import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Limite de proporção de valores ausentes para descartar uma coluna
MAX_MISSING_RATIO = 0.5
# Limite de cardinalidade para colunas categóricas (acima disso, a coluna é descartada)
MAX_CATEGORICAL_CARDINALITY = 50
# Limite de proporção do código 9 ("ignorado" no DATASUS) para descartar coluna
MAX_IGNORADO_RATIO = 0.8
# Nome esperado da coluna alvo
TARGET_COLUMN = "label_cid"

# Colunas que constituem vazamento direto do alvo (descrevem a própria causa do
# óbito a partir da qual o label_cid foi derivado). Removidas antes da análise
# para que a seleção de atributos seja informativa.
TARGET_LEAK_COLUMNS = {
    "CAUSABAS",
    "CAUSABAS_O",
    "CB_PRE",
    "causabas_categoria",
    "causabas_subcategoria",
    "causabas_capitulo",
    "causabas_grupo",
    "LINHAA",
    "LINHAB",
    "LINHAC",
    "LINHAD",
    "LINHAII",
    "ATESTADO",
    "ATESTANTE",
}

# Colunas categóricas do SIM/DATASUS em que o código 9 significa "ignorado"
# (e não uma categoria válida). Para essas, se a proporção de 9s for muito
# alta, a coluna é descartada por falta de poder discriminativo.
IGNORADO_CODE_COLUMNS = {
    "RACACOR",
    "ESTCIV",
    "ESC",
    "ESC2010",
    "ASSISTMED",
    "EXAME",
    "CIRURGIA",
    "NECROPSIA",
    "LOCOCOR",
    "ESCFALAGR1",
}


def _load_raw(data_path):
    """Lê o arquivo bruto, suportando tanto .csv quanto .xlsx."""
    ext = os.path.splitext(data_path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(data_path)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(data_path)
    else:
        raise ValueError(f"Formato de arquivo não suportado: {ext}")


def _drop_duplicates(df):
    """Remove linhas duplicadas exatas."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    return df, before - len(df)


def _drop_constant_columns(df):
    """Remove colunas em que todos os valores são iguais (variância zero).
    Inclui colunas inteiramente nulas, que também não trazem informação.
    """
    constant_cols = []
    for col in df.columns:
        # nunique(dropna=False) considera NaN como um valor possível
        if df[col].nunique(dropna=False) <= 1:
            constant_cols.append(col)
    return df.drop(columns=constant_cols), constant_cols


def _drop_ignorado_dominated_columns(df, threshold=MAX_IGNORADO_RATIO):
    """
    Para colunas conhecidas do DATASUS em que '9' significa 'ignorado',
    descarta aquelas onde a proporção de 9s ultrapassa o limite — não há
    informação útil para discriminar classes.
    """
    dropped = []
    for col in IGNORADO_CODE_COLUMNS:
        if col not in df.columns:
            continue
        ignorado_ratio = (df[col] == 9).mean()
        if ignorado_ratio > threshold:
            dropped.append((col, float(ignorado_ratio)))
            df = df.drop(columns=col)
    return df, dropped


def _drop_high_missing_columns(df, threshold=MAX_MISSING_RATIO):
    """Remove colunas com proporção de valores ausentes acima do limite."""
    missing_ratio = df.isnull().mean()
    cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
    return df.drop(columns=cols_to_drop), cols_to_drop


def _check_date_consistency(df):
    """
    Verifica inconsistências em colunas de data (DTNASC vs DTOBITO).
    O SIM codifica datas como inteiros DDMMYYYY. Retorna a quantidade de
    registros com data de nascimento posterior à data do óbito.
    Não remove nada — apenas relata.
    """
    if "DTOBITO" not in df.columns or "DTNASC" not in df.columns:
        return 0
    try:
        obito_year = df["DTOBITO"].astype(str).str.zfill(8).str[-4:].astype(int)
        nasc_year = df["DTNASC"].fillna(0).astype(np.int64).astype(str).str.zfill(8).str[-4:].astype(int)
        inconsistent = ((nasc_year > obito_year) & (nasc_year > 1900)).sum()
        return int(inconsistent)
    except Exception:
        return 0


def _check_age_outliers(df):
    """Verifica registros com idade fora do intervalo plausível [0, 120]."""
    if "idade_obito_anos" not in df.columns:
        return 0
    ages = df["idade_obito_anos"]
    invalid = ((ages < 0) | (ages > 120)).sum()
    return int(invalid)


def _encode_categoricals(df):
    """
    Converte colunas categóricas (object/string) em valores numéricos via LabelEncoder.
    Colunas com cardinalidade muito alta são descartadas (tipicamente IDs ou nomes livres).
    """
    dropped = []
    for col in df.select_dtypes(include=["object", "string"]).columns:
        n_unique = df[col].nunique(dropna=True)
        if n_unique > MAX_CATEGORICAL_CARDINALITY:
            dropped.append(col)
            df = df.drop(columns=col)
            continue
        # Preenche NaN com uma string temporária antes da codificação
        df[col] = df[col].fillna("__missing__").astype(str)
        df[col] = LabelEncoder().fit_transform(df[col])
    return df, dropped


def _impute_numeric(df):
    """Preenche valores numéricos ausentes com a mediana da coluna."""
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def _min_max_normalize(X):
    """
    Aplica a normalização linear Min-Max nas colunas de uma matriz numérica.
    x' = (x - x_min) / (x_max - x_min)
    Colunas sem variância (x_max == x_min) ficam zeradas.
    """
    X = X.astype(np.float64)
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    denom = x_max - x_min
    zero_var_mask = denom == 0
    denom[zero_var_mask] = 1.0  # evita divisão por zero
    X_norm = (X - x_min) / denom
    X_norm[:, zero_var_mask] = 0.0
    return X_norm


def load_and_preprocess_data(data_path, sample_size=None, random_state=42, verbose=True):
    """
    Pipeline de carregamento e pré-processamento da base de câncer do colo do útero.

    Etapas:
      1. Leitura do arquivo
      2. Remoção de registros sem alvo (inconsistência crítica)
      3. Remoção de duplicatas exatas
      4. Checagens de inconsistência (datas, idade) — apenas relata
      5. Remoção de colunas de vazamento de alvo
      6. Remoção de colunas constantes (variância zero)
      7. Remoção de colunas dominadas por '9' (código 'ignorado' do DATASUS)
      8. Remoção de colunas com alta proporção de valores ausentes
      9. Codificação de atributos categóricos (LabelEncoder); descarte por alta cardinalidade
      10. Imputação de valores numéricos ausentes pela mediana
      11. (Opcional) amostragem estratificada
      12. Normalização linear Min-Max em [0, 1]

    Retorna:
        X (np.ndarray): matriz de atributos normalizada
        y (np.ndarray): vetor alvo codificado em inteiros
        feature_names (list[str]): nomes das colunas finais
        n_classes (int): quantidade de classes distintas no alvo
    """
    df = _load_raw(data_path)
    n_raw = len(df)
    cols_raw = list(df.columns)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Coluna alvo '{TARGET_COLUMN}' não encontrada. "
            f"Primeiras colunas disponíveis: {df.columns.tolist()[:10]}"
        )

    # 1) Remove registros sem alvo
    df = df.dropna(subset=[TARGET_COLUMN])
    n_no_target_removed = n_raw - len(df)

    # 2) Remove duplicatas exatas
    df, n_duplicates_removed = _drop_duplicates(df)

    # Checagens de consistência (informativas)
    n_age_outliers = _check_age_outliers(df)
    n_date_inconsistent = _check_date_consistency(df)

    # Separa o alvo
    y_raw = df[TARGET_COLUMN].astype(str).values
    df = df.drop(columns=[TARGET_COLUMN])

    # 3) Remove colunas de vazamento de alvo
    leak_cols = [c for c in TARGET_LEAK_COLUMNS if c in df.columns]
    df = df.drop(columns=leak_cols)

    # 4) Remove colunas constantes
    df, constant_cols = _drop_constant_columns(df)

    # 5) Remove colunas dominadas por código "ignorado" (9)
    df, ignorado_dropped = _drop_ignorado_dominated_columns(df)

    # 6) Remove colunas com muitos NaN
    df, dropped_missing = _drop_high_missing_columns(df)

    # 7) Codifica categóricas / descarta colunas de alta cardinalidade
    df, dropped_cardinality = _encode_categoricals(df)

    # 8) Imputa valores numéricos ausentes
    df = _impute_numeric(df)

    # 9) Codifica o alvo
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y_raw)
    n_classes = len(target_encoder.classes_)

    feature_names = df.columns.tolist()
    X = df.values

    # 10) Amostragem estratificada opcional
    if sample_size is not None and sample_size < len(X):
        rng = np.random.default_rng(random_state)
        idx_per_class = []
        for c in np.unique(y):
            class_idx = np.where(y == c)[0]
            n_take = max(1, int(round(sample_size * len(class_idx) / len(y))))
            n_take = min(n_take, len(class_idx))
            idx_per_class.append(rng.choice(class_idx, size=n_take, replace=False))
        idx = np.concatenate(idx_per_class)
        rng.shuffle(idx)
        X = X[idx]
        y = y[idx]

    # 11) Normalização Min-Max
    X = _min_max_normalize(X)

    if verbose:
        print(f"[INFO] Linhas originais: {n_raw}")
        print(f"[INFO] Linhas sem alvo removidas: {n_no_target_removed}")
        print(f"[INFO] Duplicatas exatas removidas: {n_duplicates_removed}")
        print(f"[INFO] Idades fora de [0, 120]: {n_age_outliers}")
        print(f"[INFO] Datas inconsistentes (nasc > óbito): {n_date_inconsistent}")
        print(f"[INFO] Colunas originais: {len(cols_raw)}")
        print(f"[INFO] Colunas removidas por vazamento de alvo: {len(leak_cols)}")
        print(f"[INFO] Colunas removidas por variância zero: {len(constant_cols)} {constant_cols}")
        print(f"[INFO] Colunas removidas por 'ignorado' > {int(MAX_IGNORADO_RATIO*100)}%: "
              f"{[(c, f'{r*100:.0f}%') for c, r in ignorado_dropped]}")
        print(f"[INFO] Colunas removidas por excesso de NaN: {len(dropped_missing)}")
        print(f"[INFO] Colunas removidas por alta cardinalidade: {len(dropped_cardinality)}")
        print(f"[INFO] Total de atributos finais: {len(feature_names)}")
        print(f"[INFO] Total de registros finais: {len(X)}")
        print(f"[INFO] Classes: {list(target_encoder.classes_)} -> {list(range(n_classes))}")
        class_counts = np.bincount(y)
        print(f"[INFO] Distribuição: " +
              ", ".join(f"{c}={n} ({n/len(y)*100:.1f}%)"
                        for c, n in zip(target_encoder.classes_, class_counts)))

    return X, y, feature_names, n_classes
