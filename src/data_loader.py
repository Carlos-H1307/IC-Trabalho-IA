import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Limite de proporção de valores ausentes para descartar uma coluna
MAX_MISSING_RATIO = 0.5
# Limite de cardinalidade para colunas categóricas (acima disso, a coluna é descartada)
MAX_CATEGORICAL_CARDINALITY = 50
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


def _load_raw(data_path):
    """Lê o arquivo bruto, suportando tanto .csv quanto .xlsx."""
    ext = os.path.splitext(data_path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(data_path)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(data_path)
    else:
        raise ValueError(f"Formato de arquivo não suportado: {ext}")


def _drop_high_missing_columns(df, threshold=MAX_MISSING_RATIO):
    """Remove colunas com proporção de valores ausentes acima do limite."""
    missing_ratio = df.isnull().mean()
    cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
    return df.drop(columns=cols_to_drop), cols_to_drop


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


def load_and_preprocess_data(data_path, sample_size=None, random_state=42):
    """
    Pipeline de carregamento e pré-processamento da base de câncer do colo do útero.

    Etapas (seguindo a especificação do trabalho):
      1. Leitura do arquivo
      2. Remoção de registros inconsistentes (linhas sem o alvo)
      3. Remoção de colunas com alta proporção de valores ausentes
      4. Codificação de atributos categóricos para valores numéricos
      5. Imputação de valores numéricos ausentes pela mediana
      6. (Opcional) amostragem estratificada para reduzir custo computacional
      7. Normalização linear Min-Max no intervalo [0, 1]

    Args:
        data_path: caminho para o arquivo da base (CSV ou XLSX).
        sample_size: se informado, faz amostragem estratificada para esse tamanho.
        random_state: semente para a amostragem.

    Retorna:
        X (np.ndarray): matriz de atributos normalizada
        y (np.ndarray): vetor alvo codificado em inteiros
        feature_names (list[str]): nomes das colunas finais
        n_classes (int): quantidade de classes distintas no alvo
    """
    df = _load_raw(data_path)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Coluna alvo '{TARGET_COLUMN}' não encontrada. "
            f"Primeiras colunas disponíveis: {df.columns.tolist()[:10]}"
        )

    # 1) Remove registros sem alvo (inconsistentes)
    df = df.dropna(subset=[TARGET_COLUMN])

    # Separa o alvo antes de tocar no restante
    y_raw = df[TARGET_COLUMN].astype(str).values
    df = df.drop(columns=[TARGET_COLUMN])

    # 2) Remove colunas que constituem vazamento direto do alvo
    leak_cols = [c for c in TARGET_LEAK_COLUMNS if c in df.columns]
    df = df.drop(columns=leak_cols)

    # 3) Remove colunas com muitos valores ausentes
    df, dropped_missing = _drop_high_missing_columns(df)

    # 3) Codifica categóricas / descarta colunas de alta cardinalidade
    df, dropped_cardinality = _encode_categoricals(df)

    # 4) Imputa valores numéricos ausentes
    df = _impute_numeric(df)

    # 5) Codifica o alvo para inteiros (0..n_classes-1)
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y_raw)
    n_classes = len(target_encoder.classes_)

    feature_names = df.columns.tolist()
    X = df.values

    # 6) Amostragem estratificada opcional para tornar o GA tratável
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

    # 7) Normalização Min-Max
    X = _min_max_normalize(X)

    print(f"[INFO] Colunas removidas por vazamento de alvo: {len(leak_cols)}")
    print(f"[INFO] Colunas removidas por excesso de NaN: {len(dropped_missing)}")
    print(f"[INFO] Colunas removidas por alta cardinalidade: {len(dropped_cardinality)}")
    print(f"[INFO] Total de atributos finais: {len(feature_names)}")
    print(f"[INFO] Total de registros: {len(X)}")
    print(f"[INFO] Classes: {list(target_encoder.classes_)} -> {list(range(n_classes))}")

    return X, y, feature_names, n_classes
