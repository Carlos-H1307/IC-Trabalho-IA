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
# Limite de quase-constância: se a moda concentra mais do que isto, descarta
MAX_MODE_RATIO = 0.95
# Limite de correlação absoluta para considerar duas colunas redundantes.
# 0.90 remove pares residência/ocorrência semi-colineares (ex.: res_REGIAO_X
# vs ocor_REGIAO_X, r ~ 0.98) que só inflam a dimensionalidade sem trazer
# informação adicional.
MAX_CORRELATION = 0.90
# Cardinalidade máxima para one-hot encoding — acima disso a coluna nominal
# é considerada de alta cardinalidade e descartada (ou tratada explicitamente
# via HIGH_CARDINALITY_DROP_COLUMNS).
MAX_ONE_HOT_CARDINALITY = 20
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

# ---------------------------------------------------------------------------
# Classificação semântica das colunas do DATASUS/SIM
# ---------------------------------------------------------------------------
# A conversão de atributos categóricos deve respeitar a natureza do dado.
# Colunas nominais (sem ordem intrínseca) são codificadas via one-hot; ordinais
# preservam a ordem numérica; datas e códigos administrativos de alta
# cardinalidade são descartados.
#
# Referências:
#   - Potdar, K., Pardawala, T., & Pai, C. (2017). "A Comparative Study of
#     Categorical Variable Encoding Techniques for Neural Network Classifiers."
#     International Journal of Computer Applications, 175(4), 7-9.
#   - Hancock, J. T., & Khoshgoftaar, T. M. (2020). "Survey on categorical
#     data for neural networks." Journal of Big Data, 7:28.
#   - Micci-Barreca, D. (2001). "A Preprocessing Scheme for High-Cardinality
#     Categorical Attributes in Classification and Prediction Problems."
#     ACM SIGKDD Explorations, 3(1), 27-32. (Motiva por que ordinal encoding
#     de nominais em MLP introduz vieses espúrios.)
# ---------------------------------------------------------------------------

# Colunas nominais codificadas como inteiros no arquivo bruto.
# Sem tratamento explícito, escapam do _encode_categoricals (que só age em
# dtype object/string) e são normalizadas Min-Max como se fossem ordinais,
# induzindo relações de "adjacência" espúrias na MLP.
NOMINAL_INT_COLUMNS = {
    "RACACOR",     # 1=branca, 2=preta, 3=amarela, 4=parda, 5=indígena
    "ESTCIV",      # 1=solteiro, 2=casado, 3=viúvo, 4=separado, 5=união estável
    "LOCOCOR",     # 1=hospital, 2=out. saúde, 3=domicílio, 4=via pública, 5=outros
    "ocor_CODIGO_UF",  # UF de ocorrência (nominal, 8 códigos no recorte)
}

# Colunas binárias sim/não codificadas como 1/2 no DATASUS.
# Ao invés de one-hot (que criaria dois dummies para 2 valores), mapeamos
# para {0, 1} — mais eficiente para a MLP.
BINARY_INT_COLUMNS = {
    "ASSISTMED",   # 1=sim, 2=não (recebeu assistência médica)
    "EXAME",       # 1=sim, 2=não (fez exames complementares)
    "CIRURGIA",    # 1=sim, 2=não (foi submetido a cirurgia)
    "NECROPSIA",   # 1=sim, 2=não (fez necropsia)
}

# Colunas ordinais codificadas como inteiros — a magnitude tem significado
# monotônico (nível de escolaridade). Preservamos o encoding numérico.
# ORDINAL_INT_COLUMNS = {"ESC", "ESC2010", "SERIESCFAL", "ESCFALAGR1"}
# (não usamos a variável — apenas documenta a decisão de não one-hot-ar.)

# Colunas categóricas de altíssima cardinalidade (códigos administrativos ou
# geográficos: 80+ valores) que não são generalizáveis para uma MLP treinada
# com poucos milhares de amostras. One-hot inflaria a dimensionalidade sem
# ganho preditivo. Descartadas antes do encoding.
HIGH_CARDINALITY_DROP_COLUMNS = {
    "NATURAL",     # naturalidade (24+ códigos)
    "CODMUNNATU",  # município de naturalidade (188+ códigos IBGE)
    "CODMUNRES",   # município de residência (121+ códigos IBGE)
    "CODMUNOCOR",  # município de ocorrência (116+ códigos IBGE)
    "CODESTAB",    # estabelecimento de saúde (108+ códigos)
    "OCUP",        # ocupação CBO (80+ códigos)
    "COMUNSVOIM",  # município SVO/IML
}

# Colunas de data no formato inteiro DDMMYYYY. Tratá-las como numéricas
# produz distâncias inúteis (24042010 e 25042010 distam 100000 no valor bruto).
# Como já temos `idade_obito_anos` derivada dessas datas, são redundantes.
DATE_COLUMNS_TO_DROP = {"DTOBITO", "DTNASC"}


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


def _replace_ignorado_with_nan(df):
    """
    Para colunas do DATASUS em que '9' significa 'ignorado' e que NÃO foram
    descartadas (proporção de 9 dentro do tolerável), converte os 9s para
    NaN para que sejam tratados como valores ausentes pela imputação.

    Sem este passo, o LabelEncoder/imputação trataria 9 como uma categoria
    válida, misturando 'sem informação' com categorias reais.
    """
    transformed = {}
    for col in IGNORADO_CODE_COLUMNS:
        if col not in df.columns:
            continue
        n_replaced = int((df[col] == 9).sum())
        if n_replaced > 0:
            df[col] = df[col].replace(9, np.nan)
            transformed[col] = n_replaced
    return df, transformed


def _drop_near_constant_columns(df, threshold=MAX_MODE_RATIO):
    """
    Remove colunas onde um único valor concentra mais do que `threshold`
    dos registros. Estas colunas têm variância pequena demais para serem
    informativas para a MLP.
    """
    dropped = []
    for col in df.columns:
        counts = df[col].value_counts(dropna=False)
        if len(counts) == 0:
            continue
        mode_ratio = counts.iloc[0] / len(df)
        if mode_ratio > threshold:
            dropped.append((col, float(mode_ratio)))
            df = df.drop(columns=col)
    return df, dropped


def _drop_highly_correlated_columns(df, threshold=MAX_CORRELATION):
    """
    Detecta pares de colunas numéricas com |correlação| acima do limite e
    descarta uma das duas (a segunda na ordem das colunas) para reduzir
    redundância. A imputação por mediana é aplicada localmente apenas para
    o cálculo da correlação — os dados originais não são modificados.
    """
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return df, []

    corr = numeric.fillna(numeric.median(numeric_only=True)).corr().abs()
    cols = list(corr.columns)
    drop_set = set()
    pairs_dropped = []

    for i, col1 in enumerate(cols):
        if col1 in drop_set:
            continue
        for col2 in cols[i + 1:]:
            if col2 in drop_set:
                continue
            r = corr.loc[col1, col2]
            if r >= threshold:
                drop_set.add(col2)
                pairs_dropped.append((col1, col2, float(r)))

    if drop_set:
        df = df.drop(columns=list(drop_set))
    return df, pairs_dropped


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


def _drop_high_cardinality_nominal_columns(df):
    """
    Descarta colunas nominais de altíssima cardinalidade (códigos IBGE de
    município, CBO de ocupação, códigos de estabelecimento, etc.). Para
    uma MLP treinada com alguns milhares de amostras, one-hot dessas
    colunas inflaria a dimensionalidade sem ganho preditivo, e mantê-las
    como inteiros contínuos após Min-Max é semanticamente incorreto
    (não há ordem entre códigos IBGE).

    Referência: Micci-Barreca (2001) discute que atributos categóricos de
    alta cardinalidade requerem target/frequency encoding específico; na
    ausência disso, o descarte é preferível à codificação ingênua.
    """
    dropped = [c for c in HIGH_CARDINALITY_DROP_COLUMNS if c in df.columns]
    if dropped:
        df = df.drop(columns=dropped)
    return df, dropped


def _drop_date_columns(df):
    """
    Descarta colunas de data no formato DDMMYYYY. A informação de idade
    já está codificada em `idade_obito_anos`; a data bruta como inteiro
    não tem métrica de distância útil para a MLP.
    """
    dropped = [c for c in DATE_COLUMNS_TO_DROP if c in df.columns]
    if dropped:
        df = df.drop(columns=dropped)
    return df, dropped


def _map_binary_int_columns(df):
    """
    Mapeia colunas binárias DATASUS codificadas como {1, 2} para {1, 0}.
    Um único neurônio de entrada é suficiente para uma variável binária —
    one-hot criaria redundância perfeita (colineares).
    """
    mapped = []
    for col in BINARY_INT_COLUMNS:
        if col not in df.columns:
            continue
        # 1 = sim -> 1, 2 = não -> 0; qualquer NaN residual é preenchido depois
        df[col] = df[col].map({1: 1, 2: 0}).astype("float64")
        mapped.append(col)
    return df, mapped


def _one_hot_nominal_int_columns(df):
    """
    Aplica one-hot encoding às colunas nominais codificadas como inteiros
    no arquivo bruto (RACACOR, ESTCIV, LOCOCOR, ocor_CODIGO_UF).

    Sem este passo, essas colunas escapariam do `_encode_categoricals`
    (que só age em dtype object/string) e seriam normalizadas Min-Max
    como se fossem ordinais, o que induz relações de adjacência espúrias
    no espaço de entrada da MLP (ex.: "RACACOR=2 estaria entre RACACOR=1
    e RACACOR=3", o que não faz sentido semântico).

    Referência: Potdar et al. (2017) mostram que one-hot supera ordinal
    encoding em classificadores MLP para variáveis nominais em quase
    todos os cenários testados.
    """
    encoded = []
    for col in NOMINAL_INT_COLUMNS:
        if col not in df.columns:
            continue
        # Preenche NaN com sentinela ("-1") antes do get_dummies para não
        # perder a informação de "faltante" numa categoria separada.
        series = df[col].fillna(-1).astype("Int64").astype(str)
        dummies = pd.get_dummies(series, prefix=col, dtype="float64")
        # Remove eventual coluna do sentinela "-1" (mesmo comportamento de
        # imputação implícita: se todos os -1s virarem 0 nos dummies, a
        # informação de missingness é ignorada; aceitável quando raro).
        sentinel_col = f"{col}_-1"
        if sentinel_col in dummies.columns:
            dummies = dummies.drop(columns=sentinel_col)
        df = pd.concat([df.drop(columns=col), dummies], axis=1)
        encoded.append((col, dummies.shape[1]))
    return df, encoded


def _encode_categoricals(df):
    """
    Codifica colunas categóricas restantes (dtype object/string) em
    formato numérico. Estratégia:

      - Cardinalidade alta (> MAX_CATEGORICAL_CARDINALITY): descartada.
      - Cardinalidade baixa (<= MAX_ONE_HOT_CARDINALITY): one-hot.
      - Cardinalidade intermediária: LabelEncoder (fallback ordinal, com
        a ressalva de que pode introduzir relação ordinal espúria; usado
        aqui apenas para colunas que na prática desta base não caem
        nesse intervalo).

    A escolha de one-hot para nominais em MLP segue a recomendação de
    Hancock & Khoshgoftaar (2020) e Potdar et al. (2017).
    """
    dropped_high_card = []
    encoded_one_hot = []
    encoded_label = []
    for col in list(df.select_dtypes(include=["object", "string"]).columns):
        n_unique = df[col].nunique(dropna=True)
        if n_unique > MAX_CATEGORICAL_CARDINALITY:
            dropped_high_card.append(col)
            df = df.drop(columns=col)
            continue

        # Preenche NaN com sentinela textual (será tratado como categoria
        # extra que é depois descartada nos dummies para não introduzir
        # colinearidade com a imputação numérica).
        df[col] = df[col].fillna("__missing__").astype(str)

        if n_unique <= MAX_ONE_HOT_CARDINALITY:
            dummies = pd.get_dummies(df[col], prefix=col, dtype="float64")
            missing_col = f"{col}___missing__"
            if missing_col in dummies.columns:
                dummies = dummies.drop(columns=missing_col)
            df = pd.concat([df.drop(columns=col), dummies], axis=1)
            encoded_one_hot.append((col, dummies.shape[1]))
        else:
            df[col] = LabelEncoder().fit_transform(df[col])
            encoded_label.append(col)

    return df, dropped_high_card, encoded_one_hot, encoded_label


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


def _build_feature_groups(feature_names, one_hot_prefixes):
    """
    Agrupa colunas por atributo semântico original.

    Colunas geradas por one-hot compartilham o prefixo `{original}_{valor}`;
    as demais formam grupos unitários. Retorna uma lista de tuplas
    `(nome_do_grupo, [índices_de_colunas])` na ordem em que os grupos
    aparecem em `feature_names`.

    Essa estrutura habilita a codificação binária AGRUPADA usada no AG:
    cada gene do cromossomo corresponde a um grupo (atributo original),
    não a uma coluna numérica. Ativar um gene passa TODAS as dummies do
    grupo para a MLP simultaneamente, preservando a semântica categórica
    do one-hot e eliminando a fragmentação que ocorre quando o AG
    seleciona dummies parciais (ex.: `RACACOR_1` sem `RACACOR_2..5`).

    Referências:
      - Yang, J., & Honavar, V. (1998). "Feature Subset Selection Using
        a Genetic Algorithm." IEEE Intelligent Systems, 13(2), 44-49.
        Descreve GA para seleção de atributos com codificação binária
        onde cada gene representa uma variável original, não uma coluna
        de codificação.
      - Guyon, I., & Elisseeff, A. (2003). "An Introduction to Variable
        and Feature Selection." Journal of Machine Learning Research, 3,
        1157-1182. (Discute o efeito de encoding sobre a efetividade de
        wrappers de seleção.)
      - Kuhn, M., & Johnson, K. (2013). "Applied Predictive Modeling."
        Springer, cap. 3. Alerta sobre a interação entre one-hot e
        seleção de atributos: dummies isoladas raramente carregam
        informação suficiente sem o restante do grupo.
    """
    groups = []
    name_to_idx = {}
    prefixes_sorted = sorted(set(one_hot_prefixes), key=len, reverse=True)
    for col_idx, col_name in enumerate(feature_names):
        matched_prefix = None
        for prefix in prefixes_sorted:
            if col_name.startswith(prefix + "_"):
                matched_prefix = prefix
                break
        group_name = matched_prefix if matched_prefix is not None else col_name
        if group_name not in name_to_idx:
            name_to_idx[group_name] = len(groups)
            groups.append((group_name, []))
        groups[name_to_idx[group_name]][1].append(col_idx)
    return groups


def load_and_preprocess_data(data_path, sample_size=None, random_state=42, verbose=True):
    """
    Pipeline de carregamento e pré-processamento da base de câncer do colo do útero.

    Etapas:
      1.  Leitura do arquivo
      2.  Remoção de registros sem alvo
      3.  Remoção de duplicatas exatas
      4.  Checagens de inconsistência (datas, idade) — apenas relata
      5.  Remoção de colunas de vazamento de alvo
      6.  Descarte de colunas de data brutas (DDMMYYYY sem métrica útil)
      7.  Descarte de códigos administrativos de alta cardinalidade
      8.  Remoção de colunas constantes (variância zero)
      9.  Remoção de colunas dominadas por '9' (>80% 'ignorado' do DATASUS)
      10. Remoção de colunas quase-constantes (moda > 95%)
      11. Conversão de '9' residual ('ignorado') para NaN
      12. Remoção de colunas com alta proporção de valores ausentes
      13. Mapeamento de colunas binárias DATASUS (1/2) para {1, 0}
      14. One-hot de colunas nominais codificadas como inteiros
      15. One-hot de categóricas textuais (baixa cardinalidade)
      16. Imputação de valores numéricos ausentes pela mediana
      17. Remoção de colunas numéricas redundantes (|corr| > MAX_CORRELATION)
      18. (Opcional) amostragem estratificada
      19. Normalização linear Min-Max em [0, 1]
      20. Construção dos grupos de features (1 grupo por atributo
          semântico original — dimensão do cromossomo binário no AG)

    Referências para o encoding categórico:
      Potdar et al. (2017); Hancock & Khoshgoftaar (2020); Micci-Barreca (2001).

    Retorna:
        X (np.ndarray): matriz de atributos normalizada
        y (np.ndarray): vetor alvo codificado em inteiros
        feature_names (list[str]): nomes das colunas finais
        n_classes (int): quantidade de classes distintas no alvo
        class_names (list[str]): rótulos originais na ordem 0..n_classes-1
        stats (dict): estatísticas do pipeline para relatório e EDA
        feature_groups (list[(str, list[int])]): grupos de colunas por
            atributo semântico original. `len(feature_groups)` é a
            dimensão do cromossomo binário do AG.
    """
    df = _load_raw(data_path)
    n_raw = len(df)
    cols_raw = list(df.columns)
    df_raw_for_eda = df.copy()  # cópia intocada para EDA (idade, classe, missing)

    stats = {
        "n_raw": n_raw,
        "cols_raw": cols_raw,
        "missing_ratio_raw": df.isnull().mean().to_dict(),
        "cardinality_raw": {c: int(df[c].nunique(dropna=True)) for c in df.columns},
    }

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Coluna alvo '{TARGET_COLUMN}' não encontrada. "
            f"Primeiras colunas disponíveis: {df.columns.tolist()[:10]}"
        )

    # Distribuição bruta da coluna alvo (antes de qualquer remoção)
    stats["target_distribution_raw"] = (
        df[TARGET_COLUMN].astype(str).value_counts(dropna=False).to_dict()
    )

    # 1) Remove registros sem alvo
    df = df.dropna(subset=[TARGET_COLUMN])
    n_no_target_removed = n_raw - len(df)
    stats["n_no_target_removed"] = int(n_no_target_removed)

    # 2) Remove duplicatas exatas
    df, n_duplicates_removed = _drop_duplicates(df)
    stats["n_duplicates_removed"] = int(n_duplicates_removed)

    # Checagens de consistência (informativas)
    n_age_outliers = _check_age_outliers(df)
    n_date_inconsistent = _check_date_consistency(df)
    stats["n_age_outliers"] = int(n_age_outliers)
    stats["n_date_inconsistent"] = int(n_date_inconsistent)

    if "idade_obito_anos" in df.columns:
        stats["age_series"] = df["idade_obito_anos"].dropna().tolist()
        stats["age_with_target"] = df[
            ["idade_obito_anos", TARGET_COLUMN]
        ].dropna().to_dict(orient="list")

    # Proporção de "ignorado" (código 9) nas colunas DATASUS marcadas
    ignorado_ratio = {}
    for col in IGNORADO_CODE_COLUMNS:
        if col in df.columns:
            ignorado_ratio[col] = float((df[col] == 9).mean())
    stats["ignorado_ratio"] = ignorado_ratio

    # Separa o alvo
    y_raw = df[TARGET_COLUMN].astype(str).values
    df = df.drop(columns=[TARGET_COLUMN])

    # 3) Remove colunas de vazamento de alvo
    leak_cols = [c for c in TARGET_LEAK_COLUMNS if c in df.columns]
    df = df.drop(columns=leak_cols)
    stats["leak_cols"] = list(leak_cols)

    # 4) Descarte de datas brutas (redundantes com idade_obito_anos)
    df, dropped_dates = _drop_date_columns(df)
    stats["dropped_dates"] = list(dropped_dates)

    # 5) Descarte de códigos administrativos de alta cardinalidade
    df, dropped_high_card_int = _drop_high_cardinality_nominal_columns(df)
    stats["dropped_high_card_int"] = list(dropped_high_card_int)

    # 6) Remove colunas constantes (variância zero)
    df, constant_cols = _drop_constant_columns(df)
    stats["constant_cols"] = list(constant_cols)

    # 7) Remove colunas dominadas por código "ignorado" (9 > 80%)
    df, ignorado_dropped = _drop_ignorado_dominated_columns(df)
    stats["ignorado_dropped"] = [(c, float(r)) for c, r in ignorado_dropped]

    # 8) Remove colunas quase-constantes (moda > 95%)
    df, near_constant_cols = _drop_near_constant_columns(df)
    stats["near_constant_cols"] = [(c, float(r)) for c, r in near_constant_cols]

    # 9) Converte código 9 ("ignorado") residual para NaN (será imputado)
    df, ignorado_replaced = _replace_ignorado_with_nan(df)
    stats["ignorado_replaced"] = {c: int(n) for c, n in ignorado_replaced.items()}

    # 10) Remove colunas com muitos NaN
    df, dropped_missing = _drop_high_missing_columns(df)
    stats["dropped_missing"] = list(dropped_missing)

    # 11) Mapeia binárias DATASUS {1, 2} -> {1, 0}
    df, binary_mapped = _map_binary_int_columns(df)
    stats["binary_mapped"] = list(binary_mapped)

    # 12) One-hot das colunas nominais codificadas como inteiros
    df, nominal_int_encoded = _one_hot_nominal_int_columns(df)
    stats["nominal_int_encoded"] = [(c, int(n)) for c, n in nominal_int_encoded]

    # 13) Codifica categóricas textuais restantes (one-hot para baixa
    #     cardinalidade; descarte para alta cardinalidade)
    df, dropped_cardinality, str_one_hot, str_label = _encode_categoricals(df)
    stats["dropped_cardinality_text"] = list(dropped_cardinality)
    stats["str_one_hot"] = [(c, int(n)) for c, n in str_one_hot]
    stats["str_label"] = list(str_label)

    # 14) Imputa valores numéricos ausentes
    df = _impute_numeric(df)

    # 15) Remove colunas numéricas redundantes (|corr| > 0.95)
    #     Executado APÓS one-hot para capturar dummies perfeitamente
    #     colineares (raro, mas ocorre se um par de colunas nominais
    #     representa a mesma partição em amostragens pequenas).
    df, corr_pairs_dropped = _drop_highly_correlated_columns(df)
    stats["corr_pairs_dropped"] = [
        (a, b, float(r)) for a, b, r in corr_pairs_dropped
    ]

    # 16) Codifica o alvo
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y_raw)
    n_classes = len(target_encoder.classes_)

    feature_names = df.columns.tolist()
    X = df.values

    # 17) Amostragem estratificada opcional
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

    # 18) Normalização Min-Max
    X = _min_max_normalize(X)

    if verbose:
        print(f"[INFO] Linhas originais: {n_raw}")
        print(f"[INFO] Linhas sem alvo removidas: {n_no_target_removed}")
        print(f"[INFO] Duplicatas exatas removidas: {n_duplicates_removed}")
        print(f"[INFO] Idades fora de [0, 120]: {n_age_outliers}")
        print(f"[INFO] Datas inconsistentes (nasc > óbito): {n_date_inconsistent}")
        print(f"[INFO] Colunas originais: {len(cols_raw)}")
        print(f"[INFO] Colunas removidas por vazamento de alvo: {len(leak_cols)}")
        print(f"[INFO] Colunas de data removidas: {dropped_dates}")
        print(f"[INFO] Códigos administrativos de alta cardinalidade removidos: "
              f"{dropped_high_card_int}")
        print(f"[INFO] Colunas removidas por variância zero: {len(constant_cols)} {constant_cols}")
        print(f"[INFO] Colunas removidas por 'ignorado' > {int(MAX_IGNORADO_RATIO*100)}%: "
              f"{[(c, f'{r*100:.0f}%') for c, r in ignorado_dropped]}")
        print(f"[INFO] Colunas removidas por quase-constância (moda > {int(MAX_MODE_RATIO*100)}%): "
              f"{[(c, f'{r*100:.1f}%') for c, r in near_constant_cols]}")
        print(f"[INFO] '9' (ignorado) convertido para NaN em: "
              f"{[(c, n) for c, n in ignorado_replaced.items()]}")
        print(f"[INFO] Colunas removidas por excesso de NaN: {len(dropped_missing)} {dropped_missing}")
        print(f"[INFO] Binárias DATASUS (1/2 -> 1/0): {binary_mapped}")
        print(f"[INFO] One-hot de nominais inteiras: "
              f"{[(c, n) for c, n in nominal_int_encoded]}")
        print(f"[INFO] One-hot de categóricas textuais: "
              f"{[(c, n) for c, n in str_one_hot]}")
        if str_label:
            print(f"[INFO] LabelEncoder aplicado (fallback ordinal): {str_label}")
        print(f"[INFO] Colunas removidas por correlação > {MAX_CORRELATION}: "
              f"{[(a, b, f'{r:.3f}') for a, b, r in corr_pairs_dropped]}")
        print(f"[INFO] Colunas removidas por alta cardinalidade textual: "
              f"{len(dropped_cardinality)} {dropped_cardinality}")
        print(f"[INFO] Total de atributos finais: {len(feature_names)}")
        print(f"[INFO] Total de registros finais: {len(X)}")
        print(f"[INFO] Classes: {list(target_encoder.classes_)} -> {list(range(n_classes))}")
        class_counts = np.bincount(y)
        print("[INFO] Distribuição: " +
              ", ".join(f"{c}={n} ({n/len(y)*100:.1f}%)"
                        for c, n in zip(target_encoder.classes_, class_counts)))

    class_names = [str(c) for c in target_encoder.classes_]

    # 20) Grupos de features para codificação binária agrupada no AG.
    #     Coleta os prefixos das colunas que sofreram one-hot (nominais
    #     inteiras + textuais) e agrupa colunas do X final por prefixo.
    one_hot_prefixes = (
        [c for c, _ in nominal_int_encoded] +
        [c for c, _ in str_one_hot]
    )
    feature_groups = _build_feature_groups(feature_names, one_hot_prefixes)

    stats["n_final_samples"] = int(len(X))
    stats["n_final_features"] = int(len(feature_names))
    stats["feature_names_final"] = list(feature_names)
    stats["n_feature_groups"] = len(feature_groups)
    stats["feature_groups"] = [(name, list(idxs)) for name, idxs in feature_groups]
    stats["class_distribution_final"] = {
        class_names[c]: int(np.sum(y == c)) for c in range(n_classes)
    }
    stats["X_final_snapshot"] = X.copy()

    if verbose:
        print(f"[INFO] Grupos de features (atributos semânticos originais): "
              f"{len(feature_groups)}")
        for name, idxs in feature_groups:
            if len(idxs) > 1:
                print(f"  {name}: {len(idxs)} colunas -> {[feature_names[i] for i in idxs]}")

    return X, y, feature_names, n_classes, class_names, stats, feature_groups
