"""
Gráficos exploratórios sobre a base bruta e o pré-processamento.

Consomem o dicionário `stats` produzido por `data_loader.load_and_preprocess_data`.
Todos os plots são renderizados diretamente em disco (matplotlib backend Agg),
sem dependências fora de pandas/matplotlib/numpy.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TOP_N_MISSING = 25
TOP_N_CARDINALITY = 25
FIGSIZE = (11, 6)


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1) Distribuição do alvo (bruta vs. final)
# ---------------------------------------------------------------------------
def plot_class_distribution(stats, path):
    """Barras lado a lado: distribuição de classes bruta x final."""
    raw = stats.get("target_distribution_raw", {})
    fin = stats.get("class_distribution_final", {})

    labels = sorted(set(list(raw.keys()) + list(fin.keys())))
    raw_vals = [raw.get(l, 0) for l in labels]
    fin_vals = [fin.get(l, 0) for l in labels]

    x = np.arange(len(labels))
    width = 0.4

    fig, ax = plt.subplots(figsize=FIGSIZE)
    b1 = ax.bar(x - width / 2, raw_vals, width, label="Base bruta", color="#7f8c8d")
    b2 = ax.bar(x + width / 2, fin_vals, width, label="Após pré-proc.", color="#1f77b4")
    ax.set_title("Distribuição de classes — bruta vs. após pré-processamento", pad=12)
    ax.set_xlabel("Classe (label_cid)")
    ax.set_ylabel("Nº de registros")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax.legend()

    for bars in (b1, b2):
        for b in bars:
            h = int(b.get_height())
            if h:
                ax.text(b.get_x() + b.get_width() / 2, h, str(h),
                        ha="center", va="bottom", fontsize=8)
    _save(fig, path)


# ---------------------------------------------------------------------------
# 2) Waterfall de remoção de colunas
# ---------------------------------------------------------------------------
def plot_column_removal_waterfall(stats, path):
    """
    Barras empilhadas mostrando quantas colunas foram descartadas em cada
    estágio do pré-processamento. Explica visualmente o funil.
    """
    reasons = [
        ("Vazamento de alvo", len(stats.get("leak_cols", []))),
        ("Datas brutas (DDMMYYYY)", len(stats.get("dropped_dates", []))),
        ("Alta cardinalidade (int)", len(stats.get("dropped_high_card_int", []))),
        ("Variância zero (constante)", len(stats.get("constant_cols", []))),
        ("'Ignorado' (9) > 80%", len(stats.get("ignorado_dropped", []))),
        ("Quase-constante (moda > 95%)", len(stats.get("near_constant_cols", []))),
        ("NaN > 50%", len(stats.get("dropped_missing", []))),
        ("Alta cardinalidade (texto)", len(stats.get("dropped_cardinality_text", []))),
        ("Corr. |r| > 0.95", len(stats.get("corr_pairs_dropped", []))),
    ]
    added = [
        ("One-hot nominais int (dummies)", sum(n for _, n in stats.get("nominal_int_encoded", []))),
        ("One-hot textuais (dummies)", sum(n for _, n in stats.get("str_one_hot", []))),
    ]
    n_raw = len(stats.get("cols_raw", []))
    n_final = stats.get("n_final_features", 0)

    fig, ax = plt.subplots(figsize=(12, 6.5))

    labels_r = [l for l, _ in reasons]
    counts_r = [c for _, c in reasons]
    labels_a = [l for l, _ in added]
    counts_a = [c for _, c in added]

    all_labels = ["Colunas brutas"] + labels_r + labels_a + ["Colunas finais (L)"]
    values = [n_raw] + [-c for c in counts_r] + counts_a + [n_final]
    colors = (
        ["#2c3e50"]
        + ["#c0392b"] * len(reasons)
        + ["#27ae60"] * len(added)
        + ["#1f77b4"]
    )

    x = np.arange(len(all_labels))
    bars = ax.bar(x, values, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Δ colunas")
    ax.set_title(
        f"Waterfall do pré-processamento — {n_raw} colunas brutas → {n_final} atributos finais",
        pad=12,
    )
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")

    for b, v in zip(bars, values):
        if v == 0:
            continue
        ax.text(b.get_x() + b.get_width() / 2, v,
                f"{'+' if v > 0 else ''}{v}",
                ha="center",
                va="bottom" if v > 0 else "top",
                fontsize=8, fontweight="bold")
    _save(fig, path)


# ---------------------------------------------------------------------------
# 3) Missing por coluna (top-N)
# ---------------------------------------------------------------------------
def plot_missing_ratio(stats, path, top_n=TOP_N_MISSING):
    """
    Barras horizontais das TOP-N colunas com maior % de NaN na base bruta.
    Colore em vermelho as que foram efetivamente descartadas pelo pipeline.
    """
    missing = stats.get("missing_ratio_raw", {})
    if not missing:
        return
    dropped_missing = set(stats.get("dropped_missing", []))

    series = (
        pd.Series(missing)
        .sort_values(ascending=False)
        .head(top_n)
        .iloc[::-1]  # inverte para ficar ordenado ao plotar horizontal
    )

    colors = ["#c0392b" if c in dropped_missing else "#3498db" for c in series.index]

    fig, ax = plt.subplots(figsize=(11, max(6, 0.28 * len(series))))
    ax.barh(series.index, series.values * 100, color=colors)
    ax.set_xlabel("% de valores ausentes")
    ax.set_title(f"TOP-{top_n} colunas por % NaN na base bruta "
                 "(vermelho = descartada por > 50% NaN)", pad=12)
    ax.axvline(50, color="black", linestyle="--", linewidth=1,
               label="Limite (50%)")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.5, axis="x")
    for i, v in enumerate(series.values):
        ax.text(v * 100 + 0.5, i, f"{v*100:.1f}%", va="center", fontsize=8)
    _save(fig, path)


# ---------------------------------------------------------------------------
# 4) Cardinalidade por coluna (top-N)
# ---------------------------------------------------------------------------
def plot_cardinality(stats, path, top_n=TOP_N_CARDINALITY):
    """
    Barras horizontais das TOP-N colunas por cardinalidade (nº valores únicos).
    Anota as descartadas por alta cardinalidade textual/administrativa.
    """
    card = stats.get("cardinality_raw", {})
    if not card:
        return
    dropped_admin = set(stats.get("dropped_high_card_int", []))
    dropped_text = set(stats.get("dropped_cardinality_text", []))

    series = pd.Series(card).sort_values(ascending=False).head(top_n).iloc[::-1]

    def _color(c):
        if c in dropped_admin:
            return "#c0392b"
        if c in dropped_text:
            return "#e67e22"
        return "#3498db"

    colors = [_color(c) for c in series.index]

    fig, ax = plt.subplots(figsize=(11, max(6, 0.28 * len(series))))
    ax.barh(series.index, series.values, color=colors)
    ax.set_xlabel("Nº de valores únicos")
    ax.set_title(
        f"TOP-{top_n} colunas por cardinalidade\n"
        "vermelho = descarte por código administrativo | "
        "laranja = descarte por alta cardinalidade textual",
        pad=12,
    )
    ax.grid(True, linestyle=":", alpha=0.5, axis="x")
    for i, v in enumerate(series.values):
        ax.text(v + 0.5, i, str(int(v)), va="center", fontsize=8)
    _save(fig, path)


# ---------------------------------------------------------------------------
# 5) Proporção de "9" (ignorado) por coluna DATASUS
# ---------------------------------------------------------------------------
def plot_ignorado_ratio(stats, path):
    """
    Barras horizontais com a proporção de código '9' ("ignorado") em cada
    coluna DATASUS marcada. Marca visualmente o limite de 80% (descarte).
    """
    data = stats.get("ignorado_ratio", {})
    if not data:
        return
    dropped = {c: r for c, r in stats.get("ignorado_dropped", [])}

    series = pd.Series(data).sort_values(ascending=True)
    colors = ["#c0392b" if c in dropped else "#3498db" for c in series.index]

    fig, ax = plt.subplots(figsize=(11, max(4, 0.35 * len(series))))
    ax.barh(series.index, series.values * 100, color=colors)
    ax.axvline(80, color="black", linestyle="--", linewidth=1,
               label="Limite (80%)")
    ax.set_xlabel("% de registros com código 9 (ignorado)")
    ax.set_title("Colunas DATASUS: proporção de código 9 (ignorado)\n"
                 "vermelho = coluna descartada pelo limite", pad=12)
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5, axis="x")
    for i, v in enumerate(series.values):
        ax.text(v * 100 + 0.5, i, f"{v*100:.1f}%", va="center", fontsize=8)
    _save(fig, path)


# ---------------------------------------------------------------------------
# 6) Idade — distribuição geral e por classe
# ---------------------------------------------------------------------------
def plot_age_distribution(stats, path):
    """Histograma da idade de óbito na base bruta (pós-remoção de sem-alvo)."""
    ages = stats.get("age_series")
    if not ages:
        return
    ages = np.array(ages)
    ages_valid = ages[(ages >= 0) & (ages <= 120)]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(ages_valid, bins=40, color="#1f77b4", edgecolor="white")
    ax.axvline(np.median(ages_valid), color="#c0392b", linestyle="--",
               linewidth=1.5, label=f"Mediana = {np.median(ages_valid):.0f} anos")
    ax.set_xlabel("Idade no óbito (anos)")
    ax.set_ylabel("Frequência")
    ax.set_title("Distribuição da idade no óbito (base bruta)", pad=12)
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)
    _save(fig, path)


def plot_age_by_class(stats, path):
    """Box plot de idade de óbito por classe alvo (label_cid) na base bruta."""
    payload = stats.get("age_with_target")
    if not payload:
        return
    df = pd.DataFrame(payload)
    df.columns = ["idade", "classe"]
    df = df[(df["idade"] >= 0) & (df["idade"] <= 120)]
    df["classe"] = df["classe"].astype(str)

    order = sorted(df["classe"].unique())
    grouped = [df.loc[df["classe"] == c, "idade"].values for c in order]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bp = ax.boxplot(grouped, patch_artist=True, showmeans=True,
                    meanprops={"marker": "D", "markerfacecolor": "yellow",
                               "markeredgecolor": "black", "markersize": 6})
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order)
    palette = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#e67e22"]
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    ax.set_xlabel("Classe (label_cid)")
    ax.set_ylabel("Idade no óbito (anos)")
    ax.set_title("Idade no óbito por classe alvo (base bruta)", pad=12)
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")
    _save(fig, path)


# ---------------------------------------------------------------------------
# 7) Correlação entre atributos finais (top-N por variância)
# ---------------------------------------------------------------------------
def plot_final_correlation_heatmap(stats, path, top_n=25):
    """
    Heatmap da matriz de correlação entre os TOP-N atributos finais
    (ordenados por variância). Ajuda a inspecionar redundância residual
    após os drops de correlação > 0.95.
    """
    X = stats.get("X_final_snapshot")
    feature_names = stats.get("feature_names_final", [])
    if X is None or len(feature_names) == 0:
        return

    var = X.var(axis=0)
    n = min(top_n, X.shape[1])
    top_idx = np.argsort(var)[::-1][:n]
    Xs = X[:, top_idx]
    names = [feature_names[i] for i in top_idx]

    corr = np.corrcoef(Xs, rowvar=False)

    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=75, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_title(f"Correlação de Pearson — TOP-{n} atributos finais (por variância)",
                 pad=12)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Correlação")
    _save(fig, path)


# ---------------------------------------------------------------------------
# 8) Perfil de encoding (contagem por tratamento)
# ---------------------------------------------------------------------------
def plot_encoding_profile(stats, path):
    """
    Barras horizontais: quantos atributos finais foram gerados por cada
    tratamento (one-hot nominal int, one-hot textual, mapeamento binário,
    passagem direta, LabelEncoder).
    """
    n_binary = len(stats.get("binary_mapped", []))
    n_onehot_int = sum(n for _, n in stats.get("nominal_int_encoded", []))
    n_onehot_txt = sum(n for _, n in stats.get("str_one_hot", []))
    n_label = len(stats.get("str_label", []))
    n_total = stats.get("n_final_features", 0)
    n_num = max(n_total - (n_binary + n_onehot_int + n_onehot_txt + n_label), 0)

    labels = [
        "Numéricas contínuas / ordinais",
        "One-hot nominais (int)",
        "One-hot textuais",
        "Binárias mapeadas {1,2}→{1,0}",
        "LabelEncoder (fallback)",
    ]
    values = [n_num, n_onehot_int, n_onehot_txt, n_binary, n_label]
    colors = ["#3498db", "#27ae60", "#2ecc71", "#f39c12", "#e67e22"]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.barh(labels, values, color=colors)
    ax.set_xlabel("Nº de atributos finais gerados")
    ax.set_title(f"Perfil de encoding dos atributos finais (total = {n_total})",
                 pad=12)
    for b, v in zip(bars, values):
        ax.text(v + 0.3, b.get_y() + b.get_height() / 2, str(v),
                va="center", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5, axis="x")
    _save(fig, path)


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------
def generate_eda_plots(stats, output_dir):
    """
    Gera todos os gráficos de exploração em `output_dir`. Sobrescreve
    arquivos existentes. Retorna a lista de caminhos escritos.
    """
    os.makedirs(output_dir, exist_ok=True)
    outputs = []

    plots = [
        ("eda_01_class_distribution.png", plot_class_distribution),
        ("eda_02_column_removal_waterfall.png", plot_column_removal_waterfall),
        ("eda_03_missing_ratio_top.png", plot_missing_ratio),
        ("eda_04_cardinality_top.png", plot_cardinality),
        ("eda_05_ignorado_ratio.png", plot_ignorado_ratio),
        ("eda_06_age_distribution.png", plot_age_distribution),
        ("eda_07_age_by_class.png", plot_age_by_class),
        ("eda_08_final_correlation.png", plot_final_correlation_heatmap),
        ("eda_09_encoding_profile.png", plot_encoding_profile),
    ]
    for fname, fn in plots:
        path = os.path.join(output_dir, fname)
        try:
            fn(stats, path)
            outputs.append(path)
        except Exception as exc:
            print(f"[AVISO] Falha ao gerar {fname}: {exc}")
    return outputs
