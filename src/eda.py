"""
Análise Exploratória de Dados (EDA) da base de câncer do colo do útero.

Gera estatísticas descritivas, identifica inconsistências e produz gráficos
que documentam as características da base antes da execução do GA.

Saídas (em `reports/`):
  - eda_summary.txt          — resumo textual
  - eda_class_distribution.png — distribuição das classes-alvo
  - eda_missingness.png      — proporção de NaN por coluna (top 30)
  - eda_age_by_class.png     — boxplot de idade por classe
  - eda_temporal.png         — evolução temporal dos óbitos por classe
  - eda_correlation.png      — matriz de correlação das variáveis numéricas
  - eda_numeric_stats.csv    — describe() das variáveis numéricas
  - eda_categorical_stats.csv — contagem das principais categóricas
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Garante que data_loader (constantes) seja importável quando rodado como script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import (
    TARGET_COLUMN,
    TARGET_LEAK_COLUMNS,
    IGNORADO_CODE_COLUMNS,
    MAX_MISSING_RATIO,
    MAX_IGNORADO_RATIO,
)


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _save_text(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _plot_class_distribution(df, output_path):
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    pct = (counts / counts.sum() * 100).round(2)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color=["#d62728", "#ff7f0e", "#2ca02c"])
    for bar, p in zip(bars, pct.values):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h, f"{int(h):,}\n({p:.1f}%)",
                ha="center", va="bottom", fontsize=10)
    ax.set_title("Distribuição das classes (label_cid)", fontsize=14, pad=12)
    ax.set_xlabel("Código CID-10")
    ax.set_ylabel("Quantidade de registros")
    ax.set_ylim(0, counts.max() * 1.15)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def _plot_missingness(df, output_path, top_n=30):
    missing = df.isnull().mean().sort_values(ascending=False)
    missing = missing[missing > 0].head(top_n)
    if len(missing) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(missing))))
    ax.barh(missing.index[::-1], (missing.values * 100)[::-1],
            color=["#d62728" if v > MAX_MISSING_RATIO else "#1f77b4" for v in missing.values[::-1]])
    ax.set_xlabel("% de valores ausentes")
    ax.set_title(f"Top {len(missing)} colunas com valores ausentes", fontsize=14, pad=12)
    ax.axvline(MAX_MISSING_RATIO * 100, color="red", linestyle="--", alpha=0.6,
               label=f"Limite ({int(MAX_MISSING_RATIO*100)}%) → descarte")
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def _plot_age_by_class(df, output_path):
    if "idade_obito_anos" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    classes = sorted(df[TARGET_COLUMN].dropna().unique())
    data = [df.loc[df[TARGET_COLUMN] == c, "idade_obito_anos"].dropna() for c in classes]
    bp = ax.boxplot(data, tick_labels=classes, patch_artist=True, widths=0.55,
                    medianprops=dict(color="black", linewidth=2))
    colors = ["#d62728", "#ff7f0e", "#2ca02c"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_title("Distribuição de idade no óbito por classe", fontsize=14, pad=12)
    ax.set_xlabel("Classe (CID)")
    ax.set_ylabel("Idade no óbito (anos)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def _plot_temporal_evolution(df, output_path):
    if "DTOBITO" not in df.columns:
        return
    ano = df["DTOBITO"].astype(str).str.zfill(8).str[-4:].astype(int)
    df_tmp = df.assign(ano_obito=ano)
    crosstab = pd.crosstab(df_tmp["ano_obito"], df_tmp[TARGET_COLUMN], normalize="index") * 100

    fig, ax = plt.subplots(figsize=(10, 5.5))
    crosstab.plot(ax=ax, marker="o", linewidth=2,
                  color=["#d62728", "#ff7f0e", "#2ca02c"])
    ax.set_title("Proporção das classes por ano do óbito", fontsize=14, pad=12)
    ax.set_xlabel("Ano do óbito")
    ax.set_ylabel("% de óbitos no ano")
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(title="Classe", loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def _plot_correlation(df, output_path, top_n=15):
    numeric = df.select_dtypes(include=[np.number])
    # Variáveis com variação real
    numeric = numeric.loc[:, numeric.nunique() > 1]
    if numeric.shape[1] < 2:
        return
    # Limita a top_n colunas com maior variância para legibilidade
    variances = numeric.var().sort_values(ascending=False)
    cols = variances.head(top_n).index.tolist()
    corr = numeric[cols].corr()

    fig, ax = plt.subplots(figsize=(0.6 * len(cols) + 2, 0.6 * len(cols) + 2))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=60, ha="right", fontsize=9)
    ax.set_yticklabels(cols, fontsize=9)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if abs(v) < 0.6 else "white")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Matriz de correlação (top {len(cols)} variáveis numéricas)",
                 fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def _summary_lines(df):
    """Constrói o resumo textual da EDA."""
    lines = []
    lines.append("=" * 72)
    lines.append("  ANÁLISE EXPLORATÓRIA DOS DADOS — CÂNCER DO COLO DO ÚTERO")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"Total de registros: {len(df):,}")
    lines.append(f"Total de colunas:   {df.shape[1]}")
    lines.append("")

    lines.append("--- 1. DISTRIBUIÇÃO DA CLASSE-ALVO (label_cid) ---")
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    pct = (counts / counts.sum() * 100).round(2)
    for c in counts.index:
        lines.append(f"  {c}: {counts[c]:>7,} ({pct[c]:>5.2f}%)")
    lines.append("")
    lines.append(f"  >> Desbalanceamento: razão maior/menor = {counts.max()/counts.min():.2f}x")
    lines.append("")

    lines.append("--- 2. INCONSISTÊNCIAS DETECTADAS ---")
    n_dup = int(df.duplicated().sum())
    lines.append(f"  Duplicatas exatas: {n_dup}")
    n_no_target = int(df[TARGET_COLUMN].isnull().sum())
    lines.append(f"  Registros sem alvo (label_cid nulo): {n_no_target}")
    if "idade_obito_anos" in df.columns:
        ages = df["idade_obito_anos"]
        n_age_nan = int(ages.isnull().sum())
        n_age_invalid = int(((ages < 0) | (ages > 120)).sum())
        lines.append(f"  Idades nulas: {n_age_nan}")
        lines.append(f"  Idades fora de [0, 120]: {n_age_invalid}")
        lines.append(f"  Idade min/max observada: {ages.min():.0f} / {ages.max():.0f}")
    if "SEXO" in df.columns:
        sexo_vals = df["SEXO"].value_counts().to_dict()
        lines.append(f"  Distribuição de SEXO: {sexo_vals}")
        if df["SEXO"].nunique() == 1:
            lines.append("    >> SEXO é constante (todas as pacientes femininas) — coluna descartada")
    if "DTOBITO" in df.columns and "DTNASC" in df.columns:
        try:
            obito_year = df["DTOBITO"].astype(str).str.zfill(8).str[-4:].astype(int)
            nasc_year = df["DTNASC"].fillna(0).astype(np.int64).astype(str).str.zfill(8).str[-4:].astype(int)
            n_invalid_dates = int(((nasc_year > obito_year) & (nasc_year > 1900)).sum())
            lines.append(f"  Datas inconsistentes (nasc > óbito): {n_invalid_dates}")
            lines.append(f"  Período coberto: {obito_year.min()} a {obito_year.max()}")
        except Exception:
            pass
    lines.append("")

    lines.append("--- 3. COLUNAS CONSTANTES (variância zero) ---")
    constant = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    for c in constant:
        v = df[c].iloc[0] if not df[c].isnull().all() else "(todas NaN)"
        lines.append(f"  {c}: único valor = {v!r}")
    if not constant:
        lines.append("  (nenhuma)")
    lines.append("")

    lines.append("--- 4. VALORES AUSENTES (top 15) ---")
    missing = df.isnull().mean().sort_values(ascending=False)
    missing = missing[missing > 0].head(15)
    for col, ratio in missing.items():
        flag = "  ←  DESCARTAR" if ratio > MAX_MISSING_RATIO else ""
        lines.append(f"  {col:<25} {ratio*100:>6.2f}%{flag}")
    if missing.empty:
        lines.append("  (nenhuma)")
    lines.append("")

    lines.append("--- 5. CÓDIGO 9 ('IGNORADO' no DATASUS) ---")
    for col in IGNORADO_CODE_COLUMNS:
        if col not in df.columns:
            continue
        ratio = (df[col] == 9).mean()
        flag = "  ←  DESCARTAR" if ratio > MAX_IGNORADO_RATIO else ""
        lines.append(f"  {col:<15} {ratio*100:>6.2f}% com valor 9{flag}")
    lines.append("")

    lines.append("--- 6. ESTATÍSTICAS DESCRITIVAS — IDADE POR CLASSE ---")
    if "idade_obito_anos" in df.columns:
        for c in sorted(df[TARGET_COLUMN].dropna().unique()):
            s = df.loc[df[TARGET_COLUMN] == c, "idade_obito_anos"].dropna()
            lines.append(f"  {c}: n={len(s):>6,}  média={s.mean():>5.1f}  "
                         f"mediana={s.median():>4.0f}  std={s.std():>5.1f}  "
                         f"min={s.min():>3.0f}  max={s.max():>3.0f}")
    lines.append("")

    lines.append("--- 7. VAZAMENTO DE ALVO (colunas a remover) ---")
    leak = [c for c in TARGET_LEAK_COLUMNS if c in df.columns]
    lines.append(f"  {len(leak)} colunas removidas por descrever diretamente a causa:")
    for c in leak:
        lines.append(f"    - {c}")
    lines.append("")

    return lines


def run_eda(data_path, output_dir="reports"):
    """Executa todo o pipeline de EDA e salva resultados em `output_dir`."""
    _ensure_dir(output_dir)

    print(f"[EDA] Carregando dados de {data_path} ...")
    df = pd.read_excel(data_path) if data_path.endswith(".xlsx") else pd.read_csv(data_path)
    print(f"[EDA] Base carregada: {df.shape[0]} registros, {df.shape[1]} colunas")

    # Resumo textual
    lines = _summary_lines(df)
    summary_text = "\n".join(lines)
    print()
    print(summary_text)

    summary_path = os.path.join(output_dir, "eda_summary.txt")
    _save_text(summary_path, lines)
    print(f"\n[EDA] Resumo salvo em {summary_path}")

    # Estatísticas descritivas numéricas
    numeric_stats = df.select_dtypes(include=[np.number]).describe().T
    numeric_stats_path = os.path.join(output_dir, "eda_numeric_stats.csv")
    numeric_stats.to_csv(numeric_stats_path, encoding="utf-8")
    print(f"[EDA] Estatísticas numéricas salvas em {numeric_stats_path}")

    # Contagem das categóricas mais importantes
    cat_rows = []
    for col in df.columns:
        s = df[col]
        if s.nunique(dropna=False) <= 20:
            counts = s.value_counts(dropna=False).head(20)
            for val, n in counts.items():
                cat_rows.append({"coluna": col, "valor": val, "n": n,
                                 "pct": round(n / len(df) * 100, 3)})
    cat_path = os.path.join(output_dir, "eda_categorical_stats.csv")
    pd.DataFrame(cat_rows).to_csv(cat_path, index=False, encoding="utf-8")
    print(f"[EDA] Estatísticas categóricas salvas em {cat_path}")

    # Gráficos
    print("[EDA] Gerando gráficos...")
    _plot_class_distribution(df, os.path.join(output_dir, "eda_class_distribution.png"))
    _plot_missingness(df, os.path.join(output_dir, "eda_missingness.png"))
    _plot_age_by_class(df, os.path.join(output_dir, "eda_age_by_class.png"))
    _plot_temporal_evolution(df, os.path.join(output_dir, "eda_temporal.png"))
    _plot_correlation(df, os.path.join(output_dir, "eda_correlation.png"))
    print(f"[EDA] Gráficos salvos em {output_dir}/")
    print("[EDA] Concluído.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="EDA da base de câncer do colo do útero.")
    parser.add_argument("--data-path", default="data/dataset-short.xlsx")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()
    run_eda(args.data_path, args.output_dir)


if __name__ == "__main__":
    main()
