import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Convergência do GA
# ---------------------------------------------------------------------------

def _plot_per_experiment_convergence(df_ga, output_path):
    """Curva de convergência (melhor fitness) por experimento individual."""
    plt.figure(figsize=(11, 6))
    for exp_id, group in df_ga.groupby("experimento"):
        plt.plot(
            group["geracao"],
            group["melhor_fitness"],
            alpha=0.35,
            linewidth=1.0,
            label=f"Exp {exp_id}" if exp_id < 5 else None,
        )
    plt.title("Convergência do AG por Experimento", fontsize=14, pad=12)
    plt.xlabel("Geração", fontsize=12)
    plt.ylabel("Melhor Fitness", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def _plot_average_convergence(df_ga, output_path):
    """Curva média dos melhores fitness ao longo dos experimentos (com banda de desvio)."""
    pivot = df_ga.pivot_table(
        index="geracao", columns="experimento", values="melhor_fitness"
    )
    # Para gerações em que algum experimento parou cedo, mantém o último valor conhecido
    pivot = pivot.ffill()

    mean_curve = pivot.mean(axis=1)
    std_curve = pivot.std(axis=1)

    plt.figure(figsize=(11, 6))
    plt.plot(mean_curve.index, mean_curve.values, color="#1f77b4",
             linewidth=2.5, label="Média dos melhores (20 experimentos)")
    plt.fill_between(
        mean_curve.index,
        mean_curve.values - std_curve.values,
        mean_curve.values + std_curve.values,
        color="#1f77b4",
        alpha=0.18,
        label="± 1 desvio-padrão",
    )
    plt.title("Curva Média de Convergência do AG", fontsize=14, pad=12)
    plt.xlabel("Geração", fontsize=12)
    plt.ylabel("Melhor Fitness", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def _plot_fitness_components(df_ga, output_path):
    """Compara melhor, médio e pior fitness no primeiro experimento (perfil típico)."""
    first_exp = df_ga["experimento"].min()
    group = df_ga[df_ga["experimento"] == first_exp]

    plt.figure(figsize=(11, 6))
    plt.plot(group["geracao"], group["melhor_fitness"],
             label="Melhor", color="#2ca02c", linewidth=2)
    plt.plot(group["geracao"], group["fitness_medio"],
             label="Médio", color="#1f77b4", linestyle="--")
    plt.plot(group["geracao"], group["pior_fitness"],
             label="Pior", color="#d62728", alpha=0.6)
    plt.title(f"Perfil de Fitness — Experimento {first_exp}", fontsize=14, pad=12)
    plt.xlabel("Geração", fontsize=12)
    plt.ylabel("Fitness", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def _plot_aggregate_fitness_bands(df_ga, output_path):
    """Média de melhor / médio / pior fitness com bandas em todos os experimentos."""
    def _mean_std(metric):
        piv = df_ga.pivot_table(index="geracao", columns="experimento",
                                values=metric).ffill()
        return piv.mean(axis=1), piv.std(axis=1)

    m_best, s_best = _mean_std("melhor_fitness")
    m_avg, s_avg = _mean_std("fitness_medio")
    m_wor, s_wor = _mean_std("pior_fitness")

    fig, ax = plt.subplots(figsize=(11, 6))
    for m, s, color, label in [
        (m_best, s_best, "#2ca02c", "Melhor (média)"),
        (m_avg, s_avg, "#1f77b4", "Médio (média)"),
        (m_wor, s_wor, "#d62728", "Pior (média)"),
    ]:
        ax.plot(m.index, m.values, linewidth=2, color=color, label=label)
        ax.fill_between(m.index, (m - s).values, (m + s).values,
                        color=color, alpha=0.12)
    ax.set_title("Fitness melhor/médio/pior — média ± desvio nos experimentos",
                 fontsize=14, pad=12)
    ax.set_xlabel("Geração", fontsize=12)
    ax.set_ylabel("Fitness", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Distribuição e desempenho da rede
# ---------------------------------------------------------------------------

def _plot_features_vs_score(df_nn, output_path):
    """Dispersão: nº de atributos usados versus F1-Score (colorido por geração)."""
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        df_nn["num_atributos_usados"],
        df_nn["f1_score"],
        c=df_nn["geracao"],
        cmap="viridis",
        alpha=0.65,
        edgecolors="white",
        s=55,
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label("Geração", fontsize=11)
    plt.title("Atributos selecionados x F1-Score", fontsize=14, pad=12)
    plt.xlabel("Quantidade de atributos ativos", fontsize=12)
    plt.ylabel("F1-Score (teste)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def _plot_f1_weighted_vs_macro(df_nn, output_path):
    """
    F1 weighted vs. F1 macro por avaliação. Distância entre as métricas
    revela viés por classe (base desbalanceada 62/20/18).
    """
    if "f1_macro" not in df_nn.columns:
        return
    plt.figure(figsize=(8, 8))
    plt.scatter(df_nn["f1_score"], df_nn["f1_macro"], alpha=0.4, s=25,
                c=df_nn["geracao"], cmap="viridis")
    lo = min(df_nn["f1_score"].min(), df_nn["f1_macro"].min())
    hi = max(df_nn["f1_score"].max(), df_nn["f1_macro"].max())
    plt.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1,
             label="F1 weighted = F1 macro")
    plt.title("F1 weighted vs. F1 macro — diagnóstico de viés por classe",
              fontsize=13, pad=12)
    plt.xlabel("F1 weighted (fitness primário)")
    plt.ylabel("F1 macro")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Estatísticas por experimento
# ---------------------------------------------------------------------------

def _plot_experiment_boxplot(summary_df, output_path):
    """Boxplot de fitness / F1 / #atributos entre os N experimentos."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    axes[0].boxplot(summary_df["melhor_fitness"], patch_artist=True,
                    boxprops=dict(facecolor="#2ca02c", alpha=0.5),
                    medianprops=dict(color="black", linewidth=2))
    axes[0].scatter(np.ones(len(summary_df)),
                    summary_df["melhor_fitness"],
                    color="black", alpha=0.6, s=25, zorder=3)
    axes[0].set_title("Melhor fitness")
    axes[0].set_xticks([])
    axes[0].grid(True, linestyle=":", alpha=0.5, axis="y")

    axes[1].boxplot([summary_df["melhor_f1_weighted"], summary_df["melhor_f1_macro"]],
                    patch_artist=True,
                    boxprops=dict(facecolor="#1f77b4", alpha=0.5),
                    medianprops=dict(color="black", linewidth=2))
    axes[1].set_xticklabels(["F1 weighted", "F1 macro"])
    axes[1].set_title("F1-Score do melhor cromossomo")
    axes[1].grid(True, linestyle=":", alpha=0.5, axis="y")

    axes[2].boxplot(summary_df["num_atributos_ativos"], patch_artist=True,
                    boxprops=dict(facecolor="#e67e22", alpha=0.5),
                    medianprops=dict(color="black", linewidth=2))
    axes[2].scatter(np.ones(len(summary_df)),
                    summary_df["num_atributos_ativos"],
                    color="black", alpha=0.6, s=25, zorder=3)
    axes[2].set_title("Atributos ativos no melhor")
    axes[2].set_xticks([])
    axes[2].grid(True, linestyle=":", alpha=0.5, axis="y")

    fig.suptitle(f"Distribuição das métricas ao longo de {len(summary_df)} experimentos",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_num_features_histogram(summary_df, output_path):
    """Histograma do número de atributos ativos nos melhores cromossomos."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    values = summary_df["num_atributos_ativos"]
    bins = min(20, max(5, values.nunique()))
    ax.hist(values, bins=bins, color="#1f77b4", edgecolor="white", alpha=0.85)
    ax.axvline(values.mean(), color="#c0392b", linestyle="--",
               linewidth=1.5, label=f"Média = {values.mean():.1f}")
    ax.axvline(values.median(), color="#2ca02c", linestyle="--",
               linewidth=1.5, label=f"Mediana = {values.median():.1f}")
    ax.set_title(f"Distribuição do nº de atributos ativos "
                 f"nos {len(values)} melhores cromossomos", pad=12)
    ax.set_xlabel("Nº de atributos ativos")
    ax.set_ylabel("Nº de experimentos")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_top_features(freq_df, output_path, top_n=25):
    """Barras horizontais dos TOP-N atributos mais selecionados nos melhores."""
    top = freq_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, max(6, 0.35 * len(top))))
    ax.barh(top["atributo"], top["frequencia"] * 100, color="#27ae60")
    ax.set_xlabel("Frequência de seleção (% dos experimentos)")
    ax.set_title(f"TOP-{top_n} atributos mais selecionados nos melhores cromossomos",
                 pad=12)
    ax.grid(True, linestyle=":", alpha=0.5, axis="x")
    for i, (v, c) in enumerate(zip(top["frequencia"] * 100, top["vezes_selecionado"])):
        ax.text(v + 0.5, i, f"{v:.0f}% ({int(c)}x)", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_time_per_experiment(summary_df, output_path):
    """Barras do tempo de execução por experimento (para análise de custo)."""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(summary_df["experimento"], summary_df["tempo_s"],
           color="#8e44ad", alpha=0.85)
    ax.set_xlabel("Experimento")
    ax.set_ylabel("Tempo (s)")
    ax.set_title("Tempo de execução por experimento", pad=12)
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_best_generation(summary_df, output_path):
    """Barras da geração em que cada experimento encontrou seu melhor cromossomo."""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(summary_df["experimento"], summary_df["geracao_do_melhor"],
           color="#16a085", alpha=0.9, label="Geração do melhor")
    ax.plot(summary_df["experimento"], summary_df["geracoes_executadas"],
            color="#c0392b", marker="o", linewidth=1.5,
            label="Geração de parada")
    ax.set_xlabel("Experimento")
    ax.set_ylabel("Geração")
    ax.set_title("Geração em que o melhor apareceu vs. geração de parada", pad=12)
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Matriz de confusão do melhor global
# ---------------------------------------------------------------------------

def _plot_confusion_matrix(json_path, output_path):
    """Heatmap da matriz de confusão salva pelo reporter."""
    if not os.path.exists(json_path):
        return
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    labels = payload["confusion_matrix"]["labels"]
    mat = np.array(payload["confusion_matrix"]["matrix"])

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    exp_id = payload["experimento_id"]
    f1w = payload["retreino"]["f1_weighted"]
    f1m = payload["retreino"]["f1_macro"]
    ax.set_title(
        f"Matriz de Confusão — Melhor Cromossomo (exp #{exp_id})\n"
        f"F1 weighted = {f1w:.3f} | F1 macro = {f1m:.3f}",
        pad=12,
    )

    thresh = mat.max() / 2.0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, str(int(mat[i, j])),
                    ha="center", va="center",
                    color="white" if mat[i, j] > thresh else "black",
                    fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_per_class_f1(json_path, output_path):
    """Barras do F1 por classe (do classification_report do melhor global)."""
    if not os.path.exists(json_path):
        return
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    report = payload.get("classification_report", {})
    labels = payload["confusion_matrix"]["labels"]

    f1s = [report.get(l, {}).get("f1-score", 0) for l in labels]
    supports = [int(report.get(l, {}).get("support", 0)) for l in labels]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(labels, f1s, color=["#1f77b4", "#2ca02c", "#e67e22"][:len(labels)])
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1-Score")
    ax.set_title("F1-Score por classe — melhor cromossomo global", pad=12)
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")
    for b, f, s in zip(bars, f1s, supports):
        ax.text(b.get_x() + b.get_width() / 2, f + 0.02,
                f"{f:.3f}\n(n={s})", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def plot_metrics(log_dir="logs", output_dir="plots", report_dir=None):
    """
    Lê os CSVs de log (e opcionalmente relatórios em `report_dir`) e gera:
      - convergência (por exp, média, componentes, bandas agregadas)
      - dispersão atributos x F1 e F1 weighted x F1 macro
      - boxplots, histograma, top-N, tempo e geração do melhor
      - matriz de confusão e F1 por classe do melhor global
    """
    os.makedirs(output_dir, exist_ok=True)

    ga_path = os.path.join(log_dir, "ga_metrics.csv")
    nn_path = os.path.join(log_dir, "nn_metrics.csv")

    if os.path.exists(ga_path):
        df_ga = pd.read_csv(ga_path)
        _plot_average_convergence(df_ga, os.path.join(output_dir, "ga_convergencia_media.png"))
        _plot_per_experiment_convergence(df_ga, os.path.join(output_dir, "ga_convergencia_por_experimento.png"))
        _plot_fitness_components(df_ga, os.path.join(output_dir, "ga_fitness_componentes.png"))
        _plot_aggregate_fitness_bands(df_ga, os.path.join(output_dir, "ga_fitness_componentes_medios.png"))
    else:
        print(f"[AVISO] Arquivo {ga_path} não encontrado para plotagem.")

    if os.path.exists(nn_path):
        df_nn = pd.read_csv(nn_path)
        _plot_features_vs_score(df_nn, os.path.join(output_dir, "nn_atributos_vs_f1.png"))
        _plot_f1_weighted_vs_macro(df_nn, os.path.join(output_dir, "nn_f1_weighted_vs_macro.png"))
    else:
        print(f"[AVISO] Arquivo {nn_path} não encontrado para plotagem.")

    if report_dir and os.path.isdir(report_dir):
        summary_path = os.path.join(report_dir, "resumo_experimentos.csv")
        freq_path = os.path.join(report_dir, "frequencia_atributos.csv")
        best_json = os.path.join(report_dir, "melhor_global.json")

        if os.path.exists(summary_path):
            summary = pd.read_csv(summary_path)
            _plot_experiment_boxplot(summary, os.path.join(output_dir, "exp_boxplot_metricas.png"))
            _plot_num_features_histogram(summary, os.path.join(output_dir, "exp_histograma_atributos.png"))
            _plot_time_per_experiment(summary, os.path.join(output_dir, "exp_tempo_por_experimento.png"))
            _plot_best_generation(summary, os.path.join(output_dir, "exp_geracao_do_melhor.png"))

        if os.path.exists(freq_path):
            freq = pd.read_csv(freq_path)
            _plot_top_features(freq, os.path.join(output_dir, "exp_top_atributos_selecionados.png"))

        if os.path.exists(best_json):
            _plot_confusion_matrix(best_json, os.path.join(output_dir, "nn_confusao_melhor.png"))
            _plot_per_class_f1(best_json, os.path.join(output_dir, "nn_f1_por_classe.png"))
