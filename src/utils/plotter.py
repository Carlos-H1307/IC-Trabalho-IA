import os

import matplotlib.pyplot as plt
import pandas as pd


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


def plot_metrics(log_dir="logs", output_dir="plots"):
    """
    Lê os CSVs de log e gera:
      - curva média de convergência (média dos melhores em N experimentos)
      - curvas individuais por experimento
      - perfil de fitness (melhor/médio/pior) do primeiro experimento
      - dispersão atributos x F1-Score
    """
    os.makedirs(output_dir, exist_ok=True)

    ga_path = os.path.join(log_dir, "ga_metrics.csv")
    nn_path = os.path.join(log_dir, "nn_metrics.csv")

    if os.path.exists(ga_path):
        df_ga = pd.read_csv(ga_path)
        _plot_average_convergence(df_ga, os.path.join(output_dir, "ga_convergencia_media.png"))
        _plot_per_experiment_convergence(df_ga, os.path.join(output_dir, "ga_convergencia_por_experimento.png"))
        _plot_fitness_components(df_ga, os.path.join(output_dir, "ga_fitness_componentes.png"))
    else:
        print(f"[AVISO] Arquivo {ga_path} não encontrado para plotagem.")

    if os.path.exists(nn_path):
        df_nn = pd.read_csv(nn_path)
        _plot_features_vs_score(df_nn, os.path.join(output_dir, "nn_atributos_vs_f1.png"))
    else:
        print(f"[AVISO] Arquivo {nn_path} não encontrado para plotagem.")
