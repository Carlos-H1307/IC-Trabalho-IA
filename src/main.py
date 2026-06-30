import argparse
import os
import random
import time

import numpy as np

from data_loader import load_and_preprocess_data
from ga.algorithm import GeneticAlgorithm
from utils.logger import MetricsLogger
from utils.plotter import plot_metrics


# ----------------------------------------------------------------------------
# Hiperparâmetros do trabalho (conforme especificação)
# ----------------------------------------------------------------------------
POPULATION_SIZE = 150
CROSSOVER_RATE = 0.85
ELITE_SIZE = 10
GAP = 2
MAX_GENERATIONS = 200
STAGNATION_LIMIT = 20
TOURNAMENT_SIZE = 3
N_EXPERIMENTS = 20

# Caminhos
DATA_PATH = os.path.join("data", "dataset-short.xlsx")
LOG_DIR = "logs"
PLOT_DIR = "plots"

# Para tornar 20 experimentos × 200 gerações × 150 indivíduos tratáveis,
# usamos uma amostra estratificada da base. Defina 0 (via CLI) para a base inteira.
DEFAULT_SAMPLE_SIZE = 3000


def _set_seed(seed):
    """Fixa as sementes para tornar cada experimento reprodutível."""
    random.seed(seed)
    np.random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="GA + MLP para seleção de atributos (câncer do colo do útero)."
    )
    parser.add_argument("--data-path", default=DATA_PATH,
                        help="Caminho para o arquivo da base de dados.")
    parser.add_argument("--population", type=int, default=POPULATION_SIZE)
    parser.add_argument("--generations", type=int, default=MAX_GENERATIONS)
    parser.add_argument("--experiments", type=int, default=N_EXPERIMENTS)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
                        help="Amostra estratificada dos registros. Use 0 para usar a base inteira.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Nº de processos paralelos para mini-ciclos dentro de cada geração. "
                             "1 = serial. Recomendado: M3 Pro Max 36GB → 8; Ryzen 3600 16GB → 4-6.")
    parser.add_argument("--quick", action="store_true",
                        help="Modo rápido para teste: poucos experimentos e gerações.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.quick:
        # Modo smoke-test: valida o pipeline end-to-end rapidamente.
        # Com pop=20, elite=10, gap=2 → 5 mini-ciclos × 2 = 10 evals/gen
        # 2 exps × 5 gens × 10 = 100 avaliações totais → ~3 min serial
        args.experiments = 2
        args.generations = 5
        args.population = 20

    sample_size = args.sample_size if args.sample_size > 0 else None

    print("=" * 70)
    print("  Seleção de Atributos — Câncer do Colo do Útero")
    print("  Algoritmo Genético + Rede Neural (Backpropagation)")
    print("=" * 70)

    if not os.path.exists(args.data_path):
        print(f"[ERRO] Arquivo não encontrado em: {args.data_path}")
        print("       Coloque a base em data/ ou use --data-path.")
        return

    # ------------------------------------------------------------------
    # 1) Carregamento e pré-processamento
    # ------------------------------------------------------------------
    print(f"\n[INFO] Carregando base de dados de {args.data_path} ...")
    X, y, feature_names, n_classes = load_and_preprocess_data(
        args.data_path, sample_size=sample_size, random_state=42
    )
    L = X.shape[1]
    mutation_rate = 1.0 / L

    # ------------------------------------------------------------------
    # 2) Logger
    # ------------------------------------------------------------------
    logger = MetricsLogger(log_dir=LOG_DIR)

    # ------------------------------------------------------------------
    # 3) Resumo dos hiperparâmetros
    # ------------------------------------------------------------------
    n_non_elite = args.population - ELITE_SIZE
    mini_cycles = n_non_elite // GAP
    print("\n[INFO] Hiperparâmetros:")
    print(f"  População: {args.population}")
    print(f"  Crossover Uniforme, Pc = {CROSSOVER_RATE}")
    print(f"  Mutação: Pm = 1/L = {mutation_rate:.5f} (L = {L})")
    print(f"  Elitismo: {ELITE_SIZE}, Gap (steady-state): {GAP}")
    print(f"  Mini-ciclos por geração: {mini_cycles} → "
          f"{mini_cycles * GAP} avaliações novas por geração")
    print(f"  Gerações máx.: {args.generations} | Estagnação: {STAGNATION_LIMIT}")
    print(f"  Experimentos: {args.experiments}")
    print(f"  Workers (mini-ciclos paralelos): {args.workers}")
    print()

    # ------------------------------------------------------------------
    # 4) Loop de N experimentos (sempre serial, paralelismo é intra-geração)
    # ------------------------------------------------------------------
    results = []
    overall_start = time.time()

    for exp_id in range(args.experiments):
        print(f"=== Experimento {exp_id + 1}/{args.experiments} ===")
        _set_seed(seed=42 + exp_id)
        start = time.time()

        ga = GeneticAlgorithm(
            population_size=args.population,
            chromosome_length=L,
            crossover_rate=CROSSOVER_RATE,
            mutation_rate=mutation_rate,
            X=X,
            y=y,
            n_classes=n_classes,
            elite_size=ELITE_SIZE,
            gap=GAP,
            tournament_size=TOURNAMENT_SIZE,
            max_generations=args.generations,
            stagnation_limit=STAGNATION_LIMIT,
            logger=logger,
            experiment_id=exp_id,
            random_state=42 + exp_id,
            workers=args.workers,
        )

        best_genes, best_fitness, best_f1 = ga.evolve()
        elapsed = time.time() - start

        selected = [feature_names[i] for i, g in enumerate(best_genes) if g == 1]
        results.append({
            "experimento": exp_id,
            "best_fitness": best_fitness,
            "best_f1": best_f1,
            "n_atributos": len(selected),
            "atributos": selected,
            "tempo_s": elapsed,
        })

        print(f"  Melhor fitness: {best_fitness:.4f} | "
              f"F1-Score: {best_f1:.4f} | "
              f"Atributos ativos: {len(selected)}/{L} | "
              f"Tempo: {elapsed:.1f}s\n")

    total_elapsed = time.time() - overall_start

    # ------------------------------------------------------------------
    # 5) Resumo agregado
    # ------------------------------------------------------------------
    fitnesses = np.array([r["best_fitness"] for r in results])
    f1s = np.array([r["best_f1"] for r in results])
    nfeats = np.array([r["n_atributos"] for r in results])

    print("=" * 70)
    print("  RESUMO DOS EXPERIMENTOS")
    print("=" * 70)
    print(f"Fitness médio:  {fitnesses.mean():.4f} ± {fitnesses.std():.4f}")
    print(f"F1-Score médio: {f1s.mean():.4f} ± {f1s.std():.4f}")
    print(f"Atributos selecionados (média): {nfeats.mean():.1f} de {L}")
    print(f"Tempo total: {total_elapsed/60:.1f} min ({args.workers} workers)")

    best_idx = int(np.argmax(fitnesses))
    best = results[best_idx]
    print()
    print(f"Melhor experimento: #{best['experimento']}")
    print(f"  Fitness:  {best['best_fitness']:.4f}")
    print(f"  F1-Score: {best['best_f1']:.4f}")
    print(f"  Atributos selecionados ({best['n_atributos']}):")
    for feat in best["atributos"]:
        print(f"    - {feat}")

    # ------------------------------------------------------------------
    # 6) Gráficos
    # ------------------------------------------------------------------
    print("\n[INFO] Gerando gráficos...")
    plot_metrics(log_dir=LOG_DIR, output_dir=PLOT_DIR)
    print(f"[OK] Gráficos salvos em /{PLOT_DIR}/")
    print("\nProcesso finalizado.")


if __name__ == "__main__":
    main()
