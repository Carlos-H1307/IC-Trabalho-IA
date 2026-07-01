import argparse
import os
import random
import shutil
import time

import numpy as np

from data_loader import load_and_preprocess_data
from ga.algorithm import GeneticAlgorithm
from utils.logger import MetricsLogger
from utils.plotter import plot_metrics
from utils import reporter
from utils import eda


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
REPORT_DIR = "reports"

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
                        help="Nº de experimentos a rodar em paralelo (processos). "
                             "1 = serial. Sugestões: M3 Pro Max 36GB → 8; "
                             "Ryzen 3600 16GB → 4–6.")
    parser.add_argument("--quick", action="store_true",
                        help="Modo rápido para teste: poucos experimentos e gerações.")
    parser.add_argument("--fitness-repeats", type=int, default=1,
                        help="Número K de retreinos da MLP por cromossomo. "
                             "F1 do fitness = média das K execuções (seeds "
                             "distintas). K > 1 reduz o ruído do estimador em "
                             "sqrt(K) ao custo de K× runtime. Refs: "
                             "Bengio & Grandvalet (2004); Nadeau & Bengio (2003).")
    return parser.parse_args()


# ----------------------------------------------------------------------------
# Função executada por experimento (em processo separado se workers > 1)
# ----------------------------------------------------------------------------
def run_single_experiment(
    exp_id,
    X,
    y,
    n_classes,
    feature_names,
    feature_groups,
    population_size,
    max_generations,
    worker_log_dir,
    fitness_repeats=1,
):
    """
    Executa um experimento completo do GA e retorna o resultado.
    Cada chamada é independente — projetada para rodar em processos paralelos
    via joblib quando `workers > 1`.

    Codificação binária agrupada: `chromosome_length = len(feature_groups)`.
    Cada gene ativa/desativa um atributo semântico original inteiro (com
    todas as suas dummies one-hot, se aplicável). Ver `_build_feature_groups`.
    """
    # Imports locais para que cada subprocesso tenha sua própria instância
    import time as _time
    from ga.algorithm import GeneticAlgorithm
    from utils.logger import MetricsLogger

    _set_seed(seed=42 + exp_id)
    start = _time.time()

    # Cada experimento escreve em sua própria pasta para evitar contenção
    exp_log_dir = os.path.join(worker_log_dir, f"exp{exp_id}")
    os.makedirs(exp_log_dir, exist_ok=True)
    logger = MetricsLogger(log_dir=exp_log_dir)

    # L = número de atributos semânticos originais (grupos), não de colunas.
    # Pm = 1/L segue o spec, calculado sobre o novo L.
    L = len(feature_groups)
    mutation_rate = 1.0 / L

    ga = GeneticAlgorithm(
        population_size=population_size,
        chromosome_length=L,
        crossover_rate=CROSSOVER_RATE,
        mutation_rate=mutation_rate,
        X=X,
        y=y,
        n_classes=n_classes,
        elite_size=ELITE_SIZE,
        gap=GAP,
        tournament_size=TOURNAMENT_SIZE,
        max_generations=max_generations,
        stagnation_limit=STAGNATION_LIMIT,
        logger=logger,
        experiment_id=exp_id,
        random_state=42 + exp_id,
        feature_groups=feature_groups,
        fitness_repeats=fitness_repeats,
    )

    result = ga.evolve()
    elapsed = _time.time() - start

    best_genes = result["best_genes"]
    # Cada bit ativo representa um grupo; expandimos para nomes de atributos
    # semânticos (nomes de grupo) e também para as colunas que a MLP viu.
    selected_group_idx = [i for i, g in enumerate(best_genes) if g == 1]
    selected_group_names = [feature_groups[i][0] for i in selected_group_idx]
    selected_col_idx = [
        c for i in selected_group_idx for c in feature_groups[i][1]
    ]
    selected_col_names = [feature_names[c] for c in selected_col_idx]
    return {
        "experimento": exp_id,
        "best_fitness": result["best_fitness"],
        "best_f1": result["best_f1"],
        "best_f1_macro": result["best_f1_macro"],
        "best_generation": result["best_generation"],
        "total_generations": result["total_generations"],
        "n_atributos": len(selected_group_names),
        "atributos": selected_group_names,
        "atributos_idx": selected_group_idx,
        "atributos_colunas": selected_col_names,
        "atributos_colunas_idx": selected_col_idx,
        "n_colunas_mlp": len(selected_col_idx),
        "best_genes": "".join(map(str, best_genes)),
        "random_state": 42 + exp_id,
        "tempo_s": elapsed,
    }


def _merge_worker_logs(worker_log_dir, final_log_dir, exp_ids):
    """
    Consolida os CSVs de cada experimento em um único arquivo no diretório
    final. Os arquivos por experimento ficam em
    `worker_log_dir/exp{N}/ga_metrics.csv` (e `nn_metrics.csv`).
    """
    os.makedirs(final_log_dir, exist_ok=True)
    for filename in ["ga_metrics.csv", "nn_metrics.csv"]:
        merged_path = os.path.join(final_log_dir, filename)
        header_written = False
        with open(merged_path, "w", encoding="utf-8") as out:
            for exp_id in exp_ids:
                src = os.path.join(worker_log_dir, f"exp{exp_id}", filename)
                if not os.path.exists(src):
                    continue
                with open(src, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                if not lines:
                    continue
                if not header_written:
                    out.write(lines[0])
                    header_written = True
                out.writelines(lines[1:])


def main():
    args = parse_args()

    if args.quick:
        # Modo smoke-test: valida o pipeline end-to-end rapidamente.
        args.experiments = 2
        args.generations = 20
        args.population = 30

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
    (
        X, y, feature_names, n_classes, class_names, data_stats, feature_groups
    ) = load_and_preprocess_data(
        args.data_path, sample_size=sample_size, random_state=42
    )
    # L = número de atributos semânticos originais (grupos), não colunas.
    # Sob codificação binária agrupada, cada gene ativa um grupo inteiro
    # de dummies one-hot; L reflete o "número total de atributos" do spec.
    L = len(feature_groups)
    n_columns_total = X.shape[1]
    mutation_rate = 1.0 / L

    # Gera gráficos de exploração da base ANTES do GA — eles não dependem do
    # resultado dos experimentos e permitem inspeção imediata do pré-processamento.
    os.makedirs(PLOT_DIR, exist_ok=True)
    print(f"[INFO] Gerando gráficos de exploração em /{PLOT_DIR}/ ...")
    eda_paths = eda.generate_eda_plots(data_stats, PLOT_DIR)
    print(f"[OK] {len(eda_paths)} gráficos exploratórios salvos.")

    # ------------------------------------------------------------------
    # 2) Preparação dos logs (cada experimento escreve no seu subdir;
    #    no final consolidamos em logs/ga_metrics.csv e nn_metrics.csv)
    # ------------------------------------------------------------------
    worker_log_dir = os.path.join(LOG_DIR, "_workers")
    if os.path.exists(worker_log_dir):
        shutil.rmtree(worker_log_dir)
    os.makedirs(worker_log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 3) Resumo dos hiperparâmetros
    # ------------------------------------------------------------------
    print("\n[INFO] Hiperparâmetros:")
    print(f"  População: {args.population}")
    print(f"  Crossover Uniforme, Pc = {CROSSOVER_RATE}")
    print(f"  Mutação: Pm = 1/L = {mutation_rate:.5f} (L = {L} grupos, "
          f"{n_columns_total} colunas na MLP)")
    print(f"  Elitismo: {ELITE_SIZE}, Gap (steady-state): {GAP}")
    print(f"  Gerações máx.: {args.generations} | Estagnação: {STAGNATION_LIMIT}")
    print(f"  Experimentos: {args.experiments}")
    print(f"  Workers (experimentos em paralelo): {args.workers}")
    if args.fitness_repeats > 1:
        print(f"  Fitness com múltiplas seeds: K = {args.fitness_repeats} "
              f"(runtime × {args.fitness_repeats}, ruído / sqrt({args.fitness_repeats}))")
    print()

    # ------------------------------------------------------------------
    # 4) Loop de experimentos (paralelo via joblib se workers > 1)
    # ------------------------------------------------------------------
    overall_start = time.time()
    experiment_args = [
        dict(
            exp_id=exp_id,
            X=X, y=y, n_classes=n_classes,
            feature_names=feature_names,
            feature_groups=feature_groups,
            population_size=args.population,
            max_generations=args.generations,
            worker_log_dir=worker_log_dir,
            fitness_repeats=args.fitness_repeats,
        )
        for exp_id in range(args.experiments)
    ]

    if args.workers > 1:
        from joblib import Parallel, delayed
        print(f"[INFO] Rodando {args.experiments} experimentos em paralelo "
              f"({args.workers} processos via joblib/loky)...\n")
        results = Parallel(n_jobs=args.workers, backend="loky", verbose=5)(
            delayed(run_single_experiment)(**kwargs) for kwargs in experiment_args
        )
        # joblib pode embaralhar a ordem; ordenamos pelo exp_id
        results.sort(key=lambda r: r["experimento"])
        for r in results:
            print(f"  [exp {r['experimento']:02d}] fitness={r['best_fitness']:.4f} | "
                  f"F1={r['best_f1']:.4f} | "
                  f"ativos={r['n_atributos']}/{L} | "
                  f"tempo={r['tempo_s']:.1f}s")
    else:
        # Serial
        results = []
        for kwargs in experiment_args:
            print(f"=== Experimento {kwargs['exp_id'] + 1}/{args.experiments} ===")
            r = run_single_experiment(**kwargs)
            results.append(r)
            print(f"  Melhor fitness: {r['best_fitness']:.4f} | "
                  f"F1-Score: {r['best_f1']:.4f} | "
                  f"Atributos ativos: {r['n_atributos']}/{L} | "
                  f"Tempo: {r['tempo_s']:.1f}s\n")

    total_elapsed = time.time() - overall_start

    # ------------------------------------------------------------------
    # 5) Consolida logs em logs/ga_metrics.csv e logs/nn_metrics.csv
    # ------------------------------------------------------------------
    _merge_worker_logs(worker_log_dir, LOG_DIR, [r["experimento"] for r in results])

    # ------------------------------------------------------------------
    # 6) Resumo agregado
    # ------------------------------------------------------------------
    fitnesses = np.array([r["best_fitness"] for r in results])
    f1s = np.array([r["best_f1"] for r in results])
    nfeats = np.array([r["n_atributos"] for r in results])

    print()
    print("=" * 70)
    print("  RESUMO DOS EXPERIMENTOS")
    print("=" * 70)
    print(f"Fitness médio:  {fitnesses.mean():.4f} ± {fitnesses.std():.4f}")
    print(f"F1-Score médio: {f1s.mean():.4f} ± {f1s.std():.4f}")
    print(f"Atributos selecionados (média): {nfeats.mean():.1f} de {L}")
    print(f"Tempo total: {total_elapsed/60:.1f} min "
          f"({args.workers} {'workers' if args.workers > 1 else 'worker'})")

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
    # 7) Relatórios detalhados (para uso na análise dos entregáveis)
    # ------------------------------------------------------------------
    os.makedirs(REPORT_DIR, exist_ok=True)
    print(f"\n[INFO] Gerando relatórios detalhados em /{REPORT_DIR}/ ...")

    reporter.save_feature_names(
        feature_names, os.path.join(REPORT_DIR, "feature_names.csv")
    )
    reporter.save_hyperparameters(
        args, mutation_rate, L,
        constants={
            "CROSSOVER_RATE": CROSSOVER_RATE,
            "ELITE_SIZE": ELITE_SIZE,
            "GAP": GAP,
            "STAGNATION_LIMIT": STAGNATION_LIMIT,
            "TOURNAMENT_SIZE": TOURNAMENT_SIZE,
        },
        path=os.path.join(REPORT_DIR, "hiperparametros.json"),
    )
    reporter.save_dataset_info(
        X, y, feature_names, n_classes, class_names,
        os.path.join(REPORT_DIR, "dataset_info.json"),
    )
    reporter.save_experiment_summary(
        results, L, os.path.join(REPORT_DIR, "resumo_experimentos.csv")
    )
    reporter.save_convergence_aggregate(
        os.path.join(LOG_DIR, "ga_metrics.csv"),
        os.path.join(REPORT_DIR, "convergencia_agregada.csv"),
    )
    reporter.save_feature_frequency(
        results, feature_groups,
        os.path.join(REPORT_DIR, "frequencia_atributos.csv"),
    )
    print("[INFO] Retreinando o melhor cromossomo para matriz de confusão...")
    best_payload = reporter.save_best_global_report(
        results, feature_names, feature_groups, X, y, n_classes, class_names,
        json_path=os.path.join(REPORT_DIR, "melhor_global.json"),
        confusion_csv_path=os.path.join(REPORT_DIR, "nn_confusao_melhor.csv"),
    )
    print(
        f"[OK] Melhor global: exp #{best_payload['experimento_id']} | "
        f"F1w={best_payload['retreino']['f1_weighted']:.4f} | "
        f"F1m={best_payload['retreino']['f1_macro']:.4f}"
    )

    # ------------------------------------------------------------------
    # 8) Gráficos
    # ------------------------------------------------------------------
    print("\n[INFO] Gerando gráficos...")
    plot_metrics(log_dir=LOG_DIR, output_dir=PLOT_DIR, report_dir=REPORT_DIR)
    print(f"[OK] Gráficos salvos em /{PLOT_DIR}/")
    print("\nProcesso finalizado.")


if __name__ == "__main__":
    main()
