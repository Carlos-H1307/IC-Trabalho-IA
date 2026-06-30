import random

from ga.chromosome import Chromosome
from ga.fitness import evaluate_chromosome, linear_scale_population


# ---------------------------------------------------------------------------
# Worker function (top-level, picklable) — usado pelo joblib em workers > 1
# ---------------------------------------------------------------------------
def _mini_cycle_worker(
    parent1,
    parent2,
    X,
    y,
    n_classes,
    crossover_rate,
    mutation_rate,
    chromosome_id_base,
    random_state,
    gap,
):
    """
    Executa um mini-ciclo Gap=2 em um worker independente.

    Recebe dois pais já selecionados (a seleção por torneio acontece no
    processo principal, sobre um snapshot da população — assim todos os
    mini-ciclos de uma geração usam o mesmo pool de pais).

    Retorna a lista de filhos avaliados (cada um com `fitness`, `f1_score`
    e `nn_metrics` preenchidos).
    """
    # Re-semeia para reproducibilidade dentro do worker
    rng_seed = hash((random_state, chromosome_id_base)) & 0xFFFFFFFF
    random.seed(rng_seed)

    child1, child2 = parent1.crossover(parent2, crossover_rate)
    child1.mutate(mutation_rate)
    child2.mutate(mutation_rate)
    children = [child1, child2][:gap]

    for j, child in enumerate(children):
        evaluate_chromosome(
            child,
            X,
            y,
            n_classes=n_classes,
            chromosome_id=f"{chromosome_id_base}_child{j}",
            logger=None,                 # worker não loga; main loga depois
            generation=None,
            fitness_cache=None,          # cache local ao worker não ajudaria
            random_state=random_state,
        )

    return children


class GeneticAlgorithm:
    """
    Algoritmo Genético com Steady-State Gap=2 + Elitismo de 10.

    Cada "geração completa" do GA é composta por múltiplos mini-ciclos de
    substituição (gap=2). Em cada mini-ciclo:
      1. Seleciona 2 pais via torneio (sobre snapshot da população na gen)
      2. Aplica Crossover Uniforme com probabilidade Pc → 2 filhos
      3. Aplica mutação bit-flip nos filhos (Pm = 1/L)
      4. Avalia os 2 filhos via MLP + função de aptidão

    Ao final dos mini-ciclos, substitui-se os (pop_size − elite_size) piores
    indivíduos pelos filhos gerados; os 10 melhores são preservados.

    Quantidade de mini-ciclos por geração:
        n_mini_cycles = (population_size − elite_size) / gap
    Para população=150, elite=10, gap=2 → 70 mini-ciclos × 2 filhos = 140
    novas avaliações por geração (toda a parte não-elite é renovada).

    Paralelismo (workers > 1):
      - Os mini-ciclos dentro de uma geração são executados em paralelo via
        joblib (processos independentes via backend 'loky').
      - Todos usam o MESMO snapshot da população (selecionado no início da
        geração), o que é necessário para independência computacional.
      - Trade-off: na execução serial estrita, os filhos de um mini-ciclo
        poderiam servir de pais no próximo. Em paralelo isso é abdicado,
        e o GA fica mais próximo de um geracional com elitismo — mas em
        troca ganha-se um speedup quase linear em N workers.

    Critério de parada: 200 gerações OU 20 gerações sem melhoria.
    """

    def __init__(
        self,
        population_size,
        chromosome_length,
        crossover_rate,
        mutation_rate,
        X,
        y,
        n_classes,
        elite_size=10,
        gap=2,
        tournament_size=3,
        max_generations=200,
        stagnation_limit=20,
        logger=None,
        experiment_id=0,
        fitness_cache=None,
        random_state=42,
        workers=1,
    ):
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.X = X
        self.y = y
        self.n_classes = n_classes
        self.elite_size = elite_size
        self.gap = gap
        self.tournament_size = tournament_size
        self.max_generations = max_generations
        self.stagnation_limit = stagnation_limit
        self.logger = logger
        self.experiment_id = experiment_id
        self.fitness_cache = fitness_cache if fitness_cache is not None else {}
        self.random_state = random_state
        self.workers = workers

        # Quantidade de mini-ciclos por geração completa
        n_non_elite = population_size - elite_size
        self.mini_cycles_per_generation = n_non_elite // gap

        # População inicial aleatória
        self.population = [
            Chromosome(chromosome_length) for _ in range(population_size)
        ]

        # Histórico de melhor fitness por geração (para curva de convergência)
        self.history_best = []
        self.history_avg = []
        self.history_worst = []

    def _evaluate_population(self):
        """Avalia todos os indivíduos da população inicial (gen 0)."""
        for i, chromo in enumerate(self.population):
            if chromo.fitness < 0:
                cid = f"exp{self.experiment_id}_init_ind{i}"
                evaluate_chromosome(
                    chromo,
                    self.X,
                    self.y,
                    n_classes=self.n_classes,
                    chromosome_id=cid,
                    logger=self.logger,
                    generation=0,
                    fitness_cache=self.fitness_cache,
                    random_state=self.random_state,
                )

    def _select_parent(self):
        """Seleção por torneio usando fitness escalado (normalização linear)."""
        tournament = random.sample(self.population, self.tournament_size)
        return max(tournament, key=lambda c: c.scaled_fitness)

    def _select_parent_pairs(self, n_pairs):
        """Seleciona n_pairs pares de pais via torneio (no processo principal)."""
        pairs = []
        for _ in range(n_pairs):
            p1 = self._select_parent()
            p2 = self._select_parent()
            pairs.append((p1, p2))
        return pairs

    def _do_generation_serial(self, generation, parent_pairs):
        """Executa todos os mini-ciclos serialmente (workers == 1)."""
        all_children = []
        for cycle_idx, (p1, p2) in enumerate(parent_pairs):
            child1, child2 = p1.crossover(p2, self.crossover_rate)
            child1.mutate(self.mutation_rate)
            child2.mutate(self.mutation_rate)
            children = [child1, child2][: self.gap]
            for j, child in enumerate(children):
                cid = f"exp{self.experiment_id}_gen{generation}_cyc{cycle_idx}_child{j}"
                evaluate_chromosome(
                    child,
                    self.X,
                    self.y,
                    n_classes=self.n_classes,
                    chromosome_id=cid,
                    logger=None,  # log após coletar tudo (consistência com modo paralelo)
                    generation=generation,
                    fitness_cache=self.fitness_cache,
                    random_state=self.random_state,
                )
            all_children.extend(children)
        return all_children

    def _do_generation_parallel(self, generation, parent_pairs):
        """Executa os mini-ciclos em paralelo via joblib (workers > 1)."""
        from joblib import Parallel, delayed

        tasks = [
            delayed(_mini_cycle_worker)(
                p1,
                p2,
                self.X,
                self.y,
                self.n_classes,
                self.crossover_rate,
                self.mutation_rate,
                f"exp{self.experiment_id}_gen{generation}_cyc{cycle_idx}",
                self.random_state,
                self.gap,
            )
            for cycle_idx, (p1, p2) in enumerate(parent_pairs)
        ]

        # backend 'loky' = subprocessos independentes (cada um com seu TF)
        results = Parallel(n_jobs=self.workers, backend="loky")(tasks)

        all_children = []
        for children in results:
            all_children.extend(children)

        # Atualiza cache central com os resultados dos workers (que não tinham cache)
        for child in all_children:
            key = child.key()
            if key not in self.fitness_cache:
                self.fitness_cache[key] = (
                    child.f1_score,
                    child.fitness,
                    getattr(child, "nn_metrics", None),
                )
        return all_children

    def _log_children_nn_metrics(self, children):
        """Loga as métricas de NN dos filhos no logger principal."""
        if not self.logger:
            return
        for child in children:
            metrics = getattr(child, "nn_metrics", None)
            if metrics:
                self.logger.log_nn_metrics(**metrics)

    def _log_generation(self, generation):
        fitnesses = [c.fitness for c in self.population]
        best = max(fitnesses)
        avg = sum(fitnesses) / len(fitnesses)
        worst = min(fitnesses)
        self.history_best.append(best)
        self.history_avg.append(avg)
        self.history_worst.append(worst)

        if self.logger:
            best_chromo = max(self.population, key=lambda c: c.fitness)
            self.logger.log_ga_metrics(
                experiment_id=self.experiment_id,
                generation=generation,
                best_fitness=best,
                avg_fitness=avg,
                worst_fitness=worst,
                best_chromosome=best_chromo.genes,
                num_active=best_chromo.num_active,
                best_f1=best_chromo.f1_score,
            )
        return best

    def evolve(self):
        """
        Loop principal do GA. Retorna (best_chromosome_genes, best_fitness, best_f1).
        """
        # 1) Avalia população inicial (sempre serial — só acontece uma vez)
        self._evaluate_population()
        linear_scale_population(self.population)

        # Registra a geração 0
        best_overall_fitness = self._log_generation(generation=0)
        best_overall_chromo = max(self.population, key=lambda c: c.fitness)
        best_overall_genes = list(best_overall_chromo.genes)
        best_overall_f1 = best_overall_chromo.f1_score

        stagnation = 0

        # 2) Evolução: cada geração executa N mini-ciclos (serial ou paralelo)
        for generation in range(1, self.max_generations + 1):
            # Seleciona TODOS os pares de pais no início da geração
            # (snapshot da população, sem evolução dentro da geração)
            parent_pairs = self._select_parent_pairs(self.mini_cycles_per_generation)

            # Executa os mini-ciclos
            if self.workers > 1:
                all_children = self._do_generation_parallel(generation, parent_pairs)
            else:
                all_children = self._do_generation_serial(generation, parent_pairs)

            # Loga as métricas de NN dos filhos no logger principal
            self._log_children_nn_metrics(all_children)

            # Substitui os (pop_size - elite_size) piores indivíduos pelos filhos
            self.population.sort(key=lambda c: c.fitness)
            n_to_replace = self.population_size - self.elite_size
            for i, child in enumerate(all_children[:n_to_replace]):
                self.population[i] = child

            linear_scale_population(self.population)

            # Loga a geração
            best_now = self._log_generation(generation=generation)
            best_chromo_now = max(self.population, key=lambda c: c.fitness)

            improved = best_now > best_overall_fitness + 1e-9
            if improved:
                best_overall_fitness = best_now
                best_overall_genes = list(best_chromo_now.genes)
                best_overall_f1 = best_chromo_now.f1_score
                stagnation = 0
            else:
                stagnation += 1

            if stagnation >= self.stagnation_limit:
                print(
                    f"  [INFO] Parada por estagnação após {self.stagnation_limit} "
                    f"gerações sem melhoria (geração {generation})."
                )
                break

        return best_overall_genes, best_overall_fitness, best_overall_f1
