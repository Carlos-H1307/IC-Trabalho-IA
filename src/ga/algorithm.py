import random

from ga.chromosome import Chromosome
from ga.fitness import evaluate_chromosome, linear_scale_population


class GeneticAlgorithm:
    """
    Algoritmo Genético Steady-State (Gap = 2) para seleção de atributos.

    A cada geração:
      1. Seleciona 2 pais via torneio (usando fitness escalado)
      2. Aplica Crossover Uniforme com probabilidade Pc
      3. Aplica mutação bit-flip com probabilidade Pm = 1/L
      4. Avalia os 2 filhos via MLP + função de aptidão
      5. Insere os filhos substituindo os 2 piores indivíduos fora da elite

    Elitismo: os 10 melhores são preservados integralmente entre gerações.
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

        # População inicial aleatória
        self.population = [
            Chromosome(chromosome_length) for _ in range(population_size)
        ]

        # Histórico de melhor fitness por geração (para curva de convergência)
        self.history_best = []
        self.history_avg = []
        self.history_worst = []

    def _evaluate_population(self):
        """Avalia todos os indivíduos da população atual (caso ainda não tenham sido)."""
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

    def _replace_worst(self, children):
        """
        Substitui os piores indivíduos (excluindo a elite) pelos filhos gerados.
        Estratégia steady-state com gap = len(children).
        """
        # Ordena do pior para o melhor
        self.population.sort(key=lambda c: c.fitness)

        # Janela dos não-elite: os primeiros (population_size - elite_size) são os piores
        n_non_elite = self.population_size - self.elite_size
        non_elite_idx = list(range(n_non_elite))

        # Substitui os primeiros len(children) piores
        for i, child in enumerate(children):
            if i < n_non_elite:
                self.population[non_elite_idx[i]] = child

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
        # 1) Avalia população inicial
        self._evaluate_population()
        linear_scale_population(self.population)

        # Registra a geração 0 (população inicial)
        best_overall_fitness = self._log_generation(generation=0)
        best_overall_chromo = max(self.population, key=lambda c: c.fitness)
        best_overall_genes = list(best_overall_chromo.genes)
        best_overall_f1 = best_overall_chromo.f1_score

        stagnation = 0

        # 2) Evolução steady-state
        for generation in range(1, self.max_generations + 1):
            # Seleciona 2 pais e gera "gap" filhos via crossover + mutação
            parent1 = self._select_parent()
            parent2 = self._select_parent()
            child1, child2 = parent1.crossover(parent2, self.crossover_rate)
            child1.mutate(self.mutation_rate)
            child2.mutate(self.mutation_rate)
            children = [child1, child2][: self.gap]

            # Avalia os filhos
            for j, child in enumerate(children):
                cid = f"exp{self.experiment_id}_gen{generation}_child{j}"
                evaluate_chromosome(
                    child,
                    self.X,
                    self.y,
                    n_classes=self.n_classes,
                    chromosome_id=cid,
                    logger=self.logger,
                    generation=generation,
                    fitness_cache=self.fitness_cache,
                    random_state=self.random_state,
                )

            # Substitui os piores (preservando a elite)
            self._replace_worst(children)
            linear_scale_population(self.population)

            # Atualiza histórico e checa estagnação
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
