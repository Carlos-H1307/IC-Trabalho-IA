import random
from ga.chromosome import Chromosome
from ga.fitness import calculate_fitness

class GeneticAlgorithm:
    def __init__(self, population_size, chromosome_length, mutation_rate, crossover_rate, X, y, logger):
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.X = X
        self.y = y
        self.logger = logger
        
        # Inicializa a população gerando cromossomos aleatórios
        self.population = [Chromosome(chromosome_length) for _ in range(population_size)]
        
    def _select_parent(self):
        """Seleção por Torneio: escolhe alguns indivíduos e pega o melhor entre eles."""
        tournament_size = 3
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda chromo: chromo.fitness)

    def evolve(self, generations):
        best_overall_chromosome = None
        best_overall_fitness = -1.0

        for generation in range(generations):
            # 1. Avaliação (Cálculo de Fitness via Rede Neural)
            for i, chromosome in enumerate(self.population):
                if chromosome.fitness == -1.0: # Só avalia se ainda não tiver fitness
                    chromosome_id = f"gen{generation}_ind{i}"
                    chromosome.fitness = calculate_fitness(
                        chromosome, self.X, self.y, 
                        chromosome_id=chromosome_id, 
                        logger=self.logger, 
                        generation=generation
                    )

            # 2. Coleta de Métricas para os Logs e Gráficos
            fitnesses = [c.fitness for c in self.population]
            best_gen_fitness = max(fitnesses)
            worst_gen_fitness = min(fitnesses)
            avg_gen_fitness = sum(fitnesses) / self.population_size
            
            best_gen_chromosome = max(self.population, key=lambda c: c.fitness)
            
            # Salva o melhor global
            if best_gen_fitness > best_overall_fitness:
                best_overall_fitness = best_gen_fitness
                best_overall_chromosome = list(best_gen_chromosome.genes)

            # Grava no CSV se o logger estiver configurado
            if self.logger:
                self.logger.log_ga_metrics(
                    generation, best_gen_fitness, avg_gen_fitness, 
                    worst_gen_fitness, best_gen_chromosome.genes
                )

            # 3. Criação da Nova População (Reprodução)
            new_population = []
            
            # Elitismo: passa o melhor indivíduo direto para a próxima geração sem alterações
            new_population.append(best_gen_chromosome)

            # Preenche o resto da população com cruzamento e mutação
            while len(new_population) < self.population_size:
                parent1 = self._select_parent()
                parent2 = self._select_parent()
                
                child1, child2 = parent1.crossover(parent2, self.crossover_rate)
                child1.mutate(self.mutation_rate)
                child2.mutate(self.mutation_rate)
                
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)

            self.population = new_population

        return best_overall_chromosome, best_overall_fitness