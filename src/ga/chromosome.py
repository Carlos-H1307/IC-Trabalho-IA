import random


class Chromosome:
    """
    Cromossomo de codificação binária para seleção de atributos.
    Cada gene representa um atributo:
      gene = 1 -> atributo selecionado
      gene = 0 -> atributo descartado
    """

    def __init__(self, length, genes=None):
        self.length = length
        if genes is None:
            # Inicializa aleatoriamente
            self.genes = [random.choice([0, 1]) for _ in range(length)]
            self._ensure_at_least_one_active()
        else:
            self.genes = list(genes)

        self.fitness = -1.0       # -1 indica que ainda não foi avaliado
        self.f1_score = 0.0       # F1-Score bruto retornado pela rede
        self.scaled_fitness = -1.0  # Fitness após normalização linear

    def _ensure_at_least_one_active(self):
        """Garante que o cromossomo selecione pelo menos um atributo."""
        if sum(self.genes) == 0:
            self.genes[random.randint(0, self.length - 1)] = 1

    @property
    def num_active(self):
        return sum(self.genes)

    def key(self):
        """Chave hashable para uso em caches de fitness."""
        return tuple(self.genes)

    def mutate(self, mutation_rate):
        """Mutação bit-flip: inverte cada gene com probabilidade mutation_rate."""
        for i in range(self.length):
            if random.random() < mutation_rate:
                self.genes[i] = 1 - self.genes[i]
        self._ensure_at_least_one_active()

    def crossover(self, partner, crossover_rate):
        """
        Crossover uniforme: para cada posição, com 50% de chance o filho recebe
        o gene do pai 1, e com 50% de chance recebe o gene do pai 2. Pc controla
        se o cruzamento ocorre — caso contrário, os filhos são cópias dos pais.
        """
        if random.random() > crossover_rate:
            return (
                Chromosome(self.length, list(self.genes)),
                Chromosome(self.length, list(partner.genes)),
            )

        child1_genes = []
        child2_genes = []
        for i in range(self.length):
            if random.random() < 0.5:
                child1_genes.append(self.genes[i])
                child2_genes.append(partner.genes[i])
            else:
                child1_genes.append(partner.genes[i])
                child2_genes.append(self.genes[i])

        return (
            Chromosome(self.length, child1_genes),
            Chromosome(self.length, child2_genes),
        )

    def __repr__(self):
        return (
            f"Chromosome(L={self.length}, ativos={self.num_active}, "
            f"fitness={self.fitness:.4f})"
        )
