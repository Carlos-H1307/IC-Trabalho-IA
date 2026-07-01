import random


class Chromosome:
    """
    Cromossomo de codificação binária para seleção de atributos.
    Cada gene representa um atributo:
      gene = 1 -> atributo selecionado
      gene = 0 -> atributo descartado

    Inicialização estratificada por densidade
    -----------------------------------------
    A inicialização amostra Ns ~ Uniform(1, L) e ativa Ns bits escolhidos
    aleatoriamente (sem reposição), em vez de amostrar cada bit como
    Bernoulli(0.5). Isso garante que a população inicial cubra todo o
    intervalo de densidades [1/L, 1], em vez de concentrar-se em torno
    de L/2 (~ N(L/2, sqrt(L)/2) sob Bernoulli(0.5)).

    Justificativa: para o problema de seleção de atributos, subconjuntos
    esparsos ganham bônus de parcimônia (0,1 * (1 - Ns/Nt)) e podem
    dominar a elite; sob Bernoulli(0.5) o resto da população fica
    incapaz de reproduzir tal densidade via crossover uniforme, causando
    lock-in na geração 0. A amostragem uniforme sobre o tamanho do
    subconjunto é uma estratégia recorrente na literatura de seleção
    de atributos por AG:

      - Kabir, M. M., Islam, M. M., & Yao, X. (2011). "A new wrapper
        feature selection approach using neural network."
        Neurocomputing, 74(17), 3273-3283.
      - Xue, B., Zhang, M., Browne, W. N., & Yao, X. (2016). "A Survey
        on Evolutionary Computation Approaches to Feature Selection."
        IEEE Trans. Evol. Comput., 20(4), 606-626. (Sec. IV-A discute
        inicialização balanceada por número de atributos.)
      - Leardi, R. (1994). "Application of a genetic algorithm to
        feature selection under full validation conditions and to
        outlier detection." J. Chemometrics, 8(1), 65-79.
    """

    def __init__(self, length, genes=None):
        self.length = length
        if genes is None:
            # Inicialização estratificada por densidade: amostra o número
            # de atributos ativos uniformemente em [1, L] e depois escolhe
            # quais posições ativar sem reposição.
            num_active = random.randint(1, length)
            active_positions = random.sample(range(length), num_active)
            self.genes = [0] * length
            for pos in active_positions:
                self.genes[pos] = 1
        else:
            self.genes = list(genes)

        # Invariante: nenhum cromossomo pode ficar sem atributos ativos.
        # Aplicado também para o caminho `genes=` (usado por crossover),
        # onde dois pais esparsos podem gerar um filho todo-zero.
        self._ensure_at_least_one_active()

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
