import random

class Chromosome:
    def __init__(self, length, genes=None):
        self.length = length
        if genes is None:
            # Inicializa aleatoriamente com 1 (ativa) ou 0 (inativa)
            self.genes = [random.choice([0, 1]) for _ in range(length)]
            
            # Failsafe: Garante que pelo menos uma coluna do dataset esteja ativa
            if sum(self.genes) == 0:
                self.genes[random.randint(0, length - 1)] = 1
        else:
            self.genes = genes
            
        self.fitness = -1.0 # -1.0 indica que ainda não foi avaliado

    def mutate(self, mutation_rate):
        """Aplica mutação invertendo o bit (0 vira 1, 1 vira 0) com base na taxa."""
        for i in range(self.length):
            if random.random() < mutation_rate:
                self.genes[i] = 1 - self.genes[i]
                
        # Failsafe após mutação
        if sum(self.genes) == 0:
            self.genes[random.randint(0, self.length - 1)] = 1

    def crossover(self, partner, crossover_rate):
        """Realiza o cruzamento (crossover) de 1 ponto entre dois pais."""
        if random.random() > crossover_rate:
            # Não houve cruzamento, retorna cópias exatas dos pais
            return Chromosome(self.length, list(self.genes)), Chromosome(self.length, list(partner.genes))
        
        # Escolhe um ponto de corte aleatório
        crossover_point = random.randint(1, self.length - 1)
        
        # Gera os filhos misturando as partes
        child1_genes = self.genes[:crossover_point] + partner.genes[crossover_point:]
        child2_genes = partner.genes[:crossover_point] + self.genes[crossover_point:]
        
        return Chromosome(self.length, child1_genes), Chromosome(self.length, child2_genes)
