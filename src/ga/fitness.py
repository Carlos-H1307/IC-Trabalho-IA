import numpy as np

from nn.trainer import train_and_evaluate_nn

# Pesos da função de aptidão (conforme especificação do trabalho)
F1_WEIGHT = 0.9
PARSIMONY_WEIGHT = 0.1


def evaluate_chromosome(
    chromosome,
    X,
    y,
    n_classes,
    chromosome_id=None,
    logger=None,
    generation=None,
    fitness_cache=None,
    random_state=42,
):
    """
    Avalia um cromossomo treinando a MLP nos atributos selecionados e calcula
    o fitness conforme a fórmula:

        Fitness = 0.9 * F1-Score + 0.1 * (1 - Ns / Nt)

    onde Ns é o número de atributos selecionados e Nt é o total de atributos.

    Resultados são cacheados pela chave (tupla de genes) — isso evita re-treinar
    a rede para cromossomos idênticos que reaparecem durante a evolução.
    """
    active_indices = [i for i, gene in enumerate(chromosome.genes) if gene == 1]
    if not active_indices:
        chromosome.f1_score = 0.0
        chromosome.fitness = 0.0
        return 0.0

    # Cache de avaliações para cromossomos idênticos
    key = chromosome.key()
    if fitness_cache is not None and key in fitness_cache:
        f1, fitness = fitness_cache[key]
        chromosome.f1_score = f1
        chromosome.fitness = fitness
        return fitness

    X_filtered = X[:, active_indices]

    f1 = train_and_evaluate_nn(
        X_filtered,
        y,
        n_classes=n_classes,
        chromosome_id=chromosome_id,
        logger=logger,
        generation=generation,
        random_state=random_state,
    )

    Nt = chromosome.length
    Ns = len(active_indices)
    parsimony = 1.0 - (Ns / Nt)
    fitness = F1_WEIGHT * f1 + PARSIMONY_WEIGHT * parsimony

    chromosome.f1_score = float(f1)
    chromosome.fitness = float(fitness)

    if fitness_cache is not None:
        fitness_cache[key] = (chromosome.f1_score, chromosome.fitness)

    return chromosome.fitness


def linear_scale_population(population):
    """
    Normalização linear dos valores de fitness da população:
        scaled = (fitness - f_min) / (f_max - f_min)

    Quando f_max == f_min (todos iguais), todos recebem 1.0.
    Atualiza chromosome.scaled_fitness in-place.
    """
    fitnesses = np.array([c.fitness for c in population], dtype=np.float64)
    f_min = fitnesses.min()
    f_max = fitnesses.max()
    if f_max - f_min < 1e-12:
        for c in population:
            c.scaled_fitness = 1.0
    else:
        for c in population:
            c.scaled_fitness = float((c.fitness - f_min) / (f_max - f_min))
