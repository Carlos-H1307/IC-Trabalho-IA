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
    feature_groups=None,
    fitness_repeats=1,
):
    """
    Avalia um cromossomo treinando a MLP nos atributos selecionados e calcula
    o fitness conforme a fórmula:

        Fitness = 0.9 * F1-Score + 0.1 * (1 - Ns / Nt)

    onde Ns é o número de atributos selecionados e Nt é o total de atributos.

    Codificação binária agrupada
    ----------------------------
    Se `feature_groups` for fornecido (lista de (nome, [índices_de_coluna])),
    cada bit do cromossomo representa um GRUPO (atributo semântico
    original), não uma coluna individual. Ativar o bit i passa TODAS as
    colunas de `feature_groups[i]` para a MLP. Isso elimina a
    fragmentação de one-hot: em vez de o AG poder selecionar `RACACOR_1`
    isoladamente, ele escolhe "usar RACACOR" (todas as 5 dummies) ou
    "não usar RACACOR" (nenhuma).

    Sob a codificação agrupada, Ns e Nt referem-se a GRUPOS ativos e
    total de grupos, respectivamente — consistente com a definição de
    "número de atributos selecionados" do spec.

    Se `feature_groups` for None, mantém a codificação flat legada
    (1 bit por coluna).

    Resultados são cacheados pela chave (tupla de genes) — isso evita
    re-treinar a rede para cromossomos idênticos que reaparecem durante
    a evolução.

    Após a avaliação, anexa o dicionário de métricas da NN ao cromossomo
    via `chromosome.nn_metrics` para que o chamador possa logar mais
    tarde (útil quando a avaliação acontece em um worker paralelo que
    não tem acesso ao logger principal).
    """
    active_bits = [i for i, gene in enumerate(chromosome.genes) if gene == 1]
    if not active_bits:
        chromosome.f1_score = 0.0
        chromosome.fitness = 0.0
        chromosome.nn_metrics = None
        return 0.0

    # Cache de avaliações para cromossomos idênticos
    key = chromosome.key()
    if fitness_cache is not None and key in fitness_cache:
        f1, fitness, cached_metrics = fitness_cache[key]
        chromosome.f1_score = f1
        chromosome.fitness = fitness
        chromosome.nn_metrics = cached_metrics
        return fitness

    # Expande bits ativos para índices de colunas de X.
    if feature_groups is not None:
        active_col_indices = []
        for bit_idx in active_bits:
            active_col_indices.extend(feature_groups[bit_idx][1])
    else:
        active_col_indices = active_bits

    X_filtered = X[:, active_col_indices]

    # Avaliação com múltiplas seeds (opcional). Sob `fitness_repeats > 1`,
    # a MLP é retreinada K vezes com seeds distintas (split 70/15/15 e
    # inicialização de pesos diferentes) e o F1 usado no fitness é a
    # MÉDIA das K execuções. O erro-padrão do estimador cai em sqrt(K),
    # o que reduz o ruído da paisagem de busca do AG.
    # Refs: Bengio & Grandvalet (2004, JMLR); Nadeau & Bengio (2003, ML).
    if fitness_repeats <= 1:
        f1, nn_metrics = train_and_evaluate_nn(
            X_filtered,
            y,
            n_classes=n_classes,
            chromosome_id=chromosome_id,
            logger=logger,
            generation=generation,
            random_state=random_state,
        )
    else:
        f1_scores = []
        last_metrics = None
        for k in range(fitness_repeats):
            seed_k = random_state + 1000 * k
            f1_k, metrics_k = train_and_evaluate_nn(
                X_filtered,
                y,
                n_classes=n_classes,
                chromosome_id=chromosome_id,
                logger=logger if k == 0 else None,  # loga só o primeiro seed
                generation=generation,
                random_state=seed_k,
            )
            f1_scores.append(f1_k)
            last_metrics = metrics_k
        import numpy as _np
        f1 = float(_np.mean(f1_scores))
        # Anexa média/std ao dicionário de métricas para relatório posterior
        last_metrics = dict(last_metrics or {})
        last_metrics["f1_score"] = f1
        last_metrics["f1_std_over_repeats"] = float(_np.std(f1_scores))
        last_metrics["fitness_repeats"] = fitness_repeats
        nn_metrics = last_metrics

    Nt = chromosome.length
    Ns = len(active_bits)  # grupos ativos, não colunas — spec-consistent
    parsimony = 1.0 - (Ns / Nt)
    fitness = F1_WEIGHT * f1 + PARSIMONY_WEIGHT * parsimony

    chromosome.f1_score = float(f1)
    chromosome.fitness = float(fitness)
    chromosome.nn_metrics = nn_metrics

    if fitness_cache is not None:
        fitness_cache[key] = (chromosome.f1_score, chromosome.fitness, nn_metrics)

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
