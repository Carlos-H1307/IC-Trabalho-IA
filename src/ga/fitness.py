# Importa a função que vai treinar a rede neural (que estará em src/nn/trainer.py)
from nn.trainer import train_and_evaluate_nn

def calculate_fitness(chromosome, X, y, chromosome_id=None, logger=None, generation=None):
    """
    Filtra as colunas ativas e retorna o fitness baseado na acurácia da Rede Neural.
    """
    # Mapeia os índices onde o gene é igual a 1
    active_columns_indices = [i for i, gene in enumerate(chromosome.genes) if gene == 1]
    
    # Se nenhuma coluna estiver ativa, fitness é 0
    if not active_columns_indices:
        return 0.0
        
    # Filtra o dataset (assumindo X como numpy array para indexação rápida)
    X_filtered = X[:, active_columns_indices]
    
    # Executa a Rede Neural apenas com as colunas selecionadas
    # O trainer deve treinar a rede e retornar a acurácia ou o F1-Score no conjunto de validação
    fitness_score = train_and_evaluate_nn(
        X_filtered, y, 
        chromosome_id=chromosome_id, 
        logger=logger, 
        generation=generation
    )
    
    return fitness_score