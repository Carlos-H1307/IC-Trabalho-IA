import os
# Importações dos módulos internos da estrutura sugerida
from data_loader import load_and_preprocess_data
from ga.algorithm import GeneticAlgorithm
from utils.logger import MetricsLogger
from utils.plotter import plot_metrics

def main():
    # 1. Configurações de Caminhos e Hiperparâmetros
    DATA_PATH = os.path.join("files", "raw", "breast_cancer.csv")
    LOG_DIR = "logs"
    PLOT_DIR = "plots"
    
    POPULATION_SIZE = 50
    GENERATIONS = 30
    MUTATION_RATE = 0.05
    CROSSOVER_RATE = 0.8

    print("========================================================")
    print("  Seleção de Características para Cancro de Mama com GA + NN  ")
    print("========================================================\n")
    
    # 2. Carregamento e Pré-processamento dos Dados
    if not os.path.exists(DATA_PATH):
        print(f"[ERRO] O ficheiro de dados não foi encontrado em: {DATA_PATH}")
        print("Por favor, certifique-se de que o dataset está na pasta /files/raw/")
        return

    print("[INFO] A carregar a base de dados...")
    X, y, feature_names = load_and_preprocess_data(DATA_PATH)
    num_features = X.shape[1]
    print(f"[INFO] Dataset carregado com sucesso. Total de colunas (atributos): {num_features}")
    
    # 3. Inicialização do Gestor de Logs
    # Cria a pasta /logs se não existir e prepara os ficheiros CSV para as métricas
    logger = MetricsLogger(log_dir=LOG_DIR)
    
    # 4. Configuração do Algoritmo Genético
    print("[INFO] A inicializar o Algoritmo Genético...")
    ga = GeneticAlgorithm(
        population_size=POPULATION_SIZE,
        chromosome_length=num_features,
        mutation_rate=MUTATION_RATE,
        crossover_rate=CROSSOVER_RATE,
        X=X,
        y=y,
        logger=logger
    )
    
    # 5. Execução do Processo Evolutivo
    print(f"[INFO] A iniciar a evolução ao longo de {GENERATIONS} gerações. Por favor, aguarde...")
    best_chromosome, best_fitness = ga.evolve(generations=GENERATIONS)
    
    print("\n==================== EVOLUÇÃO CONCLUÍDA ====================")
    print(f"Melhor Fitness Alcançado (Acurácia da Rede Neural): {best_fitness:.4f}")
    
    # Mapeamento do melhor cromossoma para identificar as colunas relevantes
    selected_features = [feature_names[i] for i, active in enumerate(best_chromosome) if active == 1]
    
    print(f"Número de colunas originais: {num_features}")
    print(f"Número de colunas selecionadas: {len(selected_features)}")
    print("\nColunas identificadas como relevantes para o diagnóstico:")
    for feature in selected_features:
        print(f" - {feature}")
    print("============================================================\n")
    
    # 6. Geração de Gráficos de Desempenho
    print("[INFO] A ler os ficheiros de log e a gerar os gráficos das métricas...")
    plot_metrics(log_dir=LOG_DIR, output_dir=PLOT_DIR)
    print(f"[SUCESSO] Gráficos guardados na pasta /{PLOT_DIR}.")
    print("Processo finalizado.")

if __name__ == "__main__":
    main()