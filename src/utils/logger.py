import os
import csv

class MetricsLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        self.ga_log_path = os.path.join(log_dir, "ga_metrics.csv")
        self.nn_log_path = os.path.join(log_dir, "nn_metrics.csv")
        
        # Cria a pasta de logs se ela não existir
        os.makedirs(log_dir, exist_ok=True)
        
        # Inicializa os arquivos com os cabeçalhos (headers)
        self._init_ga_log()
        self._init_nn_log()

    def _init_ga_log(self):
        if not os.path.exists(self.ga_log_path):
            with open(self.ga_log_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["geracao", "melhor_fitness", "fitness_medio", "pior_fitness", "melhor_cromossomo"])

    def _init_nn_log(self):
        if not os.path.exists(self.nn_log_path):
            with open(self.nn_log_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["id_cromossomo", "geracao", "loss_treino", "loss_validacao", "acuracia_validacao", "epocas", "num_atributos_usados"])

    def log_ga_metrics(self, generation, best_fitness, avg_fitness, worst_fitness, best_chromosome):
        """Salva o resumo de desempenho de uma geração do Algoritmo Genético."""
        # Converte a lista do cromossomo (ex: [1, 0, 1]) em string para salvar no CSV
        chromosome_str = "-".join(map(str, best_chromosome))
        
        with open(self.ga_log_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([generation, best_fitness, avg_fitness, worst_fitness, chromosome_str])

    def log_nn_metrics(self, chromosome_id, generation, train_loss, val_loss, val_accuracy, epochs, num_features_used):
        """Salva os detalhes do treinamento da Rede Neural para um cromossomo específico."""
        with open(self.nn_log_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([chromosome_id, generation, train_loss, val_loss, val_accuracy, epochs, num_features_used])