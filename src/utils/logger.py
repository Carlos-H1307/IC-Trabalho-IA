import csv
import os


class MetricsLogger:
    """
    Escreve em CSV as métricas do AG (por experimento e geração) e da NN
    (por avaliação de cromossomo).
    """

    GA_HEADER = [
        "experimento",
        "geracao",
        "melhor_fitness",
        "fitness_medio",
        "pior_fitness",
        "melhor_f1",
        "num_atributos_ativos",
        "melhor_cromossomo",
    ]

    NN_HEADER = [
        "id_cromossomo",
        "geracao",
        "loss_treino",
        "loss_validacao",
        "acuracia_validacao",
        "f1_score",        # F1 weighted (fitness primário)
        "f1_macro",        # F1 macro (para diagnóstico de viés)
        "epocas",
        "num_atributos_usados",
    ]

    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        self.ga_log_path = os.path.join(log_dir, "ga_metrics.csv")
        self.nn_log_path = os.path.join(log_dir, "nn_metrics.csv")
        os.makedirs(log_dir, exist_ok=True)
        self._init_log(self.ga_log_path, self.GA_HEADER)
        self._init_log(self.nn_log_path, self.NN_HEADER)

    @staticmethod
    def _init_log(path, header):
        # Reescreve o arquivo a cada execução para evitar mistura entre rodadas
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

    def log_ga_metrics(
        self,
        experiment_id,
        generation,
        best_fitness,
        avg_fitness,
        worst_fitness,
        best_chromosome,
        num_active,
        best_f1,
    ):
        chromosome_str = "".join(map(str, best_chromosome))
        with open(self.ga_log_path, mode="a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                experiment_id,
                generation,
                best_fitness,
                avg_fitness,
                worst_fitness,
                best_f1,
                num_active,
                chromosome_str,
            ])

    def log_nn_metrics(
        self,
        chromosome_id,
        generation,
        train_loss,
        val_loss,
        val_accuracy,
        f1_score,
        f1_macro,
        epochs,
        num_features_used,
    ):
        with open(self.nn_log_path, mode="a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                chromosome_id,
                generation,
                train_loss,
                val_loss,
                val_accuracy,
                f1_score,
                f1_macro,
                epochs,
                num_features_used,
            ])
