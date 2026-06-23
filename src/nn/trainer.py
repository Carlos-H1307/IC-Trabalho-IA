import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

from nn.model import create_mlp_model

# Configurações de treinamento (compromisso entre tempo de execução e qualidade)
DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 64
DEFAULT_PATIENCE = 5  # paciência para early stopping no conjunto de validação


def _split_70_15_15(X, y, random_state):
    """Divide os dados em 70% treino / 15% validação / 15% teste de forma estratificada."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_and_evaluate_nn(
    X_filtered,
    y,
    n_classes,
    chromosome_id=None,
    logger=None,
    generation=None,
    random_state=42,
    epochs=DEFAULT_EPOCHS,
    batch_size=DEFAULT_BATCH_SIZE,
    patience=DEFAULT_PATIENCE,
):
    """
    Treina a MLP nos atributos selecionados e retorna o F1-Score no conjunto de teste.

    A divisão segue o procedimento experimental: 70% treino / 15% validação / 15% teste.
    O conjunto de validação é usado pelo EarlyStopping para escolher a melhor configuração
    da rede (menor erro de validação). O F1-Score final é medido no conjunto de teste,
    em dados não vistos durante o treino nem durante a seleção do melhor modelo.
    """
    X_train, X_val, X_test, y_train, y_val, y_test = _split_70_15_15(
        X_filtered, y, random_state=random_state
    )

    input_dim = X_train.shape[1]
    model = create_mlp_model(input_dim, n_classes=n_classes, learning_rate=0.001)

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=0,
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0,
    )

    # Avaliação no conjunto de teste
    # F1 weighted: pondera o F1 de cada classe pelo seu suporte real.
    # Para classes desbalanceadas (62/20/18), é a métrica que reflete o
    # desempenho esperado na população — usada como fitness primário.
    # F1 macro: trata as três classes igualmente (sem ponderação) — útil
    # para diagnóstico de viés, registrado em log para análise comparativa.
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    train_loss = float(history.history["loss"][-1])
    val_loss = float(history.history["val_loss"][-1])
    val_accuracy = float(history.history["val_accuracy"][-1])

    if logger:
        logger.log_nn_metrics(
            chromosome_id=chromosome_id,
            generation=generation,
            train_loss=train_loss,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            f1_score=f1_weighted,
            f1_macro=f1_macro,
            epochs=len(history.history["loss"]),
            num_features_used=input_dim,
        )

    return f1_weighted
