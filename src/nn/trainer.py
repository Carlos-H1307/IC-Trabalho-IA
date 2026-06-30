import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from nn.model import create_mlp_model

# Configurações de treinamento
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


def _oversample_minority_classes(X, y, random_state):
    """
    Reamostra com reposição as classes minoritárias para igualar o tamanho da
    classe majoritária. Tratamento do desbalanceamento (62/20/18) no nível do
    treinamento, já que `MLPClassifier` do sklearn não aceita `class_weight`
    nem `sample_weight`.

    Aplicado apenas ao conjunto de treino — validação e teste preservam a
    distribuição original para que as métricas reflitam o cenário real.
    """
    classes, counts = np.unique(y, return_counts=True)
    max_count = int(counts.max())

    X_parts, y_parts = [], []
    for c in classes:
        mask = y == c
        X_c, y_c = X[mask], y[mask]
        if len(X_c) < max_count:
            X_c, y_c = resample(
                X_c, y_c,
                n_samples=max_count,
                replace=True,
                random_state=random_state,
            )
        X_parts.append(X_c)
        y_parts.append(y_c)

    X_resampled = np.vstack(X_parts)
    y_resampled = np.concatenate(y_parts)

    # Embaralha (evita batches consecutivos da mesma classe)
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(len(y_resampled))
    return X_resampled[perm], y_resampled[perm]


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
    Treina a MLP (sklearn) nos atributos selecionados e retorna F1-Score + métricas.

    Divisão experimental (70/15/15 estratificada):
      - 70%: treino
      - 15%: validação (consumida pelo `early_stopping` interno do MLPClassifier)
      - 15%: teste (usado apenas para F1 final, nunca visto durante treino/validação)

    O sklearn `MLPClassifier(early_stopping=True)` separa internamente uma fração
    do conjunto recebido em `fit` para validação. Por isso passamos
    **treino + validação concatenados** (85% do total) e configuramos
    `validation_fraction = 15/85` para que o split interno seja idêntico ao 70/15/15.

    Métrica de aptidão: F1-Score weighted (pondera classes pelo suporte real,
    apropriado para base desbalanceada 62/20/18). Também loga F1 macro para
    diagnóstico de viés por classe.

    Retorna (f1_weighted, metrics_dict). Se `logger` for fornecido, também
    grava as métricas em `nn_metrics.csv`. Em modo paralelo, o caller chama
    com `logger=None` e loga depois.
    """
    X_train, X_val, X_test, y_train, y_val, y_test = _split_70_15_15(
        X_filtered, y, random_state=random_state
    )

    input_dim = X_train.shape[1]

    # Oversampling das classes minoritárias APENAS no conjunto de treino
    # (validação e teste preservam a distribuição original 62/20/18)
    X_train, y_train = _oversample_minority_classes(X_train, y_train, random_state)

    # Concatena treino + validação para que o sklearn faça o split interno
    # que reproduz nosso 70/15/15
    X_train_val = np.concatenate([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])
    val_fraction = len(X_val) / len(X_train_val)

    model = create_mlp_model(
        input_dim=input_dim,
        n_classes=n_classes,
        learning_rate=0.001,
        max_iter=epochs,
        batch_size=batch_size,
        validation_fraction=val_fraction,
        n_iter_no_change=patience,
        random_state=random_state,
    )

    # Treinamento (sklearn loga warnings se max_iter for atingido sem convergir;
    # silenciamos via context manager local)
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        model.fit(X_train_val, y_train_val)

    # Avaliação no conjunto de teste
    y_pred = model.predict(X_test)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    # Métricas auxiliares (do histórico de treinamento do sklearn)
    train_loss = float(model.loss_curve_[-1]) if model.loss_curve_ else 0.0
    val_scores = getattr(model, "validation_scores_", None)
    val_accuracy = float(val_scores[-1]) if val_scores else 0.0
    # Com early_stopping=True, sklearn rastreia best_validation_score_ (accuracy),
    # não best_loss_. Usamos o último train_loss como proxy para "val_loss".
    best_val_score = getattr(model, "best_validation_score_", None)
    val_loss = float(best_val_score) if best_val_score is not None else train_loss

    metrics = {
        "chromosome_id": chromosome_id,
        "generation": generation,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "f1_score": float(f1_weighted),
        "f1_macro": float(f1_macro),
        "epochs": int(model.n_iter_),
        "num_features_used": input_dim,
    }

    if logger:
        logger.log_nn_metrics(**metrics)

    return float(f1_weighted), metrics
