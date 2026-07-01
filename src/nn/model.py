from sklearn.neural_network import MLPClassifier


def create_mlp_model(
    input_dim,
    n_classes,
    learning_rate=0.001,
    max_iter=200,
    batch_size=64,
    validation_fraction=0.15,
    n_iter_no_change=15,
    random_state=42,
):
    """
    Cria a arquitetura MLP especificada no trabalho via scikit-learn:
      - Camada de entrada: input_dim neurônios (atributos selecionados pelo GA)
      - 1ª camada oculta: 32 neurônios, ativação ReLU
      - 2ª camada oculta: 16 neurônios, ativação ReLU
      - Camada de saída: n_classes neurônios, ativação Softmax (automático para multiclasse)

    Treinamento: Backpropagation com otimizador Adam (lr=0.001).
    Early stopping: "melhor configuração = menor erro de validação" (spec) — habilitado
    com `early_stopping=True` e paciência `n_iter_no_change` épocas.

    Observação: trocamos TF/Keras por scikit-learn para acelerar ~78× a avaliação
    de cada cromossomo (medido em profile). A spec não exige framework específico,
    apenas a arquitetura, otimizador, taxa de aprendizado, ativações e procedimento
    de validação — todos atendidos pelo `MLPClassifier`.

    `input_dim` é usado apenas para documentação aqui; o sklearn infere dinamicamente
    da matriz de treino.
    """
    return MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        learning_rate_init=learning_rate,
        max_iter=max_iter,
        batch_size=batch_size,
        early_stopping=True,
        validation_fraction=validation_fraction,
        n_iter_no_change=n_iter_no_change,
        alpha=0.0,        # desabilita L2 (não exigido pela spec)
        random_state=random_state,
    )
