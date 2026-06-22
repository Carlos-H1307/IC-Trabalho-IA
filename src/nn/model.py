from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam


def create_mlp_model(input_dim, n_classes, learning_rate=0.001):
    """
    Cria a arquitetura MLP especificada no trabalho:
      - Camada de entrada: input_dim neurônios (atributos selecionados pelo GA)
      - 1ª camada oculta: 32 neurônios, ativação ReLU
      - 2ª camada oculta: 16 neurônios, ativação ReLU
      - Camada de saída: n_classes neurônios, ativação Softmax

    Treinamento: Backpropagation com otimizador Adam (lr=0.001).
    Loss: sparse_categorical_crossentropy (y como inteiros, não one-hot).
    """
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(n_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
