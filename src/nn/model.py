from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def create_mlp_model(input_dim):
    """
    Cria uma arquitetura MLP (Multilayer Perceptron) para classificação binária.
    A camada de entrada adapta-se ao número de colunas selecionadas pelo GA.
    """
    model = Sequential([
        # Camada oculta 1: 16 neurônios com ativação ReLU
        Dense(16, activation='relu', input_dim=input_dim),
        Dropout(0.2), # Previne overfitting "desligando" aleatoriamente 20% dos neurônios
        
        # Camada oculta 2: 8 neurônios
        Dense(8, activation='relu'),
        
        # Camada de saída: 1 neurônio com Sigmoid para retornar probabilidade (0 ou 1)
        # Ideal para o diagnóstico de câncer de mama (Benigno/Maligno)
        Dense(1, activation='sigmoid')
    ])
    
    # Compilação do modelo focada em classificação binária
    model.compile(
        optimizer='adam', 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    
    return model