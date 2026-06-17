from sklearn.model_selection import train_test_split
from nn.model import create_mlp_model

def train_and_evaluate_nn(X_filtered, y, chromosome_id=None, logger=None, generation=None):
    """
    Recebe os dados filtrados pelo cromossomo, treina a rede MLP e 
    retorna a acurácia de validação como fitness.
    """
    # 1. Divisão dos dados: 80% para treino, 20% para validação cruzada
    # O random_state fixo garante que todos os cromossomos sejam testados na mesma divisão
    X_train, X_val, y_train, y_val = train_test_split(X_filtered, y, test_size=0.2, random_state=42)
    
    # 2. Inicialização do Modelo
    input_dim = X_train.shape[1]
    model = create_mlp_model(input_dim)
    
    # 3. Treinamento
    # Mantemos o número de épocas baixo (ex: 10 a 20) para que o GA não demore dias para rodar.
    # verbose=0 mantém o terminal limpo durante as centenas de treinamentos.
    EPOCHS = 15
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=0 
    )
    
    # 4. Extração de Métricas (Pegamos os valores da última época)
    val_accuracy = history.history['val_accuracy'][-1]
    val_loss = history.history['val_loss'][-1]
    train_loss = history.history['loss'][-1]
    
    # 5. Telemetria e Logs
    if logger:
        logger.log_nn_metrics(
            chromosome_id=chromosome_id,
            generation=generation,
            train_loss=train_loss,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            epochs=EPOCHS,
            num_features_used=input_dim
        )
        
    return val_accuracy