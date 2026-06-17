import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(data_path):
    """
    Carrega o dataset de câncer de mama, separa os atributos do alvo (diagnóstico)
    e aplica a normalização dos dados para a Rede Neural.
    """
    # 1. Lê o arquivo CSV
    df = pd.read_csv(data_path)
    
    # ATENÇÃO: Ajuste os nomes abaixo de acordo com as colunas do seu CSV!
    # Vamos assumir que a última coluna seja o alvo (ex: 'target' ou 'diagnosis')
    # e que não há colunas de ID inúteis.
    
    X = df.iloc[:, :-1].values  # Todas as colunas, exceto a última
    y = df.iloc[:, -1].values   # Apenas a última coluna (0 para benigno, 1 para maligno)
    
    # Guarda o nome das colunas para o GA nos dizer quais foram as melhores
    feature_names = df.columns[:-1].tolist()
    
    # 2. Normalização (Essencial para Redes Neurais escalonarem os pesos corretamente)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, feature_names