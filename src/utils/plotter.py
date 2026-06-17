import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics(log_dir="logs", output_dir="plots"):
    """
    Lê os arquivos de log e gera gráficos estatísticos da evolução e do treino.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    ga_path = os.path.join(log_dir, "ga_metrics.csv")
    nn_path = os.path.join(log_dir, "nn_metrics.csv")
    
    # -------------------------------------------------------------------------
    # 1. Gráfico de Convergência do Algoritmo Genético
    # -------------------------------------------------------------------------
    if os.path.exists(ga_path):
        df_ga = pd.read_csv(ga_path)
        
        plt.figure(figsize=(10, 6))
        plt.plot(df_ga['geracao'], df_ga['melhor_fitness'], label='Melhor Fitness (Máx)', color='green', linewidth=2)
        plt.plot(df_ga['geracao'], df_ga['fitness_medio'], label='Fitness Médio', color='blue', linestyle='--')
        plt.plot(df_ga['geracao'], df_ga['pior_fitness'], label='Pior Fitness', color='red', alpha=0.5)
        
        plt.title('Convergência do Algoritmo Genético (Seleção de Atributos)', fontsize=14, pad=15)
        plt.xlabel('Geração', fontsize=12)
        plt.ylabel('Acurácia da Rede Neural (Fitness)', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='lower right', fontsize=10)
        
        # Garante valores inteiros no eixo X das gerações
        plt.xticks(df_ga['geracao'])
        
        plot_ga_out = os.path.join(output_dir, "ga_convergencia.png")
        plt.savefig(plot_ga_out, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print(f"[AVISO] Arquivo {ga_path} não encontrado para plotagem.")

    # -------------------------------------------------------------------------
    # 2. Gráfico de Relação: Quantidade de Atributos vs Acurácia
    # -------------------------------------------------------------------------
    if os.path.exists(nn_path):
        df_nn = pd.read_csv(nn_path)
        
        plt.figure(figsize=(10, 6))
        # Plota a dispersão para ver se menos atributos mantêm uma acurácia alta
        scatter = plt.scatter(
            df_nn['num_atributos_usados'], 
            df_nn['acuracia_validacao'], 
            c=df_nn['geracao'], 
            cmap='viridis', 
            alpha=0.7, 
            edgecolors='w', 
            s=80
        )
        
        cbar = plt.colorbar(scatter)
        cbar.set_label('Progressão das Gerações', fontsize=11)
        
        plt.title('Análise de Redundância: Impacto do Número de Colunas na Acurácia', fontsize=14, pad=15)
        plt.xlabel('Quantidade de Colunas Ativas (Cromossomo)', fontsize=12)
        plt.ylabel('Acurácia de Validação (NN)', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6)
        
        plot_nn_out = os.path.join(output_dir, "nn_atributos_vs_acuracia.png")
        plt.savefig(plot_nn_out, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print(f"[AVISO] Arquivo {nn_path} não encontrado para plotagem.")