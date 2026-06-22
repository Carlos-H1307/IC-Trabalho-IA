# Seleção de Atributos com Algoritmos Genéticos e Redes Neurais

Trabalho da disciplina de Inteligência Computacional — CEFET-RJ  
Prof. Laércio Brito

## Descrição

Sistema de seleção automática de atributos para uma base de dados de câncer do colo do útero. Utiliza Algoritmos Genéticos (AG) como mecanismo de busca e uma Rede Neural Artificial (MLP) treinada com Backpropagation como função de avaliação de cada solução.

O objetivo é encontrar o subconjunto de atributos que maximize a capacidade preditiva do modelo (F1-Score) com o menor número de variáveis possível.

## Como funciona

Cada cromossomo do AG representa uma máscara binária sobre os atributos da base de dados. A aptidão de cada cromossomo é calculada treinando uma MLP apenas com os atributos selecionados e medindo o F1-Score no conjunto de validação.

```
Fitness = 0,9 × F1-Score + 0,1 × (1 − Ns/Nt)
```

## Setup

O projeto usa [uv](https://docs.astral.sh/uv/) para gerenciamento de dependências e ambiente virtual.

### 1. Instalar o uv

**macOS / Linux**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell)**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Após a instalação, reinicie o terminal para que o comando `uv` fique disponível.

### 2. Clonar o repositório

```bash
git clone https://github.com/Carlos-H1307/IC-Trabalho-IA.git
cd IC-Trabalho-IA
```

### 3. Instalar as dependências

```bash
uv sync
```

Isso cria automaticamente um ambiente virtual em `.venv/` e instala todas as dependências declaradas no `pyproject.toml`.

### 4. Adicionar a base de dados

Coloque o arquivo da base (formato CSV ou XLSX) em `files/raw/cervical-cancer.xlsx`. A pasta está no `.gitignore` — o dataset é distribuído separadamente pelo professor via Teams.

Por padrão o código procura `files/raw/cervical-cancer.xlsx`. Use `--data-path` para apontar para outro caminho.

## Executar

Execução completa (20 experimentos × 200 gerações, conforme o trabalho):

```bash
uv run python src/main.py
```

Modo rápido para teste (poucas gerações e indivíduos):

```bash
uv run python src/main.py --quick
```

Outras opções úteis:

```bash
uv run python src/main.py --experiments 5 --generations 50 --sample-size 5000
uv run python src/main.py --sample-size 0   # usa a base completa, sem amostragem
```

Os resultados são salvos em `logs/` (CSV) e os gráficos em `plots/`.

### Saídas geradas

- `logs/ga_metrics.csv` — por experimento e geração: melhor/médio/pior fitness, melhor F1, nº de atributos ativos e cromossomo.
- `logs/nn_metrics.csv` — por avaliação da rede: losses, acurácia, F1, épocas e nº de atributos.
- `plots/ga_convergencia_media.png` — curva média da convergência ao longo dos experimentos (com banda de desvio).
- `plots/ga_convergencia_por_experimento.png` — curvas individuais de cada experimento.
- `plots/ga_fitness_componentes.png` — melhor/médio/pior do primeiro experimento.
- `plots/nn_atributos_vs_f1.png` — dispersão de F1-Score por número de atributos ativos.

## Estrutura

```
src/
├── main.py              # ponto de entrada
├── data_loader.py       # carregamento e pré-processamento (normalização Min-Max)
├── ga/
│   ├── algorithm.py     # loop evolutivo, seleção, elitismo
│   ├── chromosome.py    # representação binária, crossover, mutação
│   └── fitness.py       # cálculo de aptidão via MLP
├── nn/
│   ├── model.py         # arquitetura MLP (32 → 16 → softmax)
│   └── trainer.py       # treinamento com Adam, divisão 70/15/15
└── utils/
    ├── logger.py        # log de métricas do AG e da NN em CSV
    └── plotter.py       # curvas de convergência e gráficos de análise
docs/
├── ga-mlp-task.pdf      # enunciado do trabalho
├── ga-material/         # material de apoio sobre Algoritmos Genéticos
└── mlp-material/        # material de apoio sobre Redes Neurais
```

## Parâmetros do AG

| Parâmetro | Valor |
|---|---|
| Tamanho da população | 150 |
| Crossover | Uniforme, Pc = 0,85 |
| Mutação | Pm = 1/L |
| Elitismo | 10 melhores preservados |
| Estratégia | Steady-State, gap = 2 |
| Critério de parada | 200 gerações ou 20 sem melhoria |

## Parâmetros da MLP

| Parâmetro | Valor |
|---|---|
| Camadas ocultas | 32 neurônios (ReLU) → 16 neurônios (ReLU) |
| Saída | Softmax |
| Loss | Sparse Categorical Crossentropy |
| Otimizador | Adam (lr = 0,001) |
| Divisão dos dados | 70% treino / 15% validação / 15% teste (estratificada) |
| Early stopping | Sim, paciência 5 sobre `val_loss` |
| Métrica de aptidão | F1-Score macro no conjunto de teste |

## Notas de implementação

- **Vazamento de alvo**: a base contém colunas que descrevem diretamente a causa do óbito (`CAUSABAS`, `causabas_categoria`, `LINHAA`..`LINHAD` etc.). Essas colunas são removidas no pré-processamento (`TARGET_LEAK_COLUMNS` em `data_loader.py`), caso contrário o classificador atinge F1 = 1,0 trivialmente.
- **Amostragem estratificada**: por padrão o pipeline trabalha com uma amostra estratificada de 3000 registros para tornar 20 experimentos × 200 gerações tratáveis em uma máquina pessoal. Use `--sample-size 0` para a base completa.
- **Cache de fitness**: cromossomos idênticos (mesma máscara binária) reaproveitam a avaliação anterior, reduzindo significativamente o número de treinos da MLP.
