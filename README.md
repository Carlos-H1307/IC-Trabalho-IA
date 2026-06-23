# Seleção de Atributos com Algoritmos Genéticos e Redes Neurais

Trabalho da disciplina de Inteligência Computacional — CEFET-RJ
Prof. Laércio Brito

## Descrição

Sistema de seleção automática de atributos para uma base de mortalidade por
câncer do colo do útero. Utiliza Algoritmos Genéticos (AG) como mecanismo de
busca e uma Rede Neural Artificial (MLP) treinada com Backpropagation como
função de avaliação de cada solução candidata.

O objetivo é encontrar o subconjunto de atributos que maximize a capacidade
preditiva do modelo (F1-Score) usando o menor número possível de variáveis,
codificado pela função de aptidão:

```
Fitness = 0,9 × F1-Score + 0,1 × (1 − Ns/Nt)
```

onde **Ns** é o número de atributos selecionados pelo cromossomo e **Nt** o
total de atributos disponíveis.

---

## Sumário

1. [Setup](#setup)
2. [Execução](#execução)
3. [Estrutura do projeto](#estrutura-do-projeto)
4. [Base de dados — exploração e limpeza](#base-de-dados--exploração-e-limpeza)
5. [Algoritmo Genético — escolhas de projeto](#algoritmo-genético--escolhas-de-projeto)
6. [Rede Neural (MLP) — escolhas de projeto](#rede-neural-mlp--escolhas-de-projeto)
7. [Função de aptidão e escalonamento](#função-de-aptidão-e-escalonamento)
8. [Procedimento experimental](#procedimento-experimental)
9. [Saídas geradas](#saídas-geradas)
10. [Decisões pragmáticas](#decisões-pragmáticas)

---

## Setup

O projeto usa [uv](https://docs.astral.sh/uv/) para gerenciar dependências e
o ambiente virtual.

### 1. Instalar o uv

**macOS / Linux**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell)**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Após instalar, reinicie o terminal.

### 2. Clonar o repositório

```bash
git clone https://github.com/Carlos-H1307/IC-Trabalho-IA.git
cd IC-Trabalho-IA
```

### 3. Instalar as dependências

```bash
uv sync
```

Cria automaticamente o `.venv/` e instala tudo declarado em `pyproject.toml`
(TensorFlow/Keras, scikit-learn, pandas, openpyxl, matplotlib).

### 4. Adicionar a base de dados

Coloque o arquivo em `files/raw/cervical-cancer.xlsx`. A pasta `files/` está no
`.gitignore` — a base é distribuída separadamente via Teams. O formato pode
ser CSV ou XLSX (o loader detecta pela extensão). Use `--data-path` para
apontar para outro caminho.

---

## Execução

Execução completa, conforme a especificação do trabalho
(20 experimentos × até 200 gerações):

```bash
uv run python src/main.py
```

Modo rápido para validar o pipeline (poucos experimentos e gerações):

```bash
uv run python src/main.py --quick
```

Outras opções:

```bash
uv run python src/main.py --experiments 5 --generations 50 --sample-size 5000
uv run python src/main.py --sample-size 0   # base completa, sem amostragem
```

CLI completa:

| Flag                | Padrão                              | Descrição |
|---------------------|-------------------------------------|-----------|
| `--data-path`       | `files/raw/cervical-cancer.xlsx`    | Caminho do arquivo de entrada |
| `--population`      | `150`                                | Tamanho da população do AG |
| `--generations`     | `200`                                | Limite superior de gerações |
| `--experiments`     | `20`                                 | Quantidade de execuções independentes |
| `--sample-size`     | `3000`                               | Amostra estratificada (0 = base completa) |
| `--quick`           | —                                    | Atalho: 2 experimentos × 20 gerações, pop 30 |

---

## Estrutura do projeto

```
src/
├── main.py              # ponto de entrada, CLI, loop de experimentos
├── data_loader.py       # carregamento, limpeza, normalização Min-Max
├── ga/
│   ├── algorithm.py     # loop steady-state, seleção, elitismo, parada
│   ├── chromosome.py    # representação binária, crossover, mutação
│   └── fitness.py       # função de aptidão e normalização linear
├── nn/
│   ├── model.py         # arquitetura MLP 32 → 16 → softmax
│   └── trainer.py       # divisão 70/15/15, treino com Adam, F1-Score
└── utils/
    ├── logger.py        # log de métricas em CSV
    └── plotter.py       # curvas de convergência e gráficos de análise

docs/
├── ga-mlp-task.pdf      # enunciado do trabalho
├── ga-material/         # material de apoio sobre Algoritmos Genéticos
└── mlp-material/        # material de apoio sobre Redes Neurais

files/raw/               # base de dados (gitignored)
logs/                    # CSVs gerados na execução (gitignored)
plots/                   # gráficos gerados na execução (gitignored)
```

---

## Base de dados — exploração e limpeza

### Origem

Base distribuída pelo professor via Teams, com registros do Sistema de
Informações sobre Mortalidade (SIM/DATASUS) referentes a óbitos por câncer do
colo do útero. Dois arquivos foram disponibilizados:

| Arquivo                    | Registros | Colunas |
|----------------------------|-----------|---------|
| `dataset-short.xlsx`       | 148.785   | 50      |
| `dataset-complete.xlsx`    | 148.785   | 167     |

**Versão utilizada:** `dataset-short.xlsx`, renomeada para
`cervical-cancer.xlsx` ao ser colocada em `files/raw/`. A versão completa
contém colunas redundantes (descritores textuais, datas decompostas,
coordenadas geográficas) que não trazem informação adicional para o
classificador e ainda aumentariam o custo computacional.

### Variável-alvo

A coluna **`label_cid`** é a variável-alvo, com três classes:

| Código | Descrição (CID-10) |
|--------|--------------------|
| C53    | Neoplasia maligna do **colo** do útero |
| C54    | Neoplasia maligna do **corpo** do útero |
| C55    | Neoplasia maligna do útero, **porção não especificada** |

A codificação para inteiros (`LabelEncoder`) gera os rótulos `{C53→0, C54→1,
C55→2}`. A rede neural tem 3 neurônios de saída (Softmax) correspondentes às
três classes.

### Pipeline de limpeza (`src/data_loader.py`)

O pré-processamento é determinístico e segue estas etapas, **nesta ordem**:

#### 1. Leitura do arquivo

`_load_raw` detecta a extensão e usa `pd.read_excel` (XLSX) ou
`pd.read_csv` (CSV). Nenhuma colunagem é inferida — todas as colunas do
arquivo são preservadas no momento da leitura.

#### 2. Descarte de registros sem alvo

Linhas com `label_cid` nulo são removidas (`dropna(subset=["label_cid"])`).
Na prática, o `label_cid` está sempre presente na base utilizada, mas a
proteção fica no código por robustez.

#### 3. Remoção de colunas de vazamento de alvo

> Esta é a decisão mais importante do pré-processamento.

A base contém colunas que descrevem **diretamente** a causa básica do óbito
da qual o `label_cid` foi derivado. Treinar a rede com essas colunas faz a
classificação ficar trivial (F1 = 1,0) e elimina qualquer sinal real para o
AG explorar. As colunas removidas estão em `TARGET_LEAK_COLUMNS`:

| Coluna                    | Por que vaza |
|---------------------------|--------------|
| `CAUSABAS`                | Código CID-10 completo da causa básica (`C539`, `C549`, etc.) — o `label_cid` é o prefixo deste campo |
| `CAUSABAS_O`              | Variante codificada do mesmo campo |
| `CB_PRE`                  | Causa básica preliminar |
| `causabas_categoria`      | Já é o `label_cid` (categoria CID a 3 dígitos) |
| `causabas_subcategoria`   | Granularidade maior da mesma informação |
| `causabas_capitulo`       | Capítulo CID (presente apenas no dataset completo) |
| `causabas_grupo`          | Grupo CID (presente apenas no dataset completo) |
| `LINHAA`, `LINHAB`, `LINHAC`, `LINHAD`, `LINHAII` | Linhas do atestado de óbito — contêm os códigos CID que originam o alvo |
| `ATESTADO`, `ATESTANTE`   | Texto/identificadores associados ao atestado, podem conter o código da causa |

**Validação:** rodando o pipeline sem este filtro, F1 chega a 1,0 já na
população inicial. Após o filtro, o F1 fica entre 0,35–0,55, refletindo a
real dificuldade do problema.

#### 4. Remoção de colunas com muitos valores ausentes

Colunas com mais de **50% de valores nulos** são descartadas
(`MAX_MISSING_RATIO = 0.5`). Exemplos da base curta:

- `CAUSAMAT`, `NUDIASINF`, `ALTCAUSA` — 100% nulas
- `ESTABDESCR`, `COMUNSVOIM` — > 95% nulas
- `SERIESCFAL`, `LINHAD`, `LINHAII` — ~80% nulas

O limiar de 50% foi escolhido como compromisso: colunas com pouca cobertura
são informacionalmente pobres e introduziriam ruído de imputação se mantidas.

#### 5. Codificação de variáveis categóricas

Para cada coluna do tipo `object` / `string`:

- Se a cardinalidade (`nunique`) for **maior que 50**
  (`MAX_CATEGORICAL_CARDINALITY`), a coluna é **descartada**. São tipicamente
  IDs, nomes livres ou códigos arbitrários (ex.: `ocor_MUNNOME`,
  `res_MUNNOMEX`) — sem semântica numérica e com poder discriminativo baixo
  individualmente.
- Caso contrário, valores nulos são preenchidos com o token `"__missing__"`
  (preservando a informação de "ausente" como categoria própria) e a coluna
  é convertida para inteiros via `LabelEncoder`.

> **Justificativa do LabelEncoder vs. One-Hot Encoding:** preferiu-se
> `LabelEncoder` para manter o cromossomo curto. Cada coluna categórica vira
> **um único gene** no cromossomo do AG, em vez de explodir em dezenas de
> variáveis dummy. Isso mantém o espaço de busca tratável (L ≈ 30) e o
> trabalho fiel à ideia de "selecionar atributos" (não "selecionar
> indicadores".) A perda em expressividade é compensada pela MLP, que
> consegue aprender funções não-monotônicas dos valores codificados.

#### 6. Imputação de valores numéricos ausentes

Para cada coluna numérica restante com pelo menos um NaN, o valor é
preenchido com a **mediana** da coluna. A mediana foi escolhida em vez da
média porque a base contém colunas com fortes assimetrias e outliers
(ex.: `CODMUN*`, `OCUP`, datas codificadas como inteiros gigantes), onde a
média seria distorcida.

#### 7. Amostragem estratificada (opcional)

Por padrão, o pipeline trabalha com uma **amostra estratificada de 3000
registros** (parâmetro `--sample-size`). A amostragem mantém a proporção
original das três classes:

```python
n_take = round(sample_size * |classe_c| / N)
```

Esta é uma **decisão pragmática** para tornar 20 experimentos × 200
gerações × ~150 cromossomos viável em uma máquina pessoal. Com a base
completa (148k registros), cada treinamento da MLP custaria vários segundos,
inviabilizando o estudo. Em `--sample-size 0` o pipeline usa a base inteira.

#### 8. Normalização Min-Max

Última etapa: **normalização linear Min-Max** em [0, 1], conforme exigido
pelo enunciado:

```
x' = (x − x_min) / (x_max − x_min)
```

Colunas constantes (variância zero) são preservadas com valor 0 e a divisão
é protegida por `denom = 1.0` quando `x_max == x_min`.

> A normalização foi feita **após** o split de classes para que o `x_min`
> e `x_max` reflitam a base efetiva usada. A normalização é aplicada antes
> do split treino/validação/teste — uma pequena fonte de "vazamento" de
> estatísticas que é aceitável neste cenário, dado que o GA não usa esses
> conjuntos para tomar decisões fora da MLP.

### Quadro-resumo da limpeza

| Etapa | Ação | Heurística |
|-------|------|------------|
| 1 | Leitura via pandas | extensão do arquivo |
| 2 | Descartar registros sem alvo | `label_cid` não nulo |
| 3 | Remover colunas de vazamento | lista explícita `TARGET_LEAK_COLUMNS` |
| 4 | Remover colunas muito esparsas | > 50% nulos |
| 5 | Codificar categóricas (LabelEncoder) | ≤ 50 valores únicos |
| 5 (alt) | Descartar categóricas | > 50 valores únicos |
| 6 | Imputar numéricos | mediana da coluna |
| 7 | Amostragem estratificada | 3000 registros (padrão) |
| 8 | Normalização Min-Max | linear em [0, 1] |

Após o pipeline, a base resultante tem tipicamente **L ≈ 32 atributos** e
3000 registros balanceados.

---

## Algoritmo Genético — escolhas de projeto

### Representação dos cromossomos (`src/ga/chromosome.py`)

**Codificação binária**: cada cromossomo é uma lista de comprimento L (= nº
total de atributos), onde:

- `gene = 1` → atributo **selecionado** (incluído no treinamento da MLP)
- `gene = 0` → atributo **descartado**

**Inicialização aleatória**: cada gene é 0 ou 1 com probabilidade 0,5,
amostragem independente. Um *failsafe* garante que pelo menos um gene esteja
ativo: se a inicialização produzir um cromossomo todo zeros (probabilidade
2⁻ᴸ, desprezível, mas possível), um gene aleatório é forçado a 1. O mesmo
*failsafe* é aplicado após a mutação.

**Atributos guardados em cada cromossomo:**

- `genes` — lista de 0s e 1s
- `fitness` — valor final da função de aptidão (após o cálculo via MLP)
- `f1_score` — F1-Score bruto (componente principal do fitness)
- `scaled_fitness` — fitness após normalização linear da população
  (usado pela seleção)
- `key()` — tupla imutável dos genes, usada como chave do cache

### Operadores genéticos

#### Crossover Uniforme (Pc = 0,85)

Implementado em `Chromosome.crossover`. Conforme a especificação do trabalho.

- Com probabilidade `Pc = 0,85`, dois pais geram dois filhos por amostragem
  independente posição-a-posição: para cada índice i, com 50% de chance o
  filho 1 recebe o gene do pai 1 (e o filho 2 do pai 2), e com 50% o
  contrário.
- Com probabilidade `1 − Pc = 0,15`, os filhos são **cópias exatas** dos pais
  (sem cruzamento naquela iteração).

> Optou-se pelo crossover **uniforme** (não 1- ou 2-pontos) porque o
> enunciado prescreve essa modalidade e porque, em problemas de seleção
> binária, o uniforme oferece maior diversidade exploratória — qualquer
> combinação de genes pode aparecer nos filhos.

#### Mutação bit-flip (Pm = 1/L)

Cada gene é invertido (`gene := 1 − gene`) de forma independente com
probabilidade `Pm = 1/L`. Isso resulta em, **em média, uma mutação por
cromossomo**, valor consagrado pela literatura para algoritmos genéticos
binários (Bäck, 1996). `L` aqui é o comprimento do cromossomo (≈ 32 após o
pré-processamento), então `Pm ≈ 0,031`.

Após a mutação, o *failsafe* garante novamente pelo menos um gene ativo.

### Seleção

**Torneio de tamanho 3** (`tournament_size = 3`):

1. Sorteia-se uma amostra aleatória de 3 indivíduos da população.
2. O vencedor é o de **maior `scaled_fitness`** (fitness normalizado
   linearmente, ver abaixo).

> A escolha do torneio em lugar de roleta foi motivada por dois fatores:
> (a) pressão seletiva controlada (não depende dos valores absolutos do
> fitness, que podem ter intervalos pequenos no final da evolução), e
> (b) compatibilidade direta com a *Normalização Linear* exigida — o
> torneio simplesmente compara valores escalados, sem precisar de roleta
> proporcional.

### Estratégia evolutiva — Steady-State com Gap = 2

Cada **geração** do algoritmo consiste em:

1. Selecionar **2 pais** via torneio (usando `scaled_fitness`).
2. Aplicar crossover uniforme → 2 filhos.
3. Aplicar mutação bit-flip aos 2 filhos.
4. Avaliar a aptidão dos 2 filhos (treinar a MLP em cada um deles).
5. Substituir os 2 piores indivíduos **fora da elite** pelos 2 filhos.
6. Recomputar `scaled_fitness` para a nova população.

A diferença para um AG geracional clássico é que apenas 2 indivíduos por
geração são trocados — o restante da população se mantém. Isso é vantajoso
porque o custo dominante (treinar a MLP) acontece poucas vezes por geração,
permitindo aplicar muitas gerações com tempo de execução administrável.

**Elitismo de 10 indivíduos:** a cada substituição, os 10 melhores
(por `fitness`) são preservados integralmente — só os indivíduos fora desse
top 10 podem ser substituídos. Isto garante monotonicidade do melhor
fitness ao longo das gerações.

### Critério de parada

Dois critérios, qualquer um que ocorra primeiro:

1. **200 gerações** (`max_generations`), ou
2. **20 gerações consecutivas sem melhoria** (`stagnation_limit`) — o melhor
   fitness global não aumenta em 20 iterações seguidas.

### Cache de fitness

A avaliação de um cromossomo é **cara** (custo de treinar uma MLP do zero).
Como a evolução produz com frequência cromossomos idênticos (especialmente
quando a população começa a convergir), guarda-se um cache em memória
indexado pela tupla `(genes)`:

```python
fitness_cache[chromosome.key()] = (f1, fitness)
```

Avaliações repetidas reutilizam o resultado anterior. Em testes empíricos,
o cache economiza de 20% a 40% das avaliações depois das primeiras 30
gerações.

### Resumo dos hiperparâmetros do AG

| Parâmetro | Valor | Onde está |
|---|---|---|
| População | 150 | `main.POPULATION_SIZE` |
| Crossover | Uniforme | `Chromosome.crossover` |
| Pc | 0,85 | `main.CROSSOVER_RATE` |
| Mutação | bit-flip | `Chromosome.mutate` |
| Pm | 1/L | calculado em `main.py` |
| Seleção | Torneio (3) | `algorithm._select_parent` |
| Elitismo | 10 | `main.ELITE_SIZE` |
| Estratégia | Steady-State, Gap = 2 | `main.GAP` |
| Máx. gerações | 200 | `main.MAX_GENERATIONS` |
| Estagnação | 20 sem melhoria | `main.STAGNATION_LIMIT` |

---

## Rede Neural (MLP) — escolhas de projeto

### Arquitetura (`src/nn/model.py`)

| Camada | Neurônios | Ativação | Observação |
|--------|-----------|----------|------------|
| Entrada | `input_dim` | — | dimensão variável: igual ao nº de genes ativos no cromossomo |
| Oculta 1 | **32** | ReLU | conforme especificação |
| Oculta 2 | **16** | ReLU | conforme especificação |
| Saída | `n_classes` (= 3) | **Softmax** | um neurônio por classe |

**Observações importantes:**

- A entrada **se adapta** ao cromossomo. Cada vez que um cromossomo é
  avaliado, instancia-se um modelo novo com `input_dim` igual ao número de
  atributos selecionados pelo cromossomo. Isso é o que permite o AG comparar
  configurações com tamanhos de entrada diferentes.
- Não há `Dropout` nem `BatchNormalization`. A especificação prescreve a
  arquitetura mínima e, dado o tamanho reduzido das camadas ocultas e o uso
  de *early stopping*, regularização adicional não foi necessária.
- A saída é **Softmax com 3 unidades**, gerando probabilidades para C53, C54
  e C55. A predição final é o `argmax`.

### Treinamento (`src/nn/trainer.py`)

| Aspecto | Valor | Justificativa |
|---------|-------|---------------|
| Otimizador | Adam (lr=0,001) | exigido pela especificação |
| Loss | `sparse_categorical_crossentropy` | rótulos como inteiros (0/1/2); evita necessidade de one-hot |
| Métrica de treino | accuracy | acompanhamento durante o treino |
| Batch size | 64 | bom compromisso entre vetorização (GPU/CPU) e ruído de gradiente |
| Épocas (máx.) | 30 | early stopping geralmente para antes |
| Early stopping | `patience=5`, `monitor="val_loss"`, `restore_best_weights=True` | escolhe a melhor configuração intermediária no conjunto de validação |
| Métrica de aptidão | F1-Score **macro** no teste | tratamento equitativo das 3 classes, mesmo com desbalanceamento |

### Divisão dos dados — 70/15/15 estratificada

A divisão segue exatamente o procedimento experimental do trabalho. É feita
**dentro do trainer**, a cada avaliação de cromossomo, de forma
**estratificada** (preserva proporção das três classes):

1. Primeiro split: 70% treino, 30% temporário (`stratify=y`).
2. Segundo split: do temporário, 50/50 entre validação e teste
   (`stratify=y_temp`).
3. Resultado: 70% treino / 15% validação / 15% teste.

> **`random_state` controlado por experimento:** o `random_state` da divisão
> depende do ID do experimento (semente = `42 + exp_id`). Isso significa que
> dentro de um mesmo experimento, todos os cromossomos avaliados são
> testados na **mesma** partição. Logo, comparar fitness entre cromossomos
> é justo (igualdade de oportunidades). Entre experimentos diferentes, as
> partições mudam, gerando variabilidade que é capturada pela curva de
> convergência média (com desvio-padrão).

### Função de aptidão — papel do conjunto de teste

- **Treino (70%)**: ajusta os pesos da MLP via Backpropagation.
- **Validação (15%)**: usada pelo `EarlyStopping` para escolher a melhor
  configuração (menor `val_loss`).
- **Teste (15%)**: usado **uma única vez** ao final, para gerar o F1-Score
  reportado como fitness.

O fitness é o F1-Score medido no conjunto de teste — dados que a rede nunca
viu durante o treino e nem usou para selecionar a melhor versão dos pesos.

---

## Função de aptidão e escalonamento

### Fórmula

```
Fitness = 0,9 × F1-Score + 0,1 × (1 − Ns/Nt)
```

Cada componente:

- **F1-Score (peso 0,9)**: macro F1 no conjunto de teste. Maximiza a
  qualidade preditiva.
- **Parcimônia (peso 0,1)**: `1 − Ns/Nt` recompensa cromossomos que usam
  poucos atributos. Por exemplo, um cromossomo com 8 de 32 atributos ativos
  recebe um bônus de `0,1 × (1 − 8/32) = 0,075`.

Cromossomos com **nenhum atributo ativo** recebem fitness = 0 (caso de borda
para o qual não é possível treinar a rede).

### Normalização linear da população

Antes de cada seleção, os fitness são escalados linearmente para [0, 1]:

```
scaled = (fitness − f_min) / (f_max − f_min)
```

Quando todos os fitness são iguais (caso degenerado: convergência total),
todos recebem `scaled = 1,0`.

A normalização tem duas funções:

1. Mantém pressão seletiva constante mesmo quando o intervalo de fitness se
   estreita (todos próximos do ótimo).
2. Evita que diferenças minúsculas no fitness bruto sejam ignoradas pela
   seleção.

A função `linear_scale_population` é chamada após cada nova avaliação.

---

## Procedimento experimental

### 20 experimentos independentes

O loop principal (`src/main.py`) executa **N experimentos independentes**
(padrão N = 20). Para cada experimento:

1. **Seed reset** — semente fixada para `numpy`, `random` e `tensorflow` em
   `42 + exp_id`, garantindo reprodutibilidade.
2. População inicial gerada aleatoriamente.
3. Avaliação completa da população inicial (geração 0).
4. Loop steady-state até atingir o critério de parada.
5. Resultado final: melhor cromossomo, fitness e F1.

### Curva média de convergência

Ao final dos 20 experimentos:

- Para cada geração g, calcula-se a **média e o desvio-padrão** do melhor
  fitness daquela geração entre os 20 experimentos.
- Experimentos que pararam antes da geração g têm seu último valor
  propagado (`ffill`), evitando "buracos" na média.
- A curva é plotada em `plots/ga_convergencia_media.png` com a média como
  linha sólida e a banda `± 1 σ` em transparência.

### Métricas finais reportadas

Após todos os experimentos:

- Fitness médio e desvio-padrão (entre os 20 melhores).
- F1-Score médio e desvio-padrão.
- Número médio de atributos selecionados.
- Identificação do **melhor experimento** (maior fitness) e a lista dos
  atributos selecionados nele.
- Tempo total de execução.

---

## Saídas geradas

Após uma execução completa, são produzidos:

### Logs (`logs/`)

`ga_metrics.csv` — uma linha por (experimento, geração):

| Coluna | Conteúdo |
|---|---|
| `experimento` | ID do experimento (0..N−1) |
| `geracao` | Número da geração |
| `melhor_fitness` | Maior fitness da população |
| `fitness_medio` | Fitness médio da população |
| `pior_fitness` | Menor fitness da população |
| `melhor_f1` | F1-Score do melhor indivíduo (sem o termo de parcimônia) |
| `num_atributos_ativos` | Nº de genes = 1 no melhor cromossomo |
| `melhor_cromossomo` | Máscara binária (string concatenada) |

`nn_metrics.csv` — uma linha por avaliação de cromossomo:

| Coluna | Conteúdo |
|---|---|
| `id_cromossomo` | Identificador único (`expX_genY_childZ`) |
| `geracao` | Geração da avaliação |
| `loss_treino` | Loss final no treino |
| `loss_validacao` | Loss final na validação |
| `acuracia_validacao` | Acurácia final na validação |
| `f1_score` | F1-Score macro no teste |
| `epocas` | Quantidade de épocas até parar (early stop ≤ 30) |
| `num_atributos_usados` | Tamanho da entrada da MLP |

### Gráficos (`plots/`)

- **`ga_convergencia_media.png`** — média do melhor fitness por geração
  (entre os N experimentos), com banda de ± 1 desvio-padrão. **Gráfico
  principal exigido pelo trabalho.**
- **`ga_convergencia_por_experimento.png`** — uma curva por experimento, em
  transparência, para inspeção visual da consistência.
- **`ga_fitness_componentes.png`** — melhor/médio/pior fitness do primeiro
  experimento, ao longo das gerações.
- **`nn_atributos_vs_f1.png`** — dispersão F1-Score versus número de
  atributos ativos, colorida por geração — útil para visualizar o
  trade-off parcimônia × qualidade.

---

## Decisões pragmáticas

A especificação é completa mas implícita sobre alguns detalhes; estas são as
escolhas adicionais feitas para tornar o trabalho executável e reproduzível:

| Decisão | Justificativa |
|---------|---------------|
| Amostragem estratificada (3000 registros, padrão) | Sem amostragem, cada treino da MLP custa ~5s; com 20 × ~400 evaluações, levaria > 11h. Com 3000 registros, fica em ~30 min. |
| LabelEncoder para categóricas em vez de One-Hot | Mantém o cromossomo curto (~32 genes) e fiel à ideia de "selecionar atributos" da base, não "selecionar dummies". |
| Limite de 50% de NaN para descartar coluna | Compromisso entre perda de informação e imputação pesada. |
| Limite de 50 valores únicos para descartar categóricas | Acima disso, são IDs ou nomes livres — sem informação discriminativa útil. |
| Imputação por mediana (não média) | Robusta a outliers presentes em colunas como `OCUP`, `CODMUN*`. |
| `EarlyStopping(patience=5)` na MLP | Evita gastar todas as 30 épocas em redes que já convergiram, sem perda de qualidade (retorna os melhores pesos). |
| Cache de fitness por tupla de genes | Reduz 20–40% das avaliações na fase tardia da evolução. |
| Seed = `42 + exp_id` | Reprodutibilidade total; experimentos independentes mas determinísticos. |
| Saída Softmax + `sparse_categorical_crossentropy` | Evita o overhead de one-hot encoding dos rótulos. |

---

## Referências de material

- `docs/ga-mlp-task.pdf` — enunciado oficial do trabalho.
- `docs/ga-material/` — material de apoio sobre Algoritmos Genéticos.
- `docs/mlp-material/` — material de apoio sobre Redes Neurais.
