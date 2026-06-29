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
4. [Análise exploratória dos dados (EDA)](#análise-exploratória-dos-dados-eda)
5. [Inconsistências detectadas](#inconsistências-detectadas)
6. [Pipeline de limpeza](#pipeline-de-limpeza)
7. [Algoritmo Genético — escolhas de projeto](#algoritmo-genético--escolhas-de-projeto)
8. [Rede Neural (MLP) — escolhas de projeto](#rede-neural-mlp--escolhas-de-projeto)
9. [Função de aptidão e escalonamento](#função-de-aptidão-e-escalonamento)
10. [Métrica F1: por que weighted e não macro](#métrica-f1-por-que-weighted-e-não-macro)
11. [Procedimento experimental](#procedimento-experimental)
12. [Saídas geradas](#saídas-geradas)
13. [Decisões pragmáticas](#decisões-pragmáticas)

---

## Setup

O projeto usa [uv](https://docs.astral.sh/uv/) para dependências e
[git-lfs](https://git-lfs.com) para versionar os arquivos pesados da base de
dados (≈ 150 MB).

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

### 2. Instalar o git-lfs

**macOS**
```bash
brew install git-lfs
```

**Ubuntu/Debian**
```bash
sudo apt install git-lfs
```

**Windows**
```powershell
choco install git-lfs
```

Após instalar, inicialize uma vez por usuário:

```bash
git lfs install
```

### 3. Clonar o repositório

```bash
git clone https://github.com/Carlos-H1307/IC-Trabalho-IA.git
cd IC-Trabalho-IA
```

O `git clone` já baixa os arquivos `.xlsx` da base via LFS automaticamente.
Verifique:

```bash
ls -lh data/
# dataset-short.xlsx     ~33M
# dataset-complete.xlsx  ~115M
```

Se os arquivos vierem com poucos KB (são ponteiros LFS), rode:

```bash
git lfs pull
```

### 4. Instalar as dependências

```bash
uv sync
```

Cria automaticamente o `.venv/` e instala tudo declarado em `pyproject.toml`
(TensorFlow/Keras, scikit-learn, pandas, openpyxl, matplotlib).

---

## Execução

### Análise Exploratória (EDA)

Gera todos os relatórios e gráficos descritivos da base bruta:

```bash
uv run python src/eda.py
```

Saídas em `reports/` (commitado no repo para fins de relatório):
- `eda_summary.txt` — resumo textual completo
- `eda_class_distribution.png` — distribuição das 3 classes
- `eda_missingness.png` — % de NaN por coluna
- `eda_age_by_class.png` — boxplot de idade por classe
- `eda_temporal.png` — evolução temporal das classes
- `eda_correlation.png` — matriz de correlação numérica
- `eda_numeric_stats.csv` — describe() das numéricas
- `eda_categorical_stats.csv` — contagem das categóricas

### Pipeline principal (GA + MLP)

Execução completa, conforme a especificação do trabalho (20 experimentos × até 200 gerações):

```bash
uv run python src/main.py
```

Modo rápido para validar o pipeline:

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
| `--data-path`       | `data/dataset-short.xlsx`           | Caminho do arquivo de entrada |
| `--population`      | `150`                                | Tamanho da população do AG |
| `--generations`     | `200`                                | Limite superior de gerações |
| `--experiments`     | `20`                                 | Quantidade de execuções independentes |
| `--sample-size`     | `3000`                               | Amostra estratificada (0 = base completa) |
| `--quick`           | —                                    | Atalho: 2 experimentos × 20 gerações, pop 30 |

---

## Estrutura do projeto

```
src/
├── main.py              # ponto de entrada do GA, CLI, loop de experimentos
├── eda.py               # análise exploratória de dados (script independente)
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

reports/                 # saídas da EDA (versionadas no repo)
data/                    # base de dados (versionada via git-lfs)
logs/                    # CSVs gerados pelo GA (gitignored)
plots/                   # gráficos do GA (gitignored)
```

---

## Análise exploratória dos dados (EDA)

Resultados produzidos por `src/eda.py` sobre a base
`dataset-short.xlsx` (148.785 registros × 50 colunas).

### Origem

Base disponibilizada pelo professor (via Teams) e versionada no repositório
através de git-lfs. Contém registros do Sistema de Informações sobre
Mortalidade (SIM/DATASUS) referentes a óbitos por câncer do colo do útero
(códigos CID-10 C53, C54, C55). Dois arquivos foram
disponibilizados:

| Arquivo                    | Registros | Colunas |
|----------------------------|-----------|---------|
| `dataset-short.xlsx`       | 148.785   | 50      |
| `dataset-complete.xlsx`    | 148.785   | 167     |

**Versão utilizada:** `dataset-short.xlsx`. A versão completa traz colunas
redundantes (descritores textuais, datas decompostas em dia/mês/ano,
coordenadas geográficas) que não adicionam sinal e aumentam o custo
computacional sem benefício.

### Variável-alvo (`label_cid`)

Três classes (códigos CID-10):

| Código | Descrição                                          | Casos    | %      |
|--------|----------------------------------------------------|----------|--------|
| **C53** | Neoplasia maligna do **colo** do útero            | 92.287   | 62,03  |
| **C55** | Neoplasia maligna do útero, **porção não esp.**   | 30.345   | 20,40  |
| **C54** | Neoplasia maligna do **corpo** do útero           | 26.153   | 17,58  |

**Razão maior/menor = 3,53×** → base **fortemente desbalanceada**. Esta é
uma observação crítica para a escolha da métrica de aptidão (ver
[Métrica F1](#métrica-f1-por-que-weighted-e-não-macro)).

### Período coberto

Óbitos de **2010 a 2024** (15 anos). Volume crescente ao longo do tempo,
saindo de ~8.100 registros/ano em 2010 para ~12.000 em 2024. A proporção das
classes muda discretamente ao longo do tempo (C53 estável em ~62%, C54
crescendo de 12% para 22%, C55 caindo de 26% para 15%) — sugerindo
melhora na precisão de classificação ao longo dos anos, e não mudança
epidemiológica real.

### Distribuição de idade por classe

Resultado descritivo (em anos):

| Classe | n      | Média | Mediana | Std  | Min | Max |
|--------|--------|-------|---------|------|-----|-----|
| C53    | 92.283 | 56,7  | 56      | 16,0 | 13  | 113 |
| C54    | 26.152 | 67,8  | 68      | 12,3 | 13  | 114 |
| C55    | 30.339 | 63,3  | 64      | 15,7 | 11  | 111 |

**Achado:** a idade tem **sinal discriminativo claro**. C53 (colo) ocorre em
média ~11 anos mais cedo que C54 (corpo). Isso é coerente com a literatura
epidemiológica (câncer do colo associado a HPV em mulheres mais jovens;
câncer do corpo associado a fatores hormonais pós-menopausa). Espera-se que
o GA selecione `idade_obito_anos` na maior parte dos cromossomos vencedores.

### Variáveis disponíveis (50 colunas brutas)

- **Identificação e geografia**: NATURAL, CODMUNNATU, DTNASC, CODMUNRES,
  CODMUNOCOR, res_/ocor_* (capital, fronteira, Amazônia, UF, região)
- **Demografia**: SEXO, RACACOR, ESTCIV, ESC, ESC2010, OCUP, idade_obito_anos
- **Circunstâncias do óbito**: TIPOBITO, DTOBITO, LOCOCOR, CODESTAB,
  ASSISTMED, EXAME, CIRURGIA, NECROPSIA
- **Causa (vazamento)**: CAUSABAS, causabas_categoria, causabas_subcategoria,
  LINHAA, LINHAB, LINHAC, LINHAD, LINHAII, CB_PRE, CAUSABAS_O
- **Outras**: ESCFALAGR1, SERIESCFAL, NUDIASOBCO, NUDIASINF, ALTCAUSA,
  CAUSAMAT, COMUNSVOIM, ESTABDESCR
- **Alvo**: label_cid

---

## Inconsistências detectadas

Resultados das checagens automáticas (`data_loader._check_*` e seções de
diagnóstico da EDA):

| Inconsistência                            | Detectado | Ação no pipeline |
|-------------------------------------------|----------|------------------|
| Registros com `label_cid` nulo            | **0**    | Removidos (`dropna(subset)`) — proteção mesmo sem casos |
| Duplicatas exatas                         | **3**    | Removidas (`drop_duplicates`) |
| Idades fora de [0, 120]                   | **0**    | — |
| Idades nulas                              | **11**   | Imputadas pela mediana (56 anos) |
| Datas inconsistentes (nasc > óbito)       | **0**    | — |
| Anos fora de [1996, 2025]                 | **0**    | — |
| `SEXO ≠ 2` (não-feminino)                 | **0**    | — (esperado: câncer de útero) |
| Coluna `SEXO` constante                   | **Sim**  | **Removida** — variância zero |
| Coluna `TIPOBITO` constante               | **Sim**  | **Removida** — variância zero (sempre 2 = óbito não-fetal) |
| Coluna `CAUSAMAT` 100% nula               | **Sim**  | Removida (variância zero) |
| Coluna `NUDIASINF` 100% nula              | **Sim**  | Removida (variância zero) |
| Código 9 ("ignorado") dominante em EXAME  | **94%**  | **Coluna removida** |
| Código 9 ("ignorado") dominante em CIRURGIA | **95%** | **Coluna removida** |
| Coluna `ESTABDESCR` 100% NaN              | **Sim**  | Removida por **quase-constância** (moda > 95%) |
| Coluna `COMUNSVOIM` 96,5% NaN             | **Sim**  | Removida por **quase-constância** |
| Coluna `ALTCAUSA` 100% NaN                | **Sim**  | Removida por **quase-constância** |
| `CODMUNOCOR` ↔ `ocor_CODIGO_UF` (r=1,000) | **Sim**  | Uma coluna removida por **correlação alta** |
| `CODMUNRES` ↔ `CODMUNOCOR` (r=0,989)      | **Sim**  | Uma coluna removida por **correlação alta** |
| Código 9 residual em ASSISTMED (34%), NECROPSIA (29%), ESC (16%)... | **Sim** | **Convertido para NaN** e imputado pela mediana |

### Sobre o código 9 ("ignorado") do DATASUS

No padrão SIM/DATASUS, o valor `9` em campos categóricos representa
*"ignorado"* — informação ausente, não uma categoria real. Tratá-lo como
categoria normal mistura ausência de informação com categorias válidas e
introduz ruído. O pipeline aplica duas estratégias:

1. **Descarte por dominância (etapa 7)**: para colunas em
   `IGNORADO_CODE_COLUMNS` (RACACOR, ESTCIV, ESC, ASSISTMED, EXAME, CIRURGIA,
   NECROPSIA, etc.) onde **mais de 80%** dos valores são 9, a coluna é
   descartada inteira — não há sinal útil. Aplicado a `EXAME` (94%) e
   `CIRURGIA` (95%).
2. **Conversão para NaN (etapa 9)**: para as colunas que sobreviveram à
   etapa anterior, todo valor 9 é substituído por NaN antes da imputação.
   Isso evita que o LabelEncoder/median fitting trate "ignorado" como uma
   categoria válida. Aplicado a 8 colunas, com ~155 mil substituições:
   ASSISTMED (50k), NECROPSIA (43k), ESC (23k), ESTCIV (15k), ESC2010 (13k),
   ESCFALAGR1 (11k), RACACOR (5k), LOCOCOR (46).

### Quase-constância (moda > 95%)

Mesmo quando uma coluna não é tecnicamente "constante" (variância zero),
se um único valor concentra mais de **95%** dos registros, ela tem variância
pequena demais para a MLP aprender padrões úteis. O detector
(`_drop_near_constant_columns`) captura colunas que escapariam dos filtros
de "variância zero" e "alto NaN" — por exemplo, colunas com mistura de NaN
+ um único valor real.

### Correlação alta (|r| > 0,95)

Pares de colunas numéricas com correlação absoluta acima de 0,95 são
redundantes. O detector (`_drop_highly_correlated_columns`) calcula a matriz
de correlação (com imputação local por mediana só para o cálculo) e remove
**a segunda coluna** de cada par. Pares descobertos na base:

| Par                                       | r     | Mantida | Removida |
|-------------------------------------------|-------|---------|----------|
| `CODMUNRES` ↔ `CODMUNOCOR`                | 0,989 | CODMUNRES | CODMUNOCOR |
| `CODMUNRES` ↔ `ocor_CODIGO_UF`            | 0,989 | CODMUNRES | ocor_CODIGO_UF |

A explicação geográfica é direta: a grande maioria dos óbitos ocorre no
mesmo município (e portanto UF) de residência. Manter as três colunas
seria redundância pura.

### Por que não tratar outliers nas numéricas

A auditoria detectou apenas **7 outliers** em `idade_obito_anos` (1,5×IQR),
todos no intervalo [11–13] ou [111–114] — valores plausíveis para casos
médicos extremos. Outliers detectados em colunas como `CODMUNRES`,
`CODESTAB`, `OCUP` não são outliers reais: são **códigos categóricos
disfarçados de números** (códigos de município, IDs de estabelecimento,
códigos especiais de ocupação). Tratá-los com IQR distorceria o significado
sem ganho informacional. A normalização Min-Max já comprime tudo para [0,1].

---

## Pipeline de limpeza

O pré-processamento (`src/data_loader.py`) é determinístico e segue 15 etapas
**nesta ordem**:

| #  | Etapa                                                    | Heurística                              |
|----|----------------------------------------------------------|-----------------------------------------|
| 1  | Leitura do arquivo (XLSX/CSV)                            | extensão do arquivo                     |
| 2  | Remoção de registros sem alvo                            | `dropna(subset=["label_cid"])`          |
| 3  | Remoção de duplicatas exatas                             | `drop_duplicates`                       |
| 4  | Checagens de inconsistência (auditoria, não remove)      | datas, idade                            |
| 5  | Remoção de colunas de **vazamento de alvo**              | lista `TARGET_LEAK_COLUMNS` (14 cols)   |
| 6  | Remoção de **colunas constantes** (variância zero)       | `nunique(dropna=False) ≤ 1`             |
| 7  | Remoção de colunas dominadas por "ignorado" (9 > 80%)    | `(col == 9).mean() > 0.80`              |
| 8  | Remoção de colunas **quase-constantes**                  | moda > 95% dos registros                |
| 9  | Conversão de "9" residual ("ignorado") para NaN          | colunas em `IGNORADO_CODE_COLUMNS`      |
| 10 | Remoção de colunas com excesso de NaN                    | `> 50%` de valores nulos                |
| 11 | Remoção de colunas numéricas **redundantes**             | `|corr| > 0,95`                          |
| 12 | Codificação de categóricas                                | `LabelEncoder` se `nunique ≤ 50`; descarte se acima |
| 13 | Imputação de NaN numéricos                                | mediana da coluna                       |
| 14 | Amostragem estratificada (opcional)                       | 3000 registros (padrão)                 |
| 15 | Normalização **Min-Max linear**                           | `(x − x_min) / (x_max − x_min)` em [0,1] |

Após o pipeline, **L = 26 atributos** (de 50 originais) com **3000 registros**
balanceados estratificadamente (padrão).

### Removidos por categoria (resumo)

| Causa do descarte                       | Qtd | Colunas |
|-----------------------------------------|-----|---------|
| Vazamento de alvo                       | 10  | `CAUSABAS`, `CAUSABAS_O`, `CB_PRE`, `causabas_categoria`, `causabas_subcategoria`, `LINHAA`, `LINHAB`, `LINHAC`, `LINHAD`, `LINHAII` |
| Variância zero                          | 4   | `TIPOBITO`, `SEXO`, `CAUSAMAT`, `NUDIASINF` |
| Ignorado dominante (9 > 80%)            | 2   | `EXAME`, `CIRURGIA` |
| Quase-constância (moda > 95%)           | 3   | `ESTABDESCR` (100% NaN), `COMUNSVOIM` (96,5% NaN), `ALTCAUSA` (100% NaN) |
| Excesso de NaN (> 50%)                  | 2   | `SERIESCFAL` (82%), `NUDIASOBCO` (81%) |
| Correlação alta (\|r\| > 0,95)           | 2   | `CODMUNOCOR`, `ocor_CODIGO_UF` (perfeitamente correlacionadas com `CODMUNRES`) |
| Alta cardinalidade categórica           | 1   | `ESTABDESCR`-like / nomes livres |
| **Total descartado**                    | **24** | de 50 originais (incluindo `label_cid`) |

### Justificativas das escolhas

- **LabelEncoder em vez de One-Hot**: mantém o cromossomo curto (~26 genes).
  Se usássemos One-Hot, colunas como `RACACOR` (5 valores) virariam 5 genes
  individuais, explodindo `L` para perto de 100 e mudando a natureza do
  problema (selecionar atributos × selecionar indicadores).
- **Limiar de 50% de NaN para descarte**: equilibra perda de informação com
  ruído de imputação massiva. Para colunas muito esparsas, a mediana seria
  representativa demais da minoria que respondeu.
- **Limiar de 95% para quase-constância**: além da variância zero (`nunique==1`),
  colunas com 95% do mesmo valor têm sinal informacional negligenciável.
- **Limiar de 0,95 para correlação**: pares com |r| ≥ 0,95 são funcionalmente
  redundantes. Mantém-se uma coluna por par; o GA fica livre para selecionar
  a representação geográfica/categórica restante sem disputa entre clones.
- **Conversão de 9→NaN antes da imputação**: a alternativa (tratar 9 como
  categoria) misturaria "ignorado" com categorias reais no mesmo eixo
  numérico, degradando o sinal. A imputação pela mediana faz uma assunção
  conservadora ("provavelmente é o valor mais comum").
- **Imputação por mediana** (não média): robusta a outliers em colunas como
  `OCUP` (códigos ocupacionais com escalas estranhas) e `CODMUN*` (códigos
  de município com grandes saltos numéricos).
- **Min-Max em [0,1]**: exigido pela especificação. Mantém todos os
  atributos na mesma escala para a entrada da MLP.
- **Outliers numéricos preservados**: a auditoria mostrou que praticamente
  todos os "outliers" detectados por IQR em colunas como `CODMUNRES`,
  `CODESTAB`, `OCUP` são **códigos categóricos**, não valores extremos de
  uma escala contínua. Tratá-los com winsorização destruiria o significado
  semântico.

---

## Algoritmo Genético — escolhas de projeto

### Representação dos cromossomos (`src/ga/chromosome.py`)

**Codificação binária**: cada cromossomo é uma lista de comprimento L (= nº
total de atributos), onde:

- `gene = 1` → atributo **selecionado**
- `gene = 0` → atributo **descartado**

**Inicialização aleatória**: cada gene é 0 ou 1 com p=0,5, independente. Um
*failsafe* garante ≥ 1 gene ativo (sorteio de cromossomo todo-zeros é
desprezível mas possível).

**Atributos armazenados em cada cromossomo:**

- `genes` — lista de 0s e 1s
- `fitness` — valor final da função de aptidão
- `f1_score` — F1 weighted bruto (componente principal do fitness)
- `scaled_fitness` — fitness após normalização linear da população
- `key()` — tupla imutável, usada como chave do cache

### Operadores genéticos

#### Crossover Uniforme (Pc = 0,85)

Implementado em `Chromosome.crossover`:

- Com probabilidade `Pc = 0,85`, dois pais geram dois filhos por amostragem
  independente posição-a-posição: para cada índice i, com 50% de chance o
  filho 1 recebe o gene do pai 1 (e o filho 2 do pai 2), e com 50% o
  contrário.
- Com probabilidade `1 − Pc = 0,15`, os filhos são **cópias exatas** dos pais.

#### Mutação bit-flip (Pm = 1/L)

Cada gene é invertido com probabilidade `Pm = 1/L`. Em média, **1 mutação
por cromossomo** — valor consagrado para AGs binários (Bäck, 1996). Com L≈28,
`Pm ≈ 0,036`.

### Seleção: Torneio de tamanho 3

1. Sorteia-se 3 indivíduos da população.
2. Vencedor = maior `scaled_fitness`.

Optou-se por torneio em vez de roleta porque:
(a) pressão seletiva controlada independentemente da escala dos fitness, e
(b) integra naturalmente com a normalização linear exigida.

### Estratégia evolutiva — Steady-State, Gap = 2

Cada **geração** do algoritmo:

1. Seleciona 2 pais via torneio.
2. Aplica crossover uniforme → 2 filhos.
3. Aplica mutação bit-flip aos 2 filhos.
4. Avalia a aptidão dos 2 filhos (treinar MLP em cada).
5. Substitui os 2 piores indivíduos **fora da elite**.
6. Recomputa `scaled_fitness` para a nova população.

### Elitismo (10 indivíduos)

Os 10 melhores por `fitness` são preservados a cada geração — só os indivíduos
fora desse top 10 podem ser substituídos. Garante monotonicidade do melhor
fitness ao longo das gerações.

### Critério de parada

Dois critérios, qualquer um que ocorra primeiro:

1. **200 gerações** (`max_generations`), ou
2. **20 gerações consecutivas sem melhoria** (`stagnation_limit`).

### Cache de fitness

Avaliações repetidas (cromossomos idênticos pela tupla de genes) reutilizam
o resultado anterior. Economia empírica: 20–40% após as primeiras 30
gerações.

### Resumo dos hiperparâmetros

| Parâmetro       | Valor                  | Onde                       |
|-----------------|------------------------|----------------------------|
| População       | 150                    | `main.POPULATION_SIZE`     |
| Crossover       | Uniforme               | `Chromosome.crossover`     |
| Pc              | 0,85                   | `main.CROSSOVER_RATE`      |
| Mutação         | bit-flip               | `Chromosome.mutate`        |
| Pm              | 1/L                    | calculado em `main.py`     |
| Seleção         | Torneio (3)            | `algorithm._select_parent` |
| Elitismo        | 10                     | `main.ELITE_SIZE`          |
| Estratégia      | Steady-State, Gap = 2  | `main.GAP`                 |
| Máx. gerações   | 200                    | `main.MAX_GENERATIONS`     |
| Estagnação      | 20 sem melhoria        | `main.STAGNATION_LIMIT`    |

---

## Rede Neural (MLP) — escolhas de projeto

### Arquitetura (`src/nn/model.py`)

| Camada    | Neurônios          | Ativação | Observação                                 |
|-----------|--------------------|----------|--------------------------------------------|
| Entrada   | `input_dim`        | —        | igual ao nº de genes ativos no cromossomo  |
| Oculta 1  | **32**             | ReLU     | conforme especificação                     |
| Oculta 2  | **16**             | ReLU     | conforme especificação                     |
| Saída     | `n_classes` (= 3)  | Softmax  | um neurônio por classe                     |

A entrada **se adapta** ao cromossomo: cada avaliação instancia um modelo
novo com `input_dim` igual aos atributos selecionados. É isso que permite
ao GA comparar configurações com tamanhos de entrada diferentes.

Sem `Dropout` ou `BatchNormalization` — não foram pedidos e o early stopping
já cobre regularização.

### Treinamento (`src/nn/trainer.py`)

| Aspecto             | Valor                                                          | Justificativa |
|---------------------|----------------------------------------------------------------|---------------|
| Otimizador          | Adam (lr=0,001)                                                | exigido pela especificação |
| Loss                | `sparse_categorical_crossentropy`                              | rótulos como inteiros 0/1/2; evita one-hot |
| Métrica de treino   | accuracy                                                       | acompanhamento durante o treino |
| Batch size          | 64                                                             | compromisso vetorização × ruído de gradiente |
| Épocas (máx.)       | 30                                                             | early stopping geralmente para antes |
| Early stopping      | `patience=5`, `monitor="val_loss"`, `restore_best_weights=True` | "melhor configuração = menor erro de validação" (spec) |
| **Métrica de aptidão** | **F1-Score weighted** no teste                              | reflete desbalanceamento real (ver seção dedicada) |
| Métrica auxiliar    | F1-Score macro                                                 | logado para diagnóstico de viés por classe |

### Divisão dos dados — 70/15/15 estratificada

A divisão acontece **dentro da função de fitness** (`trainer._split_70_15_15`),
ou seja, **a cada avaliação de cromossomo**. Não é um split único feito no
início — é refeito a cada chamada de `train_and_evaluate_nn`.

#### Mecânica: dois `train_test_split` em cascata

```python
# 1º split: 70% treino, 30% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=random_state, stratify=y
)
# 2º split: temp dividido 50/50 → 15% val, 15% teste
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
)
```

| Conjunto    | %    | Uso |
|-------------|------|-----|
| Treino      | 70%  | Ajusta pesos via Backpropagation |
| Validação   | 15%  | Monitorado pelo `EarlyStopping(monitor="val_loss")` para escolher a melhor versão dos pesos (`restore_best_weights=True`) |
| Teste       | 15%  | Mede F1-Score reportado como fitness — **uma única vez por cromossomo** |

#### Estratificação

Os dois splits recebem `stratify=y` (no primeiro) e `stratify=y_temp` (no
segundo). Isso garante que **cada conjunto preserva a proporção original das
3 classes** (~62% C53 / 20% C55 / 18% C54). Crítico para uma base
desbalanceada: sem estratificação, era possível um split degenerado com
poucos exemplos de C54 no teste, inflando ou deflando o F1 por sorte do
sorteio.

#### Propagação do `random_state`

A semente flui do `main.py` até o trainer:

```
main.py: random_state = 42 + exp_id
   ↓
GeneticAlgorithm.__init__(random_state=...)
   ↓
evaluate_chromosome(random_state=...)        em fitness.py
   ↓
train_and_evaluate_nn(random_state=...)      em trainer.py
   ↓
_split_70_15_15(random_state=...)            mesma semente para os 2 splits
```

Consequência prática:

| Cenário | Comportamento |
|---|---|
| Dois cromossomos no **mesmo experimento** | Recebem **a mesma semente** → mesma partição treino/val/teste → comparação **justa** de fitness |
| Mesmo cromossomo em **experimentos diferentes** | Recebe `42 + exp_id` diferente → partição diferente → fitness varia. Essa variância alimenta a banda de desvio na curva média de convergência |

#### Alternativas consideradas

- **K-fold cross-validation**: estatisticamente mais robusto, mas multiplica
  o custo por `k`. Para 20 experimentos × ~400 avaliações × k folds, o tempo
  fica inviável. A spec não pediu.
- **Split único globalmente fixo (mesma semente em todos os 20 experimentos)**:
  mais simples, mas a "curva média de convergência" perderia o sinal de
  variabilidade do split — só capturaria ruído da inicialização aleatória
  da população do GA.
- **Split antes do GA, fora do loop**: mais eficiente (split não é refeito
  a cada cromossomo), mas as duas abordagens são funcionalmente
  equivalentes porque dentro de um experimento o `random_state` é
  constante. Ganho marginal não justifica a refatoração.

---

## Função de aptidão e escalonamento

### Fórmula

```
Fitness = 0,9 × F1-Score + 0,1 × (1 − Ns/Nt)
```

- **F1-Score (peso 0,9)**: F1 *weighted* no conjunto de teste. Maximiza a
  qualidade preditiva ponderada pelo desbalanceamento real.
- **Parcimônia (peso 0,1)**: `1 − Ns/Nt` recompensa cromossomos com poucos
  atributos. Para Nt=26, um cromossomo com 8 atributos ganha bônus
  `0,1 × (1 − 8/26) = 0,069`.

Cromossomos com **zero atributos ativos** recebem fitness = 0 (caso de borda).

### Normalização linear da população

Antes de cada seleção, fitness escalados para [0, 1]:

```
scaled = (fitness − f_min) / (f_max − f_min)
```

Quando todos os fitness são iguais (caso degenerado), todos recebem `scaled = 1,0`.

Função: `linear_scale_population`, chamada após cada nova avaliação.

---

## Métrica F1: por que weighted e não macro

A especificação só diz "F1-Score". Para problemas multiclasse, é preciso
escolher entre os esquemas de agregação: **macro**, **weighted**, **micro**.

A escolha **importa muito** porque a base é fortemente desbalanceada:

| Classe | Frequência | Esquema macro | Esquema weighted |
|--------|-----------|---------------|------------------|
| C53    | 62,03%    | peso 1/3      | peso 0,6203      |
| C54    | 17,58%    | peso 1/3      | peso 0,1758      |
| C55    | 20,40%    | peso 1/3      | peso 0,2040      |

**Comportamento de cada esquema:**

- **F1 macro** trata as 3 classes igualmente. Um modelo que acerta 95% em C53
  mas só 30% em C54 e C55 fica com macro F1 ≈ 0,52 — fortemente penalizado.
- **F1 weighted** pondera pelo suporte real. O mesmo modelo fica com weighted F1
  ≈ 0,76, refletindo o desempenho esperado em produção (onde 62% dos casos
  reais são C53).
- **F1 micro** (= accuracy em multiclasse) sempre favorece a classe majoritária.

**Por que weighted como fitness primário:**

1. **Realismo populacional.** Se o sistema for usado para classificar novos
   óbitos, a distribuição esperada é ~62/20/18, não 33/33/33. O weighted
   reflete a métrica relevante.
2. **Decisão consciente do trade-off.** Não estamos cegos ao desbalanceamento
   — estamos **ponderando deliberadamente** pelo suporte real, em vez de
   fingir que as classes são igualmente prevalentes.
3. **Métrica macro ainda é logada** para diagnóstico. Em `nn_metrics.csv`,
   a coluna `f1_macro` permite verificar se o modelo está ignorando
   classes minoritárias. Se `f1_macro` ficar muito abaixo de `f1_weighted`,
   indica viés para C53.

**Alternativa considerada e descartada:** transformar o problema em binário
(C53 vs. não-C53) — alinharia com o título do trabalho ("câncer **do colo**
do útero"), mas a especificação pede "número de neurônios igual ao número
de classes da base", que são 3. Manter as 3 classes está correto.

---

## Procedimento experimental

### 20 experimentos independentes

Loop em `src/main.py`:

1. **Seed reset** — semente fixada para `numpy`, `random`, `tensorflow` em
   `42 + exp_id`, garantindo reprodutibilidade.
2. População inicial gerada aleatoriamente.
3. Avaliação completa da população inicial (geração 0).
4. Loop steady-state até o critério de parada.
5. Resultado final: melhor cromossomo, fitness e F1.

### Curva média de convergência

Ao final dos 20 experimentos:

- Para cada geração g, calcula-se **média e desvio-padrão** do melhor
  fitness daquela geração entre os 20 experimentos.
- Experimentos parados antes da geração g têm seu último valor propagado
  (`ffill`) para evitar buracos na média.
- A curva final é a **média dos melhores em 20 experimentos** — exatamente
  o gráfico exigido no item 8 da especificação.

### Métricas finais reportadas

- Fitness médio e desvio-padrão.
- F1-Score médio e desvio-padrão.
- Número médio de atributos selecionados.
- **Melhor experimento** com lista dos atributos selecionados.
- Tempo total de execução.

---

## Saídas geradas

### EDA (`reports/`)

Geradas por `src/eda.py`. Versionadas no repo para o relatório.

- `eda_summary.txt` — resumo textual com classes, inconsistências,
  estatísticas, colunas a remover
- `eda_class_distribution.png` — barras das 3 classes
- `eda_missingness.png` — top colunas com NaN
- `eda_age_by_class.png` — boxplot idade × classe
- `eda_temporal.png` — proporção de classes por ano
- `eda_correlation.png` — matriz de correlação numérica
- `eda_numeric_stats.csv` — `describe()` das numéricas
- `eda_categorical_stats.csv` — contagem de baixa cardinalidade

### Logs do GA (`logs/`, gitignored)

`ga_metrics.csv` — uma linha por (experimento, geração):

| Coluna                  | Conteúdo                                                       |
|-------------------------|----------------------------------------------------------------|
| `experimento`           | ID do experimento (0..N−1)                                     |
| `geracao`               | Número da geração                                              |
| `melhor_fitness`        | Maior fitness da população                                     |
| `fitness_medio`         | Fitness médio da população                                     |
| `pior_fitness`          | Menor fitness da população                                     |
| `melhor_f1`             | F1 weighted do melhor indivíduo                                |
| `num_atributos_ativos`  | Nº de genes = 1 no melhor cromossomo                           |
| `melhor_cromossomo`     | Máscara binária (string concatenada)                           |

`nn_metrics.csv` — uma linha por avaliação de cromossomo:

| Coluna                | Conteúdo                                                |
|-----------------------|---------------------------------------------------------|
| `id_cromossomo`       | Identificador único (`expX_genY_childZ`)                |
| `geracao`             | Geração da avaliação                                    |
| `loss_treino`         | Loss final no treino                                    |
| `loss_validacao`      | Loss final na validação                                 |
| `acuracia_validacao`  | Acurácia final na validação                             |
| `f1_score`            | **F1 weighted** no teste (fitness primário)             |
| `f1_macro`            | F1 macro no teste (diagnóstico de viés)                 |
| `epocas`              | Quantidade de épocas até parar (early stop ≤ 30)        |
| `num_atributos_usados`| Tamanho da entrada da MLP                               |

### Gráficos do GA (`plots/`, gitignored)

- **`ga_convergencia_media.png`** — média do melhor fitness por geração
  (entre N experimentos), banda de ± 1 σ. **Gráfico principal exigido pelo
  trabalho.**
- **`ga_convergencia_por_experimento.png`** — uma curva por experimento.
- **`ga_fitness_componentes.png`** — melhor/médio/pior do primeiro experimento.
- **`nn_atributos_vs_f1.png`** — dispersão F1 × atributos ativos.

---

## Decisões pragmáticas

Escolhas técnicas feitas além do que a especificação prescreve:

| Decisão | Justificativa |
|---------|---------------|
| Amostragem estratificada de 3000 registros (padrão) | Sem amostragem, cada treino da MLP custa ~5s; 20 × ~400 avaliações levaria > 11h. Com 3000 registros, cai para ~30 min. Use `--sample-size 0` para a base completa. |
| **F1 weighted como fitness primário** (e não macro) | Desbalanceamento real é 62/20/18. Macro penaliza excessivamente erros nas minoritárias; weighted reflete o desempenho esperado em produção. F1 macro é logado em paralelo para diagnóstico. |
| Remoção de 14 colunas de vazamento de alvo | Sem isso, F1 = 1,0 trivialmente e o GA não tem espaço de busca. |
| Remoção de colunas constantes (SEXO, TIPOBITO) | Variância zero → zero poder discriminativo. |
| Remoção de colunas dominadas por "ignorado" (9 > 80%) | EXAME e CIRURGIA têm 94–95% de "ignorado" — sem sinal. |
| LabelEncoder para categóricas em vez de One-Hot | Mantém o cromossomo curto (~28 genes) e fiel à ideia de "selecionar atributos". |
| Limite de 50% de NaN para descartar coluna | Compromisso entre perda de informação e imputação massiva. |
| Limite de 50 valores únicos para descartar categóricas | Acima disso, IDs ou nomes livres sem informação discriminativa. |
| Imputação por mediana (não média) | Robusta a outliers em `CODMUN*`, `OCUP` (códigos com escala estranha). |
| `EarlyStopping(patience=5)` na MLP | "Melhor configuração = menor erro de validação" (spec). Retorna os melhores pesos via `restore_best_weights`. |
| Cache de fitness por tupla de genes | Reduz 20–40% das avaliações na fase tardia da evolução. |
| Seed = `42 + exp_id` | Reprodutibilidade total; experimentos independentes mas determinísticos. |
| Saída Softmax + `sparse_categorical_crossentropy` | Evita overhead de one-hot encoding dos rótulos. |
| Torneio de seleção (em vez de roleta) | Pressão seletiva controlada, independente da escala do fitness. |

---

## Referências de material

- `docs/ga-mlp-task.pdf` — enunciado oficial do trabalho.
- `docs/ga-material/` — material de apoio sobre Algoritmos Genéticos.
- `docs/mlp-material/` — material de apoio sobre Redes Neurais.
- `reports/eda_summary.txt` — resumo completo da análise exploratória.
