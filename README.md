# Seleção de Atributos com Algoritmos Genéticos e Redes Neurais

Trabalho da disciplina de Inteligência Computacional, CEFET-RJ
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
7. [Algoritmo Genético, escolhas de projeto](#algoritmo-genético-escolhas-de-projeto)
8. [Rede Neural (MLP), escolhas de projeto](#rede-neural-mlp-escolhas-de-projeto)
9. [Função de aptidão e escalonamento](#função-de-aptidão-e-escalonamento)
10. [Métrica F1: por que weighted e não macro](#métrica-f1-por-que-weighted-e-não-macro)
11. [Sample size, ruído do F1 e reprodutibilidade](#sample-size-ruído-do-f1-e-reprodutibilidade)
12. [Procedimento experimental](#procedimento-experimental)
13. [Saídas geradas](#saídas-geradas)
14. [Decisões pragmáticas](#decisões-pragmáticas)

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
(scikit-learn, pandas, openpyxl, matplotlib, joblib).

---

## Execução

### Análise Exploratória (EDA)

Gera todos os relatórios e gráficos descritivos da base bruta:

```bash
uv run python src/eda.py
```

Saídas em `reports/` (commitado no repo para fins de relatório):
- `eda_summary.txt`, resumo textual completo
- `eda_class_distribution.png`, distribuição das 3 classes
- `eda_missingness.png`, % de NaN por coluna
- `eda_age_by_class.png`, boxplot de idade por classe
- `eda_temporal.png`, evolução temporal das classes
- `eda_correlation.png`, matriz de correlação numérica
- `eda_numeric_stats.csv`, describe() das numéricas
- `eda_categorical_stats.csv`, contagem das categóricas

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
| `--workers`         | `1`                                  | Nº de experimentos a rodar em paralelo (processos). Cada worker roda um experimento completo; logs são consolidados ao fim. Sugestões: M3 Pro Max 36 GB → 8; Ryzen 3600 16 GB → 4–6 |
| `--fitness-repeats` | `1`                                  | K retreinos da MLP por cromossomo. F1 do fitness = média das K execuções com seeds distintas. Reduz o ruído do estimador em √K ao custo de K× runtime. Ver seção *Sample size, ruído do F1 e reprodutibilidade* |
| `--quick`           | —                                    | Atalho: 2 experimentos × 20 gerações, pop 30 |

### Comando de produção sugerido

Para uma execução com baixa variância entre execuções e resultado estável para o relatório final:

```bash
uv run python src/main.py \
    --sample-size 6000 \
    --fitness-repeats 3 \
    --workers 4
```

Racional detalhado na seção *Sample size, ruído do F1 e reprodutibilidade*.

### Paralelismo

`--workers N` distribui os **experimentos** (não as gerações dentro de um experimento) entre N processos via `joblib.Parallel` com backend `loky`. Os 20 experimentos são totalmente independentes (sementes distintas, sem dependência de estado), então a paralelização é perfeita.

Cada worker:
1. Roda um experimento completo (200 gerações × 2 evals + 150 iniciais)
2. Escreve seu próprio CSV em `logs/_workers/exp{id}/`
3. Ao fim, o processo principal consolida tudo em `logs/ga_metrics.csv` e `logs/nn_metrics.csv`

Tempo total esperado para a spec (20 experimentos × 200 gerações × amostra de 3000):

| Hardware                 | `--workers 1` | `--workers 4` | `--workers 8` |
|--------------------------|--------------:|--------------:|--------------:|
| M3 Pro Max 36 GB         | ~10–15 min    | ~3–5 min      | **~2–3 min**  |
| Ryzen 3600 + 16 GB       | ~15–25 min    | **~4–7 min**  | (RAM limita)  |

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

#### Nota terminológica

"Câncer do colo do útero" (português) e "cervical cancer" (inglês) são
sinônimos perfeitos e correspondem **estritamente ao código CID-10 C53**.
"Colo do útero" = "cervix" = parte inferior do útero.

Apesar do título do trabalho ser "Câncer do Colo do Útero", a base
disponibilizada cobre **câncer uterino em sentido amplo**, incluindo:

- **C53** (colo do útero / cervical), 62%
- **C54** (corpo do útero / endometrial), 18%
- **C55** (útero, não especificado), 20%

C54 e C55 **não são** câncer cervical em sentido estrito. Por essa razão,
quando este relatório discute classes específicas, sempre usa o código
CID-10 (C53/C54/C55) em vez de termos como "câncer cervical", esse último
seria ambíguo num contexto multiclasse. O termo "câncer do colo do útero"
é mantido apenas como rótulo do trabalho, respeitando o título dado pelo
professor.

### Período coberto

Óbitos de **2010 a 2024** (15 anos). Volume crescente ao longo do tempo,
saindo de ~8.100 registros/ano em 2010 para ~12.000 em 2024. A proporção das
classes muda discretamente ao longo do tempo (C53 estável em ~62%, C54
crescendo de 12% para 22%, C55 caindo de 26% para 15%), sugerindo
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

### Distribuição geográfica por classe

Análise das colunas de residência (`res_MUNNOMEX`, `res_REGIAO`, `res_SIGLA_UF`,
`ocor_MUNNOMEX`) revelou **desbalanceamento sistemático da distribuição espacial**
entre as três classes — não é ruído de amostragem, é viés estrutural. Este
achado tem implicações diretas para o oversampling (ver mais abaixo).

**Residência em capital (baseline geral: 28,2%)**

| Classe   | n      | % em capital | Odds capital/interior      |
|----------|--------|--------------|----------------------------|
| C53      | 92.287 | 28,1%        | 0,39                       |
| **C54**  | 26.153 | **36,0%**    | **0,56** (+44% vs C53)     |
| **C55**  | 30.345 | **22,2%**    | **0,29** (−27% vs C53)     |

A diferença entre C54 (36,0%) e C55 (22,2%) é de **13,8 pontos percentuais** — muito
acima do que se esperaria por amostragem aleatória sob a hipótese nula de
independência entre classe e local de residência. C54 está **sobre-representado**
em capitais; C55, **sub-representado**. Para óbito em capital (`ocor_MUNNOMEX`),
o padrão se repete e é ainda mais acentuado: C54 = 52,5%, C53 = 49,2%, C55 = 31,5%.

**Distribuição regional (% dentro da classe)**

| Classe   | Sudeste   | Nordeste | Sul   | Norte | Centro-Oeste |
|----------|-----------|----------|-------|-------|--------------|
| C53      | 33,1%     | 31,3%    | 14,6% | 13,2% | 7,9%         |
| **C54**  | **53,2%** | 20,6%    | 15,8% | 4,0%  | 6,4%         |
| **C55**  | **47,5%** | 25,3%    | 15,8% | 5,6%  | 5,9%         |

Baseline populacional da base: Sudeste 39,6%, Nordeste 28,2%, Sul 15,0%,
Norte 10,0%, Centro-Oeste 7,2%.

C54 está **~34% mais concentrado no Sudeste** que a base geral; C55 também,
embora menos. C53 (majoritária) segue mais de perto a distribuição populacional
real, enquanto as minoritárias trazem viés espacial marcado. Top 5 UFs por
classe (residência): C53 lidera com SP=15,1% enquanto C54 tem SP=28,1% (quase
o dobro).

**Interpretação epidemiológica**

O padrão é consistente com **variação regional na precisão diagnóstica**, não
com epidemiologia diferencial: capitais e Sudeste concentram serviços de
patologia especializados, permitindo classificação precisa como **C54** (corpo
do útero — exige análise histopatológica). Interior e regiões com menos
infraestrutura tendem a receber diagnóstico genérico **C55** (útero não
especificado) — o "C55 alto no Nordeste/Norte" é sinal de **subclassificação
por falta de acesso**, não de incidência real distinta.

**Implicações para o pipeline (interação com oversampling)**

Este viés interage diretamente com o **oversampling das minoritárias**
(ver [subseção](#tratamento-do-desbalanceamento-de-classes)). Ao duplicar
com reposição as linhas de C54 (~2,8×) e C55 (~2,4×) no conjunto de treino,
amplificamos o sinal "residência em capital + Sudeste" nas amostras C54, e
o sinal "interior + Nordeste" nas amostras C55. A MLP pode aprender essa
correlação geográfica espúria como **atalho** para a classe, em vez do sinal
clínico real. Como as colunas de município (`CODMUNRES`, `CODMUNOCOR`) são
armazenadas como códigos IBGE inteiros no XLSX, elas atravessam o pipeline
como features **numéricas ordinais** — o filtro `MAX_CATEGORICAL_CARDINALITY=50`
não as remove.

**Mitigações consideradas mas não aplicadas** (para manter a base o mais
próxima possível da spec do trabalho):

- **Deixar o GA decidir**: as colunas `res_*` e `ocor_*` fazem parte do espaço
  de busca — se o modelo consegue F1 melhor sem elas, o GA tende a desativá-las
  ao longo das gerações. Proteção parcial: só funciona se o benefício médio for
  maior que o ruído de fold entre os 20 experimentos.
- **Diagnóstico auxiliar**: F1 macro é logado em paralelo ao F1 weighted
  (ver [seção](#métrica-f1-por-que-weighted-e-não-macro)). Se o modelo estivesse
  explorando o atalho geográfico de forma severa, esperaríamos F1 macro
  desproporcionalmente alto (recall inflado nas minorias via viés). Como
  observamos macro < weighted consistentemente, o efeito parece controlado.
- **Não aplicado**: SMOTE (interpolação em vez de duplicação, reduziria a
  amplificação mas ainda opera em códigos IBGE tratados como ordinais); remoção
  ativa das colunas de município antes do GA. Ambas mudariam a base em relação
  à spec e comprometeriam a comparabilidade.

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
| Registros com `label_cid` nulo            | **0**    | Removidos (`dropna(subset)`), proteção mesmo sem casos |
| Duplicatas exatas                         | **3**    | Removidas (`drop_duplicates`) |
| Idades fora de [0, 120]                   | **0**    |, |
| Idades nulas                              | **11**   | Imputadas pela mediana (56 anos) |
| Datas inconsistentes (nasc > óbito)       | **0**    |, |
| Anos fora de [1996, 2025]                 | **0**    |, |
| `SEXO ≠ 2` (não-feminino)                 | **0**    |, (esperado: câncer de útero) |
| Coluna `SEXO` constante                   | **Sim**  | **Removida**, variância zero |
| Coluna `TIPOBITO` constante               | **Sim**  | **Removida**, variância zero (sempre 2 = óbito não-fetal) |
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
| Desbalanceamento severo das classes (62/20/18, razão 3,53×) | **Sim** | **Oversampling das minoritárias** no conjunto de treino (ver [subseção](#tratamento-do-desbalanceamento-de-classes)) |

### Sobre o código 9 ("ignorado") do DATASUS

No padrão SIM/DATASUS, o valor `9` em campos categóricos representa
*"ignorado"*, informação ausente, não uma categoria real. Tratá-lo como
categoria normal mistura ausência de informação com categorias válidas e
introduz ruído. O pipeline aplica duas estratégias:

1. **Descarte por dominância (etapa 7)**: para colunas em
   `IGNORADO_CODE_COLUMNS` (RACACOR, ESTCIV, ESC, ASSISTMED, EXAME, CIRURGIA,
   NECROPSIA, etc.) onde **mais de 80%** dos valores são 9, a coluna é
   descartada inteira, não há sinal útil. Aplicado a `EXAME` (94%) e
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
de "variância zero" e "alto NaN", por exemplo, colunas com mistura de NaN
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
todos no intervalo [11–13] ou [111–114], valores plausíveis para casos
médicos extremos. Outliers detectados em colunas como `CODMUNRES`,
`CODESTAB`, `OCUP` não são outliers reais: são **códigos categóricos
disfarçados de números** (códigos de município, IDs de estabelecimento,
códigos especiais de ocupação). Tratá-los com IQR distorceria o significado
sem ganho informacional. A normalização Min-Max já comprime tudo para [0,1].

---

## Pipeline de limpeza

Todos os tratamentos de dados aplicados, do arquivo bruto até a entrada da MLP,
estão organizados em **dois estágios**:

- **Estágio A — Pré-processamento global** (`src/data_loader.py`): roda **uma vez** por experimento, antes do GA. 17 etapas.
- **Estágio B — Tratamento por avaliação** (`src/nn/trainer.py`): roda **a cada avaliação de cromossomo** (treino de MLP). 2 etapas.

### Estágio A — Pré-processamento global (`src/data_loader.py`)

Determinístico, executado na ordem abaixo:

| #  | Etapa                                                    | O que faz                                                   | Heurística / configuração                                |
|----|----------------------------------------------------------|-------------------------------------------------------------|----------------------------------------------------------|
| 1  | **Leitura do arquivo**                                   | Carrega XLSX (via openpyxl) ou CSV (via pandas)             | Extensão do arquivo                                      |
| 2  | **Drop de registros sem alvo**                           | Remove linhas com `label_cid` nulo                          | `dropna(subset=["label_cid"])`                           |
| 3  | **Drop de duplicatas exatas**                            | Remove linhas idênticas em todas as colunas                 | `drop_duplicates()` — detecta 3 na base                  |
| 4  | **Auditoria de idade** (não remove)                      | Relata idades fora de [0, 120]                              | `_check_age_outliers` — detecta 0 inválidas, 11 nulas    |
| 5  | **Auditoria de datas** (não remove)                      | Relata registros com `DTNASC > DTOBITO`                     | `_check_date_consistency` — detecta 0 inconsistências    |
| 6  | **Drop de colunas de vazamento de alvo**                 | Remove 14 colunas que codificam diretamente a causa do óbito | Lista explícita em `TARGET_LEAK_COLUMNS`                 |
| 7  | **Drop de colunas constantes**                           | Remove colunas com `nunique ≤ 1` (variância zero)           | Detecta `TIPOBITO`, `SEXO`, `CAUSAMAT`, `NUDIASINF`      |
| 8  | **Drop de colunas dominadas por "9" (ignorado DATASUS)** | Remove colunas em `IGNORADO_CODE_COLUMNS` com > 80% de 9s   | `MAX_IGNORADO_RATIO = 0.8` — descarta `EXAME`, `CIRURGIA` |
| 9  | **Drop de colunas quase-constantes**                     | Remove colunas onde a moda concentra > 95% dos valores      | `MAX_MODE_RATIO = 0.95` — descarta `ESTABDESCR`, `COMUNSVOIM`, `ALTCAUSA` |
| 10 | **Conversão de "9" residual → NaN**                      | Em colunas DATASUS restantes, transforma 9 em NaN           | Para que a imputação (etapa 14) trate "ignorado" como ausente |
| 11 | **Drop de colunas com excesso de NaN**                   | Remove colunas com > 50% de valores nulos                   | `MAX_MISSING_RATIO = 0.5` — descarta `SERIESCFAL`, `NUDIASOBCO` |
| 12 | **Drop de colunas numéricas redundantes**                | Remove uma coluna de cada par com `|corr| > 0,95`           | `MAX_CORRELATION = 0.95` — descarta `CODMUNOCOR`, `ocor_CODIGO_UF` |
| 13 | **Codificação de variáveis categóricas**                 | Aplica `LabelEncoder` em colunas object/string              | Se `nunique > 50` → descarta (alta cardinalidade)        |
| 14 | **Imputação de NaN numéricos**                           | Preenche valores ausentes com a **mediana** da coluna       | Robusto a outliers em códigos (`OCUP`, `CODMUN*`)         |
| 15 | **Codificação do target (`label_cid`)**                  | Mapeia `{C53, C54, C55}` → `{0, 1, 2}` via `LabelEncoder`   | Inteiros consumidos pela softmax da MLP                  |
| 16 | **Amostragem estratificada** (opcional)                  | Reduz a base mantendo a proporção 62/20/18 das classes      | `DEFAULT_SAMPLE_SIZE = 3000` (use `--sample-size 0` para a base inteira) |
| 17 | **Normalização Min-Max**                                 | `x' = (x − x_min) / (x_max − x_min)` em [0, 1]              | Exigido pela especificação (seção 3 do enunciado)        |

**Resultado**: de **50 colunas brutas** + ~149k registros → **26 atributos** + amostra estratificada de 3.000 registros (na configuração padrão), prontos para a MLP.

### Estágio B — Tratamento por avaliação (`src/nn/trainer.py`)

Executado **uma vez para cada cromossomo avaliado** pelo GA (com os atributos filtrados pela máscara do cromossomo):

| #  | Etapa                                       | O que faz                                                                | Heurística / configuração                                  |
|----|---------------------------------------------|--------------------------------------------------------------------------|------------------------------------------------------------|
| 18 | **Divisão estratificada 70/15/15**          | Splita em treino / validação / teste preservando proporção das classes   | `_split_70_15_15`, `train_test_split(stratify=y)` em cascata |
| 19 | **Oversampling das classes minoritárias**   | Reamostra com reposição C54 e C55 para igualar o tamanho de C53          | `_oversample_minority_classes`, **apenas no conjunto de treino**; val e teste preservam a distribuição original |

Após o Estágio B, o conjunto de treino balanceado vai para o `MLPClassifier`.
Validação e teste **nunca são oversampleados** — preservam a distribuição
populacional real (62/20/18) para que as métricas reflitam o cenário de produção.

> **Mapa rápido**: para detalhes sobre cada etapa, ver:
> - Vazamento de alvo: [seção "Inconsistências detectadas"](#inconsistências-detectadas)
> - Tratamento de "9" (DATASUS): [subseção dedicada](#sobre-o-código-9-ignorado-do-datasus)
> - Quase-constância: [subseção dedicada](#quase-constância-moda--95)
> - Correlação alta: [subseção dedicada](#correlação-alta-r--095)
> - Divisão 70/15/15: [seção dedicada na parte da MLP](#divisão-dos-dados-701515-estratificada)
> - Oversampling: [subseção "Tratamento do desbalanceamento de classes"](#tratamento-do-desbalanceamento-de-classes)

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

## Algoritmo Genético, escolhas de projeto

### Representação dos cromossomos (`src/ga/chromosome.py`)

**Codificação binária**: cada cromossomo é uma lista de comprimento L (= nº
total de atributos), onde:

- `gene = 1` → atributo **selecionado**
- `gene = 0` → atributo **descartado**

**Inicialização aleatória**: cada gene é 0 ou 1 com p=0,5, independente. Um
*failsafe* garante ≥ 1 gene ativo (sorteio de cromossomo todo-zeros é
desprezível mas possível).

**Atributos armazenados em cada cromossomo:**

- `genes`, lista de 0s e 1s
- `fitness`, valor final da função de aptidão
- `f1_score`, F1 weighted bruto (componente principal do fitness)
- `scaled_fitness`, fitness após normalização linear da população
- `key()`, tupla imutável, usada como chave do cache

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
por cromossomo**, valor consagrado para AGs binários (Bäck, 1996). Com L≈28,
`Pm ≈ 0,036`.

### Seleção: Torneio de tamanho 3

1. Sorteia-se 3 indivíduos da população.
2. Vencedor = maior `scaled_fitness`.

Optou-se por torneio em vez de roleta porque:
(a) pressão seletiva controlada independentemente da escala dos fitness, e
(b) integra naturalmente com a normalização linear exigida.

### Estratégia evolutiva, Steady-State, Gap = 2

Cada **geração** do algoritmo:

1. Preserva os **10 melhores** indivíduos (elitismo).
2. Seleciona **2 pais** via torneio de 3 (usando `scaled_fitness`).
3. Aplica **Crossover Uniforme** com `Pc = 0,85` → 2 filhos.
4. Aplica **mutação bit-flip** nos filhos com `Pm = 1/L`.
5. Avalia os 2 filhos (treina MLP em cada).
6. Substitui os 2 piores indivíduos da população pelos 2 filhos.
7. Recomputa `scaled_fitness` para a nova população.

Os 138 indivíduos intermediários permanecem inalterados de uma geração
para a próxima — apenas o "fundo" da população é renovado.

Ambos `Pc = 0,85` e `Pm = 1/L` são aplicados no loop conforme a especificação.

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

## Rede Neural (MLP), escolhas de projeto

### Arquitetura (`src/nn/model.py`)

| Camada    | Neurônios          | Ativação | Observação                                 |
|-----------|--------------------|----------|--------------------------------------------|
| Entrada   | `input_dim`        |,        | igual ao nº de genes ativos no cromossomo  |
| Oculta 1  | **32**             | ReLU     | conforme especificação                     |
| Oculta 2  | **16**             | ReLU     | conforme especificação                     |
| Saída     | `n_classes` (= 3)  | Softmax  | um neurônio por classe                     |

A entrada **se adapta** ao cromossomo: cada avaliação instancia um modelo
novo com `input_dim` igual aos atributos selecionados. É isso que permite
ao GA comparar configurações com tamanhos de entrada diferentes.

Sem `Dropout` ou `BatchNormalization`, não foram pedidos e o early stopping
já cobre regularização.

### Treinamento (`src/nn/trainer.py`)

| Aspecto             | Valor                                                                                       | Justificativa |
|---------------------|---------------------------------------------------------------------------------------------|---------------|
| Framework           | **scikit-learn** `MLPClassifier`                                                            | spec exige arquitetura/otimizador/ativações, não framework — sklearn é ~78× mais rápido que Keras para MLPs deste porte (medido em profile) |
| Otimizador          | Adam (lr=0,001)                                                                             | exigido pela especificação |
| Loss                | Categorical crossentropy (default do sklearn para multiclasse)                              | equivalente ao `sparse_categorical_crossentropy` do Keras |
| Batch size          | 64                                                                                          | compromisso vetorização × ruído de gradiente |
| Épocas (máx.)       | 30                                                                                          | `max_iter=30` (early stopping geralmente para antes) |
| Early stopping      | `early_stopping=True`, `n_iter_no_change=5`, fração de validação interna ≈ 15/85            | "melhor configuração = menor erro de validação" (spec) |
| L2 regularization   | `alpha=0.0` (desabilitado)                                                                  | spec não exige; mantém comportamento equivalente ao Keras (Keras tampouco regulariza por default) |
| **Desbalanceamento (treino)** | **Oversampling das minorias** para igualar a majoritária         | spec não exige, mas necessário porque `MLPClassifier` não aceita `class_weight` (ver subseção) |
| **Métrica de aptidão** | **F1-Score weighted** no teste                                                           | reflete desbalanceamento real (ver seção dedicada) |
| Métrica auxiliar    | F1-Score macro                                                                              | logado para diagnóstico de viés por classe |

#### Tratamento do desbalanceamento de classes

A base tem distribuição **62% C53 / 20% C55 / 18% C54** (razão maior/menor = 3,53×).
Sem tratamento, a MLP aprende a "chutar C53" para qualquer entrada — F1 macro
cai drasticamente porque as minorias C54 e C55 nunca são preditas.

**Limitação técnica do sklearn**: o `MLPClassifier` **não aceita** `class_weight`
nem `sample_weight` (limitação conhecida da API). Para compensar, aplicamos
**oversampling com reposição** nas classes minoritárias do conjunto de treino:

```python
# Antes do treino:
# C53: 730 amostras (majoritária)
# C54: 207 amostras → reamostradas para 730 (replicação aleatória)
# C55: 240 amostras → reamostradas para 730
# Total de treino: 730 × 3 = 2.190 amostras balanceadas
```

Implementado via `sklearn.utils.resample` em `trainer._oversample_minority_classes`.
Aplicado **apenas ao conjunto de treino** — validação e teste preservam a
distribuição original 62/20/18 para que as métricas reflitam o cenário real
de produção. Efeito empírico (--quick): F1 weighted subiu de ~0,48 (sem
oversampling) para ~0,60 (com).

**Ressalva: amplificação de viés geográfico.** Random oversampling com reposição
**duplica linhas existentes** em vez de sintetizar exemplos novos. Isso amplifica
qualquer viés estrutural já presente nas minorias. A EDA
([subseção](#distribuição-geográfica-por-classe)) mostrou que C54 e C55 têm
distribuições geográficas sistematicamente diferentes de C53 (C54 sobre-representa
capitais e Sudeste; C55 sub-representa capitais). Ao replicar 2–3× essas linhas
no treino, reforçamos "residência em capital + Sudeste → C54" e
"interior + Nordeste → C55" como sinais espúrios. Aceito como trade-off conhecido;
diagnóstico via F1 macro em paralelo (não observamos macro > weighted, o que
seria indicativo de exploração severa do atalho).

**Alternativas consideradas e descartadas:**
- `class_weight` no sklearn MLP: não suportado.
- SMOTE (sintético): adiciona dependência (`imbalanced-learn`) e a vantagem
  vs. oversampling simples é marginal para nossa cardinalidade baixa de features.
- Undersampling da majoritária: descartaria ~70% dos dados de C53 (perde sinal).
- Trocar de framework (PyTorch): regrediria o speedup de 78× do sklearn.

#### Por que sklearn em vez de Keras

A spec define arquitetura, otimizador, taxa de aprendizado, ativações e
procedimento de validação — todos atendíveis tanto por `tf.keras.Sequential`
quanto por `sklearn.neural_network.MLPClassifier`. Profile no `--quick` mostrou
que ~96% do tempo do Keras estava em `model.fit()` por causa de overheads de
framework (graph compilation, `tf.function` retracing a cada novo `input_dim`,
kernel launches CUDA mesmo em CPU). A MLP tem apenas ~1.400 pesos — o
**compute real é nanossegundos**; tudo o resto é overhead.

Speedup medido por avaliação: **78,15×** (1,59s no Keras vs. 0,02s no sklearn).
Resultado prático: spec completa (20 exp × 200 gen × 140 evals) cai de
~1–2 dias para ~30 min em CPU serial. Sem custo de qualidade — sklearn
implementa MLP padrão com backprop+Adam, exatamente como exigido.

### Divisão dos dados, 70/15/15 estratificada

A divisão acontece **dentro da função de fitness** (`trainer._split_70_15_15`),
ou seja, **a cada avaliação de cromossomo**. Não é um split único feito no
início, é refeito a cada chamada de `train_and_evaluate_nn`.

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
| Teste       | 15%  | Mede F1-Score reportado como fitness, **uma única vez por cromossomo** |

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
  variabilidade do split, só capturaria ruído da inicialização aleatória
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
  mas só 30% em C54 e C55 fica com macro F1 ≈ 0,52, fortemente penalizado.
- **F1 weighted** pondera pelo suporte real. O mesmo modelo fica com weighted F1
  ≈ 0,76, refletindo o desempenho esperado em produção (onde 62% dos casos
  reais são C53).
- **F1 micro** (= accuracy em multiclasse) sempre favorece a classe majoritária.

**Por que weighted como fitness primário:**

1. **Realismo populacional.** Se o sistema for usado para classificar novos
   óbitos, a distribuição esperada é ~62/20/18, não 33/33/33. O weighted
   reflete a métrica relevante.
2. **Decisão consciente do trade-off.** Não estamos cegos ao desbalanceamento
  , estamos **ponderando deliberadamente** pelo suporte real, em vez de
   fingir que as classes são igualmente prevalentes.
3. **Métrica macro ainda é logada** para diagnóstico. Em `nn_metrics.csv`,
   a coluna `f1_macro` permite verificar se o modelo está ignorando
   classes minoritárias. Se `f1_macro` ficar muito abaixo de `f1_weighted`,
   indica viés para C53.

**Alternativa considerada e descartada:** transformar o problema em binário
(C53 vs. não-C53), alinharia com o título do trabalho ("câncer **do colo**
do útero"), mas a especificação pede "número de neurônios igual ao número
de classes da base", que são 3. Manter as 3 classes está correto.

---

## Sample size, ruído do F1 e reprodutibilidade

Uma questão prática: por que os F1 obtidos pelo AG oscilam entre execuções
e por que 0,60 nunca aparece de forma consistente? A resposta tem três
componentes.

### Teto do problema (empírico)

Treinar a MLP com **todos os 57 atributos** (sem seleção do AG) sobre
`--sample-size 3000` produz:

| baseline                             | F1-weighted (5 seeds) |
|--------------------------------------|-----------------------|
| MLP em 57 atributos (todos)          | **0,53 ± 0,01**       |
| MLP em 28 atributos aleatórios       | 0,49 ± 0,02           |

Aumentar o sample não move esse teto significativamente:

| `--sample-size` | F1_all_features |
|-----------------|-----------------|
| 3 000           | 0,531           |
| 6 000           | 0,528           |
| 10 000          | 0,518           |
| 20 000          | 0,533           |

Isso é o **ceiling intrínseco** da tarefa (distinguir C53/C54/C55 a partir
dos atributos que sobram após remover o vazamento). O AG rotineiramente
encontra subconjuntos com F1 = 0,55–0,57, **superando** o baseline — a
seleção está ajudando.

### Por que "0,60+" apareceu antes

O que era relatado como "0,60" era o **fitness** (0,9 · F1 + 0,1 ·
parcimônia), não o F1 puro. Um cromossomo com 3 atributos ativos e F1 =
0,573 produz fitness = 0,9 · 0,573 + 0,1 · (1 − 3/23) ≈ **0,599** — o
número mágico "0,60" nunca correspondeu a F1 real.

### Piso de ruído do F1

Com split 15% de teste sobre `sample_size = 3000` (450 linhas de teste,
das quais ~79 pertencem à classe minoritária C54), o erro-padrão do F1
ponderado é aproximadamente:

$$\mathrm{SE}(F_1^{w}) \approx \sqrt{\sum_c w_c^2 \cdot \frac{F_1^c (1 - F_1^c)}{n_c}}$$

Numericamente (com F1 por classe ≈ 0,6 / 0,5 / 0,4):

| `sample_size` | tamanho do teste | mín. classe C54 | SE(F1_weighted) |
|--------------:|-----------------:|----------------:|----------------:|
|         3 000 |              450 |              79 |         ≈ 0,017 |
|         6 000 |              900 |             158 |         ≈ 0,012 |
|        10 000 |            1 500 |             264 |         ≈ 0,009 |
|        20 000 |            3 000 |             528 |         ≈ 0,007 |

O AG está tentando distinguir cromossomos cujos F1 reais diferem em
0,01–0,05. Com ruído de 0,017 por avaliação, grande parte da busca
acontece **dentro** do piso de ruído — ordenamentos são pouco confiáveis.

### Regra prática para escolher `sample_size`

Para ter pelo menos 150–200 exemplos da classe minoritária no conjunto de
teste (bom para estabilizar o F1 minoritário, que puxa o ruído do
weighted):

$$\text{sample\_size} \gtrsim \frac{200}{0{,}15 \cdot \pi_{\min}}$$

Para esta base ($\pi_{\min} = 17{,}6\%$): `sample_size ≥ 7 600`. Na prática,
6 000 já entrega SE ~ 0,012 — bom trade-off runtime × variância.

### Fitness com múltiplas seeds (`--fitness-repeats K`)

Alternativa/complemento à `sample_size`: cada cromossomo é avaliado K
vezes com seeds distintas (splits 70/15/15 diferentes, inicialização de
pesos diferente) e o fitness usa a **média** dos K F1. O erro-padrão do
estimador cai em √K, o que é equivalente estatisticamente a aumentar o
sample em K vezes (do ponto de vista do estimador), mas mantém a
diversidade dos splits.

- `K=1` (default): rápido, alto ruído
- `K=3`: SE / √3 ≈ 0,58×, runtime 3×
- `K=5`: SE / √5 ≈ 0,45×, runtime 5×

Refs: Bengio & Grandvalet (2004) "No Unbiased Estimator of the Variance
of K-Fold Cross-Validation", JMLR 5:1089-1105; Nadeau & Bengio (2003)
"Inference for the Generalization Error", Machine Learning 52(3):239-281.

### Reprodutibilidade entre execuções

Todos os seeds são derivados de `42 + exp_id` (numpy, random, MLP,
splits, oversampling). Duas execuções do mesmo comando devem produzir
resultados bit-idênticos por experimento. Se você vê variação
run-to-run com os mesmos flags, é sintoma de não-determinismo (BLAS
multi-thread, ordem de agregação em `joblib`, etc.) — abra uma issue.

O que varia legitimamente é o **resultado entre os 20 experimentos**:
cada um usa uma seed diferente, é isso que dá a curva média que o spec
pede.

---

## Procedimento experimental

### 20 experimentos independentes

Loop em `src/main.py`:

1. **Seed reset**, semente fixada para `numpy` e `random` em
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
- A curva final é a **média dos melhores em 20 experimentos**, exatamente
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

- `eda_summary.txt`, resumo textual com classes, inconsistências,
  estatísticas, colunas a remover
- `eda_class_distribution.png`, barras das 3 classes
- `eda_missingness.png`, top colunas com NaN
- `eda_age_by_class.png`, boxplot idade × classe
- `eda_temporal.png`, proporção de classes por ano
- `eda_correlation.png`, matriz de correlação numérica
- `eda_numeric_stats.csv`, `describe()` das numéricas
- `eda_categorical_stats.csv`, contagem de baixa cardinalidade

### Logs do GA (`logs/`, gitignored)

`ga_metrics.csv`, uma linha por (experimento, geração):

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

`nn_metrics.csv`, uma linha por avaliação de cromossomo:

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

- **`ga_convergencia_media.png`**, média do melhor fitness por geração
  (entre N experimentos), banda de ± 1 σ. **Gráfico principal exigido pelo
  trabalho.**
- **`ga_convergencia_por_experimento.png`**, uma curva por experimento.
- **`ga_fitness_componentes.png`**, melhor/médio/pior do primeiro experimento.
- **`nn_atributos_vs_f1.png`**, dispersão F1 × atributos ativos.

---

## Decisões pragmáticas

Escolhas técnicas feitas além do que a especificação prescreve:

| Decisão | Justificativa |
|---------|---------------|
| Amostragem estratificada de 3000 registros (padrão) | Sem amostragem, cada treino da MLP custa ~5s; 20 × ~400 avaliações levaria > 11h. Com 3000 registros, cai para ~30 min. Use `--sample-size 0` para a base completa. |
| **F1 weighted como fitness primário** (e não macro) | Desbalanceamento real é 62/20/18. Macro penaliza excessivamente erros nas minoritárias; weighted reflete o desempenho esperado em produção. F1 macro é logado em paralelo para diagnóstico. |
| Remoção de 14 colunas de vazamento de alvo | Sem isso, F1 = 1,0 trivialmente e o GA não tem espaço de busca. |
| Remoção de colunas constantes (SEXO, TIPOBITO) | Variância zero → zero poder discriminativo. |
| Remoção de colunas dominadas por "ignorado" (9 > 80%) | EXAME e CIRURGIA têm 94–95% de "ignorado", sem sinal. |
| LabelEncoder para categóricas em vez de One-Hot | Mantém o cromossomo curto (~26 genes) e fiel à ideia de "selecionar atributos". |
| Limite de 50% de NaN para descartar coluna | Compromisso entre perda de informação e imputação massiva. |
| Limite de 50 valores únicos para descartar categóricas | Acima disso, IDs ou nomes livres sem informação discriminativa. |
| Imputação por mediana (não média) | Robusta a outliers em `CODMUN*`, `OCUP` (códigos com escala estranha). |
| `EarlyStopping(patience=5)` na MLP | "Melhor configuração = menor erro de validação" (spec). Retorna os melhores pesos via `restore_best_weights`. |
| Cache de fitness por tupla de genes | Reduz 20–40% das avaliações na fase tardia da evolução. |
| Seed = `42 + exp_id` | Reprodutibilidade total; experimentos independentes mas determinísticos. |
| sklearn `MLPClassifier` em vez de Keras | 78× mais rápido para MLP de ~1.400 pesos; spec preservada integralmente. |
| Torneio de seleção (em vez de roleta) | Pressão seletiva controlada, independente da escala do fitness. |
| Oversampling das minorias no conjunto de treino | Compensa o desbalanceamento 62/20/18 sem alterar val/teste. Necessário porque `MLPClassifier` do sklearn não suporta `class_weight` nem `sample_weight`. Subiu F1 weighted de ~0,48 → ~0,60. |
| Conversão de "9" residual para NaN (após etapa 8) | Trata o código DATASUS "ignorado" como ausente, não como categoria real. ~155 mil substituições em 8 colunas. |
| Detecção de quase-constância (moda > 95%) | Captura colunas que escapam dos filtros de variância zero e excesso de NaN (ex.: 100% NaN + 1 valor real). |
| Detecção de correlação alta (\|r\| > 0,95) entre numéricas | Remove redundância de pares clones (`CODMUNRES` ↔ `CODMUNOCOR` ↔ `ocor_CODIGO_UF`). |

---

## Referências de material

- `docs/ga-mlp-task.pdf`, enunciado oficial do trabalho.
- `docs/ga-material/`, material de apoio sobre Algoritmos Genéticos.
- `docs/mlp-material/`, material de apoio sobre Redes Neurais.
- `reports/eda_summary.txt`, resumo completo da análise exploratória.
