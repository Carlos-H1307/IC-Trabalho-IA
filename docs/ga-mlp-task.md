# Trabalho de Algoritmo Genético e Redes Neurais

**CEFET-RJ** — Rio, 4/06/2026
**Prof.:** Laercio Brito

## Seleção de Atributos em Base de Dados de Câncer do Colo do Útero Utilizando Algoritmos Genéticos e Redes Neurais

---

## 1. Objetivo

O objetivo deste trabalho é desenvolver um sistema de seleção automática de atributos para uma base de dados relacionada ao câncer do colo do útero, utilizando Algoritmos Genéticos (AG) como mecanismo de busca e Redes Neurais Artificiais (RNA) treinadas pelo algoritmo Backpropagation como função de avaliação das soluções geradas.

O foco do trabalho é investigar o impacto da redução de dimensionalidade sobre o desempenho de classificação da rede neural, identificando subconjuntos de atributos que maximizem a capacidade preditiva do modelo.

---

## 2. Base de Dados

Será utilizada uma base de dados relacionada ao câncer do colo do útero. A base de dados se encontra na plataforma Teams em uma versão completa e outra pré-reduzida.

---

## 3. Pré-processamento dos Dados

Antes da execução do algoritmo genético deverão ser realizadas as seguintes etapas:

1. Tratamento de valores ausentes;
2. Remoção de registros inconsistentes;
3. Conversão de atributos categóricos para formato numérico;
4. Análise exploratória dos dados;
5. Aplicação da normalização linear Min-Max.

A normalização deverá ser realizada conforme:

$$
x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$

onde:

- $x$ = valor original;
- $x_{\min}$ = menor valor do atributo;
- $x_{\max}$ = maior valor do atributo.

Após a normalização, todos os atributos deverão estar no intervalo $[0, 1]$.

---

## 4. Representação do Cromossomo

Cada cromossomo representará um subconjunto de atributos da base de dados.

Será utilizada **codificação binária**, onde:

- Gene = 1 → atributo selecionado;
- Gene = 0 → atributo não selecionado.

**Exemplo:**

| A1 | A2 | A3 | A4 | A5 | A6 |
|----|----|----|----|----|----|
| 1  | 0  | 1  | 1  | 0  | 1  |

Nesse exemplo, apenas os atributos A1, A3, A4 e A6 serão utilizados pela rede neural.

O comprimento do cromossomo será igual ao número total de atributos da base de dados.

---

## 5. Configuração do Algoritmo Genético

- **Tamanho da população:** 150 cromossomos.
- Será utilizado o operador de **Crossover Uniforme**.
- **Probabilidade de crossover (Pc):** 0,85.
- **Probabilidade de mutação:** $P_m = 1 / L$, onde $L$ representa o comprimento do cromossomo.
- **Elitismo:** os 10 melhores indivíduos deverão ser preservados integralmente a cada geração.
- Será utilizada a abordagem **Steady-State Genetic Algorithm. Gap = 2**.
- **Critérios de parada:**
  - 200 gerações completas; ou
  - 20 gerações consecutivas sem melhoria da melhor solução.

---

## 6. Função de Aptidão

A aptidão de cada cromossomo será determinada pelo desempenho de uma Rede Neural Artificial treinada utilizando exclusivamente os atributos selecionados pelo cromossomo.

A medida principal de desempenho será o **F1-Score**.

A função de aptidão será definida por:

$$
\text{Fitness} = 0{,}9 \times \text{F1-Score} + 0{,}1 \times \left(1 - \frac{N_s}{N_t}\right)
$$

onde:

- $N_s$ = número de atributos selecionados;
- $N_t$ = número total de atributos.

Dessa forma, busca-se simultaneamente:

- Maximizar a capacidade preditiva;
- Minimizar o número de atributos utilizados.

Após o cálculo da aptidão, deverá ser aplicada a **normalização linear dos valores de fitness** para escalonamento da população.

---

## 7. Configuração da Rede Neural

A Rede Neural será utilizada como avaliadora dos indivíduos gerados pelo algoritmo genético.

### 7.1 Arquitetura

**Camada de Entrada**

- Número de neurônios igual ao número de atributos selecionados.

**Primeira Camada Oculta**

- 32 neurônios;
- Função de ativação **ReLU**.

**Segunda Camada Oculta**

- 16 neurônios;
- Função de ativação **ReLU**.

**Camada de Saída**

- Quantidade de neurônios igual ao número de classes da base de dados;
- Função de ativação: **Softmax**.

A saída representará a distribuição de probabilidade entre as classes.

#### Observações sobre as funções de ativação

**ReLU (Rectified Linear Unit)** é uma função que zera valores negativos e mantém valores positivos:

$$
f(x) = \max(0, x)
$$

**Softmax** transforma um vetor de valores em probabilidades que somam 1:

$$
\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}
$$

Se a rede gera:

$$
z = [2{,}0,\ 1{,}0,\ 0{,}1]
$$

Depois do Softmax:

- Classe 1: 0,66
- Classe 2: 0,24
- Classe 3: 0,10

### 7.2 Treinamento

**Algoritmo:**

- Backpropagation.

**Otimizador:**

- Adam.

O **Adam** é um algoritmo de otimização usado para treinar redes neurais. Ele é, basicamente, uma versão "inteligente" do **gradiente descendente**, com melhorias para tornar o aprendizado mais rápido e estável.

**Adam (Adaptive Moment Estimation)** é um otimizador que combina duas ideias:

- **Momento (Momentum):** lembra a direção dos gradientes anteriores.
- **Taxa de aprendizado adaptativa:** ajusta o passo automaticamente para cada peso.

**Taxa de aprendizado:**

- 0,001.
- A melhor configuração da rede neural será a que der **menor erro no conjunto de validação**.

---

## 8. Procedimento Experimental

Os dados deverão ser divididos em:

- **70%** para treinamento;
- **15%** para validação;
- **15%** para teste.

Para cada cromossomo avaliado:

1. Selecionar os atributos indicados pelos genes ativos;
2. Construir a rede neural correspondente;
3. Treinar a rede neural;
4. Calcular o F1-Score;
5. Calcular o fitness;
6. Atualizar a população do algoritmo genético.

Ao final do processo deverão ser apresentados:

- Melhor cromossomo encontrado;
- Número de atributos selecionados;
- Curva de convergência do algoritmo genético (média dos melhores em **20 experimentos completos**).

---

## Observações

Cada grupo deverá entregar:

1. Código-fonte completo;
2. Relatório técnico;
3. Descrição dos experimentos realizados;
4. Tabela contendo os parâmetros utilizados;
5. Gráficos de evolução do algoritmo genético;
6. Análise crítica dos resultados.
