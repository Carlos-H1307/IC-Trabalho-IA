import csv
import json
import os
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

from nn.trainer import _split_70_15_15, _oversample_minority_classes
from nn.model import create_mlp_model


# ---------------------------------------------------------------------------
# Metadados estáticos
# ---------------------------------------------------------------------------

def save_feature_names(feature_names, path):
    """
    Salva o mapeamento índice -> nome de atributo. Essencial para
    decodificar as strings binárias de cromossomos (`melhor_cromossomo`
    no ga_metrics.csv).
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["indice", "atributo"])
        for i, name in enumerate(feature_names):
            writer.writerow([i, name])


def save_hyperparameters(args, mutation_rate, L, constants, path):
    """
    Snapshot dos hiperparâmetros efetivamente usados no experimento.

    `constants` é um dict com os valores fixos definidos em `main.py`
    (CROSSOVER_RATE, ELITE_SIZE, GAP, STAGNATION_LIMIT, TOURNAMENT_SIZE).
    Passar explicitamente evita `from main import ...` — que falha porque
    `main.py` é carregado como `__main__` e não como módulo importável.
    """
    payload = {
        "data_path": args.data_path,
        "sample_size": args.sample_size if args.sample_size > 0 else None,
        "population_size": args.population,
        "max_generations": args.generations,
        "n_experiments": args.experiments,
        "workers": args.workers,
        "crossover_rate": constants["CROSSOVER_RATE"],
        "mutation_rate": mutation_rate,
        "mutation_rate_formula": "1/L",
        "chromosome_length_L": L,
        "elite_size": constants["ELITE_SIZE"],
        "steady_state_gap": constants["GAP"],
        "tournament_size": constants["TOURNAMENT_SIZE"],
        "stagnation_limit": constants["STAGNATION_LIMIT"],
        "f1_weight": 0.9,
        "parsimony_weight": 0.1,
        "mlp_hidden_layers": [32, 16],
        "mlp_activation": "relu",
        "mlp_output_activation": "softmax",
        "mlp_optimizer": "adam",
        "mlp_learning_rate": 0.001,
        "mlp_early_stopping": True,
        "mlp_patience": 15,
        "mlp_max_epochs": 200,
        "mlp_batch_size": 64,
        "split_train_val_test": [0.70, 0.15, 0.15],
        "oversample_minority": True,
        "fitness_repeats": getattr(args, "fitness_repeats", 1),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_dataset_info(X, y, feature_names, n_classes, class_names, path):
    """Descreve a base após pré-processamento (dimensões, classes, distribuição)."""
    class_counts = np.bincount(y).tolist()
    payload = {
        "n_samples": int(X.shape[0]),
        "n_features_L": int(X.shape[1]),
        "n_classes": int(n_classes),
        "classes": class_names,
        "class_distribution": {
            str(class_names[c]): {
                "n": int(class_counts[c]),
                "pct": round(class_counts[c] / len(y) * 100, 2),
            }
            for c in range(len(class_counts))
        },
        "feature_names": list(feature_names),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Consolidação por experimento
# ---------------------------------------------------------------------------

def save_experiment_summary(results, L, path):
    """
    1 linha por experimento. Colunas incluem fitness, F1 weighted/macro,
    contagem de atributos, geração em que o melhor foi encontrado, tempo
    e a string binária do melhor cromossomo.
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "experimento",
            "melhor_fitness",
            "melhor_f1_weighted",
            "melhor_f1_macro",
            "num_atributos_ativos",
            "razao_atributos",
            "geracao_do_melhor",
            "geracoes_executadas",
            "tempo_s",
            "random_state",
            "melhor_cromossomo",
            "atributos_selecionados",
        ])
        for r in results:
            writer.writerow([
                r["experimento"],
                f"{r['best_fitness']:.6f}",
                f"{r['best_f1']:.6f}",
                f"{r['best_f1_macro']:.6f}",
                r["n_atributos"],
                f"{r['n_atributos'] / L:.4f}",
                r["best_generation"],
                r["total_generations"],
                f"{r['tempo_s']:.2f}",
                r["random_state"],
                r["best_genes"],
                "|".join(r["atributos"]),
            ])


# ---------------------------------------------------------------------------
# Convergência agregada
# ---------------------------------------------------------------------------

def save_convergence_aggregate(ga_csv_path, out_path):
    """
    A partir do ga_metrics consolidado, calcula por geração:
      - média, desvio, min, max de melhor_fitness
      - média de fitness_medio e pior_fitness
      - média de num_atributos_ativos e melhor_f1
      - contagem de experimentos ativos na geração
    Experimentos que pararam por estagnação são preenchidos com o último
    valor observado (ffill) — mesma convenção do plotter.
    """
    df = pd.read_csv(ga_csv_path)
    metrics = [
        "melhor_fitness",
        "fitness_medio",
        "pior_fitness",
        "melhor_f1",
        "num_atributos_ativos",
    ]
    pivots = {
        m: df.pivot_table(index="geracao", columns="experimento", values=m).ffill()
        for m in metrics
    }

    n_exp_ativos = df.pivot_table(
        index="geracao", columns="experimento", values="melhor_fitness"
    ).notna().sum(axis=1)

    agg = pd.DataFrame({
        "geracao": pivots["melhor_fitness"].index,
        "melhor_fitness_media": pivots["melhor_fitness"].mean(axis=1).values,
        "melhor_fitness_desvio": pivots["melhor_fitness"].std(axis=1).values,
        "melhor_fitness_min": pivots["melhor_fitness"].min(axis=1).values,
        "melhor_fitness_max": pivots["melhor_fitness"].max(axis=1).values,
        "fitness_medio_media": pivots["fitness_medio"].mean(axis=1).values,
        "pior_fitness_media": pivots["pior_fitness"].mean(axis=1).values,
        "melhor_f1_media": pivots["melhor_f1"].mean(axis=1).values,
        "num_atributos_ativos_media": pivots["num_atributos_ativos"].mean(axis=1).values,
        "n_experimentos_ativos": n_exp_ativos.reindex(pivots["melhor_fitness"].index).values,
    })
    agg.to_csv(out_path, index=False, float_format="%.6f")


# ---------------------------------------------------------------------------
# Frequência de seleção por atributo
# ---------------------------------------------------------------------------

def save_feature_frequency(results, feature_groups, path):
    """
    Para cada atributo semântico (grupo), conta em quantos dos N experimentos
    ele apareceu no melhor cromossomo. Permite identificar quais atributos
    são considerados "essenciais" pela busca do GA.

    Sob codificação binária agrupada, cada índice de gene corresponde a
    um atributo original inteiro (ex.: `RACACOR` com todas as dummies),
    não a uma coluna numérica isolada.
    """
    n_exp = len(results)
    counter = Counter()
    for r in results:
        for idx in r["atributos_idx"]:
            counter[idx] += 1

    rows = []
    for i, (name, col_indices) in enumerate(feature_groups):
        c = counter.get(i, 0)
        rows.append({
            "indice_grupo": i,
            "atributo": name,
            "n_colunas_no_grupo": len(col_indices),
            "vezes_selecionado": c,
            "frequencia": c / n_exp if n_exp > 0 else 0.0,
        })
    df = pd.DataFrame(rows)
    df = df.sort_values(by="vezes_selecionado", ascending=False)
    df.to_csv(path, index=False, float_format="%.4f")


# ---------------------------------------------------------------------------
# Melhor cromossomo global — retreino para matriz de confusão
# ---------------------------------------------------------------------------

def _retrain_best_and_predict(best_genes, X, y, n_classes, random_state, feature_groups):
    """
    Retreina a MLP no melhor cromossomo (com os mesmos hiperparâmetros e
    random_state usados no experimento) e retorna (y_true, y_pred, f1_w, f1_m).

    `best_genes` é um vetor binário de comprimento `len(feature_groups)`.
    Cada gene ativo expande para todas as colunas de X do grupo correspondente.
    """
    active_bits = [i for i, g in enumerate(best_genes) if g == 1]
    col_indices = [c for i in active_bits for c in feature_groups[i][1]]
    Xf = X[:, col_indices]
    X_train, X_val, X_test, y_train, y_val, y_test = _split_70_15_15(
        Xf, y, random_state=random_state
    )
    X_train, y_train = _oversample_minority_classes(X_train, y_train, random_state)
    X_train_val = np.concatenate([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])
    val_fraction = len(X_val) / len(X_train_val)

    model = create_mlp_model(
        input_dim=X_train.shape[1],
        n_classes=n_classes,
        learning_rate=0.001,
        max_iter=200,
        batch_size=64,
        validation_fraction=val_fraction,
        n_iter_no_change=15,
        random_state=random_state,
    )
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        model.fit(X_train_val, y_train_val)

    y_pred = model.predict(X_test)
    f1_w = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    f1_m = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    return y_test, y_pred, f1_w, f1_m


def save_best_global_report(
    results,
    feature_names,
    feature_groups,
    X,
    y,
    n_classes,
    class_names,
    json_path,
    confusion_csv_path,
):
    """
    Identifica o melhor experimento (por fitness) e retreina sua MLP no
    subconjunto de atributos escolhido para produzir:
      - matriz de confusão (CSV com rótulos)
      - classification_report por classe (JSON)
      - metadados do melhor cromossomo (genes, atributos selecionados)
    """
    best_idx = int(np.argmax([r["best_fitness"] for r in results]))
    best = results[best_idx]
    genes = [int(g) for g in best["best_genes"]]

    y_true, y_pred, f1_w, f1_m = _retrain_best_and_predict(
        genes, X, y, n_classes,
        random_state=best["random_state"],
        feature_groups=feature_groups,
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    class_labels = [str(class_names[c]) for c in range(n_classes)]

    with open(confusion_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["real \\ previsto"] + class_labels)
        for i, row in enumerate(cm):
            writer.writerow([class_labels[i]] + list(map(int, row)))

    report = classification_report(
        y_true, y_pred,
        labels=list(range(n_classes)),
        target_names=class_labels,
        output_dict=True,
        zero_division=0,
    )

    payload = {
        "experimento_id": best["experimento"],
        "melhor_fitness": best["best_fitness"],
        "melhor_f1_weighted": best["best_f1"],
        "melhor_f1_macro": best["best_f1_macro"],
        "retreino": {
            "f1_weighted": f1_w,
            "f1_macro": f1_m,
        },
        "num_atributos_ativos": best["n_atributos"],
        "geracao_do_melhor": best["best_generation"],
        "geracoes_executadas": best["total_generations"],
        "tempo_s": best["tempo_s"],
        "random_state": best["random_state"],
        "melhor_cromossomo": best["best_genes"],
        "atributos_selecionados": best["atributos"],
        "classification_report": report,
        "confusion_matrix": {
            "labels": class_labels,
            "matrix": cm.astype(int).tolist(),
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return payload
