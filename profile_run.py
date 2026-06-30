"""
Profiling script: mede onde o tempo está sendo gasto numa avaliação típica
de cromossomo (load → split → criar MLP → treinar → predict → métrica).

Rode com: uv run python profile_run.py
"""

import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

sys.path.insert(0, "src")

import numpy as np

from data_loader import load_and_preprocess_data


def time_block(label, fn):
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    return elapsed, result


def profile_keras_pipeline(X, y, n_classes, n_iters=5):
    """Profila a pipeline atual (Keras): import + criar modelo + treinar + predict."""
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam

    timings = {
        "split": [],
        "build_model": [],
        "compile": [],
        "fit": [],
        "predict": [],
        "f1": [],
        "total": [],
    }

    print(f"\n=== KERAS PIPELINE (n={n_iters} iterações) ===")

    for i in range(n_iters):
        rng = np.random.default_rng(42 + i)
        n_features = rng.integers(5, X.shape[1] + 1)
        mask = rng.choice(X.shape[1], size=n_features, replace=False)
        X_filt = X[:, sorted(mask)]

        total_start = time.perf_counter()

        # 1. Split
        t, (X_tr, X_te, y_tr, y_te) = time_block(
            "split",
            lambda: train_test_split(X_filt, y, test_size=0.30, random_state=42, stratify=y),
        )
        t2, (X_val, X_test, y_val, y_test) = time_block(
            "split",
            lambda: train_test_split(X_te, y_te, test_size=0.50, random_state=42, stratify=y_te),
        )
        timings["split"].append(t + t2)

        # 2. Build model
        def build():
            return Sequential([
                Input(shape=(n_features,)),
                Dense(32, activation="relu"),
                Dense(16, activation="relu"),
                Dense(n_classes, activation="softmax"),
            ])

        t, model = time_block("build_model", build)
        timings["build_model"].append(t)

        # 3. Compile
        t, _ = time_block(
            "compile",
            lambda: model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            ),
        )
        timings["compile"].append(t)

        # 4. Fit
        es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=0)
        t, _ = time_block(
            "fit",
            lambda: model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                              epochs=30, batch_size=64, callbacks=[es], verbose=0),
        )
        timings["fit"].append(t)

        # 5. Predict
        t, y_pred_proba = time_block(
            "predict",
            lambda: model.predict(X_test, verbose=0),
        )
        timings["predict"].append(t)

        # 6. F1
        t, _ = time_block(
            "f1",
            lambda: f1_score(y_test, np.argmax(y_pred_proba, axis=1),
                             average="weighted", zero_division=0),
        )
        timings["f1"].append(t)

        timings["total"].append(time.perf_counter() - total_start)
        print(f"  iter {i+1}/{n_iters}: {timings['total'][-1]:.2f}s "
              f"(n_features={n_features})")

    return timings


def profile_sklearn_pipeline(X, y, n_classes, n_iters=5):
    """Profila uma alternativa usando sklearn.neural_network.MLPClassifier."""
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier

    timings = {
        "split": [],
        "build_model": [],
        "fit": [],
        "predict": [],
        "f1": [],
        "total": [],
    }

    print(f"\n=== SKLEARN PIPELINE (n={n_iters} iterações) ===")

    for i in range(n_iters):
        rng = np.random.default_rng(42 + i)
        n_features = rng.integers(5, X.shape[1] + 1)
        mask = rng.choice(X.shape[1], size=n_features, replace=False)
        X_filt = X[:, sorted(mask)]

        total_start = time.perf_counter()

        # 1. Split (mesmo que keras)
        t, (X_tr, X_te, y_tr, y_te) = time_block(
            "split",
            lambda: train_test_split(X_filt, y, test_size=0.30, random_state=42, stratify=y),
        )
        t2, (X_val, X_test, y_val, y_test) = time_block(
            "split",
            lambda: train_test_split(X_te, y_te, test_size=0.50, random_state=42, stratify=y_te),
        )
        timings["split"].append(t + t2)

        # 2. Build model — sklearn cria preguiçoso, só conta o init
        # Para early stopping equivalente ao keras (validação separada),
        # passamos train+val concatenado e usamos validation_fraction
        X_train_val = np.concatenate([X_tr, X_val])
        y_train_val = np.concatenate([y_tr, y_val])
        val_fraction = len(X_val) / len(X_train_val)

        def build():
            return MLPClassifier(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                solver="adam",
                learning_rate_init=0.001,
                max_iter=30,
                batch_size=64,
                early_stopping=True,
                validation_fraction=val_fraction,
                n_iter_no_change=5,
                alpha=0.0,            # desabilita L2 (TF não usa por padrão)
                random_state=42,
            )

        t, clf = time_block("build_model", build)
        timings["build_model"].append(t)

        # 3. Fit (compile + fit do keras → uma chamada só)
        t, _ = time_block("fit", lambda: clf.fit(X_train_val, y_train_val))
        timings["fit"].append(t)

        # 4. Predict
        t, y_pred = time_block("predict", lambda: clf.predict(X_test))
        timings["predict"].append(t)

        # 5. F1
        t, _ = time_block(
            "f1",
            lambda: f1_score(y_test, y_pred, average="weighted", zero_division=0),
        )
        timings["f1"].append(t)

        timings["total"].append(time.perf_counter() - total_start)
        print(f"  iter {i+1}/{n_iters}: {timings['total'][-1]:.2f}s "
              f"(n_features={n_features})")

    return timings


def print_summary(name, timings):
    print(f"\n--- Resumo: {name} ---")
    keys = [k for k in timings.keys() if k != "total"]
    total_avg = np.mean(timings["total"])
    print(f"{'fase':<15s}  {'média (s)':>10s}  {'% total':>10s}")
    for k in keys:
        m = np.mean(timings[k])
        pct = m / total_avg * 100
        print(f"{k:<15s}  {m:>10.4f}  {pct:>9.1f}%")
    print(f"{'TOTAL':<15s}  {total_avg:>10.4f}  {'100.0':>9s}%")
    print(f"Std deviation total: {np.std(timings['total']):.3f}s")


def main():
    print("Carregando base de dados (amostra estratificada de 1500 registros)...")
    X, y, _, n_classes = load_and_preprocess_data(
        "data/dataset-short.xlsx", sample_size=1500, random_state=42, verbose=False
    )
    print(f"  X.shape={X.shape}, classes={n_classes}")

    # Warmup TF
    print("\nWarmup do TF (primeira chamada compila o backend)...")
    t0 = time.perf_counter()
    _ = profile_keras_pipeline(X, y, n_classes, n_iters=1)
    print(f"  Warmup: {time.perf_counter() - t0:.2f}s\n")

    # Profile real
    keras_t = profile_keras_pipeline(X, y, n_classes, n_iters=5)
    sklearn_t = profile_sklearn_pipeline(X, y, n_classes, n_iters=5)

    print_summary("KERAS", keras_t)
    print_summary("SKLEARN", sklearn_t)

    keras_total = np.mean(keras_t["total"])
    sklearn_total = np.mean(sklearn_t["total"])
    print(f"\n>>> Speedup sklearn vs keras: {keras_total/sklearn_total:.2f}×")


if __name__ == "__main__":
    main()
