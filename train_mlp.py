"""
Train MLPClassifier for TAU 2020 Mobile / DCASE 2020 Task 1A.

Labels are integer-encoded before fit() and inverse-transformed back to
scene names for reporting. This avoids a scikit-learn >=1.7 incompatibility
where MLPClassifier's internal early-stopping validation calls np.isnan()
on the predicted labels, which fails on string dtypes.

Usage:
  python train_mlp.py --features C:\\exercices\\projectsem2\\features_split.csv ^
                      --outdir   C:\\exercices\\projectsem2\\results_mlp
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


HIDDEN_LAYERS_GRID = [(64,), (128,), (64, 64), (128, 64)]
ALPHA_GRID         = [1e-4, 1e-3, 1e-2, 1e-1]
ACTIVATION_GRID    = ["relu", "tanh"]


def load_data(features_path: Path):
    df = pd.read_csv(features_path)
    if "fold" not in df.columns:
        raise SystemExit("Input CSV missing 'fold' column. Run make_split.py first.")

    meta_cols = {"filename", "identifier", "source_label", "target", "split", "fold"}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    logging.info("Loaded %d rows, %d features", len(df), len(feature_cols))

    train = df[df["fold"] == "train"]
    val   = df[df["fold"] == "val"]
    test  = df[df["fold"] == "test"]
    logging.info("Fold sizes: train=%d val=%d test=%d", len(train), len(val), len(test))

    X_train = train[feature_cols].to_numpy()
    y_train = train["target"].to_numpy()
    X_val   = val[feature_cols].to_numpy()
    y_val   = val["target"].to_numpy()
    X_test  = test[feature_cols].to_numpy()
    y_test  = test["target"].to_numpy()
    test_devices = test["source_label"].to_numpy()

    scene_classes = sorted(df["target"].unique())
    return (X_train, y_train, X_val, y_val, X_test, y_test,
            test_devices, feature_cols, scene_classes)


def make_pipeline(estimator) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     estimator),
    ])


def tune_mlp(X_train, y_train_enc, X_val, y_val_enc):
    logging.info("=== Tuning MLPClassifier ===")
    rows = []
    best = {"val_acc": -1.0, "params": None, "pipe": None}

    combos = list(itertools.product(HIDDEN_LAYERS_GRID, ALPHA_GRID, ACTIVATION_GRID))
    n_combos = len(combos)
    logging.info("Sweeping %d configurations", n_combos)

    for i, (hidden, alpha, act) in enumerate(combos, start=1):
        t0 = time.time()
        clf = MLPClassifier(
            hidden_layer_sizes=hidden,
            activation=act,
            alpha=alpha,
            solver="adam",
            learning_rate_init=1e-3,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=0,
            verbose=False,
        )
        pipe = make_pipeline(clf)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            pipe.fit(X_train, y_train_enc)

        train_acc = accuracy_score(y_train_enc, pipe.predict(X_train))
        val_acc   = accuracy_score(y_val_enc,   pipe.predict(X_val))
        elapsed = time.time() - t0
        logging.info(
            "[%2d/%d] hidden=%-10s alpha=%-6g act=%-5s "
            "train_acc=%.4f val_acc=%.4f iters=%-3d (%.1fs)",
            i, n_combos, str(hidden), alpha, act,
            train_acc, val_acc, clf.n_iter_, elapsed,
        )
        rows.append({
            "hidden_layers": str(hidden), "alpha": alpha, "activation": act,
            "train_acc": train_acc, "val_acc": val_acc,
            "n_iter": int(clf.n_iter_), "fit_time_s": elapsed,
        })
        if val_acc > best["val_acc"]:
            best = {
                "val_acc": val_acc,
                "params": {"hidden_layer_sizes": hidden, "alpha": alpha,
                           "activation": act},
                "pipe": pipe,
            }

    grid = pd.DataFrame(rows).sort_values("val_acc", ascending=False)
    logging.info("Best MLP: %s val_acc=%.4f", best["params"], best["val_acc"])
    return best["pipe"], grid, best


def evaluate_on_test(pipe, X_test, y_test_str, test_devices, scene_classes,
                     label_encoder: LabelEncoder, outdir: Path) -> dict:
    logging.info("=== Final evaluation on test ===")
    y_pred_enc = pipe.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_enc)

    test_acc = accuracy_score(y_test_str, y_pred)
    macro_f1 = f1_score(y_test_str, y_pred, average="macro", labels=scene_classes)
    logging.info("MLP test_acc = %.4f  macro_F1 = %.4f", test_acc, macro_f1)

    cm = confusion_matrix(y_test_str, y_pred, labels=scene_classes)
    pd.DataFrame(cm, index=scene_classes, columns=scene_classes).to_csv(
        outdir / "mlp_confusion_test.csv"
    )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(scene_classes)))
        ax.set_yticks(range(len(scene_classes)))
        ax.set_xticklabels(scene_classes, rotation=45, ha="right")
        ax.set_yticklabels(scene_classes)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title("MLP — confusion matrix on test set")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=8)
        fig.colorbar(im, ax=ax); fig.tight_layout()
        fig.savefig(outdir / "mlp_confusion_test.png", dpi=150)
        plt.close(fig)
    except ImportError:
        logging.warning("matplotlib not available; skipping heatmap PNG")

    per_device = {}
    for dev in sorted(np.unique(test_devices)):
        mask = test_devices == dev
        per_device[dev] = float(accuracy_score(y_test_str[mask], y_pred[mask]))
    pd.Series(per_device, name="test_accuracy").to_csv(
        outdir / "mlp_per_device_test.csv", header=True
    )
    logging.info("Per-device test accuracy:")
    for dev, acc in per_device.items():
        logging.info("  %s: %.4f", dev, acc)

    per_class = {}
    for cls in scene_classes:
        mask = y_test_str == cls
        per_class[cls] = float(accuracy_score(y_test_str[mask], y_pred[mask]))
    pd.Series(per_class, name="test_accuracy").to_csv(
        outdir / "mlp_per_class_test.csv", header=True
    )

    return {
        "test_acc": float(test_acc),
        "test_macro_f1": float(macro_f1),
        "per_device_test_acc": per_device,
        "per_class_test_acc": per_class,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--outdir",   required=True, type=Path)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    logging.info("Output directory: %s", args.outdir.resolve())

    (X_train, y_train, X_val, y_val, X_test, y_test,
     test_devices, feature_cols, scene_classes) = load_data(args.features)

    # Encode string labels to integers to dodge sklearn>=1.7 MLPClassifier
    # incompatibility with string labels under early_stopping=True.
    le = LabelEncoder().fit(np.concatenate([y_train, y_val, y_test]))
    y_train_enc = le.transform(y_train)
    y_val_enc   = le.transform(y_val)
    logging.info("Encoded %d classes: %s", len(le.classes_), list(le.classes_))

    t0 = time.time()
    pipe, grid, best = tune_mlp(X_train, y_train_enc, X_val, y_val_enc)
    grid.to_csv(args.outdir / "mlp_grid.csv", index=False)
    joblib.dump(pipe, args.outdir / "mlp_final.joblib")

    test_results = evaluate_on_test(
        pipe, X_test, y_test, test_devices, scene_classes, le, args.outdir,
    )

    summary = {
        "model": "MLPClassifier",
        "n_features": len(feature_cols),
        "n_train": int(len(y_train)),
        "n_val":   int(len(y_val)),
        "n_test":  int(len(y_test)),
        "scene_classes": scene_classes,
        "best_params": {k: (list(v) if isinstance(v, tuple) else v)
                        for k, v in best["params"].items()},
        "val_acc": best["val_acc"],
        "train_acc_at_best": float(accuracy_score(y_train_enc, pipe.predict(X_train))),
        **test_results,
        "total_time_s": time.time() - t0,
    }
    with (args.outdir / "mlp_results.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    print()
    print("=== Summary ===")
    print(f"MLP: {best['params']}  val={best['val_acc']:.4f}  "
          f"test={test_results['test_acc']:.4f}  macroF1={test_results['test_macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())