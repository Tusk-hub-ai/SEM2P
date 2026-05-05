"""
Train SVM (Linear + RBF) for TAU 2020 Mobile / DCASE 2020 Task 1A.

Protocol:
  1. Load features_split.csv (must have a 'fold' column).
  2. Fit a Pipeline of: SimpleImputer(median) -> StandardScaler -> SVM
     - SimpleImputer fills the 9 NaN HNR rows
     - StandardScaler is fitted on TRAIN ONLY, transform applied to all
  3. For each kernel (linear, rbf):
       - sweep a hyperparameter grid
       - score each candidate on the validation fold (device s3)
       - pick the best by val accuracy
  4. Refit the chosen config on train, evaluate ONCE on test.
  5. Report train / val / test accuracy, confusion matrix on test,
     per-device test accuracy, and save artifacts.

Outputs (in --outdir):
  svm_linear_grid.csv         -- val accuracy for every linear C
  svm_rbf_grid.csv            -- val accuracy for every (C, gamma) on RBF
  svm_linear_final.joblib     -- best linear pipeline, refit on train
  svm_rbf_final.joblib        -- best RBF pipeline, refit on train
  svm_results.json            -- summary numbers for the chapter
  svm_rbf_confusion_test.csv  -- confusion matrix on test (rows=true, cols=pred)
  svm_rbf_confusion_test.png  -- the same as a heatmap
  svm_rbf_per_device_test.csv -- accuracy on test broken down by device

Usage:
  python train_svm.py --features C:\\exercices\\projectsem2\\features_split.csv ^
                      --outdir   C:\\exercices\\projectsem2\\results_svm
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC


# ---------------------------------------------------------------------------
# Hyperparameter grids
# ---------------------------------------------------------------------------
# Linear: just C. LinearSVC uses squared hinge by default and is much faster
# than SVC(kernel='linear') on this problem size.
LINEAR_C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]

# RBF: C * gamma. 4x4 = 16 fits. Standard log-scale grid.
RBF_C_GRID = [0.1, 1.0, 10.0, 100.0]
RBF_GAMMA_GRID = ["scale", 0.001, 0.01, 0.1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_data(features_path: Path):
    """Return X_train, y_train, X_val, y_val, X_test, y_test, test_devices, feature_cols, scene_classes."""
    df = pd.read_csv(features_path)
    if "fold" not in df.columns:
        raise SystemExit(
            "Input CSV missing 'fold' column. Run make_split.py first."
        )

    meta_cols = {"filename", "identifier", "source_label", "target", "split", "fold"}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    logging.info("Loaded %d rows, %d features", len(df), len(feature_cols))

    train = df[df["fold"] == "train"]
    val = df[df["fold"] == "val"]
    test = df[df["fold"] == "test"]

    logging.info("Fold sizes: train=%d val=%d test=%d", len(train), len(val), len(test))

    X_train = train[feature_cols].to_numpy()
    y_train = train["target"].to_numpy()
    X_val = val[feature_cols].to_numpy()
    y_val = val["target"].to_numpy()
    X_test = test[feature_cols].to_numpy()
    y_test = test["target"].to_numpy()
    test_devices = test["source_label"].to_numpy()

    scene_classes = sorted(df["target"].unique())
    return (X_train, y_train, X_val, y_val, X_test, y_test,
            test_devices, feature_cols, scene_classes)


def make_pipeline(estimator) -> Pipeline:
    """Imputer -> Scaler -> classifier. Fit only on train; transforms inherited."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", estimator),
    ])


# ---------------------------------------------------------------------------
# Linear SVM tuning
# ---------------------------------------------------------------------------
def tune_linear(X_train, y_train, X_val, y_val) -> tuple[Pipeline, pd.DataFrame, dict]:
    logging.info("=== Tuning LinearSVC ===")
    rows = []
    best = {"val_acc": -1.0, "C": None, "pipe": None}

    for C in LINEAR_C_GRID:
        t0 = time.time()
        pipe = make_pipeline(LinearSVC(
            C=C,
            dual="auto",         # picks dual vs primal automatically
            max_iter=10_000,
            random_state=0,
        ))
        pipe.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, pipe.predict(X_train))
        val_acc = accuracy_score(y_val, pipe.predict(X_val))
        elapsed = time.time() - t0
        logging.info("Linear C=%-6g  train_acc=%.4f  val_acc=%.4f  (%.1fs)",
                     C, train_acc, val_acc, elapsed)
        rows.append({"C": C, "train_acc": train_acc, "val_acc": val_acc,
                     "fit_time_s": elapsed})
        if val_acc > best["val_acc"]:
            best = {"val_acc": val_acc, "C": C, "pipe": pipe}

    grid = pd.DataFrame(rows).sort_values("val_acc", ascending=False)
    logging.info("Best LinearSVC: C=%g  val_acc=%.4f", best["C"], best["val_acc"])
    return best["pipe"], grid, {"C": best["C"], "val_acc": best["val_acc"]}


# ---------------------------------------------------------------------------
# RBF SVM tuning
# ---------------------------------------------------------------------------
def tune_rbf(X_train, y_train, X_val, y_val) -> tuple[Pipeline, pd.DataFrame, dict]:
    logging.info("=== Tuning SVC(kernel='rbf') ===")
    rows = []
    best = {"val_acc": -1.0, "C": None, "gamma": None, "pipe": None}

    n_combos = len(RBF_C_GRID) * len(RBF_GAMMA_GRID)
    i = 0
    for C in RBF_C_GRID:
        for gamma in RBF_GAMMA_GRID:
            i += 1
            t0 = time.time()
            pipe = make_pipeline(SVC(
                kernel="rbf",
                C=C,
                gamma=gamma,
                cache_size=8000,    # MB of kernel cache; speeds up rbf
                random_state=0,
            ))
            pipe.fit(X_train, y_train)
            train_acc = accuracy_score(y_train, pipe.predict(X_train))
            val_acc = accuracy_score(y_val, pipe.predict(X_val))
            elapsed = time.time() - t0
            logging.info("[%2d/%2d] RBF C=%-6g gamma=%-8s  train_acc=%.4f  val_acc=%.4f  (%.1fs)",
                         i, n_combos, C, str(gamma), train_acc, val_acc, elapsed)
            rows.append({
                "C": C, "gamma": gamma,
                "train_acc": train_acc, "val_acc": val_acc,
                "fit_time_s": elapsed,
            })
            if val_acc > best["val_acc"]:
                best = {"val_acc": val_acc, "C": C, "gamma": gamma, "pipe": pipe}

    grid = pd.DataFrame(rows).sort_values("val_acc", ascending=False)
    logging.info("Best RBF: C=%g gamma=%s  val_acc=%.4f",
                 best["C"], best["gamma"], best["val_acc"])
    return best["pipe"], grid, {
        "C": best["C"], "gamma": best["gamma"], "val_acc": best["val_acc"],
    }


# ---------------------------------------------------------------------------
# Test-set evaluation (called once per kernel, on the chosen config)
# ---------------------------------------------------------------------------
def evaluate_on_test(pipe: Pipeline, X_test, y_test, test_devices, scene_classes,
                     tag: str, outdir: Path) -> dict:
    logging.info("=== Final evaluation on test, model=%s ===", tag)
    y_pred = pipe.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    logging.info("%s test_acc = %.4f", tag, test_acc)

    # Confusion matrix on test
    cm = confusion_matrix(y_test, y_pred, labels=scene_classes)
    cm_df = pd.DataFrame(cm, index=scene_classes, columns=scene_classes)
    cm_df.to_csv(outdir / f"svm_{tag}_confusion_test.csv")

    # Heatmap
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
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"SVM ({tag}) — confusion matrix on test set")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=8)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(outdir / f"svm_{tag}_confusion_test.png", dpi=150)
        plt.close(fig)
    except ImportError:
        logging.warning("matplotlib not available; skipping heatmap PNG")

    # Per-device test accuracy — the headline Task 1A diagnostic
    per_device = {}
    for dev in sorted(np.unique(test_devices)):
        mask = test_devices == dev
        per_device[dev] = float(accuracy_score(y_test[mask], y_pred[mask]))
    pd.Series(per_device, name="test_accuracy").to_csv(
        outdir / f"svm_{tag}_per_device_test.csv", header=True
    )
    logging.info("Per-device test accuracy:")
    for dev, acc in per_device.items():
        logging.info("  %s: %.4f", dev, acc)

    return {"test_acc": float(test_acc), "per_device_test_acc": per_device}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", required=True, type=Path,
                   help="features_split.csv produced by make_split.py")
    p.add_argument("--outdir", required=True, type=Path,
                   help="Output directory for models, grids, results")
    p.add_argument("--linear-only", action="store_true",
                   help="Skip RBF (smoke test)")
    p.add_argument("--rbf-only", action="store_true",
                   help="Skip Linear")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    logging.info("Output directory: %s", args.outdir.resolve())

    # ---- Load --------------------------------------------------------------
    (X_train, y_train, X_val, y_val, X_test, y_test,
     test_devices, feature_cols, scene_classes) = load_data(args.features)

    summary = {
        "n_features": len(feature_cols),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "scene_classes": scene_classes,
    }

    # ---- Linear ------------------------------------------------------------
    if not args.rbf_only:
        t0 = time.time()
        lin_pipe, lin_grid, lin_best = tune_linear(X_train, y_train, X_val, y_val)
        lin_grid.to_csv(args.outdir / "svm_linear_grid.csv", index=False)
        joblib.dump(lin_pipe, args.outdir / "svm_linear_final.joblib")
        lin_test = evaluate_on_test(lin_pipe, X_test, y_test, test_devices,
                                    scene_classes, "linear", args.outdir)
        summary["linear"] = {
            "best_C": lin_best["C"],
            "val_acc": lin_best["val_acc"],
            "train_acc_at_best": float(accuracy_score(y_train, lin_pipe.predict(X_train))),
            **lin_test,
            "tuning_time_s": time.time() - t0,
        }

    # ---- RBF ---------------------------------------------------------------
    if not args.linear_only:
        t0 = time.time()
        rbf_pipe, rbf_grid, rbf_best = tune_rbf(X_train, y_train, X_val, y_val)
        rbf_grid.to_csv(args.outdir / "svm_rbf_grid.csv", index=False)
        joblib.dump(rbf_pipe, args.outdir / "svm_rbf_final.joblib")
        rbf_test = evaluate_on_test(rbf_pipe, X_test, y_test, test_devices,
                                    scene_classes, "rbf", args.outdir)
        summary["rbf"] = {
            "best_C": rbf_best["C"],
            "best_gamma": str(rbf_best["gamma"]),
            "val_acc": rbf_best["val_acc"],
            "train_acc_at_best": float(accuracy_score(y_train, rbf_pipe.predict(X_train))),
            **rbf_test,
            "tuning_time_s": time.time() - t0,
        }

    # ---- Summary -----------------------------------------------------------
    with (args.outdir / "svm_results.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    logging.info("Wrote summary to %s", args.outdir / "svm_results.json")

    print()
    print("=== Summary ===")
    if "linear" in summary:
        s = summary["linear"]
        print(f"Linear:  C={s['best_C']}  val={s['val_acc']:.4f}  test={s['test_acc']:.4f}")
    if "rbf" in summary:
        s = summary["rbf"]
        print(f"RBF:     C={s['best_C']} gamma={s['best_gamma']}  "
              f"val={s['val_acc']:.4f}  test={s['test_acc']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
