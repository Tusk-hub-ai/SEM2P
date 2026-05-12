"""
Train XGBoost for TAU 2020 Mobile / DCASE 2020 Task 1A.

Protocol mirrors train_svm.py:
  1. Load features_split.csv (must have a 'fold' column).
  2. Pipeline: SimpleImputer(median) -> StandardScaler -> XGBClassifier
     - Imputer and scaler are kept for consistency with the other two
       models; XGBoost itself is scale-invariant and NaN-tolerant.
  3. Sweep a hyperparameter grid; score each candidate on the validation
     fold (device s3); pick the best by val accuracy.
  4. Refit chosen config on train, evaluate ONCE on test.
  5. Save artifacts (grid, final model, confusion matrix, per-device
     test accuracy, results JSON).

Outputs (in --outdir):
  xgb_grid.csv                -- val accuracy for every (n_est, depth, lr, lambda)
  xgb_final.joblib            -- best pipeline, refit on train
  xgb_results.json            -- summary numbers
  xgb_confusion_test.csv/.png -- confusion matrix on test
  xgb_per_device_test.csv     -- accuracy on test broken down by device
  xgb_per_class_test.csv      -- accuracy on test broken down by class
  xgb_feature_importance.csv  -- gain-based feature importance from the
                                 final fitted model

Usage:
  python train_xgb.py --features C:\\exercices\\projectsem2\\features_split.csv ^
                      --outdir   C:\\exercices\\projectsem2\\results_xgb
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# Hyperparameter grid
# ---------------------------------------------------------------------------
# A modest grid: 3 * 3 * 3 * 2 = 54 configurations.
# XGBoost is fast enough on the 9900X that the whole sweep finishes in
# ~10-20 minutes. The grid covers the conventional 'safe' region for
# moderately-sized tabular classification.
N_ESTIMATORS_GRID  = [200, 500, 1000]   # number of boosting rounds (trees)
MAX_DEPTH_GRID     = [4, 6, 10]         # tree depth
LEARNING_RATE_GRID = [0.05, 0.1, 0.3]   # eta -- shrinkage on each tree
REG_LAMBDA_GRID    = [1.0, 10.0]        # L2 regularisation on leaf weights


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------
def tune_xgb(X_train, y_train_enc, X_val, y_val_enc, n_classes):
    """Grid search over XGBoost hyperparameters, scored on val."""
    logging.info("=== Tuning XGBClassifier ===")
    rows = []
    best = {"val_acc": -1.0, "params": None, "pipe": None}

    combos = list(itertools.product(
        N_ESTIMATORS_GRID, MAX_DEPTH_GRID, LEARNING_RATE_GRID, REG_LAMBDA_GRID
    ))
    n_combos = len(combos)
    logging.info("Sweeping %d configurations", n_combos)

    for i, (n_est, depth, lr, reg_l) in enumerate(combos, start=1):
        t0 = time.time()
        clf = XGBClassifier(
            n_estimators=n_est,
            max_depth=depth,
            learning_rate=lr,
            reg_lambda=reg_l,
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            tree_method="hist",     # fastest CPU algorithm
            n_jobs=-1,              # all cores -- XGBoost parallelises natively
            random_state=0,
            verbosity=0,
        )
        pipe = make_pipeline(clf)
        pipe.fit(X_train, y_train_enc)
        train_acc = accuracy_score(y_train_enc, pipe.predict(X_train))
        val_acc   = accuracy_score(y_val_enc,   pipe.predict(X_val))
        elapsed = time.time() - t0
        logging.info(
            "[%2d/%d] n_est=%-4d depth=%-2d lr=%-4s lambda=%-5s "
            "train_acc=%.4f val_acc=%.4f (%.1fs)",
            i, n_combos, n_est, depth, lr, reg_l, train_acc, val_acc, elapsed,
        )
        rows.append({
            "n_estimators": n_est, "max_depth": depth,
            "learning_rate": lr, "reg_lambda": reg_l,
            "train_acc": train_acc, "val_acc": val_acc,
            "fit_time_s": elapsed,
        })
        if val_acc > best["val_acc"]:
            best = {
                "val_acc": val_acc,
                "params": {"n_estimators": n_est, "max_depth": depth,
                           "learning_rate": lr, "reg_lambda": reg_l},
                "pipe": pipe,
            }

    grid = pd.DataFrame(rows).sort_values("val_acc", ascending=False)
    logging.info("Best XGB: %s val_acc=%.4f", best["params"], best["val_acc"])
    return best["pipe"], grid, best


# ---------------------------------------------------------------------------
# Test evaluation
# ---------------------------------------------------------------------------
def evaluate_on_test(pipe, X_test, y_test_str, test_devices, scene_classes,
                     label_encoder: LabelEncoder, outdir: Path) -> dict:
    logging.info("=== Final evaluation on test ===")
    y_pred_enc = pipe.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_enc)

    test_acc = accuracy_score(y_test_str, y_pred)
    macro_f1 = f1_score(y_test_str, y_pred, average="macro", labels=scene_classes)
    logging.info("XGB test_acc = %.4f  macro_F1 = %.4f", test_acc, macro_f1)

    # Confusion matrix
    cm = confusion_matrix(y_test_str, y_pred, labels=scene_classes)
    pd.DataFrame(cm, index=scene_classes, columns=scene_classes).to_csv(
        outdir / "xgb_confusion_test.csv"
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
        ax.set_title("XGBoost — confusion matrix on test set")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=8)
        fig.colorbar(im, ax=ax); fig.tight_layout()
        fig.savefig(outdir / "xgb_confusion_test.png", dpi=150)
        plt.close(fig)
    except ImportError:
        logging.warning("matplotlib not available; skipping heatmap PNG")

    # Per-device accuracy
    per_device = {}
    for dev in sorted(np.unique(test_devices)):
        mask = test_devices == dev
        per_device[dev] = float(accuracy_score(y_test_str[mask], y_pred[mask]))
    pd.Series(per_device, name="test_accuracy").to_csv(
        outdir / "xgb_per_device_test.csv", header=True
    )
    logging.info("Per-device test accuracy:")
    for dev, acc in per_device.items():
        logging.info("  %s: %.4f", dev, acc)

    # Per-class accuracy
    per_class = {}
    for cls in scene_classes:
        mask = y_test_str == cls
        per_class[cls] = float(accuracy_score(y_test_str[mask], y_pred[mask]))
    pd.Series(per_class, name="test_accuracy").to_csv(
        outdir / "xgb_per_class_test.csv", header=True
    )

    return {
        "test_acc": float(test_acc),
        "test_macro_f1": float(macro_f1),
        "per_device_test_acc": per_device,
        "per_class_test_acc": per_class,
    }


def save_feature_importance(pipe, feature_cols, outdir: Path) -> None:
    """Pull XGBoost's gain-based feature importance from the fitted pipeline."""
    clf: XGBClassifier = pipe.named_steps["clf"]
    importance = clf.feature_importances_  # shape (n_features,)
    df = (
        pd.DataFrame({"feature": feature_cols, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    df.to_csv(outdir / "xgb_feature_importance.csv", index=False)
    logging.info("Top 10 features by importance:")
    for _, row in df.head(10).iterrows():
        logging.info("  %-30s %.4f", row["feature"], row["importance"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
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

    # XGBoost requires integer labels (0..n_classes-1). LabelEncoder handles
    # string labels and lets us round-trip back to scene names for reporting.
    le = LabelEncoder().fit(np.concatenate([y_train, y_val, y_test]))
    y_train_enc = le.transform(y_train)
    y_val_enc   = le.transform(y_val)
    n_classes = len(le.classes_)
    logging.info("Encoded %d classes: %s", n_classes, list(le.classes_))

    t0 = time.time()
    pipe, grid, best = tune_xgb(X_train, y_train_enc, X_val, y_val_enc, n_classes)
    grid.to_csv(args.outdir / "xgb_grid.csv", index=False)
    joblib.dump(pipe, args.outdir / "xgb_final.joblib")

    test_results = evaluate_on_test(
        pipe, X_test, y_test, test_devices, scene_classes, le, args.outdir,
    )
    save_feature_importance(pipe, feature_cols, args.outdir)

    summary = {
        "model": "XGBoost",
        "n_features": len(feature_cols),
        "n_train": int(len(y_train)),
        "n_val":   int(len(y_val)),
        "n_test":  int(len(y_test)),
        "scene_classes": scene_classes,
        "best_params": best["params"],
        "val_acc": best["val_acc"],
        "train_acc_at_best": float(accuracy_score(y_train_enc, pipe.predict(X_train))),
        **test_results,
        "total_time_s": time.time() - t0,
    }
    with (args.outdir / "xgb_results.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    print()
    print("=== Summary ===")
    print(f"XGBoost: {best['params']}  val={best['val_acc']:.4f}  "
          f"test={test_results['test_acc']:.4f}  macroF1={test_results['test_macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
