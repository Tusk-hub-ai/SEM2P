"""
Feature selection for TAU 2020 Mobile / DCASE 2020 Task 1A.

Protocol:
  1. Load features_split.csv and the trained XGBoost model from results_xgb/.
  2. Extract gain-based feature importance from the XGBoost model.
  3. Sweep N (number of top features retained); for each N:
       - Retrain XGBoost on top-N features (same hyperparameters as the
         tuned best config).
       - Score on the validation set (device s3).
       - Record train/val accuracy and fit time.
  4. Select the smallest N where val accuracy is within a tolerance of the
     full-feature baseline (default: within 0.5 percentage points).
  5. Write features_topN.csv -- a reduced-feature CSV with the same fold
     column and metadata. This file can be passed directly to train_svm.py,
     train_xgb.py, and train_mlp.py via their --features argument.

Outputs (in --outdir):
  featsel_sweep.csv          -- N vs train_acc/val_acc/fit_time
  featsel_sweep.png          -- plot of val accuracy vs N (if matplotlib)
  features_topN.csv          -- reduced-feature dataset for re-running models
  featsel_selected.json      -- chosen N, retained feature names, accuracies
  featsel_importance.csv     -- the full XGBoost importance ranking
                                (copied from results_xgb/ for convenience)

Usage:
  python feature_selection.py ^
      --features C:\\exercices\\projectsem2\\features_split.csv ^
      --xgb-model C:\\exercices\\projectsem2\\results_xgb\\xgb_final.joblib ^
      --outdir   C:\\exercices\\projectsem2\\results_featsel
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
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# Sweep grid -- which top-N values to try
# ---------------------------------------------------------------------------
# Roughly log-spaced. Covers very aggressive selection (top 10) through to
# the full set.
N_GRID = [10, 20, 30, 50, 75, 100, 150, 200, 300, 414]

# Tolerance for "matches the full-feature baseline": pick the smallest N
# whose val accuracy is within this many percentage points of the best.
TOLERANCE_PP = 0.5


def load_data(features_path: Path):
    df = pd.read_csv(features_path)
    if "fold" not in df.columns:
        raise SystemExit("Input CSV missing 'fold' column. Run make_split.py first.")

    meta_cols = {"filename", "identifier", "source_label", "target", "split", "fold"}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    logging.info("Loaded %d rows, %d features", len(df), len(feature_cols))
    return df, feature_cols, list(meta_cols & set(df.columns))


def make_pipeline(estimator) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     estimator),
    ])


def get_xgb_params(pipe) -> dict:
    """Extract the hyperparameters of the tuned XGBClassifier inside the pipeline."""
    clf: XGBClassifier = pipe.named_steps["clf"]
    p = clf.get_params()
    # Keep only the knobs we deliberately tuned + the fixed ones.
    return {
        "n_estimators":   p["n_estimators"],
        "max_depth":      p["max_depth"],
        "learning_rate":  p["learning_rate"],
        "reg_lambda":     p["reg_lambda"],
        "objective":      p.get("objective", "multi:softprob"),
        "num_class":      p.get("num_class"),
        "eval_metric":    p.get("eval_metric", "mlogloss"),
        "tree_method":    p.get("tree_method", "hist"),
        "n_jobs":         -1,
        "random_state":   0,
        "verbosity":      0,
    }


def get_importance_ranking(pipe, feature_cols) -> pd.DataFrame:
    """Pull gain-based importance from the trained pipeline, sorted descending."""
    clf: XGBClassifier = pipe.named_steps["clf"]
    importance = clf.feature_importances_
    df = pd.DataFrame({"feature": feature_cols, "importance": importance})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--features",  required=True, type=Path,
                   help="features_split.csv produced by make_split.py")
    p.add_argument("--xgb-model", required=True, type=Path,
                   help="xgb_final.joblib produced by train_xgb.py")
    p.add_argument("--outdir",    required=True, type=Path)
    p.add_argument("--tolerance-pp", type=float, default=TOLERANCE_PP,
                   help="Match-the-baseline tolerance in percentage points")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    logging.info("Output directory: %s", args.outdir.resolve())

    # ---- Load -----------------------------------------------------------
    df, feature_cols, meta_present = load_data(args.features)
    logging.info("Loading XGBoost model from %s", args.xgb_model)
    pipe = joblib.load(args.xgb_model)
    xgb_params = get_xgb_params(pipe)
    logging.info("Using tuned XGB params: %s",
                 {k: v for k, v in xgb_params.items()
                  if k in {"n_estimators","max_depth","learning_rate","reg_lambda"}})

    # ---- Importance ranking ---------------------------------------------
    ranking = get_importance_ranking(pipe, feature_cols)
    ranking.to_csv(args.outdir / "featsel_importance.csv", index=False)
    logging.info("Top 10 features by importance:")
    for _, row in ranking.head(10).iterrows():
        logging.info("  %-32s %.5f", row["feature"], row["importance"])

    # ---- Encode labels & build matrices ---------------------------------
    train = df[df["fold"] == "train"]
    val   = df[df["fold"] == "val"]
    test  = df[df["fold"] == "test"]
    y_all = pd.concat([train["target"], val["target"], test["target"]]).to_numpy()
    le = LabelEncoder().fit(y_all)

    y_train_enc = le.transform(train["target"].to_numpy())
    y_val_enc   = le.transform(val["target"].to_numpy())

    # ---- Sweep N --------------------------------------------------------
    logging.info("=== Sweeping top-N feature counts ===")
    rows = []
    for n in N_GRID:
        if n > len(feature_cols):
            continue
        top_n = ranking["feature"].head(n).tolist()
        X_train_n = train[top_n].to_numpy()
        X_val_n   = val[top_n].to_numpy()

        t0 = time.time()
        clf = XGBClassifier(**xgb_params)
        pipe_n = make_pipeline(clf)
        pipe_n.fit(X_train_n, y_train_enc)
        train_acc = accuracy_score(y_train_enc, pipe_n.predict(X_train_n))
        val_acc   = accuracy_score(y_val_enc,   pipe_n.predict(X_val_n))
        elapsed = time.time() - t0

        logging.info("N=%-3d  train_acc=%.4f  val_acc=%.4f  (%.1fs)",
                     n, train_acc, val_acc, elapsed)
        rows.append({"N": n, "train_acc": train_acc, "val_acc": val_acc,
                     "fit_time_s": elapsed})

    sweep = pd.DataFrame(rows)
    sweep.to_csv(args.outdir / "featsel_sweep.csv", index=False)

    # ---- Plot -----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(sweep["N"], sweep["train_acc"], "o-", label="train", color="C0")
        ax.plot(sweep["N"], sweep["val_acc"],   "o-", label="val (device s3)", color="C1")
        ax.set_xlabel("Number of retained features (top-N by XGBoost gain)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Feature selection — accuracy vs. number of features")
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_xscale("log")
        fig.tight_layout()
        fig.savefig(args.outdir / "featsel_sweep.png", dpi=150)
        plt.close(fig)
        logging.info("Wrote plot: %s", args.outdir / "featsel_sweep.png")
    except ImportError:
        logging.warning("matplotlib not available; skipping plot")

    # ---- Pick the chosen N ---------------------------------------------
    # Smallest N whose val_acc is within tolerance of the best val_acc in the sweep.
    best_val_acc = sweep["val_acc"].max()
    threshold = best_val_acc - args.tolerance_pp / 100.0
    candidates = sweep[sweep["val_acc"] >= threshold].sort_values("N")
    chosen_n = int(candidates.iloc[0]["N"])
    chosen_val_acc = float(candidates.iloc[0]["val_acc"])
    logging.info("Best val_acc in sweep: %.4f  (threshold=%.4f, tolerance=%.1fpp)",
                 best_val_acc, threshold, args.tolerance_pp)
    logging.info("Selected N = %d  (val_acc=%.4f)", chosen_n, chosen_val_acc)

    # ---- Write the reduced-feature CSV ---------------------------------
    chosen_features = ranking["feature"].head(chosen_n).tolist()
    keep_cols = [c for c in df.columns if c in meta_present] + chosen_features
    df_reduced = df[keep_cols]
    out_csv = args.outdir / f"features_top{chosen_n}.csv"
    df_reduced.to_csv(out_csv, index=False)
    logging.info("Wrote reduced-feature CSV (%d rows x %d cols) to %s",
                 df_reduced.shape[0], df_reduced.shape[1], out_csv)

    # ---- Summary --------------------------------------------------------
    summary = {
        "n_features_original": len(feature_cols),
        "chosen_n": chosen_n,
        "chosen_val_acc": chosen_val_acc,
        "best_val_acc_in_sweep": float(best_val_acc),
        "tolerance_pp": args.tolerance_pp,
        "xgb_params_used": {k: v for k, v in xgb_params.items()
                            if k in {"n_estimators","max_depth",
                                     "learning_rate","reg_lambda"}},
        "selected_features": chosen_features,
        "output_csv": str(out_csv),
        "sweep": rows,
    }
    with (args.outdir / "featsel_selected.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    print()
    print("=== Summary ===")
    print(f"Original features:    {len(feature_cols)}")
    print(f"Chosen N:             {chosen_n}")
    print(f"Val accuracy at N:    {chosen_val_acc:.4f}")
    print(f"Best val in sweep:    {best_val_acc:.4f}")
    print(f"Reduced CSV written:  {out_csv}")
    print()
    print("Next: re-run the three model scripts pointing --features at the")
    print("reduced CSV to get post-selection numbers, e.g.:")
    print(f"  python train_xgb.py --features {out_csv} --outdir results_xgb_topN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
