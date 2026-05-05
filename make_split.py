"""
Add a `fold` column to features.csv with values:
    train    — fold1_train clips on devices a, b, c, s1, s2
    val      — fold1_train clips on device s3 (held out for hyperparameter tuning)
    test     — all fold1_evaluate clips (untouched until final reporting)
    unused   — the 6,105 leftover clips (kept out of every model)

Rationale (device-aware validation):
    DCASE 2020 Task 1A's test set contains unseen devices (s4-s6) which
    do not appear in training. A random validation split would not reflect
    this device shift, so hyperparameters tuned on it could be optimistic.
    Holding out an in-domain device (s3) from training gives validation a
    device-generalisation signal that mimics the train->test gap.

Usage:
    python make_split.py --features C:\\exercices\\projectsem2\\features.csv \\
                         --output   C:\\exercices\\projectsem2\\features_split.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

VAL_DEVICE = "s3"  # held out from train to act as validation


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True, type=Path,
                   help="Input features CSV (from extract_features.py)")
    p.add_argument("--output", required=True, type=Path,
                   help="Output CSV path (with fold column added)")
    args = p.parse_args()

    df = pd.read_csv(args.features)
    print(f"Loaded {len(df)} rows from {args.features}")

    # Sanity: required columns
    for col in ("split", "source_label", "target"):
        if col not in df.columns:
            raise SystemExit(f"Required column '{col}' missing from input CSV.")

    # Build the fold column.
    #   split == 'evaluate' -> 'test'
    #   split == 'train' & device == VAL_DEVICE -> 'val'
    #   split == 'train' & device != VAL_DEVICE -> 'train'
    #   split == 'unused' -> 'unused'
    fold = pd.Series(index=df.index, dtype="object")
    fold[df["split"] == "evaluate"] = "test"
    fold[df["split"] == "unused"] = "unused"
    train_mask = df["split"] == "train"
    fold[train_mask & (df["source_label"] == VAL_DEVICE)] = "val"
    fold[train_mask & (df["source_label"] != VAL_DEVICE)] = "train"

    if fold.isna().any():
        raise SystemExit("Bug: some rows did not get assigned a fold.")
    df["fold"] = fold

    # --- Reporting -------------------------------------------------------
    print()
    print("=== Fold totals ===")
    print(df["fold"].value_counts().to_string())
    print()

    print("=== Devices per fold ===")
    print(pd.crosstab(df["fold"], df["source_label"]).to_string())
    print()

    # The table for §3.1 / Table 3.1.
    # Order matches your Table 3.1 layout (airport first, urban park last).
    scene_order = [
        "airport", "shopping_mall", "metro_station", "street_pedestrian",
        "public_square", "street_traffic", "tram", "bus", "metro", "park",
    ]
    pretty = {
        "airport": "Airport",
        "shopping_mall": "Indoor shopping mall",
        "metro_station": "Metro station",
        "street_pedestrian": "Pedestrian street",
        "public_square": "Public square",
        "street_traffic": "Street, medium traffic",
        "tram": "Tram",
        "bus": "Bus",
        "metro": "Underground metro",
        "park": "Urban park",
    }

    counts = (
        df[df["fold"].isin(["train", "val", "test"])]
        .pivot_table(index="target", columns="fold", values="filename",
                     aggfunc="count", fill_value=0)
        .reindex(scene_order)
    )
    counts = counts[["train", "val", "test"]]  # column order
    counts["Total"] = counts.sum(axis=1)

    print("=== Table 3.1 — Per-class sample counts ===")
    print(f"{'#':>2}  {'Class':<24} {'Train':>6} {'Validation':>11} {'Test':>6} {'Total':>6}")
    print("-" * 60)
    for i, scene in enumerate(scene_order):
        row = counts.loc[scene]
        print(f"{i:>2}  {pretty[scene]:<24} "
              f"{row['train']:>6d} {row['val']:>11d} {row['test']:>6d} {row['Total']:>6d}")
    print("-" * 60)
    totals = counts.sum(axis=0)
    print(f"    {'TOTAL':<24} "
          f"{totals['train']:>6d} {totals['val']:>11d} {totals['test']:>6d} {totals['Total']:>6d}")
    print()

    # --- Write -----------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows x {df.shape[1]} cols to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
