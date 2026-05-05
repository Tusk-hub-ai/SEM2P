"""
Extract per-clip features from TAU Urban Acoustic Scenes 2020 Mobile.

Features per clip:
  - Log-mel energies (n_mels=64) + delta + delta-delta, each summarised
    as mean and std across time -> 384 features
  - Spectral/temporal: ZCR, RMS, spectral centroid, bandwidth, rolloff,
    flatness, flux/onset strength, 7-band spectral contrast, HNR
    (mean+std where applicable)

Output CSV columns:
  filename, identifier, source_label, target, split, <features...>

Usage:
  python extract_features.py \
      --dataset-root "C:/exercices/projectsem2/data/TAU-urban-acoustic-scenes-2020-mobile-development" \
      --output "C:/exercices/projectsem2/features.csv" \
      [--limit 50] [--n-jobs -1]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
from joblib import Parallel, delayed

# parselmouth is only needed inside workers; import lazily there too,
# but importing here lets us fail fast if it's missing.
try:
    import parselmouth  # noqa: F401
except ImportError:
    print(
        "ERROR: parselmouth not installed. Run: pip install praat-parselmouth",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Audio / feature parameters
# ---------------------------------------------------------------------------
SR = 44100
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 64
N_CONTRAST_BANDS = 7  # librosa default -> 7 bands (6 + 1)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def _mean_std(name: str, x: np.ndarray) -> dict:
    """Summarise a 1D time series as mean and std."""
    return {f"{name}_mean": float(np.mean(x)), f"{name}_std": float(np.std(x))}


def _per_band_mean_std(prefix: str, M: np.ndarray) -> dict:
    """Summarise a (bands, frames) matrix as per-band mean + std."""
    means = np.mean(M, axis=1)
    stds = np.std(M, axis=1)
    out = {}
    for i in range(M.shape[0]):
        out[f"{prefix}{i:02d}_mean"] = float(means[i])
        out[f"{prefix}{i:02d}_std"] = float(stds[i])
    return out


def _hnr_mean_std(y: np.ndarray, sr: int) -> dict:
    """
    Harmonic-to-noise ratio via Praat (parselmouth).

    Praat's HNR is autocorrelation-based and tuned for voiced speech.
    On non-speech audio it can return undefined values; we catch any
    failure and emit NaN rather than crashing.
    """
    import parselmouth
    try:
        snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)
        # Defaults: time_step=0.01s, min_pitch=75 Hz, silence_threshold=0.1,
        # periods_per_window=1.0. These are Praat's standard.
        harm = snd.to_harmonicity_cc(
            time_step=0.01,
            minimum_pitch=75.0,
            silence_threshold=0.1,
            periods_per_window=1.0,
        )
        values = harm.values[harm.values != -200]  # -200 == undefined in Praat
        if values.size == 0:
            return {"hnr_mean": np.nan, "hnr_std": np.nan}
        return {"hnr_mean": float(np.mean(values)), "hnr_std": float(np.std(values))}
    except Exception:
        return {"hnr_mean": np.nan, "hnr_std": np.nan}


def extract_features_for_clip(audio_path: Path) -> dict:
    """
    Extract all features for a single wav file.

    Returns a dict of feature_name -> float. Raises on hard failures
    so the caller can log and skip.
    """
    # sr=None preserves native sample rate (44.1 kHz for TAU)
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    if sr != SR:
        # TAU is 44.1 kHz; if a file isn't, resample so feature params
        # are consistent across the dataset.
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        sr = SR

    feats: dict = {}

    # --- Log-mel energies + deltas + delta-deltas -------------------------
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)  # (n_mels, frames)
    delta = librosa.feature.delta(log_mel, order=1)
    delta2 = librosa.feature.delta(log_mel, order=2)

    feats.update(_per_band_mean_std("logmel", log_mel))
    feats.update(_per_band_mean_std("logmel_d1_", delta))
    feats.update(_per_band_mean_std("logmel_d2_", delta2))

    # --- Spectral / temporal features -------------------------------------
    # All use the same n_fft / hop_length so frame counts line up.
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))

    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(S=S)[0]
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)  # (bands, frames)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]

    feats.update(_mean_std("spec_centroid", centroid))
    feats.update(_mean_std("spec_bandwidth", bandwidth))
    feats.update(_mean_std("spec_rolloff", rolloff))
    feats.update(_mean_std("spec_flatness", flatness))
    feats.update(_per_band_mean_std("spec_contrast", contrast))
    feats.update(_mean_std("spec_flux", onset_env))  # onset strength == half-wave-rectified flux
    feats.update(_mean_std("zcr", zcr))
    feats.update(_mean_std("rms", rms))

    # --- HNR via Praat ----------------------------------------------------
    feats.update(_hnr_mean_std(y, sr))

    return feats


def process_one(row: dict, audio_root: Path) -> dict | None:
    """Worker: process a single meta.csv row. Returns row-dict or None on failure."""
    rel = row["filename"]  # e.g. "audio/airport-barcelona-0-0-a.wav"
    audio_path = audio_root / rel
    try:
        feats = extract_features_for_clip(audio_path)
    except Exception as exc:  # noqa: BLE001
        logging.warning("FAILED %s: %s", rel, exc)
        return None

    out = {
        "filename": rel,
        "identifier": row["identifier"],
        "source_label": row["source_label"],
        "target": row["scene_label"],
        "split": row["split"],
    }
    out.update(feats)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def build_split_map(dataset_root: Path) -> dict[str, str]:
    """Map filename -> 'train' / 'evaluate' / 'unused' from fold1 CSVs."""
    setup = dataset_root / "evaluation_setup"
    train_csv = setup / "fold1_train.csv"
    eval_csv = setup / "fold1_evaluate.csv"
    if not train_csv.exists() or not eval_csv.exists():
        raise FileNotFoundError(
            f"Could not find fold1_train.csv / fold1_evaluate.csv in {setup}"
        )

    # TAU CSVs are tab-separated.
    train_df = pd.read_csv(train_csv, sep="\t")
    eval_df = pd.read_csv(eval_csv, sep="\t")
    split: dict[str, str] = {}
    for fn in train_df["filename"]:
        split[fn] = "train"
    for fn in eval_df["filename"]:
        split[fn] = "evaluate"
    return split


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", required=True, type=Path,
                   help="Path to TAU-urban-acoustic-scenes-2020-mobile-development/")
    p.add_argument("--output", required=True, type=Path,
                   help="Output CSV path")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N clips (for smoke test)")
    p.add_argument("--n-jobs", type=int, default=-1,
                   help="joblib n_jobs (default -1 = all cores)")
    p.add_argument("--log-every", type=int, default=200,
                   help="Log a heartbeat every N completed clips")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        logging.error("Dataset root does not exist: %s", dataset_root)
        return 1

    meta_path = dataset_root / "meta.csv"
    if not meta_path.exists():
        logging.error("meta.csv not found at %s", meta_path)
        return 1

    logging.info("Reading meta.csv ...")
    meta = pd.read_csv(meta_path, sep="\t")
    logging.info("meta.csv: %d rows, columns=%s", len(meta), list(meta.columns))

    split_map = build_split_map(dataset_root)
    meta["split"] = meta["filename"].map(split_map).fillna("unused")
    counts = meta["split"].value_counts().to_dict()
    logging.info("Split counts: %s", counts)

    if args.limit is not None:
        meta = meta.head(args.limit).copy()
        logging.info("Limiting to first %d clips (smoke test)", len(meta))

    rows = meta.to_dict(orient="records")
    n = len(rows)
    logging.info("Extracting features for %d clips with n_jobs=%d ...",
                 n, args.n_jobs)
    t0 = time.time()

    # joblib backend: 'loky' (default) is process-based and avoids GIL/BLAS
    # contention; librosa releases the GIL in C code but spectral_contrast
    # etc. don't always, so processes are safer.
    results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=10)(
        delayed(process_one)(row, dataset_root) for row in rows
    )

    elapsed = time.time() - t0
    ok = [r for r in results if r is not None]
    failed = n - len(ok)
    logging.info("Done in %.1fs. ok=%d failed=%d", elapsed, len(ok), failed)

    if not ok:
        logging.error("No clips processed successfully; not writing output.")
        return 2

    df = pd.DataFrame(ok)

    # Stable column order: meta cols first, then features sorted.
    meta_cols = ["filename", "identifier", "source_label", "target", "split"]
    feat_cols = sorted(c for c in df.columns if c not in meta_cols)
    df = df[meta_cols + feat_cols]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    logging.info("Wrote %d rows x %d cols to %s",
                 df.shape[0], df.shape[1], args.output)

    # Quick post-write summary
    n_nan_rows = df.isna().any(axis=1).sum()
    logging.info("Rows with any NaN: %d (HNR is the usual culprit on non-speech)",
                 n_nan_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())