"""
detector.py
===========
Anomaly-detection pipeline.

Runs inference on the test set, applies the reconstruction-error threshold,
and exports a CSV of detected anomalous sequences.
"""

import os
import time

import numpy as np
import pandas as pd
import keras

from .config import HyperParams, Paths, FEATURE_COLUMNS, get_device
from .model import LSTMAutoencoder
from .preprocess import Preprocessor
from .dataset import make_sequences, make_seq_labels


def detect(
    hp: HyperParams = HyperParams(),
    paths: Paths = Paths(),
    threshold_percentile: int = 95,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Identify anomalous sequences in the test dataset.

    The anomaly threshold is the *threshold_percentile*-th percentile of
    reconstruction errors observed on **normal** sequences.

    Args:
        hp                   : Hyperparameter configuration.
        paths                : File-path configuration.
        threshold_percentile : Percentile of normal errors used as threshold.
        verbose              : Print progress to stdout when True.

    Returns:
        DataFrame of detected anomalous sequences with metadata columns:
        ``Original_Row_Index``, ``Reconstruction_Error``, ``Label``, and all
        raw CSV columns of the last event in each anomalous window.
    """
    device = get_device()

    if verbose:
        print("=" * 60)
        print("  LSTM AUTOENCODER -- ANOMALY DETECTION PIPELINE (Keras)")
        print("=" * 60)
        print(f"  Device: {device}")
        print()

    if not os.path.exists(paths.model):
        raise FileNotFoundError(f"Model not found at {paths.model}. Run train.py first.")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    if verbose:
        print(f"[1/4] Loading data from: {paths.eval_data}")
    df_raw = pd.read_csv(paths.eval_data, low_memory=False)
    labels = df_raw["Label"].values

    pp = Preprocessor.load(FEATURE_COLUMNS, paths.encoders, paths.scaler)
    X = pp.transform(df_raw)

    # ------------------------------------------------------------------
    # 2. Build sequences
    # ------------------------------------------------------------------
    if verbose:
        print(f"[2/4] Building sequences (seq_len={hp.seq_len}) ...")
    sequences = make_sequences(X, hp.seq_len)
    seq_labels = make_seq_labels(labels, hp.seq_len)

    # ------------------------------------------------------------------
    # 3. Compute reconstruction errors
    # ------------------------------------------------------------------
    if verbose:
        print("[3/4] Loading model and computing reconstruction errors ...")
    model = keras.models.load_model(paths.model, custom_objects={"LSTMAutoencoder": LSTMAutoencoder})

    t0 = time.time()
    reconstructed = model.predict(sequences, batch_size=hp.batch_size, verbose=1 if verbose else 0)
    errors = np.mean(np.square(reconstructed - sequences), axis=(1, 2))
    if verbose:
        print(f"      Completed in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # 4. Apply threshold and extract anomalous rows
    # ------------------------------------------------------------------
    normal_errors = errors[seq_labels == 0]
    threshold = np.percentile(normal_errors, threshold_percentile)
    anomaly_mask = errors > threshold
    anomaly_idx = np.where(anomaly_mask)[0]

    # Each sequence i ends at raw row (i + seq_len - 1)
    trigger_rows = df_raw.iloc[anomaly_idx + hp.seq_len - 1].copy()
    trigger_rows["Original_Row_Index"] = anomaly_idx
    trigger_rows["Reconstruction_Error"] = errors[anomaly_idx]
    trigger_rows["Label"] = seq_labels[anomaly_idx]

    # Reorder key columns to the front for readability
    key_cols = ["Original_Row_Index", "SystemTime", "Image", "CommandLine", "User", "EventID", "Label", "Reconstruction_Error"]
    present = [c for c in key_cols if c in trigger_rows.columns]
    other = [c for c in trigger_rows.columns if c not in present]
    trigger_rows = trigger_rows[present + other]

    os.makedirs(os.path.dirname(paths.detected_anomalies), exist_ok=True)
    trigger_rows.to_csv(paths.detected_anomalies, index=False)

    if verbose:
        print(f"\n[4/4] Detected {len(trigger_rows):,} anomalous sequences.")
        print(f"      Threshold (p{threshold_percentile} of normal): {threshold:.6f}")
        print(f"      Saved -> {paths.detected_anomalies}")

    return trigger_rows
