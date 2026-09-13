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
from .model import LSTMAutoencoder, EventSpecificMaskedMSE
from .preprocess import Preprocessor
from .dataset import build_tree_dataset, vectorize_tree_sequences


def detect(
    hp: HyperParams = HyperParams(),
    paths: Paths = Paths(),
    threshold_percentile: float = 80.0,
    verbose: bool = True,
    stride: int = 3,
) -> pd.DataFrame:
    """
    Identify anomalous sequences in the test dataset using 15-step hierarchical tree sequences.

    The anomaly threshold is the *threshold_percentile*-th percentile of
    reconstruction errors observed on **normal** sequences.

    Args:
        hp                   : Hyperparameter configuration.
        paths                : File-path configuration.
        threshold_percentile : Percentile of normal errors used as threshold.
        verbose              : Print progress to stdout when True.
        stride               : Action sequence sliding stride.

    Returns:
        DataFrame of detected anomalous sequences with metadata columns.
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
    # 1. Load data & generate 15-step tree sequences
    # ------------------------------------------------------------------
    if verbose:
        print(f"[1/4] Generating 15-step process-action sequences from: {paths.eval_data} (stride={stride})")
    pp = Preprocessor.load(FEATURE_COLUMNS, paths.encoders, paths.scaler)
    raw_seqs, seq_labels, metadatas = build_tree_dataset(paths.eval_data, stride=stride)
    sequences = vectorize_tree_sequences(raw_seqs, pp, fit=False, seq_len=hp.seq_len)

    if verbose:
        print(f"      Total sequences: {len(sequences):,} (normal={np.sum(seq_labels==0):,}, anomalous={np.sum(seq_labels==1):,})")

    # ------------------------------------------------------------------
    # 2. Compute reconstruction errors
    # ------------------------------------------------------------------
    if verbose:
        print("[2/4] Loading model and computing reconstruction errors ...")
    model = keras.models.load_model(
        paths.model, 
        custom_objects={
            "LSTMAutoencoder": LSTMAutoencoder, 
            "EventSpecificMaskedMSE": EventSpecificMaskedMSE
        }
    )

    t0 = time.time()
    reconstructed = model.predict(sequences, batch_size=hp.batch_size, verbose=1 if verbose else 0)
    
    # Retrieve the event-specific feature mask from preprocessor
    feature_names = list(pp._scaler.feature_names_in_)
    event_col_indices = [feature_names.index(f"EventID_{eid}") for eid in pp.event_ids]
    mask_matrix = pp.get_event_feature_mask()
    
    # Construct 3D feature mask using numpy matrix multiplication
    one_hot_events = sequences[:, :, event_col_indices]
    feature_mask = np.matmul(one_hot_events, mask_matrix)
    
    # Mask out zero-padded blank steps (all features are zero)
    mask = np.any(sequences != 0, axis=2)  # shape (N, seq_len)
    step_counts = np.sum(mask, axis=1, keepdims=True)  # shape (N, 1)
    
    # Compute squared error and apply feature mask
    sq_err = np.square(reconstructed - sequences)
    masked_sq_err = sq_err * feature_mask
    
    # Sum over features and divide by active feature counts
    active_feature_counts = np.sum(feature_mask, axis=2, keepdims=True)
    active_feature_counts = np.maximum(active_feature_counts, 1.0)
    mse_per_step = np.sum(masked_sq_err, axis=2) / np.squeeze(active_feature_counts, axis=2)
    
    errors = np.sum(mse_per_step * mask, axis=1) / np.squeeze(step_counts, axis=1)
    
    if verbose:
        print(f"      Completed in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # 3. Apply threshold and extract anomalous sequences
    # ------------------------------------------------------------------
    normal_errors = errors[seq_labels == 0]
    threshold = float(np.percentile(normal_errors, threshold_percentile))
    anomaly_mask = errors > threshold
    anomaly_idx = np.where(anomaly_mask)[0]

    # Build detailed output rows
    rows = []
    for idx in anomaly_idx:
        m = metadatas[idx]
        trig = m.get("trigger_event", {})
        row_dict = {
            "Sequence_ID": idx,
            "Original_Row_Index": trig.get("_row_idx", -1),
            "SystemTime": trig.get("SystemTime", ""),
            "Image": trig.get("Image", ""),
            "CommandLine": trig.get("CommandLine", ""),
            "User": trig.get("User", ""),
            "EventID": trig.get("EventID", 0),
            "ParentImage": m.get("parent", {}).get("Image", ""),
            "ParentCommandLine": m.get("parent", {}).get("CommandLine", ""),
            "GrandparentImage": m.get("gp", {}).get("Image", ""),
            "Active_Steps": m.get("active_steps", 20),
            "Reconstruction_Error": errors[idx],
            "Threshold": threshold,
            "Label": seq_labels[idx],
            "DestinationIp": trig.get("DestinationIp", ""),
            "DestinationPort": trig.get("DestinationPort", 0),
            "QueryName": trig.get("QueryName", ""),
            "Computer": trig.get("Computer", "") or m.get("current", {}).get("Computer", ""),
        }
        rows.append(row_dict)

    trigger_rows = pd.DataFrame(rows)
    trigger_rows["Source_Data_Path"] = os.path.abspath(paths.eval_data)

    os.makedirs(os.path.dirname(paths.detected_anomalies), exist_ok=True)
    trigger_rows.to_csv(paths.detected_anomalies, index=False)

    if verbose:
        print(f"\n[3/4] Detected {len(trigger_rows):,} anomalous sequences.")
        print(f"      Threshold (p{threshold_percentile}): {threshold:.6f}")
        print(f"      Saved -> {paths.detected_anomalies}")

    return trigger_rows
