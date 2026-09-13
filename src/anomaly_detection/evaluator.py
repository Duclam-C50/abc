"""
evaluator.py
============
Model evaluation pipeline.

Computes per-sequence reconstruction errors, selects an anomaly threshold
(95th percentile of normal errors), and reports classification metrics.
"""

import json
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import keras
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

from .config import HyperParams, Paths, FEATURE_COLUMNS, get_device
from .model import LSTMAutoencoder, EventSpecificMaskedMSE
from .preprocess import Preprocessor
from .dataset import build_tree_dataset, vectorize_tree_sequences
from .visualization import plot_error_distribution


@dataclass
class EvalResult:
    """Outcome of an evaluation run."""

    auc: float
    threshold: float
    n_normal: int
    n_anomalous: int
    report: str
    confusion: np.ndarray
    thresholds: dict = None


def evaluate(
    hp: HyperParams = HyperParams(),
    paths: Paths = Paths(),
    verbose: bool = True,
    stride: int = 3,
    threshold_percentile: float = 80.0,
) -> EvalResult:
    """
    Evaluate the saved model on labelled test data.

    Args:
        hp                   : Hyperparameter configuration (seq_len, batch_size).
        paths                : File-path configuration.
        verbose              : Print results to stdout when True.
        stride               : Action sequence sliding stride.
        threshold_percentile : Percentile of normal errors used as anomaly threshold.

    Returns:
        :class:`EvalResult` dataclass.
    """
    device = get_device()

    if verbose:
        print("=" * 60)
        print("  LSTM AUTOENCODER -- EVALUATION PIPELINE (Keras)")
        print("=" * 60)
        print(f"  Device: {device}")
        print()

    _require_file(paths.model, "Model not found. Run train.py first.")

    # ------------------------------------------------------------------
    # 1. Load & preprocess test data
    # ------------------------------------------------------------------
    # 1. Load & vectorize test data (15-step tree sequences)
    # ------------------------------------------------------------------
    if verbose:
        print(f"[1/4] Generating 15-step process-action sequences from: {paths.eval_data} (stride={stride})")
    pp = Preprocessor.load(FEATURE_COLUMNS, paths.encoders, paths.scaler)
    raw_seqs, seq_labels, metadatas = build_tree_dataset(paths.eval_data, stride=stride)
    sequences = vectorize_tree_sequences(raw_seqs, pp, fit=False, seq_len=hp.seq_len)

    if verbose:
        print(f"      Sequences: {len(sequences):,}  (normal={np.sum(seq_labels==0):,}, anomalous={np.sum(seq_labels==1):,})")

    # ------------------------------------------------------------------
    # 2. Reconstruct + compute errors
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
    # one_hot_events: shape (N, seq_len, num_event_types)
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
    # 3. Metrics (Tiered Thresholds by Process Depth: Bậc 1, 2, 3+)
    # ------------------------------------------------------------------
    # Map each sequence to depth tier: 1, 2, or 3 (3 for depth >= 3)
    # ------------------------------------------------------------------
    # 3. Metrics (Unified Anomaly Threshold at 95th Percentile)
    # ------------------------------------------------------------------
    normal_err = errors[seq_labels == 0]
    anomalous_err = errors[seq_labels >= 1]
    
    threshold = float(np.percentile(normal_err, threshold_percentile))
    predictions = (errors > threshold).astype(int)
    binary_labels = (seq_labels >= 1).astype(int)

    auc = roc_auc_score(binary_labels, errors)
    report = classification_report(binary_labels, predictions, target_names=["Normal", "Anomalous"])
    cm = confusion_matrix(binary_labels, predictions)

    if verbose:
        print(f"\n[3/4] Results (Anomaly Threshold at p{int(threshold_percentile)})")
        print(f"  Normal    -- count: {len(normal_err):,}  mean error: {np.mean(normal_err):.6f}")
        print(f"  Anomalous -- count: {len(anomalous_err):,}  mean error: {np.mean(anomalous_err):.6f}")
        print(f"  ROC AUC   : {auc:.4f}")
        print(f"  Threshold (p{int(threshold_percentile)}) : {threshold:.6f}")
        detected_cnt = np.sum((predictions == 1) & (binary_labels == 1))
        print(f"  Detected  : {detected_cnt}/{len(anomalous_err)} ({detected_cnt/len(anomalous_err)*100:.1f}%)")
        print("\n  Classification Report:")
        print(report)
        print("  Confusion Matrix:")
        print(cm)

    # Save error-distribution plot
    dist_plot_path = paths.eval_report.replace(".json", "_error_dist.png")
    plot_error_distribution(normal_err, anomalous_err, threshold, save_path=dist_plot_path)

    # Save JSON report
    os.makedirs(os.path.dirname(paths.eval_report), exist_ok=True)
    with open(paths.eval_report, "w") as f:
        json.dump(
            {
                "auc": auc,
                "threshold": threshold,
                "global_threshold": threshold,
                "n_normal": int(len(normal_err)),
                "n_anomalous": int(len(anomalous_err))
            },
            f,
            indent=2,
        )

    return EvalResult(
        auc=auc,
        threshold=threshold,
        thresholds={"global": threshold},
        n_normal=len(normal_err),
        n_anomalous=len(anomalous_err),
        report=report,
        confusion=cm,
    )


def _require_file(path: str, msg: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{msg}\n  Expected: {path}")
