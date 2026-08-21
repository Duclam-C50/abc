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
from .model import LSTMAutoencoder
from .preprocess import Preprocessor
from .dataset import make_sequences, make_seq_labels
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


def evaluate(hp: HyperParams = HyperParams(), paths: Paths = Paths(), verbose: bool = True) -> EvalResult:
    """
    Evaluate the saved model on labelled test data.

    Args:
        hp      : Hyperparameter configuration (seq_len, batch_size).
        paths   : File-path configuration.
        verbose : Print results to stdout when True.

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
    if verbose:
        print(f"[1/4] Loading test data from: {paths.eval_data}")
    df = pd.read_csv(paths.eval_data, low_memory=False)
    labels = df["Label"].values

    pp = Preprocessor.load(FEATURE_COLUMNS, paths.encoders, paths.scaler)
    X = pp.transform(df)

    # ------------------------------------------------------------------
    # 2. Build sequences + sequence labels
    # ------------------------------------------------------------------
    if verbose:
        print(f"[2/4] Building sequences (seq_len={hp.seq_len}) ...")
    sequences = make_sequences(X, hp.seq_len)
    seq_labels = make_seq_labels(labels, hp.seq_len)
    if verbose:
        print(f"      Sequences: {len(sequences):,}  (normal={np.sum(seq_labels==0):,}, anomalous={np.sum(seq_labels==1):,})")

    # ------------------------------------------------------------------
    # 3. Reconstruct + compute errors
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
    # 4. Metrics
    # ------------------------------------------------------------------
    normal_err = errors[seq_labels == 0]
    anomalous_err = errors[seq_labels == 1]
    threshold = np.percentile(normal_err, 95)
    predictions = (errors > threshold).astype(int)
    auc = roc_auc_score(seq_labels, errors)
    report = classification_report(seq_labels, predictions, target_names=["Normal", "Anomalous"])
    cm = confusion_matrix(seq_labels, predictions)

    if verbose:
        print("\n[4/4] Results")
        print(f"  Normal   -- count: {len(normal_err):,}  mean error: {np.mean(normal_err):.6f}")
        print(f"  Anomalous -- count: {len(anomalous_err):,}  mean error: {np.mean(anomalous_err):.6f}")
        print(f"  ROC AUC   : {auc:.4f}")
        print(f"  Threshold : {threshold:.6f}  (95th pct of normal errors)")
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
            {"auc": auc, "threshold": threshold, "n_normal": int(len(normal_err)), "n_anomalous": int(len(anomalous_err))},
            f,
            indent=2,
        )

    return EvalResult(
        auc=auc,
        threshold=threshold,
        n_normal=len(normal_err),
        n_anomalous=len(anomalous_err),
        report=report,
        confusion=cm,
    )


def _require_file(path: str, msg: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{msg}\n  Expected: {path}")
