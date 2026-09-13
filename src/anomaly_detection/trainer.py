"""
trainer.py
==========
Training pipeline for the LSTM Autoencoder.

Exposes a single public function :func:`train` that accepts configuration
objects and returns a :class:`TrainResult` dataclass.
"""

import os
import json
import time
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
import keras
import torch
from torch.utils.data import TensorDataset, DataLoader

from .config import HyperParams, Paths, FEATURE_COLUMNS, get_device
from .model import LSTMAutoencoder, EventSpecificMaskedMSE
from .preprocess import Preprocessor
from .dataset import build_tree_dataset, vectorize_tree_sequences
from .visualization import plot_loss


@dataclass
class TrainResult:
    """Outcome of a training run."""

    best_val_loss: float
    train_losses: List[float]
    val_losses: List[float]
    duration_seconds: float
    model_path: str


def train(hp: HyperParams = HyperParams(), paths: Paths = Paths(), verbose: bool = True) -> TrainResult:
    """
    Train the LSTM Autoencoder on normal-traffic data.

    Args:
        hp      : Hyperparameter configuration.
        paths   : File-path configuration.
        verbose : Print progress to stdout when True.

    Returns:
        :class:`TrainResult` with losses and metadata.
    """
    device = get_device()

    if verbose:
        print("=" * 60)
        print("  LSTM AUTOENCODER -- TRAINING PIPELINE (Keras)")
        print("=" * 60)
        print(f"  Backend : {keras.backend.backend()}")
        print(f"  Device  : {device}")
        print(f"  SEQ_LEN={hp.seq_len}  BATCH={hp.batch_size}  EPOCHS={hp.epochs}  LR={hp.lr}")
        print()

    os.makedirs(os.path.dirname(paths.model), exist_ok=True)
    os.makedirs(os.path.dirname(paths.train_history), exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data & generate 20-step tree sequences
    # ------------------------------------------------------------------
    if verbose:
        print(f"[1/4] Generating 20-step hierarchical tree sequences from: {paths.train_data}")
    raw_seqs, _, _ = build_tree_dataset(paths.train_data, stride=3)

    split_idx = int(len(raw_seqs) * (1 - hp.val_split))
    train_raw, val_raw = raw_seqs[:split_idx], raw_seqs[split_idx:]

    if verbose:
        print(f"      Total tree sequences: {len(raw_seqs):,}  (Train: {len(train_raw):,}, Val: {len(val_raw):,})")

    # ------------------------------------------------------------------
    # 2. Preprocess & Vectorize
    # ------------------------------------------------------------------
    if verbose:
        print(f"[2/4] Preprocessing & vectorizing 15-step process-action sequences ...")
    pp = Preprocessor(FEATURE_COLUMNS)
    train_seqs = vectorize_tree_sequences(train_raw, pp, fit=True, seq_len=hp.seq_len)
    val_seqs = vectorize_tree_sequences(val_raw, pp, fit=False, seq_len=hp.seq_len)
    pp.save(paths.encoders, paths.scaler)

    # ------------------------------------------------------------------
    # 4. Build model + compile
    # ------------------------------------------------------------------
    n_features = train_seqs.shape[2]
    if verbose:
        print(f"[4/5] Building model (features={n_features}, hidden={hp.hidden_size}, layers={hp.num_layers}) ...")

    model = LSTMAutoencoder(n_features=n_features, hidden_size=hp.hidden_size, num_layers=hp.num_layers)
    # Warm up so weights are initialised before compile
    model(keras.ops.convert_to_tensor(train_seqs[:1], dtype="float32"))
    
    # Retrieve the event-specific feature mask from preprocessor
    feature_names = list(pp._scaler.feature_names_in_)
    event_col_indices = [feature_names.index(f"EventID_{eid}") for eid in pp.event_ids]
    mask_matrix = pp.get_event_feature_mask().tolist()
    
    custom_loss = EventSpecificMaskedMSE(mask_matrix=mask_matrix, event_col_indices=event_col_indices)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp.lr, clipnorm=1.0), loss=custom_loss)

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    if verbose:
        print(f"[5/5] Training for {hp.epochs} epoch(s) ...")

    train_loader = DataLoader(
        TensorDataset(torch.tensor(train_seqs, dtype=torch.float32)),
        batch_size=hp.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(val_seqs, dtype=torch.float32)),
        batch_size=hp.batch_size,
        shuffle=False,
    )

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    start = time.time()

    for epoch in range(hp.epochs):
        # --- train ---
        epoch_loss = sum(
            _extract_loss(model.train_on_batch(b[0].numpy(), b[0].numpy()))
            for b in train_loader
        )
        avg_train = epoch_loss / len(train_loader)

        # --- validate ---
        val_loss = sum(
            _extract_loss(model.test_on_batch(b[0].numpy(), b[0].numpy()))
            for b in val_loader
        )
        avg_val = val_loss / len(val_loader)

        train_losses.append(avg_train)
        val_losses.append(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            model.save(paths.model)

        if verbose:
            marker = " [saved]" if avg_val == best_val_loss else ""
            print(f"  Epoch [{epoch+1:02d}/{hp.epochs:02d}]  train={avg_train:.6f}  val={avg_val:.6f}{marker}")

    duration = time.time() - start
    
    # Unconditionally save the final trained model at the very end
    model.save(paths.model)

    # ------------------------------------------------------------------
    # 6. Save history + plot
    # ------------------------------------------------------------------
    history = {
        "params": hp.to_dict(),
        "train_loss": train_losses,
        "val_loss": val_losses,
        "best_val_loss": best_val_loss,
        "duration_seconds": duration,
    }
    with open(paths.train_history, "w") as f:
        json.dump(history, f, indent=2)

    plot_loss(
        train_losses=train_losses,
        val_losses=val_losses,
        title=f"LSTM Autoencoder Loss  (seq={hp.seq_len}, batch={hp.batch_size})",
        save_path=paths.train_plot,
    )

    if verbose:
        print(f"\n  Done in {duration:.1f}s  |  Best val loss: {best_val_loss:.6f}")
        print(f"  Model saved -> {paths.model}")

    return TrainResult(
        best_val_loss=best_val_loss,
        train_losses=train_losses,
        val_losses=val_losses,
        duration_seconds=duration,
        model_path=paths.model,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _extract_loss(raw) -> float:
    """Normalise the various return types of train_on_batch / test_on_batch."""
    if isinstance(raw, dict):
        return float(raw["loss"])
    if isinstance(raw, (list, tuple, np.ndarray)):
        return float(raw[0])
    return float(raw)
