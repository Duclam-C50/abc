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
from .model import LSTMAutoencoder
from .preprocess import Preprocessor
from .dataset import make_sequences
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
    # 1. Load data
    # ------------------------------------------------------------------
    if verbose:
        print(f"[1/5] Loading data from: {paths.train_data}")
    df = pd.read_csv(paths.train_data, low_memory=False)

    split_idx = int(len(df) * (1 - hp.val_split))
    df_train, df_val = df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

    if verbose:
        print(f"      Train rows: {len(df_train):,}  |  Val rows: {len(df_val):,}")

    # ------------------------------------------------------------------
    # 2. Preprocess
    # ------------------------------------------------------------------
    if verbose:
        print(f"  Preprocessing (fit on train, apply to val) ...")
    pp = Preprocessor(FEATURE_COLUMNS)
    X_train = pp.fit_transform(df_train)
    X_val = pp.transform(df_val)
    pp.save(paths.encoders, paths.scaler)

    # ------------------------------------------------------------------
    # 3. Build sequences
    # ------------------------------------------------------------------
    if verbose:
        print(f"[3/5] Building sequences (seq_len={hp.seq_len}) ...")
    train_seqs = make_sequences(X_train, hp.seq_len)
    val_seqs = make_sequences(X_val, hp.seq_len)
    if verbose:
        print(f"      Train sequences: {len(train_seqs):,}  |  Val sequences: {len(val_seqs):,}")

    # ------------------------------------------------------------------
    # 4. Build model + compile
    # ------------------------------------------------------------------
    n_features = train_seqs.shape[2]
    if verbose:
        print(f"[4/5] Building model (features={n_features}, hidden={hp.hidden_size}, layers={hp.num_layers}) ...")

    model = LSTMAutoencoder(n_features=n_features, hidden_size=hp.hidden_size, num_layers=hp.num_layers)
    # Warm up so weights are initialised before compile
    model(keras.ops.convert_to_tensor(train_seqs[:1], dtype="float32"))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp.lr), loss=keras.losses.MeanSquaredError())

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
