"""
visualization.py
================
Matplotlib helpers for training and evaluation visualisations.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Optional


def plot_loss(
    train_losses: List[float],
    val_losses: List[float],
    title: str,
    save_path: str,
) -> None:
    """
    Plot training and validation loss curves and save to *save_path*.

    Args:
        train_losses : Per-epoch training losses.
        val_losses   : Per-epoch validation losses.
        title        : Figure title.
        save_path    : Destination file path (PNG).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, color="#1f77b4", linewidth=2, label="Train Loss")
    ax.plot(epochs, val_losses, color="#d62728", linewidth=2, label="Val Loss")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=11)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_error_distribution(
    normal_errors: np.ndarray,
    anomalous_errors: np.ndarray,
    threshold: float,
    save_path: str,
) -> None:
    """
    Plot the reconstruction-error distributions for normal and anomalous
    sequences, with a vertical threshold line.

    Args:
        normal_errors   : MSE errors for normal sequences.
        anomalous_errors: MSE errors for anomalous sequences.
        threshold       : Detection threshold value.
        save_path       : Destination file path (PNG).
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.hist(normal_errors, bins=100, color="#1f77b4", alpha=0.6, label="Normal")
    ax.hist(anomalous_errors, bins=100, color="#d62728", alpha=0.6, label="Anomalous")
    ax.axvline(threshold, color="#ff7f0e", linewidth=2, linestyle="--", label=f"Threshold = {threshold:.4f}")

    ax.set_title("Reconstruction Error Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Mean Squared Error", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
