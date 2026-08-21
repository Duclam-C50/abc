"""
dataset.py
==========
Utilities for constructing sliding-window sequences from time-series data.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def make_sequences(data: np.ndarray, seq_len: int) -> np.ndarray:
    """
    Build overlapping fixed-length sequences from a 2-D data array.

    Uses ``numpy.lib.stride_tricks.sliding_window_view`` (O(1) memory copy)
    instead of a Python loop for maximum efficiency.

    Args:
        data    : Array of shape (n_timesteps, n_features).
        seq_len : Length of each sequence window.

    Returns:
        Array of shape (n_timesteps - seq_len, seq_len, n_features).
    """
    if len(data) <= seq_len:
        raise ValueError(
            f"data length ({len(data)}) must be greater than seq_len ({seq_len})."
        )
    # sliding_window_view returns shape (N - seq_len + 1, 1, seq_len, n_features).
    # We drop the last window ([:-1]) so the count is N - seq_len, matching
    # the sequence-label array produced by make_seq_labels.
    windows = sliding_window_view(data, window_shape=(seq_len, data.shape[1]))
    return windows[:-1, 0, :, :]  # (N - seq_len, seq_len, n_features)


def make_seq_labels(labels: np.ndarray, seq_len: int) -> np.ndarray:
    """
    Compute sequence-level binary labels.

    A sequence is labelled **anomalous (1)** if *any* event within the window
    has a non-zero label (i.e., at least one malicious event is present).

    Args:
        labels  : 1-D integer array of per-event labels, shape (n_timesteps,).
        seq_len : Sequence window length.

    Returns:
        1-D binary array of shape (n_timesteps - seq_len,).
    """
    windows = sliding_window_view(labels, seq_len)  # (n_windows, seq_len)
    return np.any(windows[:-1] != 0, axis=1).astype(np.int32)
