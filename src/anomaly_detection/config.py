"""
config.py
=========
Centralised project configuration.

All hyperparameters and file paths are stored in frozen dataclasses so they can
be safely passed around, overridden per-script, and serialised to JSON.
"""

import os

# ---------------------------------------------------------------------------
# Backend must be set BEFORE any Keras import anywhere in the codebase.
# Setting it here (imported by every module) guarantees correct ordering.
# ---------------------------------------------------------------------------
os.environ.setdefault("KERAS_BACKEND", "torch")

from dataclasses import dataclass, field, asdict
from typing import List

# ---------------------------------------------------------------------------
# Root directory (parent of src/)
# ---------------------------------------------------------------------------
ROOT_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Feature columns
# ---------------------------------------------------------------------------
FEATURE_COLUMNS: List[str] = [
    "EventID", "Image", "CommandLine", "User",
    "ParentImage", "ParentCommandLine", "TargetFilename",
    "TargetObject", "ImageLoaded", "Signed", "Signature", "SignatureStatus",
    "SourceImage", "TargetImage", "QueryName", "QueryStatus", "Computer"
]


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
@dataclass
class HyperParams:
    """Training and model hyperparameters."""

    seq_len: int = 15
    batch_size: int = 64
    epochs: int = 20
    lr: float = 1e-3
    hidden_size: int = 32
    num_layers: int = 1
    val_split: float = 0.2

    # Grid-search candidates (used only by the grid-search script)
    grid_seq_lens: List[int] = field(default_factory=lambda: [10, 20, 30])
    grid_batch_sizes: List[int] = field(default_factory=lambda: [64, 128])
    grid_epochs: List[int] = field(default_factory=lambda: [20, 30, 40])

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
@dataclass
class Paths:
    """All project file paths derived from ROOT_DIR."""

    root: str = field(default_factory=lambda: ROOT_DIR)

    # Artifacts
    model: str = field(default_factory=lambda: os.path.join(ROOT_DIR, "artifacts", "models", "lstm_autoencoder.keras"))
    scaler: str = field(default_factory=lambda: os.path.join(ROOT_DIR, "artifacts", "models", "scaler.pkl"))
    encoders: str = field(default_factory=lambda: os.path.join(ROOT_DIR, "artifacts", "models", "encoders.pkl"))

    # Training results
    train_history: str = field(default_factory=lambda: os.path.join(ROOT_DIR, "artifacts", "results", "train_history.json"))
    train_plot: str = field(default_factory=lambda: os.path.join(ROOT_DIR, "artifacts", "results", "train_loss.png"))

    # Evaluation results
    eval_report: str = field(default_factory=lambda: os.path.join(ROOT_DIR, "artifacts", "results", "eval_report.json"))

    # Anomaly detection results
    detected_anomalies: str = field(default_factory=lambda: os.path.join(ROOT_DIR, "artifacts", "results", "detected_anomalies.csv"))

    # Explanation results
    ig_dir: str = field(default_factory=lambda: os.path.join(ROOT_DIR, "artifacts", "results", "ig"))
    dashboard_js: str = field(default_factory=lambda: os.path.join(ROOT_DIR, "artifacts", "results", "ig", "dashboard_data.js"))

    # Data
    train_data: str = field(default_factory=lambda: os.path.join(ROOT_DIR, "data", "train", "log_csv", "LMD-2023-Normal.csv"))
    eval_data: str = field(default_factory=lambda: os.path.join(ROOT_DIR, "data", "evaluate", "log_csv", "LMD-2023.csv"))


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def get_device() -> str:
    """Return 'cuda' if a CUDA-capable GPU is available, else 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ---------------------------------------------------------------------------
# Default singletons (convenient for scripts that do not need overrides)
# ---------------------------------------------------------------------------
DEFAULT_HP = HyperParams()
DEFAULT_PATHS = Paths()
