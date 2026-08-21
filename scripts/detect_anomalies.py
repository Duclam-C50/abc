"""Detect anomalous sequences and export a CSV report."""

import argparse
from anomaly_detection.config import HyperParams, Paths, DEFAULT_HP
from anomaly_detection.detector import detect


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detect anomalies with LSTM Autoencoder")
    p.add_argument("--batch_size",  type=int, default=DEFAULT_HP.batch_size)
    p.add_argument("--seq_len",     type=int, default=DEFAULT_HP.seq_len)
    p.add_argument("--threshold_p", type=int, default=95, help="Percentile for anomaly threshold")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    hp = HyperParams(batch_size=args.batch_size, seq_len=args.seq_len)
    detect(hp=hp, paths=Paths(), threshold_percentile=args.threshold_p)
