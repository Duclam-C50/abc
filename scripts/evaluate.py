"""Evaluate the saved model on labelled test data."""

import argparse
from anomaly_detection.config import HyperParams, Paths, DEFAULT_HP
from anomaly_detection.evaluator import evaluate


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate LSTM Autoencoder")
    p.add_argument("--batch_size", type=int, default=DEFAULT_HP.batch_size)
    p.add_argument("--seq_len",    type=int, default=DEFAULT_HP.seq_len)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    hp = HyperParams(batch_size=args.batch_size, seq_len=args.seq_len)
    evaluate(hp=hp, paths=Paths())
