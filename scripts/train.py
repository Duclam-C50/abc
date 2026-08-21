"""Train the LSTM Autoencoder on normal-traffic data."""

import argparse
from anomaly_detection.config import HyperParams, Paths, DEFAULT_HP
from anomaly_detection.trainer import train


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LSTM Autoencoder")
    p.add_argument("--epochs",     type=int,   default=DEFAULT_HP.epochs)
    p.add_argument("--batch_size", type=int,   default=DEFAULT_HP.batch_size)
    p.add_argument("--seq_len",    type=int,   default=DEFAULT_HP.seq_len)
    p.add_argument("--lr",         type=float, default=DEFAULT_HP.lr)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    hp = HyperParams(
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
    )
    train(hp=hp, paths=Paths())
