"""Train the LSTM Autoencoder on normal-traffic data."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from anomaly_detection.config import HyperParams, Paths, DEFAULT_HP
from anomaly_detection.trainer import train


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LSTM Autoencoder")
    p.add_argument("--epochs",     type=int,   default=DEFAULT_HP.epochs)
    p.add_argument("--batch_size", type=int,   default=DEFAULT_HP.batch_size)
    p.add_argument("--seq_len",    type=int,   default=DEFAULT_HP.seq_len)
    p.add_argument("--lr",         type=float, default=DEFAULT_HP.lr)
    p.add_argument("--data",       type=str,   default=None, help="Đường dẫn file dữ liệu train CSV (mặc định lấy từ config)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    hp = HyperParams(
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
    )
    paths = Paths()
    if args.data:
        paths.train_data = args.data
    train(hp=hp, paths=paths)
