"""Detect anomalous sequences and export a CSV report."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from anomaly_detection.config import HyperParams, Paths, DEFAULT_HP
from anomaly_detection.detector import detect


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detect anomalies with LSTM Autoencoder")
    p.add_argument("--batch_size",  type=int,   default=DEFAULT_HP.batch_size)
    p.add_argument("--seq_len",     type=int,   default=DEFAULT_HP.seq_len)
    p.add_argument("--stride",      type=int,   default=3, help="Stride for sliding window of actions")
    p.add_argument("--threshold_p", type=float, default=80.0, help="Percentile for anomaly threshold (mặc định 80)")
    p.add_argument("--data",        type=str,   default=None, help="Đường dẫn file dữ liệu evaluate CSV (mặc định lấy từ config)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    hp = HyperParams(batch_size=args.batch_size, seq_len=args.seq_len)
    paths = Paths()
    if args.data:
        paths.eval_data = args.data
    detect(hp=hp, paths=paths, threshold_percentile=args.threshold_p, stride=args.stride)
