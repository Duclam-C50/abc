"""Generate dashboard_data.js for the web explanation UI."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from anomaly_detection.config import HyperParams, Paths, DEFAULT_HP
from anomaly_detection.explainer import generate_dashboard


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate dashboard data")
    p.add_argument("--limit",   type=int, default=100, help="Số lượng anomaly xuất ra (0 = tất cả)")
    p.add_argument("--steps",   type=int, default=15, help="Số bước tích phân Integrated Gradients")
    p.add_argument("--seq_len", type=int, default=DEFAULT_HP.seq_len)
    p.add_argument("--stride",  type=int, default=3, help="Stride for sliding window of actions")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    hp = HyperParams(seq_len=args.seq_len)
    generate_dashboard(limit=args.limit, steps=args.steps, hp=hp, paths=Paths(), stride=args.stride)
