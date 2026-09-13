"""Compute Integrated Gradients explanations for top-N anomalies."""

import argparse
from anomaly_detection.config import HyperParams, Paths, DEFAULT_HP
from anomaly_detection.explainer import explain


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Explain anomalies with Integrated Gradients")
    p.add_argument("--limit",   type=int, default=-1, help="Number of anomalies to explain (-1 = all)")
    p.add_argument("--steps",   type=int, default=50, help="IG integration steps")
    p.add_argument("--seq_len", type=int, default=DEFAULT_HP.seq_len)
    p.add_argument("--data",    type=str, default=None, help="Đường dẫn file dữ liệu evaluate CSV (mặc định lấy từ config)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    hp = HyperParams(seq_len=args.seq_len)
    paths = Paths()
    if args.data:
        paths.eval_data = args.data
    explain(limit=args.limit, steps=args.steps, hp=hp, paths=paths)
