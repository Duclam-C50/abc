"""Generate dashboard_data.js for the web explanation UI."""

import argparse
from anomaly_detection.config import HyperParams, Paths, DEFAULT_HP
from anomaly_detection.explainer import generate_dashboard


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate dashboard data")
    p.add_argument("--limit",   type=int, default=30)
    p.add_argument("--steps",   type=int, default=50)
    p.add_argument("--seq_len", type=int, default=DEFAULT_HP.seq_len)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    hp = HyperParams(seq_len=args.seq_len)
    generate_dashboard(limit=args.limit, steps=args.steps, hp=hp, paths=Paths())
