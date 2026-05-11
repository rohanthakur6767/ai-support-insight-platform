"""CLI: generate the synthetic dataset.

Usage:
    python -m scripts.generate_data --n 5000 --out data/tickets.csv
"""
from __future__ import annotations

import argparse

from app.data.synthesize import generate_dataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/tickets.csv")
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    path = generate_dataset(args.out, n=args.n, seed=args.seed)
    print(f"Wrote {args.n} tickets to {path}")


if __name__ == "__main__":
    main()
