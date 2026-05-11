"""CLI: run the AI pipeline against a CSV.

Usage:
    python -m scripts.run_pipeline --csv data/tickets.csv
"""
from __future__ import annotations

import argparse
import logging

from app.pipeline.runner import process_csv


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/tickets.csv")
    p.add_argument("--no-cluster", action="store_true", help="Skip the KMeans clustering stage")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=args.log_level)
    summary = process_csv(args.csv, do_cluster=not args.no_cluster)
    print(summary)


if __name__ == "__main__":
    main()
