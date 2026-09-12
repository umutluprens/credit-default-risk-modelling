"""Command-line entry point for credit default risk modelling."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from credit_risk.modelling import generate_demo_data, load_credit_data, run_modelling


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Path to cs-training.csv")
    source.add_argument("--demo", action="store_true", help="Run with synthetic data")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--threshold", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = generate_demo_data() if args.demo else load_credit_data(args.input)
    metrics = run_modelling(frame, args.output, threshold=args.threshold)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()

