"""CLI entry point for the CPLEX Python API multi-period model."""

from __future__ import annotations

import argparse
from pathlib import Path

from .runner import run_cplex_api_monolithic


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Solve the multi-period monolithic model with the official "
            "CPLEX Python API."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets") / "multiple-periods" / "small",
    )
    parser.add_argument("--dataset-name", default="dataset_1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(Path("OutPut-CPLEX-API") / "multiple-periods" / "small" / "dataset_1"),
    )
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--mip-gap", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260408)
    args = parser.parse_args()

    try:
        summary = run_cplex_api_monolithic(
            args.data_dir,
            args.output_dir,
            args.dataset_name,
            time_limit=args.time_limit,
            mip_gap=args.mip_gap,
            seed=args.seed,
        )
    except ImportError as exc:
        parser.exit(2, f"error: {exc}\n")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
