"""CLI for the CPLEX Python API single-period column-generation model."""

from __future__ import annotations

import argparse
from pathlib import Path

from .runner import run_cplex_api_column_generation


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Solve the single-period model with CPLEX Python API column "
            "generation."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets") / "single-period" / "small",
    )
    parser.add_argument("--dataset-name", default="dataset_9")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path("OutPut-CPLEX-API-CG")
            / "single-period"
            / "small"
            / "dataset_9"
        ),
    )
    parser.add_argument("--final-time-limit", type=float, default=60.0)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260408)
    parser.add_argument(
        "--seed-result-csv",
        type=Path,
        default=None,
        help="Optional feasible opt_result.csv used as an initial column set.",
    )
    args = parser.parse_args()
    summary = run_cplex_api_column_generation(
        args.data_dir,
        args.output_dir,
        args.dataset_name,
        final_time_limit=args.final_time_limit,
        max_iterations=args.max_iterations,
        seed_result_csv=args.seed_result_csv,
        seed=args.seed,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
