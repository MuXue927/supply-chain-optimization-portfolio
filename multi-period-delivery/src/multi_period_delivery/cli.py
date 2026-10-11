"""Command-line interface for the multi-period delivery package."""

from __future__ import annotations

import argparse
from pathlib import Path

from .alns_config import default_config as ALNSConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="multi-period-delivery",
        description=(
            "Run the production-driven multi-period delivery optimizer. "
            "ALNS-B is the base configuration, ALNS-T enables the parameter "
            "tuner, ALNS-ML enables ML operator selection, and ALNS-F enables "
            "both mechanisms."
        ),
    )
    parser.add_argument(
        "--scale",
        choices=("small", "medium", "large"),
        default="small",
        help="Dataset scale. The included sample instance is under small.",
    )
    parser.add_argument(
        "--dataset-id",
        type=int,
        default=1,
        help="Dataset number, for example 1 for dataset_1.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Parent directory containing dataset_* folders. Defaults to "
            "datasets/multiple-periods/<scale> under the current directory."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output root. Defaults to "
            "OutPut-ALNS/multiple-periods/<scale> under the current directory."
        ),
    )
    parser.add_argument(
        "--max-runtime",
        type=int,
        default=60,
        help="Maximum runtime in seconds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=15926535,
        help="Random seed.",
    )
    parser.add_argument(
        "--enable-param-tuner",
        action="store_true",
        help="Enable online parameter tuning (ALNS-T or ALNS-F).",
    )
    parser.add_argument(
        "--enable-ml-operators",
        action="store_true",
        help="Enable machine-learning operator selection (ALNS-ML or ALNS-F).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Directory for the ALNS log file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scale = args.scale
    data_dir = args.data_dir or (
        Path.cwd() / "datasets" / "multiple-periods" / scale
    )
    output_dir = args.output_dir or (
        Path.cwd() / "OutPut-ALNS" / "multiple-periods" / scale
    )

    ALNSConfig.update_from_dict(
        {
            "DATASET_TYPE": scale,
            "DATASET_IDX": args.dataset_id,
            "MAX_RUNTIME": args.max_runtime,
            "SEED": args.seed,
            "ENABLE_PARAM_TUNER": args.enable_param_tuner,
            "ENABLE_ML_OPERATORS": args.enable_ml_operators,
            "LOG_DIR": str(args.log_dir),
        }
    )

    # Import after updating the shared config. multi_period_delivery.alns_optimizer reads
    # selected configuration values at import time.
    from .main import run_model

    run_model(str(data_dir), str(output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
