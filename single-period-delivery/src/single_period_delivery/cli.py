"""Command-line interface for the single-period delivery package."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="single-period-delivery",
        description="Run algorithms for the single-period delivery problem.",
    )
    subparsers = parser.add_subparsers(dest="algorithm", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--data-dir",
            type=Path,
            default=Path("datasets") / "single-period" / "small",
            help="Directory containing dataset_* instance directories.",
        )
        subparser.add_argument(
            "--dataset-name",
            default="dataset_9",
            help="Dataset directory name, for example dataset_9.",
        )

    monolithic = subparsers.add_parser(
        "monolithic",
        help="Run the direct integer-programming model.",
    )
    add_common(monolithic)
    monolithic.add_argument(
        "--output-dir",
        type=Path,
        default=Path("OutPut") / "single-period" / "small",
        help="Directory for solver output files.",
    )

    heuristic = subparsers.add_parser(
        "heuristic",
        help="Run the two-stage heuristic.",
    )
    add_common(heuristic)
    heuristic.add_argument(
        "--output-dir",
        type=Path,
        default=Path("OutPut-Heuristic") / "single-period" / "small",
        help="Directory for heuristic output files.",
    )
    heuristic.add_argument(
        "--experiments",
        type=int,
        default=1,
        help="Number of heuristic repetitions.",
    )

    column_generation = subparsers.add_parser(
        "column-generation",
        help="Run the column-generation algorithm.",
    )
    add_common(column_generation)
    column_generation.add_argument(
        "--output-dir",
        type=Path,
        default=Path("OutPut-CG") / "single-period" / "small",
        help="Directory for column-generation output files.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.algorithm == "monolithic":
        from .monolithic.optLogicSingle import run_model

        run_model(
            str(args.data_dir),
            str(args.output_dir),
            dataset_name=args.dataset_name,
        )
    elif args.algorithm == "heuristic":
        from .heuristic.main import run_model

        run_model(
            str(args.data_dir),
            str(args.output_dir),
            exp_nums=args.experiments,
            dataset_name=args.dataset_name,
        )
    elif args.algorithm == "column-generation":
        from .column_generation.OPTLogicCG import run_model

        run_model(
            str(args.data_dir),
            str(args.output_dir),
            dataset_name=args.dataset_name,
        )
    else:  # pragma: no cover - argparse enforces this branch.
        raise ValueError(f"Unknown algorithm: {args.algorithm}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
