"""Write model configuration files for the single-period project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write model_config.json and model_config_cg.json."
    )
    parser.add_argument(
        "--scale",
        choices=("small", "medium", "large"),
        default="small",
        help="Dataset scale whose parent directory receives the config files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Target config directory. Defaults to "
            "datasets/single-period/<scale>."
        ),
    )
    return parser


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {path}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir or (
        project_root / "datasets" / "single-period" / args.scale
    )

    write_json(
        output_dir / "model_config.json",
        {
            "solver_id": "GUROBI",
            "alpha": 2,
            "BIG_NUM": 99999,
            "gap_limit": 0,
            "time_limit": 180,
            "output_flag": 1,
        },
    )
    write_json(
        output_dir / "model_config_cg.json",
        {
            "solver_id": "GUROBI",
            "pun_factor1": 5000,
            "pun_factor2": 5000,
            "pun_factor3": 5000,
            "pattern_batch": 100,
            "pattern_num_shrink_to": 10000,
            "trigger_shrink_level": 20000,
            "gap_limit": 0,
            "time_limit": 180,
            "output_flag": 1,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
