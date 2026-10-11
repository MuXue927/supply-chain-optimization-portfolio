"""Runner for the official CPLEX Python API multi-period model."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict

from ..monolithic.InputDataMultiple import InputDataMultiple
from .model import MultiPeriodCplexApiMonolithic


def run_cplex_api_monolithic(
    input_dir: str | Path,
    output_dir: str | Path,
    dataset_name: str,
    *,
    time_limit: float = 60.0,
    mip_gap: float = 0.0,
    seed: int = 20260408,
) -> Dict[str, object]:
    """Solve one dataset with the CPLEX Python API and write result files."""
    random.seed(seed)
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    data = InputDataMultiple(
        str(input_path),
        str(output_path),
        dataset_name,
    )
    data.load()

    model = MultiPeriodCplexApiMonolithic(data)
    solution = model.solve(
        time_limit=time_limit,
        mip_gap=mip_gap,
        random_seed=seed,
        write_lp=output_path / "cplex_api_model.lp",
    )

    (output_path / "cplex_console.log").write_text(
        solution.stdout,
        encoding="utf-8",
    )
    summary = model.write_outputs(solution, output_path)
    summary.update(
        {
            "dataset_name": dataset_name,
            "time_limit": time_limit,
            "mip_gap": mip_gap,
            "seed": seed,
        }
    )

    # Keep the same JSON filename and core fields as the existing CLI model.
    (output_path / "cplex_result.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    api_summary = dict(summary)
    api_summary.update(
        {
            "solver_status": solution.solver_status,
            "solver_status_code": solution.solver_status_code,
            "best_bound": solution.best_bound,
            "solver_mip_gap": solution.mip_gap,
            "solver_log": "cplex_console.log",
        }
    )
    (output_path / "cplex_api_result.json").write_text(
        json.dumps(api_summary, indent=2),
        encoding="utf-8",
    )
    return api_summary
