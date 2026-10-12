"""Runner for the CPLEX Python API single-period monolithic model."""

from __future__ import annotations

import json
import random
from pathlib import Path

from ..monolithic.InputDataSingle import InputDataSingle
from .model import SinglePeriodCplexApiMonolithic


def run_cplex_api_monolithic(
    input_dir: str | Path,
    output_dir: str | Path,
    dataset_name: str,
    *,
    time_limit: float = 60.0,
    mip_gap: float = 0.0,
    seed: int = 20260408,
) -> dict[str, object]:
    """Solve one single-period dataset with the CPLEX Python API."""
    random.seed(seed)
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    work_dir = output_path / "_cplex_work"

    data = InputDataSingle(str(input_path), str(output_path), dataset_name)
    data.load()
    model = SinglePeriodCplexApiMonolithic(data)
    solution = model.solve(
        work_dir,
        time_limit=time_limit,
        mip_gap=mip_gap,
        random_seed=seed,
    )
    (output_path / "cplex_api_console.log").write_text(
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
            "solver_status": solution.solver_status,
            "solver_status_code": solution.solver_status_code,
            "best_bound": solution.best_bound,
            "solver_mip_gap": solution.mip_gap,
        }
    )
    (output_path / "cplex_api_result.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary
