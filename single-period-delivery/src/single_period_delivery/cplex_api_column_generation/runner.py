"""Runner for the CPLEX Python API single-period column-generation model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from ..column_generation.InputDataCG import InputDataCG
from .model import CplexApiColumnGeneration


def run_cplex_api_column_generation(
    input_dir: str | Path,
    output_dir: str | Path,
    dataset_name: str,
    *,
    final_time_limit: float = 60.0,
    max_iterations: int = 100,
    seed_result_csv: str | Path | None = None,
    seed: int = 20260408,
) -> Dict[str, object]:
    """Run the CPLEX Python API column-generation workflow for one dataset."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    work_dir = output_path / "_cplex_work"
    data = InputDataCG(str(input_path), str(output_path), dataset_name)
    data.load()

    algorithm = CplexApiColumnGeneration(
        data,
        max_iterations=max_iterations,
        random_seed=seed,
    )
    if seed_result_csv is not None:
        algorithm.initialize_patterns()
        algorithm.seed_patterns_from_result_csv(seed_result_csv)
    result = algorithm.run(
        work_dir,
        final_time_limit=final_time_limit,
    )
    summary = algorithm.write_outputs(result, output_path)
    summary.update(
        {
            "dataset_name": dataset_name,
            "final_time_limit": final_time_limit,
            "max_iterations": max_iterations,
            "alpha": algorithm.alpha,
            "seed": seed,
            "seed_result_csv": str(seed_result_csv)
            if seed_result_csv is not None
            else None,
        }
    )
    (output_path / "cplex_api_cg_result.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary
