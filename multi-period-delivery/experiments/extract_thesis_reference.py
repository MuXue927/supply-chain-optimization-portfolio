#!/usr/bin/env python3
"""Re-score the archived thesis ALNS plans under the corrected objective.

The thesis code set ``param_pun_objective`` to 0 by default, so its reported
``best_actual_objective`` was the **fixed vehicle cost only** and silently
omitted the minimum-load penalty. The fixed ALNS in this repository includes
that penalty (``alpha * vehicle_cost`` for every used vehicle below its
minimum shipment quantity), so the numbers are only comparable once the
thesis plans are re-scored under the same metric.

This script reads each archived thesis run's ``opt_summary.csv`` plan, looks up
the vehicle costs and minimum loads from the dataset's ``vehicle.csv``, and
writes both the penalty-free and the corrected objective for the same five
instances used by the portfolio experiments.

The archived results are outside the repository, so the output CSV is
committed and this script is kept for provenance. Point ``--thesis-results``
and ``--thesis-datasets`` at the archived thesis tree to re-run it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

INSTANCES: Dict[str, List[int]] = {
    "small": [12, 30],
    "medium": [13, 26],
    "large": [3],
}
VARIANTS = ["basic", "tuner", "ml", "full"]
BUDGETS = {"small": 180, "medium": 360, "large": 720}
RUN_DIRS = {
    "small": "small_3min_20260408_173620",
    "medium": "medium_6min_20260408_173620",
    "large": "large_12min_20260408_173620",
}


def vehicle_tables(dataset_dir: Path) -> tuple[Dict[str, float], Dict[str, float]]:
    veh = pd.read_csv(dataset_dir / "vehicle.csv")
    costs = dict(zip(veh["vehicle_type"], veh["cost_to_use"]))
    min_loads = dict(zip(veh["vehicle_type"], veh["min_standard_size"]))
    return costs, min_loads


def rescore_plan(
    plan_path: Path, costs: Dict[str, float], min_loads: Dict[str, float], alpha: float
) -> tuple[float, float, int]:
    df = pd.read_csv(plan_path)
    fixed = 0.0
    penalty = 0.0
    under = 0
    for row in df.itertuples():
        cost = float(costs[row.vehicle_type])
        fixed += cost
        if float(row.occupied_size) < float(min_loads[row.vehicle_type]):
            under += 1
            penalty += alpha * cost
    return round(fixed, 4), round(fixed + penalty, 4), under


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thesis-results", type=Path, required=True)
    parser.add_argument("--thesis-datasets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for scale, dataset_ids in INSTANCES.items():
        for dataset_id in dataset_ids:
            dataset_dir = args.thesis_datasets / scale / f"dataset_{dataset_id}"
            costs, min_loads = vehicle_tables(dataset_dir)
            config = json.loads(
                (dataset_dir.parent / "model_config_alns.json").read_text(encoding="utf-8")
            )
            alpha = float(config.get("alpha", 2.0))
            for variant in VARIANTS:
                budget = BUDGETS[scale]
                run_dir = (
                    args.thesis_results
                    / RUN_DIRS[scale]
                    / f"dataset_{dataset_id}__{variant}__seed0__t{budget}"
                )
                plan = run_dir / f"dataset_{dataset_id}" / "opt_summary.csv"
                if not plan.exists():
                    print(f"missing: {plan}")
                    continue
                fixed, corrected, under = rescore_plan(plan, costs, min_loads, alpha)
                rows.append(
                    {
                        "scale": scale,
                        "dataset_id": dataset_id,
                        "variant": variant,
                        "seed": 0,
                        "time_budget_s": budget,
                        "thesis_fixed_cost": fixed,
                        "thesis_final_objective": corrected,
                        "thesis_underloaded_vehicles": under,
                    }
                )

    out = pd.DataFrame(rows).sort_values(["scale", "dataset_id", "variant"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
