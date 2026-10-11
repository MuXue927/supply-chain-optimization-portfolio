#!/usr/bin/env python3
"""Reproducible portfolio experiment driver for the single-period project.

For each curated instance it runs two approaches:

* ``heuristic`` - the fast two-stage constructive heuristic (always returns a
  feasible loading plan);
* ``cplex``     - the CPLEX Python API monolithic model under a fixed time
  limit, used as an exact-solver reference (objective and dual bound).

Outputs:

* ``runs.csv``    - one row per (instance, approach).
* ``summary.csv`` - per-scale pivoted results.

The monolithic model is hard to close to optimality, so the CPLEX run reports
its feasible incumbent (if found) and its best bound, which lets the report
state the heuristic's optimality gap honestly.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

INSTANCES: Dict[str, List[str]] = {
    "small": ["dataset_1", "dataset_3"],
    "medium": ["dataset_1", "dataset_3"],
    "large": ["dataset_1"],
}

# CPLEX time limit in seconds per instance.
CPLEX_TIME_LIMITS: Dict[str, int] = {"small": 600, "medium": 600, "large": 600}


@dataclass
class Job:
    scale: str
    dataset_name: str
    approach: str

    @property
    def name(self) -> str:
        return f"{self.scale}_{self.dataset_name}__{self.approach}"


def heuristic_objective(run_dir: Path, dataset_name: str) -> Optional[float]:
    summary = run_dir / dataset_name / "opt_summary.csv"
    if not summary.exists():
        return None
    total = 0.0
    rows = 0
    with summary.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                total += float(row["total_cost"])
            except (KeyError, TypeError, ValueError):
                return None
            rows += 1
    return total if rows else None


def cplex_result(run_dir: Path) -> dict:
    result_file = run_dir / "cplex_api_result.json"
    if not result_file.exists():
        return {}
    try:
        return json.loads(result_file.read_text(encoding="utf-8"))
    except Exception:
        return {}


def run_job(
    job: Job,
    project_root: Path,
    results_dir: Path,
    python_exe: str,
) -> dict:
    data_dir = project_root / "datasets" / "single-period" / job.scale
    run_dir = results_dir / "raw" / job.name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "raw" / f"{job.name}.log"

    row = {
        "scale": job.scale,
        "dataset_name": job.dataset_name,
        "approach": job.approach,
        "objective": None,
        "best_bound": None,
        "mip_gap_pct": None,
        "status": None,
        "wall_seconds": None,
    }

    started = time.time()
    if job.approach == "heuristic":
        cmd = [
            python_exe,
            "-m",
            "single_period_delivery",
            "heuristic",
            "--data-dir",
            str(data_dir),
            "--dataset-name",
            job.dataset_name,
            "--output-dir",
            str(run_dir),
        ]
    else:
        cmd = [
            python_exe,
            "-m",
            "single_period_delivery.cplex_api_monolithic",
            "--data-dir",
            str(data_dir),
            "--dataset-name",
            job.dataset_name,
            "--output-dir",
            str(run_dir),
            "--time-limit",
            str(CPLEX_TIME_LIMITS[job.scale]),
            "--mip-gap",
            "0",
        ]

    proc = subprocess.run(
        cmd,
        cwd=str(project_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    wall = time.time() - started
    log_path.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
    row["wall_seconds"] = round(wall, 2)
    row["return_code"] = proc.returncode

    if job.approach == "heuristic":
        row["objective"] = heuristic_objective(run_dir, job.dataset_name)
        row["status"] = "success" if row["objective"] is not None else "failed"
    else:
        payload = cplex_result(run_dir)
        row["objective"] = payload.get("objective")
        row["best_bound"] = payload.get("best_bound")
        gap = payload.get("solver_mip_gap")
        row["mip_gap_pct"] = None if gap is None else round(float(gap) * 100.0, 4)
        row["status"] = payload.get("status") or (
            "success" if proc.returncode == 0 else "failed"
        )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--approaches",
        default="heuristic,cplex",
        help="Comma-separated subset of {heuristic,cplex}.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    results_dir = args.results_dir
    if not results_dir.is_absolute():
        results_dir = project_root / results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    approaches = [a.strip() for a in args.approaches.split(",") if a.strip()]
    jobs = [
        Job(scale, dataset_name, approach)
        for scale, names in INSTANCES.items()
        for dataset_name in names
        for approach in approaches
    ]

    print(f"Scheduling {len(jobs)} runs with {args.workers} workers...")
    rows: List[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_job, job, project_root, results_dir, sys.executable): job
            for job in jobs
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"  [{row['status']}] {row['scale']}/{row['dataset_name']}"
                f" {row['approach']}: obj={row['objective']} bound={row['best_bound']}"
                f" ({row['wall_seconds']}s)"
            )

    rows.sort(key=lambda r: (r["scale"], r["dataset_name"], r["approach"]))
    fieldnames = [
        "scale",
        "dataset_name",
        "approach",
        "objective",
        "best_bound",
        "mip_gap_pct",
        "status",
        "wall_seconds",
        "return_code",
    ]
    runs_csv = results_dir / "runs.csv"
    if runs_csv.exists():
        # Merge so that re-running a subset of approaches keeps the others.
        existing = list(csv.DictReader(runs_csv.open(newline="", encoding="utf-8")))
        replaced = {(r["scale"], r["dataset_name"], r["approach"]) for r in rows}
        rows = [
            r
            for r in existing
            if (r["scale"], r["dataset_name"], r["approach"]) not in replaced
        ] + rows
        rows.sort(key=lambda r: (r["scale"], r["dataset_name"], r["approach"]))
    with runs_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {runs_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
