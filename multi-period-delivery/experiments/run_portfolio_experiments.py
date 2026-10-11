#!/usr/bin/env python3
"""Reproducible portfolio experiment driver for the multi-period project.

This is the publication-quality experiment entry point used by the
portfolio's ``results/`` folder. It runs the multi-period ALNS on a small,
curated instance set that covers three scales, then writes two tidy CSV files:

* ``runs.csv``    - one row per (instance, variant) run.
* ``summary.csv`` - per-scale aggregate statistics.

Design notes
------------
* Each run is a separate ``python -m multi_period_delivery`` subprocess, so a
  crash or timeout in one run cannot affect the others.
* The final objective is recomputed independently from the run's
  ``opt_summary.csv`` (``sum(total_cost)``) instead of trusting the log line.
  The log is only used for the initial-solution baseline.
* Budgets and instance ids are declared in one place so the experiment is
  reproducible: change them here, not in ad-hoc shell commands.

Example
-------
    python experiments/run_portfolio_experiments.py --workers 6
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Curated instance set: 2 small + 2 medium + 1 large.
# These are the thesis "representative" instances, so the new results can be
# compared directly with the recorded thesis numbers on identical data.
INSTANCES: Dict[str, List[int]] = {
    "small": [12, 30],
    "medium": [13, 26],
    "large": [3],
}

# Time budget in seconds per scale. The small budget matches the thesis
# comparison so the ALNS-vs-thesis numbers stay comparable.
BUDGETS: Dict[str, int] = {"small": 180, "medium": 360, "large": 720}

# ALNS variant -> extra CLI flags.
VARIANTS: Dict[str, List[str]] = {
    "basic": [],
    "tuner": ["--enable-param-tuner"],
    "ml": ["--enable-ml-operators"],
    "full": ["--enable-param-tuner", "--enable-ml-operators"],
}

INITIAL_RE = re.compile(
    r"Initial Solution Objective \(Gurobi-aligned\):\s*([0-9.]+)"
)
FINAL_RE = re.compile(
    r"Best Actual Objective \(Gurobi-aligned\):\s*([0-9.]+)"
)


@dataclass
class Job:
    scale: str
    dataset_id: int
    variant: str

    @property
    def name(self) -> str:
        return f"{self.scale}_dataset_{self.dataset_id}__{self.variant}"


def parse_log(text: str) -> Tuple[Optional[float], Optional[float]]:
    """Return (initial_objective, final_objective) parsed from an ALNS log."""
    initial = INITIAL_RE.search(text)
    final = FINAL_RE.search(text)
    return (
        float(initial.group(1)) if initial else None,
        float(final.group(1)) if final else None,
    )


def independent_objective(run_dir: Path) -> tuple[Optional[float], Optional[float]]:
    """Recompute the objective from opt_summary.csv.

    Returns ``(corrected_total, fixed_cost_only)``.

    * ``corrected_total`` is ``sum(total_cost)`` = vehicle fixed cost plus the
      minimum-load penalty (the corrected metric used by this repository).
    * ``fixed_cost_only`` is ``sum(fixed_cost)`` = the penalty-free metric that
      the thesis actually reported, kept for a like-for-like comparison.
    """
    summary = run_dir / "opt_summary.csv"
    if not summary.exists():
        return None, None
    total = 0.0
    fixed = 0.0
    rows = 0
    with summary.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                total += float(row["total_cost"])
                fixed += float(row["fixed_cost"])
            except (KeyError, TypeError, ValueError):
                return None, None
            rows += 1
    if not rows:
        return None, None
    return total, fixed


def run_job(
    job: Job,
    project_root: Path,
    results_dir: Path,
    seed: int,
    python_exe: str,
    budget_factor: float = 1.0,
) -> dict:
    """Execute a single (instance, variant) run and return its result row."""
    budget = max(2, int(round(BUDGETS[job.scale] * budget_factor)))
    data_dir = project_root / "datasets" / "multiple-periods" / job.scale
    run_output = results_dir / "raw" / job.name
    run_output.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "raw" / f"{job.name}.log"

    cmd = [
        python_exe,
        "-m",
        "multi_period_delivery",
        "--scale",
        job.scale,
        "--dataset-id",
        str(job.dataset_id),
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(run_output),
        "--max-runtime",
        str(budget),
        "--seed",
        str(seed),
        *VARIANTS[job.variant],
    ]

    started = time.time()
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

    initial_obj, final_from_log = parse_log(proc.stdout)
    dataset_output = run_output / f"dataset_{job.dataset_id}"
    independent, fixed_only = independent_objective(dataset_output)
    final_obj = independent if independent is not None else final_from_log

    improvement_pct = None
    if initial_obj is not None and final_obj is not None and initial_obj > 0:
        improvement_pct = (initial_obj - final_obj) / initial_obj * 100.0

    return {
        "scale": job.scale,
        "dataset_id": job.dataset_id,
        "variant": job.variant,
        "seed": seed,
        "time_budget_s": budget,
        "initial_objective": initial_obj,
        "final_objective": final_obj,
        "fixed_cost_objective": fixed_only,
        "improvement_pct": improvement_pct,
        "log_final_objective": final_from_log,
        "wall_seconds": round(wall, 2),
        "return_code": proc.returncode,
        "status": "success" if proc.returncode == 0 and final_obj is not None else "failed",
    }


def write_summary(rows: List[dict], results_dir: Path) -> None:
    """Write per-scale mean/median improvement and objective."""
    import statistics

    def stat(values: List[float], fn) -> Optional[float]:
        return round(fn(values), 4) if values else None

    summary_rows = []
    for scale in ("small", "medium", "large"):
        for variant in VARIANTS:
            subset = [
                r
                for r in rows
                if r["scale"] == scale
                and r["variant"] == variant
                and r["status"] == "success"
            ]
            improvements = [r["improvement_pct"] for r in subset if r["improvement_pct"] is not None]
            finals = [r["final_objective"] for r in subset if r["final_objective"] is not None]
            summary_rows.append(
                {
                    "scale": scale,
                    "variant": variant,
                    "run_count": len(subset),
                    "mean_final_objective": stat(finals, statistics.fmean),
                    "mean_improvement_pct": stat(improvements, statistics.fmean),
                    "median_improvement_pct": stat(improvements, statistics.median),
                }
            )

    fieldnames = [
        "scale",
        "variant",
        "run_count",
        "mean_final_objective",
        "mean_improvement_pct",
        "median_improvement_pct",
    ]
    with (results_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory for runs.csv, summary.csv and raw logs.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--budget-factor",
        type=float,
        default=1.0,
        help="Scale all time budgets; use a small value for a smoke test.",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Optional comma-separated variant filter, e.g. basic,full.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    results_dir = args.results_dir
    if not results_dir.is_absolute():
        results_dir = project_root / results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    variants = list(VARIANTS)
    if args.only:
        variants = [v.strip() for v in args.only.split(",") if v.strip()]

    jobs = [
        Job(scale, dataset_id, variant)
        for scale, dataset_ids in INSTANCES.items()
        for dataset_id in dataset_ids
        for variant in variants
    ]

    print(f"Scheduling {len(jobs)} runs with {args.workers} workers...")
    rows: List[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_job,
                job,
                project_root,
                results_dir,
                args.seed,
                sys.executable,
                args.budget_factor,
            ): job
            for job in jobs
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"  [{row['status']}] {row['scale']}/dataset_{row['dataset_id']}"
                f" {row['variant']}: final={row['final_objective']}"
                f" improvement={row['improvement_pct']} ({row['wall_seconds']}s)"
            )

    rows.sort(key=lambda r: (r["scale"], r["dataset_id"], r["variant"]))
    fieldnames = [
        "scale",
        "dataset_id",
        "variant",
        "seed",
        "time_budget_s",
        "initial_objective",
        "final_objective",
        "fixed_cost_objective",
        "improvement_pct",
        "log_final_objective",
        "wall_seconds",
        "return_code",
        "status",
    ]
    with (results_dir / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_summary(rows, results_dir)
    print(f"Wrote {results_dir / 'runs.csv'} and {results_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
