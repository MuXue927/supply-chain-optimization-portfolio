#!/usr/bin/env python3
"""
Sample a fixed number of dataset_* instances per scale and run experiments.

Features:
- Randomly sample N instances from each scale directory (small/medium/large)
- Persist selection manifests (CSV + JSON) for reproducibility
- Run experiments/run_experiments.py per scale with scale-specific variants/time-limits
- Continue running remaining scales even if one scale fails

Typical usage (PowerShell with venv already activated):
python experiments/sample_and_run_experiments.py --sample-size 20 --selection-seed 20260408
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_EXPERIMENTS = PROJECT_ROOT / "experiments" / "run_experiments.py"


def utc_now_iso_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def dataset_sort_key(p: Path) -> tuple[int, str]:
    name = p.name
    if name.startswith("dataset_"):
        suffix = name.split("dataset_", 1)[1]
        if suffix.isdigit():
            return int(suffix), name
    return 10**9, name


def load_stats_csv(stats_file: Path) -> dict[str, dict[str, str]]:
    if not stats_file.exists():
        return {}
    with stats_file.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        out = {}
        for row in reader:
            key = row.get("dataset_name", "")
            if key:
                out[key] = row
        return out


def collect_dataset_dirs(scale_dir: Path) -> list[Path]:
    ds = [p for p in scale_dir.iterdir() if p.is_dir() and p.name.startswith("dataset_")]
    ds.sort(key=dataset_sort_key)
    return ds


def write_selection_csv(
    file_path: Path,
    rows: list[dict[str, str]],
    extra_headers: list[str],
) -> None:
    base_headers = [
        "selection_order",
        "scale",
        "dataset_name",
        "instance_path",
    ]
    headers = base_headers + [h for h in extra_headers if h not in base_headers]
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow({h: r.get(h, "") for h in headers})


def run_scale(
    *,
    python_exe: str,
    instances: list[Path],
    variants: str,
    seeds: str,
    time_limit: int,
    parallel: int,
    out_dir: Path,
) -> tuple[int, list[str]]:
    cmd = [
        python_exe,
        str(RUN_EXPERIMENTS),
        "--instances",
    ] + [str(p) for p in instances] + [
        "--variants",
        variants,
        "--seeds",
        seeds,
        "--time-limit",
        str(time_limit),
        "--parallel",
        str(parallel),
        "--out-dir",
        str(out_dir),
    ]

    rc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False).returncode
    return rc, cmd


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sample 20 instances per scale and run experiment batches."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="datasets/multiple-periods",
        help="Base directory containing small/medium/large",
    )
    parser.add_argument(
        "--scales",
        type=str,
        default="small,medium,large",
        help="Comma-separated scales to run",
    )
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--selection-seed", type=int, default=20260408)
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--out-root", type=str, default="ExperimentsResults")
    parser.add_argument(
        "--run-tag",
        type=str,
        default="",
        help="Optional run tag; default uses timestamp",
    )
    parser.add_argument("--small-time", type=int, default=180)
    parser.add_argument("--medium-time", type=int, default=360)
    parser.add_argument("--large-time", type=int, default=720)
    parser.add_argument(
        "--small-variants",
        type=str,
        default="monolithic,ALNS-B,ALNS-T,ALNS-ML,ALNS-F",
    )
    parser.add_argument(
        "--medium-variants",
        type=str,
        default="ALNS-B,ALNS-T,ALNS-ML,ALNS-F",
    )
    parser.add_argument(
        "--large-variants",
        type=str,
        default="ALNS-B,ALNS-T,ALNS-ML,ALNS-F",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only sample and write manifests, do not launch experiments",
    )

    args = parser.parse_args()

    run_tag = args.run_tag.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    scales = [s.strip() for s in args.scales.split(",") if s.strip()]

    time_map = {
        "small": args.small_time,
        "medium": args.medium_time,
        "large": args.large_time,
    }
    variant_map = {
        "small": args.small_variants,
        "medium": args.medium_variants,
        "large": args.large_variants,
    }

    base_dir = (PROJECT_ROOT / args.base_dir).resolve()
    out_root = (PROJECT_ROOT / args.out_root).resolve()
    selection_dir = out_root / f"selection_{run_tag}"
    selection_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.selection_seed)

    all_rows: list[dict[str, str]] = []
    per_scale_selected: dict[str, list[Path]] = {}

    # 1) Sample and persist selection info.
    for scale in scales:
        scale_dir = base_dir / scale
        if not scale_dir.exists() or not scale_dir.is_dir():
            raise FileNotFoundError(f"Scale directory not found: {scale_dir}")

        datasets = collect_dataset_dirs(scale_dir)
        if len(datasets) < args.sample_size:
            raise ValueError(
                f"Scale '{scale}' has only {len(datasets)} datasets, "
                f"but sample-size={args.sample_size}"
            )

        selected = rng.sample(datasets, args.sample_size)
        selected.sort(key=dataset_sort_key)
        per_scale_selected[scale] = selected

        stats_map = load_stats_csv(scale_dir / "instance_statistics.csv")
        extra_headers = []
        if stats_map:
            sample_row = next(iter(stats_map.values()))
            extra_headers = [k for k in sample_row.keys() if k != "dataset_name"]

        rows = []
        for idx, ds in enumerate(selected, start=1):
            row = {
                "selection_order": str(idx),
                "scale": scale,
                "dataset_name": ds.name,
                "instance_path": str(ds.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            }
            if ds.name in stats_map:
                for k, v in stats_map[ds.name].items():
                    if k == "dataset_name":
                        continue
                    row[k] = v
            rows.append(row)
            all_rows.append(row)

        write_selection_csv(
            selection_dir / f"selected_instances_{scale}.csv", rows, extra_headers
        )

    # unified csv for all scales
    all_extra = sorted(
        {
            key
            for r in all_rows
            for key in r.keys()
            if key
            not in {
                "selection_order",
                "scale",
                "dataset_name",
                "instance_path",
            }
        }
    )
    write_selection_csv(selection_dir / "selected_instances_all.csv", all_rows, all_extra)

    # 2) Run experiments scale-by-scale.
    run_records = []
    any_failure = False
    for scale in scales:
        selected = per_scale_selected[scale]
        scale_out_dir = out_root / f"{scale}_{time_map[scale] // 60}min_{run_tag}"

        if args.dry_run:
            cmd_preview = [
                args.python_exe,
                "experiments/run_experiments.py",
                "--instances",
            ] + [
                str(p.relative_to(PROJECT_ROOT)).replace("\\", "/") for p in selected
            ] + [
                "--variants",
                variant_map[scale],
                "--seeds",
                args.seeds,
                "--time-limit",
                str(time_map[scale]),
                "--parallel",
                str(args.parallel),
                "--out-dir",
                str(scale_out_dir.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            ]
            run_records.append(
                {
                    "scale": scale,
                    "status": "dry_run",
                    "return_code": 0,
                    "command": subprocess.list2cmdline(cmd_preview),
                    "out_dir": str(scale_out_dir.relative_to(PROJECT_ROOT)).replace(
                        "\\", "/"
                    ),
                    "n_instances": len(selected),
                }
            )
            continue

        t0 = time.time()
        rc, cmd = run_scale(
            python_exe=args.python_exe,
            instances=selected,
            variants=variant_map[scale],
            seeds=args.seeds,
            time_limit=time_map[scale],
            parallel=args.parallel,
            out_dir=scale_out_dir,
        )
        elapsed = time.time() - t0
        failed = rc != 0
        any_failure = any_failure or failed

        run_records.append(
            {
                "scale": scale,
                "status": "failed" if failed else "ok",
                "return_code": rc,
                "elapsed_seconds": elapsed,
                "command": subprocess.list2cmdline(cmd),
                "out_dir": str(scale_out_dir.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "n_instances": len(selected),
            }
        )

    manifest = {
        "created_at": utc_now_iso_z(),
        "project_root": str(PROJECT_ROOT),
        "base_dir": str(base_dir),
        "scales": scales,
        "sample_size": args.sample_size,
        "selection_seed": args.selection_seed,
        "seeds": args.seeds,
        "parallel": args.parallel,
        "run_tag": run_tag,
        "dry_run": args.dry_run,
        "time_limits": {k: time_map[k] for k in scales if k in time_map},
        "variants": {k: variant_map[k] for k in scales if k in variant_map},
        "selection_dir": str(selection_dir),
        "runs": run_records,
    }

    (selection_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Selection manifests written to: {selection_dir}")
    print(f"Run records saved to: {selection_dir / 'selection_manifest.json'}")

    return 1 if any_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
