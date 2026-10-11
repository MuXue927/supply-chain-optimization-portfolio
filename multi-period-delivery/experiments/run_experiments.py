#!/usr/bin/env python3
"""
Batch experiment runner for the multi-period project.

Responsibilities:
- Accept instance paths, variants, seeds, time limits, parallelism, and output paths.
- Run the multi-period optimizer for each (instance, variant, seed) tuple.
- Update the shared ALNS configuration before each run.
- Write run_metadata.json so prepare_thesis_results.py can aggregate results.

Notes:
- On Windows, keep --parallel=1 unless concurrent solver processes are known to be safe.
- This script assumes the ALNSOptimizer signature:
    ALNSOptimizer(input_file_loc, output_file_loc)
  and the method run_optimization(dataset_name) -> bool.
- Adjust the adapter logic if the public interface changes.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import glob
import time
from contextlib import redirect_stdout, redirect_stderr, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import List
import multiprocessing as mp

RUN_META_NAME = "run_metadata.json"

# Ensure repository root is importable when the script is executed directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def utc_now_iso_z() -> str:
    """Return a timezone-aware UTC ISO-8601 timestamp ending with 'Z'."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_seeds(seeds_spec: str) -> List[int]:
    """
    Supported forms:
      - "0-9" -> [0..9]
      - "0,2,5" -> [0,2,5]
      - "0" -> [0]
    """
    if "-" in seeds_spec and "," not in seeds_spec:
        a, b = seeds_spec.split("-")
        return list(range(int(a), int(b) + 1))
    parts = [p.strip() for p in seeds_spec.split(",")]
    out = []
    for p in parts:
        if p == "":
            continue
        out.append(int(p))
    return out


def is_valid_instance(path_str: str):
    """
    Validate whether a path looks like a runnable instance.
    Returns (ok: bool, reason: str).

    Supported format:
    - A concrete dataset_N directory such as
      datasets/multiple-periods/small/dataset_1 containing order.csv.

    Validation rules:
     - Files are invalid.
     - dataset_* directories are accepted only when order.csv exists.
     - Container directories must be expanded before validation.
    """
    p = Path(path_str)
    if p.is_file():
        return False, "path is a file; expected a dataset directory"

    # Accept only concrete dataset_* directories.
    if p.name.lower().startswith("dataset_"):
        orderf = p / "order.csv"
        if orderf.exists():
            return True, "ok (dataset_* directory)"
        else:
            return False, "dataset_* directory but order.csv missing"

    # Reject container directories such as small, medium, and large.
    return False, "not a dataset_* directory (use dataset_1, dataset_2, ...)"


def expand_container_to_datasets(container_path: str) -> List[str]:
    """
    Expand a container directory such as small, medium, or large into all
    contained dataset_* directories.

    Example:
    - Input: datasets/multiple-periods/small
    - Output: [
        datasets/multiple-periods/small/dataset_1,
        datasets/multiple-periods/small/dataset_2,
        ...
      ]

    Returns the sorted dataset path list.
    """
    container = Path(container_path)
    if not container.is_dir():
        return []

    datasets = []
    try:
        for sub in sorted(container.iterdir()):
            if sub.is_dir() and sub.name.lower().startswith("dataset_"):
                datasets.append(str(sub))
    except Exception:
        pass

    return datasets


def expand_instances(patterns: List[str]) -> List[str]:
    paths = []
    for p in patterns:
        if "*" in p or "?" in p:
            paths.extend(sorted(glob.glob(p)))
        else:
            # If the path is a directory, treat the directory itself as an instance
            # (do not expand to individual files). This avoids passing config files
            # or unrelated CSVs as instances.
            if os.path.isdir(p):
                paths.append(p)
            else:
                paths.append(p)
    # dedupe and keep order
    seen = set()
    out = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    # Remove configuration files such as model_config*.json.
    filtered = [p for p in out if "model_config" not in Path(p).name.lower()]

    # Validate each path first. Container directories are expanded into their
    # dataset_* children.
    pre_validated = []
    for path_str in filtered:
        ok, reason = is_valid_instance(path_str)
        if ok:
            pre_validated.append(path_str)
        else:
            # Check whether this path is a scale container directory.
            container_datasets = expand_container_to_datasets(path_str)
            if container_datasets:
                print(
                    f"Auto-expanding container: {path_str} -> {len(container_datasets)} datasets",
                    flush=True,
                )
                pre_validated.extend(container_datasets)
            else:
                # Record the validation failure reason.
                pre_validated.append((path_str, reason))

    # Apply final validation to all candidate instances.
    valid_instances = []
    skipped = []
    for item in pre_validated:
        if isinstance(item, tuple):
            path_str, reason = item
            skipped.append((path_str, reason))
        else:
            path_str = item
            ok, reason = is_valid_instance(path_str)
            if ok:
                valid_instances.append(path_str)
            else:
                skipped.append((path_str, reason))

    # Report skipped invalid instances.
    if skipped:
        print("Skipped instances (invalid):", flush=True)
        for path_str, reason in skipped:
            print(f"  - {path_str}  => {reason}", flush=True)

    return valid_instances


def make_variant_flags(variant: str):
    """
    variant -> (enable_tuner, enable_ml)
    """
    v = variant.strip().lower()
    if v in ("alns-b", "basic"):
        return False, False
    if v in ("alns-t", "tuner", "paramtuner"):
        return True, False
    if v in ("alns-ml", "ml", "mlo", "mlops", "mloperators"):
        return False, True
    if v in ("alns-f", "full", "tuner+ml", "full+ml"):
        return True, True
    if v in ("monolithic", "gurobi", "mono"):
        return None, None
    raise ValueError(f"Unknown variant: {variant}")


def run_single(args_tuple):
    """
    Worker entry point that imports the package and runs one experiment.
    args_tuple: (instance_path, variant, seed, time_limit, out_base_dir)
    """
    if len(args_tuple) >= 6:
        instance_path, variant, seed, time_limit, out_base_dir, isolate_worker_log = (
            args_tuple
        )
    else:
        instance_path, variant, seed, time_limit, out_base_dir = args_tuple
        isolate_worker_log = False
    start_ts = time.time()
    t0 = utc_now_iso_z()
    instance_path = str(Path(instance_path).resolve())
    basename = Path(instance_path).stem
    safe_variant = variant.replace(",", "_")
    run_dir = (
        Path(out_base_dir)
        / f"{basename}__{safe_variant}__seed{seed}__t{int(time_limit)}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    worker_log_file = run_dir / "worker_console.log"

    # If a concrete dataset directory lacks model_config_alns.json, try to
    # copy a fallback configuration from a parent datasets directory.
    try:
        inst_path_obj = Path(instance_path)
        if inst_path_obj.is_dir():
            cfg_name = "model_config_alns.json"
            dst_cfg = inst_path_obj / cfg_name
            if not dst_cfg.exists():
                # Look for a shared configuration under the parent directory.
                parent_cfg = (
                    Path("datasets") / "multiple-periods" / "model_config_alns.json"
                )
                if parent_cfg.exists():
                    try:
                        import shutil

                        shutil.copy(parent_cfg, dst_cfg)
                    except Exception:
                        # Best-effort copy; later validation reports any missing file.
                        pass

    except Exception:
        # Ignore any issues in best-effort fallback; main run will record errors if data still missing
        pass

    # Import inside worker to avoid pickling/import issues
    try:
        from multi_period_delivery.alns_config import default_config as ALNSConfig
        from multi_period_delivery.alns_optimizer import ALNSOptimizer
    except Exception as e:
        meta = {
            "instance": instance_path,
            "variant": variant,
            "seed": seed,
            "time_limit": time_limit,
            "start_time": t0,
            "status": "import_error",
            "error": str(e),
        }
        (run_dir / RUN_META_NAME).write_text(
            json.dumps(meta, indent=2, ensure_ascii=False)
        )
        return meta

    # Compute flags for variant
    try:
        enable_tuner, enable_ml = make_variant_flags(variant)
    except Exception as e:
        meta = {
            "instance": instance_path,
            "variant": variant,
            "seed": seed,
            "time_limit": time_limit,
            "start_time": t0,
            "status": "bad_variant",
            "error": str(e),
        }
        (run_dir / RUN_META_NAME).write_text(
            json.dumps(meta, indent=2, ensure_ascii=False)
        )
        return meta

    v = variant.lower()

    # Run optimizer
    ok = False
    err = None
    alns_meta = {}
    mono_meta = {}
    out_ctx = nullcontext()
    log_fh = None
    if isolate_worker_log:
        log_fh = open(worker_log_file, "a", encoding="utf-8", buffering=1)
        out_ctx = nullcontext(log_fh)

    try:
        with out_ctx as _fh:
            std_ctx = redirect_stdout(_fh) if (_fh is not None) else nullcontext()
            err_ctx = redirect_stderr(_fh) if (_fh is not None) else nullcontext()
            with std_ctx, err_ctx:
                # For a concrete dataset_N directory, use its parent as the
                # input root and the directory name as the dataset name.
                inst_obj = Path(instance_path)
                if inst_obj.name.lower().startswith("dataset_") and inst_obj.is_dir():
                    input_loc = str(inst_obj.parent)
                    dataset_name = inst_obj.name
                else:
                    input_loc = instance_path
                    dataset_name = basename

                if v in ("monolithic", "gurobi", "mono"):
                    # Monolithic / Gurobi baseline path
                    from multi_period_delivery.monolithic.optLogicMultiple import (
                        run_model_from_paths,
                    )

                    mono_meta = run_model_from_paths(
                        input_file_loc=input_loc,
                        output_file_loc=str(run_dir),
                        dataset_name=dataset_name,
                        time_limit=int(time_limit),
                        write_outputs=True,
                    )
                    # Merge monolithic metadata into the unified record.
                    if isinstance(mono_meta, dict):
                        status = mono_meta.get("status", "unknown")
                        ok = status in (
                            "optimal",
                            "success",
                            "time_limit",
                            "suboptimal",
                        )
                    else:
                        mono_meta = {}
                        ok = False
                else:
                    # Configure ALNS runtime via default_config
                    cfg_updates = {
                        "ENABLE_PARAM_TUNER": bool(enable_tuner),
                        "ENABLE_ML_OPERATORS": bool(enable_ml),
                        "SEED": int(seed),
                        "MAX_RUNTIME": int(time_limit),
                        # Use a separate log directory for each run.
                        "LOG_DIR": str(run_dir / "logs"),
                    }
                    ALNSConfig.update_from_dict(cfg_updates)

                    # Pass the output directory to ALNSOptimizer. input_loc must
                    # be the parent directory containing dataset_* folders.
                    optimizer = ALNSOptimizer(input_loc, str(run_dir))
                    ok = optimizer.run_optimization(dataset_name)

                    # Collect lightweight ALNS tracking metrics for run_metadata.json
                    # Keep only scalar values to avoid huge metadata and non-serializable objects.
                    try:
                        tracker = getattr(optimizer, "tracker", None)
                        if tracker is not None and hasattr(tracker, "get_statistics"):
                            stats = tracker.get_statistics() or {}
                            total_iterations = int(
                                stats.get("total_iterations", 0) or 0
                            )
                            best_actual_obj = stats.get("best_actual_objective", None)
                            best_search_obj = stats.get("best_objective", None)
                            elapsed_time = stats.get("elapsed_time", None)

                            # Prefer tracker elapsed_time; fall back to this run wall-clock later.
                            iter_per_second = None
                            try:
                                elapsed_val = (
                                    float(elapsed_time)
                                    if elapsed_time is not None
                                    else None
                                )
                                if elapsed_val is not None and elapsed_val > 1e-9:
                                    iter_per_second = (
                                        float(total_iterations) / elapsed_val
                                    )
                            except Exception:
                                iter_per_second = None

                            alns_meta = {
                                "total_iterations": total_iterations,
                                "best_actual_objective": best_actual_obj,
                                "best_search_objective": best_search_obj,
                                "tracker_elapsed_seconds": elapsed_time,
                                "iter_per_second": iter_per_second,
                            }
                    except Exception:
                        # Best-effort only; experiment should not fail due to metadata enrichment.
                        alns_meta = {}

                status = "success" if ok else "failed"
    except Exception as e:
        ok = False
        status = "runtime_error"
        err = str(e)
    finally:
        if log_fh is not None:
            log_fh.close()
    end_ts = time.time()
    meta = {
        "instance": instance_path,
        "variant": variant,
        "seed": seed,
        "time_limit": time_limit,
        "start_time": t0,
        "end_time": utc_now_iso_z(),
        "wall_seconds": end_ts - start_ts,
        "status": status,
    }
    # attach error if present (covers both exception and explicit failure cases)
    if err is not None:
        meta["error"] = err

    # merge monolithic/gurobi specific metadata when available
    if isinstance(mono_meta, dict) and mono_meta:
        for k in (
            "final_obj",
            "best_bound",
            "gap_pct",
            "mip_gap",
            "gurobi_runtime",
            "iterations",
            "node_count",
            "vehicle_count",
            "output_files",
            "warnings",
        ):
            if k in mono_meta:
                meta[k] = mono_meta[k]

    # merge ALNS-specific tracking metadata when available
    if isinstance(alns_meta, dict) and alns_meta:
        for k, v in alns_meta.items():
            meta[k] = v

        # Ensure iter_per_second is always available when total_iterations exists.
        if meta.get("iter_per_second", None) is None:
            try:
                total_iterations = int(meta.get("total_iterations", 0) or 0)
                wall = float(meta.get("wall_seconds", 0.0) or 0.0)
                if wall > 1e-9:
                    meta["iter_per_second"] = float(total_iterations) / wall
            except Exception:
                pass

    # write metadata
    try:
        (run_dir / RUN_META_NAME).write_text(
            json.dumps(meta, indent=2, ensure_ascii=False)
        )
    except Exception:
        # best-effort
        pass

    return meta


def main():
    parser = argparse.ArgumentParser(description="Run ALNS experiments (template).")
    parser.add_argument(
        "--instances",
        nargs="+",
        required=True,
        help="Instance path or glob pattern; multiple values are supported.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="ALNS-B,ALNS-F",
        help=(
            "Comma-separated variants. Thesis names: "
            "ALNS-B,ALNS-T,ALNS-ML,ALNS-F; monolithic is also supported."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0-4",
        help="Random seed range or list, for example 0-9 or 0,2,5.",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=300,
        help="Wall-clock time limit in seconds for each run.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="OutPut-ALNS/experiments",
        help="Root directory for experiment output.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel worker processes. Use 1 on Windows by default.",
    )

    args = parser.parse_args()

    instances = expand_instances(args.instances)
    if not instances:
        print("No instances found after validation. Exiting.")
        return

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    seeds = parse_seeds(args.seeds)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # build tasks
    tasks = []
    for inst in instances:
        for var in variants:
            for s in seeds:
                tasks.append((inst, var, s, args.time_limit, str(out_dir)))

    print(
        f"Discovered {len(instances)} instances, {len(variants)} variants, {len(seeds)} seeds -> total runs: {len(tasks)}"
    )
    print(f"Output base dir: {out_dir}")
    start = time.time()

    if args.parallel is None or args.parallel <= 1:
        results = []
        for t in tasks:
            print(f"Running: instance={Path(t[0]).name} variant={t[1]} seed={t[2]}")
            res = run_single((*t, False))
            results.append(res)
    else:
        # multiprocessing with spawn (Windows safe)
        mp.set_start_method("spawn", force=True)
        with mp.Pool(processes=args.parallel) as pool:
            parallel_tasks = [(*t, True) for t in tasks]
            for res in pool.imap_unordered(run_single, parallel_tasks):
                print(json.dumps(res, ensure_ascii=False))
        results = []  # pool already printed metadata

    elapsed = time.time() - start
    summary = {
        "n_instances": len(instances),
        "n_variants": len(variants),
        "n_seeds": len(seeds),
        "total_runs": len(tasks),
        "elapsed_seconds": elapsed,
        "timestamp": utc_now_iso_z(),
    }
    (out_dir / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(
        "Experiment finished. Summary written to:", out_dir / "experiment_summary.json"
    )


if __name__ == "__main__":
    main()
