#!/usr/bin/env python3
"""
Prepare thesis-ready experiment tables and normalized run records.

This script scans the experiment output tree, collects run metadata, rebuilds
initial-solution baselines for ALNS runs when needed, and exports unified CSV
tables plus publication-friendly plots.

Outputs:
- long-format run table
- scale/variant summary table
- instance-level comparison table

Usage example:
  python runExperiments/prepare_thesis_results.py \
    --results-root ExperimentsResults \
    --out-dir ExperimentsResults/thesis_analysis
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_META_NAME = "run_metadata.json"
SELECTED_ALL_NAME = "selected_instances_all.csv"


def canonical_variant(value: Any) -> str:
    aliases = {
        "basic": "ALNS-B",
        "alns-b": "ALNS-B",
        "tuner": "ALNS-T",
        "paramtuner": "ALNS-T",
        "alns-t": "ALNS-T",
        "ml": "ALNS-ML",
        "mlo": "ALNS-ML",
        "mlops": "ALNS-ML",
        "mloperators": "ALNS-ML",
        "alns-ml": "ALNS-ML",
        "full": "ALNS-F",
        "tuner+ml": "ALNS-F",
        "full+ml": "ALNS-F",
        "alns-f": "ALNS-F",
    }
    text = str(value).strip()
    return aliases.get(text.lower(), text)


def safe_float(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        value = float(value)
        if math.isfinite(value):
            return value
    except Exception:
        pass
    return float("nan")


def safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def utc_now_stamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def infer_scale_from_instance_path(instance_path: Path) -> Optional[str]:
    parts = [p.lower() for p in instance_path.parts]
    for scale in ("small", "medium", "large"):
        if scale in parts:
            return scale
    return None


def load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def discover_run_dirs(results_root: Path) -> List[Path]:
    run_dirs = [p.parent for p in results_root.rglob(RUN_META_NAME)]
    run_dirs = sorted(set(run_dirs))
    return run_dirs


def discover_selection_manifest(results_root: Path) -> Optional[Path]:
    candidates = [
        p
        for p in results_root.iterdir()
        if p.is_dir() and p.name.startswith("selection_")
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_selection_table(selection_dir: Optional[Path]) -> pd.DataFrame:
    if selection_dir is None:
        return pd.DataFrame()
    path = selection_dir / SELECTED_ALL_NAME
    return read_csv_if_exists(path)


@contextmanager
def temporary_config(updates: Dict[str, Any]):
    from ALNSCode.alns_config import default_config as ALNSConfig

    old_values = {k: copy.deepcopy(getattr(ALNSConfig, k, None)) for k in updates}
    ALNSConfig.update_from_dict(updates)
    try:
        yield ALNSConfig
    finally:
        ALNSConfig.update_from_dict(old_values)


def compute_initial_baseline(
    *,
    instance_path: Path,
    variant: str,
    seed: int,
    cache: Dict[Tuple[str, str, int], Dict[str, Any]],
) -> Dict[str, Any]:
    cache_key = (str(instance_path.resolve()), variant.lower(), int(seed))
    if cache_key in cache:
        return cache[cache_key]

    from ALNSCode.alns_optimizer import ALNSOptimizer

    variant_map = {
        "basic": (False, False),
        "tuner": (True, False),
        "ml": (False, True),
        "full": (True, True),
    }
    enable_tuner, enable_ml = variant_map.get(variant.lower(), (False, False))

    dataset_name = instance_path.name
    input_loc = instance_path.parent

    with temporary_config(
        {
            "ENABLE_PARAM_TUNER": bool(enable_tuner),
            "ENABLE_ML_OPERATORS": bool(enable_ml),
            "SEED": int(seed),
            "QUIET_INITIAL_SOLUTION_LOGS": True,
            "QUIET_OPERATOR_LOGS": True,
        }
    ):
        with tempfile.TemporaryDirectory(prefix="thesis_initial_") as tmp_dir:
            optimizer = ALNSOptimizer(str(input_loc), tmp_dir)
            optimizer.rng = np.random.default_rng(int(seed))
            ok = optimizer.load_data(dataset_name)
            if not ok:
                result = {
                    "initial_actual_objective": float("nan"),
                    "initial_search_objective": float("nan"),
                    "initial_vehicle_count": float("nan"),
                    "initial_feasible": False,
                    "initial_available": False,
                }
                cache[cache_key] = result
                return result

            init_state = optimizer.create_initial_solution()
            if init_state is None:
                result = {
                    "initial_actual_objective": float("nan"),
                    "initial_search_objective": float("nan"),
                    "initial_vehicle_count": float("nan"),
                    "initial_feasible": False,
                    "initial_available": False,
                }
                cache[cache_key] = result
                return result

            try:
                feasible, _ = init_state.validate()
            except Exception:
                feasible = False
            try:
                init_actual = float(init_state.calculate_cost())
            except Exception:
                init_actual = float("nan")
            try:
                init_search = float(init_state.objective())
            except Exception:
                init_search = float("nan")
            try:
                init_vehicle_count = float(len(list(init_state.iter_used_vehicles())))
            except Exception:
                try:
                    init_vehicle_count = float(
                        len(getattr(init_state, "vehicles", []) or [])
                    )
                except Exception:
                    init_vehicle_count = float("nan")

            result = {
                "initial_actual_objective": init_actual,
                "initial_search_objective": init_search,
                "initial_vehicle_count": init_vehicle_count,
                "initial_feasible": bool(feasible),
                "initial_available": True,
            }
            cache[cache_key] = result
            return result


def normalize_run_record(
    *,
    run_dir: Path,
    meta: Dict[str, Any],
    selection_df: pd.DataFrame,
    initial_cache: Dict[Tuple[str, str, int], Dict[str, Any]],
) -> Dict[str, Any]:
    instance_raw = meta.get("instance")
    instance_path = Path(instance_raw).expanduser() if instance_raw else run_dir
    if not instance_path.is_absolute():
        instance_path = (PROJECT_ROOT / instance_path).resolve()
    else:
        instance_path = instance_path.resolve()

    scale = infer_scale_from_instance_path(instance_path)
    dataset_name = instance_path.name
    variant = str(meta.get("variant", ""))
    seed = safe_int(meta.get("seed"))
    time_limit = safe_int(meta.get("time_limit"))

    row: Dict[str, Any] = {
        "scale": scale,
        "dataset_name": dataset_name,
        "instance": str(instance_path),
        "variant": variant,
        "seed": seed,
        "time_limit": time_limit,
        "status": meta.get("status"),
        "start_time": meta.get("start_time"),
        "end_time": meta.get("end_time"),
        "wall_seconds": safe_float(meta.get("wall_seconds")),
        "total_iterations": safe_int(meta.get("total_iterations")),
        "tracker_elapsed_seconds": safe_float(meta.get("tracker_elapsed_seconds")),
        "iter_per_second": safe_float(meta.get("iter_per_second")),
        "best_actual_objective": safe_float(meta.get("best_actual_objective")),
        "best_search_objective": safe_float(meta.get("best_search_objective")),
        "final_obj": safe_float(meta.get("final_obj")),
        "gap_pct": safe_float(meta.get("gap_pct")),
        "mip_gap": safe_float(meta.get("mip_gap")),
        "best_bound": safe_float(meta.get("best_bound")),
    }

    if not selection_df.empty:
        match = selection_df[
            (selection_df["dataset_name"] == dataset_name)
            & (selection_df["scale"] == scale)
        ]
        if not match.empty:
            for col in match.columns:
                if col in {"selection_order", "scale", "dataset_name", "instance_path"}:
                    continue
                row[col] = match.iloc[0][col]

    if variant.lower() in {"monolithic", "gurobi", "mono"}:
        row["comparison_baseline"] = "milp"
        row["milp_gap_pct"] = (
            row["gap_pct"] if np.isfinite(row["final_obj"]) else float("nan")
        )
        row["initial_actual_objective"] = float("nan")
        row["initial_search_objective"] = float("nan")
        row["improvement_pct"] = float("nan")
        row["gap_vs_initial_pct"] = float("nan")
        row["gap_vs_milp_pct"] = float("nan")
        row["initial_available"] = False
        row["initial_feasible"] = False
        row["initial_vehicle_count"] = float("nan")
        return row

    row["comparison_baseline"] = "initial"
    if seed is None:
        row["initial_actual_objective"] = float("nan")
        row["initial_search_objective"] = float("nan")
        row["initial_vehicle_count"] = float("nan")
        row["initial_feasible"] = False
        row["initial_available"] = False
    else:
        baseline = compute_initial_baseline(
            instance_path=instance_path,
            variant=variant,
            seed=seed,
            cache=initial_cache,
        )
        row.update(baseline)

    init_obj = safe_float(row.get("initial_actual_objective"))
    best_actual = safe_float(row.get("best_actual_objective"))
    final_milp = safe_float(row.get("final_obj"))

    if np.isfinite(init_obj) and np.isfinite(best_actual) and abs(init_obj) > 1e-12:
        row["improvement_pct"] = (init_obj - best_actual) / abs(init_obj) * 100.0
        row["gap_vs_initial_pct"] = (best_actual - init_obj) / abs(init_obj) * 100.0
    else:
        row["improvement_pct"] = float("nan")
        row["gap_vs_initial_pct"] = float("nan")

    if np.isfinite(final_milp) and abs(final_milp) > 1e-12 and np.isfinite(best_actual):
        row["gap_vs_milp_pct"] = (best_actual - final_milp) / abs(final_milp) * 100.0
    else:
        row["gap_vs_milp_pct"] = float("nan")

    return row


def collect_records(results_root: Path, selection_df: pd.DataFrame) -> pd.DataFrame:
    initial_cache: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []

    for run_dir in discover_run_dirs(results_root):
        meta_path = run_dir / RUN_META_NAME
        meta = load_json(meta_path)
        if not meta:
            continue
        row = normalize_run_record(
            run_dir=run_dir,
            meta=meta,
            selection_df=selection_df,
            initial_cache=initial_cache,
        )
        row["run_dir"] = str(run_dir)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    numeric_cols = [
        "seed",
        "time_limit",
        "wall_seconds",
        "total_iterations",
        "tracker_elapsed_seconds",
        "iter_per_second",
        "best_actual_objective",
        "best_search_objective",
        "final_obj",
        "gap_pct",
        "mip_gap",
        "best_bound",
        "initial_actual_objective",
        "initial_search_objective",
        "initial_vehicle_count",
        "improvement_pct",
        "gap_vs_initial_pct",
        "gap_vs_milp_pct",
        "milp_gap_pct",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def filter_scales(df: pd.DataFrame, scales: List[str]) -> pd.DataFrame:
    if df.empty or not scales or "scale" not in df.columns:
        return df
    scale_set = {s.strip().lower() for s in scales if s.strip()}
    if not scale_set:
        return df
    return df[df["scale"].astype(str).str.lower().isin(scale_set)].copy()


def compute_instance_wide_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    instance_cols = ["scale", "dataset_name", "variant", "run_dir"]
    value_cols = [
        "best_actual_objective",
        "final_obj",
        "initial_actual_objective",
        "improvement_pct",
        "gap_vs_initial_pct",
        "gap_vs_milp_pct",
        "wall_seconds",
        "status",
        "total_iterations",
        "iter_per_second",
        "mip_gap",
        "gap_pct",
    ]
    available_cols = [c for c in instance_cols + value_cols if c in df.columns]
    long_df = df[available_cols].copy()

    wide = df.pivot_table(
        index=["scale", "dataset_name"],
        columns="variant",
        values="best_actual_objective",
        aggfunc="median",
    ).reset_index()
    return long_df, wide


def summarize_by_scale_variant(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    grouped = (
        df.groupby(["scale", "variant"])
        .agg(
            run_count=("run_dir", "count"),
            success_count=("status", lambda s: int((s == "success").sum())),
            mean_final_obj=("best_actual_objective", "mean"),
            median_final_obj=("best_actual_objective", "median"),
            std_final_obj=("best_actual_objective", "std"),
            mean_wall_seconds=("wall_seconds", "mean"),
            median_wall_seconds=("wall_seconds", "median"),
            mean_iterations=("total_iterations", "mean"),
            mean_initial_obj=("initial_actual_objective", "mean"),
            median_initial_obj=("initial_actual_objective", "median"),
            mean_improvement_pct=("improvement_pct", "mean"),
            median_improvement_pct=("improvement_pct", "median"),
            std_improvement_pct=("improvement_pct", "std"),
            mean_gap_vs_milp_pct=("gap_vs_milp_pct", "mean"),
            median_gap_vs_milp_pct=("gap_vs_milp_pct", "median"),
            std_gap_vs_milp_pct=("gap_vs_milp_pct", "std"),
        )
        .reset_index()
    )
    return grouped


def write_outputs(df: pd.DataFrame, out_dir: Path):
    ensure_dir(out_dir)

    long_csv = out_dir / "thesis_results_long.csv"
    df.to_csv(long_csv, index=False)

    summary_df = summarize_by_scale_variant(df)
    summary_csv = out_dir / "thesis_summary_by_scale_variant.csv"
    summary_df.to_csv(summary_csv, index=False)

    long_table, wide_table = compute_instance_wide_tables(df)
    long_table.to_csv(out_dir / "thesis_instance_long.csv", index=False)
    wide_table.to_csv(out_dir / "thesis_instance_wide.csv", index=False)

    manifest = {
        "n_runs": int(df.shape[0]),
        "n_instances": int(df["dataset_name"].nunique())
        if "dataset_name" in df.columns
        else 0,
        "n_variants": int(df["variant"].nunique()) if "variant" in df.columns else 0,
        "tables": {
            "long": str(long_csv),
            "summary": str(summary_csv),
            "instance_long": str(out_dir / "thesis_instance_long.csv"),
            "instance_wide": str(out_dir / "thesis_instance_wide.csv"),
        },
    }
    (out_dir / "thesis_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare thesis-ready experiment tables and normalized run records."
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="ExperimentsResults",
        help="Root directory containing small/medium/large result folders.",
    )
    parser.add_argument(
        "--selection-dir",
        type=str,
        default="",
        help="Optional selection manifest directory; auto-detected if omitted.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output directory; defaults to a timestamped thesis_analysis folder under results-root.",
    )
    parser.add_argument(
        "--scales",
        type=str,
        default="small,medium,large",
        help="Comma-separated scales to include, e.g. small,medium.",
    )
    args = parser.parse_args()

    results_root = (PROJECT_ROOT / args.results_root).resolve()
    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")

    if args.selection_dir.strip():
        selection_dir = (PROJECT_ROOT / args.selection_dir).resolve()
    else:
        selection_dir = discover_selection_manifest(results_root)

    selection_df = load_selection_table(selection_dir)

    if args.out_dir.strip():
        out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    else:
        out_dir = results_root / f"thesis_analysis_{utc_now_stamp()}"

    df = collect_records(results_root, selection_df)
    if df.empty:
        print(f"No runs found under {results_root}")
        return 1

    if "variant" in df.columns:
        df["variant"] = df["variant"].map(canonical_variant)

    scales = [s.strip() for s in args.scales.split(",") if s.strip()]
    df = filter_scales(df, scales)
    if df.empty:
        print(f"No runs matched scales={scales} under {results_root}")
        return 1

    df = df.sort_values(
        ["scale", "dataset_name", "variant"], na_position="last"
    ).reset_index(drop=True)

    manifest = write_outputs(df, out_dir)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
