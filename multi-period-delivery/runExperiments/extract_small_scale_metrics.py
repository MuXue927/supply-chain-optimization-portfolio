#!/usr/bin/env python3
"""
Extract small-scale metrics from thesis_results_long.csv and summarize statistics.

Requested metrics:
- best_actual_objective
- initial_actual_objective
- final_obj
- gap_vs_milp_pct (recomputed)
- mip_gap

Outputs:
1) detail CSV with one row per small-scale run
2) summary CSV with mean / best / worst for each metric

Note:
- Missing values are left blank in CSV output.
- "best" is defined as the minimum value, "worst" as the maximum value.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def safe_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def recompute_gap_vs_milp_pct(best_actual_objective: float, final_obj: float) -> float:
    if not np.isfinite(best_actual_objective):
        return float("nan")
    if not np.isfinite(final_obj) or abs(final_obj) < 1e-12:
        return float("nan")
    return (best_actual_objective - final_obj) / abs(final_obj) * 100.0


def load_and_filter_small(input_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if "scale" not in df.columns:
        raise ValueError("Missing required column: scale")

    for col in [
        "best_actual_objective",
        "initial_actual_objective",
        "final_obj",
        "mip_gap",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = float("nan")

    if "gap_vs_milp_pct" not in df.columns:
        df["gap_vs_milp_pct"] = float("nan")

    small = df[df["scale"].astype(str).str.lower() == "small"].copy()

    # Build MILP baseline per dataset from monolithic/gurobi rows.
    mono_mask = (
        small["variant"].astype(str).str.lower().isin(["monolithic", "gurobi", "mono"])
    )
    mono = small[mono_mask].copy()
    mono["_milp_obj"] = pd.to_numeric(mono["final_obj"], errors="coerce")
    milp_by_dataset = (
        mono.groupby("dataset_name")["_milp_obj"].median().to_dict()
        if not mono.empty
        else {}
    )

    # For rows missing final_obj, fill with corresponding MILP baseline value.
    small["_final_obj_for_gap"] = pd.to_numeric(small["final_obj"], errors="coerce")
    small["_final_obj_for_gap"] = small.apply(
        lambda r: (
            safe_float(milp_by_dataset.get(str(r.get("dataset_name", ""))))
            if not np.isfinite(safe_float(r.get("_final_obj_for_gap")))
            else safe_float(r.get("_final_obj_for_gap"))
        ),
        axis=1,
    )

    small["gap_vs_milp_pct"] = small.apply(
        lambda r: recompute_gap_vs_milp_pct(
            safe_float(r.get("best_actual_objective")),
            safe_float(r.get("_final_obj_for_gap")),
        ),
        axis=1,
    )

    # Keep final output clean.
    small = small.drop(columns=["_final_obj_for_gap"], errors="ignore")

    return small


def build_detail_table(small: pd.DataFrame) -> pd.DataFrame:
    keep_cols: List[str] = [
        "scale",
        "dataset_name",
        "instance",
        "variant",
        "seed",
        "status",
        "best_actual_objective",
        "initial_actual_objective",
        "final_obj",
        "gap_vs_milp_pct",
        "mip_gap",
        "run_dir",
    ]
    for col in keep_cols:
        if col not in small.columns:
            small[col] = ""

    detail = small[keep_cols].copy()
    detail = detail.sort_values(
        by=["dataset_name", "variant", "seed"],
        na_position="last",
    ).reset_index(drop=True)
    return detail


def build_summary_table(detail: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "best_actual_objective",
        "initial_actual_objective",
        "final_obj",
        "gap_vs_milp_pct",
        "mip_gap",
    ]

    rows: List[Dict[str, Any]] = []
    for m in metrics:
        s = pd.to_numeric(detail[m], errors="coerce")
        valid = s.dropna()
        if valid.empty:
            rows.append(
                {
                    "metric": m,
                    "count": 0,
                    "mean": float("nan"),
                    "best": float("nan"),
                    "worst": float("nan"),
                }
            )
            continue

        rows.append(
            {
                "metric": m,
                "count": int(valid.shape[0]),
                "mean": float(valid.mean()),
                "best": float(valid.min()),
                "worst": float(valid.max()),
            }
        )

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract small-scale metrics and compute summary stats."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="ExperimentsResults/thesis_analysis/thesis_results_long.csv",
        help="Path to thesis_results_long.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="ExperimentsResults/thesis_analysis/small_scale_analysis",
        help="Output directory for detail and summary CSV files",
    )
    args = parser.parse_args()

    input_csv = (PROJECT_ROOT / args.input_csv).resolve()
    out_dir = ensure_dir((PROJECT_ROOT / args.out_dir).resolve())

    small = load_and_filter_small(input_csv)
    detail = build_detail_table(small)
    summary = build_summary_table(detail)

    detail_path = out_dir / "small_scale_metrics_detail.csv"
    summary_path = out_dir / "small_scale_metrics_summary.csv"

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"detail_csv={detail_path}")
    print(f"summary_csv={summary_path}")
    print(f"small_rows={len(detail)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
