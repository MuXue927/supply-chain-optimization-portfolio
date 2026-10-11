#!/usr/bin/env python3
"""
聚合实验输出脚本 (runExperiments/aggregator.py)

功能摘要:
- 遍历实验输出目录，收集每个 run 的 run_metadata.json 与可选的 per-run result JSON
- 生成汇总 CSV (aggregated_results.csv) 与分组统计 JSON (aggregated_stats.json)
- 支持按 instance / variant / time_limit 聚合并计算均值、中位数、std、样本量

依赖: pandas
用法示例:
    python scripts/aggregator.py --root OutPut-ALNS/experiments --out-dir results/aggregated
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

DEFAULT_META_NAME = "run_metadata.json"
POSSIBLE_RESULT_FILES = ["result.json", "results.json", "output.json"]


def find_run_dirs(root: Path) -> List[Path]:
    runs = []
    for p in root.rglob(DEFAULT_META_NAME):
        runs.append(p.parent)
    return sorted(runs)


def load_json_if_exists(p: Path) -> Dict[str, Any]:
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def collect_runs(root: Path) -> List[Dict[str, Any]]:
    run_dirs = find_run_dirs(root)
    rows = []
    for d in run_dirs:
        meta = load_json_if_exists(d / DEFAULT_META_NAME)
        # try to find richer per-run result JSON
        extra = {}
        for fname in POSSIBLE_RESULT_FILES:
            f = d / fname
            if f.exists():
                extra = load_json_if_exists(f)
                break
        row = {}
        row.update(meta if isinstance(meta, dict) else {})
        # merge some common fields from extra if absent
        for k in ("final_obj", "best_bound", "gap_pct", "time_series"):
            if k not in row and k in extra:
                row[k] = extra[k]
        # attach run_dir for traceability
        row["_run_dir"] = str(d)
        rows.append(row)
    return rows


def aggregate_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.json_normalize(rows)
    # normalize expected columns
    expected_cols = [
        "instance",
        "variant",
        "seed",
        "time_limit",
        "wall_seconds",
        "status",
        "final_obj",
        "best_bound",
        "gap_pct",
        "best_actual_objective",
        "best_search_objective",
        "total_iterations",
        "tracker_elapsed_seconds",
        "iter_per_second",
        "start_time",
        "end_time",
        "_run_dir",
    ]
    for c in expected_cols:
        if c not in df.columns:
            df[c] = pd.NA
    # ensure numeric types where appropriate
    for c in ("seed", "time_limit"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in (
        "wall_seconds",
        "final_obj",
        "best_bound",
        "gap_pct",
        "best_actual_objective",
        "best_search_objective",
        "tracker_elapsed_seconds",
        "iter_per_second",
    ):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[expected_cols]


def compute_group_stats(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    agg = (
        df.groupby(group_cols)
        .agg(
            n_runs=("seed", "count"),
            mean_obj=("final_obj", "mean"),
            median_obj=("final_obj", "median"),
            std_obj=("final_obj", "std"),
            mean_wall=("wall_seconds", "mean"),
            median_wall=("wall_seconds", "median"),
        )
        .reset_index()
    )
    return agg


def main():
    parser = argparse.ArgumentParser(description="Aggregate experiment run outputs.")
    parser.add_argument(
        "--root", type=str, default="OutPut-ALNS/experiments", help="实验输出根目录"
    )
    parser.add_argument(
        "--out-dir", type=str, default="results/aggregated", help="聚合结果输出目录"
    )
    parser.add_argument(
        "--group-by", type=str, default="instance,variant", help="按逗号分隔的聚合列"
    )
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_runs(root)
    df = aggregate_dataframe(rows)
    if df.empty:
        print("No runs found under", root)
        return

    # write aggregated csv
    agg_csv = out_dir / "aggregated_results.csv"
    df.to_csv(agg_csv, index=False)
    print("Wrote aggregated CSV:", agg_csv)

    # compute grouped stats
    group_cols = [c.strip() for c in args.group_by.split(",") if c.strip()]
    stats_df = compute_group_stats(df, group_cols)
    stats_csv = out_dir / "aggregated_stats.csv"
    stats_df.to_csv(stats_csv, index=False)
    print("Wrote aggregated stats CSV:", stats_csv)

    # write a small JSON summary
    summary = {
        "n_runs": int(df.shape[0]),
        "n_instances": int(df["instance"].nunique()) if "instance" in df.columns else 0,
        "n_variants": int(df["variant"].nunique()) if "variant" in df.columns else 0,
        "groups": group_cols,
    }
    (out_dir / "aggregated_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print("Wrote aggregated summary JSON")


if __name__ == "__main__":
    main()
