#!/usr/bin/env python3
"""
Build medium/large improvement summaries for ALNS variants.

Input:
- one or more thesis_results_long.csv files produced by prepare_thesis_results.py

Output:
1) detail CSV: one row per medium/large ALNS run with improvement_pct
2) summary CSV: per-scale, per-variant statistics for improvement_pct
3) wide CSV: one row per dataset with ALNS variant improvement means
4) per-scale detail/summary/wide CSVs for medium and large individually
5) per-scale instance tables: one row per instance with ALNS variant improvements

Metric definition:
    improvement_pct = (initial_actual_objective - best_actual_objective)
                      / abs(initial_actual_objective) * 100

Positive values mean the ALNS solution improves on the initial solution.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def canonical_variant(value: Any) -> str:
    aliases = {
        "basic": "ALNS-B",
        "alns-b": "ALNS-B",
        "full": "ALNS-F",
        "alns-f": "ALNS-F",
        "ml": "ALNS-ML",
        "alns-ml": "ALNS-ML",
        "tuner": "ALNS-T",
        "alns-t": "ALNS-T",
    }
    text = str(value).strip()
    return aliases.get(text.lower(), text)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def discover_selection_dir(results_root: Path) -> Optional[Path]:
    if not results_root.exists():
        return None
    candidates = [
        p
        for p in results_root.iterdir()
        if p.is_dir() and p.name.startswith("selection_")
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_selection_features(selection_dir: Optional[Path]) -> pd.DataFrame:
    if selection_dir is None:
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    for name in ["selected_instances_medium.csv", "selected_instances_large.csv"]:
        path = selection_dir / name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        needed = [
            "scale",
            "dataset_name",
            "plant_nums",
            "dealer_nums",
            "sku_nums",
            "vehicle_type_nums",
            "horizons",
        ]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            continue
        frames.append(df[needed].copy())

    if not frames:
        return pd.DataFrame()

    sel = pd.concat(frames, ignore_index=True)
    sel["scale"] = sel["scale"].astype(str).str.lower()
    sel["dataset_name"] = sel["dataset_name"].astype(str)
    sel["structure_quadruple"] = sel.apply(
        lambda r: (
            f"({int(r['plant_nums'])},{int(r['dealer_nums'])},"
            f"{int(r['sku_nums'])},{int(r['vehicle_type_nums'])})"
        ),
        axis=1,
    )
    sel["structure_quadruple_t"] = sel.apply(
        lambda r: f"{r['structure_quadruple']}, T={int(r['horizons'])}", axis=1
    )

    return sel[
        [
            "scale",
            "dataset_name",
            "structure_quadruple",
            "structure_quadruple_t",
        ]
    ].drop_duplicates(subset=["scale", "dataset_name"])


def safe_float(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        value = float(value)
        return value if math.isfinite(value) else float("nan")
    except Exception:
        return float("nan")


def load_results_long(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)
    required = ["scale", "dataset_name", "variant", "best_actual_objective"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in thesis_results_long.csv: {missing}"
        )

    for col in [
        "initial_actual_objective",
        "best_actual_objective",
        "improvement_pct",
        "gap_vs_initial_pct",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = float("nan")

    for col in ["scale", "dataset_name", "variant", "status", "run_dir"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower() == "success"].copy()

    df["scale"] = df["scale"].astype(str).str.lower()
    df["variant"] = df["variant"].astype(str).str.lower()
    return df


def recompute_improvement(row: pd.Series) -> float:
    initial_obj = safe_float(row.get("initial_actual_objective"))
    best_obj = safe_float(row.get("best_actual_objective"))
    if (
        not np.isfinite(initial_obj)
        or not np.isfinite(best_obj)
        or abs(initial_obj) < 1e-12
    ):
        return float("nan")
    return (initial_obj - best_obj) / abs(initial_obj) * 100.0


def load_and_filter_medium_large(results_csv: Path) -> pd.DataFrame:
    df = load_results_long(results_csv)
    df = df[df["scale"].isin(["medium", "large"])].copy()
    if df.empty:
        return df

    # Keep ALNS variants only.
    variant_order = ["basic", "full", "ml", "tuner"]
    df = df[df["variant"].isin(variant_order)].copy()
    if df.empty:
        return df

    missing_improvement = ~np.isfinite(df["improvement_pct"])
    if missing_improvement.any():
        df.loc[missing_improvement, "improvement_pct"] = df.loc[
            missing_improvement
        ].apply(recompute_improvement, axis=1)

    df["gap_vs_initial_pct"] = np.where(
        np.isfinite(df["improvement_pct"]),
        -df["improvement_pct"],
        float("nan"),
    )

    return df.sort_values(
        ["scale", "dataset_name", "variant", "seed"], na_position="last"
    ).reset_index(drop=True)


def load_and_filter_many(results_csvs: List[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in results_csvs:
        frame = load_and_filter_medium_large(path)
        if not frame.empty:
            frame = frame.copy()
            frame["source_file"] = str(path)
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_summary(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()

    summary = (
        detail.groupby(["scale", "variant"])["improvement_pct"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "count": "n",
                "mean": "improvement_mean_pct",
                "median": "improvement_median_pct",
                "std": "improvement_std_pct",
                "min": "best_pct",
                "max": "worst_pct",
                "variant": "method",
            }
        )
        .sort_values(["scale", "method"])
        .reset_index(drop=True)
    )
    return summary


def build_wide(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()

    per_dataset = (
        detail.groupby(["scale", "dataset_name", "variant"], as_index=False)[
            "improvement_pct"
        ]
        .mean()
        .rename(columns={"improvement_pct": "improvement_mean_pct"})
    )
    per_dataset["method"] = per_dataset["variant"].map(canonical_variant)

    wide = per_dataset.pivot_table(
        index=["scale", "dataset_name"],
        columns="method",
        values="improvement_mean_pct",
        aggfunc="first",
        observed=False,
    ).reset_index()

    for method in ["ALNS-B", "ALNS-F", "ALNS-ML", "ALNS-T"]:
        if method not in wide.columns:
            wide[method] = np.nan

    wide = wide[["scale", "dataset_name", "ALNS-B", "ALNS-F", "ALNS-ML", "ALNS-T"]]

    def _best_method(row: pd.Series) -> str:
        vals = row[["ALNS-B", "ALNS-F", "ALNS-ML", "ALNS-T"]].dropna()
        if vals.empty:
            return ""
        return str(vals.idxmax())

    def _best_improvement(row: pd.Series) -> float:
        vals = row[["ALNS-B", "ALNS-F", "ALNS-ML", "ALNS-T"]].dropna()
        if vals.empty:
            return float("nan")
        return float(vals.max())

    wide["best_method"] = wide.apply(_best_method, axis=1)
    wide["best_improvement_pct"] = wide.apply(_best_improvement, axis=1)
    return wide


def build_instance_wide(
    detail: pd.DataFrame, selection_features: pd.DataFrame
) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()

    method_labels = {
        "ALNS-B": "ALNS-B",
        "ALNS-F": "ALNS-F",
        "ALNS-ML": "ALNS-ML",
        "ALNS-T": "ALNS-T",
    }

    keys = ["scale", "dataset_name"]
    if "instance" in detail.columns:
        keys.append("instance")

    sub = detail[keys + ["variant", "improvement_pct"]].copy()
    sub["method"] = sub["variant"].map(canonical_variant).map(method_labels)

    instance_wide = sub.pivot_table(
        index=keys,
        columns="method",
        values="improvement_pct",
        aggfunc="mean",
        observed=False,
    ).reset_index()

    for method in ["ALNS-B", "ALNS-F", "ALNS-ML", "ALNS-T"]:
        if method not in instance_wide.columns:
            instance_wide[method] = np.nan

    if not selection_features.empty:
        instance_wide = instance_wide.merge(
            selection_features,
            on=["scale", "dataset_name"],
            how="left",
        )

    ordered_cols: List[str] = ["scale"]
    if "structure_quadruple" in instance_wide.columns:
        ordered_cols.append("structure_quadruple")
    if "structure_quadruple_t" in instance_wide.columns:
        ordered_cols.append("structure_quadruple_t")
    if "instance" in instance_wide.columns:
        ordered_cols.append("instance")
    ordered_cols += ["ALNS-B", "ALNS-F", "ALNS-ML", "ALNS-T"]
    # dataset_name is intentionally not included in the final table.
    instance_wide = instance_wide[ordered_cols]
    return instance_wide


def build_scale_tables(
    detail: pd.DataFrame, selection_features: pd.DataFrame
) -> Dict[str, Dict[str, pd.DataFrame]]:
    tables: Dict[str, Dict[str, pd.DataFrame]] = {}
    for scale in ["medium", "large"]:
        sub = detail[detail["scale"] == scale].copy()
        if sub.empty:
            continue
        tables[scale] = {
            "detail": sub,
            "summary": build_summary(sub),
            "wide": build_wide(sub),
            "instance_wide": build_instance_wide(sub, selection_features),
        }
    return tables


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute ALNS improvement over initial solution for medium and large scales."
    )
    parser.add_argument(
        "--results-csv",
        type=str,
        nargs="+",
        default=["ExperimentsResults/thesis_analysis/thesis_results_long.csv"],
        help="One or more paths to thesis_results_long.csv files",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="ExperimentsResults/thesis_analysis/medium_large_analysis",
        help="Output directory for improvement tables",
    )
    parser.add_argument(
        "--selection-dir",
        type=str,
        default="",
        help="Optional selection_* directory containing selected_instances_medium/large.csv",
    )
    args = parser.parse_args()

    results_csvs = [(PROJECT_ROOT / p).resolve() for p in args.results_csv]
    out_dir = ensure_dir((PROJECT_ROOT / args.out_dir).resolve())

    if args.selection_dir.strip():
        selection_dir = (PROJECT_ROOT / args.selection_dir).resolve()
    else:
        selection_dir = discover_selection_dir(PROJECT_ROOT / "ExperimentsResults")
    selection_features = load_selection_features(selection_dir)

    detail = load_and_filter_many(results_csvs)
    summary = build_summary(detail)
    wide = build_wide(detail)
    scale_tables = build_scale_tables(detail, selection_features)

    detail_path = out_dir / "medium_large_improvement_detail.csv"
    summary_path = out_dir / "medium_large_improvement_summary.csv"
    wide_path = out_dir / "medium_large_improvement_wide.csv"

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    wide.to_csv(wide_path, index=False)

    for scale, tables in scale_tables.items():
        tables["detail"].to_csv(
            out_dir / f"{scale}_improvement_detail.csv", index=False
        )
        tables["summary"].to_csv(
            out_dir / f"{scale}_improvement_summary.csv", index=False
        )
        tables["wide"].to_csv(out_dir / f"{scale}_improvement_wide.csv", index=False)
        tables["instance_wide"].to_csv(
            out_dir / f"{scale}_improvement_by_instance.csv", index=False
        )

    print(f"detail_csv={detail_path}")
    print(f"summary_csv={summary_path}")
    print(f"wide_csv={wide_path}")
    print(f"detail_rows={len(detail)}")
    print(f"summary_rows={len(summary)}")
    print(f"wide_rows={len(wide)}")
    for scale, tables in scale_tables.items():
        print(f"{scale}_detail_rows={len(tables['detail'])}")
        print(f"{scale}_summary_rows={len(tables['summary'])}")
        print(f"{scale}_wide_rows={len(tables['wide'])}")
        print(f"{scale}_instance_rows={len(tables['instance_wide'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
