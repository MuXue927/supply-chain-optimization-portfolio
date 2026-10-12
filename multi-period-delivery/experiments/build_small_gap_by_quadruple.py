#!/usr/bin/env python3
"""
Build small-scale per-dataset Gap summary table using dataset feature quadruples.

Inputs:
- selected_instances_small.csv (dataset key features)
- small_scale_metrics_detail.csv (per-run gap_vs_milp_pct)

Outputs:
1) gap_by_dataset_quadruple_wide.csv
   One row per dataset (identified by quadruple), with one column per ALNS method.
2) gap_by_dataset_quadruple_long.csv
   Long-format table suitable for plotting/reporting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _quadruple_str(row: pd.Series) -> str:
    return (
        f"({int(row['plant_nums'])},"
        f"{int(row['dealer_nums'])},"
        f"{int(row['sku_nums'])},"
        f"{int(row['vehicle_type_nums'])})"
    )


def load_selected_instances(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"selected instances file not found: {path}")

    sel = pd.read_csv(path)
    required = [
        "selection_order",
        "dataset_name",
        "plant_nums",
        "dealer_nums",
        "sku_nums",
        "vehicle_type_nums",
        "horizons",
    ]
    missing = [c for c in required if c not in sel.columns]
    if missing:
        raise ValueError(f"selected_instances_small.csv missing columns: {missing}")

    sel = sel.copy()
    sel["dataset_name"] = sel["dataset_name"].astype(str)
    sel["quadruple"] = sel.apply(_quadruple_str, axis=1)
    sel["quadruple_with_t"] = sel["quadruple"] + sel["horizons"].map(
        lambda t: f", T={int(t)}"
    )
    return sel


def load_gap_detail(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"small-scale detail file not found: {path}")

    df = pd.read_csv(path)
    required = ["dataset_name", "variant", "gap_vs_milp_pct", "scale"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"small_scale_metrics_detail.csv missing columns: {missing}")

    df = df.copy()
    df["scale"] = df["scale"].astype(str).str.lower()
    df = df[df["scale"] == "small"].copy()
    df["dataset_name"] = df["dataset_name"].astype(str)
    df["variant"] = df["variant"].astype(str).str.lower()
    df["gap_vs_milp_pct"] = pd.to_numeric(df["gap_vs_milp_pct"], errors="coerce")

    # Keep ALNS variants only.
    variant_map = {
        "basic": "ALNS-B",
        "alns-b": "ALNS-B",
        "full": "ALNS-F",
        "alns-f": "ALNS-F",
        "ml": "ALNS-ML",
        "alns-ml": "ALNS-ML",
        "tuner": "ALNS-T",
        "alns-t": "ALNS-T",
    }
    df = df[df["variant"].isin(variant_map.keys())].copy()
    df["method"] = df["variant"].map(variant_map)
    return df


def build_tables(
    selected_df: pd.DataFrame, detail_df: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    # Aggregate by dataset/method in case there are multiple seeds.
    agg = (
        detail_df.groupby(["dataset_name", "method"], as_index=False)["gap_vs_milp_pct"]
        .mean()
        .rename(columns={"gap_vs_milp_pct": "gap_pct"})
    )

    long_df = selected_df[
        ["selection_order", "dataset_name", "quadruple", "quadruple_with_t"]
    ].merge(
        agg,
        on="dataset_name",
        how="left",
    )

    method_order = ["ALNS-B", "ALNS-F", "ALNS-ML", "ALNS-T"]
    long_df["method"] = pd.Categorical(
        long_df["method"], categories=method_order, ordered=True
    )
    long_df = long_df.sort_values(
        ["selection_order", "method"], na_position="last"
    ).reset_index(drop=True)

    gap_wide = agg.pivot_table(
        index="dataset_name",
        columns="method",
        values="gap_pct",
        aggfunc="first",
        observed=False,
    ).reset_index()

    wide = selected_df[
        ["selection_order", "dataset_name", "quadruple", "quadruple_with_t"]
    ].merge(gap_wide, on="dataset_name", how="left")

    for m in method_order:
        if m not in wide.columns:
            wide[m] = np.nan

    wide = wide[
        [
            "selection_order",
            "dataset_name",
            "quadruple",
            "quadruple_with_t",
            *method_order,
        ]
    ]
    wide = wide.sort_values("selection_order").reset_index(drop=True)

    # Also provide best method per dataset (minimum gap is better).
    def _best_method(row: pd.Series) -> str:
        vals = row[method_order].dropna()
        if vals.empty:
            return ""
        return str(vals.idxmin())

    def _best_gap(row: pd.Series) -> float:
        vals = row[method_order].dropna()
        if vals.empty:
            return float("nan")
        return float(vals.min())

    wide["best_method"] = wide.apply(_best_method, axis=1)
    wide["best_gap_pct"] = wide.apply(_best_gap, axis=1)

    return {"long": long_df, "wide": wide}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build per-dataset Gap summary table represented by feature quadruples."
    )
    parser.add_argument(
        "--selected-csv",
        type=str,
        default="ExperimentsResults/selection_20260408_173620/selected_instances_small.csv",
        help="Path to selected_instances_small.csv",
    )
    parser.add_argument(
        "--detail-csv",
        type=str,
        default="ExperimentsResults/thesis_analysis/small_scale_analysis/small_scale_metrics_detail.csv",
        help="Path to small_scale_metrics_detail.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="ExperimentsResults/thesis_analysis/small_scale_analysis",
        help="Output directory",
    )
    args = parser.parse_args()

    selected_csv = (PROJECT_ROOT / args.selected_csv).resolve()
    detail_csv = (PROJECT_ROOT / args.detail_csv).resolve()
    out_dir = ensure_dir((PROJECT_ROOT / args.out_dir).resolve())

    selected_df = load_selected_instances(selected_csv)
    detail_df = load_gap_detail(detail_csv)
    tables = build_tables(selected_df, detail_df)

    long_path = out_dir / "gap_by_dataset_quadruple_long.csv"
    wide_path = out_dir / "gap_by_dataset_quadruple_wide.csv"

    tables["long"].to_csv(long_path, index=False)
    tables["wide"].to_csv(wide_path, index=False)

    print(f"long_csv={long_path}")
    print(f"wide_csv={wide_path}")
    print(f"dataset_rows={tables['wide'].shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
