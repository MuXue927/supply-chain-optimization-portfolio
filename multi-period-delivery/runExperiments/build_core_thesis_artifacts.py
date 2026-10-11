#!/usr/bin/env python3
"""
Build core thesis artifacts from prepared thesis_analysis outputs.

Required outputs:
1) Table 1 (small): gap statistics vs MILP baseline
2) Table 2 (medium/large): improvement vs initial-solution statistics
3) Figure 1 (small): gap boxplot
4) Figure 2 (medium): improvement boxplot
5) Figure 3 (large): improvement boxplot
6) Figure 4 (representative instance): convergence curve (prefer time-axis source;
   fallback to existing Objective.svg exported by run)

Usage example:
  python runExperiments/build_core_thesis_artifacts.py \
    --analysis-dir ExperimentsResults/thesis_analysis \
    --out-dir ExperimentsResults/thesis_analysis/core_artifacts
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def load_results_long(analysis_dir: Path) -> pd.DataFrame:
    path = analysis_dir / "thesis_results_long.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    df = pd.read_csv(path)

    for col in [
        "best_actual_objective",
        "improvement_pct",
        "wall_seconds",
        "final_obj",
        "gap_pct",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["scale", "variant", "dataset_name", "status", "run_dir"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    if "status" in df.columns:
        df = df[df["status".strip()] == "success"].copy()

    if "variant" in df.columns:
        df["variant"] = df["variant"].str.lower()
    if "scale" in df.columns:
        df["scale"] = df["scale"].str.lower()

    return df


def is_mono_variant(v: str) -> bool:
    return v in {"monolithic", "gurobi", "mono"}


def compute_small_gap_vs_milp(df: pd.DataFrame) -> pd.DataFrame:
    small = df[df["scale"] == "small"].copy()
    if small.empty:
        return pd.DataFrame()

    mono = small[small["variant"].map(is_mono_variant)].copy()
    mono_obj = pd.to_numeric(mono.get("best_actual_objective"), errors="coerce")
    mono_obj = mono_obj.where(
        np.isfinite(mono_obj), pd.to_numeric(mono.get("final_obj"), errors="coerce")
    )
    mono = mono.assign(_mono_obj=mono_obj)
    mono = mono[np.isfinite(mono["_mono_obj"])].copy()
    if mono.empty:
        fallback = small[~small["variant"].map(is_mono_variant)].copy()
        fallback["gap_to_milp_pct"] = pd.to_numeric(
            fallback.get("gap_pct"), errors="coerce"
        )
        fallback = fallback[np.isfinite(fallback["gap_to_milp_pct"])].copy()
        return fallback

    milp_by_dataset = mono.groupby("dataset_name")["_mono_obj"].median().to_dict()

    alns = small[~small["variant"].map(is_mono_variant)].copy()
    if alns.empty:
        return pd.DataFrame()

    def _gap(row: pd.Series) -> float:
        milp = safe_float(milp_by_dataset.get(str(row["dataset_name"])))
        best_obj = safe_float(row.get("best_actual_objective"))
        if not np.isfinite(milp) or abs(milp) < 1e-12 or not np.isfinite(best_obj):
            return float("nan")
        return (best_obj - milp) / abs(milp) * 100.0

    alns["gap_to_milp_pct"] = alns.apply(_gap, axis=1)
    alns = alns[np.isfinite(alns["gap_to_milp_pct"])].copy()
    if alns.empty:
        fallback = small[~small["variant"].map(is_mono_variant)].copy()
        fallback["gap_to_milp_pct"] = pd.to_numeric(
            fallback.get("gap_pct"), errors="coerce"
        )
        fallback = fallback[np.isfinite(fallback["gap_to_milp_pct"])].copy()
        return fallback
    return alns


def build_table1_small_gap(small_gap_df: pd.DataFrame) -> pd.DataFrame:
    if small_gap_df.empty:
        return pd.DataFrame()
    t1 = (
        small_gap_df.groupby("variant")["gap_to_milp_pct"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
        .rename(
            columns={
                "variant": "method",
                "count": "n",
                "mean": "gap_mean_pct",
                "median": "gap_median_pct",
                "std": "gap_std_pct",
            }
        )
        .sort_values("method")
    )
    return t1


def build_table2_medium_large_improvement(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["scale"].isin(["medium", "large"])].copy()
    sub = sub[~sub["variant"].map(is_mono_variant)].copy()
    sub = sub[np.isfinite(sub["improvement_pct"])].copy()
    if sub.empty:
        return pd.DataFrame()

    t2 = (
        sub.groupby(["scale", "variant"])["improvement_pct"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
        .rename(
            columns={
                "scale": "scale",
                "variant": "method",
                "count": "n",
                "mean": "improvement_mean_pct",
                "median": "improvement_median_pct",
                "std": "improvement_std_pct",
            }
        )
        .sort_values(["scale", "method"])
    )
    return t2


def plot_boxplot(
    df: pd.DataFrame,
    *,
    value_col: str,
    out_path: Path,
    title: str,
    ylabel: str,
) -> Optional[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if df.empty or value_col not in df.columns or "variant" not in df.columns:
        return None

    variants = sorted(v for v in df["variant"].dropna().unique())
    values: List[List[float]] = []
    labels: List[str] = []
    for v in variants:
        arr = pd.to_numeric(
            df.loc[df["variant"] == v, value_col], errors="coerce"
        ).dropna()
        if arr.empty:
            continue
        labels.append(str(v))
        values.append(arr.to_list())

    if not values:
        return None

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(values, tick_labels=labels, showmeans=True)
    ax.set_title(title)
    ax.set_xlabel("Method")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_small_gap_boxplot(
    df: pd.DataFrame,
    out_path: Path,
) -> Optional[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import Patch
    from matplotlib import rcParams

    if df.empty or "gap_to_milp_pct" not in df.columns or "variant" not in df.columns:
        return None

    rcParams["svg.fonttype"] = "none"
    rcParams["mathtext.fontset"] = "stix"

    simsun_path = Path(r"C:\Windows\Fonts\simsun.ttc")
    times_path = Path(r"C:\Windows\Fonts\times.ttf")
    chinese_font = (
        FontProperties(fname=str(simsun_path))
        if simsun_path.exists()
        else FontProperties(family="SimSun")
    )
    english_font = (
        FontProperties(fname=str(times_path))
        if times_path.exists()
        else FontProperties(family="Times New Roman")
    )

    variant_order = ["basic", "full", "ml", "tuner"]
    variant_labels = {
        "basic": "ALNS-B",
        "full": "ALNS-F",
        "ml": "ALNS-ML",
        "tuner": "ALNS-T",
    }
    warm_palette = ["#f6c1a5", "#f3d2a8", "#f7b7a3", "#f8d6b3"]

    values: List[List[float]] = []
    labels: List[str] = []
    facecolors: List[str] = []
    for idx, variant in enumerate(variant_order):
        series = pd.to_numeric(
            df.loc[df["variant"] == variant, "gap_to_milp_pct"], errors="coerce"
        ).dropna()
        if series.empty:
            continue
        values.append(series.to_list())
        labels.append(variant_labels.get(variant, variant))
        facecolors.append(warm_palette[idx % len(warm_palette)])

    if not values:
        return None

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    bp = ax.boxplot(
        values,
        patch_artist=True,
        tick_labels=labels,
        showmeans=True,
        meanprops={
            "marker": "^",
            "markerfacecolor": "#2ca02c",
            "markeredgecolor": "#2ca02c",
            "markersize": 7,
        },
        medianprops={"color": "#d97706", "linewidth": 2.0},
        whiskerprops={"color": "#666666", "linewidth": 1.4},
        capprops={"color": "#666666", "linewidth": 1.4},
        flierprops={
            "marker": "o",
            "markerfacecolor": "#999999",
            "markeredgecolor": "#999999",
            "markersize": 3.5,
            "alpha": 0.6,
        },
    )

    for patch, facecolor in zip(bp["boxes"], facecolors):
        patch.set_facecolor(facecolor)
        patch.set_edgecolor("#8b5e3c")
        patch.set_linewidth(1.4)

    ax.set_xlabel("算法类别", fontsize=13, fontproperties=chinese_font)
    ax.set_ylabel(
        r"相对差异指标$Gap_{a}^{ILP}$(%)", fontsize=13, fontproperties=chinese_font
    )
    ax.tick_params(axis="both", labelsize=11)
    for label in ax.get_xticklabels():
        label.set_fontproperties(english_font)
    for label in ax.get_yticklabels():
        label.set_fontproperties(english_font)
    ax.grid(False)

    legend_handles = [
        Patch(facecolor="#f6c1a5", edgecolor="#8b5e3c", label="箱体"),
        Line2D([0], [0], color="#666666", linewidth=1.4, label="须线"),
        Line2D(
            [0],
            [0],
            color="#d97706",
            linewidth=2.0,
            label="中位数",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor="#2ca02c",
            markeredgecolor="#2ca02c",
            markersize=7,
            label="均值",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        fontsize=10,
        prop=chinese_font,
    )

    fig.tight_layout()
    fig.savefig(out_path, format="svg", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _variant_priority(v: str) -> int:
    order = {"full": 0, "ml": 1, "tuner": 2, "basic": 3}
    return order.get(v, 99)


def find_representative_row(df: pd.DataFrame) -> Optional[pd.Series]:
    # Prefer medium scale for a representative curve; fallback to large, then small.
    candidates = df[~df["variant"].map(is_mono_variant)].copy()
    if candidates.empty:
        return None

    scales = ["medium", "large", "small"]
    for scale in scales:
        s = candidates[candidates["scale"] == scale].copy()
        if s.empty:
            continue

        cover = (
            s.groupby("dataset_name")["variant"].nunique().sort_values(ascending=False)
        )
        if cover.empty:
            continue
        best_dataset = str(cover.index[0])

        dsub = s[s["dataset_name"] == best_dataset].copy()
        dsub = dsub.sort_values(
            by=["variant", "best_actual_objective"],
            key=lambda col: col.map(_variant_priority)
            if col.name == "variant"
            else col,
            na_position="last",
        )
        if not dsub.empty:
            return dsub.iloc[0]
    return None


def export_representative_convergence(
    df: pd.DataFrame, out_dir: Path
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "source": None,
        "output_svg": None,
        "output_png": None,
        "note": None,
    }
    row = find_representative_row(df)
    if row is None:
        out["note"] = "No representative run found."
        return out

    run_dir = Path(str(row.get("run_dir", "")))
    dataset_name = str(row.get("dataset_name", ""))
    source_svg = run_dir / dataset_name / "images" / "Objective.svg"

    if not source_svg.exists():
        out["note"] = f"Objective.svg not found for representative run: {run_dir}"
        return out

    target_svg = out_dir / "figure4_representative_convergence.svg"
    shutil.copy2(source_svg, target_svg)

    out["source"] = str(source_svg)
    out["output_svg"] = str(target_svg)
    out["note"] = (
        "Used existing Objective.svg from representative run. "
        "If tracker time-series CSV becomes available, this can be replaced by true time-axis redraw."
    )

    try:
        import cairosvg  # type: ignore

        target_png = out_dir / "figure4_representative_convergence.png"
        cairosvg.svg2png(url=str(target_svg), write_to=str(target_png), dpi=180)
        out["output_png"] = str(target_png)
    except Exception:
        pass

    return out


def write_csv(df: pd.DataFrame, path: Path) -> Optional[Path]:
    if df.empty:
        return None
    df.to_csv(path, index=False)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build core thesis tables/figures from thesis_analysis outputs."
    )
    parser.add_argument(
        "--analysis-dir",
        type=str,
        default="ExperimentsResults/thesis_analysis",
        help="Directory containing thesis_results_long.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output directory. Defaults to <analysis-dir>/core_artifacts",
    )
    args = parser.parse_args()

    analysis_dir = (PROJECT_ROOT / args.analysis_dir).resolve()
    if not analysis_dir.exists():
        raise FileNotFoundError(f"analysis-dir not found: {analysis_dir}")

    out_dir = (
        (PROJECT_ROOT / args.out_dir).resolve()
        if args.out_dir.strip()
        else analysis_dir / "core_artifacts"
    )
    ensure_dir(out_dir)

    df = load_results_long(analysis_dir)
    if df.empty:
        raise RuntimeError("Input results are empty after filtering successful runs.")

    small_gap_df = compute_small_gap_vs_milp(df)
    table1 = build_table1_small_gap(small_gap_df)
    table2 = build_table2_medium_large_improvement(df)

    p_table1 = write_csv(table1, out_dir / "table1_small_gap_stats.csv")
    p_table2 = write_csv(table2, out_dir / "table2_medium_large_improvement_stats.csv")

    fig1 = plot_small_gap_boxplot(
        small_gap_df,
        out_dir / "figure1_small_gap_boxplot.svg",
    )

    medium_df = df[
        (df["scale"] == "medium")
        & (~df["variant"].map(is_mono_variant))
        & (np.isfinite(df["improvement_pct"]))
    ].copy()
    fig2 = plot_boxplot(
        medium_df,
        value_col="improvement_pct",
        out_path=out_dir / "figure2_medium_improvement_boxplot.png",
        title="Medium-scale Improvement over Initial Solution",
        ylabel="Improvement over Initial Solution (%)",
    )

    large_df = df[
        (df["scale"] == "large")
        & (~df["variant"].map(is_mono_variant))
        & (np.isfinite(df["improvement_pct"]))
    ].copy()
    fig3 = plot_boxplot(
        large_df,
        value_col="improvement_pct",
        out_path=out_dir / "figure3_large_improvement_boxplot.png",
        title="Large-scale Improvement over Initial Solution",
        ylabel="Improvement over Initial Solution (%)",
    )

    fig4 = export_representative_convergence(df, out_dir)

    manifest = {
        "inputs": {
            "analysis_dir": str(analysis_dir),
            "results_long": str(analysis_dir / "thesis_results_long.csv"),
        },
        "outputs": {
            "table1_small_gap_stats": str(p_table1) if p_table1 else None,
            "table2_medium_large_improvement_stats": str(p_table2)
            if p_table2
            else None,
            "figure1_small_gap_boxplot": str(fig1) if fig1 else None,
            "figure2_medium_improvement_boxplot": str(fig2) if fig2 else None,
            "figure3_large_improvement_boxplot": str(fig3) if fig3 else None,
            "figure4_representative_convergence": fig4,
        },
        "notes": [
            "Table 1 uses gap computed against per-instance monolithic objective in small scale.",
            "Table 2 uses improvement_pct from thesis_results_long.csv for medium/large.",
            "Figure 1 is exported as SVG with Chinese labels, SimSun for Chinese glyphs, and Times New Roman for English/math text.",
            "Figure 4 currently reuses Objective.svg from representative run when no explicit tracker time-series CSV is available.",
        ],
    }

    manifest_path = out_dir / "core_artifacts_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
