#!/usr/bin/env python3
"""
绘图与统计分析脚本 (scripts/plot_and_stats.py)

功能:
- 从 aggregator 输出的 aggregated_results.csv 读取结果
- 绘制箱线图 (final_obj per variant)、收敛曲线占位（若 time_series 可用则绘制）、以及 Dolan–Moré 性能剖面
- 进行成对 Wilcoxon signed-rank 检验并输出 p-values CSV（含 Holm 校正）
- 输出图像与统计表到指定输出目录

依赖:
- pandas, matplotlib, seaborn (可选), scipy

用法示例:
    python scripts/plot_and_stats.py --agg results/aggregated/aggregated_results.csv --out results/plots --alpha 0.05
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# optional imports
try:
    import seaborn as sns  # type: ignore
except Exception:
    sns = None

from itertools import combinations
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

def load_aggregated(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def plot_boxplot(df: pd.DataFrame, out_dir: Path, value_col: str = "final_obj"):
    plt.figure(figsize=(8,6))
    order = sorted(df["variant"].dropna().unique())
    if sns:
        sns.boxplot(data=df, x="variant", y=value_col, order=order)
    else:
        # fallback to matplotlib
        groups = [df[df["variant"]==v][value_col].dropna() for v in order]
        plt.boxplot(groups, labels=order)
    plt.ylabel(value_col)
    plt.xlabel("variant")
    plt.title(f"Distribution of {value_col} by variant")
    p = out_dir / "boxplot_final_obj.png"
    plt.tight_layout()
    plt.savefig(p, dpi=150)
    plt.close()
    return p

def performance_profile(df: pd.DataFrame, out_dir: Path, value_col: str = "final_obj"):
    # require instance + variant + final_obj
    sub = df[["instance", "variant", value_col]].dropna()
    # pivot to instance x variant
    pivot = sub.pivot_table(index="instance", columns="variant", values=value_col, aggfunc="median")
    # drop instances without any full data
    pivot = pivot.dropna(how="all")
    # compute best per instance (minimization)
    best = pivot.min(axis=1)
    # compute ratios
    ratios = pivot.div(best, axis=0)
    # replace zeros or negatives -> set to large number
    ratios = ratios.applymap(lambda x: np.nan if pd.isna(x) else (x if x > 0 else np.nan))
    # define tau grid
    taus = np.logspace(0, 2, 200)  # from 1 to 100
    profiles = {}
    for variant in ratios.columns:
        r = ratios[variant]
        vals = []
        for tau in taus:
            vals.append((r <= tau).sum() / float(r.shape[0] - r.isna().sum()))
        profiles[variant] = vals
    # plot
    plt.figure(figsize=(8,6))
    for v, vals in profiles.items():
        plt.plot(taus, vals, label=v)
    plt.xscale("log")
    plt.xlabel("Performance ratio (tau)")
    plt.ylabel("Proportion of instances with ratio <= tau")
    plt.title("Dolan–Moré performance profile")
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.5)
    p = out_dir / "performance_profile.png"
    plt.tight_layout()
    plt.savefig(p, dpi=150)
    plt.close()
    return p

def paired_wilcoxon(df: pd.DataFrame, out_dir: Path, value_col: str = "final_obj", alpha: float = 0.05):
    # prepare wide table: instances x variants (median over seeds)
    sub = df[["instance", "variant", "seed", value_col]].dropna()
    wide = sub.groupby(["instance","variant"])[value_col].median().unstack()
    variants = [c for c in wide.columns if c is not None]
    results = []
    pairs = list(combinations(variants, 2))
    pvals = []
    for a,b in pairs:
        xa = wide[a]
        xb = wide[b]
        # keep only instances where both exist
        mask = xa.notna() & xb.notna()
        if mask.sum() < 3:
            p = np.nan
            stat = np.nan
        else:
            try:
                stat, p = wilcoxon(xa[mask], xb[mask])
            except Exception:
                p = np.nan
                stat = np.nan
        results.append({"variant_a": a, "variant_b": b, "n_pairs": int(mask.sum()), "stat": stat, "pval": p})
        pvals.append(p if not pd.isna(p) else 1.0)
    # multiple testing correction (Holm)
    reject, pvals_corrected, _, _ = multipletests(pvals, alpha=alpha, method="holm")
    for i, r in enumerate(results):
        r["pval_adj"] = float(pvals_corrected[i])
        r["reject_null"] = bool(reject[i])
    out = out_dir / "wilcoxon_pairwise.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    # also write csv
    pd.DataFrame(results).to_csv(out_dir / "wilcoxon_pairwise.csv", index=False)
    return out

def main():
    parser = argparse.ArgumentParser(description="Plot and compute statistics for aggregated results.")
    parser.add_argument("--agg", type=str, required=True, help="aggregated_results.csv 路径")
    parser.add_argument("--out", type=str, default="results/plots", help="输出目录")
    parser.add_argument("--alpha", type=float, default=0.05, help="显著性水平")
    args = parser.parse_args()

    agg_path = Path(args.agg)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_aggregated(agg_path)
    if df.empty:
        print("Aggregated file empty:", agg_path)
        return

    # basic cleaning: keep succeeded runs
    df_ok = df[df["status"] == "success"].copy()
    if df_ok.empty:
        print("No successful runs in aggregated data")
        return

    boxplot_path = plot_boxplot(df_ok, out_dir)
    perf_path = performance_profile(df_ok, out_dir)
    wilcoxon_path = paired_wilcoxon(df_ok, out_dir, alpha=args.alpha)

    summary = {
        "boxplot": str(boxplot_path),
        "performance_profile": str(perf_path),
        "wilcoxon": str(wilcoxon_path),
        "n_runs": int(df_ok.shape[0]),
        "n_instances": int(df_ok["instance"].nunique()) if "instance" in df_ok.columns else 0,
        "n_variants": int(df_ok["variant"].nunique()) if "variant" in df_ok.columns else 0,
    }
    (out_dir / "plot_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Plots and stats written to", out_dir)

if __name__ == "__main__":
    main()
