#!/usr/bin/env python3
"""Build the multi-period portfolio report from the experiment outputs.

Inputs:

* ``results/multi-period/runs.csv``             - fixed ALNS runs.
* ``results/multi-period/thesis_reference.csv`` - archived thesis results on
  the identical instances.

Outputs:

* ``results/multi-period/comparison.csv``       - side-by-side comparison.
* ``results/multi-period/figures/*.png``        - presentation figures.
* ``results/multi-period/README.md``            - human-readable summary.

Two metrics are reported:

* **published** - penalty-free fixed vehicle cost, which is the metric the
  thesis actually reported. This is the primary, like-for-like comparison.
* **corrected** - fixed cost plus the minimum-load penalty. The thesis metric
  omitted this penalty; the corrected column shows both sides re-scored under
  the corrected objective.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

VARIANTS = ["basic", "tuner", "ml", "full"]


def load_inputs(results_dir: Path) -> pd.DataFrame:
    runs = pd.read_csv(results_dir / "runs.csv")
    thesis = pd.read_csv(results_dir / "thesis_reference.csv")
    merged = runs.merge(
        thesis,
        on=["scale", "dataset_id", "variant"],
        how="left",
        suffixes=("", "_thesis"),
    )
    merged["delta_published"] = (
        merged["fixed_cost_objective"] - merged["thesis_published_objective"]
    ).round(3)
    merged["delta_corrected"] = (
        merged["final_objective"] - merged["thesis_rescored_corrected_objective"]
    ).round(3)
    return merged


def _instance_labels(df: pd.DataFrame):
    instances = (
        df[["scale", "dataset_id"]]
        .drop_duplicates()
        .sort_values(["scale", "dataset_id"])
    )
    return instances, [f"{r.scale}/{r.dataset_id}" for r in instances.itertuples()]


def figure_published(df: pd.DataFrame, out_path: Path) -> None:
    instances, labels = _instance_labels(df)
    x = range(len(labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(11, 5))
    for offset, variant in enumerate(VARIANTS):
        new_vals, thesis_vals = [], []
        for row in instances.itertuples():
            subset = df[
                (df.scale == row.scale)
                & (df.dataset_id == row.dataset_id)
                & (df.variant == variant)
            ]
            new_vals.append(
                float(subset["fixed_cost_objective"].iloc[0]) if not subset.empty else float("nan")
            )
            thesis_vals.append(
                float(subset["thesis_published_objective"].iloc[0])
                if not subset.empty
                else float("nan")
            )
        positions = [i + (offset - 1.5) * width for i in x]
        ax.bar(positions, new_vals, width, label=f"{variant} (new)")
        ax.plot(positions, thesis_vals, "k_", markersize=12, label="thesis" if offset == 0 else None)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Published objective, fixed cost (lower is better)")
    ax.set_title("Multi-period ALNS: new vs thesis on the published metric (markers = thesis)")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def figure_delta(df: pd.DataFrame, out_path: Path) -> None:
    instances, labels = _instance_labels(df)
    x = range(len(labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(11, 5))
    for offset, variant in enumerate(VARIANTS):
        deltas = []
        for row in instances.itertuples():
            subset = df[
                (df.scale == row.scale)
                & (df.dataset_id == row.dataset_id)
                & (df.variant == variant)
            ]
            deltas.append(
                float(subset["delta_published"].iloc[0]) if not subset.empty else float("nan")
            )
        positions = [i + (offset - 1.5) * width for i in x]
        colors = ["#2b8a3e" if (d == d and d < 0) else "#c92a2a" for d in deltas]
        ax.bar(positions, deltas, width, color=colors, edgecolor="black", linewidth=0.4)

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("New fixed cost - thesis fixed cost\n(negative = new is better)")
    ax.set_title("Multi-period ALNS: change vs thesis on the published metric")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_markdown(df: pd.DataFrame) -> str:
    lines = [
        "# Multi-period results",
        "",
        "The fixed ALNS is compared with the archived thesis ALNS on the same five",
        "instances (2 small, 2 medium, 1 large), the same seed, and the same time",
        "budgets. Lower is better; negative delta means the fixed ALNS is better.",
        "",
        "The revised ALNS is a descendant of the thesis code. It keeps the thesis",
        "model, data schema, initial solution, and operator families, and fixes the",
        "identifiable defects of the thesis baseline (objective accounting, search",
        "objective scaling, destroyed-state scoring, repair completeness, and a",
        "missing cross-period consolidation neighborhood). See",
        "`multi-period-delivery/docs/thesis_relationship.md` for the full account.",
        "",
        "## Published metric (fixed vehicle cost)",
        "",
        "This is the metric the thesis actually reported. The thesis code left the",
        "minimum-load penalty factor at zero, so its objective counted only the",
        "fixed vehicle cost. Both columns below use that same metric.",
        "",
        "| Scale | Dataset | Variant | New fixed cost | Thesis | Delta |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in df.sort_values(["scale", "dataset_id", "variant"]).itertuples():
        lines.append(
            f"| {row.scale} | {row.dataset_id} | {row.variant} | "
            f"{row.fixed_cost_objective:.2f} | {row.thesis_published_objective:.2f} | "
            f"{row.delta_published:+.2f} |"
        )
    wins = int((df["delta_published"] < 0).sum())
    total = int(df["delta_published"].notna().sum())
    mean_delta = df["delta_published"].mean()

    lines += [
        "",
        f"**Summary:** the fixed ALNS is better on **{wins}/{total}** runs, with a",
        f"mean fixed-cost change of **{mean_delta:+.2f}**.",
        "",
        "## Corrected metric (fixed cost + minimum-load penalty)",
        "",
        "The corrected objective adds `alpha * vehicle_cost` for every used vehicle",
        "below its minimum shipment quantity. The thesis numbers in this column are",
        "its written plans re-scored under the corrected objective, so they are a",
        "reference rather than the thesis's original headline.",
        "",
        "| Scale | Dataset | Variant | New corrected | Thesis plan re-scored | Delta |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in df.sort_values(["scale", "dataset_id", "variant"]).itertuples():
        lines.append(
            f"| {row.scale} | {row.dataset_id} | {row.variant} | "
            f"{row.final_objective:.2f} | {row.thesis_rescored_corrected_objective:.2f} | "
            f"{row.delta_corrected:+.2f} |"
        )

    lines += [
        "",
        "Figures:",
        "",
        "- `figures/final_objective_new_vs_thesis.png`",
        "- `figures/delta_vs_thesis.png`",
        "",
        "`summary.csv` reports, per scale and variant, the mean improvement of",
        "each run over its own initial solution under the corrected objective.",
        "",
        "The re-scored thesis column is produced by",
        "`experiments/extract_thesis_reference.py`, which reads the archived",
        "thesis run plans and applies the corrected objective.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=None)
    args = parser.parse_args()
    results_dir = args.results_dir or (
        Path(__file__).resolve().parents[2] / "results" / "multi-period"
    )
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_inputs(results_dir)
    df.to_csv(results_dir / "comparison.csv", index=False)
    figure_published(df, figures_dir / "final_objective_new_vs_thesis.png")
    figure_delta(df, figures_dir / "delta_vs_thesis.png")
    (results_dir / "README.md").write_text(build_markdown(df), encoding="utf-8")
    print(f"Wrote comparison.csv, figures, and README.md under {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
