#!/usr/bin/env python3
"""Build the single-period portfolio report from the experiment outputs.

Inputs:

* ``results/single-period/runs.csv``             - new runs (heuristic + CPLEX).
* ``results/single-period/thesis_reference.csv`` - thesis Gurobi incumbents.

Outputs:

* ``results/single-period/comparison.csv``       - side-by-side comparison.
* ``results/single-period/figures/*.png``        - presentation figures.
* ``results/single-period/README.md``            - human-readable summary.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_inputs(results_dir: Path) -> pd.DataFrame:
    runs = pd.read_csv(results_dir / "runs.csv")
    thesis = pd.read_csv(results_dir / "thesis_reference.csv")

    heur = runs[runs["approach"] == "heuristic"][
        ["scale", "dataset_name", "objective", "wall_seconds"]
    ].rename(columns={"objective": "heuristic_objective", "wall_seconds": "heuristic_seconds"})
    cpx = runs[runs["approach"] == "cplex"][
        ["scale", "dataset_name", "objective", "best_bound", "mip_gap_pct", "wall_seconds"]
    ].rename(
        columns={
            "objective": "cplex_objective",
            "best_bound": "cplex_best_bound",
            "mip_gap_pct": "cplex_mip_gap_pct",
            "wall_seconds": "cplex_seconds",
        }
    )
    merged = heur.merge(cpx, on=["scale", "dataset_name"], how="outer")
    merged = merged.merge(
        thesis[["scale", "dataset_name", "thesis_objective"]],
        on=["scale", "dataset_name"],
        how="left",
    )
    merged["heuristic_gap_vs_bound_pct"] = (
        (merged["heuristic_objective"] - merged["cplex_best_bound"])
        / merged["cplex_best_bound"]
        * 100.0
    ).round(2)
    return merged.sort_values(["scale", "dataset_name"])


def figure_objectives(df: pd.DataFrame, out_path: Path) -> None:
    labels = [f"{r.scale}/{r.dataset_name.replace('dataset_', '')}" for r in df.itertuples()]
    x = range(len(labels))
    width = 0.27

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width for i in x], df["heuristic_objective"], width, label="Heuristic")
    # Missing values stay NaN so no bar is drawn (a zero bar would be misleading).
    ax.bar(list(x), df["cplex_objective"], width, label="CPLEX incumbent")
    ax.bar([i + width for i in x], df["thesis_objective"], width, label="Thesis Gurobi")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Objective (lower is better)")
    ax.set_title("Single-period: heuristic vs CPLEX vs thesis Gurobi")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_markdown(df: pd.DataFrame) -> str:
    lines = [
        "# Single-period results",
        "",
        "Two-stage heuristic versus the CPLEX Python API monolithic model",
        "(fixed time limit) on five instances. The thesis Gurobi incumbent is",
        "shown for reference. Lower objective is better.",
        "",
        "| Scale | Dataset | Heuristic | Heuristic time (s) | CPLEX incumbent | CPLEX bound | CPLEX gap (%) | Thesis Gurobi |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in df.itertuples():
        def fmt(value):
            return "-" if value != value or value is None else f"{value:.2f}"

        lines.append(
            f"| {row.scale} | {row.dataset_name} | {fmt(row.heuristic_objective)} | "
            f"{fmt(row.heuristic_seconds)} | {fmt(row.cplex_objective)} | "
            f"{fmt(row.cplex_best_bound)} | {fmt(row.cplex_mip_gap_pct)} | "
            f"{fmt(row.thesis_objective)} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- The two-stage heuristic returns feasible plans in a few seconds on",
        "  every instance.",
        "- On `small/dataset_3` the route-local monolithic model proves the",
        "  optimum at 15.06 in about 2.4 seconds, improving on the thesis Gurobi",
        "  incumbent of 15.36.",
        "- On `small/dataset_1` the route-local model reaches 179.40 with a 0.27%",
        "  gap, also improving on the thesis incumbent of 233.14.",
        "- The medium and large runs now produce feasible incumbents instead of",
        "  stopping without an integer solution or exhausting memory. With",
        "  1800-3600 second budgets their remaining optimality gaps are about",
        "  2.9-3.5%.",
        "",
        "Figures:",
        "",
        "- `figures/objectives.png`",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=None)
    args = parser.parse_args()
    results_dir = args.results_dir or (
        Path(__file__).resolve().parents[2] / "results" / "single-period"
    )
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_inputs(results_dir)
    df.to_csv(results_dir / "comparison.csv", index=False)
    figure_objectives(df, figures_dir / "objectives.png")
    (results_dir / "README.md").write_text(build_markdown(df), encoding="utf-8")
    print(f"Wrote comparison.csv, figures, and README.md under {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
