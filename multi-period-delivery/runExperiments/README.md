# Multi-period Experiment Pipeline

This directory contains the experiment pipeline used for the multi-period
production-driven delivery problem in the thesis. The scripts reproduce the
comparison of the integer linear programming (ILP) benchmark with four ALNS
variants:

| Thesis name | Parameter tuner | Machine-learning operator selection |
| ----------- | --------------: | ----------------------------------: |
| `ALNS-B`  |              No |                                  No |
| `ALNS-T`  |             Yes |                                  No |
| `ALNS-ML` |              No |                                 Yes |
| `ALNS-F`  |             Yes |                                 Yes |

The legacy names `basic`, `tuner`, `ml`, and `full` are still accepted as
aliases, but new experiments should use the thesis names above.

## Experimental Design

The thesis uses a stratified design with three scales:

| Scale  | Plants | Dealers |   SKUs | Vehicle types | Periods |
| ------ | -----: | ------: | -----: | ------------: | ------: |
| Small  |    2-3 |    2-30 |  10-40 |           2-3 |       5 |
| Medium |    3-4 |   30-50 |  40-60 |           2-4 |       5 |
| Large  |    4-6 |   50-90 | 60-100 |           2-4 |       5 |

For each scale, 30 candidate instances are generated and 20 instances are
sampled with a fixed random seed for the formal experiments. The sampling
manifest is stored so that the comparison is reproducible.

The time budgets are:

- Small-scale ILP comparison: 180 seconds.
- Medium-scale ALNS comparison: 360 seconds.
- Large-scale ALNS comparison: 720 seconds.

Small-scale experiments measure solution quality against the ILP objective.
Medium- and large-scale experiments measure improvement over the constructive
initial solution because the ILP model is not computationally practical for
the full experiment set.

## Evaluation Metrics

For small-scale instances, the thesis reports

```text
Gap_a_ILP = (Z_a_best - Z_ILP) / Z_ILP * 100%
```

where `Z_a_best` is the best feasible objective of an ALNS variant and `Z_ILP`
is the feasible ILP objective returned under the same time budget. A negative
value means that the ALNS solution is better than the ILP solution returned
within the time limit.

For medium- and large-scale instances, the thesis reports the relative
improvement over the initial solution:

```text
Imp_a = (Z_a_init - Z_a_best) / Z_a_init * 100%
```

Positive values represent cost reduction. The summary tables report the number
of instances, mean, median, and standard deviation.

The thesis does not use Wilcoxon tests or Dolan-More performance profiles in
the final multi-period experiment chapter. Scripts for those discarded
comparisons have been removed from this directory.

## Main Scripts

### `sample_and_run_experiments.py`

Samples a fixed number of `dataset_*` instances per scale, stores the selection
manifest, and calls `run_experiments.py` with the scale-specific variants and
time budgets.

```powershell
python runExperiments/sample_and_run_experiments.py `
  --sample-size 20 `
  --selection-seed 20260408
```

### `run_experiments.py`

Runs each `(instance, variant, seed)` combination and writes one output
directory per run. Each run contains metadata, status, objective values, and a
log for traceability.

```powershell
python runExperiments/run_experiments.py `
  --instances datasets/multiple-periods/medium `
  --variants ALNS-B,ALNS-T,ALNS-ML,ALNS-F `
  --seeds 1,2,3 `
  --max-runtime 360 `
  --out-dir ExperimentsResults/medium
```

### `prepare_thesis_results.py`

Scans the experiment output tree, reconstructs the initial-solution baseline
when required, normalizes run records, and writes the canonical analysis data:

- `thesis_results_long.csv`
- `thesis_summary_by_scale_variant.csv`
- `thesis_instance_long.csv`
- `thesis_instance_wide.csv`
- `thesis_analysis_manifest.json`

This script performs data preparation only. Tables and thesis figures are built
by the scripts below.

```powershell
python runExperiments/prepare_thesis_results.py `
  --results-root ExperimentsResults `
  --out-dir ExperimentsResults/thesis_analysis
```

### `extract_small_scale_metrics.py`

Extracts small-scale run-level metrics and reports the thesis statistics:
count, mean, median, and standard deviation.

```powershell
python runExperiments/extract_small_scale_metrics.py `
  --input-csv ExperimentsResults/thesis_analysis/thesis_results_long.csv `
  --out-dir ExperimentsResults/thesis_analysis/small_scale_analysis
```

### `build_small_gap_by_quadruple.py`

Builds the dataset-level table for `Gap_a_ILP`, keyed by the instance
quadruple `(plants, dealers, SKUs, vehicle types)`. This supports the detailed
small-scale Gap table in the thesis.

### `build_medium_large_improvement.py`

Builds the medium- and large-scale improvement tables, including:

- one row per instance and ALNS variant;
- scale-level mean, median, and standard deviation;
- wide tables for direct comparison across variants.

This script supports the thesis tables for improvement over the initial
solution.

### `build_thesis_tables_and_figures.py`

Builds the final thesis-oriented tables and figures from
`thesis_results_long.csv`:

- small-scale Gap statistics;
- medium- and large-scale improvement statistics;
- improvement boxplots;
- representative convergence-source SVGs for the three structural quadruples:
  - small: `(3,17,17,3)`
  - medium: `(4,46,54,4)`
  - large: `(6,70,93,3)`

```powershell
python runExperiments/build_thesis_tables_and_figures.py `
  --analysis-dir ExperimentsResults/thesis_analysis `
  --out-dir ExperimentsResults/thesis_analysis/core_artifacts
```

The convergence export copies one `Objective.svg` per ALNS variant. The
separate `restyle_selected_convergence_svgs.py` script is optional and applies
final thesis styling, font choices, and axis formatting.

## Recommended Workflow

1. Generate or place the small, medium, and large instance directories under
   `datasets/multiple-periods/`.
2. Run `sample_and_run_experiments.py` with the fixed selection seed.
3. Run `prepare_thesis_results.py` to produce the canonical long table.
4. Use `extract_small_scale_metrics.py` and
   `build_small_gap_by_quadruple.py` for the small-scale Gap tables.
5. Use `build_medium_large_improvement.py` for the medium- and large-scale
   improvement tables.
6. Run `build_thesis_tables_and_figures.py` for the final summary artifacts and
   representative convergence plots.

## Expected Run Output

Each run directory should contain:

- `run_metadata.json`
- `result.json`
- `log.txt`
- `images/Objective.svg` when convergence plotting is enabled

Generated outputs, full datasets, logs, and publication files are intentionally
excluded from the Git repository.
