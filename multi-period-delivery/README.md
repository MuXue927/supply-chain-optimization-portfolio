# Multi-period Production-driven Delivery Optimization

This project studies a production-driven, multi-period delivery problem with
in-warehouse inventory, scheduled-production inventory, and cross-period
inventory flow. The resulting inventory balance has a staircase structure and
couples delivery and loading decisions across periods.

The project is installable as the `multi_period_delivery` Python package:

```text
multi-period-delivery/
├── src/multi_period_delivery/
│   ├── alns/
│   ├── cplex_api_monolithic/
│   └── monolithic/
├── experiments/
├── scripts/
│   └── write_config.py
├── datasets/multiple-periods/small/dataset_1/
└── pyproject.toml
```

The main methods are:

- `multi_period_delivery.alns`: adaptive large neighborhood search.
- `monolithic`: integer-programming benchmark solved by Gurobi.
- `cplex_api_monolithic`: CPLEX Python API implementation of the monolithic
  benchmark.
- `experiments`: reproducible thesis experiment pipeline.

The checked-in CSV files under `datasets/multiple-periods/` are the
authoritative benchmark source. The shared generator has drifted from these
files, so regenerating with the documented seed is not a reproducible way to
reconstruct a historical dataset. Tight-capacity variants are derived from the
archived CSVs without changing any other input field.

`INVENTORY_HOLDING_COST` is an optional ALNS objective term, defaulting to
`0.0`. At a positive value it charges
`rate * sum(max(0, s_ikt) * sku_size)` for positive periods. The default keeps
archived objectives bit-for-bit unchanged; use a positive value only with the
tight-capacity variants, where it creates the cost signal that the surplus
inventory operator was designed to respond to.

For attribution experiments, `DISABLED_DESTROY_OPERATORS` (default: empty
tuple) omits named destroy operators from registration. An omitted operator
does not execute and does not consume a roulette draw; it is not the same as a
zero weight. The default preserves the historical operator set and trajectory.

The ALNS implementation includes 10 destroy operators, 9 repair operators,
plus a minimum-load fill post-processing pass,
adaptive operator weights, parameter tuning, feasibility repair, and
machine-learning-based operator selection.

## Problem Description

The full production-driven multi-period problem, inventory structure,
constraints, assumptions, and English problem diagram are documented here:

[`docs/problem_description.md`](docs/problem_description.md)

The complete English multi-period monolithic mathematical model, including
the route-local slot reformulation and the limitations of the legacy global
fleet index, is documented here:

[`docs/mathematical_models.md`](docs/mathematical_models.md)

The result, cost, constraint, and plot validation for `dataset_1` is documented
in:

[`docs/dataset_1_validation.md`](docs/dataset_1_validation.md)

The CPLEX-versus-ALNS objective reconciliation and over-delivery analysis are
documented in:

[`docs/objective_analysis.md`](docs/objective_analysis.md)

The full ALNS implementation review, root causes, fixes, and regression
results are documented in:

[`docs/alns_review.md`](docs/alns_review.md)

How this implementation relates to the thesis code, what was inherited, what
was changed, and why the revised ALNS performs better than the thesis baseline
is documented in:

[`docs/thesis_relationship.md`](docs/thesis_relationship.md)

## Requirements

- Python 3.10 or newer.
- A valid Gurobi installation and license for the original monolithic
  benchmark.
- PowerShell on Windows, or an equivalent terminal on Linux/macOS.
- The CPLEX Python API is the supported CPLEX route for this project. Install
  it with `uv pip install --python ..\.venv ".[cplex]"`.
  The extra pins `cplex==22.1.2.0` and `docplex==2.32.264`. The PyPI wheel
  ships the Community Edition, which refuses models beyond roughly 1,000
  variables or constraints (CPLEX error `1016`). Run
  `..\.venv\Scripts\docplex.exe config --upgrade "C:\Program Files\IBM\ILOG\CPLEX_Studio2212"`
  once to copy in the size-unlimited Studio runtime. See
  [`../docs/uv_environment.md`](../docs/uv_environment.md).

## CPLEX Python API

The `cplex_api_monolithic/` module is the Python-API version of the
multi-period monolithic model. It uses `import cplex`, `cplex.Cplex`, and
`SparsePair` directly.

```powershell
python -m multi_period_delivery.cplex_api_monolithic `
  --data-dir datasets/multiple-periods/small `
  --dataset-name dataset_1 `
  --output-dir OutPut-CPLEX-API/multiple-periods/small/dataset_1 `
  --time-limit 120 `
  --mip-gap 0 `
  --seed 1
```

The shared `uv` environment setup, including the CPLEX Community-to-Full runtime
upgrade, is documented in
[`../docs/uv_environment.md`](../docs/uv_environment.md).

The API model is exposed through a lazy package-level function:

```python
from multi_period_delivery import run_cplex_api_monolithic
```

## Install With pip

Run the following commands from the `multi-period-delivery` directory:

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

What each command does:

- `python -m venv .venv` creates a project-local virtual environment.
- `& .\.venv\Scripts\Activate.ps1` activates it in the current PowerShell.
- `python -m pip install --upgrade pip` updates pip.
- `python -m pip install -e .` installs the project in editable mode. The
  `multi_period_delivery` package becomes importable and the
  `multi-period-delivery`
  command becomes available.

If script activation is blocked in PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

This changes the execution policy only for the current PowerShell session.

## Verify the Installation

```powershell
python -c "import multi_period_delivery as mpd; print(mpd.__all__)"
multi-period-delivery --help
python -m multi_period_delivery --help
```

Explanation:

- The first command imports the installed package and prints its public API.
- The second command calls the console entry point created by `pip`.
- The third command runs the package through
  `src/multi_period_delivery/__main__.py`.

Recommended public imports:

```python
from multi_period_delivery import ALNSConfig, DataALNS, run_model
```

Run the package smoke tests:

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
```

The first command installs development dependencies and the second command
checks the command-line parser.

## Run the included sample

The included instance is:

```text
datasets/multiple-periods/small/dataset_1
```

Run the base ALNS configuration:

```powershell
multi-period-delivery --scale small --dataset-id 1 --max-runtime 60
```

What each option does:

- `--scale small` selects the `small` dataset directory.
- `--dataset-id 1` selects `dataset_1`.
- `--max-runtime 60` limits the run to 60 seconds.

The default data path is relative to the current directory:

```text
datasets/multiple-periods/small
```

The default output path is:

```text
OutPut-ALNS/multiple-periods/small/dataset_1
```

## Select an ALNS Variant

The thesis compares four variants:

| Variant     | Parameter tuner | ML operator selection | Command flags             |
| ----------- | --------------: | --------------------: | ------------------------- |
| `ALNS-B`  |              No |                    No | none                      |
| `ALNS-T`  |             Yes |                    No | `--enable-param-tuner`  |
| `ALNS-ML` |              No |                   Yes | `--enable-ml-operators` |
| `ALNS-F`  |             Yes |                   Yes | both flags                |

Examples:

```powershell
multi-period-delivery --enable-param-tuner
multi-period-delivery --enable-ml-operators
multi-period-delivery --enable-param-tuner --enable-ml-operators
```

The commands run `ALNS-T`, `ALNS-ML`, and `ALNS-F`, respectively.

## Run With Custom Paths

```powershell
multi-period-delivery `
  --scale medium `
  --dataset-id 3 `
  --data-dir datasets/multiple-periods/medium `
  --output-dir ExperimentsResults/medium `
  --max-runtime 360 `
  --seed 42 `
  --enable-param-tuner `
  --enable-ml-operators `
  --log-dir logs/medium_dataset_3
```

Explanation:

- `--data-dir` is the parent directory containing `dataset_*` folders.
- `--output-dir` is the run output root.
- `--max-runtime` is the stopping time in seconds.
- `--seed` controls stochastic operator selection and search behavior.
- `--enable-param-tuner` enables online parameter tuning.
- `--enable-ml-operators` enables the machine-learning operator selector.
- `--log-dir` selects the directory for the runtime log.

## Use the Package From Python

Configure the package before the first optimizer import:

```python
from multi_period_delivery import ALNSConfig, run_model

ALNSConfig.update_from_dict(
    {
        "DATASET_TYPE": "small",
        "DATASET_IDX": 1,
        "MAX_RUNTIME": 60,
        "SEED": 42,
        "ENABLE_PARAM_TUNER": True,
        "ENABLE_ML_OPERATORS": False,
        "LOG_DIR": "logs/small_dataset_1",
    }
)

run_model(
    input_file_loc="datasets/multiple-periods/small",
    output_file_loc="OutPut-ALNS/multiple-periods/small",
)
```

This runs the `ALNS-T` configuration. Set the two feature flags independently
to obtain `ALNS-B`, `ALNS-T`, `ALNS-ML`, or `ALNS-F`.

The optimizer reads selected configuration values when its implementation is
imported. For reproducible programmatic runs, update `ALNSConfig` before the
first call to `run_model`. The command-line interface handles this ordering
automatically.

## Experiment Pipeline

The thesis batch experiments, metric definitions, table construction, and
figure generation are documented in:

[`experiments/README.md`](experiments/README.md)

The pipeline covers:

- fixed-seed instance sampling;
- ILP and ALNS batch runs;
- small-scale `Gap_a_ILP` metrics;
- medium- and large-scale improvement metrics;
- thesis summary tables and representative convergence figures.

## Development Utilities

```powershell
python tools/generate_data.py --problem multi --periods 5 --scale small --count 3 --seed 20260408
python scripts/write_config.py --scale small
```

Explanation:

- `tools/generate_data.py` is the shared generator for both projects.
  `--problem multi --periods 5` creates multi-period instances with a
  five-period planning horizon.
- `scripts/write_config.py --scale small` writes `model_config.json` and
  `model_config_alns.json` into `datasets/multiple-periods/small`.

## Common Problems

`ModuleNotFoundError: No module named 'multi_period_delivery'`

- Activate the correct virtual environment.
- Run `python -m pip install -e .` from `multi-period-delivery`.

`FileNotFoundError` for `dataset_*`

- Check `--data-dir`, `--scale`, and `--dataset-id`.
- The repository ships a curated instance set: two small, two medium, and one
  large instance.

Gurobi error

- The ALNS path does not require Gurobi for its normal search.
- The monolithic benchmark requires a valid Gurobi installation and license.

Generated outputs, logs, full datasets, and local virtual environments are
excluded from Git.

## Experiments and Results

The portfolio experiment pipeline runs four ALNS variants on the curated
five-instance set and compares them with the archived thesis results on the
identical instances:

```powershell
python experiments/run_portfolio_experiments.py `
  --results-dir ../results/multi-period --workers 6
python experiments/build_portfolio_report.py
```

The summary tables and figures are committed under
[`../results/multi-period`](../results/multi-period). Raw per-run directories
are written to `results/multi-period/raw/` and are not committed.
