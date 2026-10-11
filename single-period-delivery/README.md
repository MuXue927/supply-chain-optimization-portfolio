# Single-period Delivery Optimization

This project studies a single-period, two-level retail distribution problem
with multiple plants, multiple dealers, multiple SKUs, heterogeneous vehicle
types, and minimum shipment quantities.

The code is organized as an installable Python package:

```text
single-period-delivery/
├── src/single_period_delivery/
│   ├── monolithic/
│   ├── heuristic/
│   ├── column_generation/
│   ├── cplex_api_monolithic/
│   ├── cplex_api_column_generation/
│   └── cplex_api_support.py
├── scripts/
│   └── write_config.py
├── datasets/single-period/small/dataset_9/
└── pyproject.toml
```

The package supports three algorithms:

- `monolithic`: direct integer-programming model solved by Gurobi.
- `heuristic`: fast two-stage constructive heuristic.
- `column-generation`: master problem and pricing subproblem implementation.

## Problem Description

The full research problem, system structure, constraints, assumptions, and
English problem diagram are documented here:

[`docs/problem_description.md`](docs/problem_description.md)

The static code review and heuristic result checks are documented in:

[`docs/code_review.md`](docs/code_review.md)

## Requirements

- Python 3.10 or newer. Python 3.10 is the recommended interpreter when
  using the official CPLEX Python API because the matching CPLEX wheel is
  available for Python 3.10.
- A valid Gurobi installation and license for the original monolithic and
  column-generation algorithms.
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

`cplex_api_monolithic/` and `cplex_api_column_generation/` are the
Python-API versions of the same models. They use `import cplex`,
`cplex.Cplex`, and `SparsePair` directly instead of writing an LP file.

```powershell
python -m single_period_delivery.cplex_api_monolithic `
  --data-dir datasets/single-period/small `
  --dataset-name dataset_9 `
  --output-dir OutPut-CPLEX-API/single-period/small/dataset_9 `
  --time-limit 180 `
  --mip-gap 0 `
  --seed 1
```

```powershell
python -m single_period_delivery.cplex_api_column_generation `
  --data-dir datasets/single-period/small `
  --dataset-name dataset_9 `
  --output-dir OutPut-CPLEX-API-CG/single-period/small/dataset_9 `
  --max-iterations 20 `
  --final-time-limit 60 `
  --seed 20260408
```

The complete shared `uv` environment setup, including the CPLEX
Community-to-Full runtime upgrade, is documented in the portfolio-level
[`docs/uv_environment.md`](../docs/uv_environment.md).

The API models are exposed through lazy package-level functions:

```python
from single_period_delivery import (
    run_cplex_api_column_generation,
    run_cplex_api_monolithic,
)
```

## Install With pip

Run the following commands from the `single-period-delivery` directory:

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

What each command does:

- `python -m venv .venv` creates an isolated virtual environment in the
  `.venv` folder.
- `& .\.venv\Scripts\Activate.ps1` activates that environment. After this,
  `python` and `pip` refer to the virtual environment.
- `python -m pip install --upgrade pip` upgrades the package installer.
- `python -m pip install -e .` installs this project in editable mode. The
  package becomes importable, the `single-period-delivery` command becomes
  available, and later source edits take effect without reinstalling.

If PowerShell blocks script activation, run this once in the current
PowerShell window:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

This changes the policy only for the current terminal session.

## Verify the Installation

```powershell
python -c "import single_period_delivery as spd; print(spd.__version__)"
single-period-delivery --help
python -m single_period_delivery --help
```

What each command does:

- The first command imports the installed package and prints its version.
- The second command calls the console entry point created by `pip`.
- The third command uses the package's `__main__.py`; it is equivalent to the
  console command and is useful when testing the package directly.

Check the Gurobi installation when using an exact solver:

```powershell
python -c "import gurobipy; print(gurobipy.gurobi.version())"
```

If this command imports successfully but solving later reports a license
error, configure an academic or commercial Gurobi license.

Run the package smoke tests:

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
```

The first command installs the development extras, including pytest and Ruff.
The second command checks the package entry point.

## Run the included sample

The sample instance is:

```text
datasets/single-period/small/dataset_9
```

Run the three algorithms from the project root:

```powershell
single-period-delivery monolithic
single-period-delivery heuristic --experiments 1
single-period-delivery column-generation
```

The defaults select `datasets/single-period/small` and `dataset_9`.

What each command does:

- `monolithic` loads the sample instance, constructs the integer-programming
  model, solves it with Gurobi, and writes result CSV files.
- `heuristic` runs the two-stage heuristic for one repetition by default.
  Increase `--experiments` for repeated stochastic runs.
- `column-generation` builds the master and pricing models and writes
  column-generation result files.

Default output directories:

- Monolithic: `OutPut/single-period/small/dataset_9`
- Heuristic: `OutPut-Heuristic/single-period/small/dataset_9`
- Column generation: `OutPut-CG/single-period/small/dataset_9`

## Run a different instance

Use `--data-dir`, `--output-dir`, and `--dataset-name` to override defaults:

```powershell
single-period-delivery monolithic `
  --data-dir datasets/single-period/large `
  --output-dir OutPut/single-period/large `
  --dataset-name dataset_30
```

Explanation:

- `--data-dir` is the parent directory that contains `dataset_*` folders.
- `--output-dir` is the parent directory for result folders.
- `--dataset-name` selects the concrete instance directory, such as
  `dataset_30`.

The same arguments are available for `heuristic` and `column-generation`.
The heuristic also accepts `--experiments`:

```powershell
single-period-delivery heuristic `
  --data-dir datasets/single-period/large `
  --output-dir OutPut-Heuristic/single-period/large `
  --dataset-name dataset_30 `
  --experiments 50
```

This runs the heuristic 50 times and records the repeated-run results.

## Use the package from Python

```python
from single_period_delivery import (
    run_column_generation,
    run_heuristic,
    run_monolithic,
)

run_monolithic(
    input_file_loc="datasets/single-period/large",
    output_file_loc="OutPut/single-period/large",
    dataset_name="dataset_30",
)

run_heuristic(
    input_file_loc="datasets/single-period/large",
    output_file_loc="OutPut-Heuristic/single-period/large",
    dataset_name="dataset_30",
    exp_nums=50,
)

run_column_generation(
    input_file_loc="datasets/single-period/large",
    output_file_loc="OutPut-CG/single-period/large",
    dataset_name="dataset_30",
)
```

The three exported functions load the selected algorithm lazily, so importing
the package itself does not immediately import Gurobi.

## Development Utilities

Dataset generation is shared at the monorepo root:

```powershell
python tools/generate_data.py --problem single --scale small --count 3 --seed 20260408
python scripts/write_config.py --scale small
```

Explanation:

- `tools/generate_data.py` is the shared generator for both projects.
  `--problem single` writes single-period instances into this project's
  `datasets/single-period/small` directory.
- `scripts/write_config.py --scale small` writes `model_config.json` and
  `model_config_cg.json` into `datasets/single-period/small`.

These are development utilities and are not required for importing or using
the installed package.

## Common Problems

`ModuleNotFoundError: No module named 'single_period_delivery'`

- Activate the correct virtual environment.
- Run `python -m pip install -e .` from the `single-period-delivery` directory.

`gurobipy` import or license error

- Install Gurobi and configure a valid license before running the exact
  algorithms.

`FileNotFoundError` for a CSV file

- Check that `--data-dir` points to the parent directory containing
  `dataset_*`, not to the `dataset_*` directory itself.
- Check that `--dataset-name` matches the actual folder name.

Generated output, virtual environments, and caches are excluded from Git.

## Experiments and Results

The portfolio experiment pipeline runs the two-stage heuristic and the CPLEX
Python API monolithic model on the curated five-instance set:

```powershell
python experiments/run_portfolio_experiments.py `
  --results-dir ../results/single-period --workers 3
python experiments/build_portfolio_report.py
```

The summary table and figure are committed under
[`../results/single-period`](../results/single-period). Raw per-run
directories are written to `results/single-period/raw/` and are not committed.
