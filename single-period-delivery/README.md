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
│   └── column_generation/
├── datasets/single-period/small/dataset_9/
├── GeneratingData.py
├── params_config.py
└── pyproject.toml
```

The package supports three algorithms:

- `monolithic`: direct integer-programming model solved by Gurobi.
- `heuristic`: fast two-stage constructive heuristic.
- `column-generation`: master problem and pricing subproblem implementation.

## Requirements

- Python 3.11 or newer.
- A valid Gurobi installation and license for the monolithic and
  column-generation algorithms.
- PowerShell on Windows, or an equivalent terminal on Linux/macOS.

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

The source tree also contains scripts used to generate instances and write
solver configuration:

```powershell
python GeneratingData.py --help
python params_config.py
```

Explanation:

- `GeneratingData.py` creates synthetic instances for the configured scale.
- `params_config.py` writes a model configuration JSON file into the selected
  dataset directory.

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
