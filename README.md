# Supply-Chain Distribution Optimization Portfolio

[![CI](https://github.com/MuXue927/supply-chain-optimization-portfolio/actions/workflows/ci.yml/badge.svg)](https://github.com/MuXue927/supply-chain-optimization-portfolio/actions/workflows/ci.yml)

Two standalone, reproducible Python projects for a retail supply-chain
distribution problem: shipping products from production plants to dealers with
heterogeneous vehicles, limited plant inventory, and minimum shipment
quantities.

- `single-period-delivery` - single-period, multi-plant, multi-dealer,
  multi-SKU delivery planning.
- `multi-period-delivery` - the production-driven extension with in-warehouse
  and scheduled-production inventory across a planning horizon.

Each project ships an installable package, an exact mixed-integer programming
model, an alternative algorithm, and a reproducible experiment pipeline. The
portfolio demonstrates optimization modelling, algorithm implementation,
experiment engineering, and Python packaging practice.

## What Each Project Contains

### Single-period delivery

- Monolithic integer-programming model (Gurobi).
- Column generation with explicit master and pricing subproblems.
- Fast two-stage constructive heuristic.
- CPLEX Python API implementations of both the monolithic and
  column-generation models.

### Multi-period delivery

- Production-driven multi-period model with in-warehouse inventory,
  production inventory, and cross-period inventory balance.
- Adaptive large neighborhood search (ALNS) with 10 destroy and 9 repair
  operators, a minimum-load fill post-processing pass, and a cross-period
  consolidation neighborhood.
- Online parameter tuning and machine-learning operator selection.
- CPLEX Python API implementation of the multi-period monolithic benchmark.

## Results

The curated experiment results are committed under [`results/`](results):
five instances per project across three scales, with summary tables and
figures. See [`results/README.md`](results/README.md) for the design and the
per-project READMEs for the headline numbers.

## Repository Layout

```text
.
├── docs/
│   └── uv_environment.md            # shared environment + CPLEX runtime notes
├── results/                          # committed summary tables and figures
├── tools/
│   └── generate_data.py              # shared data generator
├── single-period-delivery/
│   ├── src/single_period_delivery/
│   │   ├── monolithic/               # Gurobi integer-programming model
│   │   ├── column_generation/        # Gurobi column generation
│   │   ├── heuristic/                # two-stage constructive heuristic
│   │   ├── cplex_api_monolithic/     # CPLEX Python API monolithic model
│   │   └── cplex_api_column_generation/
│   ├── experiments/                  # experiment runner + report builder
│   ├── tests/
│   └── datasets/single-period/{small,medium,large}/
└── multi-period-delivery/
    ├── src/multi_period_delivery/
    │   ├── alns/                      # ALNS search implementation
    │   ├── monolithic/                # Gurobi multi-period model
    │   └── cplex_api_monolithic/      # CPLEX Python API benchmark
    ├── experiments/                   # experiment runner + report builder
    ├── tests/
    └── datasets/multiple-periods/{small,medium,large}/
```

The two project READMEs document the algorithms, the module map, and every
run command:

- [`single-period-delivery/README.md`](single-period-delivery/README.md)
- [`multi-period-delivery/README.md`](multi-period-delivery/README.md)

## Environment

Both projects form one [uv](https://docs.astral.sh/uv/) workspace: the
repository root declares the two members, pins Python 3.10 in
`.python-version` and locks every dependency in `uv.lock`. Full setup,
including the CPLEX Community-to-Full runtime upgrade and the solver-version
matching rules, is documented in
[`docs/uv_environment.md`](docs/uv_environment.md).

```powershell
uv sync                  # locked environment for both packages

# optional: CPLEX Python API (see docs/uv_environment.md)
uv sync --extra cplex
.\.venv\Scripts\docplex.exe config --upgrade "C:\Program Files\IBM\ILOG\CPLEX_Studio2212"
```

Gurobi requires a valid license; the ALNS, the heuristic, and the CPLEX Python
API paths do not.
## Testing

Run the full suite from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Both subprojects ship tests with the same file names (for example
`test_cli.py`, `test_cplex_api.py`), and a bare `pytest` from the root would
otherwise recurse into generated folders such as `outputs/`. The root
`pytest.ini` pins the collection scope and the import mode, so no extra flags
are needed. Each project also carries an equivalent
`[tool.pytest.ini_options]` block in its `pyproject.toml`, so running pytest
from inside a single project works the same way.

## Reproducing the Experiments

```powershell
# multi-period: 4 ALNS variants x 5 instances -> results/multi-period
.\.venv\Scripts\python.exe multi-period-delivery\experiments\run_portfolio_experiments.py `
  --results-dir results\multi-period --workers 6
.\.venv\Scripts\python.exe multi-period-delivery\experiments\build_portfolio_report.py

# single-period: heuristic + CPLEX Python API -> results/single-period
.\.venv\Scripts\python.exe single-period-delivery\experiments\run_portfolio_experiments.py `
  --results-dir results\single-period --workers 3
.\.venv\Scripts\python.exe single-period-delivery\experiments\build_portfolio_report.py
```

The shared data generator, used to create new instances in the same format, is
documented in [`tools/README.md`](tools/README.md).

## Data and Reproducibility

- The committed datasets are the five curated instances per project used by
  the experiment pipelines, plus the original sample instances.
- Raw per-run directories and console logs are written under
  `results/*/raw/` and are intentionally not committed.
- Generated output folders (`OutPut*/`, `outputs/`, `ExperimentsResults/`),
  virtual environments, and build artifacts are excluded from version control.

## Research Context

The single-period problem was developed for a study of two-level inventory
systems and product distribution; a version was accepted for the LESCDT 2026
international conference. The multi-period project extends the model with
production-driven inventory dynamics and adaptive ALNS search.

This repository is a reorganized and revised version of the original thesis
codebase. It keeps the thesis model, data schema, initial-solution procedure,
and operator families, and fixes the implementation defects and missing
neighborhoods described in
[`multi-period-delivery/docs/thesis_relationship.md`](multi-period-delivery/docs/thesis_relationship.md).

The complete English mathematical model references are:

- [single-period monolithic and column-generation models](single-period-delivery/docs/mathematical_models.md);
- [multi-period monolithic model](multi-period-delivery/docs/mathematical_models.md).

## Notes

- Do not commit credentials, cookies, private databases, virtual
  environments, or generated result folders.
- The repository ships code, small datasets, and curated result summaries
  rather than a copy of every local experiment artifact.
