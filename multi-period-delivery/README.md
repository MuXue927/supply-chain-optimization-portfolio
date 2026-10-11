# Multi-period Production-driven Delivery Optimization

This project extends the delivery optimization setting to multiple periods
with in-warehouse inventory, scheduled-production inventory, and cross-period
inventory flow. The resulting inventory balance has a staircase structure and
makes the routing and loading decisions coupled across periods.

## Implemented Methods

- `monolithic/`: compact mathematical model for benchmark comparisons.
- `ALNSCode/`: adaptive large neighborhood search implementation.
- `runExperiments/`: experiment orchestration and result aggregation.

The ALNS implementation includes 10 destroy operators, 8 repair operators,
adaptive operator weights, parameter tuning, feasibility repair, and
machine-learning-based operator selection.

## Setup

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

The project can be installed in editable mode from this directory. Gurobi is
required for the monolithic model, while the ALNS path can run independently
when its input data and configuration are available.

## Running

```powershell
python -m ALNSCode.main
python -m pytest -q
```

The small sample instance is under
`datasets/multiple-periods/small/dataset_1`.

## Experiment Pipeline

The `runExperiments/` folder contains scripts for:

- running repeated ALNS configurations;
- extracting small-, medium-, and large-scale metrics;
- comparing improvement distributions;
- preparing figures and tables for the thesis.

Generated outputs, logs, and full datasets are intentionally excluded from
this repository.
