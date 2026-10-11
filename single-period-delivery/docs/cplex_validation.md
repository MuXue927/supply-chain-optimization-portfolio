# CPLEX Python API Validation

## Purpose

The `cplex_api_monolithic/` and `cplex_api_column_generation/` packages are
independent CPLEX Python API implementations of the single-period models.
They use `cplex.Cplex` and `SparsePair` directly and do not modify the
thesis-aligned Gurobi packages. The shared environment setup, including the
Community-to-Full runtime upgrade, is documented in
[`../../docs/uv_environment.md`](../../docs/uv_environment.md).

## Reproduction Commands

Run the API monolithic model from `single-period-delivery`:

```powershell
python -m single_period_delivery.cplex_api_monolithic `
  --data-dir datasets/single-period/small `
  --dataset-name dataset_9 `
  --output-dir OutPut-CPLEX-API/single-period/small/dataset_9 `
  --time-limit 180 `
  --mip-gap 0 `
  --seed 1
```

Run the API column-generation model with the validated heuristic loads as an
initial master solution:

```powershell
python -m single_period_delivery.cplex_api_column_generation `
  --data-dir datasets/single-period/small `
  --dataset-name dataset_9 `
  --output-dir OutPut-CPLEX-API-CG/single-period/small/dataset_9 `
  --max-iterations 0 `
  --final-time-limit 60 `
  --seed-result-csv OutPut-Heuristic/single-period/small/dataset_9/opt_result.csv
```

## Verified Results

| Model | Status | Objective | Recomputed objective | Feasibility checks |
| --- | --- | ---: | ---: | --- |
| Monolithic | `time_limit_feasible` | 209.73 | 209.73 | 0 shortages, 0 capacity violations, 0 plant-inventory violations |
| Seeded column generation | `optimal` | 230.52 | 230.52 | 0 demand mismatches, 0 capacity violations, 0 plant-inventory violations |

The monolithic result is a feasible incumbent found within the time limit,
not a zero-gap optimality certificate. The seeded column-generation run starts
from a validated heuristic solution and reproduces its objective exactly; it
validates the corrected minimum-load penalty and objective accounting in the
master model.

## Model Consistency Checks

Before solving, the Python API model construction was checked against the
reviewed formulation:

| Model statistic | Value |
| --- | ---: |
| Variables | 100,928 |
| Constraints | 10,294 |
| Objective terms | 304 |

The variable-name set was consistent with the reviewed model, and the API
smoke test solved a small integer model with a positive incumbent. A
30-second monolithic run on `dataset_9` reached the time limit without an
integer incumbent, which is recorded here as a limitation of that time budget
rather than as a claim about the global optimum.
