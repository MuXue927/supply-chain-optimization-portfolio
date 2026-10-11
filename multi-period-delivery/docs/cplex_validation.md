# CPLEX Python API Validation

## Purpose

The `cplex_api_monolithic/` package is an independent CPLEX Python API
implementation of the multi-period monolithic model. It uses `cplex.Cplex`
and `SparsePair` directly and does not modify the thesis-aligned Gurobi
`monolithic/` package. The shared environment setup, including the
Community-to-Full runtime upgrade, is documented in
[`../../docs/uv_environment.md`](../../docs/uv_environment.md).

## Reproduction Command

Run from `multi-period-delivery` with the project environment active:

```powershell
python -m multi_period_delivery.cplex_api_monolithic `
  --data-dir datasets/multiple-periods/small `
  --dataset-name dataset_1 `
  --output-dir OutPut-CPLEX-API/multiple-periods/small/dataset_1 `
  --time-limit 120 `
  --mip-gap 0 `
  --seed 1
```

## Verified Result

| Item | Result |
| --- | ---: |
| Solver status | `time_limit_feasible` |
| Time limit | 120 seconds |
| Objective | 38.57 |
| Delivery records | 195 |
| Distinct vehicles | 25 |
| Vehicle-period records | 35 |
| Demand shortage | 0 |
| Demand over-delivery | 12 |
| Maximum over-delivery | 5 |
| Inventory-balance mismatches | 0 |
| Negative inventory | 0 |
| Plant-capacity violations | 0 |
| Vehicle-capacity violations | 0 |
| Below-minimum-load vehicles | 0 |
| Recomputed objective | 38.57 |

The independent checks recompute the objective from the result CSV files and
verify the inventory staircase, plant capacity, vehicle capacity, and demand
coverage directly. The result is a time-limited feasible incumbent, not a
zero-gap certificate or evidence that a previously reported objective is
globally optimal.

## Model Consistency

The Python API model was constructed independently from the reviewed
multi-period formulation and checked before solving: 41,136 variables and
3,944 constraints, with objective coefficients, variable bounds, and
constraint contents matched against the reviewed model.
