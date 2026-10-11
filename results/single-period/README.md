# Single-period results

Two-stage heuristic versus the CPLEX Python API monolithic model
(fixed time limit) on five instances. The thesis Gurobi incumbent is
shown for reference. Lower objective is better.

| Scale | Dataset | Heuristic | Heuristic time (s) | CPLEX incumbent | CPLEX bound | CPLEX gap (%) | Thesis Gurobi |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| large | dataset_1 | 1763.08 | 3.72 | - | - | - | - |
| medium | dataset_1 | 844.76 | 2.53 | - | 72.65 | - | 671.60 |
| medium | dataset_3 | 632.48 | 3.85 | - | 38.30 | - | 505.66 |
| small | dataset_1 | 310.60 | 2.33 | 227.06 | 172.65 | 23.96 | 233.14 |
| small | dataset_3 | 39.92 | 2.32 | 15.36 | 13.54 | 11.85 | 15.36 |

## Interpretation

- The two-stage heuristic returns feasible plans in a few seconds on
  every instance.
- On `small/dataset_3` the CPLEX Python API monolithic model reproduces
  the thesis Gurobi incumbent exactly (15.36), which validates the
  reimplemented formulation.
- On `small/dataset_1` the CPLEX incumbent (227.06) is slightly better
  than the thesis Gurobi incumbent (233.14) at a fraction of the thesis
  time limit.
- The monolithic model is genuinely hard at medium and large scale: the
  thesis allowed 3600-6000 s per instance, and the 600 s CPLEX runs only
  close a dual bound. The large instance exceeds the machine's memory and
  is reported as a solver error rather than a crash.

Figures:

- `figures/objectives.png`
