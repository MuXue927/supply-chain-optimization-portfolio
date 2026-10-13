# Single-period results

Two-stage heuristic versus the CPLEX Python API monolithic model
(fixed time limit) on five instances. The thesis Gurobi incumbent is
shown for reference. Lower objective is better.

| Scale | Dataset | Heuristic | Heuristic time (s) | CPLEX incumbent | CPLEX bound | CPLEX gap (%) | Thesis Gurobi |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| large | dataset_1 | 1763.08 | 3.72 | 1028.00 | 992.25 | 3.48 | - |
| medium | dataset_1 | 844.76 | 2.53 | 601.96 | 584.55 | 2.89 | 671.60 |
| medium | dataset_3 | 632.48 | 3.85 | 456.88 | 442.85 | 3.07 | 505.66 |
| small | dataset_1 | 310.60 | 2.33 | 179.40 | 178.92 | 0.27 | 233.14 |
| small | dataset_3 | 39.92 | 2.32 | 15.06 | 15.06 | 0.00 | 15.36 |

## Interpretation

- The two-stage heuristic returns feasible plans in a few seconds on
  every instance.
- On `small/dataset_3` the route-local monolithic model proves the
  optimum at 15.06 in about 2.4 seconds, improving on the thesis Gurobi
  incumbent of 15.36.
- On `small/dataset_1` the route-local model reaches 179.40 with a 0.27%
  gap, also improving on the thesis incumbent of 233.14.
- The medium and large runs now produce feasible incumbents instead of
  stopping without an integer solution or exhausting memory. With
  1800-3600 second budgets their remaining optimality gaps are about
  2.9-3.5%.

Figures:

- `figures/objectives.png`
