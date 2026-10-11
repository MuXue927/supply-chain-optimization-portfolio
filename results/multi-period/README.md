# Multi-period results

The fixed ALNS is compared with the archived thesis ALNS on the same five
instances (2 small, 2 medium, 1 large), the same seed, and the same time
budgets. Lower is better; negative delta means the fixed ALNS is better.

The revised ALNS is a descendant of the thesis code. It keeps the thesis
model, data schema, initial solution, and operator families, and fixes the
identifiable defects of the thesis baseline (objective accounting, search
objective scaling, destroyed-state scoring, repair completeness, and a
missing cross-period consolidation neighborhood). See
`multi-period-delivery/docs/thesis_relationship.md` for the full account.

## Published metric (fixed vehicle cost)

This is the metric the thesis actually reported. The thesis code left the
minimum-load penalty factor at zero, so its objective counted only the
fixed vehicle cost. Both columns below use that same metric.

| Scale | Dataset | Variant | New fixed cost | Thesis | Delta |
| --- | ---: | --- | ---: | ---: | ---: |
| large | 3 | basic | 2356.59 | 2801.20 | -444.61 |
| large | 3 | full | 2355.85 | 3031.54 | -675.69 |
| large | 3 | ml | 2355.61 | 3031.54 | -675.93 |
| large | 3 | tuner | 2354.89 | 3036.91 | -682.02 |
| medium | 13 | basic | 901.50 | 1127.10 | -225.60 |
| medium | 13 | full | 906.03 | 1105.68 | -199.65 |
| medium | 13 | ml | 903.54 | 1105.68 | -202.14 |
| medium | 13 | tuner | 899.68 | 1182.12 | -282.44 |
| medium | 26 | basic | 748.04 | 889.22 | -141.18 |
| medium | 26 | full | 730.60 | 879.52 | -148.92 |
| medium | 26 | ml | 753.91 | 879.52 | -125.61 |
| medium | 26 | tuner | 722.87 | 847.41 | -124.54 |
| small | 12 | basic | 37.84 | 63.00 | -25.16 |
| small | 12 | full | 38.90 | 71.00 | -32.10 |
| small | 12 | ml | 36.50 | 71.00 | -34.50 |
| small | 12 | tuner | 38.08 | 50.00 | -11.92 |
| small | 30 | basic | 139.20 | 185.57 | -46.37 |
| small | 30 | full | 147.72 | 160.20 | -12.48 |
| small | 30 | ml | 138.61 | 160.20 | -21.59 |
| small | 30 | tuner | 142.11 | 204.41 | -62.30 |

**Summary:** the fixed ALNS is better on **20/20** runs, with a
mean fixed-cost change of **-208.74**.

## Corrected metric (fixed cost + minimum-load penalty)

The corrected objective adds `alpha * vehicle_cost` for every used vehicle
below its minimum shipment quantity. The thesis numbers in this column are
its written plans re-scored under the corrected objective, so they are a
reference rather than the thesis's original headline.

| Scale | Dataset | Variant | New corrected | Thesis plan re-scored | Delta |
| --- | ---: | --- | ---: | ---: | ---: |
| large | 3 | basic | 2650.59 | 3456.91 | -806.32 |
| large | 3 | full | 2643.85 | 3453.54 | -809.69 |
| large | 3 | ml | 2647.61 | 3453.54 | -805.93 |
| large | 3 | tuner | 2636.89 | 3456.91 | -820.02 |
| medium | 13 | basic | 951.50 | 1296.12 | -344.62 |
| medium | 13 | full | 956.03 | 1310.94 | -354.91 |
| medium | 13 | ml | 953.54 | 1310.94 | -357.40 |
| medium | 13 | tuner | 915.68 | 1296.12 | -380.44 |
| medium | 26 | basic | 928.04 | 1255.22 | -327.18 |
| medium | 26 | full | 898.60 | 1241.52 | -342.92 |
| medium | 26 | ml | 937.91 | 1241.52 | -303.61 |
| medium | 26 | tuner | 926.87 | 1255.22 | -328.35 |
| small | 12 | basic | 51.84 | 163.00 | -111.16 |
| small | 12 | full | 56.90 | 163.00 | -106.10 |
| small | 12 | ml | 44.50 | 163.00 | -118.50 |
| small | 12 | tuner | 54.08 | 94.00 | -39.92 |
| small | 30 | basic | 145.20 | 350.93 | -205.73 |
| small | 30 | full | 155.72 | 214.90 | -59.18 |
| small | 30 | ml | 150.61 | 214.90 | -64.29 |
| small | 30 | tuner | 156.11 | 350.93 | -194.82 |

Figures:

- `figures/final_objective_new_vs_thesis.png`
- `figures/delta_vs_thesis.png`

`summary.csv` reports, per scale and variant, the mean improvement of
each run over its own initial solution under the corrected objective.

The re-scored thesis column is produced by
`experiments/extract_thesis_reference.py`, which reads the archived
thesis run plans and applies the corrected objective.
