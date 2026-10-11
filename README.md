# Supply Chain Optimization Thesis Code

Research code for two related optimization problems in retail supply-chain
distribution. The repository is organized as two standalone Python projects:

- `single-period-delivery`: single-period, multi-plant, multi-dealer,
  multi-SKU delivery optimization with heterogeneous vehicles and minimum
  shipment quantities.
- `multi-period-delivery`: production-driven, multi-period delivery
  optimization with in-warehouse and scheduled-production inventory.

The projects implement exact optimization models, decomposition algorithms,
and metaheuristics. They are intended as reproducible research code and as a
portfolio of optimization, Python, and experiment-engineering work.

## Technical Highlights

### Single-period delivery

- Integer-programming formulation for multi-plant, multi-dealer delivery.
- Column generation with master and pricing subproblems.
- Two-stage heuristic for fast response scenarios.
- Monolithic Gurobi benchmark and sensitivity analysis.

### Multi-period delivery

- Multi-period model with in-warehouse inventory, production inventory, and
  cross-period inventory balance.
- Adaptive large neighborhood search with 10 destroy and 8 repair operators.
- Two-level adaptive parameter tuning and feasibility-repair callbacks.
- Ridge-regression and random-forest operator selectors.
- Experiment runners for small-, medium-, and large-scale instances.

## Repository Layout

```text
.
├── single-period-delivery/
│   ├── column-generation/
│   ├── heuristic/
│   ├── monolithic/
│   └── datasets/single-period/small/dataset_9/
└── multi-period-delivery/
    ├── ALNSCode/
    ├── monolithic/
    ├── runExperiments/
    └── datasets/multiple-periods/small/dataset_1/
```

Only small sample datasets are included in this portfolio repository. The
full experimental instances and generated output folders are intentionally
excluded.

## Reproducing

Each subproject has its own `pyproject.toml` and README with commands.
The multi-period project requires Python 3.10+, while the single-period
environment was locked with Python 3.13. A valid Gurobi installation is needed
for the mathematical-programming components. The ALNS code also uses the
public `alns` package.

Start with the project-level README:

- [`single-period-delivery/README.md`](single-period-delivery/README.md)
- [`multi-period-delivery/README.md`](multi-period-delivery/README.md)

## Research Context

The single-period project was developed for a study of two-level inventory
systems and product distribution. The multi-period project extends the model
with production-driven inventory dynamics and adaptive ALNS search. A version
of the single-period work was accepted for the LESCDT 2026 international
conference.

## Notes

- Do not commit credentials, cookies, private databases, virtual
  environments, or generated result folders.
- The repository intentionally contains code and small sample data rather
  than a copy of every local experiment artifact.
