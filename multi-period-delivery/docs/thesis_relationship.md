# Relationship to the Thesis Implementation

This project is a direct descendant of the ALNS code written for the master's
thesis on two-echelon production and distribution. The published repository is
that codebase, reorganized into an installable package and revised. This
document states exactly what was inherited, what was changed, and why the
revised ALNS produces better results than the thesis baseline **without
claiming that the thesis work was wrong**.

The correct framing is: the thesis implementation is a sound model and a
working metaheuristic, but it carries a set of identifiable defects and
limitations. The revision keeps the thesis design and removes those defects,
so the comparison is "same algorithm family, fixed implementation".

## Provenance

The revised multi-period code is derived from the thesis sources:

- the same mixed-integer formulation and data schema;
- the same constructive initial solution (`improved_initial_solution` and its
  sequential-allocation, repair, and strict-feasibility stages);
- the same destroy-operator families (random, Shaw, periodic Shaw, worst,
  surplus, path, cross-period dealer, period-range, infeasible, inventory
  level);
- the same repair-operator families (greedy, local search, inventory balance,
  smart batch, regret, fleet consolidation, sequential, lookahead);
- the same adaptive parameter tuner and machine-learning operator-selection
  framework;
- the same objective definition as the mathematical model.

What changed is packaging, correctness, and the search neighborhoods. The
public repository re-lays the code as an installable `src/` package, adds a
clean CLI, adds regression tests, and fixes the defect classes below.

## Defect Classes in the Thesis Baseline

### 1. The reported objective omitted the minimum-load penalty (metric defect)

The thesis objective is *vehicle fixed cost + underload penalty*, implemented
as `underload_vehicle_count * param_pun_objective`. The configuration leaves
`param_pun_objective` at its default of `0`, so the recorded
`best_actual_objective` was in practice the **fixed vehicle cost only**.

Concrete evidence from the archived thesis run on `medium/dataset_26`
(variant `basic`, seed 0): the archived plan contains 888 vehicle-period rows
and 183 vehicles below their minimum load. Its fixed cost is `889.22`, which is
exactly the thesis's reported `best_actual_objective`. Re-scored with the
minimum-load penalty (`alpha = 2`), the same plan is worth `1255.22`.

The revised code reads the penalty from configuration and, when the legacy
factor is unset, applies the intended `alpha * vehicle_cost` penalty. This is
the corrected objective used throughout this repository.

### 2. The search objective was mis-scaled and contained non-model terms (guidance defect)

The old search objective was scaled by `1e-3` while temperatures and
tolerances stayed in the original unit, and it added a volume-based underload
gap term and a fixed fleet-size term that are not part of the mathematical
objective. The ALNS was therefore partly optimizing something other than the
problem it reports.

The revised search objective uses the same unit as the actual objective
(`SCALE_FACTOR = 1.0`), keeps the finite unmet-demand penalty as the only extra
guidance term, and no longer adds the non-model penalties.

### 3. Destroyed intermediate states were scored as infeasible (learning defect)

`SolutionState.objective()` returned `inf` whenever demand was unmet. Destroy
operators routinely produce such states, so demand penalties, acceptance,
parameter learning, and ML rewards all received no usable signal in exactly the
states where guidance matters most.

The revised version returns a **finite** penalty for unmet demand while keeping
`inf` only for physical violations (negative inventory, vehicle overload, plant
capacity overflow). Final output still requires full physical feasibility.

### 4. Repair was incomplete and non-monotone (search defect)

Two problems compounded:

- `post_repair()` treated unmet demand as a hard violation and rolled back any
  partial but physically valid repair, discarding real progress.
- The minimum-load fill step was registered as a standalone repair operator,
  but it only fills existing underloaded vehicles and does not repair removed
  demand, so it frequently returned an incomplete state.

The revised version rolls back only on physical violations and runs the
minimum-load fill after every completed repair, so partial progress survives
and underloaded vehicles are consolidated away.

### 5. No cross-period consolidation neighborhood (structural defect)

Demand is aggregated across the planning horizon, so a load can legitimately
move from one period to another on the same plant-dealer route. The thesis
operators could reduce the load on individual vehicles but could not move an
entire load onto a same-route vehicle in a different period. That is the
dominant cost lever on medium and large instances, where vehicle-period count
drives the objective.

The revision adds the `cross_period_consolidation_repair` neighborhood. It is
registered as a first-class repair operator and also runs once before final
output, and it upgrades the target vehicle type when a merge requires it.

### 6. Implementation and caching defects (engineering defect)

Independent of the algorithm design, the thesis code contained defects that
corrupted individual evaluations: a wrong inventory namespace in
`inventory_level_removal` (named `inventory_level_destruction` in the thesis
code), division by remaining instead of rated
capacity, a Shaw off-by-one, a missing inventory restore in periodic Shaw,
stale objective caches, empty vehicles counted in fleet-size guidance, mutable
shared demand dictionaries, and a final output that could continue after the
selected incumbent failed validation. Round 2 additionally fixed integer
write-back in the Numba inventory kernel, ML operator-map mismatches, a
double-counting path in `post_destroy`, and an unbounded cross-period
consolidation call that could spend minutes inside a single repair.

Each fixed defect makes one more evaluation trustworthy; together they make the
search behave as designed.

## What Stayed the Same

| Area | Status |
| --- | --- |
| Mathematical model and constraints | unchanged |
| Dataset schema and generator | unchanged |
| Initial-solution procedure | unchanged structure |
| Destroy operator families | unchanged set, corrected implementations |
| Repair operator families | unchanged set, one added, corrected implementations |
| Parameter tuner and ML selector framework | unchanged design, hardened |
| Objective definition | unchanged formula, corrected evaluation |

## Why the Revised Version Performs Better

The improvements follow directly from the defect classes:

1. **The search now optimizes the reported objective.** Fixing the unit
   mismatch and removing the non-model penalties means accepted moves actually
   reduce the objective that is later reported.
2. **Adaptive components get real feedback.** Finite destroyed-state penalties
   make the tuner and ML selector see meaningful rewards instead of `inf`.
3. **Good partial solutions survive.** Not rolling back physically valid
   partial repairs, and filling minimum loads after each repair, keeps cheaper
   states in the population.
4. **A missing cost lever is now available.** Cross-period consolidation
   directly removes vehicle-period activations, which is why the largest gains
   appear on medium and large instances and the smallest on small ones.
5. **Every evaluation can be trusted.** The caching and accounting defects are
   exactly the kind that silently mislead a metaheuristic; removing them makes
   the search's own signals correct.

## How the Comparison Is Reported

The comparison in [`../../results/multi-period`](../../results/multi-period)
uses the thesis's five representative instances, the thesis's time budgets
(180 / 360 / 720 seconds), and seed 0, and reports two metrics:

- **published metric** (fixed vehicle cost only): this is the metric the thesis
  actually recorded, so both columns are directly comparable;
- **corrected metric** (fixed cost + minimum-load penalty): the thesis plans are
  re-scored under the corrected objective for reference.

On the published metric the revised ALNS is better on **20 of 20** runs (five
instances x four variants), with a mean fixed-cost reduction of about 209. On
the corrected metric it is also better on 20 of 20.

## Honest Caveats

- The revised implementation is **not** the thesis code; it is a reorganized,
  corrected descendant. The two should be described as "thesis baseline" and
  "revised implementation", not as the same program.
- The thesis recorded a tracker best that can differ from the plan it wrote to
  disk. The corrected-metric column therefore re-scores the archived plans as a
  reference; the published-metric comparison uses the thesis's own recorded
  numbers and does not depend on that.
- The comparison is on fixed instances and budgets. No claim is made that the
  revised ALNS is better on every possible instance, only on the benchmark set
  reported here.
