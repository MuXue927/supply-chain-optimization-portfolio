# ALNS Implementation Review

> **Status note (2026-10-13).** References below to the `38.57` CPLEX
> incumbent describe the historical global vehicle-indexed validation run.
> The ALNS review and regression results remain the public record for the
> search path; the current monolithic formulation is documented in
> [`mathematical_models.md`](mathematical_models.md).

For the provenance of this code relative to the thesis implementation, and for
the explanation of why the revised ALNS outperforms the thesis baseline, see
[`thesis_relationship.md`](thesis_relationship.md).

## Scope

This review covers the complete multi-period ALNS path:

- data loading and normalization;
- solution state, inventory recursion, feasibility, and objective evaluation;
- destroy operators and shared destroy utilities;
- repair operators, initialization, consolidation, and minimum-load handling;
- ALNS acceptance, tracker, callbacks, stopping rules, parameter tuning, ML
  operator selection, output, and reporting.

The review distinguishes the **search objective** used to guide ALNS from the
**actual objective** used to evaluate the final mathematical-programming
solution. They must share the same units whenever possible.

## Root Causes Found

### 1. Intermediate destroyed states had an infinite objective

`SolutionState.objective()` returned `inf` when any demand was not fulfilled.
ALNS frequently evaluates a destroyed state before repair is complete. This
made demand penalties, underload guidance, parameter learning, and ML rewards
effectively disappear for those states.

**Fix**

- Hard violations remain `inf`: negative inventory, vehicle overload, and
  plant inventory capacity overflow.
- Unmet demand now produces a finite penalty.
- `actual_objective()` still requires full hard feasibility, so final output
  cannot use an infeasible solution.

### 2. Search and actual objective units differed

The old search objective was scaled by `1e-3`, while simulated-annealing
temperatures and several tolerances remained in the original unit. The search
objective also added a large volume-based underload gap penalty and an extra
fixed fleet penalty that are not part of the mathematical objective.

**Fix**

- `SCALE_FACTOR` is now `1.0`.
- The default search objective no longer adds a separate volume-based
  underload penalty.
- The default search objective no longer adds a separate fleet-size penalty.
- Once demand is feasible, the search objective equals the actual objective.
- The unmet-demand penalty remains the only additional guidance term.

### 3. Partial repairs were rolled back

`post_repair()` treated unmet demand as a hard violation. A time-limited
repair that made progress but did not finish all demand was therefore rolled
back completely.

**Fix**

- Only physical violations trigger rollback:
  negative inventory, vehicle overload, and plant capacity overflow.
- Unmet demand remains visible to the finite search penalty and is repaired
  by later repair operators.

### 4. The minimum-load fill operator was used as a standalone repair

`fill_minimum_load_repair()` only fills existing underloaded vehicles. It does
not repair demand removed by a destroy operator. As a normal repair operator,
it frequently returned an incomplete state.

**Fix**

- The fill step now runs after every completed repair through `post_repair()`.
- It is no longer registered as a standalone repair operator.

### 5. Same-route consolidation could not upgrade vehicle types

The old consolidation logic could only merge cargo into an existing target
vehicle. It did not replace two `10.3` vehicles with one `11.4` vehicle.

**Fix**

- Route consolidation now tries the smallest larger vehicle type that can
  hold the combined load before giving up.

### 6. There was no cross-period consolidation neighborhood

The major remaining gap was structural: ALNS could reduce loads on individual
vehicles but could not move an entire load from one period to an existing
vehicle on the same plant-dealer route in another period.

**Fix**

- Added `cross_period_consolidation_repair`.
- It moves cargo from one vehicle to an existing same-route vehicle in a
  different period.
- It can upgrade the target vehicle type when necessary.
- Every move is accepted only when the resulting state is feasible and the
  actual objective strictly decreases.
- The operator is registered in ALNS and also runs once before final output.

### 7. Cache and correctness defects

The review also fixed:

- `_unmet_cache` was not invalidated by `mark_objective_dirty()`;
- `get_unmet_demand()` returned mutable cached dictionaries;
- fleet-size guidance counted empty vehicles;
- `local_search_repair` did not invalidate the objective cache after changing a
  vehicle type;
- `sequential_repair` and `lookahead_repair` compared SKU quantities directly
  with remaining volume capacity;
- `inventory_level_removal` looked up `(dealer, sku, day)` in a
  `(plant, sku, day)` inventory table and was therefore effectively using the
  wrong namespace;
- two destroy operators divided by remaining capacity instead of rated
  capacity when ranking low-load vehicles;
- `periodic_shaw_removal` could omit inventory restoration when it emptied a
  vehicle;
- operator wrappers now refresh tunable parameters from the online tuner
  instead of permanently using the values bound at registration time;
- the cross-period consolidation operator is registered in both the standard
  ALNS repair registry and the ML repair-operator map;
- Shaw removal had an off-by-one deletion count;
- batch inventory-update failures now trigger a full inventory rebuild;
- repeated `setup_alns()` calls could accumulate operator counts;
- tracker progress was synchronized after the stopping limit had already been
  enlarged;
- a lightweight-intensification callback read the obsolete attribute
  `best_feasible_objective`;
- `_update_demands()` mutated the shared input `DataALNS.demands` during
  reporting;
- cross-period cargo is now rejected by validation unless its cargo day
  matches the vehicle day;
- duplicate production, inventory, and demand keys are aggregated instead of
  silently keeping only the last row;
- final output could continue after the selected incumbent failed validation.

## Validation

All comparisons use `dataset_1`.

### Objective evolution

| Configuration | Actual objective | Vehicle-period records | Below-minimum vehicles |
| --- | ---: | ---: | ---: |
| Original ALNS result | 153.00 | 75 | 39 |
| After finite search objective and first repair fixes | 118.05 | 69 | 24 |
| After cross-period consolidation | 47.52 | 41 | 2 |
| After final unit alignment, best observed 120-second run | 41.47 | 40 | 0 |
| Final 30-second confirmation run after the complete patch set | 43.26 | 42 | 0 |
| CPLEX Python API, 120-second feasible incumbent | 38.57 | 35 | 0 |

The best observed ALNS result has:

- zero demand shortage;
- one over-delivered dealer-SKU pair, maximum over-delivery `5`;
- zero vehicle-capacity violations;
- zero below-minimum-load vehicles;
- no inventory-balance mismatch;
- no negative inventory;
- recomputed actual objective `41.47`.

The remaining `41.47 - 38.57 = 2.90` difference is no longer an objective-unit
or feasibility mismatch. It is a remaining search-quality gap between a
metaheuristic and a time-limited exact MIP incumbent. The final confirmation
run after the complete patch set produced the same feasible structure with
`43.26` and no underloaded vehicle. The current implementation should
therefore be reported as a feasible metaheuristic result in the `41-46` range
for this 120-second benchmark rather than as a deterministic optimum.

### Regression commands

```powershell
python -m multi_period_delivery `
  --scale small `
  --dataset-id 1 `
  --data-dir datasets/multiple-periods/small `
  --output-dir OutPut-ALNS/review `
  --max-runtime 120 `
  --seed 1
```

```powershell
python -m multi_period_delivery.cplex_api_monolithic `
  --data-dir datasets/multiple-periods/small `
  --dataset-name dataset_1 `
  --output-dir OutPut-CPLEX-API/multiple-periods/small/dataset_1 `
  --time-limit 120 `
  --mip-gap 0 `
  --seed 1
```

See also:

- `docs/objective_analysis.md`;
- `docs/cplex_validation.md`.

## Round 2 Review: Inventory Kernels, ML Selection, and Post-Processing

This round re-audited the three subsystems that the ALNS relies on most
heavily: the inventory recursion (including its optional Numba kernel), the
machine-learning operator selection, and the destroy/repair post-processing.

### Inventory kernels (`inventory_numba.py`, `inventory_manager.py`, `inventory_utils.py`)

The Numba kernel reproduces the reference `SolutionState.compute_inventory()`
recursion exactly on `dataset_1`: the day-0 opening balance is the baseline,
production is applied on day `d`, and only cargo whose `cargo_day == veh.day`
counts as a same-day shipment. Verified on 276 inventory entries with an exact
match between the array path and the pure-Python path.

Fixes:

- **Integer write-back.** `arrays_to_state_s_ikt()` used to write `float`
  values into `s_ikt`, which is typed as an integer inventory model. Values are
  now written back as plain `int`, and a regression test asserts the type
  invariant so equality checks and batch-delta updates stay integer-safe.
- **Write-back scope.** The array path now updates only `(plant, sku)` pairs
  present in `state.s_indices` and only refreshes the day-0 key when it already
  exists, so it no longer invents zero-valued keys the reference never creates.
- **Cache validation.** The cached production array, baseline slice, and
  shipment array are shape-checked, and the vehicle signature now includes the
  vehicle type, day, cargo count, total quantity, and a cargo-content hash, so a
  content change invalidates the cached shipment array.
- **Forced NumPy fallback test.** A test disables Numba and asserts the NumPy
  fallback produces results identical to the Numba kernel.
- **`build_resource_pool()` semantics.** `UnifiedInventoryManager` now computes
  the allocatable pool as `max(0, ending_inventory(day-1) + production(day) -
  same-day usage)` for days `1..H`, matching the repair path instead of treating
  the raw ending-inventory snapshot as available stock.
- **Dealer shipment aggregation.** `precompute_dealer_shipments()` now ignores
  cross-day cargo, matching the inventory and objective shipment semantics.

### ML operator selection (`ml_operator_selector.py`, `ml_destroy_selection.py`, `ml_repair_selection.py`)

The ML candidate maps are now derived from a single registration-order
constant that mirrors the operators actually registered by
`ALNSOptimizer._register_destroy_operators()` / `_register_repair_operators()`.

Authoritative registered operators:

- destroy (10): `random_removal`, `shaw_removal`, `periodic_shaw_removal`,
  `worst_removal`, `surplus_inventory_removal`, `path_removal`,
  `cross_period_dealer_removal`, `period_range_removal`, `infeasible_removal`,
  `inventory_level_removal`.
- repair (9): `greedy_repair`, `cross_period_consolidation_repair`,
  `local_search_repair`, `inventory_balance_repair`, `smart_batch_repair`,
  `regret_based_repair`, `fleet_consolidation_repair`, `sequential_repair`,
  `lookahead_repair`.

Fixes:

- Removed the phantom `infeasible_repair` entry, which was listed in the ML
  maps but never registered as an ALNS repair operator.
- Kept `cross_period_consolidation_repair` in both the selector map and the
  `ml_based_repair` map; `fill_minimum_load_repair` stays out of the ML
  candidate set because it runs inside `post_repair()`, not as a standalone
  operator.
- Added `_finite_float()` and finite-guards in `extract_state_features()`,
  `record_operator_performance()`, and `train_models()`. `NaN`/`inf` rewards
  are dropped before training; a finite feature vector is enforced with
  `np.nan_to_num()`.
- Predictions that are `NaN`/`inf` or raise no longer enter the ranking; if
  every prediction fails, selection falls back to a random valid operator.
  Empty candidate sets, unknown names, untrained models, and no-parameter
  operators all have explicit fallbacks.

### Destroy/repair post-processing (`destroy_utils.py`, `repair_utils.py`)

Fixes:

- **Overlapping inventory deltas.** `post_destroy()` could double-count when a
  removed vehicle's cargo also appeared in the operator's `partial_updates`.
  When the two key sets overlap, the code now rebuilds the authoritative
  inventory from the final vehicle set instead of applying incremental deltas.
- **In-place vehicle upgrade.** `consolidate_route_vehicles()` upgrades a
  vehicle in place and keeps its id, so a later hard-violation rollback can
  still find and remove the vehicle by the id it inserted.
- **Always validate before rollback.** `post_repair()` now always runs the full
  `state.validate()` before deciding on a rollback, so a state with demand
  strictness disabled cannot skip the physical checks for negative inventory or
  plant-capacity overflow.
- **Bounded cross-period consolidation.** `cross_period_consolidation_repair`
  evaluates each candidate move with a full `SolutionState.copy()`, so an
  unbounded call is expensive on medium and large instances. The operator now
  takes `time_limit` and `max_moves` from `ALNSConfig`
  (`CROSS_PERIOD_CONSOLIDATION_TIME_LIMIT`,
  `CROSS_PERIOD_CONSOLIDATION_MAX_MOVES`), and the final pre-output pass uses
  its own larger-but-still-bounded budget
  (`FINAL_CROSS_PERIOD_CONSOLIDATION_TIME_LIMIT`,
  `FINAL_CROSS_PERIOD_CONSOLIDATION_MAX_MOVES`). Before this, Step 9
  ("Processing results") could take minutes on a medium instance whose ALNS
  search itself ran for a few seconds.

Confirmed already-correct behavior:

- destroy's `batch_update_inventory(..., op_type="destroy")` restores inventory
  over `day..horizon` with positive deltas, and repair's `veh_loading()`
  subtracts over the same range; the keys and signs are symmetric.
- The rollback predicate covers only physical hard violations
  (`veh_over_load`, `negative_inventory`, `plant_inv_exceed`,
  `cargo_day_mismatch`) and explicitly excludes `unmet_demand`; over-delivery is
  allowed by the model and is not a hard violation.
- Minimum-load filling runs before consolidation; merges re-check capacity and
  pick a suitable larger vehicle type; merged cargo does not share a mutable
  dictionary with the source vehicle.
- Non-finite objectives are not recorded in operator-performance statistics.
- `_inventory_cache`, `shipped_cache`, and the objective-dirty flag are
  invalidated or rebuilt after every batch change, load, and rollback.

### Round 2 validation

All comparisons use `dataset_1`, 45-second budget, seed 1.

| Configuration | Actual objective | Vehicle-period records | Below-minimum vehicles | Demand shortage |
| --- | ---: | ---: | ---: | ---: |
| ALNS base (no tuner, no ML) | 44.31 | 40 | 1 | 0 |
| ALNS with ML operators enabled | 43.73 | 41 | 0 | 0 |

The reported objective was independently recomputed from `opt_summary.csv`
(`fixed_cost + minimum_load_penalty`) and matches the run output in both cases.

### Test coverage added this round

- `tests/test_inventory_kernels.py`: kernel equivalence, NumPy fallback, cache
  invalidation, integer write-back.
- `tests/test_ml_operators.py`: ML maps vs registered operators, non-finite
  features/rewards, cold start, empty candidates, degenerate zero-weight model,
  NaN prediction, and training-sample filtering.
- `tests/test_destroy_repair_postprocess.py`: destroy-to-repair inventory
  round-trip, no rollback for physically feasible partial solutions, rollback
  for negative inventory, and post-repair minimum-load ordering.

Run them from the portfolio root with the shared environment:

```powershell
.\.venv\Scripts\python.exe -m pytest multi-period-delivery/tests -q
.\.venv\Scripts\python.exe -m pytest single-period-delivery/tests -q
```
