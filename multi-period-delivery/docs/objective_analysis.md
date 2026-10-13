# CPLEX / ALNS Objective Reconciliation

> **Status note (2026-10-13).** The `38.57` incumbent below was produced by
> the pre-refactor, global vehicle-indexed CPLEX model. The objective
> reconciliation and over-delivery analysis remain valid for the ALNS path;
> the current route-local formulation is documented in
> [`mathematical_models.md`](mathematical_models.md).

## Question

Why can CPLEX report a much smaller objective than ALNS on `dataset_1`?

The answer has two parts:

1. The mathematical model allows over-delivery because demand is constrained by
   `sum(x) >= demand`, not equality.
2. The actual ALNS objective is correctly aligned with the mathematical
   objective, but its repair neighbourhoods mostly try to satisfy exactly the
   remaining demand and rarely attempt to fill an existing underloaded vehicle
   with surplus cargo.

## Numerical Decomposition

The previous ALNS result had:

- 75 vehicle-period records;
- fixed vehicle cost `75.0`;
- 39 vehicles below the minimum load;
- minimum-load penalty `39 * 2 = 78`;
- total actual objective `153.0`.

The CPLEX command-line result had:

- 37 vehicle-period records;
- fixed vehicle cost `39.73`;
- 0 vehicles below the minimum load;
- total actual objective `39.73`.

The 120-second CPLEX Python API run independently found the slightly better
feasible value `38.57`.

The gap was therefore:

- fixed-cost difference: `35.27`;
- minimum-load-penalty difference: `78.0`;
- total difference: `113.27`.

## Is Over-Delivery Allowed to Reduce the Objective?

Under the current model, yes.

The objective contains:

- a fixed cost for each used vehicle-period;
- an `alpha * vehicle_cost` penalty for each used vehicle below its minimum
  load.

It contains no per-unit transportation charge and no over-delivery penalty.
Therefore, adding extra units to an already-used vehicle can remove a
minimum-load penalty without adding another vehicle. It can also make a
different packing feasible with fewer vehicle-period activations.

This does not mean that over-delivery is free in reality. It means that the
current mathematical objective does not price the extra units of cargo. If the
business rule requires exact fulfilment, the model must use `==` demand
constraints or add a unit shipping cost or an over-delivery penalty.

## What Was Wrong With ALNS

The ALNS implementation already distinguishes:

- the search objective used by acceptance and operator selection;
- the actual objective reported to the user and used to select the final
  feasible solution.

The final `153.0` was an actual objective, not a search-objective value.
The problem was the repair neighbourhood:

- repair operators normally loaded only the remaining unmet demand;
- several paths explicitly prevented excess shipment;
- the fallback repair filled unmet demand but did not fill minimum-load gaps;
- the search proxy penalises minimum-load gaps by volume with
  `param_pun_factor3`, while the mathematical objective penalises each
  underloaded vehicle by `alpha * vehicle_cost`.

## Implemented Verification Fix

A new repair operator, `fill_minimum_load_repair`, was added to
`src/multi_period_delivery/alns/repair_operators.py`.

It:

- visits existing non-empty vehicles that are below their minimum load;
- uses only SKUs available from the vehicle's plant to its dealer;
- respects remaining vehicle capacity and available inventory;
- adds only enough extra cargo to cross the minimum-load threshold where
  possible;
- never creates a new vehicle merely to perform this fill.

A 30-second ALNS smoke run on `dataset_1` improved the actual objective from
the previous `153.0` to `139.0`. Independent checks still found:

- no demand shortage;
- no vehicle-capacity violation;
- no negative inventory;
- no plant-capacity violation;
- the recomputed actual objective equals `139.0`.

This confirmed the neighbourhood diagnosis. A subsequent full repair and
search-objective alignment pass added cross-period consolidation and removed
the remaining unit mismatch. The best observed 120-second ALNS run reached
`41.47` versus the CPLEX feasible incumbent `38.57`; a later 30-second
confirmation run reached `43.26` with the same feasible structure.

The implementation should therefore be treated as a stochastic metaheuristic
whose 120-second result is now in the same range as the time-limited CPLEX
incumbent. The remaining difference is a search-quality gap, not a difference
in the final objective formula or constraint interpretation.
