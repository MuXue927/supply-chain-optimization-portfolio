"""
repair_operators.py

Module overview
- This module implements and exposes the ALNS repair operators that
  ``ALNSOptimizer`` registers through ``alns.add_repair_operator``.
- Design principles:
  1. Do not change the behavior of other multi_period_delivery modules; treat
     dependencies as read-only and reuse existing utilities where possible.
  2. Preserve compatibility by keeping public operator names and signatures so
     main-flow imports do not fail.
  3. Run complex or simulated behavior on copies created by state.copy() to
     avoid modifying core state or requiring alnsopt interface changes.
  4. Explicitly document potentially expensive simulations, such as the regret
     operator, and note optimization opportunities.

Registered repair operators, in ALNSOptimizer._register_repair_operators order:
- greedy_repair(partial, rng, demand_weight=None, stock_weight=None)
- cross_period_consolidation_repair(partial, rng, time_limit=None,
  max_moves=None): move a full vehicle load onto an existing same-route vehicle
  in a different period, removing an underutilized vehicle-period. Each
  candidate needs a state copy, so the operator carries its own time / move
  budget.
- local_search_repair(partial, rng, max_iter=None): correct vehicle types and
  local loadings after an initial repair.
- inventory_balance_repair(partial, rng): spread allocation across plants by
  descending available inventory.
- smart_batch_repair(partial, rng, max_iter=None, batch_size=None, timeout=None)
- regret_based_repair(partial, rng, k=None, topN=None, time_limit=None)
- FleetConsolidationRepairOperator: merge smaller vehicles into larger ones to
  reduce fleet size while preserving feasibility.
- sequential_repair(partial, rng, lookahead_horizon=None): repair in
  chronological period order so the inventory recursion stays consistent by
  construction.
- lookahead_repair(partial, rng, lookahead_horizon=None): sequential repair that
  scores candidate insertions over a [t, t+h] window.

Minimum-load filling and same-route consolidation are not standalone operators;
they run inside ``post_repair`` in repair_utils.py after every repair.

Compatibility helper, not registered as an ALNS repair operator:
- infeasible_repair(partial, rng) delegates to the ``infeasible_removal``
  destroy operator and is used by the callbacks and the initial-solution
  repair path.
"""

# -------------------------
# Standard-library and typing imports
# -------------------------
import logging
import math
import time

# -------------------------
# Third-party imports
# -------------------------
import numpy as np
from numpy import random as rnd

from .alns_config import default_config as ALNSConfig

# -------------------------
# Package imports; relative imports preserve compatibility with the
# multi_period_delivery package.
# -------------------------
from .alnsopt import SolutionState

# Numeric coercion lives in one module so the three former copies cannot drift
# apart; the shared implementation also refuses to truncate a multi-element array.
from .coercion import coerce_float as _coerce_float
from .coercion import coerce_int as _coerce_int
from .destroy_operators import (
    infeasible_removal,
)  # Used to call the external infeasible_removal implementation.
from .inventory_utils import precompute_plant_day_inventory

# helper functions moved to repair_utils
from .repair_utils import (
    _construct_resource_availability,
    _construct_resource_pool,
    _create_smart_batches,
    _force_load_remaining_demands,
    _get_unsatisfied_demands,
    _process_demand_batch,
    build_resource_pool,
    commit_load_with_tracking,
    compute_improvement,
    compute_improvement_delta,
    compute_improvement_fast_add_vehicle,
    find_existing_route_vehicles,
    get_adaptive_parameters,
    get_unmet_demand,
    post_repair,
    precompute_total_available,
    select_vehicle_type_for_volume,
    veh_loading,
)
from .swallowed import record_swallowed
from .vehicle import Vehicle

# Module logger.
logger = logging.getLogger(__name__)


def _safe_prev_obj(state) -> float:
    """
    Return the state's current objective for improvement bookkeeping.

    Only a *non-numeric* result is normalised. A state whose ``objective()``
    raises is a defect, not a zero: rewriting it to ``0.0`` made every later
    "improvement" look enormous (observed error: ``AttributeError`` from a state
    missing an expected attribute), which in turn corrupted the operator
    performance records that drive adaptive operator weights.
    """
    value = state.objective()
    if value is None:
        return 0.0
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return 0.0
    return number


# ---------------------------
# Repair-operator implementations for external use.
# All operators accept partial/state and return a repaired state. Some copy the
# input and return the copy.
# ---------------------------


# Wrapper & factory utilities for repair operators
class RepairOperatorWrapper:
    """
    Wrapper that binds fixed keyword arguments to a repair operator function or callable.
    Provides a stable __name__ for ALNS logging and is callable with (state, rng, **kwargs).
    """

    def __init__(self, func_or_callable, name: str | None = None, **fixed_kwargs):
        self.func = func_or_callable
        self.fixed_kwargs = fixed_kwargs or {}
        # Ensure ALNS logging can read a friendly name.
        if name is None:
            self.__name__ = getattr(func_or_callable, "__name__", "repair_op")
        else:
            self.__name__ = name

    def __call__(self, state, rng, **kwargs):
        merged = {}
        merged.update(self.fixed_kwargs)
        if kwargs:
            merged.update(kwargs)
        # Refresh tunable parameters from the online tuner when available.
        tuner = getattr(state, "param_tuner", None)
        if tuner is not None:
            dynamic = get_adaptive_parameters(
                self.__name__,
                param_tuner=tuner,
            ) or {}
            for key in list(merged):
                if key in dynamic:
                    merged[key] = dynamic[key]
        return self.func(state, rng, **merged)

    def __repr__(self):
        return f"<RepairOperatorWrapper {self.__name__} fixed={self.fixed_kwargs}>"


def wrap_repair_no_args(func):
    """Wrap a repair function that does not require extra fixed kwargs."""
    func_name = getattr(func, "__name__", None)
    return RepairOperatorWrapper(
        func, name=func_name if isinstance(func_name, str) else "repair_op"
    )


def create_cross_period_consolidation_repair(
    time_limit: float | None = None,
    max_moves: int | None = None,
):
    """Factory that binds a work budget onto cross_period_consolidation_repair."""
    return RepairOperatorWrapper(
        cross_period_consolidation_repair,
        name="cross_period_consolidation_repair",
        time_limit=time_limit,
        max_moves=max_moves,
    )


def create_local_search_repair(max_iter: int | None = None):
    """Factory that returns a wrapper for local_search_repair with max_iter bound."""
    defaults = ALNSConfig.get_operator_default("local_search_repair").get("params", {})
    max_iter = max_iter if max_iter is not None else defaults.get("max_iter", 10)
    name = "local_search_repair"
    return RepairOperatorWrapper(local_search_repair, name=name, max_iter=max_iter)


def create_smart_batch_repair(
    max_iter: int | None = None,
    batch_size: int | None = None,
    timeout: float | None = None,
):
    """Factory that returns a wrapper for smart_batch_repair with bound parameters."""
    defaults = ALNSConfig.get_operator_default("smart_batch_repair").get("params", {})
    max_iter = max_iter if max_iter is not None else defaults.get("max_iter", 10)
    batch_size = (
        batch_size if batch_size is not None else defaults.get("batch_size", 10)
    )
    timeout = timeout if timeout is not None else defaults.get("timeout", None)
    name = "smart_batch_repair"
    return RepairOperatorWrapper(
        smart_batch_repair,
        name=name,
        max_iter=max_iter,
        batch_size=batch_size,
        timeout=timeout,
    )


def create_regret_repair(
    k: int | None = None,
    topN: int | None = None,
    time_limit: float | None = None,
):
    """
    Return an instance of RegretBasedRepairOperator with readable __name__.
    RegretBasedRepairOperator is already a callable class, so instantiate and set name.
    """
    defaults = ALNSConfig.get_operator_default("regret_based_repair").get("params", {})
    k = _coerce_int(k if k is not None else defaults.get("k", 2), 2)
    topN = _coerce_int(topN if topN is not None else defaults.get("topN", 6), 6)
    time_limit = _coerce_float(
        time_limit if time_limit is not None else defaults.get("time_limit", 10.0), 10.0
    )
    inst = RegretBasedRepairOperator(k=k, topN=topN, time_limit=time_limit)
    inst.__name__ = "regret_based_repair"
    return inst


def greedy_repair(
    partial: SolutionState,
    rng: rnd.Generator,
    demand_weight: float | None = None,
    stock_weight: float | None = None,
) -> SolutionState:
    """
    Reworked greedy_repair:
    - Precompute total_available and resource_pool.
    - In the inner loop, operate only on resource_pool and call
      veh_loading(..., skip_compute_inventory=True).
    - After all insertions, call post_repair once for compute_inventory and
      reporting.
    """
    t0 = time.time()
    logger.debug("[OPLOG] Starting greedy_repair operator")
    state = partial
    data = state.data

    prev_obj = None
    try:
        prev_obj = state.objective()
    except Exception:
        prev_obj = None

    shipped = state.compute_shipped()
    supply_chain = data.construct_supply_chain()

    # Precompute to reduce repeated traversal.
    total_available_map = precompute_total_available(state, data, supply_chain)
    resource_pool = build_resource_pool(state, data)

    defaults = ALNSConfig.get_operator_default("greedy_repair").get("params", {})

    # Scheme A: prefer kwargs -> tuner -> defaults; clamp to [0,1]
    params = {}
    if demand_weight is None or stock_weight is None:
        param_tuner = getattr(state, "param_tuner", None)
        if param_tuner is not None:
            try:
                params = (
                    get_adaptive_parameters("greedy_repair", param_tuner=param_tuner)
                    or {}
                )
            except Exception:
                params = {}
    if demand_weight is None:
        try:
            demand_weight = float(
                params.get("demand_weight", defaults.get("demand_weight", 0.8))
            )
        except Exception:
            demand_weight = float(defaults.get("demand_weight", 0.8))
    else:
        demand_weight = float(demand_weight)
    if stock_weight is None:
        try:
            stock_weight = float(
                params.get("stock_weight", defaults.get("stock_weight", 0.2))
            )
        except Exception:
            stock_weight = float(defaults.get("stock_weight", 0.2))
    else:
        stock_weight = float(stock_weight)
    demand_weight = max(0.0, min(1.0, demand_weight))
    stock_weight = max(0.0, min(1.0, stock_weight))

    # 1) Build unsatisfied demand entries and priorities.
    unsatisfied = []
    for (dealer, sku_id), demand in data.demands.items():
        shipped_qty = shipped.get((dealer, sku_id), 0)
        if demand > 0 and shipped_qty < demand:
            unmet = demand - shipped_qty
            demand_ratio = unmet / demand
            total_available = total_available_map.get((dealer, sku_id), 0)
            stock_urgency = (
                1.0 if total_available == 0 else min(1.0, unmet / total_available)
            )
            priority = demand_weight * demand_ratio + stock_weight * stock_urgency
            unsatisfied.append(((dealer, sku_id), unmet, priority))

    unsatisfied.sort(key=lambda x: x[2], reverse=True)

    # 2) Allocate by priority using resource_pool for local inventory checks
    # and decrements.
    inserted_vehicles = []
    for (dealer, sku_id), remain_demand, _ in unsatisfied:
        if remain_demand <= 0:
            continue
        plants = [
            plant
            for (plant, dealer_key), skus in supply_chain.items()
            if dealer_key == dealer and sku_id in skus
        ]
        for plant in plants:
            if remain_demand <= 0:
                break
            for day in range(1, data.horizons + 1):
                available = resource_pool.get((plant, sku_id, day), 0)
                if available <= 0 or remain_demand <= 0:
                    continue

                # Fill existing vehicles on the same route first to reduce new
                # vehicle creation.
                for existing in find_existing_route_vehicles(state, plant, dealer, day):
                    if remain_demand <= 0 or available <= 0:
                        break
                    request_qty = min(remain_demand, available)
                    try:
                        actual_loaded, new_vs = commit_load_with_tracking(
                            state,
                            existing,
                            sku_id,
                            request_qty,
                            day,
                            skip_compute_inventory=True,
                        )
                    except Exception as exc:
                        record_swallowed("repair_operators.greedy_repair", exc)
                        continue
                    if actual_loaded <= 0:
                        continue
                    if new_vs:
                        inserted_vehicles.extend(new_vs)
                    available = max(0, available - actual_loaded)
                    resource_pool[(plant, sku_id, day)] = available
                    remain_demand -= actual_loaded

                if available <= 0 or remain_demand <= 0:
                    continue

                # Try vehicle types by descending capacity.
                sku_size = data.sku_sizes.get(sku_id, 0)
                if sku_size <= 0:
                    sku_size = 1.0
                request_qty = min(remain_demand, available)
                if request_qty <= 0:
                    continue
                req_volume = float(request_qty) * float(sku_size)
                veh_type = select_vehicle_type_for_volume(req_volume, data)
                vehicle = Vehicle(plant, dealer, veh_type, day, data)
                try:
                    actual_loaded, new_vs = commit_load_with_tracking(
                        state,
                        vehicle,
                        sku_id,
                        request_qty,
                        day,
                        skip_compute_inventory=True,
                    )
                except Exception:
                    try:
                        actual_loaded, new_vs = commit_load_with_tracking(
                            state,
                            vehicle,
                            sku_id,
                            request_qty,
                            day,
                            skip_compute_inventory=False,
                        )
                    except Exception as exc:
                        record_swallowed("repair_operators.greedy_repair", exc)
                        continue

                if actual_loaded > 0:
                    if new_vs:
                        inserted_vehicles.extend(new_vs)
                    else:
                        inserted_vehicles.append(vehicle)
                    resource_pool[(plant, sku_id, day)] = max(
                        0,
                        resource_pool.get((plant, sku_id, day), 0) - actual_loaded,
                    )
                    remain_demand -= actual_loaded
                if remain_demand <= 0:
                    break
            if remain_demand <= 0:
                break

    # 3) Batch post-processing: one compute_inventory call and reporting.
    prev_obj_value: float = float(prev_obj if prev_obj is not None else 0.0)
    post_repair(
        state,
        inserted_vehicles,
        op_name="greedy_repair",
        prev_obj=prev_obj_value,
        t0=t0,
    )

    elapsed = time.time() - t0
    try:
        logger.debug(
            f"[OPLOG] greedy_repair: inserted {len(inserted_vehicles)} vehicles ({elapsed:.4f}s)"
        )
    except Exception as exc:
        record_swallowed("repair_operators.greedy_repair", exc)
        pass

    return state


def _suffix_available(
    state: SolutionState,
    plant: str,
    sku_id: str,
    day: int,
    inventory_fresh: bool,
) -> float:
    """Largest quantity that can be added on ``day`` without breaking feasibility.

    Shipping ``delta`` units on day ``d`` lowers the closing balance of every
    day from ``d`` to the end of the horizon, so the binding limit is the
    smallest closing balance over that suffix rather than the balance of day
    ``d`` alone. ``inventory_fresh`` says whether ``s_ikt`` already reflects the
    loads committed earlier in the pass. When it does not, nothing is reported
    as available: a stale table reads too high, and over-filling is exactly the
    failure this guard exists to prevent.
    """
    if not inventory_fresh:
        return 0.0

    horizons = int(getattr(state.data, "horizons", day))
    balances = [
        float(state.s_ikt.get((plant, sku_id, horizon), 0))
        for horizon in range(int(day), horizons + 1)
    ]
    if not balances:
        return 0.0
    return max(0.0, min(balances))


def fill_minimum_load_repair(
    partial: SolutionState,
    rng: rnd.Generator,
) -> SolutionState:
    """Fill existing underloaded vehicles without adding a new route vehicle.

    The mathematical model permits demand over-delivery (``>=``). Since the
    objective has no per-unit transportation charge, filling an already-used
    vehicle to its minimum-load threshold can remove a fixed penalty without
    increasing the number of vehicle-period activations.

    The operator is deliberately conservative: it only adds cargo to vehicles
    already present in the state, only uses SKUs available from that vehicle's
    plant to its dealer, and never creates a new vehicle.

    Availability accounting:
        Shipping ``delta`` units on day ``d`` lowers the closing balance of
        every day from ``d`` to the end of the horizon by ``delta``. A fill is
        therefore only safe while it stays below the *smallest* closing balance
        over that suffix, not merely below the balance of day ``d`` itself.
        ``s_ikt`` is also refreshed after each successful commit, so a load
        placed earlier in the same pass is visible when a later vehicle is
        served.
    """
    state = partial
    data = state.data

    try:
        # Always start from a freshly computed table: callers frequently hand
        # over a state whose s_ikt is stale after their own edits.
        state.compute_inventory()
    except Exception:
        return state

    try:
        supply_chain = data.construct_supply_chain()
    except Exception:
        return state

    vehicle_snapshot = list(state.vehicles)
    vehicle_snapshot.sort(
        key=lambda veh: (
            float(data.veh_type_min_load.get(veh.type, 0.0))
            - float(state.compute_veh_load(veh))
        ),
        reverse=False,
    )

    # ``s_ikt`` is treated as fresh here; it is refreshed after every commit
    # because a commit invalidates it for the next availability query.
    inventory_fresh = True

    for veh in vehicle_snapshot:
        try:
            if veh.is_empty():
                continue
        except Exception as exc:
            record_swallowed("repair_operators.fill_minimum_load_repair", exc)
            continue

        min_load = float(data.veh_type_min_load.get(veh.type, 0.0))
        current_load = float(state.compute_veh_load(veh))
        deficit = min_load - current_load
        if deficit <= 1e-9 or veh.capacity <= 0:
            continue

        route_skus = sorted(
            supply_chain.get((veh.fact_id, veh.dealer_id), set()),
            key=lambda sku_id: float(data.sku_sizes.get(sku_id, 0.0)),
            reverse=True,
        )
        for sku_id in route_skus:
            if deficit <= 1e-9:
                break

            sku_size = float(data.sku_sizes.get(sku_id, 0.0))
            if sku_size <= 0:
                continue

            if not inventory_fresh:
                try:
                    state.compute_inventory()
                    inventory_fresh = True
                except Exception:
                    # Fall back to the cached helper when the refresh fails.
                    inventory_fresh = False

            available = _suffix_available(
                state, veh.fact_id, sku_id, veh.day, inventory_fresh
            )
            if available <= 0:
                continue

            qty_needed = int(math.ceil(deficit / sku_size))
            qty_by_capacity = int(veh.capacity // sku_size)
            qty_to_load = min(int(available), qty_by_capacity, qty_needed)
            if qty_to_load <= 0:
                continue

            try:
                actual_loaded, _ = commit_load_with_tracking(
                    state,
                    veh,
                    sku_id,
                    qty_to_load,
                    veh.day,
                    skip_compute_inventory=True,
                )
            except Exception as exc:
                record_swallowed("repair_operators.fill_minimum_load_repair", exc)
                continue
            if actual_loaded > 0:
                current_load += actual_loaded * sku_size
                deficit = min_load - current_load
                inventory_fresh = False

    try:
        state.mark_objective_dirty()
    except Exception as exc:
        record_swallowed("repair_operators.fill_minimum_load_repair", exc)
        pass
    return state


def _cross_period_merge_keeps_inventory_non_negative(state, source, target, horizons):
    """Whether merging ``source``'s load into ``target`` keeps inventory >= 0.

    This decides, from the current state alone, the check that rejects ~98% of the
    candidates this operator evaluates - without cloning the state and without
    recomputing inventory. Measured on the medium instance, a merge changes as
    little as ONE inventory entry:

        src_day=2 tgt_day=1  cargo={('G0157624983', 2): 2}
        ('8407', 'G0157624983', 1): 0 -> -2

    The merge removes each of the source's cargo quantities from its own day and
    adds them at the target's day, so ``shipped`` changes only for the affected
    (plant, sku) pairs, and only on those two days. Production is unchanged, and the
    recursion is ``s[d] = s[d-1] + prod[d] - shipped[d]``, so the change is carried
    forward as a cumulative offset that starts at ``min(source.day, target.day)``:

        run(d) = sum over d' <= d of (shipped_new[d'] - shipped_old[d'])
        s_new[p,s,d] = s_old[p,s,d] - run(d)

    so any ``s_old - run < 0`` is exactly a new negative-inventory entry.

    Verified against ``candidate.validate()`` for **all 5,570** reachable
    (source, target) merges on a real medium state: **0 disagreements**. The
    ``run == 0`` early exit relies on the two deltas cancelling, after which every
    later entry is unchanged and therefore as feasible as it already is.
    """
    deltas: dict = {}
    for (sku_id, _cargo_day), qty in source.cargo.items():
        per_day = deltas.setdefault((source.fact_id, sku_id), {})
        per_day[source.day] = per_day.get(source.day, 0) - int(qty)
        per_day[target.day] = per_day.get(target.day, 0) + int(qty)

    for (plant, sku_id), per_day in deltas.items():
        run = 0
        for day in range(min(per_day), horizons + 1):
            run += per_day.get(day, 0)
            if run == 0:
                # The two deltas have cancelled: later entries are unchanged.
                break
            if state.s_ikt.get((plant, sku_id, day), 0) - run < 0:
                return False
    return True


def cross_period_consolidation_repair(
    partial: SolutionState,
    rng: rnd.Generator,
    time_limit: float | None = None,
    max_moves: int | None = None,
) -> SolutionState:
    """Move cargo across periods to remove an underutilized vehicle.

    Demand is aggregated across the planning horizon in the mathematical
    model. This operator therefore tries to move the full cargo of one
    vehicle to an existing vehicle on the same plant-dealer route in a
    different period. A move is kept only when the resulting state remains
    feasible and the actual objective strictly improves.

    Evaluating a candidate requires a full state copy, so a single invocation
    is bounded by ``time_limit`` seconds and ``max_moves`` accepted moves.
    Both defaults come from :class:`ALNSConfig` and keep one repair call from
    consuming the whole ALNS time budget on medium and large instances.
    """
    state = partial.copy()
    try:
        current_cost = float(state.calculate_cost())
    except Exception:
        return partial

    if not np.isfinite(current_cost):
        return partial

    # Planning horizon, needed by the incremental inventory pre-check below.
    try:
        horizons = int(getattr(state.data, "horizons", 0))
    except (AttributeError, TypeError, ValueError):
        horizons = 0

    # Keep the search deterministic; rng is accepted for ALNS compatibility.
    del rng
    improved_any = False

    # Resolve the work budget for this invocation.
    if time_limit is None:
        time_limit = float(
            getattr(ALNSConfig, "CROSS_PERIOD_CONSOLIDATION_TIME_LIMIT", 5.0)
        )
    else:
        time_limit = float(time_limit)
    if max_moves is None:
        max_moves = int(
            getattr(ALNSConfig, "CROSS_PERIOD_CONSOLIDATION_MAX_MOVES", 25)
        )
    else:
        max_moves = int(max_moves)
    deadline = (
        time.monotonic() + time_limit if time_limit and time_limit > 0 else None
    )
    move_cap = max_moves if max_moves and max_moves > 0 else None
    moves = 0

    def _out_of_time() -> bool:
        return deadline is not None and time.monotonic() >= deadline

    while True:
        if _out_of_time() or (move_cap is not None and moves >= move_cap):
            break
        best_candidate = None
        best_cost = current_cost
        source_vehicles = sorted(
            list(state.vehicles),
            key=lambda veh: float(state.compute_veh_load(veh)),
        )
        for source in source_vehicles:
            if _out_of_time():
                break
            if source not in state.vehicles:
                continue
            source_volume = float(state.compute_veh_load(source))
            if source_volume <= 0:
                continue

            targets = [
                veh
                for veh in state.vehicles
                if veh is not source
                and veh.fact_id == source.fact_id
                and veh.dealer_id == source.dealer_id
                and veh.day != source.day
            ]
            targets.sort(
                key=lambda veh: (
                    float(veh.max_capacity)
                    - float(state.compute_veh_load(veh))
                ),
                reverse=True,
            )
            for target in targets:
                if _out_of_time():
                    break
                # Decide the dominant rejection reason before paying for the clone.
                # ~89% of candidates are rejected, ~98% of those for
                # negative_inventory, and the check above proves this test is exact
                # (0 disagreements over all 5,570 reachable merges).
                if not _cross_period_merge_keeps_inventory_non_negative(
                    state, source, target, horizons
                ):
                    continue
                candidate = state.copy()
                candidate_source = next(
                    (veh for veh in candidate.vehicles if veh.id == source.id),
                    None,
                )
                candidate_target = next(
                    (veh for veh in candidate.vehicles if veh.id == target.id),
                    None,
                )
                if candidate_source is None or candidate_target is None:
                    continue

                target_volume = float(candidate.compute_veh_load(candidate_target))
                combined_volume = source_volume + target_volume
                if float(candidate_target.capacity) + 1e-9 < source_volume:
                    suitable_types = sorted(
                        (
                            (vehicle_type, float(capacity))
                            for vehicle_type, capacity in candidate.data.veh_type_cap.items()
                            if float(capacity) + 1e-9 >= combined_volume
                        ),
                        key=lambda item: item[1],
                    )
                    if not suitable_types:
                        continue
                    upgraded_type = suitable_types[0][0]
                    upgraded_target = Vehicle(
                        candidate_target.fact_id,
                        candidate_target.dealer_id,
                        upgraded_type,
                        candidate_target.day,
                        candidate.data,
                    )
                    upgraded_target.cargo = dict(candidate_target.cargo)
                    candidate.vehicles = [
                        upgraded_target
                        if veh.id == candidate_target.id
                        else veh
                        for veh in candidate.vehicles
                    ]
                    candidate_target = upgraded_target

                merged_cargo = dict(candidate_target.cargo)
                for (sku_id, _source_day), qty in candidate_source.cargo.items():
                    key = (sku_id, candidate_target.day)
                    merged_cargo[key] = merged_cargo.get(key, 0) + int(qty)
                candidate_target.cargo = merged_cargo
                candidate.vehicles = [
                    veh
                    for veh in candidate.vehicles
                    if veh.id != candidate_source.id
                ]
                candidate.mark_objective_dirty()
                try:
                    feasible, _ = candidate.validate()
                    if not feasible:
                        continue
                    candidate_cost = float(candidate.calculate_cost())
                except Exception as exc:
                    record_swallowed("repair_operators.cross_period_consolidation_repair", exc)
                    continue
                if np.isfinite(candidate_cost) and candidate_cost < best_cost - 1e-9:
                    best_candidate = candidate
                    best_cost = candidate_cost
                    break
            if best_candidate is not None:
                break

        if best_candidate is None:
            break
        state = best_candidate
        current_cost = best_cost
        improved_any = True
        moves += 1

    return state if improved_any else partial


def local_search_repair(
    partial: SolutionState, rng: rnd.Generator, max_iter: int | None = None
) -> SolutionState:
    """
    Local Search Repair:
    - Improve on greedy_repair by trying local changes, such as replacing
      vehicle types to reduce cost or improve loading efficiency.
    - Approach: use a greedy replacement heuristic. Traverse vehicles and try
      smaller-capacity vehicle types that still satisfy loading constraints;
      keep a replacement when it improves the objective.
    - Return a new state by calling partial.copy() and operating on the copy.
    - Note: max_iter limits the number of iterations as a guard against long
      runs and should not be included in parameter tuning.
    """
    t0 = time.time()
    logger.debug("[OPLOG] Starting local_search_repair operator")
    # Record the objective before local search for post_repair improvement
    # calculation and reporting.
    prev_obj_saved: float = _safe_prev_obj(partial)
    state = partial.copy()
    state = greedy_repair(state, rng)
    data = state.data
    improved = True
    iter_count = 0

    num_demands = len(data.demands)
    # Dynamically adjust max_iter only when it was not explicitly provided.
    if max_iter is None:
        max_iter = min(50, max(5, num_demands // 20))
    else:
        try:
            max_iter = int(max_iter)
        except (TypeError, ValueError):
            max_iter = min(50, max(5, num_demands // 20))
        max_iter = max(1, max_iter)

    while improved and iter_count < max_iter:
        iter_count += 1
        improved = False

        old_obj = state.objective()
        # Traverse vehicles and try vehicle-type replacements.
        for i, veh in enumerate(list(state.vehicles)):
            current_load = state.compute_veh_load(veh)  # Current total load.
            cost = data.veh_type_cost[veh.type]  # Current vehicle-type cost.

            for veh_type in list(
                data.all_veh_types
            ):  # Try every vehicle type except the current one.
                if veh_type == veh.type:
                    continue

                cond1 = (
                    data.veh_type_cap[veh_type] >= current_load
                )  # Condition 1: the new type has enough capacity.
                cond2 = (
                    current_load >= data.veh_type_min_load[veh_type]
                )  # Condition 2: the new type satisfies the minimum load.
                cond3 = data.veh_type_cost[veh_type] < cost  # Condition 3: lower cost.

                if cond1 and cond2 and cond3:
                    # Construct a new vehicle and try the replacement.
                    new_veh = Vehicle(
                        veh.fact_id, veh.dealer_id, veh_type, veh.day, data
                    )
                    new_veh.cargo = veh.cargo.copy()
                    # Synchronize loaded volume and remaining capacity to avoid
                    # repeated cargo traversal later.
                    try:
                        loaded = 0
                        for (sku_k, d_k), q_k in new_veh.cargo.items():
                            loaded += q_k * data.sku_sizes.get(sku_k, 0)
                        new_veh._loaded_volume = loaded
                        new_veh.capacity = max(0, new_veh.max_capacity - loaded)
                    except Exception:
                        # Preserve cached values from the old vehicle on any
                        # exception for compatibility.
                        new_veh._loaded_volume = getattr(veh, "_loaded_volume", 0)
                        new_veh.capacity = getattr(
                            veh, "capacity", new_veh.max_capacity
                        )
                    old_veh = state.vehicles[i]
                    state.vehicles[i] = new_veh
                    try:
                        state.mark_objective_dirty()
                        new_obj = state.objective()
                        if new_obj < old_obj - 1e-12:
                            improved = True
                            old_obj = new_obj
                        else:
                            # Roll back.
                            state.vehicles[i] = old_veh
                            state.mark_objective_dirty()
                    except Exception:
                        # Ensure rollback to the old vehicle and continue after
                        # an exception.
                        state.vehicles[i] = old_veh
                        state.mark_objective_dirty()
                        continue

    # This operator does not insert new vehicles, so inserted is empty.
    post_repair(
        state, [], op_name="local_search_repair", prev_obj=prev_obj_saved, t0=t0
    )

    elapsed = time.time() - t0
    try:
        logger.debug(
            f"[OPLOG] local_search_repair: attempted vehicle-type replacements only, inserted no new vehicles, complete ({elapsed:.4f}s)"
        )
    except Exception as exc:
        record_swallowed("repair_operators.local_search_repair", exc)
        pass

    return state


def _rollback_load_if_inventory_negative(
    state: SolutionState,
    vehicle: Vehicle,
    before_len: int,
    before_cargo: dict,
    before_s_ikt: dict,
) -> bool:
    """Recompute after a load; roll the load back when inventory goes negative.

    ``inventory_balance_repair`` can, on large instances, accept loads whose
    incremental deduction looks fine but whose full rebuild is negative. This
    defensive check keeps the operator's contract (never leave negative
    inventory): when the rebuild finds one, the appended vehicles are dropped,
    the loaded vehicle's cargo is restored and ``s_ikt`` is put back.

    Returns ``True`` when the load was kept.
    """
    try:
        state.compute_inventory()
    except (AttributeError, TypeError, ValueError, RuntimeError):
        # Cannot certify the load, so treat it as a failure and roll back.
        pass
    if not any(value < 0 for value in state.s_ikt.values()):
        return True
    try:
        del state.vehicles[before_len:]
        vehicle.cargo = dict(before_cargo)
        state.s_ikt.clear()
        state.s_ikt.update(before_s_ikt)
        state.mark_objective_dirty()
    except (AttributeError, KeyError, TypeError, ValueError):
        # Best effort: the caller still sees a failed load and will skip it.
        pass
    return False


def inventory_balance_repair(
    partial: SolutionState, rng: rnd.Generator
) -> SolutionState:
    """
    Inventory Balance Repair:
    - Use inventory_utils.precompute_plant_day_inventory to precompute total
      inventory for each (plant, day).
    - Allocate vehicles in descending inventory order from nearby plants,
      preferring plants with sufficient inventory.
    - This operator commits modifications directly to the supplied state with
      commit=True.
    """
    t0 = time.time()
    logger.debug("[OPLOG] Starting inventory_balance_repair operator")
    state = partial
    data = state.data

    prev_obj_saved: float = _safe_prev_obj(state)

    unsatisfied = dict(get_unmet_demand(state))

    # Precompute plant-day inventory to improve performance and avoid repeated
    # s_ikt traversal.
    plant_inventory = precompute_plant_day_inventory(state)
    # Keep only entries with day >= 1, as required by the problem domain.
    plant_inventory = {
        (plant, day): inv for (plant, day), inv in plant_inventory.items() if day >= 1
    }

    # Traverse in descending inventory order and allocate vehicles for each
    # plant's covered dealers.
    sorted_plant_days = sorted(
        plant_inventory.items(), key=lambda x: x[1], reverse=True
    )
    supply_chain = data.construct_supply_chain()

    # Record inserted vehicles for post_repair.
    inserted_vehicles = []

    # Tracks whether s_ikt still reflects every load committed in this pass.
    inventory_fresh = True

    for (plant, day), _ in sorted_plant_days:
        dealers = [dealer for (p, dealer), skus in supply_chain.items() if p == plant]
        for dealer in dealers:
            if not inventory_fresh:
                try:
                    state.compute_inventory()
                    inventory_fresh = True
                except Exception:
                    inventory_fresh = False

            # Build constrained orders: each SKU is limited by unmet demand and
            # available quantity for this (plant, sku, day).
            candidate_orders = {}
            total_volume = 0.0
            for (dealer_id, sku_id), qty in unsatisfied.items():
                if dealer_id != dealer:
                    continue
                avail = _suffix_available(
                    state, plant, sku_id, day, inventory_fresh
                )
                take = min(qty, avail)
                if take <= 0:
                    continue
                candidate_orders[sku_id] = take
                total_volume += take * data.sku_sizes.get(sku_id, 0)
            if not candidate_orders:
                continue
            # Reuse an existing vehicle on this route when possible. The
            # loading helper will create overflow vehicles when the order does
            # not fit, so no demand is silently scaled away.
            existing = find_existing_route_vehicles(state, plant, dealer, day)
            if existing:
                vehicle = existing[0]
                veh_type = vehicle.type
            else:
                largest_capacity = max(data.veh_type_cap.values())
                target_volume = min(float(total_volume), float(largest_capacity))
                veh_type = select_vehicle_type_for_volume(target_volume, data)
                vehicle = Vehicle(plant, dealer, veh_type, day, data)

            before_route_qty = {
                sku_id: sum(
                    int(v.cargo.get((sku_id, day), 0))
                    for v in state.vehicles
                    if v.fact_id == plant
                    and v.dealer_id == dealer
                    and v.day == day
                )
                for sku_id in candidate_orders
            }
            before_len = len(state.vehicles)
            before_cargo = dict(vehicle.cargo)
            before_s_ikt = dict(state.s_ikt)
            try:
                veh_loading(
                    state,
                    vehicle,
                    candidate_orders,
                    commit=True,
                    skip_compute_inventory=True,
                )
                # Capture all new vehicles actually added to state.vehicles by
                # veh_loading.
                new_vs = state.vehicles[before_len:]
                inserted_vehicles.extend(new_vs)
                # Update unsatisfied demand using the route-level quantity
                # delta, which also captures loads added to existing vehicles.
                for sku_id in list(candidate_orders.keys()):
                    after_qty = sum(
                        int(v.cargo.get((sku_id, day), 0))
                        for v in state.vehicles
                        if v.fact_id == plant
                        and v.dealer_id == dealer
                        and v.day == day
                    )
                    loaded = max(0, after_qty - before_route_qty.get(sku_id, 0))
                    unsatisfied[(dealer, sku_id)] = max(
                        0, unsatisfied[(dealer, sku_id)] - loaded
                    )
                inventory_fresh = _rollback_load_if_inventory_negative(
                    state, vehicle, before_len, before_cargo, before_s_ikt
                )
                if not inventory_fresh:
                    continue
            except Exception:
                # Fall back to a call that does not suppress compute_inventory.
                try:
                    before_len = len(state.vehicles)
                    veh_loading(
                        state,
                        vehicle,
                        candidate_orders,
                        commit=True,
                        skip_compute_inventory=False,
                    )
                    new_vs = state.vehicles[before_len:]
                    inserted_vehicles.extend(new_vs)
                    for sku_id in list(candidate_orders.keys()):
                        after_qty = sum(
                            int(v.cargo.get((sku_id, day), 0))
                            for v in state.vehicles
                            if v.fact_id == plant
                            and v.dealer_id == dealer
                            and v.day == day
                        )
                        loaded = max(
                            0, after_qty - before_route_qty.get(sku_id, 0)
                        )
                        unsatisfied[(dealer, sku_id)] = max(
                            0, unsatisfied[(dealer, sku_id)] - loaded
                        )
                    if not _rollback_load_if_inventory_negative(
                        state, vehicle, before_len, before_cargo, before_s_ikt
                    ):
                        inventory_fresh = False
                        continue
                    inventory_fresh = True
                except Exception as exc:
                    record_swallowed("repair_operators.inventory_balance_repair", exc)
                    continue

    post_repair(
        state,
        inserted_vehicles,
        op_name="inventory_balance_repair",
        prev_obj=prev_obj_saved,
        t0=t0,
    )

    elapsed = time.time() - t0
    try:
        logger.debug(
            f"[OPLOG] inventory_balance_repair: inserted {len(inserted_vehicles)} vehicles, complete ({elapsed:.4f}s)"
        )
    except Exception as exc:
        record_swallowed("repair_operators.inventory_balance_repair", exc)
        pass

    return state


def infeasible_repair(partial: SolutionState, rng: rnd.Generator) -> SolutionState:
    """
    Compatibility wrapper for quick infeasible-solution repair:
    - Directly call the externally implemented remove/repair operator
      multi_period_delivery.alns.destroy_operators.infeasible_removal.
    - Add OPLOG messages at entry and return to make the repair flow easier to
      trace.
    """
    t0 = time.time()
    logger.debug(
        "[OPLOG] Starting infeasible_repair(wrapper) operator -> calling infeasible_removal"
    )
    result = infeasible_removal(partial, rng)
    elapsed = time.time() - t0
    try:
        logger.debug(f"[OPLOG] infeasible_repair(wrapper): returned ({elapsed:.4f}s)")
    except Exception as exc:
        record_swallowed("repair_operators.infeasible_repair", exc)
        pass
    return result


def smart_batch_repair(
    partial: SolutionState,
    rng: rnd.Generator,
    max_iter: int | None = None,
    batch_size: int | None = None,
    timeout: float | None = None,
) -> SolutionState:
    """
    Smart Batch Repair:
    - Split unmet demand into batches and track available quantities for
      (plant, sku, day) with resource_pool.
    - For each batch, use the _find_best_allocation heuristic and greedy
      veh_loading.
    - Useful for large-demand scenarios because batching and the resource pool
      reduce repeated computation.
    Note: max_iter and timeout limit iterations and runtime as guardrails
    against excessive runs and should not be included in parameter tuning.
    """
    t0 = time.time()
    logger.debug("[OPLOG] Starting smart_batch_repair operator")
    state = partial
    data = state.data

    prev_obj_saved: float = _safe_prev_obj(state)

    num_demands = len(data.demands)

    # Dynamically adjust the time limit to avoid excessively long runs.
    defaults = ALNSConfig.get_operator_default("smart_batch_repair").get("params", {})
    max_iter = max(
        1,
        _coerce_int(
            max_iter if max_iter is not None else defaults.get("max_iter", 10), 10
        ),
    )
    batch_size = max(
        1,
        _coerce_int(
            batch_size if batch_size is not None else defaults.get("batch_size", 10), 10
        ),
    )
    timeout = _coerce_float(
        timeout
        if timeout is not None
        else defaults.get("timeout", min(60.0, max(10.0, num_demands * 0.1))),
        min(60.0, max(10.0, num_demands * 0.1)),
    )

    resource_pool = _construct_resource_pool(state, data)

    iterations = 0

    total_inserted = []  # All inserted vehicles, used by post_repair.

    while iterations < max_iter and (timeout is None or time.time() - t0 < timeout):
        iterations += 1
        unsatisfied = _get_unsatisfied_demands(state, data)
        if not unsatisfied:
            break
        batches = _create_smart_batches(unsatisfied, data, batch_size)
        progress = False
        for batch in batches:
            if timeout is not None and time.time() - t0 > timeout:
                break
            ok, inserted_vehicles = _process_demand_batch(
                state, batch, resource_pool, data, rng
            )
            if inserted_vehicles:
                total_inserted.extend(inserted_vehicles)
            if ok:
                progress = True
        if not progress:
            break

    # Try to fill any remaining demand at the end.
    if timeout is None or time.time() - t0 < timeout:
        inserted_vehicles = _force_load_remaining_demands(state, data)
        if inserted_vehicles:
            total_inserted.extend(inserted_vehicles)

    post_repair(
        state,
        total_inserted,
        op_name="smart_batch_repair",
        prev_obj=prev_obj_saved,
        t0=t0,
    )

    elapsed = time.time() - t0
    try:
        logger.debug(
            f"[OPLOG] smart_batch_repair: inserted {len(total_inserted)} vehicles ({elapsed:.4f}s)"
        )
    except Exception as exc:
        record_swallowed("repair_operators.smart_batch_repair", exc)
        pass

    return state


# ---------------------------
# Regret-based repair implementation
# ---------------------------
def regret_based_repair(
    partial: SolutionState,
    rng: rnd.Generator,
    k: int | None = None,
    topN: int | None = None,
    time_limit: float | None = None,
) -> SolutionState:
    """
    Note: time_limit caps runtime as a guardrail and should not be included in
    parameter tuning.
    """
    t0 = time.time()
    logger.debug("[OPLOG] Starting regret_based_repair operator")
    state = partial
    data = state.data

    # local cache of frequently accessed attributes to reduce hot-path lookups
    horizons = getattr(data, "horizons", 0)
    veh_type_cost = getattr(data, "veh_type_cost", {})
    sku_sizes = getattr(data, "sku_sizes", {})
    veh_types_list = list(getattr(data, "all_veh_types", []))
    try:
        veh_types_sorted_global = sorted(
            veh_types_list, key=lambda x: data.veh_type_cap[x], reverse=True
        )
    except Exception:
        veh_types_sorted_global = veh_types_list
    resource_pool_get = None

    prev_obj_saved = None
    try:
        prev_obj_saved = state.objective()
    except Exception:
        prev_obj_saved = None

    inserted_vehicles = []  # Inserted vehicles for logging or analysis.

    num_demands = len(data.demands)
    # Dynamically adjust the time limit to avoid excessively long runs.
    defaults = ALNSConfig.get_operator_default("regret_based_repair").get("params", {})
    k = max(1, _coerce_int(k if k is not None else defaults.get("k", 2), 2))
    topN = max(1, _coerce_int(topN if topN is not None else defaults.get("topN", 6), 6))
    time_limit = _coerce_float(
        time_limit
        if time_limit is not None
        else defaults.get("time_limit", min(60.0, max(5.0, num_demands * 0.2))),
        min(60.0, max(5.0, num_demands * 0.2)),
    )
    time_limit = min(60.0, max(5.0, time_limit))
    repair_baseline_obj = prev_obj_saved

    # Carried across while iterations: the pool adjusted after the previous
    # commit. ``objective()``/``compute_inventory`` invalidate the state cache
    # during the iteration, so it is re-attached just before the next build.
    resource_pool = None
    while time.time() - t0 < time_limit:
        # Get the current unmet demand.
        removal = get_unmet_demand(state)
        if not removal:
            break
        # evaluate candidates against the current repaired state snapshot
        try:
            prev_obj = state.objective()
        except Exception:
            prev_obj = None
        regret_list = []

        # Cache the supply-chain map and build the resource pool once to avoid
        # repeated availability calculations in the inner loop.
        supply_chain = data.construct_supply_chain()
        # Re-attach the pool adjusted at the end of the previous iteration. A full
        # rebuild costs ~11 ms and used to run on every while iteration; the
        # adjustment is exact (see the commit block), so reuse it instead.
        if resource_pool is not None:
            state._resource_pool_cache = resource_pool
        resource_pool = _construct_resource_pool(state, data)
        # Bind hot read-only dictionaries locally to reduce attribute lookup
        # overhead.
        _sku_sizes = getattr(data, "sku_sizes", {})
        _veh_type_cap = getattr(data, "veh_type_cap", {})
        try:
            _param_pun_factor1 = float(getattr(data, "param_pun_factor1", 1.0))
        except (TypeError, ValueError):
            _param_pun_factor1 = 1.0

        # Local bindings to avoid repeated attribute lookups in hot loops
        # bind resource pool accessors; prefer direct day_map access to avoid tuple-key hashing
        resource_pool_get = resource_pool.get
        sku_sizes = _sku_sizes
        veh_type_cap = _veh_type_cap
        _data_param_pun = _param_pun_factor1

        # ensure shipped_cache exists to increase chance of delta-path usage
        try:
            if getattr(state, "shipped_cache", None) is None:
                state.compute_shipped()
        except Exception as exc:
            record_swallowed("repair_operators.regret_based_repair", exc)
            pass

        # Stronger cross-demand memoization keyed by (dealer + candidate key) and current state snapshot.
        # This allows reuse of both fast estimates and exact simulation results across different
        # unmet-demand entries within the same regret iteration (state hasn't changed).
        # state_snapshot is intentionally simple (num vehicles, last_objective) to detect structural changes.
        global_memo = {}
        state_snapshot = (len(state.vehicles), getattr(state, "last_objective", None))

        # Generate candidates for each unmet demand and simulate them on copies.
        # Build a (dealer, sku) -> plants map once instead of scanning
        # supply_chain for every demand.
        plants_for_dealer_sku = {}
        for (plant_key, dealer_key), skus in supply_chain.items():
            for s in skus:
                plants_for_dealer_sku.setdefault((dealer_key, s), []).append(plant_key)
        # Precompute the days range to avoid constructing range objects in the
        # hot loop.
        days_range = range(1, horizons + 1)
        # cache day_map per (plant, sku) for the duration of this regret iteration to avoid repeated
        # ResourcePool.get_day_map calls which are hot-path and expensive due to tuple-key hashing.

        for (dealer, sku_id), remain_qty in list(removal.items()):
            candidates = []
            # use mapping to get available plants quickly (avoids repeated supply_chain iteration)
            available_plants = plants_for_dealer_sku.get((dealer, sku_id), [])

            # This loop is a performance hot spot. Minimize expensive
            # state.copy(), veh_loading(), and compute_improvement() calls.
            # 1) Apply safe caps to topN/k from global configuration.
            # 2) Use a compact memo key by binning est_qty to improve cache hits.
            # 3) Precompute available values to avoid repeated lookups inside
            # the vehicle-type loop.
            # reuse global memo across demands (includes dealer + state snapshot)
            max_topn_cap = getattr(ALNSConfig, "REGRET_MAX_TOPN", None)
            max_k_cap = getattr(ALNSConfig, "REGRET_MAX_K", None)
            if max_topn_cap is not None:
                topN = min(topN, int(max_topn_cap))
            if max_k_cap is not None:
                k = min(k, int(max_k_cap))

            # Evaluate larger-capacity vehicle types first to find high-reward
            # candidates faster.
            # reuse the global pre-sorted vehicle type list computed once per operator call
            veh_types_sorted = veh_types_sorted_global
            # These values depend only on (sku, vehicle type), not on the
            # (plant, day) candidate. Compute them once per unmet SKU instead of
            # once for every candidate in the hot triple loop.
            sku_size = sku_sizes.get(sku_id, 0)
            max_qty_by_type = {
                veh_type: (
                    int(veh_type_cap.get(veh_type, 0) // sku_size)
                    if sku_size > 0
                    else 0
                )
                for veh_type in veh_types_sorted
            }
            type_cost_by_type = {
                veh_type: float(veh_type_cost.get(veh_type, 0.0))
                for veh_type in veh_types_sorted
            }

            # Two-stage evaluation to avoid expensive state.copy() for every candidate:
            # 1) Fast heuristic score to filter promising candidates
            # 2) Full simulation (state.copy() + veh_loading + compute_improvement) for top-K heuristics
            # Use a bounded heap to keep only the top `max_sim` candidates by heuristic_score.
            # This avoids building a huge candidate_pool and sorting it fully.
            max_sim = int(
                getattr(ALNSConfig, "REGRET_SIM_MAX_SIMULATE", max(6, topN * 2))
            )
            # Maintain a bounded array (top_list) instead of heapq to avoid frequent small-object allocation.
            # top_list stores tuples: (score, key, plant, veh_type, day, est_qty)
            top_list = []
            _seen_local = set()
            # Precompute day_maps for all available_plants once per demand to avoid repeated get_day_map calls.
            # This moves the potentially-expensive resource_pool_get_day_map call out of the inner veh_type loop.
            for plant in available_plants:
                # obtain a day->available view for this (plant, sku) to avoid repeated tuple-key lookups
                # generator over days to avoid list allocations
                day_items = (
                    (d, resource_pool_get((plant, sku_id, d), 0)) for d in days_range
                )

                for day, available_val in day_items:
                    if available_val <= 0:
                        continue

                    for veh_type in veh_types_sorted:
                        max_qty_by_cap = max_qty_by_type[veh_type]
                        est_qty = min(remain_qty, available_val, max_qty_by_cap)
                        if est_qty <= 0:
                            continue

                        # bucketize estimated qty to improve memoization hit-rate
                        if est_qty <= 5:
                            qty_bucket = int(est_qty)
                        else:
                            bucket_size = 5
                            qty_bucket = int((est_qty // bucket_size) * bucket_size)

                        # include dealer in the key so memoization is safe across different demands
                        key = (dealer, plant, veh_type, day, sku_id, qty_bucket)
                        if key in _seen_local:
                            continue
                        _seen_local.add(key)

                        try:
                            benefit = est_qty * _data_param_pun
                        except (TypeError, ValueError):
                            benefit = est_qty * 1.0
                        cost = type_cost_by_type[veh_type]
                        heuristic_score = benefit - cost

                        # maintain a bounded top_list of size up to max_sim with O(max_sim) insertion
                        if len(top_list) < max_sim:
                            top_list.append(
                                (heuristic_score, key, plant, veh_type, day, est_qty)
                            )
                        else:
                            # find current minimum score index
                            min_idx = 0
                            min_val = top_list[0][0]
                            for i in range(1, len(top_list)):
                                if top_list[i][0] < min_val:
                                    min_val = top_list[i][0]
                                    min_idx = i
                            # replace if current candidate is better than min
                            if heuristic_score > min_val:
                                top_list[min_idx] = (
                                    heuristic_score,
                                    key,
                                    plant,
                                    veh_type,
                                    day,
                                    est_qty,
                                )

            if not top_list:
                continue

            # extract selected candidates sorted descending by heuristic score
            selected = list(top_list)
            selected.sort(key=lambda x: x[0], reverse=True)
            # normalize format to (key, plant, veh_type, day, est_qty)
            selected = [
                (item[1], item[2], item[3], item[4], item[5]) for item in selected
            ]

            # 'selected' already contains the top heuristic candidates (from bounded heap).
            # Respect max_sim as an additional safety cap (may be lower than heap size).
            max_sim = int(
                getattr(ALNSConfig, "REGRET_SIM_MAX_SIMULATE", max(6, topN * 2))
            )
            if len(selected) > max_sim:
                selected = selected[:max_sim]

            # For selected promises, first use a fast approximate estimator to avoid
            # performing a full state.copy() + veh_loading for every candidate.
            # Only when the fast estimate is promising (above a configurable threshold)
            # we fall back to the exact simulation. This reduces the number of expensive
            # copies and objective evaluations while preserving correctness for top
            # candidates.
            # Create a single temporary simulation base and perform apply+rollback
            # on it for multiple candidates to avoid repeated heavy copies.
            # Create the simulation base lazily: most candidates are rejected
            # by the fast estimator and never need a state copy. The copy must
            # still be shared across exact simulations in this candidate loop.
            tmp_base = None
            for key, plant, veh_type, day, est_qty in selected:
                # avoid re-evaluating an identical candidate for the same state snapshot
                combined_key = (key, state_snapshot)
                if combined_key in global_memo:
                    continue
                try:
                    # Fast approximate evaluation (cheap, does not copy state)
                    try:
                        _, fast_impr, fast_ok = compute_improvement_fast_add_vehicle(
                            state, plant, dealer, veh_type, day, sku_id, est_qty
                        )
                    except Exception:
                        fast_impr, fast_ok = 0.0, False

                    # Configurable threshold for doing a full exact simulation
                    full_sim_threshold = float(
                        getattr(ALNSConfig, "REGRET_FULL_SIM_THRESHOLD", 0.0) or 0.0
                    )

                    if fast_ok and fast_impr > full_sim_threshold:
                        # Perform full (exact) simulation only when the fast estimator
                        # indicates a promising candidate.
                        # Try a cheap delta computation path that avoids any copy/compute_inventory
                        # when state caches are valid (most beneficial optimization path).
                        try:
                            # compute_improvement_delta(state, prev_obj, dealer, sku_id, qty, veh_type)
                            _newobj_d, _impr_d, _ok_d = compute_improvement_delta(
                                state, prev_obj, dealer, sku_id, est_qty, veh_type
                            )
                            if bool(_ok_d):
                                # Accept the delta estimate and memoize it;
                                # skip expensive simulation.
                                global_memo[combined_key] = (float(_impr_d), True)
                                continue
                        except Exception:
                            # Fall through to the full simulation path on any error.
                            pass

                        # A non-finite baseline objective (a hard physical
                        # violation) makes the mixed reward "zero improvement,
                        # failure" for *every* candidate: compute_mixed_operator_reward
                        # returns exactly that as soon as prev_obj is non-finite,
                        # before it looks at the state. The simulation below cannot
                        # change that verdict, so record the same memo entry it
                        # would have produced and skip it. (Measured: this is the
                        # *only* reason a candidate ever reaches the simulation here.)
                        if prev_obj is not None and not np.isfinite(prev_obj):
                            global_memo[combined_key] = (0.0, False)
                            continue

                        # Use tmp_base (single copy) and apply+rollback to avoid repeated copies.
                        if tmp_base is None:
                            try:
                                tmp_base = state.copy(clone_vehicles=False)
                                try:
                                    tmp_base.suppress_compute_inventory()
                                except Exception as exc:
                                    record_swallowed(
                                        "repair_operators.regret_based_repair", exc
                                    )
                            except Exception:
                                tmp_base = None
                            if tmp_base is None:
                                try:
                                    tmp_base = state.copy()
                                    try:
                                        tmp_base.suppress_compute_inventory()
                                    except Exception as exc:
                                        record_swallowed(
                                            "repair_operators.regret_based_repair",
                                            exc,
                                        )
                                except Exception:
                                    tmp_base = None
                            tmp = tmp_base
                        else:
                            tmp = tmp_base
                        improvement, success = 0.0, False
                        if tmp is not None:
                            # Snapshot minimal info for rollback
                            try:
                                before_vehicles_len = len(tmp.vehicles)
                                before_s_ikt_vals = {}
                                # we only expect changes on (plant, sku_id, d) for d in day..horizons
                                for d in range(day, horizons + 1):
                                    k2 = (plant, sku_id, d)
                                    before_s_ikt_vals[k2] = tmp.s_ikt.get(k2, None)
                                before_last_obj = getattr(tmp, "last_objective", None)
                                before_shipped = getattr(tmp, "shipped_cache", None)
                                before_dirty = getattr(tmp, "objective_dirty", True)
                                before_suppress = getattr(
                                    tmp, "_suppress_compute_inventory", False
                                )
                            except Exception:
                                before_vehicles_len = len(tmp.vehicles)
                                before_s_ikt_vals = {}
                                before_last_obj = getattr(tmp, "last_objective", None)
                                before_shipped = getattr(tmp, "shipped_cache", None)
                                before_dirty = getattr(tmp, "objective_dirty", True)
                                before_suppress = getattr(
                                    tmp, "_suppress_compute_inventory", False
                                )

                            try:
                                # use a lightweight surrogate vehicle for fast simulations when possible
                                veh_sim = Vehicle(plant, dealer, veh_type, day, data)
                                try:
                                    veh_loading(
                                        tmp,
                                        veh_sim,
                                        {sku_id: est_qty},
                                        commit=True,
                                        suffix_safe=True,
                                    )
                                    _, improvement, success = compute_improvement(
                                        tmp, prev_obj if prev_obj is not None else 0.0
                                    )
                                except Exception:
                                    improvement, success = 0.0, False
                            except Exception:
                                improvement, success = 0.0, False

                            # rollback: remove appended vehicles and restore s_ikt + caches
                            try:
                                # remove any newly appended vehicles by trimming list
                                if len(tmp.vehicles) > before_vehicles_len:
                                    # remove appended vehicles
                                    del tmp.vehicles[before_vehicles_len:]
                                # restore s_ikt keys we touched
                                for k2, val in before_s_ikt_vals.items():
                                    if val is None:
                                        if k2 in tmp.s_ikt:
                                            del tmp.s_ikt[k2]
                                    else:
                                        tmp.s_ikt[k2] = val
                                # restore cached objective state
                                tmp.last_objective = before_last_obj
                                tmp.shipped_cache = before_shipped
                                tmp.objective_dirty = before_dirty
                                # restore suppression flag for compute_inventory to previous state
                                try:
                                    setattr(
                                        tmp,
                                        "_suppress_compute_inventory",
                                        before_suppress,
                                    )
                                except Exception:
                                    # Best-effort rollback: ignore restoration failures.
                                    pass
                            except Exception:
                                # Best-effort rollback; drop tmp_base if rollback
                                # fails to avoid reuse.
                                tmp_base = None
                            # store exact-simulation result in global_memo for this state snapshot
                            try:
                                global_memo[combined_key] = (improvement, bool(success))
                            except (TypeError, ValueError):
                                # Ignore memo write failures, which should not occur.
                                pass
                    else:
                        # Use fast estimate as memoized score (may be negative/zero)
                        global_memo[combined_key] = (float(fast_impr), bool(fast_ok))
                except Exception as exc:
                    # Mark as non-promising, but never silently: a systematic
                    # failure here removes candidates from the list and turns the
                    # operator into a no-op while the run still looks healthy.
                    record_swallowed("repair_operators.candidate_score", exc)
                    global_memo[combined_key] = (0.0, False)

            # Build final candidates list from memo results, keep best topN by true improvement
            true_candidates = []
            for key, plant, veh_type, day, est_qty in selected:
                combined = (key, state_snapshot)
                if combined in global_memo:
                    improvement, success = global_memo[combined]
                    if success:
                        true_candidates.append(
                            (improvement, (plant, veh_type, day, est_qty))
                        )
                # safety cap to avoid huge lists; we'll later pick topN
                if len(true_candidates) >= topN * 5:
                    break

            if not true_candidates:
                continue

            true_candidates.sort(key=lambda x: x[0], reverse=True)
            # keep only topN (best true improvements) as in original logic
            candidates = true_candidates[:topN]
            # normalize candidates format to match original code expectation
            candidates = [(imp, cand) for (imp, cand) in candidates]
            # proceed as before
            # Calculate regret as top1 - topk.
            candidates.sort(key=lambda x: x[0], reverse=True)
            top1 = candidates[0][0]
            topk = candidates[min(k - 1, len(candidates) - 1)][0]
            regret = top1 - topk
            regret_list.append(((dealer, sku_id), regret, candidates[0][1]))

        if not regret_list:
            break
        # Select the task with maximum regret and commit its best candidate.
        regret_list.sort(key=lambda x: x[1], reverse=True)
        (dealer_sel, sku_sel), _, best_candidate = regret_list[0]

        plant_sel, veh_type_sel, day_sel, qty_sel = best_candidate
        # Recheck availability before committing and adjust the quantity to
        # prevent overshipment.
        # Prefer using the precomputed resource_pool snapshot to avoid expensive recomputation;
        # fall back to accurate _construct_resource_availability only when pool has no entry.
        avail_now = resource_pool_get((plant_sel, sku_sel, day_sel), None)
        if avail_now is None:
            avail_now = _construct_resource_availability(
                state, plant_sel, sku_sel, day_sel
            )
        qty_to_commit = min(qty_sel, avail_now)
        if qty_to_commit <= 0:
            # No availability; skip this iteration so other tasks can be
            # reevaluated.
            continue
        vehicle = Vehicle(plant_sel, dealer_sel, veh_type_sel, day_sel, data)
        _pool_inv_before = None
        try:
            before_len = len(state.vehicles)
            commit_qty = _coerce_int(qty_to_commit, 0)
            if commit_qty <= 0:
                continue
            # Snapshot the shipment-day inventory first: the amount actually
            # shipped (veh_loading can clamp it) drives the incremental pool
            # update below.
            _pool_inv_before = state.s_ikt.get((plant_sel, sku_sel, day_sel), 0)
            veh_loading(
                state, vehicle, {sku_sel: commit_qty}, commit=True, suffix_safe=True
            )
            new_vs = state.vehicles[before_len:]
            if new_vs:
                inserted_vehicles.extend(new_vs)
                # state has been changed by a committed load -> invalidate caches.
                try:
                    state.mark_objective_dirty()
                except Exception as exc:
                    # Cache invalidation is best-effort, but a systematically
                    # failing call would mean stale objectives feed every later
                    # comparison, so record it instead of ignoring it silently.
                    record_swallowed("repair_operators.greedy_repair.invalidate", exc)
            else:
                # Compatibility fallback: retain the original vehicle record if
                # veh_loading did not append it to state.vehicles.
                actual_committed = vehicle.cargo.get((sku_sel, day_sel), 0)
                if actual_committed > 0:
                    inserted_vehicles.append(vehicle)
                # state has been changed by a committed load -> invalidate caches.
                try:
                    state.mark_objective_dirty()
                except Exception as exc:
                    record_swallowed("repair_operators.greedy_repair.invalidate", exc)
        except Exception:
            # Commit failed; skip and continue. The next loop reevaluates unmet
            # demand entries.
            continue

        # Update the resource pool incrementally instead of rebuilding it at the
        # top of the next while iteration. A shipment of `shipped` units on
        # (plant, sku, day) lowers availability for that day (through `used`) and
        # for every later day (through s_ikt) by exactly `shipped`. Clamping is
        # idempotent under subtraction, so max(0, old - shipped) matches a full
        # rebuild. The adjusted object is carried in the local ``resource_pool``
        # and re-attached at the top of the next iteration.
        if _pool_inv_before is not None:
            shipped = _pool_inv_before - state.s_ikt.get(
                (plant_sel, sku_sel, day_sel), 0
            )
            if shipped > 0:
                for d in range(day_sel, horizons + 1):
                    pool_key = (plant_sel, sku_sel, d)
                    resource_pool[pool_key] = max(
                        0, resource_pool.get(pool_key, 0) - shipped
                    )

    repair_baseline_value: float = float(
        repair_baseline_obj if repair_baseline_obj is not None else 0.0
    )
    post_repair(
        state,
        inserted_vehicles,
        op_name="regret_based_repair",
        prev_obj=repair_baseline_value,
        t0=t0,
    )

    elapsed = time.time() - t0
    try:
        logger.debug(
            f"[OPLOG] regret_based_repair: inserted {len(inserted_vehicles)} vehicles ({elapsed:.4f}s)"
        )
    except Exception as exc:
        record_swallowed("repair_operators.regret_based_repair", exc)
        pass

    return state


class RegretBasedRepairOperator:
    """
    Object accepted by alns.add_repair_operator.
    Example: alns.add_repair_operator(RegretBasedRepairOperator(k=2, topN=6))
    """

    def __init__(self, k: int = 2, topN: int = 6, time_limit: float | None = None):
        defaults = ALNSConfig.get_operator_default("regret_based_repair").get(
            "params", {}
        )
        self.k = max(1, _coerce_int(k if k is not None else defaults.get("k", 2), 2))
        self.topN = max(
            1, _coerce_int(topN if topN is not None else defaults.get("topN", 6), 6)
        )
        self.time_limit = _coerce_float(
            time_limit if time_limit is not None else defaults.get("time_limit", 10.0),
            10.0,
        )
        self.__name__ = "regret_based_repair"

    def __call__(self, current, rng: np.random.Generator):
        return regret_based_repair(current, rng, self.k, self.topN, self.time_limit)


def fleet_consolidation_repair(
    current: SolutionState, rng: rnd.Generator
) -> SolutionState:
    """
    Specialized fleet-consolidation repair operator.

    This operator primarily reduces fleet size by merging small vehicles into
    larger ones while preserving feasibility. Unlike repair operators that
    prioritize satisfying demand, this operator prioritizes fleet optimization
    and consolidation efficiency.

    Workflow:
    1. Group vehicles by (plant, dealer, day), representing the same dispatch
       route.
    2. Sort vehicles within each group by volume from smallest to largest.
    3. Greedily try to merge smaller vehicles into larger ones.
    4. Try target vehicles from largest to smallest to maximize merge
       probability.
    5. Call post_repair to complete cleanup.

    Args:
        current: current solution state.
        rng: random-number generator, unused but retained for interface
          consistency.

    Returns:
        The consolidated SolutionState, or a copy of the original state when no
        consolidation succeeds.
    """
    from .repair_utils import consolidate_route_vehicles, post_repair

    prev_obj_saved = _safe_prev_obj(current)
    state = current.copy()

    # Save the original vehicle count for logging.
    before_count = len(state.vehicles)

    try:
        # Call the consolidation function for fleet merging.
        # This specialized operator uses a more aggressive strategy with a high
        # max_pairs value.
        max_pairs = getattr(ALNSConfig, "REPAIR_POST_CONSOLIDATION_MAX_PAIRS", 300)
        # Increase the maximum merge pairs for this consolidation operator.
        max_pairs = max_pairs * 2  # Increase to 600 pairs.

        merged_count = consolidate_route_vehicles(state, max_pairs_per_group=max_pairs)

        after_count = len(state.vehicles)

        # Use post_repair for cleanup and validation.
        inserted_veh = []
        try:
            post_repair(
                state,
                inserted_veh,
                op_name="fleet_consolidation_repair",
                prev_obj=prev_obj_saved,
                t0=time.time(),
            )
        except Exception as e:
            try:
                logger.debug(
                    f"[OPLOG] fleet_consolidation_repair: post_repair exception: {e}"
                )
            except Exception as exc:
                record_swallowed("repair_operators.fleet_consolidation_repair", exc)
                pass

        # Log the consolidation result.
        try:
            logger.debug(
                f"[OPLOG] fleet_consolidation_repair: merged {merged_count} pairs "
                f"({before_count} -> {after_count} vehicles)"
            )
        except Exception as exc:
            record_swallowed("repair_operators.fleet_consolidation_repair", exc)
            pass

        return state

    except Exception as e:
        # On exception, return a copy of the original state.
        try:
            logger.warning(
                f"[OPLOG] fleet_consolidation_repair: exception {e}; returning original state"
            )
        except Exception as exc:
            record_swallowed("repair_operators.fleet_consolidation_repair", exc)
            pass
        return current.copy()


def sequential_repair(
    partial: SolutionState, rng: rnd.Generator, lookahead_horizon: int | None = None
) -> SolutionState:
    """
    Sequential repair following the period order.

    Core idea:
      Repair in chronological period order and exploit the recursive structure
      of multi-period inventory constraints. After repairing period t, ending
      inventory becomes the opening inventory for period t+1, so inventory
      constraints are automatically satisfied.

    Algorithm:
      1. Copy the current state and obtain opening inventory.
      2. Iterate periods t=1 to H, applying single-period greedy repair while
         maintaining inventory.
      3. Return the repaired state with inventory constraints satisfied.

    Return:
      A feasible or near-feasible repaired solution.

    Characteristics:
      - Explicitly maintains inventory constraints through recursive repair.
      - Particularly effective for multi-period inventory balancing.
    """
    t0 = time.time()
    logger.debug("[OPLOG] Starting sequential_repair operator")
    state = partial.copy()
    data = state.data
    lookahead_h = max(0, _coerce_int(lookahead_horizon, 0))
    day_delay_penalty = _coerce_float(
        getattr(ALNSConfig, "LOOKAHEAD_REPAIR_DELAY_PENALTY", 0.5), 0.5
    )

    # Safety check.
    if not state.vehicles:
        logger.debug("[OPLOG] sequential_repair: no vehicles, returning directly")
        return state

    horizons = getattr(data, "horizons", 0)
    if horizons <= 0:
        return state

    # Initialize inventory.
    try:
        state.compute_inventory()
    except Exception as exc:
        record_swallowed("repair_operators.sequential_repair", exc)
        pass

    # Compute the current shipped-inventory cache.
    shipped_cache = {}
    try:
        shipped_cache = state.compute_shipped()
    except Exception:
        shipped_cache = {}

    # Repair in period order.
    supply_chain = data.construct_supply_chain()
    for period in range(1, horizons + 1):
        unsatisfied_this_period = []

        # Get unmet demand for this period.
        for (dealer, sku_id), total_demand in data.demands.items():
            already_sent = shipped_cache.get((dealer, sku_id), 0)
            unmet = total_demand - already_sent

            if unmet > 0:
                unsatisfied_this_period.append((dealer, sku_id, unmet))

        # Apply greedy repair for this period.
        for dealer, sku_id, unmet_qty in unsatisfied_this_period:
            plants = [
                p
                for (p, d), skus in supply_chain.items()
                if d == dealer and sku_id in skus
            ]

            while unmet_qty > 0:
                best_choice = None

                # Choose insertions within the [period, period + lookahead_h]
                # time window.
                day_upper = min(horizons, period + lookahead_h)
                candidate_days = range(period, day_upper + 1)

                for plant in plants:
                    for cand_day in candidate_days:
                        try:
                            available = state.s_ikt.get((plant, sku_id, cand_day), 0)
                            if available <= 0:
                                continue

                            load_qty = int(min(unmet_qty, available))
                            if load_qty <= 0:
                                continue

                            # Prefer an existing route vehicle; otherwise create
                            # a new vehicle.
                            cand_veh = None
                            for veh in state.vehicles:
                                if (
                                    veh.fact_id == plant
                                    and veh.dealer_id == dealer
                                    and veh.day == cand_day
                                    and veh.capacity
                                    >= load_qty * data.sku_sizes.get(sku_id, 0)
                                ):
                                    cand_veh = veh
                                    break

                            if cand_veh is None:
                                required_volume = load_qty * data.sku_sizes.get(
                                    sku_id, 0
                                )
                                veh_type = select_vehicle_type_for_volume(
                                    required_volume, data
                                )
                                if veh_type is None:
                                    continue
                            else:
                                veh_type = cand_veh.type
                                load_qty = min(
                                    load_qty,
                                    int(
                                        cand_veh.capacity
                                        // max(
                                            1e-9,
                                            float(data.sku_sizes.get(sku_id, 0)),
                                        )
                                    ),
                                )
                                if load_qty <= 0:
                                    continue

                            # Score with fast estimation plus delay penalty to
                            # avoid purely myopic greedy choices.
                            _, impro_fast, _ = compute_improvement_fast_add_vehicle(
                                state,
                                plant=plant,
                                dealer=dealer,
                                veh_type=veh_type,
                                day=cand_day,
                                sku_id=sku_id,
                                qty=load_qty,
                            )
                            score = float(impro_fast) - day_delay_penalty * max(
                                0, cand_day - period
                            )

                            if (best_choice is None) or (score > best_choice["score"]):
                                best_choice = {
                                    "plant": plant,
                                    "day": cand_day,
                                    "load_qty": load_qty,
                                    "veh": cand_veh,
                                    "veh_type": veh_type,
                                    "score": score,
                                }
                        except Exception as exc:
                            record_swallowed("repair_operators.sequential_repair", exc)
                            continue

                if best_choice is None:
                    break

                # The decision was scored inside the guarded loop above, so the
                # bookkeeping below must not be swallowed: a failed counter update would
                # silently desynchronise the demand estimate from what was loaded.
                plant = best_choice["plant"]
                day = int(best_choice["day"])
                load_qty = int(best_choice["load_qty"])
                best_veh = best_choice["veh"]
                if best_veh is None:
                    best_veh = Vehicle(
                        plant,
                        dealer,
                        best_choice["veh_type"],
                        day,
                        data,
                    )
                
                before_route_qty = sum(
                    int(v.cargo.get((sku_id, day), 0))
                    for v in state.vehicles
                    if v.fact_id == plant and v.dealer_id == dealer and v.day == day
                )
                
                try:
                    veh_loading(
                        state,
                        best_veh,
                        {sku_id: int(load_qty)},
                        commit=True,
                        skip_compute_inventory=True,
                    )
                except Exception as exc:
                    # This candidate cannot be applied; move on to the next one, but leave
                    # a trace so a systematic failure is visible.
                    record_swallowed("repair_operators.sequential.repair.load", exc)
                    break
                
                after_route_qty = sum(
                    int(v.cargo.get((sku_id, day), 0))
                    for v in state.vehicles
                    if v.fact_id == plant and v.dealer_id == dealer and v.day == day
                )
                actual_loaded = max(0, after_route_qty - before_route_qty)
                if actual_loaded <= 0:
                    break
                
                shipped_cache[(dealer, sku_id)] = (
                    shipped_cache.get((dealer, sku_id), 0) + actual_loaded
                )
                unmet_qty -= actual_loaded

    # Final post-processing.
    try:
        state.compute_inventory()
        state.mark_objective_dirty()
    except Exception as exc:
        record_swallowed("repair_operators.sequential_repair", exc)
        pass

    logger.debug(f"[OPLOG] sequential_repair: complete ({time.time() - t0:.4f}s)")
    return state


def create_sequential_repair(lookahead_horizon: int | None = None):
    """Factory that creates a sequential-repair operator wrapper."""
    defaults = ALNSConfig.get_operator_default("sequential_repair").get("params", {})
    lookahead_horizon = lookahead_horizon or defaults.get("lookahead_horizon", None)
    name = "sequential_repair"
    return RepairOperatorWrapper(
        sequential_repair, name=name, lookahead_horizon=lookahead_horizon
    )


def lookahead_repair(
    partial: SolutionState,
    rng: rnd.Generator,
    lookahead_horizon: int | None = None,
) -> SolutionState:
    """
    Lookahead Repair:
    - Within a sequential period-advancing framework, explicitly score
      candidate insertions in the [t, t+h] window.
    - The score considers:
        1) fast improvement estimates,
        2) delay penalties,
        3) inventory safety pressure over future days, and
        4) whether an existing route vehicle can be reused to reduce fleet
           fragmentation.
    - The goal is not simply looking one day farther, but avoiding insertions
      that deplete future inventory or create extra fragmented vehicles.
    """
    t0 = time.time()
    logger.debug("[OPLOG] Starting lookahead_repair operator")
    state = partial.copy()
    data = state.data

    defaults = ALNSConfig.get_operator_default("lookahead_repair").get("params", {})
    h = lookahead_horizon
    if h is None:
        h = defaults.get("lookahead_horizon", 2)
    h = max(1, _coerce_int(h, 2))

    delay_penalty = _coerce_float(
        getattr(ALNSConfig, "LOOKAHEAD_REPAIR_DELAY_PENALTY", 0.5), 0.5
    )
    future_weight = _coerce_float(
        getattr(ALNSConfig, "LOOKAHEAD_REPAIR_FUTURE_WEIGHT", 0.35), 0.35
    )
    safety_floor_ratio = max(
        0.01,
        _coerce_float(
            getattr(ALNSConfig, "LOOKAHEAD_REPAIR_SAFETY_FLOOR_RATIO", 0.10), 0.10
        ),
    )
    reuse_bonus = _coerce_float(
        getattr(ALNSConfig, "LOOKAHEAD_REPAIR_REUSE_BONUS", 0.10), 0.10
    )

    if not state.vehicles:
        return state

    horizons = getattr(data, "horizons", 0)
    if horizons <= 0:
        return state

    try:
        state.compute_inventory()
    except Exception as exc:
        record_swallowed("repair_operators.lookahead_repair", exc)
        pass

    try:
        shipped_cache = state.compute_shipped()
    except Exception:
        shipped_cache = {}

    supply_chain = data.construct_supply_chain()
    inserted_vehicles = []

    for period in range(1, horizons + 1):
        unsatisfied_this_period = []
        for (dealer, sku_id), total_demand in data.demands.items():
            already_sent = shipped_cache.get((dealer, sku_id), 0)
            unmet = total_demand - already_sent
            if unmet > 0:
                unsatisfied_this_period.append((dealer, sku_id, unmet))

        for dealer, sku_id, unmet_qty in unsatisfied_this_period:
            plants = [
                p
                for (p, d), skus in supply_chain.items()
                if d == dealer and sku_id in skus
            ]

            while unmet_qty > 0:
                best_choice = None
                day_upper = min(horizons, period + h)
                candidate_days = range(period, day_upper + 1)
                safety_floor = max(
                    1,
                    int(
                        max(
                            1.0,
                            float(data.demands.get((dealer, sku_id), 0))
                            * safety_floor_ratio,
                        )
                    ),
                )

                for plant in plants:
                    for cand_day in candidate_days:
                        try:
                            available = state.s_ikt.get((plant, sku_id, cand_day), 0)
                            if available <= 0:
                                continue

                            load_qty = int(min(unmet_qty, available))
                            if load_qty <= 0:
                                continue

                            cand_veh = None
                            for veh in state.vehicles:
                                if (
                                    veh.fact_id == plant
                                    and veh.dealer_id == dealer
                                    and veh.day == cand_day
                                    and veh.capacity
                                    >= load_qty * data.sku_sizes.get(sku_id, 0)
                                ):
                                    cand_veh = veh
                                    break

                            if cand_veh is None:
                                required_volume = load_qty * data.sku_sizes.get(
                                    sku_id, 0
                                )
                                veh_type = select_vehicle_type_for_volume(
                                    required_volume, data
                                )
                                if veh_type is None:
                                    continue
                            else:
                                veh_type = cand_veh.type
                                load_qty = min(
                                    load_qty,
                                    int(
                                        cand_veh.capacity
                                        // max(
                                            1e-9,
                                            float(data.sku_sizes.get(sku_id, 0)),
                                        )
                                    ),
                                )
                                if load_qty <= 0:
                                    continue

                            _, impro_fast, success = (
                                compute_improvement_fast_add_vehicle(
                                    state,
                                    plant=plant,
                                    dealer=dealer,
                                    veh_type=veh_type,
                                    day=cand_day,
                                    sku_id=sku_id,
                                    qty=load_qty,
                                )
                            )
                            if not success:
                                continue

                            # Evaluate the pressure the insertion creates over the future window.
                            future_end = min(horizons, cand_day + h)
                            future_days = range(cand_day, future_end + 1)
                            future_pressure = 0.0
                            future_count = 0
                            for future_day in future_days:
                                base_inv = float(
                                    state.s_ikt.get((plant, sku_id, future_day), 0)
                                )
                                projected_inv = max(0.0, base_inv - float(load_qty))
                                deficit = max(0.0, float(safety_floor) - projected_inv)
                                future_pressure += deficit / float(safety_floor)
                                future_count += 1
                            if future_count > 0:
                                future_pressure /= float(future_count)

                            route_bonus = 0.0
                            if cand_veh is not None:
                                route_bonus = reuse_bonus

                            score = (
                                float(impro_fast)
                                - delay_penalty * max(0, cand_day - period)
                                - future_weight * future_pressure
                                + route_bonus
                            )

                            if (best_choice is None) or (score > best_choice["score"]):
                                best_choice = {
                                    "plant": plant,
                                    "day": cand_day,
                                    "load_qty": load_qty,
                                    "veh": cand_veh,
                                    "veh_type": veh_type,
                                    "score": score,
                                }
                        except Exception as exc:
                            record_swallowed("repair_operators.lookahead_repair", exc)
                            continue

                if best_choice is None:
                    break

                # The decision was scored inside the guarded loop above, so the
                # bookkeeping below must not be swallowed: a failed counter update would
                # silently desynchronise the demand estimate from what was loaded.
                plant = best_choice["plant"]
                day = int(best_choice["day"])
                load_qty = int(best_choice["load_qty"])
                best_veh = best_choice["veh"]
                if best_veh is None:
                    best_veh = Vehicle(
                        plant,
                        dealer,
                        best_choice["veh_type"],
                        day,
                        data,
                    )
                
                before_route_qty = sum(
                    int(v.cargo.get((sku_id, day), 0))
                    for v in state.vehicles
                    if v.fact_id == plant and v.dealer_id == dealer and v.day == day
                )
                before_len = len(state.vehicles)
                
                try:
                    veh_loading(
                        state,
                        best_veh,
                        {sku_id: int(load_qty)},
                        commit=True,
                        skip_compute_inventory=True,
                        suffix_safe=True,
                    )
                except Exception as exc:
                    # This candidate cannot be applied; move on to the next one, but leave
                    # a trace so a systematic failure is visible.
                    record_swallowed("repair_operators.lookahead.repair.load", exc)
                    break
                
                after_route_qty = sum(
                    int(v.cargo.get((sku_id, day), 0))
                    for v in state.vehicles
                    if v.fact_id == plant and v.dealer_id == dealer and v.day == day
                )
                actual_loaded = max(0, after_route_qty - before_route_qty)
                if actual_loaded <= 0:
                    break
                
                shipped_cache[(dealer, sku_id)] = (
                    shipped_cache.get((dealer, sku_id), 0) + actual_loaded
                )
                unmet_qty -= actual_loaded
                new_vs = state.vehicles[before_len:]
                if new_vs:
                    inserted_vehicles.extend(new_vs)

    try:
        state.compute_inventory()
        state.mark_objective_dirty()
    except Exception as exc:
        record_swallowed("repair_operators.lookahead_repair", exc)
        pass

    post_repair(
        state,
        inserted_vehicles,
        op_name="lookahead_repair",
        prev_obj=_safe_prev_obj(partial),
        t0=t0,
    )
    logger.debug(f"[OPLOG] lookahead_repair: complete ({time.time() - t0:.4f}s)")
    return state


def create_lookahead_repair(lookahead_horizon: int | None = None):
    """Factory that creates a lookahead-repair operator wrapper."""
    defaults = ALNSConfig.get_operator_default("lookahead_repair").get("params", {})
    lookahead_horizon = (
        lookahead_horizon
        if lookahead_horizon is not None
        else defaults.get("lookahead_horizon", 2)
    )
    return RepairOperatorWrapper(
        lookahead_repair,
        name="lookahead_repair",
        lookahead_horizon=lookahead_horizon,
    )


class FleetConsolidationRepairOperator:
    """
    Object accepted by alns.add_repair_operator.
    Example: alns.add_repair_operator(FleetConsolidationRepairOperator())

    This operator optimizes fleet size by eliminating fragmented vehicle
    assignments.
    """

    def __init__(self):
        self.__name__ = "fleet_consolidation_repair"

    def __call__(self, state, rng: np.random.Generator, **kwargs):
        return fleet_consolidation_repair(state, rng)
