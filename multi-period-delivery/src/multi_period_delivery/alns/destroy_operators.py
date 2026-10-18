"""
Destroy-operator collection for Adaptive Large Neighborhood Search (ALNS).
These operators remove or release parts of a solution to create perturbation
and give repair operators room to improve it.

Design notes
1. Shared post-processing hook post_destroy:
   - Restores inventory for cargo carried by removed vehicles when needed.
   - Computes the objective improvement before and after the operator, with
     robust handling of inf and NaN values.
   - Reports performance to ParamAutoTuner for adaptive learning and weight
     updates.
   - Emits structured [OPJSON] logs for parsing, visualization, and ML feature
     extraction.
   - Updates the exploration phase marker through adjust_exploration to support
     exploration, exploitation, and refinement strategies.

2. Operator overview, in registration order. Every operator name ends in
   "_removal" and mirrors ALNSOptimizer._register_destroy_operators:
   - random_removal: stratified random removal of less efficient vehicles to
     introduce diversity.
   - shaw_removal: remove related vehicles by plant, dealer, day, and cargo
     similarity to create structural disruption.
   - periodic_shaw_removal: combine clustering, multidimensional feature
     engineering, and relatedness scoring for multi-period decisions.
   - worst_removal: remove inferior vehicles using a combined utilization and
     cost-efficiency score, favoring known heuristics.
   - surplus_inventory_removal: release shipments from plants under high
     inventory pressure to reduce source-side overflow risk.
   - path_removal: evaluate plant-to-dealer paths and remove inefficient
     subsets.
   - cross_period_dealer_removal: prioritize dealers that span many periods
     with many vehicles, then remove their vehicles in batches to trigger
     larger cross-period restructuring.
   - period_range_removal: choose a contiguous window of days and remove
     low-load vehicles inside it to break cross-day local optima.
   - infeasible_removal: targeted reduction or partial rollback for negative
     inventory and the infeasibility it causes.
   - inventory_level_removal: remove deliveries for dealers whose inventory is
     abnormally low or high; the dealer-side counterpart of
     surplus_inventory_removal, which looks at plant inventory.

3. Performance optimizations:
   - precompute_dealer_shipments precomputes dealer-level shipments and demand
     to avoid repeated O(V*S) aggregation.
   - Reduce repeated scans of state.vehicles and state.s_ikt inside operators.

4. Robustness strategy:
   - Do not use inf or NaN objectives for positive learning; set improvement to
     zero to avoid contaminating statistics.
   - Early exits caused by insufficient vehicles, no candidates, or clustering
     failures are recorded through post_destroy(..., early=True).
   - Protect probability and division operations with denominator checks and
     normalization checks.

5. Extension points:
   - post_destroy can include additional context fields, such as average load
     of removed vehicles or released capacity.
   - [OPJSON] output can be redirected to a file or unified logger for easier
     tracing.

Maintenance guidance:
- When adding a destroy operator, capture prev_obj = state.objective() inside
  the operator and call post_destroy after the removal step.
- If a removal can produce partial changes, ensure that the removed array
  represents only whole-vehicle removals for inventory restoration. Explicitly
  update state.s_ikt for non-whole-vehicle adjustments.
"""

# =========================
# Standard library
# =========================
import time
from collections import defaultdict
from typing import TYPE_CHECKING

# =========================
# Third-party libraries
# =========================
import numpy as np
import numpy.random as rnd

# =========================
# Internal project dependencies
# =========================
# Import SolutionState lazily to avoid a circular import with alnsopt; use
# object as the runtime placeholder.
if TYPE_CHECKING:
    from .alnsopt import SolutionState
else:
    SolutionState = object
# batch_update_inventory is imported locally when needed to avoid a top-level
# circular dependency.

from .alns_config import default_config as ALNSConfig

# Numeric coercion lives in one module so the three former copies cannot drift
# apart; the shared implementation also refuses to truncate a multi-element array.
from .coercion import coerce_float as _coerce_float
from .coercion import coerce_int as _coerce_int

# Import shared helpers from destroy_utils to keep this file lightweight.
from .destroy_utils import (
    ensure_pyint,
    ensure_pylist,
    get_adaptive_parameters,
    post_destroy,
    resolve_degree,
)
from .inventory_utils import precompute_dealer_shipments, precompute_plant_day_inventory

# Short-lived cache for periodic_shaw_removal KMeans results keyed by
# (n_clusters, rounded_mean_features). This avoids re-running KMeans when
# feature summaries are effectively unchanged between successive calls.
_periodic_shaw_cache = {}


# Wrapper utilities for registering destroy operators with ALNS
class DestroyOperatorWrapper:
    """
    Lightweight wrapper that holds fixed keyword parameters for a destroy operator
    and exposes a callable compatible with ALNS (state, rng, **kwargs).
    """

    def __init__(self, func, name: str | None = None, **fixed_kwargs):
        self.func = func
        self.fixed_kwargs = fixed_kwargs or {}
        self.__name__ = name or getattr(func, "__name__", "destroy_op")

    def __call__(self, state, rng, **kwargs):
        # Merge fixed kwargs (lower priority) with runtime kwargs (higher priority)
        merged = {}
        merged.update(self.fixed_kwargs)
        if kwargs:
            merged.update(kwargs)
        # When online tuning is enabled, let the tuner refresh parameter
        # values while preserving explicit call-site overrides.
        tuner = getattr(state, "param_tuner", None)
        if tuner is not None:
            dynamic = get_adaptive_parameters(
                self.__name__,
                param_tuner=tuner,
            ) or {}
            for key in list(merged):
                if key in dynamic:
                    merged[key] = dynamic[key]
        before_signature = (
            state_structure_signature(state)
            if self.__name__
            in {
                "surplus_inventory_removal",
                "path_removal",
                "infeasible_removal",
            }
            else None
        )
        result = self.func(state, rng, **merged)
        if before_signature is not None and result is not None:
            try:
                result._destroy_noop = (
                    before_signature == state_structure_signature(result)
                )
            except (AttributeError, TypeError):
                pass
        return result

    def __repr__(self):
        return f"<DestroyOperatorWrapper {self.__name__} fixed={self.fixed_kwargs}>"


def state_structure_signature(state) -> tuple:
    """Cheap exact signature used to identify a destroy no-op."""
    return (
        tuple(
            (
                getattr(vehicle, "id", None),
                getattr(vehicle, "day", None),
                tuple(sorted(getattr(vehicle, "cargo", {}).items())),
            )
            for vehicle in getattr(state, "vehicles", ())
        ),
        tuple(sorted(getattr(state, "s_ikt", {}).items())),
    )


# Factory helpers for common destroy operators
def create_random_removal(degree: float):
    name = "random_removal"
    return DestroyOperatorWrapper(func=random_removal, name=name, degree=degree)


def create_shaw_removal(degree: float):
    name = "shaw_removal"
    return DestroyOperatorWrapper(func=shaw_removal, name=name, degree=degree)


def create_periodic_shaw_removal(**params):
    params = dict(params)
    name = params.pop("name", "periodic_shaw_removal")
    # Support legacy configuration or external callers that pass "k_clusters"
    # and map it to the operator's expected "k_cluster" parameter.
    if "k_cluster" not in params and "k_clusters" in params:
        params["k_cluster"] = params.pop("k_clusters")
    return DestroyOperatorWrapper(func=periodic_shaw_removal, name=name, **params)


def wrap_operator_no_args(func):
    """Return a wrapper for operators that take no fixed parameters."""
    return DestroyOperatorWrapper(func=func, name=getattr(func, "__name__", None))


def create_worst_removal(degree: float):
    """Factory for worst_removal with a fixed degree parameter."""
    name = "worst_removal"
    return DestroyOperatorWrapper(func=worst_removal, name=name, degree=degree)


def create_surplus_inventory_removal(degree: float):
    """Factory for surplus_inventory_removal with a fixed degree parameter."""
    name = "surplus_inventory_removal"
    return DestroyOperatorWrapper(
        func=surplus_inventory_removal, name=name, degree=degree
    )


def create_path_removal(degree: float):
    """Factory for path_removal with a fixed degree parameter."""
    name = "path_removal"
    return DestroyOperatorWrapper(func=path_removal, name=name, degree=degree)


def create_cross_period_dealer_removal(degree: float):
    """Factory for cross_period_dealer_removal with a fixed degree parameter."""
    name = "cross_period_dealer_removal"
    return DestroyOperatorWrapper(
        func=cross_period_dealer_removal, name=name, degree=degree
    )


def create_period_range_removal(degree: float, window_ratio: float | None = None):
    """Factory for period_range_removal with fixed parameters."""
    name = "period_range_removal"
    fixed_kwargs = {"degree": degree}
    if window_ratio is not None:
        fixed_kwargs["window_ratio"] = window_ratio
    return DestroyOperatorWrapper(func=period_range_removal, name=name, **fixed_kwargs)


def random_removal(
    current: SolutionState, rng: rnd.Generator, degree: float | None = None
):
    """
    Concise, importable random_removal implementation:
      - Randomly remove vehicles while retaining at least one vehicle.
      - Compute prev_obj, call post_destroy, and mark the objective dirty.
      - Avoid undefined helpers and convert values to native Python types for
        compatibility.
    """
    t0 = time.time()
    print("[OPLOG] Starting random_removal operator")
    state = current.copy()
    prev_obj = state.objective()

    if len(state.vehicles) <= 1:
        post_destroy(state, [], "random_removal", early=True, prev_obj=prev_obj, t0=t0)
        print(f"[OPLOG] random_removal: insufficient vehicles ({time.time() - t0:.4f}s)")
        return state

    degree = float(resolve_degree("random_removal", state, degree))

    # Determine the removal count, removing at least one vehicle while
    # retaining at least one.
    num_remove = int(len(state.vehicles) * degree)
    num_remove = max(1, num_remove)
    num_remove = min(num_remove, max(0, len(state.vehicles) - 1))

    if num_remove <= 0:
        post_destroy(state, [], "random_removal", early=True, prev_obj=prev_obj, t0=t0)
        print(f"[OPLOG] random_removal: num_remove=0 ({time.time() - t0:.4f}s)")
        return state

    try:
        sel = rng.choice(range(len(state.vehicles)), size=num_remove, replace=False)
        sel = np.atleast_1d(sel)
        remove_indices = [int(x) for x in sel]
    except Exception:
        # Fallback: take the first num_remove indices in order.
        remove_indices = list(range(min(num_remove, len(state.vehicles))))

    remove_indices = sorted(remove_indices, reverse=True)
    removed = [state.vehicles[i] for i in remove_indices]
    state.vehicles = [
        veh for i, veh in enumerate(state.vehicles) if i not in remove_indices
    ]

    state.mark_objective_dirty()

    post_destroy(state, removed, "random_removal", prev_obj=prev_obj, t0=t0)
    print(
        f"[OPLOG] random_removal: removed {len(removed)}/{len(current.vehicles)} vehicles ({time.time() - t0:.4f}s)"
    )
    return state


def worst_removal(
    current: SolutionState,
    rng: rnd.Generator,
    degree: float | None = None,
    value_bias: float | None = None,
):
    """
    Concise, importable worst_removal implementation:
      - Score each vehicle using space utilization and cost/value efficiency.
      - Select low-scoring vehicles for removal.
      - Call post_destroy and invalidate the objective cache.
    """
    t0 = time.time()
    print("[OPLOG] Starting worst_removal operator")
    state = current.copy()
    prev_obj = state.objective()
    if len(state.vehicles) <= 1:
        post_destroy(state, [], "worst_removal", early=True, prev_obj=prev_obj, t0=t0)
        print(f"[OPLOG] worst_removal: insufficient vehicles ({time.time() - t0:.4f}s)")
        return state

    shipments_by_dealer_sku_day, _, _ = precompute_dealer_shipments(state)
    # Use cross-day accumulated shipments for each (dealer, sku) to calculate
    # satisfaction and avoid distortion from a single day's values.
    shipments_by_dealer_sku_total = {}
    for (dlr, sku, _day), q in shipments_by_dealer_sku_day.items():
        shipments_by_dealer_sku_total[(dlr, sku)] = (
            shipments_by_dealer_sku_total.get((dlr, sku), 0) + q
        )

    # The cargo-value bias is configurable; the default 2.0 controls how
    # strongly cargo value decays at high satisfaction levels.
    defaults = ALNSConfig.get_operator_default("worst_removal").get("params", {})
    if value_bias is None:
        params = get_adaptive_parameters(
            op_name="worst_removal", param_tuner=state.param_tuner
        )
        try:
            value_bias = float(
                params.get("value_bias", defaults.get("value_bias", 2.0))
            )
        except Exception:
            value_bias = defaults.get("value_bias", 2.0)
    value_bias = _coerce_float(value_bias, float(defaults.get("value_bias", 2.0)))

    vehicle_metrics = []
    for i, veh in enumerate(state.vehicles):
        capacity = state.data.veh_type_cap[veh.type]
        load = state.compute_veh_load(veh)
        space_util = load / capacity if capacity > 0 else 0
        veh_cost = state.data.veh_type_cost[veh.type]

        cargo_value = 0
        for (sku_id, day), qty in veh.cargo.items():
            dealer = veh.dealer_id
            demand = state.data.demands.get((dealer, sku_id), 0)
            if demand > 0:
                shipped_total = shipments_by_dealer_sku_total.get((dealer, sku_id), 0)
                satisfaction = min(1.0, shipped_total / demand)
                # Use the configurable value bias with a hinge to keep the
                # result non-negative.
                sku_value = (
                    qty
                    * state.data.sku_sizes[sku_id]
                    * max(0.0, value_bias - satisfaction)
                )
                cargo_value += sku_value

        cost_efficiency = cargo_value / veh_cost if veh_cost > 0 else 0
        score = 0.4 * space_util + 0.6 * cost_efficiency
        vehicle_metrics.append((i, score))

    vehicle_metrics.sort(key=lambda x: x[1])

    degree = float(resolve_degree("worst_removal", state, degree))
    num_remove = int(len(state.vehicles) * degree)
    if num_remove <= 0:
        post_destroy(state, [], "worst_removal", early=True, prev_obj=prev_obj, t0=t0)
        print(
            f"[OPLOG] worst_removal: degree={degree:.3f}, no vehicles removed ({time.time() - t0:.4f}s)"
        )
        return state

    selection_pool = vehicle_metrics[: min(len(vehicle_metrics), int(num_remove * 2))]
    if not selection_pool:
        post_destroy(state, [], "worst_removal", early=True, prev_obj=prev_obj, t0=t0)
        print(f"[OPLOG] worst_removal: no selectable vehicles ({time.time() - t0:.4f}s)")
        return state

    scores = np.array([1.0 / (1.0 + item[1]) for item in selection_pool])
    scores_sum = scores.sum()
    if scores_sum <= 0:
        probabilities = None
    else:
        probabilities = scores / scores_sum

    try:
        selected_indices = rng.choice(
            [idx for idx, _ in selection_pool],
            size=min(num_remove, len(selection_pool)),
            replace=False,
            p=probabilities,
        )
    except Exception:
        selected_indices = rng.choice(
            [idx for idx, _ in selection_pool],
            size=min(num_remove, len(selection_pool)),
            replace=False,
        )

    # Convert numpy scalar or array indices to a list of Python ints to avoid
    # indexing and serialization issues from np.int64 or ndarray values.
    try:
        selected_indices = np.atleast_1d(selected_indices)
        selected_indices = [int(x) for x in selected_indices]
    except Exception:
        if isinstance(selected_indices, (list, tuple)):
            selected_indices = [int(x) for x in selected_indices]
        else:
            selected_indices = [int(selected_indices)]

    removed = [state.vehicles[i] for i in sorted(selected_indices, reverse=True)]
    state.vehicles = [
        veh for i, veh in enumerate(state.vehicles) if i not in selected_indices
    ]

    state.mark_objective_dirty()

    post_destroy(state, removed, "worst_removal", prev_obj=prev_obj, t0=t0)
    print(
        f"[OPLOG] worst_removal: removed {len(removed)}/{len(current.vehicles)} vehicles ({time.time() - t0:.4f}s)"
    )
    return state


# ---------------------------------------------------------------------
# 3. Infeasibility-repair-oriented removal for negative inventory relief
# ---------------------------------------------------------------------
def infeasible_removal(current: SolutionState, rng: rnd.Generator):
    """
    Target negative inventory entries in state.s_ikt for (plant, sku, day):
      Procedure:
        1. Collect all entries with inv < 0 and sort by severity.
        2. Partially or fully roll back related vehicles with the same
           plant/day/SKU.
        3. For partial adjustments, reduce veh.cargo, free capacity, and add
           inventory back from day through horizons.
    Applicability:
      - Can also serve as a destroy operator during repair to quickly restore
        feasibility.
    """
    t0 = time.time()
    print("[OPLOG] Starting infeasible_removal operator")
    state = current.copy()
    prev_obj = state.objective()

    if len(state.vehicles) <= 1:
        post_destroy(
            state, [], "infeasible_removal", early=True, prev_obj=prev_obj, t0=t0
        )
        print(f"[OPLOG] infeasible_removal: insufficient vehicles ({time.time() - t0:.4f}s)")
        return state

    neg_inv_items = [
        (plant, sku_id, day, inv)
        for (plant, sku_id, day), inv in state.s_ikt.items()
        if inv < 0
    ]

    if not neg_inv_items:
        print(
            f"[OPLOG] infeasible_removal: no negative inventory, repair not needed ({time.time() - t0:.4f}s)"
        )
        post_destroy(
            state, [], "infeasible_removal", early=True, prev_obj=prev_obj, t0=t0
        )
        return state

    neg_inv_items.sort(key=lambda x: x[3])  # More negative values take priority.

    remove_indices = []  # Whole-vehicle removals.
    partial_adjustments = []  # (vehicle_index, sku_id, day, reduce_qty)

    for plant, sku_id, day, neg_amount in neg_inv_items:
        related_vehicles = []
        for idx, veh in enumerate(state.vehicles):
            if veh.fact_id == plant and veh.day == day and (sku_id, day) in veh.cargo:
                shipped_qty = veh.cargo[(sku_id, day)]
                related_vehicles.append((idx, veh, shipped_qty))
        if not related_vehicles:
            continue

        related_vehicles.sort(key=lambda x: x[2], reverse=True)

        amount_to_reduce = abs(neg_amount)
        reduced_so_far = 0
        for idx, veh, shipped_qty in related_vehicles:
            if reduced_so_far + shipped_qty <= amount_to_reduce:
                reduced_so_far += shipped_qty
                # Convert numpy aggregate results and shipped_qty to Python int
                # before comparison to avoid subtle np.int64/np.generic issues.
                try:
                    total_cargo_qty = int(np.sum(list(veh.cargo.values())))
                except Exception:
                    total_cargo_qty = sum(veh.cargo.values()) if veh.cargo else 0
                if total_cargo_qty == int(
                    shipped_qty
                ):  # Remove the whole vehicle when its cargo equals the rollback quantity.
                    if idx not in remove_indices:
                        remove_indices.append(idx)
                else:
                    partial_adjustments.append(
                        (idx, sku_id, day, shipped_qty)
                    )  # Record a partial adjustment.
            else:
                adjust_qty = amount_to_reduce - reduced_so_far
                partial_adjustments.append((idx, sku_id, day, adjust_qty))
                reduced_so_far = amount_to_reduce

    # Apply partial adjustments to vehicle cargo and record inventory add-backs.
    remove_set = set(remove_indices)
    partial_updates = defaultdict(int)
    for idx, sku_id, day, reduce_qty in partial_adjustments:
        if idx >= len(state.vehicles) or idx in remove_set:
            continue
        try:
            veh = state.vehicles[idx]
            current_qty = veh.cargo.get((sku_id, day), 0)
            reduce_qty = min(int(reduce_qty), int(current_qty))
            if reduce_qty <= 0:
                continue

            veh.cargo[(sku_id, day)] = current_qty - reduce_qty
            if veh.cargo[(sku_id, day)] == 0:
                del veh.cargo[(sku_id, day)]

            for d in range(day, state.data.horizons + 1):
                key = (veh.fact_id, sku_id, d)
                partial_updates[key] += reduce_qty

            if veh.is_empty():
                remove_set.add(idx)
        except Exception:
            continue

    remove_indices = sorted(remove_set, reverse=True)
    removed = [state.vehicles[i] for i in remove_indices if i < len(state.vehicles)]
    state.vehicles = [
        veh for i, veh in enumerate(state.vehicles) if i not in remove_set
    ]

    state.mark_objective_dirty()

    # Pass partial_updates to post_destroy through extra so post_destroy can
    # perform batch restoration and validation centrally. The operator does not
    # update state.s_ikt internally; post_destroy calls batch_update_inventory.
    try:
        extra_payload = (
            {"partial_updates": dict(partial_updates)} if partial_updates else None
        )
    except Exception:
        extra_payload = None
    post_destroy(
        state,
        removed,
        "infeasible_removal",
        prev_obj=prev_obj,
        t0=t0,
        skip_batch_update=False,
        extra=extra_payload,
    )
    print(
        f"[OPLOG] infeasible_removal: removed {len(removed)} vehicles, applied {len(partial_updates)} partial inventory updates ({time.time() - t0:.4f}s)"
    )
    return state


# ---------------------------------------------------------------------
# 4. High-inventory-pressure-oriented removal
# ---------------------------------------------------------------------
def surplus_inventory_removal(
    current: SolutionState, rng: rnd.Generator, degree: float | None = None
):
    """
    Focus on plants with high inventory risk:
      - Aggregate plant_sku_inventory and measure risk using maximum-period
        utilization through the amplification factor max_util - 0.8.
      - Select vehicles from high-risk plants that serve dealers with high
        demand satisfaction, freeing transport capacity.
      - Priority = risk coefficient * destination satisfaction.
    Applicability:
      - Reduce plant inventory overflow risk and create room for later
        replanning by withdrawing shipments.

    Unit note:
      - plant_inv_limit is the maximum SKU quantity a production site can
        store, not a volume limit, and uses the same units as s_ikt.
    """
    t0 = time.time()
    print("[OPLOG] Starting surplus_inventory_removal operator")
    state = current.copy()
    prev_obj = state.objective()
    data = state.data

    if len(state.vehicles) <= 1:
        post_destroy(
            state, [], "surplus_inventory_removal", early=True, prev_obj=prev_obj, t0=t0
        )
        print(
            f"[OPLOG] surplus_inventory_removal: insufficient vehicles ({time.time() - t0:.4f}s)"
        )
        return state

    plant_sku_inventory = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for (plant, sku_id, day), inv in state.s_ikt.items():
        plant_sku_inventory[plant][day][sku_id] += inv

    plant_risk = {}
    for plant, day_dict in plant_sku_inventory.items():
        max_capacity = data.plant_inv_limit[
            plant
        ]  # Maximum SKU quantity, not volume, using the same units as s_ikt.
        if max_capacity <= 0:
            plant_risk[plant] = 0.0
            continue
        utilizations = []
        for day, sku_invs in day_dict.items():
            total_inv_day = np.sum(list(sku_invs.values()))
            utilizations.append(total_inv_day / max_capacity)
        max_util = (
            max(utilizations) if utilizations else 0.0
        )  # Maximum-period utilization indicating the inventory peak.
        risk = max(0.0, max_util - 0.8) * 5  # Amplify linearly above 80% utilization.
        plant_risk[plant] = risk

    high_risk_plants = sorted(plant_risk.items(), key=lambda x: x[1], reverse=True)

    candidates = []
    _, shipments_by_dealer_total, dealer_total_demand = precompute_dealer_shipments(
        state
    )

    for plant, risk in high_risk_plants:
        if risk <= 0:
            continue
        for i, veh in enumerate(state.vehicles):
            if veh.fact_id == plant:
                total_qty = sum(qty for (_, _), qty in veh.cargo.items())
                dealer = veh.dealer_id
                dealer_demand = dealer_total_demand.get(dealer, 0)
                dealer_shipped = shipments_by_dealer_total.get(dealer, 0)
                satisfaction = (
                    dealer_shipped / dealer_demand if dealer_demand > 0 else 1.0
                )
                priority = risk * satisfaction
                candidates.append((i, veh, priority, total_qty))

    if not candidates:
        print(
            f"[OPLOG] surplus_inventory_removal: no high-risk plants ({time.time() - t0:.4f}s)"
        )
        post_destroy(
            state, [], "surplus_inventory_removal", early=True, prev_obj=prev_obj, t0=t0
        )
        return state

    candidates.sort(key=lambda x: x[2], reverse=True)

    degree = float(resolve_degree("surplus_inventory_removal", state, degree))

    num_remove = min(int(len(state.vehicles) * degree), len(candidates))
    if num_remove <= 0:
        print(
            f"[OPLOG] surplus_inventory_removal: degree={degree:.3f}, no vehicles removed ({time.time() - t0:.4f}s)"
        )
        post_destroy(
            state, [], "surplus_inventory_removal", early=True, prev_obj=prev_obj, t0=t0
        )
        return state

    selected_indices = [idx for idx, _, _, _ in candidates[:num_remove]]
    selected_indices.sort(reverse=True)

    removed = [veh for i, veh in enumerate(state.vehicles) if i in selected_indices]
    state.vehicles = [
        veh for i, veh in enumerate(state.vehicles) if i not in selected_indices
    ]

    state.mark_objective_dirty()

    post_destroy(state, removed, "surplus_inventory_removal", prev_obj=prev_obj, t0=t0)
    print(
        f"[OPLOG] surplus_inventory_removal: removed {len(removed)}/{len(current.vehicles)} vehicles ({time.time() - t0:.4f}s)"
    )
    return state


# ---------------------------------------------------------------------
# 5. Shaw similarity-based removal
# ---------------------------------------------------------------------
def shaw_removal(
    current: SolutionState, rng: rnd.Generator, degree: float | None = None
):
    """
    Focus on a related vehicle cluster for similarity-based removal:
      Similarity dimensions: plant/dealer/day proximity, cargo SKU overlap,
      and random noise.
      Weights: fact(3), dealer(2), day(1), cargo(2).
      Seed selection: low utilization, later periods, and high plant inventory
      produce higher scores, then roulette-wheel selection chooses a seed.
    Goal:
      - Create structural gaps, such as removing vehicles from one plant and
        period, to give repair operators room to rebuild loading decisions.

    Unit note:
      - plant_inv_limit is the maximum SKU quantity a plant can store, not a
        volume limit, and uses the same units as s_ikt.
    """
    t0 = time.time()
    print("[OPLOG] Starting shaw_removal operator")
    state = current.copy()
    prev_obj = state.objective()

    if len(state.vehicles) <= 1:
        post_destroy(state, [], "shaw_removal", early=True, prev_obj=prev_obj, t0=t0)
        print(f"[OPLOG] shaw_removal: insufficient vehicles ({time.time() - t0:.4f}s)")
        return state

    degree = float(resolve_degree("shaw_removal", state, degree))

    seed_candidates = []
    plant_inv_limit = state.data.plant_inv_limit
    # Pre-aggregate inventory by (plant, day) to avoid repeated state.s_ikt
    # scans inside the vehicle loop.
    plant_day_inventory = precompute_plant_day_inventory(state)
    for i, veh in enumerate(state.vehicles):
        load = state.compute_veh_load(veh)
        capacity = state.data.veh_type_cap[veh.type]
        utilization = load / capacity if capacity > 0 else 0

        plant_inventory = plant_day_inventory.get((veh.fact_id, veh.day), 0)
        seed_score = (
            (1 - utilization) * 0.4
            + (veh.day / state.data.horizons) * 0.3
            + (plant_inventory / (plant_inv_limit[veh.fact_id] + 1)) * 0.3
        )
        seed_candidates.append((i, seed_score))

    seed_scores = np.array([score for _, score in seed_candidates])
    if seed_scores.sum() > 0:
        probabilities = seed_scores / seed_scores.sum()
        try:
            # Use ensure_pyint to convert possible numpy return values to Python int.
            sel = rng.choice(len(seed_candidates), p=probabilities)
            pos = ensure_pyint(sel)
        except Exception:
            pos = ensure_pyint(rng.integers(0, len(seed_candidates)))
        # pos is returned as a Python int by ensure_pyint.
        seed_idx = int(seed_candidates[pos][0])
    else:
        pos = ensure_pyint(rng.integers(0, len(seed_candidates)))
        seed_idx = int(seed_candidates[pos][0])

    seed_veh = state.vehicles[seed_idx]
    seed_skus = set(sku_id for (sku_id, _) in seed_veh.cargo.keys())

    relatedness = []
    weight_fact, weight_dealer, weight_day, weight_cargo = 3.0, 2.0, 1.0, 2.0
    for i, veh in enumerate(state.vehicles):
        if i == seed_idx:
            continue
        fact_sim = 1.0 if veh.fact_id == seed_veh.fact_id else 0.0
        dealer_sim = 1.0 if veh.dealer_id == seed_veh.dealer_id else 0.0
        day_diff = abs(veh.day - seed_veh.day)
        day_sim = max(0.0, 1.0 - day_diff / state.data.horizons)
        veh_skus = set(sku_id for (sku_id, _) in veh.cargo.keys())
        if seed_skus and veh_skus:
            common_skus = len(seed_skus & veh_skus)
            all_skus = len(seed_skus | veh_skus)
            cargo_sim = common_skus / all_skus if all_skus > 0 else 0.0
        else:
            cargo_sim = 0.0
        total_sim = (
            weight_fact * fact_sim
            + weight_dealer * dealer_sim
            + weight_day * day_sim
            + weight_cargo * cargo_sim
        )
        noise = rng.uniform(-0.1, 0.1)
        total_sim = max(0.0, total_sim + noise)
        relatedness.append((i, total_sim))

    relatedness.sort(key=lambda x: x[1], reverse=True)

    # ``degree`` is the total share to remove, including the seed vehicle.
    num_remove = min(
        max(1, int(np.ceil(len(state.vehicles) * degree))),
        len(relatedness),
    )
    if num_remove <= 0:
        post_destroy(state, [], "shaw_removal", early=True, prev_obj=prev_obj, t0=t0)
        print(
            f"[OPLOG] shaw_removal: degree={degree:.3f}, no vehicles removed ({time.time() - t0:.4f}s)"
        )
        return state

    max_remove_total = max(0, len(state.vehicles) - 1)
    remove_set = {seed_idx}
    for idx, _ in relatedness:
        if len(remove_set) >= max_remove_total:
            break
        remove_set.add(idx)
        if len(remove_set) >= min(max_remove_total, num_remove):
            break
    remove_indices = sorted(remove_set, reverse=True)

    removed = [veh for i, veh in enumerate(state.vehicles) if i in remove_indices]
    state.vehicles = [
        veh for i, veh in enumerate(state.vehicles) if i not in remove_indices
    ]

    state.mark_objective_dirty()

    # Include operator context in the log for attribution and debugging:
    # seed_idx, params, and degree.
    try:
        extra_payload = {
            "seed_idx": int(seed_idx) if "seed_idx" in locals() else None,
            "params": {"degree": float(degree) if degree is not None else None},
            "degree": float(degree) if degree is not None else None,
        }
    except Exception:
        extra_payload = None

    post_destroy(
        state, removed, "shaw_removal", prev_obj=prev_obj, t0=t0, extra=extra_payload
    )
    print(
        f"[OPLOG] shaw_removal: removed {len(removed)}/{len(current.vehicles)} vehicles ({time.time() - t0:.4f}s)"
    )
    return state


# ---------------------------------------------------------------------
# 6. Multi-period / clustering-enhanced Shaw removal
# ---------------------------------------------------------------------
def periodic_shaw_removal(
    current: SolutionState,
    rng: rnd.Generator,
    degree: float | None = None,
    alpha: float | None = None,
    beta: float | None = None,
    gamma: float | None = None,
    k_cluster: int | None = None,
) -> SolutionState:
    """
    Multidimensional feature clustering plus relatedness-based removal:
      Features: [normalized day, dealer demand satisfaction, previous-day
      inventory (normalized), vehicle utilization, plant inventory pressure].
      Procedure:
        1. Build a feature row for each (veh_idx, veh, sku, current day).
        2. Run KMeans clustering with adaptive cluster counts and IQR
           normalization to limit outlier influence.
        3. Select one of the larger clusters, choose a random seed within it,
           and compute member similarity using alpha/beta/gamma dimensional
           differences.
        4. Remove a proportion of low-similarity or nearby allocations at the
           SKU-allocation level.
      Fallback:
        If clustering fails, use simple random whole-vehicle removal.
    """
    t0 = time.time()
    print("[OPLOG] Starting periodic_shaw_removal operator")
    state = current.copy()
    prev_obj = state.objective()
    data = state.data

    if len(state.vehicles) <= 1:
        post_destroy(
            state, [], "periodic_shaw_removal", early=True, prev_obj=prev_obj, t0=t0
        )
        print(f"[OPLOG] periodic_shaw_removal: insufficient vehicles ({time.time() - t0:.4f}s)")
        return state

    defaults = ALNSConfig.get_operator_default("periodic_shaw_removal").get(
        "params", {}
    )
    degree = float(resolve_degree("periodic_shaw_removal", state, degree))
    # other params: prefer explicit kwargs, then tuner (only for missing), then defaults
    params_other = {}
    if alpha is None or beta is None or gamma is None or k_cluster is None:
        params_other = get_adaptive_parameters(
            op_name="periodic_shaw_removal", param_tuner=state.param_tuner
        )
    alpha = _coerce_float(
        alpha
        if alpha is not None
        else params_other.get("alpha", defaults.get("alpha", 0.4)),
        float(defaults.get("alpha", 0.4)),
    )
    beta = _coerce_float(
        beta
        if beta is not None
        else params_other.get("beta", defaults.get("beta", 0.3)),
        float(defaults.get("beta", 0.3)),
    )
    gamma = _coerce_float(
        gamma
        if gamma is not None
        else params_other.get("gamma", defaults.get("gamma", 0.3)),
        float(defaults.get("gamma", 0.3)),
    )
    k_cluster = _coerce_int(
        k_cluster
        if k_cluster is not None
        else params_other.get(
            "k_cluster",
            params_other.get("k_clusters", defaults.get("k_clusters", 3)),
        ),
        int(defaults.get("k_clusters", 3)),
    )
    k_cluster = max(1, k_cluster)

    allocations = []  # (veh_idx, veh, sku, day, feature_vector)

    # Precompute auxiliary data to avoid repeated scans of state.s_ikt inside
    # the vehicle loop.
    _, shipments_by_dealer_total, dealer_total_demand = precompute_dealer_shipments(
        state
    )
    plant_day_inventory = precompute_plant_day_inventory(state)

    for veh_idx, veh in enumerate(state.vehicles):
        plant = veh.fact_id
        dealer = veh.dealer_id
        day = veh.day
        load = state.compute_veh_load(veh)
        cap = data.veh_type_cap[veh.type]

        veh_util = load / cap if cap > 0 else 0
        total_demand = dealer_total_demand.get(dealer, 0)
        dealer_shipped = shipments_by_dealer_total.get(dealer, 0)
        satisfaction = dealer_shipped / total_demand if total_demand > 0 else 1.0

        plant_inv = plant_day_inventory.get((plant, day), 0)

        plant_capacity = data.plant_inv_limit[plant]
        plant_pressure = plant_inv / plant_capacity if plant_capacity > 0 else 1.0
        for (sku, d), qty in list(veh.cargo.items()):
            if d != day:
                continue
            # Previous-day inventory uses day - 1 only when day >= 2.
            # For day == 1, use the current day's inventory because there is no
            # previous day. This avoids reading the immutable day-0 opening
            # inventory, which would make the feature ineffective.
            inv_transfer = (
                state.s_ikt.get((plant, sku, day - 1), 0)
                if day >= 2
                else state.s_ikt.get((plant, sku, day), 0)
            )
            norm_day = day / data.horizons
            norm_inv = (
                min(1.0, inv_transfer / (plant_capacity * 0.1))
                if plant_capacity > 0
                else 0.0
            )
            feature = np.array(
                [norm_day, satisfaction, norm_inv, veh_util, plant_pressure]
            )
            allocations.append((veh_idx, veh, sku, day, feature))

    if not allocations:
        print(
            f"[OPLOG] periodic_shaw_removal: no processable allocations ({time.time() - t0:.4f}s)"
        )
        post_destroy(
            state, [], "periodic_shaw_removal", early=True, prev_obj=prev_obj, t0=t0
        )
        return state

    try:
        features = np.array([a[4] for a in allocations])
        features_median = np.median(features, axis=0)
        features_iqr = np.percentile(features, 75, axis=0) - np.percentile(
            features, 25, axis=0
        )
        features_iqr[features_iqr == 0] = 1.0
        features_normalized = (features - features_median) / features_iqr
        optimal_clusters = min(
            k_cluster, len(allocations), max(1, len(allocations) // 3)
        )

        from sklearn.cluster import KMeans

        kmeans = KMeans(
            n_clusters=optimal_clusters,
            # Use ensure_pyint to normalize numpy return values and prevent
            # np.int64 or 0-d array leakage.
            random_state=ensure_pyint(rng.integers(0, 1000)),
            n_init=3,
            max_iter=50,
            tol=1e-3,
        )  # KMeans uses the "lloyd" algorithm by default.
        labels = kmeans.fit_predict(features_normalized)
        del kmeans
    except Exception as e:
        print(
            f"[OPLOG] periodic_shaw_removal: clustering failed ({type(e).__name__}); falling back to simple random removal"
        )
        num_remove = int(len(state.vehicles) * degree)
        if num_remove > 0:
            # Handle numpy scalar/ndarray returns by converting them to a
            # Python int list with ensure_pylist.
            remove_indices = ensure_pylist(
                rng.choice(range(len(state.vehicles)), size=num_remove, replace=False)
            )
            remove_indices = sorted(remove_indices, reverse=True)
            removed = [
                veh for i, veh in enumerate(state.vehicles) if i in remove_indices
            ]
            state.vehicles = [
                veh for i, veh in enumerate(state.vehicles) if i not in remove_indices
            ]
            state.mark_objective_dirty()
            post_destroy(
                state, removed, "periodic_shaw_removal", prev_obj=prev_obj, t0=t0
            )
            print(
                f"[OPLOG] periodic_shaw_removal: fallback mode - removed {num_remove} vehicles ({time.time() - t0:.4f}s)"
            )
        else:
            post_destroy(
                state, [], "periodic_shaw_removal", early=True, prev_obj=prev_obj, t0=t0
            )
            print(
                f"[OPLOG] periodic_shaw_removal: fallback mode - no vehicles removed ({time.time() - t0:.4f}s)"
            )
        return state

    unique_labels = np.unique(labels)
    if len(unique_labels) == 0:
        print(
            f"[OPLOG] periodic_shaw_removal: no valid cluster labels ({time.time() - t0:.4f}s)"
        )
        post_destroy(
            state, [], "periodic_shaw_removal", early=True, prev_obj=prev_obj, t0=t0
        )
        return state

    # Convert cluster labels and counts to native Python ints to prevent numpy
    # type leakage.
    cluster_sizes = [
        (int(label), int(np.sum(labels == label))) for label in unique_labels
    ]
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)
    eligible_clusters = cluster_sizes[
        : max(1, len(cluster_sizes) // 2)
    ]  # Take the larger half of the clusters.
    if not eligible_clusters:
        print(f"[OPLOG] periodic_shaw_removal: no eligible clusters ({time.time() - t0:.4f}s)")
        post_destroy(
            state, [], "periodic_shaw_removal", early=True, prev_obj=prev_obj, t0=t0
        )
        return state

    # Handle numpy scalar/ndarray returns by converting the first item to
    # Python int, then fall back to integers and finally to an available
    # cluster label.
    try:
        # Prefer ensure_pyint to convert possible numpy return values to Python int.
        selected_cluster = ensure_pyint(
            rng.choice([label for label, _ in eligible_clusters])
        )
    except Exception:
        try:
            selected_cluster = ensure_pyint(rng.integers(0, len(eligible_clusters)))
        except Exception:
            # Final fallback: use the first eligible cluster label as Python int.
            try:
                selected_cluster = int(eligible_clusters[0][0])
            except Exception:
                selected_cluster = ensure_pyint([eligible_clusters[0][0]])
    # Convert numpy index arrays to Python int lists to avoid type
    # inconsistencies in later indexing and serialization.
    cluster_indices = ensure_pylist(np.where(labels == selected_cluster)[0])
    cluster_allocs = [allocations[i] for i in cluster_indices]
    if len(cluster_allocs) <= 1:
        print(f"[OPLOG] periodic_shaw_removal: too few elements in cluster ({time.time() - t0:.4f}s)")
        post_destroy(
            state, [], "periodic_shaw_removal", early=True, prev_obj=prev_obj, t0=t0
        )
        return state

    # Use ensure_pyint to normalize random integer return values.
    seed_idx = ensure_pyint(rng.integers(0, len(cluster_allocs)))
    seed_feature = cluster_allocs[int(seed_idx)][4]

    similarities = []
    for i, alloc in enumerate(cluster_allocs):
        if i == seed_idx:
            continue
        feat = alloc[4]
        time_diff = abs(feat[0] - seed_feature[0])
        demand_cos = max(0, np.dot(feat[1:3], seed_feature[1:3])) / (
            np.linalg.norm(feat[1:3]) * np.linalg.norm(seed_feature[1:3]) + 1e-8
        )
        op_diff = np.linalg.norm(feat[3:5] - seed_feature[3:5])
        sim = alpha * time_diff + beta * (1 - demand_cos) + gamma * op_diff
        noise = rng.uniform(-0.1, 0.1) * sim
        sim += noise
        similarities.append((alloc, sim))

    similarities.sort(key=lambda x: x[1])
    num_remove = int(len(cluster_allocs) * degree)
    num_remove = min(num_remove, len(similarities))

    # Clean up empty vehicles after the loop instead of modifying the list
    # repeatedly inside it.
    empties_idx = set()

    # Collect partial rollback updates and update vehicle cargo accordingly.
    partial_updates = defaultdict(int)
    for i in range(num_remove):
        alloc, _sim = similarities[i]
        veh_idx, veh, sku_id, day, _ = alloc
        qty = veh.cargo.get((sku_id, day), 0)
        try:
            reduce_qty = int(qty)
        except (TypeError, ValueError):
            try:
                reduce_qty = int(np.array(qty).flat[0])
            except Exception:
                reduce_qty = 0
        if reduce_qty <= 0:
            continue

        veh.cargo[(sku_id, day)] = qty - reduce_qty
        if veh.cargo[(sku_id, day)] == 0:
            del veh.cargo[(sku_id, day)]

        for d in range(day, data.horizons + 1):
            key = (veh.fact_id, sku_id, d)
            partial_updates[key] += reduce_qty

        if veh.is_empty():
            empties_idx.add(veh_idx)

    # Remove vehicles by index in reverse order after the loop.
    if empties_idx:
        remove_list = sorted(list(empties_idx), reverse=True)
        removed = [state.vehicles[i] for i in remove_list if i < len(state.vehicles)]
        state.vehicles = [
            veh for idx, veh in enumerate(state.vehicles) if idx not in empties_idx
        ]
    else:
        removed = []

    state.mark_objective_dirty()
    # Let post_destroy apply partial_updates in batch. The operator no longer
    # directly modifies state.s_ikt or veh.cargo outside this controlled path.
    try:
        extra_payload = (
            {"partial_updates": dict(partial_updates)} if partial_updates else None
        )
    except Exception:
        extra_payload = None
    post_destroy(
        state,
        removed,
        "periodic_shaw_removal",
        prev_obj=prev_obj,
        t0=t0,
        skip_batch_update=False,
        extra=extra_payload,
    )
    print(
        f"[OPLOG] periodic_shaw_removal: processed cluster {selected_cluster}, removed {len(removed)} empty vehicles/allocations ({time.time() - t0:.4f}s)"
    )
    return state


# ---------------------------------------------------------------------
# 7. Cross-period dealer block removal for large jumps
# ---------------------------------------------------------------------
def cross_period_dealer_removal(
    current: SolutionState, rng: rnd.Generator, degree: float | None = None
):
    """
    Large-jump destroy operator:
      - Aggregate each dealer's cross-period vehicles, then prioritize dealers
        with large period spans and many vehicles.
      - Remove vehicles in batches within the selected dealer's cluster to
        create room for cross-period reconstruction.
      - The design goal is to escape local jitter and trigger larger follow-up
        repair changes.
    """
    t0 = time.time()
    print("[OPLOG] Starting cross_period_dealer_removal operator")
    state = current.copy()
    prev_obj = state.objective()

    total_vehicles = len(state.vehicles)
    if total_vehicles <= 2:
        post_destroy(
            state,
            [],
            "cross_period_dealer_removal",
            early=True,
            prev_obj=prev_obj,
            t0=t0,
        )
        return state

    degree = float(resolve_degree("cross_period_dealer_removal", state, degree))

    # Phase-based intensity: more aggressive early, more conservative late, with
    # temporary pressure when stagnation persists.
    if bool(getattr(ALNSConfig, "CROSS_PERIOD_REMOVAL_PHASED", True)):
        try:
            progress = 0.0
            if getattr(state, "tracker", None) is not None:
                cur_it = int(state.tracker.get_iteration())
                max_it = int(state.tracker.get_max_iterations())
                if max_it > 0:
                    progress = max(0.0, min(1.0, float(cur_it) / float(max_it)))

            if progress < 0.45:
                degree *= float(
                    getattr(ALNSConfig, "CROSS_PERIOD_REMOVAL_EARLY_BOOST", 1.35)
                )
            elif progress > 0.80:
                degree *= float(
                    getattr(ALNSConfig, "CROSS_PERIOD_REMOVAL_LATE_SHRINK", 0.75)
                )

            stagnation_window = int(
                getattr(ALNSConfig, "CROSS_PERIOD_REMOVAL_STAGNATION_WINDOW", 80)
            )
            no_improve = int(getattr(state.tracker, "no_improve_iters", 0))
            if no_improve >= stagnation_window:
                degree *= float(
                    getattr(ALNSConfig, "CROSS_PERIOD_REMOVAL_STAGNATION_BOOST", 1.20)
                )
        except Exception:
            pass

    degree = max(0.15, min(0.90, degree))
    target_remove = max(2, int(total_vehicles * degree))
    target_remove = min(target_remove, total_vehicles - 1)
    if target_remove <= 0:
        post_destroy(
            state,
            [],
            "cross_period_dealer_removal",
            early=True,
            prev_obj=prev_obj,
            t0=t0,
        )
        return state

    by_dealer = defaultdict(list)
    for idx, veh in enumerate(state.vehicles):
        by_dealer[getattr(veh, "dealer_id", None)].append((idx, veh))

    if not by_dealer:
        post_destroy(
            state,
            [],
            "cross_period_dealer_removal",
            early=True,
            prev_obj=prev_obj,
            t0=t0,
        )
        return state

    dealer_candidates = []
    for dealer, items in by_dealer.items():
        if dealer is None or len(items) == 0:
            continue
        days = {int(getattr(v, "day", 0)) for _, v in items}
        span = (max(days) - min(days) + 1) if days else 1
        score = float(len(items)) * float(span)
        dealer_candidates.append((dealer, score, items))

    if not dealer_candidates:
        post_destroy(
            state,
            [],
            "cross_period_dealer_removal",
            early=True,
            prev_obj=prev_obj,
            t0=t0,
        )
        return state

    dealer_candidates.sort(key=lambda x: x[1], reverse=True)
    topn = max(1, len(dealer_candidates) // 3)
    pool = dealer_candidates[:topn]
    weights = np.array([x[1] for x in pool], dtype=float)
    if float(weights.sum()) > 0:
        weights = weights / weights.sum()
    else:
        weights = None

    try:
        sel = int(np.atleast_1d(rng.choice(len(pool), p=weights))[0])
    except Exception:
        sel = int(np.atleast_1d(rng.integers(0, len(pool)))[0])

    _dealer, _score, selected_items = pool[sel]
    selected_items = list(selected_items)

    # Prefer removing low-load vehicles to clean up less efficient vehicles in
    # the block first.
    def _load_ratio(v):
        try:
            loaded = float(state.compute_veh_load(v))
            cap = float(getattr(v, "max_capacity", 0.0))
            return loaded / cap if cap > 1e-9 else 0.0
        except Exception:
            return 0.0

    selected_items.sort(key=lambda x: _load_ratio(x[1]))
    remove_indices = [idx for idx, _ in selected_items[:target_remove]]

    if not remove_indices:
        post_destroy(
            state,
            [],
            "cross_period_dealer_removal",
            early=True,
            prev_obj=prev_obj,
            t0=t0,
        )
        return state

    remove_index_set = set(remove_indices)
    removed = [veh for i, veh in enumerate(state.vehicles) if i in remove_index_set]
    state.vehicles = [
        veh for i, veh in enumerate(state.vehicles) if i not in remove_index_set
    ]

    state.mark_objective_dirty()

    post_destroy(
        state,
        removed,
        "cross_period_dealer_removal",
        prev_obj=prev_obj,
        t0=t0,
    )
    print(
        f"[OPLOG] cross_period_dealer_removal: removed {len(removed)}/{len(current.vehicles)} vehicles ({time.time() - t0:.4f}s)"
    )
    return state


def period_range_removal(
    current: SolutionState,
    rng: rnd.Generator,
    degree: float | None = None,
    window_ratio: float | None = None,
):
    """
    Destroy by continuous period segment:
      - Select a continuous day segment from [1, horizons].
      - Prefer removing low-load vehicles in that segment to create room for
        replanning.
      - Goal: break cross-day local optima while keeping the removal size
        controlled.
    """
    t0 = time.time()
    print("[OPLOG] Starting period_range_removal operator")
    state = current.copy()
    prev_obj = state.objective()

    total_vehicles = len(state.vehicles)
    if total_vehicles <= 2:
        post_destroy(
            state, [], "period_range_removal", early=True, prev_obj=prev_obj, t0=t0
        )
        return state

    defaults = ALNSConfig.get_operator_default("period_range_removal").get("params", {})
    degree = float(resolve_degree("period_range_removal", state, degree))
    if window_ratio is None:
        try:
            params_other = get_adaptive_parameters(
                op_name="period_range_removal", param_tuner=state.param_tuner
            )
        except Exception:
            params_other = {}
        window_ratio = params_other.get(
            "window_ratio", defaults.get("window_ratio", 0.30)
        )

    window_ratio = _coerce_float(
        window_ratio, float(defaults.get("window_ratio", 0.30))
    )
    window_ratio = max(0.15, min(0.70, window_ratio))
    degree = max(0.15, min(0.85, degree))

    horizons = max(1, int(getattr(state.data, "horizons", 1)))
    window_len = max(1, min(horizons, int(round(horizons * window_ratio))))

    # Candidate windows [start_day, end_day].
    windows = []
    day_counts = defaultdict(int)
    for veh in state.vehicles:
        day_counts[int(getattr(veh, "day", 1))] += 1

    for start_day in range(1, horizons - window_len + 2):
        end_day = start_day + window_len - 1
        veh_cnt = 0
        span_fill = 0
        for d in range(start_day, end_day + 1):
            c = int(day_counts.get(d, 0))
            veh_cnt += c
            if c > 0:
                span_fill += 1
        if veh_cnt <= 0:
            continue
        # Score windows higher when they contain more vehicles and provide
        # more complete period coverage.
        score = float(veh_cnt) * (1.0 + float(span_fill) / float(window_len))
        windows.append((start_day, end_day, veh_cnt, score))

    if not windows:
        post_destroy(
            state, [], "period_range_removal", early=True, prev_obj=prev_obj, t0=t0
        )
        return state

    weights = np.array([w[3] for w in windows], dtype=float)
    if float(weights.sum()) > 0:
        weights = weights / weights.sum()
    else:
        weights = None

    try:
        widx = ensure_pyint(rng.choice(len(windows), p=weights))
    except Exception:
        widx = ensure_pyint(rng.integers(0, len(windows)))

    start_day, end_day, _veh_cnt, _score = windows[int(widx)]

    candidates = []
    for idx, veh in enumerate(state.vehicles):
        day = int(getattr(veh, "day", 1))
        if start_day <= day <= end_day:
            candidates.append((idx, veh))

    if not candidates:
        post_destroy(
            state, [], "period_range_removal", early=True, prev_obj=prev_obj, t0=t0
        )
        return state

    target_remove = max(1, int(total_vehicles * degree))
    target_remove = min(target_remove, len(candidates), total_vehicles - 1)
    if target_remove <= 0:
        post_destroy(
            state, [], "period_range_removal", early=True, prev_obj=prev_obj, t0=t0
        )
        return state

    def _load_ratio(v):
        try:
            loaded = float(state.compute_veh_load(v))
            cap = float(getattr(v, "max_capacity", 0.0))
            return loaded / cap if cap > 1e-9 else 0.0
        except Exception:
            return 0.0

    candidates.sort(key=lambda x: _load_ratio(x[1]))
    remove_indices = [idx for idx, _ in candidates[:target_remove]]
    remove_set = set(remove_indices)

    removed = [veh for i, veh in enumerate(state.vehicles) if i in remove_set]
    state.vehicles = [
        veh for i, veh in enumerate(state.vehicles) if i not in remove_set
    ]

    state.mark_objective_dirty()

    extra_payload = {
        "segment_start_day": int(start_day),
        "segment_end_day": int(end_day),
        "segment_window_len": int(window_len),
        "segment_window_ratio": float(window_ratio),
        "segment_candidate_count": int(len(candidates)),
    }
    post_destroy(
        state,
        removed,
        "period_range_removal",
        prev_obj=prev_obj,
        t0=t0,
        extra=extra_payload,
    )
    print(
        f"[OPLOG] period_range_removal: segment [{start_day},{end_day}] removed {len(removed)}/{len(current.vehicles)} vehicles ({time.time() - t0:.4f}s)"
    )
    return state


# ---------------------------------------------------------------------
# 7. Whole-path removal from plant to dealer
# ---------------------------------------------------------------------
def path_removal(
    current: SolutionState, rng: rnd.Generator, degree: float | None = None
):
    """
    Evaluate (plant, dealer) paths as a whole:
      - Path metrics: demand satisfaction, average vehicle cost, and share of
        total vehicles.
      - Prefer removing less efficient vehicles from paths with high
        satisfaction, high cost, and many vehicles.
      - If a path contains more than half of all vehicles, reduce the removal
        ratio to avoid structural collapse from clearing it at once.
    """
    t0 = time.time()
    print("[OPLOG] Starting path_removal operator")
    state = current.copy()
    prev_obj = state.objective()
    data = state.data

    if len(state.vehicles) <= 1:
        post_destroy(state, [], "path_removal", early=True, prev_obj=prev_obj, t0=t0)
        print(f"[OPLOG] path_removal: insufficient vehicles ({time.time() - t0:.4f}s)")
        return state

    path_info = {}
    for veh in state.vehicles:
        path = (veh.fact_id, veh.dealer_id)
        if path not in path_info:
            path_info[path] = {
                "vehicles": [],
                "total_demand": 0,
                "shipped": 0,
                "total_cost": 0,
            }
        path_info[path]["vehicles"].append(veh)
        path_info[path]["total_cost"] += data.veh_type_cost[veh.type]
        for (sku_id, _), qty in veh.cargo.items():
            path_info[path]["shipped"] += qty

    for (plant, dealer), info in path_info.items():
        dealer_demand = sum(data.demands.get((dealer, sku), 0) for sku in data.all_skus)
        info["total_demand"] = dealer_demand

    if not path_info:
        print(f"[OPLOG] path_removal: no valid paths ({time.time() - t0:.4f}s)")
        post_destroy(state, [], "path_removal", early=True, prev_obj=prev_obj, t0=t0)
        return state

    path_scores = []
    for path, info in path_info.items():
        if not info["vehicles"]:
            continue
        satisfaction = (
            info["shipped"] / info["total_demand"] if info["total_demand"] > 0 else 1.0
        )
        avg_cost = info["total_cost"] / len(info["vehicles"]) if info["vehicles"] else 0
        path_score = (
            satisfaction * 0.4
            + (avg_cost / max(data.veh_type_cost.values())) * 0.3
            + (len(info["vehicles"]) / len(state.vehicles)) * 0.3
        )
        path_scores.append((path, path_score))

    if not path_scores:
        print(f"[OPLOG] path_removal: no valid path scores ({time.time() - t0:.4f}s)")
        post_destroy(state, [], "path_removal", early=True, prev_obj=prev_obj, t0=t0)
        return state

    path_scores.sort(key=lambda x: x[1], reverse=True)
    top_paths = path_scores[: max(1, len(path_scores) // 3)]  # Top one-third by score.
    weights = np.array([score for _, score in top_paths])
    weights = weights / np.sum(weights) if np.sum(weights) > 0 else None

    try:
        # Handle numpy scalar/ndarray returns; prefer ensure_pyint to convert
        # possible numpy values to Python int.
        try:
            sel = rng.choice(len(top_paths), p=weights)
            selected_idx = ensure_pyint(sel)
        except Exception:
            selected_idx = ensure_pyint(rng.integers(0, len(top_paths)))
        target_path = top_paths[int(selected_idx)][0]
    except Exception:
        # Fall back to robust integer handling and ensure Python int values;
        # finally fall back to a random path.
        try:
            selected_idx = int(np.atleast_1d(rng.integers(0, len(top_paths)))[0])
            target_path = top_paths[selected_idx][0]
        except Exception:
            # Choose a random path and ensure it is a native Python tuple or
            # original path type to avoid serialization/comparison issues from
            # numpy.object_.
            candidate = rng.choice([path for path, _ in path_scores])
            # Convert a potentially numpy-typed candidate to a native Python
            # type for compatibility with np.ndarray, np.generic, np.str_, and
            # similar values.
            try:
                if isinstance(candidate, np.ndarray):
                    # ndarray -> tuple of native elements.
                    try:
                        candidate = tuple(candidate.tolist())
                    except Exception:
                        candidate = tuple(candidate)
                elif isinstance(
                    candidate, (np.generic, np.integer, np.floating, np.bool_)
                ):
                    # numpy scalar -> Python scalar via item().
                    try:
                        candidate = candidate.item()
                    except Exception:
                        try:
                            candidate = int(candidate)
                        except (TypeError, ValueError):
                            try:
                                candidate = float(candidate)
                            except (TypeError, ValueError):
                                candidate = str(candidate)
                # list -> tuple.
                if isinstance(candidate, list):
                    candidate = tuple(candidate)
                # If candidate is bytes-like or numpy.str_, convert to str
                if isinstance(candidate, (bytes,)):
                    try:
                        candidate = candidate.decode()
                    except Exception:
                        candidate = str(candidate)
            except Exception:
                # Final fallback to guarantee a serializable native Python
                # candidate.
                try:
                    candidate = (
                        tuple(candidate)
                        if isinstance(candidate, (list, tuple))
                        else str(candidate)
                    )
                except Exception:
                    candidate = str(candidate)

            # Ensure target_path is a pure Python tuple or original scalar.
            if isinstance(candidate, (list, tuple)):
                target_path = tuple(candidate)
            else:
                target_path = candidate

    path_vehicles = [
        (i, veh)
        for i, veh in enumerate(state.vehicles)
        if (veh.fact_id, veh.dealer_id) == target_path
    ]

    if not path_vehicles or len(path_vehicles) <= 1:
        print(f"[OPLOG] path_removal: insufficient vehicles on selected path ({time.time() - t0:.4f}s)")
        post_destroy(state, [], "path_removal", early=True, prev_obj=prev_obj, t0=t0)
        return state

    degree = float(resolve_degree("path_removal", state, degree))

    path_ratio = len(path_vehicles) / len(state.vehicles)
    if path_ratio > 0.5:
        degree *= 0.5

    num_remove = int(len(path_vehicles) * degree)
    if num_remove <= 0:
        post_destroy(state, [], "path_removal", early=True, prev_obj=prev_obj, t0=t0)
        print(
            f"[OPLOG] path_removal: degree={degree:.3f}, no vehicles removed ({time.time() - t0:.4f}s)"
        )
        return state

    num_remove = (
        min(num_remove, len(path_vehicles) - 1) if len(path_vehicles) > 1 else 0
    )
    if num_remove <= 0:
        post_destroy(state, [], "path_removal", early=True, prev_obj=prev_obj, t0=t0)
        print(
            f"[OPLOG] path_removal: num_remove=0 after calculation, early exit ({time.time() - t0:.4f}s)"
        )
        return state

    vehicle_metrics = []
    max_cost = max(data.veh_type_cost.values())
    for i, veh in path_vehicles:
        load = state.compute_veh_load(veh)
        capacity = data.veh_type_cap[veh.type]
        utilization = load / capacity if capacity > 0 else 0

        cost = data.veh_type_cost[veh.type]
        unit_cost = cost / load if load > 0 else max_cost
        score = 0.5 * utilization + 0.5 * (1 - unit_cost / (max_cost + 1))

        vehicle_metrics.append((i, score))

    vehicle_metrics.sort(key=lambda x: x[1])
    remove_indices = [idx for idx, _ in vehicle_metrics[:num_remove]]
    remove_indices.sort(reverse=True)

    removed = [veh for i, veh in enumerate(state.vehicles) if i in remove_indices]
    state.vehicles = [
        veh for i, veh in enumerate(state.vehicles) if i not in remove_indices
    ]

    state.mark_objective_dirty()

    post_destroy(state, removed, "path_removal", prev_obj=prev_obj, t0=t0)
    print(
        f"[OPLOG] path_removal: removed {len(removed)}/{len(path_vehicles)} vehicles from path {target_path} ({time.time() - t0:.4f}s)"
    )
    return state


def inventory_level_removal(
    current: SolutionState, rng: rnd.Generator, degree: float | None = None
) -> SolutionState:
    """
    Inventory-level removal.

    Core idea:
      Identify abnormal dealer inventory levels, either too low or too high,
      and remove delivery decisions related to those dealers to promote
      reconstruction.

    Characteristics:
      - Adds a dealer-side perspective to surplus_inventory_removal, which
        focuses on plant inventory.
      - Identifies both stockout and overstock problems.
      - Promotes reoptimization toward inventory balance.
    """
    t0 = time.time()
    print("[OPLOG] Starting inventory_level_removal operator")
    state = current.copy()
    prev_obj = state.objective()

    total_vehicles = len(state.vehicles)
    if total_vehicles <= 1:
        post_destroy(
            state,
            [],
            "inventory_level_removal",
            early=True,
            prev_obj=prev_obj,
            t0=t0,
        )
        print(
            f"[OPLOG] inventory_level_removal: insufficient vehicles ({time.time() - t0:.4f}s)"
        )
        return state

    degree = float(resolve_degree("inventory_level_removal", state, degree))
    degree = max(0.1, min(0.8, degree))

    # Calculate plant-SKU inventory anomaly scores. ``s_ikt`` is keyed by
    # (plant, sku, day); there is no dealer inventory state in this model.
    anomaly_score_map = {}  # (plant, sku) -> anomaly_score

    try:
        # Sorted: both sources are sets, so insertion order would otherwise
        # depend on PYTHONHASHSEED and change which vehicles this removes.
        for plant in sorted(state.data.plants):
            for sku_id in sorted(state.data.skus_plant.get(plant, set())):
                total_demand = sum(
                    qty
                    for (_dealer, demand_sku), qty in state.data.demands.items()
                    if demand_sku == sku_id
                )
                if total_demand <= 0:
                    continue

                plant_invs = [
                    state.s_ikt.get((plant, sku_id, t), 0)
                    for t in range(state.data.horizons + 1)
                ]
                if not plant_invs:
                    continue

                min_inv = min(plant_invs)
                max_inv = max(plant_invs)
                safety_threshold = max(5, int(total_demand * 0.1))
                upper_bound = max(1, int(total_demand * 1.5))

                anomaly = 0.0
                if min_inv < safety_threshold:
                    anomaly += (safety_threshold - min_inv) / max(
                        1.0, float(safety_threshold)
                    )
                if max_inv > upper_bound:
                    anomaly += (
                        (max_inv - upper_bound) / max(1.0, float(upper_bound))
                    ) * 0.8

                if anomaly > 0.01:
                    anomaly_score_map[(plant, sku_id)] = anomaly

    except Exception as e:
        print(f"[OPLOG] inventory_level_removal: failed to calculate anomaly scores: {e}")
        return state

    if not anomaly_score_map:
        post_destroy(
            state,
            [],
            "inventory_level_removal",
            early=True,
            prev_obj=prev_obj,
            t0=t0,
        )
        print(
            f"[OPLOG] inventory_level_removal: no anomalous dealers ({time.time() - t0:.4f}s)"
        )
        return state

    # Select targets with weighted sampling.
    anomaly_pairs = list(anomaly_score_map.keys())
    anomaly_scores = np.array([anomaly_score_map[p] for p in anomaly_pairs])
    anomaly_scores = anomaly_scores / float(anomaly_scores.sum())

    num_targets = int(
        np.clip(len(anomaly_pairs) * degree, 1, min(3, len(anomaly_pairs)))
    )

    try:
        target_indices = rng.choice(
            len(anomaly_pairs), size=num_targets, replace=False, p=anomaly_scores
        )
        target_pairs = [anomaly_pairs[int(i)] for i in target_indices]
    except Exception:
        target_pairs = anomaly_pairs[:num_targets]

    # Remove related vehicles.
    removed_vehicles = []
    for plant, sku_id in target_pairs:
        to_remove = [
            veh
            for veh in state.vehicles
            if veh.fact_id == plant
            and any(sku_key == sku_id for (sku_key, _) in veh.cargo.keys())
        ]
        removed_vehicles.extend(to_remove)

    removed_vehicles = list(set(removed_vehicles))

    if removed_vehicles:
        state.vehicles = [veh for veh in state.vehicles if veh not in removed_vehicles]
        state.mark_objective_dirty()

    post_destroy(
        state, removed_vehicles, "inventory_level_removal", prev_obj=prev_obj, t0=t0
    )
    print(
        f"[OPLOG] inventory_level_removal: removed {len(removed_vehicles)} vehicles, "
        f"targeted {len(target_pairs)} pairs ({time.time() - t0:.4f}s)"
    )
    return state


def create_inventory_level_removal(degree: float | None = None):
    """Factory that creates an inventory-level removal wrapper."""
    defaults = ALNSConfig.get_operator_default("inventory_level_removal").get(
        "params", {}
    )
    degree = degree if degree is not None else defaults.get("degree", 0.25)
    name = "inventory_level_removal"
    return DestroyOperatorWrapper(
        func=inventory_level_removal, name=name, degree=degree
    )
