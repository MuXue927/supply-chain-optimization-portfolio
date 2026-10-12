"""
repair_utils.py

Helper and utility functions extracted from
multi_period_delivery/alns/repair_operators.py.
They are used by operators in repair_operators.py and preserve the original
implementation behavior.
"""

import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .alns_config import default_config as ALNSConfig
from .alnsopt import SolutionState
from .InputDataALNS import DataALNS
from .param_tuner import ParamAutoTuner
from .vehicle import Vehicle


# MinimalVehicle: lightweight vehicle surrogate for fast simulation in regret operator
# Purpose:
# - Provide a minimal API compatible with alnsopt.veh_loading so regret simulations can
#   avoid the full Vehicle dataclass construction overhead in tight inner loops.
# - Keep implementation intentionally small and allocation-cheap.
# Notes:
# - Instances are NOT registered with global Vehicle._id_counter and thus ids are local-only.
# - Attributes/methods implemented: fact_id, dealer_id, type, day, data, id, max_capacity,
#   capacity, cargo (dict), _loaded_volume, load(), is_empty().
# - This surrogate is intended only for temporary simulation on copied states (tmp) where
#   strict global id uniqueness is not required. Do NOT use in persistent state.
class MinimalVehicle:
    _local_id_seq = 0

    def __init__(self, fact_id, dealer_id, veh_type, day, data):
        self.fact_id = fact_id
        self.dealer_id = dealer_id
        self.type = veh_type
        self.day = day
        self.data = data
        # lightweight id (unique per process but not interfering with Vehicle._id_counter)
        self.id = f"minveh_{MinimalVehicle._local_id_seq}"
        MinimalVehicle._local_id_seq += 1
        # capacities
        try:
            self.max_capacity = data.veh_type_cap[veh_type]
        except Exception:
            self.max_capacity = 0
        self.capacity = self.max_capacity
        self.cargo = {}
        self._loaded_volume = 0

    def load(self, sku_id: str, num: int) -> int:
        # simplified load similar to Vehicle.load but without heavy checks
        if num <= 0:
            return num
        sku_size = self.data.sku_sizes.get(sku_id, 0)
        if sku_size <= 0:
            raise ValueError(f"Invalid sku_size for {sku_id}")
        max_by_cap = self.capacity // sku_size
        num_loaded = min(num, max_by_cap)
        if num_loaded <= 0:
            return num
        prev = self.cargo.get((sku_id, self.day), 0)
        self.cargo[(sku_id, self.day)] = prev + num_loaded
        self.capacity -= num_loaded * sku_size
        self._loaded_volume = getattr(self, "_loaded_volume", 0) + num_loaded * sku_size
        return num - num_loaded

    def is_empty(self):
        return not bool(self.cargo)


def _get_actual_reference(state) -> Optional[float]:
    tracker = getattr(state, "tracker", None)
    if tracker is None:
        return None

    for attr in ("best_actual_obj", "initial_actual_obj"):
        try:
            ref = float(getattr(tracker, attr, float("inf")))
        except Exception:
            continue
        if np.isfinite(ref) and abs(ref) > 1e-12:
            return ref
    return None


def compute_mixed_operator_reward(
    state: SolutionState,
    prev_obj: Optional[float],
    new_obj: Optional[float] = None,
    actual_obj: Optional[float] = None,
):
    """Compute a mixed search+actual reward for operator feedback."""
    if prev_obj is None:
        return None, 0.0, False

    if new_obj is None:
        try:
            new_obj = state.objective()
        except Exception:
            return None, 0.0, False

    if actual_obj is None:
        try:
            actual_obj = float(state.calculate_cost())
        except Exception:
            actual_obj = None

    try:
        if not np.isfinite(float(prev_obj)) or not np.isfinite(float(new_obj)):
            return new_obj, 0.0, False
    except Exception:
        return new_obj, 0.0, False

    try:
        search_weight = float(getattr(ALNSConfig, "MIXED_REWARD_SEARCH_WEIGHT", 0.60))
        actual_weight = float(getattr(ALNSConfig, "MIXED_REWARD_ACTUAL_WEIGHT", 0.40))
        score_scale = float(getattr(ALNSConfig, "MIXED_REWARD_SCORE_SCALE", 100.0))
    except Exception:
        search_weight = 0.60
        actual_weight = 0.40
        score_scale = 100.0

    search_norm = 0.0
    if np.isfinite(prev_obj) and np.isfinite(new_obj):
        search_norm = float((prev_obj - new_obj) / max(1.0, abs(prev_obj)))

    actual_norm = 0.0
    actual_ref = _get_actual_reference(state)
    if actual_ref is not None and actual_obj is not None and np.isfinite(actual_obj):
        actual_norm = float((actual_ref - actual_obj) / max(1.0, abs(actual_ref)))

    mixed_reward = score_scale * (
        search_weight * search_norm + actual_weight * actual_norm
    )
    if not np.isfinite(mixed_reward):
        mixed_reward = 0.0

    success = mixed_reward > 0.0
    return new_obj, float(mixed_reward), bool(success)


# ---------------------------
# Basic resource and demand utilities
# ---------------------------
def _construct_resource_availability(
    state: SolutionState, plant: str, sku_id: str, day: int
) -> float:
    """
    Return available inventory for (plant, sku_id, day) without modifying state.

    Performance notes:
      - The original implementation scanned state.vehicles on every call, which
        was expensive on hot paths.
      - This implementation uses a lightweight cache stored in
        state._vehicles_by_plant_day_map:
        dict[(plant, day)] -> dict[sku_id] -> used_qty. It is built lazily and
        cleared when state changes structurally.
      - The cache reduces repeated O(V) scans to one O(V) build followed by
        O(1) lookups.
    """
    # Previous-period inventory and current-period production.
    prev_inv = state.s_ikt.get((plant, sku_id, day - 1), 0)
    production = state.data.sku_prod_each_day.get((plant, sku_id, day), 0)

    # Reuse the cached loaded-quantity map aggregated by (plant, day).
    vbkey = "_vehicles_by_plant_day_map"
    vehicles_map = getattr(state, vbkey, None)
    if vehicles_map is None:
        # Build once and cache on state; mark_objective_dirty clears it when the
        # structure changes.
        vehicles_map = {}
        for veh in state.vehicles:
            key = (veh.fact_id, veh.day)
            m = vehicles_map.get(key)
            if m is None:
                m = {}
                vehicles_map[key] = m
            for (sku_k, d), q in veh.cargo.items():
                if d == veh.day:
                    m[sku_k] = m.get(sku_k, 0) + q
        try:
            setattr(state, vbkey, vehicles_map)
        except Exception:
            # If the attribute cannot be set, continue without caching for
            # compatibility with read-only objects.
            pass

    used_inv = vehicles_map.get((plant, day), {}).get(sku_id, 0)
    return max(0, prev_inv + production - used_inv)


def _construct_resource_pool(
    state: SolutionState, data: DataALNS
) -> Dict[Tuple[str, str, int], float]:
    """
    Build a resource pool by precomputing available inventory for
    (plant, sku, day).

    Performance notes:
      - Use a per-state cache (_resource_pool_cache) to avoid rebuilding the
        resource pool repeatedly in a short period.
      - mark_objective_dirty clears the cache when state changes structurally.
    """
    # Reuse the cache when available.
    cache = getattr(state, "_resource_pool_cache", None)
    if cache is not None:
        return cache

    pool = build_resource_pool(state, data)
    try:
        setattr(state, "_resource_pool_cache", pool)
    except Exception:
        # Skip caching when the attribute cannot be set, such as on read-only
        # objects.
        pass
    return pool


def get_unmet_demand(state: SolutionState) -> Dict[Tuple[str, str], int]:
    """
    Return all unmet demand entries for the current state:
      key: (dealer, sku_id)
      value: unmet_quantity

    Performance notes:
      - This is a hot function in the regret operator and may be called
        frequently in a short period.
      - A lightweight state snapshot cache
        ((len(vehicles), last_objective)) avoids repeated compute_shipped()
        calls and demand scans when state has not changed structurally.
      - The cache is stored in state._unmet_cache and should be invalidated
        when state.mark_objective_dirty() is called. If invalidation is not
        handled elsewhere, this function rebuilds the cache when it detects a
        structural change.
    """
    try:
        # lightweight snapshot that changes when vehicles list or objective baseline changes
        snapshot = (
            len(getattr(state, "vehicles", [])),
            getattr(state, "last_objective", None),
        )
    except Exception:
        snapshot = None

    # try to reuse cached value if present and snapshot matches
    cache = getattr(state, "_unmet_cache", None)
    if cache is not None:
        try:
            cached_snapshot, cached_removal = cache
            if snapshot is not None and cached_snapshot == snapshot:
                return dict(cached_removal)
        except Exception:
            # fallthrough to recompute
            pass

    # compute shipped only once for this call
    shipped = state.compute_shipped()
    removal_list: Dict[Tuple[str, str], int] = {}
    # iterate demands directly (avoid extra allocations where possible)
    for key, total_demand in state.data.demands.items():
        shipped_qty = shipped.get(key, 0)
        unmet_qty = total_demand - shipped_qty
        if unmet_qty > 0:
            removal_list[key] = unmet_qty

    # store into cache (best-effort; ignore failures on read-only wrappers)
    try:
        setattr(state, "_unmet_cache", (snapshot, removal_list))
    except Exception:
        pass

    return dict(removal_list)


# ---------------------------
# veh_loading wrapper around alnsopt.veh_loading
# ---------------------------
def veh_loading(
    state: SolutionState,
    veh: Vehicle,
    orders: Dict[str, int],
    commit: bool = True,
    skip_compute_inventory: bool = True,
) -> bool:
    """
    Wrapper that delegates to
    multi_period_delivery.alns.alnsopt.veh_loading.

    Added parameter:
      skip_compute_inventory: when commit=True and this flag is True,
        temporarily suppress calls to state.compute_inventory to reduce full
        table recomputation in batch scenarios. The caller must invoke
        state.compute_inventory() once after the batch completes. This
        optimization is optional and falls back to the original behavior on
        exceptions.

    Notes:
      - Do not change the signature of alnsopt.veh_loading. This wrapper
        temporarily replaces state.compute_inventory to avoid repeated full
        updates.
      - Always restore the original compute_inventory function after any
        exception or incompatible condition to preserve consistency.
    """
    try:
        from . import alnsopt as _alnsopt
    except Exception:
        raise RuntimeError(
            "Cannot import multi_period_delivery.alns.alnsopt; the veh_loading "
            "wrapper cannot work."
        )

    if not commit:
        # Simulate on a copy without touching the real state.
        tmp = state.copy()
        _alnsopt.veh_loading(tmp, veh, orders)
        return True

    # commit == True: suppress compute_inventory when possible for batch updates.
    if skip_compute_inventory and hasattr(state, "compute_inventory"):
        orig_compute = getattr(state, "compute_inventory")
        try:
            # Establish the inventory baseline before suppressing updates so
            # incremental deductions are not applied to uninitialized s_ikt.
            if not getattr(state, "s_initialized", False):
                orig_compute()
            # Temporarily disable compute_inventory with a no-op replacement.
            state.compute_inventory = lambda: None
            _alnsopt.veh_loading(state, veh, orders)
        except Exception:
            # On any exception, restore the original function and re-raise so
            # the caller can execute fallback logic.
            try:
                state.compute_inventory = orig_compute
            except Exception:
                pass
            raise
        finally:
            # Protectively restore the original function.
            try:
                state.compute_inventory = orig_compute
            except Exception:
                pass
        return True
    else:
        # Original behavior: do not suppress compute_inventory.
        _alnsopt.veh_loading(state, veh, orders)
        return True


def select_vehicle_type_for_volume(required_volume: float, data: DataALNS) -> str:
    """Choose the smallest vehicle type that can carry required volume; fallback to the largest type."""
    suitable = []
    for vt, cap in data.veh_type_cap.items():
        if cap >= required_volume:
            suitable.append((vt, cap))
    if suitable:
        return min(suitable, key=lambda x: x[1])[0]
    return max(data.veh_type_cap.keys(), key=lambda x: data.veh_type_cap[x])


def find_existing_route_vehicles(
    state: SolutionState, plant: str, dealer: str, day: int
) -> List[Vehicle]:
    """Return non-full vehicles on the same (plant, dealer, day), sorted by remaining capacity desc."""
    candidates: List[Vehicle] = []
    for veh in state.vehicles:
        if veh.fact_id != plant or veh.dealer_id != dealer or veh.day != day:
            continue
        try:
            if veh.is_full():
                continue
        except Exception:
            pass
        candidates.append(veh)
    candidates.sort(key=lambda v: getattr(v, "capacity", 0), reverse=True)
    return candidates


def commit_load_with_tracking(
    state: SolutionState,
    veh: Vehicle,
    sku_id: str,
    qty: int,
    day: int,
    skip_compute_inventory: bool = True,
) -> Tuple[int, List[Vehicle]]:
    """
    Commit a single-SKU load and return (actual_loaded_qty, appended_new_vehicles).
    actual_loaded_qty is measured as route-level sku delta to robustly include overflow-created vehicles.
    """
    request_qty = int(max(0, qty))
    if request_qty <= 0:
        return 0, []

    before_len = len(state.vehicles)
    before_route_qty = 0
    for v in state.vehicles:
        if v.fact_id == veh.fact_id and v.day == day:
            before_route_qty += int(v.cargo.get((sku_id, day), 0))

    veh_loading(
        state,
        veh,
        {sku_id: request_qty},
        commit=True,
        skip_compute_inventory=skip_compute_inventory,
    )

    after_route_qty = 0
    for v in state.vehicles:
        if v.fact_id == veh.fact_id and v.day == day:
            after_route_qty += int(v.cargo.get((sku_id, day), 0))

    actual_loaded = max(0, int(after_route_qty - before_route_qty))
    new_vs = state.vehicles[before_len:]
    return actual_loaded, new_vs


# ---------------------------
# Precomputation and resource-pool utilities for greedy repair
# ---------------------------
def precompute_total_available(
    state: SolutionState, data: DataALNS, supply_chain: Dict[Tuple[str, str], Set[str]]
) -> Dict[Tuple[str, str], float]:
    """
    Precompute theoretical cross-plant, cross-period availability for each
    (dealer, sku_id) as previous inventory plus production.
    Return dict[(dealer, sku_id)] -> total_available (int/float).
    """
    total_avail = {}
    for (dealer, sku_id), demand in data.demands.items():
        s = 0
        # Aggregate only plants that can supply this SKU to the dealer.
        plants = [
            plant
            for (plant, dealer_key), skus in supply_chain.items()
            if dealer_key == dealer and sku_id in skus
        ]
        for plant in plants:
            for day in range(1, data.horizons + 1):
                s += state.s_ikt.get(
                    (plant, sku_id, day - 1), 0
                ) + data.sku_prod_each_day.get((plant, sku_id, day), 0)
        total_avail[(dealer, sku_id)] = s
    return total_avail


class ResourcePool:
    """
    Lightweight proxy that stores resource availability in a nested structure
    self._data[plant][sku_id][day] = available

    Provides a dict-like tuple-key interface so existing callers that use
    resource_pool.get((plant, sku, day), 0) or resource_pool[(plant, sku, day)]
    continue to work without change, but hot-path tuple lookups avoid building
    large flat dicts repeatedly.
    """

    def __init__(self):
        self._data = {}

    def _ensure(self, plant, sku):
        if plant not in self._data:
            self._data[plant] = {}
        if sku not in self._data[plant]:
            self._data[plant][sku] = {}

    def get(self, key, default=0):
        # Accept either tuple key (plant, sku, day) or explicit (plant, sku, day)
        try:
            plant, sku, day = key
        except Exception:
            return default
        return self._data.get(plant, {}).get(sku, {}).get(day, default)

    def get_day_map(self, key_or_tuple):
        """
        Return the internal day->available dict for (plant, sku).
        Accepts either (plant, sku) tuple or the same tuple wrapped as (plant, sku).
        Returns an empty dict if not present to allow safe .items()/get() usage in hot loops.
        This avoids tuple-key lookups for per-day access in performance-critical code paths.
        """
        try:
            plant, sku = key_or_tuple
        except Exception:
            return {}
        return self._data.get(plant, {}).get(sku, {})

    def __getitem__(self, key):
        try:
            plant, sku, day = key
        except Exception:
            raise KeyError(key)
        return self._data[plant][sku][day]

    def __setitem__(self, key, value):
        try:
            plant, sku, day = key
        except Exception:
            raise KeyError(key)
        self._ensure(plant, sku)
        self._data[plant][sku][day] = value

    def __contains__(self, key):
        try:
            plant, sku, day = key
        except Exception:
            return False
        return day in self._data.get(plant, {}).get(sku, {})

    def items(self):
        # iterate flat (plant, sku, day) -> value pairs
        for plant, sku_map in self._data.items():
            for sku, day_map in sku_map.items():
                for day, val in day_map.items():
                    yield (plant, sku, day), val

    def keys(self):
        for plant, sku_map in self._data.items():
            for sku, day_map in sku_map.items():
                for day in day_map.keys():
                    yield (plant, sku, day)

    def __repr__(self):
        return f"<ResourcePool plants={len(self._data)}>"


def build_resource_pool(state: SolutionState, data: DataALNS):
    """
    Build and return a ResourcePool object.
    Preserve compatibility with the legacy tuple-key get/set interface while
    using a nested structure internally to reduce tuple-key hash overhead.
    """
    pool = ResourcePool()
    # Aggregate current vehicle usage into a local map to avoid repeatedly
    # scanning state.vehicles inside the loop.
    vehicles_by_plant_day = {}
    for veh in state.vehicles:
        key = (veh.fact_id, veh.day)
        vehicles_by_plant_day.setdefault(key, []).append(veh)

    for plant in data.plants:
        for sku_id in data.skus_plant.get(plant, []):
            for day in range(1, data.horizons + 1):
                prev_inv = state.s_ikt.get((plant, sku_id, day - 1), 0)
                prod = data.sku_prod_each_day.get((plant, sku_id, day), 0)
                used = 0
                for veh in vehicles_by_plant_day.get((plant, day), []):
                    used += veh.cargo.get((sku_id, day), 0)
                pool[(plant, sku_id, day)] = max(0, prev_inv + prod - used)
    return pool


# ---------------------------------------------------------------------
# Adaptive operator parameters and batch post-processing
# ---------------------------------------------------------------------
def get_adaptive_parameters(op_name: str, param_tuner: ParamAutoTuner):
    """
    Return the current dynamic tuning parameters for an operator, or an empty
    dictionary when ParamAutoTuner is unavailable or raises an exception.
    :param op_name: operator name, such as 'random_removal'
    :return: dict
    """
    if not param_tuner:
        return {}
    try:
        params = param_tuner.get_operator_params(op_name)
        return params if params else {}
    except Exception as e:
        print(f"[WARNING] get_adaptive_parameters({op_name}) exception: {e}")
        return {}


def _veh_volume(veh: Vehicle, data: DataALNS) -> float:
    # Use cargo-based recomputation here (instead of _loaded_volume cache) because
    # some operators may leave cached volume temporarily stale during complex edits.
    # Consolidation must be conservative to avoid accidental overload merges.
    total = 0.0
    for (sku_id, _day), qty in getattr(veh, "cargo", {}).items():
        total += float(qty) * float(data.sku_sizes.get(sku_id, 0.0))
    return total


def consolidate_route_vehicles(
    state: SolutionState, max_pairs_per_group: int = 300
) -> int:
    """
    Merge small vehicles into larger ones for the same (plant, dealer, day) route.
    Returns merged vehicle count.
    """
    data = state.data
    groups = {}
    for veh in state.vehicles:
        try:
            if veh.is_empty():
                continue
        except Exception:
            pass
        key = (veh.fact_id, veh.dealer_id, veh.day)
        groups.setdefault(key, []).append(veh)

    merged_total = 0
    new_vehicles = []

    for _key, vehicles in groups.items():
        if len(vehicles) <= 1:
            new_vehicles.extend(vehicles)
            continue

        # Small-first packing into larger-capacity candidates.
        vehicles.sort(key=lambda v: _veh_volume(v, data))
        pairs_used = 0

        while len(vehicles) > 1 and pairs_used < max_pairs_per_group:
            smallest = vehicles[0]
            small_vol = _veh_volume(smallest, data)
            merged = False

            for i in range(len(vehicles) - 1, 0, -1):
                target = vehicles[i]
                target_vol = _veh_volume(target, data)
                target_cap = float(data.veh_type_cap.get(target.type, 0.0))
                if target_vol + small_vol <= target_cap + 1e-9:
                    for (sku, d), qty in smallest.cargo.items():
                        target.cargo[(sku, d)] = target.cargo.get((sku, d), 0) + qty
                    vehicles.pop(0)
                    merged_total += 1
                    pairs_used += 1
                    merged = True
                    break

                # If neither current vehicle can hold the combined load,
                # upgrade the target to the smallest vehicle type that can.
                combined_volume = target_vol + small_vol
                upgraded_type = None
                for candidate_type, candidate_cap in sorted(
                    data.veh_type_cap.items(), key=lambda item: item[1]
                ):
                    if float(candidate_cap) + 1e-9 >= combined_volume:
                        upgraded_type = candidate_type
                        break
                if upgraded_type is not None and upgraded_type != target.type:
                    merged_cargo = dict(target.cargo)
                    for (sku, d), qty in smallest.cargo.items():
                        merged_cargo[(sku, d)] = (
                            merged_cargo.get((sku, d), 0) + qty
                        )
                    # Upgrade in place so the vehicle keeps its identity. This
                    # allows post_repair() to roll back an inserted vehicle even
                    # when consolidation changed its type.
                    target.type = upgraded_type
                    target.max_capacity = int(data.veh_type_cap[upgraded_type])
                    target.cargo = merged_cargo
                    vehicles.pop(0)
                    merged_total += 1
                    pairs_used += 1
                    merged = True
                    break

            if not merged:
                new_vehicles.append(vehicles.pop(0))

        new_vehicles.extend(vehicles)

    if merged_total > 0:
        state.vehicles = new_vehicles
        try:
            state.mark_objective_dirty()
        except Exception:
            pass
    return merged_total


def post_repair(
    state: SolutionState,
    inserted: List[Vehicle],
    op_name: str,
    prev_obj: float = None,
    t0: float = None,
    extra: dict = None,
):
    """
    Post-process a batch repair:
    - Call state.compute_inventory() once.
    - If hard-constraint violations such as vehicle overload are found, try to
      roll back newly inserted vehicles by id and validate again.
    - Compute improvement, report it to param_tuner and logs, and print a
      short JSON-style log entry.
    """
    merged_after_repair = 0
    # Apply the minimum-load post-processing step to every completed repair.
    # This keeps the fill logic independent of operator selection while still
    # allowing repairs to cross the minimum-load threshold using spare stock.
    try:
        from .repair_operators import fill_minimum_load_repair

        fill_minimum_load_repair(state, None)
    except Exception:
        pass
    try:
        if getattr(ALNSConfig, "REPAIR_ENABLE_POST_CONSOLIDATION", True):
            max_pairs = int(
                getattr(ALNSConfig, "REPAIR_POST_CONSOLIDATION_MAX_PAIRS", 300)
            )
            merged_after_repair = consolidate_route_vehicles(
                state, max_pairs_per_group=max(1, max_pairs)
            )
    except Exception:
        merged_after_repair = 0

    try:
        # Lightweight inserted-only feasibility check to avoid a full validate/compute_inventory
        # in the common (no-violation) case. Conservative semantics:
        #  - Check inserted vehicles for obvious vehicle-overload violations locally.
        #  - If the local check fails or raises, fall back to suppressed validate() as before.
        feasible_tmp = True
        violations_tmp = {}
        try:
            if inserted:
                data = state.data
                veh_over_ids = []
                for v in inserted:
                    try:
                        # compute loaded volume for this vehicle (consider only cargo entries matching the vehicle's day)
                        loaded_vol = 0.0
                        for (sku_k, d), q in getattr(v, "cargo", {}).items():
                            if d == getattr(v, "day", None):
                                sku_size = float(data.sku_sizes.get(sku_k, 0.0))
                                loaded_vol += float(q) * sku_size
                        cap = float(
                            data.veh_type_cap.get(getattr(v, "type", None), 0.0)
                        )
                        # small epsilon to avoid floating point jitter
                        if loaded_vol > cap + 1e-9:
                            veh_over_ids.append(getattr(v, "id", None))
                    except Exception:
                        # any unexpected error -> mark as inconclusive so we fallback to validate()
                        veh_over_ids = ["__error__"]
                        break
                if veh_over_ids:
                    feasible_tmp = False
                    violations_tmp = {"veh_over_load": veh_over_ids}
                else:
                    feasible_tmp = True
                    violations_tmp = {}
            else:
                feasible_tmp = True
                violations_tmp = {}
        except Exception:
            # Fallback: try suppressed validate() (original safe path)
            try:
                with state.suppress_inventory_ctx():
                    feasible_tmp, violations_tmp = state.validate()
            except Exception:
                feasible_tmp, violations_tmp = True, {}
        # If either check detected potential overload or negative inventory, perform full compute to get an
        # accurate baseline before continuing (same conservative behavior as before).
        if (
            not feasible_tmp
            and violations_tmp
            and (
                violations_tmp.get("veh_over_load")
                or violations_tmp.get("negative_inventory")
            )
        ):
            try:
                state.compute_inventory()
            except Exception:
                # If compute_inventory fails, continue; later logic will handle consistency/rollback.
                pass
    except Exception:
        # Defensive: do not let post_repair fail due to inventory checks
        pass

    # Build a diagnostic summary of inserted vehicles. Limit sampling to avoid
    # excessive string construction and I/O overhead.
    try:
        total_inserted = len(inserted) if inserted else 0
        # Sample only the first 20 vehicles to avoid constructing a large
        # diagnostic list when many vehicles are inserted.
        inserted_summary = []
        if inserted:
            for v in inserted[:20]:
                try:
                    inserted_summary.append(
                        {
                            "veh_id": getattr(v, "id", None),
                            "fact_id": getattr(v, "fact_id", None),
                            "dealer_id": getattr(v, "dealer_id", None),
                            "type": getattr(v, "type", None),
                            "day": getattr(v, "day", None),
                            "cargo": dict(getattr(v, "cargo", {})),
                        }
                    )
                except Exception:
                    inserted_summary.append({"veh": str(v)})
        # Print the total count and a sample summary of at most 20 entries.
        try:
            print(
                f"[DIAG] post_repair op={op_name} inserted_cnt={total_inserted} merged_after={merged_after_repair} sample_cnt={len(inserted_summary)}"
            )
        except Exception:
            pass
    except Exception:
        inserted_summary = []

    # Quickly validate the current state and roll back vehicles inserted by
    # this operation when vehicle overload is detected. If the lightweight
    # inserted-only check already produced feasible_tmp, reuse it to avoid
    # repeated validation calls.
    try:
        # Always run the full validator before rollback decisions. The
        # lightweight inserted-only check above covers vehicle overload but not
        # negative inventory or plant-capacity violations. Demand strictness is
        # handled inside validate(), not by skipping the physical checks.
        feasible, violations = state.validate()
    except Exception:
        feasible, violations = False, {}

    rolled_back = False
    hard_violation_exists = bool(
        violations
        and (
            violations.get("veh_over_load")
            or violations.get("negative_inventory")
            or violations.get("plant_inv_exceed")
            or violations.get("cargo_day_mismatch")
        )
    )
    if not feasible and hard_violation_exists:
        try:
            # Collect vehicle ids to roll back.
            to_rollback_ids = {
                getattr(v, "id", None)
                for v in (inserted or [])
                if getattr(v, "id", None) is not None
            }
            if to_rollback_ids:
                before_cnt = len(state.vehicles)
                # Keep vehicles that were not inserted by this operation.
                state.vehicles = [
                    v
                    for v in state.vehicles
                    if getattr(v, "id", None) not in to_rollback_ids
                ]
                rolled_back = True
                # Invalidate caches and rebuild the baseline inventory.
                try:
                    state.mark_objective_dirty()
                except Exception:
                    pass
                try:
                    state.compute_inventory()
                except Exception:
                    pass
                # Validate again after rollback.
                try:
                    feasible_after, violations_after = state.validate()
                except Exception:
                    feasible_after, violations_after = False, {}
                print(
                    f"[DIAG] post_repair op={op_name} rollback_applied removed={before_cnt - len(state.vehicles)} feasible_after={feasible_after} violations_after_count={sum(len(v) for v in violations_after.values() if v)}"
                )
            else:
                print(
                    f"[DIAG] post_repair op={op_name} detected veh_over_load but no inserted ids to rollback"
                )
        except Exception as e:
            try:
                print(f"[DIAG] post_repair op={op_name} rollback error: {e}")
            except Exception:
                pass

    # Compute and report improvement. Prefer this module's incremental,
    # cache-friendly compute_improvement() to avoid unnecessary full
    # recomputation. Only the improvement delta feeds the tuner and audit
    # below; the objective/cached flags returned alongside it are ignored.
    improvement = 0.0
    try:
        if prev_obj is not None:
            # compute_improvement will use state's cached last_objective / objective_dirty semantics
            # and avoid calling state.objective() when possible.
            _, improvement, _ = compute_improvement(state, prev_obj)
        else:
            # No prev_obj provided: try to reuse cached objective if present without forcing compute.
            if getattr(state, "last_objective", None) is not None and not getattr(
                state, "objective_dirty", True
            ):
                improvement = 0.0
            else:
                # last resort: attempt to call objective() which may trigger compute_inventory()
                try:
                    state.objective()
                    improvement = 0.0
                except Exception:
                    improvement = 0.0
    except Exception:
        improvement = 0.0

    # Emit a short JSON-style log entry.
    elapsed = time.time() - t0 if t0 is not None else None
    try:
        print(
            f"[OPJSON] {{'tag':'repair','op':'{op_name}','inserted_cnt':{len(inserted) if inserted else 0},'rolled_back':{rolled_back},'improvement':{improvement},'elapsed_sec':{elapsed}}}"
        )
    except Exception:
        pass

    # Record repair-operator feasibility statistics for offline analysis of
    # success rates and failure types.
    try:
        if hasattr(state, "record_repair_audit") and callable(
            getattr(state, "record_repair_audit")
        ):
            state.record_repair_audit(
                op_name=op_name,
                feasible=bool(feasible),
                violations=violations,
                rolled_back=bool(rolled_back),
                inserted_cnt=len(inserted) if inserted else 0,
                improvement=float(improvement),
            )
    except Exception:
        pass


# ---------------------------
# Smart batch helpers
# ---------------------------
def _get_unsatisfied_demands(state: SolutionState, data: DataALNS):
    shipped = state.compute_shipped()
    unsatisfied = []
    for (dealer, sku_id), demand in data.demands.items():
        shipped_qty = shipped.get((dealer, sku_id), 0)
        if demand > shipped_qty:
            unsatisfied.append(
                {
                    "dealer": dealer,
                    "sku_id": sku_id,
                    "remain_demand": demand - shipped_qty,
                    "total_demand": demand,
                    "priority": demand - shipped_qty,
                }
            )
    return unsatisfied


def _create_smart_batches(unsatisfied_demands, data, batch_size: int):
    if not unsatisfied_demands:
        return []
    sorted_demands = sorted(
        unsatisfied_demands, key=lambda x: x["priority"], reverse=True
    )
    batches = []
    for i in range(0, len(sorted_demands), batch_size):
        batches.append(sorted_demands[i : i + batch_size])
    return batches


def _process_demand_batch(state, batch, resource_pool, data, rng):
    """
    Process one batch: find the best allocation and commit loads through
    veh_loading.
    Return batch_progress (bool), indicating whether at least one load was
    committed.
    """
    batch_progress = False
    inserted_vehicles = []  # Vehicles inserted in this batch for logs or analysis.

    for demand_info in batch:
        dealer = demand_info["dealer"]
        sku_id = demand_info["sku_id"]
        remain_demand = demand_info["remain_demand"]
        if remain_demand <= 0:
            continue
        supply_chain = data.construct_supply_chain()
        available_plants = [
            plant
            for (plant, dealer_key), skus in supply_chain.items()
            if dealer_key == dealer and sku_id in skus
        ]
        if not available_plants:
            continue
        best = _find_best_allocation(
            remain_demand, sku_id, available_plants, resource_pool, data
        )
        if not best:
            continue
        plant, day, available_qty = best
        loaded_total = 0

        # Fill existing vehicles first to avoid creating unnecessary vehicles.
        for existing in find_existing_route_vehicles(state, plant, dealer, day):
            if remain_demand <= 0 or available_qty <= 0:
                break
            request_qty = min(remain_demand, available_qty)
            try:
                actual_loaded, new_vs = commit_load_with_tracking(
                    state,
                    existing,
                    sku_id,
                    request_qty,
                    day,
                    skip_compute_inventory=True,
                )
            except Exception:
                continue
            if actual_loaded <= 0:
                continue
            loaded_total += actual_loaded
            remain_demand -= actual_loaded
            available_qty -= actual_loaded
            if new_vs:
                inserted_vehicles.extend(new_vs)
            batch_progress = True

        # Create new vehicles to cover any remaining demand.
        if remain_demand > 0 and available_qty > 0:
            veh_type = _select_optimal_vehicle_type(remain_demand, sku_id, data)
            request_qty = min(remain_demand, available_qty)
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
                if actual_loaded > 0:
                    loaded_total += actual_loaded
                    if new_vs:
                        inserted_vehicles.extend(new_vs)
                    else:
                        inserted_vehicles.append(vehicle)
                    batch_progress = True
            except Exception:
                continue

        if loaded_total > 0:
            resource_pool[(plant, sku_id, day)] = max(
                0, resource_pool.get((plant, sku_id, day), 0) - loaded_total
            )
    return batch_progress, inserted_vehicles


def _find_best_allocation(remain_demand, sku_id, available_plants, resource_pool, data):
    best_score = -1
    best = None
    for plant in available_plants:
        for day in range(1, data.horizons + 1):
            available_qty = resource_pool.get((plant, sku_id, day), 0)  # Available inventory.
            if available_qty <= 0:
                continue
            satisfaction_ratio = min(1.0, available_qty / remain_demand)  # Demand satisfaction ratio.
            efficiency_score = available_qty / max(
                1, available_qty + remain_demand
            )  # Share of available inventory.
            score = satisfaction_ratio * 0.7 + efficiency_score * 0.3
            if score > best_score:
                best_score = score
                best = (plant, day, available_qty)
    return best


def _select_optimal_vehicle_type(demand_qty, sku_id, data):
    required_volume = demand_qty * data.sku_sizes[sku_id]
    return select_vehicle_type_for_volume(required_volume, data)


def _force_load_remaining_demands(state: SolutionState, data: DataALNS):
    """
    When the main loop cannot fully satisfy all demand, attempt a conservative
    fill of remaining demand using the smallest suitable vehicles and preferred
    plant/day choices. This is a fallback mechanism to reduce omissions when an
    operator reaches its time limit.
    """
    shipped = state.compute_shipped()
    inserted_vehicles = []  # Vehicles inserted for logs or analysis.
    for (dealer, sku_id), demand in data.demands.items():
        remain = demand - shipped.get((dealer, sku_id), 0)
        if remain <= 0:
            continue
        supply_chain = data.construct_supply_chain()
        available_plants = [
            plant
            for (plant, dealer_key), skus in supply_chain.items()
            if dealer_key == dealer and sku_id in skus
        ]
        if not available_plants:
            continue
        best = None
        best_avail = 0
        for plant in available_plants:
            for day in range(1, data.horizons + 1):
                avail = _construct_resource_availability(state, plant, sku_id, day)
                if avail > best_avail:
                    best_avail = avail
                    best = (plant, day)
        if best is None or best_avail <= 0:
            continue

        plant, day = best
        required_volume = remain * data.sku_sizes.get(sku_id, 0)
        veh_type = select_vehicle_type_for_volume(required_volume, data)

        remain_to_load = min(remain, int(best_avail))
        for existing in find_existing_route_vehicles(state, plant, dealer, day):
            if remain_to_load <= 0:
                break
            try:
                actual_loaded, new_vs = commit_load_with_tracking(
                    state,
                    existing,
                    sku_id,
                    remain_to_load,
                    day,
                    skip_compute_inventory=True,
                )
            except Exception:
                continue
            if actual_loaded <= 0:
                continue
            remain_to_load -= actual_loaded
            if new_vs:
                inserted_vehicles.extend(new_vs)

        if remain_to_load > 0:
            vehicle = Vehicle(plant, dealer, veh_type, day, data)
            try:
                actual_loaded, new_vs = commit_load_with_tracking(
                    state,
                    vehicle,
                    sku_id,
                    remain_to_load,
                    day,
                    skip_compute_inventory=True,
                )
                if actual_loaded > 0:
                    if new_vs:
                        inserted_vehicles.extend(new_vs)
                    else:
                        inserted_vehicles.append(vehicle)
            except Exception:
                continue

    return inserted_vehicles


# ---------------------------
# Improvement computation: fast estimation and exact evaluation
# ---------------------------
def compute_improvement_fast_add_vehicle(
    state: SolutionState,
    plant: str,
    dealer: str,
    veh_type: str,
    day: int,
    sku_id: str,
    qty: int,
):
    """
    Quickly approximate the objective impact of adding one vehicle loaded with
    a single SKU and quantity qty.
    Notes:
      - This function does not modify state or copy the full SolutionState.
      - Use structural information from the objective for approximation:
          improvement_unscaled ≈ param_pun_factor1 * qty - veh_cost - min_load_penalty
      - Return (new_obj_est, improvement_scaled, success_flag).
    Important optimization:
      - To avoid expensive state.objective()/compute_inventory() calls during
        candidate filtering, this implementation does not call
        state.objective(). new_obj_est is None, and callers should use
        impro_scaled and success_flag only for filtering. Run an exact
        simulation on a copy when precise evaluation is needed.
    Limitation:
      - This estimate ignores infeasibility caused by inventory overflows and
        other constraints. Use it only as a cheap filter.
    """
    data = state.data
    try:
        scale = float(getattr(ALNSConfig, "SCALE_FACTOR", 1e-3))
    except Exception:
        scale = 1e-3

    # Basic parameters.
    veh_cost = float(data.veh_type_cost.get(veh_type, 0.0))
    try:
        benefit = float(data.param_pun_factor1) * float(qty)
    except Exception:
        benefit = float(qty)

    # Estimate the minimum-load penalty when loaded volume is below the
    # vehicle's min_load.
    sku_size = float(data.sku_sizes.get(sku_id, 0.0))
    load_volume = sku_size * float(qty)
    min_load = float(data.veh_type_min_load.get(veh_type, 0.0))
    min_load_penalty = 0.0
    if load_volume < min_load:
        try:
            min_load_penalty = float(data.param_pun_factor3) * (min_load - load_volume)
        except Exception:
            min_load_penalty = 0.0

    # Estimate the unscaled improvement, where positive means improvement.
    impro_unscaled = benefit - veh_cost - min_load_penalty

    # Convert to the objective's scaled units.
    impro_scaled = impro_unscaled * scale

    # Do not call state.objective() here, to avoid expensive operations such as
    # compute_inventory.
    new_obj_est = None

    success = impro_scaled > 0.0
    return new_obj_est, float(impro_scaled), bool(success)


def compute_improvement_delta(
    state: SolutionState,
    prev_obj: Optional[float],
    dealer: str,
    sku_id: str,
    qty: int,
    veh_type: str,
):
    """
    Best-effort incremental delta computation for adding a single-SKU vehicle.

    Returns (new_obj_est, improvement, success).
    This function avoids calling state.objective() / compute_inventory() when possible by
    relying on state.shipped_cache and state.objective_dirty semantics. It is intentionally
    conservative: it only succeeds when a safe incremental estimate can be produced.

    Preconditions for success:
      - prev_obj is provided (scaled objective)
      - state.shipped_cache is present and state.objective_dirty is False (i.e., cached)
      - The candidate is a single-SKU load described by (dealer, sku_id, qty, veh_type)

    The incremental model:
      delta_unscaled = veh_cost + min_load_penalty_of_added_vehicle - demand_penalty_reduction
      new_obj = prev_obj + delta_unscaled * scale_factor
      improvement = prev_obj - new_obj

    If any required information is missing or non-finite, returns (None, 0.0, False).
    """
    try:
        if prev_obj is None:
            return None, 0.0, False

        # Only use the delta path when the state's cached shipped info appears valid.
        if getattr(state, "objective_dirty", True):
            return None, 0.0, False
        shipped_cache = getattr(state, "shipped_cache", None)
        if shipped_cache is None:
            return None, 0.0, False

        data = state.data
        try:
            scale_factor = float(getattr(ALNSConfig, "SCALE_FACTOR", 1e-3))
        except Exception:
            scale_factor = 1e-3

        # current shipped and demand
        current_shipped = int(shipped_cache.get((dealer, sku_id), 0))
        demand = int(data.demands.get((dealer, sku_id), 0))

        # reduction in unmet demand due to adding qty (cannot exceed current unmet)
        unmet_now = max(0, demand - current_shipped)
        reduction = min(int(qty), unmet_now)

        # unscaled components
        try:
            demand_penalty_factor = float(getattr(data, "param_pun_factor1", 1.0))
        except Exception:
            demand_penalty_factor = 1.0
        demand_penalty_reduction = demand_penalty_factor * reduction

        vehicle_cost = float(data.veh_type_cost.get(veh_type, 0.0))
        sku_size = float(data.sku_sizes.get(sku_id, 0.0))
        loaded_vol = float(qty) * sku_size
        min_load = float(data.veh_type_min_load.get(veh_type, 0.0))
        min_load_penalty = 0.0
        if loaded_vol < min_load:
            try:
                min_load_penalty = float(getattr(data, "param_pun_factor3", 0.0)) * (
                    min_load - loaded_vol
                )
            except Exception:
                min_load_penalty = 0.0

        delta_unscaled = vehicle_cost + min_load_penalty - demand_penalty_reduction
        delta_scaled = delta_unscaled * scale_factor

        new_obj = float(prev_obj + delta_scaled)
        improvement = float(np.clip(prev_obj - new_obj, -1e9, 1e9))
        success = improvement > 0.0
        return new_obj, improvement, success
    except Exception:
        return None, 0.0, False


def compute_improvement(state: SolutionState, prev_obj: Optional[float] = None):
    """
    Compute improvement and return (new_obj, improvement, success).

    Performance notes:
      - Accept an optional prev_obj from the caller to avoid recomputing the
        baseline objective.
      - Use state.last_objective and state.objective_dirty to avoid repeated
        state.objective() calls on the same state. Cache new_obj after the
        first computation for later reuse, with a safe fallback.
      - Preserve existing behavior: when prev_obj is None, still return
        (new_obj, 0.0, False).
    """
    try:
        # Prefer cached objective when available to avoid recomputation
        last_obj = getattr(state, "last_objective", None)
        dirty = getattr(state, "objective_dirty", True)
        if last_obj is not None and not dirty:
            new_obj = last_obj
        else:
            new_obj = state.objective()
            # Best-effort cache the computed objective for subsequent calls.
            try:
                state.last_objective = new_obj
                state.objective_dirty = False
            except Exception:
                # Ignore caching failures, such as on read-only wrappers.
                pass

        _, improvement, success = compute_mixed_operator_reward(
            state,
            prev_obj,
            new_obj=new_obj,
        )
    except Exception:
        new_obj = None
        improvement = 0.0
        success = False
    return new_obj, improvement, success
