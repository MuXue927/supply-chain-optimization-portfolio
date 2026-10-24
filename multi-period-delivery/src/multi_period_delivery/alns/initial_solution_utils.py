"""
High-performance utilities for constructing an initial ALNS solution.

Overall approach in ``improved_initial_solution``:
    1. Initialize a resource pool of available inventory across plants, SKUs,
       and periods.
    2. Prioritize demands by scarcity, supply-demand ratio, and demand size.
    3. Allocate demands sequentially to vehicles.
    4. Repair inventory, remove empty vehicles, and merge small vehicles.
    5. Return a feasible solution after recomputing inventory.

Key performance characteristics:
    - Local variable caching reduces repeated attribute lookups.
    - Plants are ranked by available inventory to improve allocation matches.
    - Timeout checks prevent initialization from stalling.
    - Vehicle merging groups by ``(plant, dealer, day)`` to limit combinations.
    - Negative-inventory repair uses local incremental adjustments.

Function overview:
    improved_initial_solution : Entry point for the five-stage process.
    _initialize_resource_pool : Build cumulative ``(plant, sku, day)`` supply.
    _prioritize_demands       : Compute and sort demand priorities.
    _sequential_allocation    : Allocate demands to vehicles.
    _repair_solution          : Repair inventory, empty vehicles, and small loads.
    _merge_small_vehicles     : Merge small vehicles to reduce fleet size.
    _select_best_vehicle      : Select the vehicle type that best fits capacity.

Notes:
    - Local imports from ``.alnsopt`` avoid circular dependencies.
    - The top-level Vehicle import is retained for the existing dependency
      structure and compatibility.

Maintenance:
    - New initial-solution strategies can be added as functions and selected in
      ``improved_initial_solution``.
    - ``initial_build_stats`` is attached to the solution:
          {
            'resource_pool_time': float,
            'prioritize_time': float,
            'allocation_time': float,
            'repair_time': float,
            'parallel_used': bool,
            'repair_iterations': int
          }
"""

# =========================
# Standard library
# =========================
import time
from collections import defaultdict

from .alns_config import default_config as ALNSConfig

# =========================
# Internal project modules
# =========================
from .vehicle import Vehicle  # Top-level reference; local imports avoid cycles.


def improved_initial_solution(state, rng):
    """
    Construct an initial feasible solution through a timed multi-stage process.

    Stages:
        1) Initialize the resource pool.
        2) Prioritize demands.
        3) Allocate sequentially.
        4) Repair negative inventory, empty vehicles, and small vehicles.
        5) Recompute final inventory.

    Parameters
    ----------
    state : SolutionState
        Initial state created by the caller.
    rng : numpy.random.Generator
        Random number generator retained for interface compatibility.

    Returns
    -------
    SolutionState
        Feasible initial solution.
    """
    data = state.data

    # Dynamic timeout based on demand size; configurable through ALNSConfig.
    min_timeout = getattr(ALNSConfig, "MIN_INIT_TIMEOUT", 30)
    max_timeout = getattr(ALNSConfig, "MAX_INIT_TIMEOUT", 120)
    per_demand = getattr(ALNSConfig, "INIT_TIMEOUT_PER_DEMAND", 0.05)
    timeout = min(max_timeout, max(min_timeout, len(data.demands) * per_demand))
    t_start = time.time()

    # 1. Resource pool.
    print("Initializing resource pool...")
    t0 = time.time()
    resource_pool = _initialize_resource_pool(data)
    print(f" Resource pool initialized in {time.time() - t0:.2f} seconds.")

    # 2. Demand priority.
    print("Prioritizing demands...")
    t0 = time.time()
    prioritized_demands = _prioritize_demands(data)
    print(f" Demands prioritized in {time.time() - t0:.2f} seconds.")

    # 3. Sequential allocation.
    print("Using sequential allocation...")
    t0 = time.time()
    stats = {}
    sol = _sequential_allocation(
        state, prioritized_demands, resource_pool, rng, timeout
    )
    stats["allocation_time"] = time.time() - t0
    print(
        f" Sequential allocation completed in {stats['allocation_time']:.2f} seconds."
    )

    # 4. Repair stage.
    print("Repairing solution if necessary...")
    t0 = time.time()
    # Track the repair-stage start time.
    repair_start = time.time()
    _repair_solution(sol, timeout - (time.time() - t_start), stats=stats)
    repair_time = time.time() - repair_start
    print(f" Repair completed in {time.time() - t0:.2f} seconds.")

    # 5. Final inventory computation.
    sol.compute_inventory()

    # Strict-feasibility finalization: apply a few deterministic repairs.
    _finalize_strict_initial_feasibility(sol, rng, timeout - (time.time() - t_start))

    # Aggregate stage statistics without changing the original logic.
    # Resource-pool and priority timings are intentionally left as None because
    # the original implementation only printed them.
    stats.setdefault("repair_iterations", 0)
    stats["parallel_used"] = False
    if "allocation_time" not in stats:
        stats["allocation_time"] = None
    stats["resource_pool_time"] = None  # The original implementation only printed timing.
    stats["prioritize_time"] = None
    stats["repair_time"] = repair_time

    # Attach the statistics to the solution for external analysis.
    sol.initial_build_stats = stats
    return sol


def _initialize_resource_pool(data):
    """
    Build cumulative available inventory for each ``(plant, sku, day)``.

    Rules:
        day=0 uses initial inventory.
        day>=1 adds same-day production to the previous period total.
    """
    resource_pool = {}

    # Initial inventory.
    for (plant, sku), inv in data.sku_initial_inv.items():
        resource_pool[(plant, sku, 0)] = inv

    # Accumulate by day.
    for day in range(1, data.horizons + 1):
        for plant in data.plants:
            for sku in data.skus_plant.get(plant, []):
                prev_inv = resource_pool.get((plant, sku, day - 1), 0)
                production = data.sku_prod_each_day.get((plant, sku, day), 0)
                resource_pool[(plant, sku, day)] = prev_inv + production

    return resource_pool


def _prioritize_demands(data):
    """
    Compute demand priorities.

    Metrics:
        scarcity_score      = 1 / supply-demand ratio
        supply_chain_score  = 1 / number of available plants
        volume_score        = normalized demand size
        priority = 0.5*scarcity + 0.3*supply_chain + 0.2*volume

    Returns
    -------
    list
        ``[((dealer, sku), demand, priority_score), ...]`` in descending order.
    """
    # Total SKU supply from production and initial inventory.
    sku_total_supply = defaultdict(int)
    for (_, sku, _), qty in data.sku_prod_each_day.items():
        sku_total_supply[sku] += qty
    for (_, sku), inv in data.sku_initial_inv.items():
        sku_total_supply[sku] += inv

    # Total SKU demand.
    sku_total_demand = defaultdict(int)
    for (_, sku), demand in data.demands.items():
        sku_total_demand[sku] += demand

    # Supply-demand ratio.
    sku_supply_demand_ratio = {}
    for sku in data.all_skus:
        supply = sku_total_supply.get(sku, 0)
        demand = sku_total_demand.get(sku, 0)
        sku_supply_demand_ratio[sku] = supply / demand if demand > 0 else float("inf")

    demand_priorities = []
    supply_chain = data.construct_supply_chain()
    max_demand = max(data.demands.values()) if data.demands else 1

    for (dealer, sku), demand in data.demands.items():
        if demand <= 0:
            continue

        available_plants = [
            plant
            for (plant, dealer_id), skus in supply_chain.items()
            if dealer_id == dealer and sku in skus
        ]
        supply_chain_score = 1.0 / (
            len(available_plants) if available_plants else float("inf")
        )
        scarcity_score = 1.0 / sku_supply_demand_ratio.get(sku, float("inf"))
        volume_score = demand / max_demand
        priority_score = (
            0.5 * scarcity_score + 0.3 * supply_chain_score + 0.2 * volume_score
        )
        demand_priorities.append(((dealer, sku), demand, priority_score))

    demand_priorities.sort(key=lambda x: x[2], reverse=True)
    return demand_priorities


def _sequential_allocation(state, prioritized_demands, resource_pool, rng, timeout):
    """
    Allocate demands sequentially with dealer grouping and cumulative limits.

    Strategy:
        - Group demands by dealer and sort by average priority.
        - Main allocation respects both the resource pool and cumulative map.
        - Supplemental allocation uses only the resource pool to fill unmet demand.
        - Successful loads update both the resource pool and cumulative map.
    """
    data = state.data
    from .alnsopt import veh_loading

    if not bool(getattr(ALNSConfig, "INITIAL_USE_DEALER_GROUPING", True)):
        return _sequential_allocation_legacy(
            state,
            prioritized_demands,
            resource_pool,
            timeout,
            veh_loading,
        )

    sol = state.copy()
    t_start = time.time()

    # ``veh_loading`` rebuilds ``used_inv`` by scanning every vehicle unless it
    # is handed the vehicles for the (plant, day) it is loading. Here that scan
    # ran once per allocation over the whole fleet: 5,887 allocations x 2,958
    # vehicles on large, measured at 2.5 s of a 5.6 s stage. The index below is
    # the same one ``greedy_repair`` already keeps (A8b), and it is kept current
    # at the load site.
    plant_day_vehicles: dict[tuple[str, int], list] = {}
    for existing in sol.vehicles:
        plant_day_vehicles.setdefault((existing.fact_id, existing.day), []).append(
            existing
        )

    supply_chain = data.construct_supply_chain()
    demand_map = {
        (dealer, sku): int(demand)
        for (dealer, sku), demand, _ in prioritized_demands
        if demand > 0
    }
    dealer_groups = _group_demands_by_dealer(prioritized_demands)
    cumulative_map = _compute_cumulative_map(resource_pool, data)

    use_cumulative = bool(
        getattr(ALNSConfig, "INITIAL_ENABLE_CUMULATIVE_CONSTRAINT", True)
    )
    enable_relaxed = bool(
        getattr(ALNSConfig, "INITIAL_ENABLE_RELAXED_REALLOCATION", True)
    )

    # Main allocation: cumulative and physical resource constraints.
    for dealer, sku_demands, _avg_priority in dealer_groups:
        if time.time() - t_start > timeout:
            break
        _allocate_dealer_demands(
            sol,
            dealer,
            sku_demands,
            demand_map,
            resource_pool,
            cumulative_map,
            supply_chain,
            use_cumulative=use_cumulative,
            timeout=timeout,
            t_start=t_start,
            veh_loading=veh_loading,
            plant_day_vehicles=plant_day_vehicles,
        )

    # Supplemental allocation relaxes cumulative constraints to fill unmet demand.
    if enable_relaxed and time.time() - t_start <= timeout:
        unmet_groups = []
        for dealer, sku_demands, avg_priority in dealer_groups:
            unmet_items = []
            for sku, _demand, sku_priority in sku_demands:
                remain = int(demand_map.get((dealer, sku), 0))
                if remain > 0:
                    unmet_items.append((sku, remain, sku_priority))
            if unmet_items:
                unmet_groups.append((dealer, unmet_items, avg_priority))

        for dealer, sku_demands, _avg_priority in unmet_groups:
            if time.time() - t_start > timeout:
                break
            _allocate_dealer_demands(
                sol,
                dealer,
                sku_demands,
                demand_map,
                resource_pool,
                cumulative_map,
                supply_chain,
                use_cumulative=False,
                timeout=timeout,
                t_start=t_start,
                veh_loading=veh_loading,
                plant_day_vehicles=plant_day_vehicles,
            )

    return sol


def _sequential_allocation_legacy(
    state,
    prioritized_demands,
    resource_pool,
    timeout,
    veh_loading,
):
    """Legacy allocation path: demand-by-demand greedy traversal."""
    data = state.data
    sol = state.copy()
    t_start = time.time()

    allocated = defaultdict(int)
    supply_chain = data.construct_supply_chain()
    supply_chain_items = list(supply_chain.items())
    horizons_range = range(1, data.horizons + 1)
    resource_get = resource_pool.get

    for (dealer, sku), total_demand, _ in prioritized_demands:
        if time.time() - t_start > timeout:
            break

        remain_demand = int(total_demand) - int(allocated[(dealer, sku)])
        if remain_demand <= 0:
            continue

        available_plants = [
            plant
            for (plant, dealer_id), skus in supply_chain_items
            if dealer_id == dealer and sku in skus
        ]
        if not available_plants:
            continue

        plant_inventory = [
            (
                plant,
                sum(int(resource_get((plant, sku, d), 0)) for d in horizons_range),
            )
            for plant in available_plants
        ]
        plant_inventory.sort(key=lambda x: x[1], reverse=True)

        for plant, _ in plant_inventory:
            if remain_demand <= 0:
                break
            for day in horizons_range:
                if remain_demand <= 0:
                    break
                available = int(resource_get((plant, sku, day), 0))
                if available <= 0:
                    continue

                assign_qty = int(min(remain_demand, available))
                if assign_qty <= 0:
                    continue

                veh_type = _select_best_vehicle(assign_qty, sku, data)
                vehicle = Vehicle(plant, dealer, veh_type, day, data)

                shipped_before = int(sol.compute_shipped().get((dealer, sku), 0))
                success = veh_loading(sol, vehicle, {sku: assign_qty})
                if not success:
                    continue

                shipped_after = int(sol.compute_shipped().get((dealer, sku), 0))
                actual_loaded = max(0, shipped_after - shipped_before)
                if actual_loaded <= 0:
                    continue

                for d_upd in range(day, data.horizons + 1):
                    resource_pool[(plant, sku, d_upd)] = int(
                        resource_pool.get((plant, sku, d_upd), 0) - actual_loaded
                    )

                allocated[(dealer, sku)] += actual_loaded
                remain_demand -= actual_loaded

    return sol


def _group_demands_by_dealer(prioritized_demands):
    """Group demands by dealer and compute average priority per dealer."""
    grouped = defaultdict(list)
    for (dealer, sku), demand, priority in prioritized_demands:
        if demand <= 0:
            continue
        grouped[dealer].append((sku, int(demand), float(priority)))

    dealer_groups = []
    for dealer, items in grouped.items():
        if not items:
            continue
        avg_priority = sum(p for _, _, p in items) / float(len(items))
        items.sort(key=lambda x: x[2], reverse=True)
        dealer_groups.append((dealer, items, avg_priority))

    dealer_groups.sort(key=lambda x: x[2], reverse=True)
    return dealer_groups


def _compute_cumulative_map(resource_pool, data):
    """Compute cumulative availability map C_ikt = sum_{u=t..T} R_iku."""
    cumulative = {}
    for plant in data.plants:
        for sku in data.skus_plant.get(plant, []):
            running = 0
            for day in range(data.horizons, 0, -1):
                running += int(resource_pool.get((plant, sku, day), 0))
                cumulative[(plant, sku, day)] = running
    return cumulative


def _refresh_cumulative_for_plant_sku(cumulative_map, resource_pool, plant, sku, horizons):
    """Refresh cumulative suffix values for a single (plant, sku) pair."""
    running = 0
    for day in range(horizons, 0, -1):
        running += int(resource_pool.get((plant, sku, day), 0))
        cumulative_map[(plant, sku, day)] = running


def _select_vehicle_type_for_volume(required_volume, data):
    """Pick smallest feasible vehicle type by volume; fallback to largest type."""
    suitable = [
        (veh_type, cap)
        for veh_type, cap in data.veh_type_cap.items()
        if cap >= required_volume
    ]
    if suitable:
        return min(suitable, key=lambda x: x[1])[0]
    return max(data.veh_type_cap.items(), key=lambda x: x[1])[0]


def _fit_orders_to_capacity(order_pairs, capacity_volume, data):
    """Trim sku orders to fit vehicle capacity while preserving priority order."""
    if capacity_volume <= 0:
        return {}

    fitted = {}
    used_volume = 0.0
    for sku, qty in order_pairs:
        if qty <= 0:
            continue
        sku_size = float(data.sku_sizes.get(sku, 0))
        if sku_size <= 0:
            continue
        remain_volume = capacity_volume - used_volume
        if remain_volume <= 1e-9:
            break
        max_qty = int(remain_volume // sku_size)
        take = int(min(int(qty), max_qty))
        if take <= 0:
            continue
        fitted[sku] = take
        used_volume += take * sku_size
    return fitted


def _allocate_dealer_demands(
    sol,
    dealer,
    sku_demands,
    demand_map,
    resource_pool,
    cumulative_map,
    supply_chain,
    use_cumulative,
    timeout,
    t_start,
    veh_loading,
    plant_day_vehicles=None,
):
    """Allocate one dealer's demands in sku-priority order with optional cumulative checks."""
    data = sol.data
    horizons_range = range(1, data.horizons + 1)
    supply_chain_items = list(supply_chain.items())

    for sku, _demand, _priority in sku_demands:
        if time.time() - t_start > timeout:
            break

        remain = int(demand_map.get((dealer, sku), 0))
        if remain <= 0:
            continue

        available_plants = [
            plant
            for (plant, dealer_id), skus in supply_chain_items
            if dealer_id == dealer and sku in skus
        ]
        if not available_plants:
            continue

        plant_inventory = [
            (
                plant,
                sum(int(resource_pool.get((plant, sku, d), 0)) for d in horizons_range),
            )
            for plant in available_plants
        ]
        plant_inventory.sort(key=lambda x: x[1], reverse=True)

        for plant, _ in plant_inventory:
            if remain <= 0 or time.time() - t_start > timeout:
                break

            for day in horizons_range:
                if remain <= 0 or time.time() - t_start > timeout:
                    break

                avail_day = int(resource_pool.get((plant, sku, day), 0))
                if avail_day <= 0:
                    continue

                if use_cumulative:
                    avail_cum = int(cumulative_map.get((plant, sku, day), 0))
                    avail = min(avail_day, avail_cum)
                else:
                    avail = avail_day
                if avail <= 0:
                    continue

                assign_qty = int(min(remain, avail))
                if assign_qty <= 0:
                    continue

                veh_type = _select_best_vehicle(assign_qty, sku, data)
                vehicle = Vehicle(plant, dealer, veh_type, day, data)

                # ``veh_loading`` invalidates the shipment cache, so bracketing
                # it with two ``compute_shipped()`` calls cost two full fleet
                # rebuilds per allocation - measured on large: 11,777 calls,
                # 10.26 s of a 15 s initial solution. The loaded amount is
                # already visible on the vehicles the call touched, and
                # ``_init_alloc_equivalence.py`` confirmed the two definitions
                # agree on all 5,887 allocations (the finished solution matches
                # on all 3,606 (dealer, sku) pairs as well).
                vehicles_before = len(sol.vehicles)
                day_index = (
                    plant_day_vehicles.setdefault((plant, day), [])
                    if plant_day_vehicles is not None
                    else None
                )
                success = veh_loading(
                    sol,
                    vehicle,
                    {sku: assign_qty},
                    plant_day_vehicles=day_index,
                )
                # Keep the index current before any early exit: the loader may
                # have appended vehicles (and split a load) before returning.
                if plant_day_vehicles is not None:
                    for added in sol.vehicles[vehicles_before:]:
                        day_index.append(added)
                if not success:
                    continue

                # Sum over everything this call appended: a load normally fits
                # one vehicle, but summing keeps the count exact if a split
                # ever happens.
                appended = sol.vehicles[vehicles_before:]
                if appended:
                    actual_loaded = sum(
                        int(veh.cargo.get((sku, day), 0)) for veh in appended
                    )
                else:
                    actual_loaded = int(vehicle.cargo.get((sku, day), 0))
                if actual_loaded <= 0:
                    continue

                demand_map[(dealer, sku)] = max(
                    0,
                    int(demand_map.get((dealer, sku), 0)) - actual_loaded,
                )
                remain -= actual_loaded

                for d_upd in range(day, data.horizons + 1):
                    pool_key = (plant, sku, d_upd)
                    resource_pool[pool_key] = int(
                        resource_pool.get(pool_key, 0) - actual_loaded
                    )

                _refresh_cumulative_for_plant_sku(
                    cumulative_map,
                    resource_pool,
                    plant,
                    sku,
                    data.horizons,
                )


def _select_best_vehicle(qty, sku, data):
    """
    Select a vehicle type for the quantity of a SKU.

    Compute required volume as ``qty * sku_size``. Choose the smallest vehicle
    that fits. If no vehicle type fits, choose the largest available type so the
    remaining quantity can be handled by additional vehicles.
    """
    required_volume = qty * data.sku_sizes[sku]
    suitable_types = [
        (vt, cap) for vt, cap in data.veh_type_cap.items() if cap >= required_volume
    ]
    if suitable_types:
        return min(suitable_types, key=lambda x: x[1])[0]
    return max(data.veh_type_cap.items(), key=lambda x: x[1])[0]


def _repair_solution(sol, timeout, stats=None):
    """
    Repair the initial solution.

    Steps:
        1. Adjust negative inventory across multiple iterations by reducing loads.
        2. Remove empty vehicles.
        3. Merge small vehicles to reduce solution size.

    Notes:
        - Inventory is recomputed from current cargo before each correction pass.
        - The repair loop exits early on timeout or when no correction is possible.
    """
    t_start = time.time()
    max_iter = getattr(ALNSConfig, "INITIAL_REPAIR_MAX_ITER", 5)
    iter_count = 0

    for _ in range(max_iter):
        iter_count += 1
        if time.time() - t_start > timeout:
            break

        # Recompute from current cargo each pass to avoid repairing stale state.
        sol.compute_inventory()
        neg_inv = [(k, v) for k, v in sol.s_ikt.items() if v < 0]
        if not neg_inv:
            break

        changed = False
        # Repair earlier periods first to avoid amplifying later-period errors.
        neg_inv.sort(key=lambda x: x[0][2])
        for (plant, sku, day), _neg_val in neg_inv:
            # Recompute for each negative point to avoid over-reduction in one pass.
            sol.compute_inventory()
            current_short = sol.s_ikt.get((plant, sku, day), 0)
            if current_short >= 0:
                continue

            relevant_vehicles = [
                veh
                for veh in sol.vehicles
                if veh.fact_id == plant
                and veh.day <= day
                and any(
                    c_sku == sku and c_day <= day for (c_sku, c_day) in veh.cargo.keys()
                )
            ]
            if not relevant_vehicles:
                continue

            to_reduce = -current_short
            relevant_vehicles.sort(
                key=lambda veh: (
                    veh.day,
                    sum(q for (s, d), q in veh.cargo.items() if s == sku and d <= day),
                ),
                reverse=True,
            )

            for veh in relevant_vehicles:
                if to_reduce <= 0:
                    break

                candidate_keys = [
                    (s, d)
                    for (s, d) in veh.cargo.keys()
                    if s == sku and d <= day and veh.cargo.get((s, d), 0) > 0
                ]
                candidate_keys.sort(key=lambda x: x[1], reverse=True)

                current_qty = sum(veh.cargo.get(k, 0) for k in candidate_keys)
                reduce_qty = min(current_qty, to_reduce)
                if reduce_qty <= 0:
                    continue

                left = reduce_qty
                for cargo_key in candidate_keys:
                    if left <= 0:
                        break
                    q0 = veh.cargo.get(cargo_key, 0)
                    if q0 <= 0:
                        continue
                    dq = min(q0, left)
                    veh.cargo[cargo_key] = q0 - dq
                    if veh.cargo[cargo_key] == 0:
                        del veh.cargo[cargo_key]
                    left -= dq

                if left < reduce_qty:
                    changed = True
                    to_reduce -= reduce_qty - left

        if not changed:
            break

    sol.vehicles = [veh for veh in sol.vehicles if not veh.is_empty()]
    if time.time() - t_start < timeout:
        _merge_small_vehicles(sol, timeout - (time.time() - t_start))

    sol.compute_inventory()
    if stats is not None:
        stats["repair_iterations"] = (
            iter_count
            if "repair_iterations" not in stats
            else stats["repair_iterations"] + iter_count
        )
    return sol


def _merge_small_vehicles(sol, timeout):
    """
    Merge small vehicles with the same ``(plant, dealer, day)`` key.

    Sort vehicles by current loaded volume, merge the smallest vehicle into the
    first target with sufficient capacity, and retain remaining vehicles when the
    timeout is reached.
    """
    data = sol.data
    t_start = time.time()
    vehicle_groups = defaultdict(list)
    for veh in sol.vehicles:
        key = (veh.fact_id, veh.dealer_id, veh.day)
        vehicle_groups[key].append(veh)

    merged_vehicles = []

    for key, vehicles in vehicle_groups.items():
        if time.time() - t_start > timeout:
            merged_vehicles.extend(vehicles)
            continue
        if len(vehicles) <= 1:
            merged_vehicles.extend(vehicles)
            continue

        def veh_volume(v):
            return sum(qty * data.sku_sizes[sku] for (sku, _), qty in v.cargo.items())

        vehicles.sort(key=veh_volume)

        while len(vehicles) > 1:
            if time.time() - t_start > timeout:
                break
            smallest = vehicles[0]
            smallest_volume = veh_volume(smallest)
            merged = False
            for i in range(1, len(vehicles)):
                target = vehicles[i]
                target_volume = veh_volume(target)
                total_volume = smallest_volume + target_volume
                if total_volume <= data.veh_type_cap.get(target.type, 0):
                    for (sku, day), qty in smallest.cargo.items():
                        target.cargo[(sku, day)] = target.cargo.get((sku, day), 0) + qty
                    vehicles.pop(0)
                    merged = True
                    break
            if not merged:
                merged_vehicles.append(vehicles.pop(0))

        merged_vehicles.extend(vehicles)

    sol.vehicles = merged_vehicles
    sol.compute_inventory()
    return sol


def _finalize_strict_initial_feasibility(sol, rng, remaining_timeout):
    """
    Apply final strict-feasibility repairs after initial construction.

    Steps:
      1) Return immediately when validation succeeds.
      2) Repair unmet demand with ``inventory_balance_repair`` first.
      3) Repair hard violations such as negative inventory or overloads with
         ``_repair_solution`` and ``infeasible_repair``.

    This stage runs only in strict mode and is bounded by timeout and pass count.
    """
    if not bool(getattr(ALNSConfig, "STRICT_FEASIBILITY_MODE", True)):
        return

    max_passes = int(getattr(ALNSConfig, "INITIAL_STRICT_FINALIZE_MAX_PASSES", 3))
    if max_passes <= 0:
        return

    t0 = time.time()
    for _ in range(max_passes):
        if remaining_timeout is not None and remaining_timeout > 0:
            if time.time() - t0 > remaining_timeout:
                break

        sol.compute_inventory()
        feasible, violations = sol.validate()
        if feasible:
            return

        # Repair unmet demand first because it is part of strict feasibility.
        if violations.get("unmet_demand"):
            try:
                from .repair_operators import inventory_balance_repair

                inventory_balance_repair(sol, rng)
            except Exception:
                pass

        # Route negative inventory, overload, and plant-capacity violations
        # through inventory and infeasibility repair. An incomplete check counts
        # as hard too: the state could not be verified.
        hard_bad = bool(
            violations.get("negative_inventory")
            or violations.get("veh_over_load")
            or violations.get("plant_inv_exceed")
            or violations.get("check_error")
        )
        if hard_bad:
            try:
                _repair_solution(sol, max(5.0, (remaining_timeout or 5.0) * 0.3))
            except Exception:
                pass
            try:
                from .repair_operators import infeasible_repair

                repaired = infeasible_repair(sol, rng)
                if repaired is not None:
                    sol.replace_vehicles(list(repaired.vehicles))
            except Exception:
                pass

    # Force one final recomputation to align inventory with vehicles.
    sol.compute_inventory()
