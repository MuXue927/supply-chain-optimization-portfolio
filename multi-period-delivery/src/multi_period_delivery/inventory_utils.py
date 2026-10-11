"""
inventory_utils.py

Responsibilities:
- Provide reusable aggregation helpers for destroy, repair, and selection
  operators.
- Precompute common O(V*S) aggregation work in O(V + D*S) or O(P*S) form
  where practical, reducing repeated scans over the vehicle list.

Design notes:
- Read state.s_ikt and state.vehicles as read-only snapshots. Callers are
  responsible for recomputing or rebuilding caches after mutating state.
- Return simple, serializable aggregate structures (dict / defaultdict) so the
  results are easy to log, assert on, and test.
- Document complexity, preconditions, and invariants so new operators can use
  these helpers correctly.

Important:
- This module never mutates state.s_ikt or state.vehicles. For batch inventory
  updates, use alnsopt.batch_update_inventory, then explicitly call
  compute_inventory / mark_objective_dirty if a consistent state is required.
"""
from collections import defaultdict


def precompute_plant_day_inventory(state):
    """
    Aggregate state.s_ikt into a (plant, day) inventory total mapping.

    Returns:
      defaultdict(int): keys are (plant, day), and values are total plant
      inventory for that day.

    Usage notes:
      - This is a read-only aggregation over the current s_ikt snapshot.
      - If state.s_ikt is modified afterwards (for example through veh_loading
        or batch_update_inventory), call this function again before relying on
        the aggregate.
      - Typical uses include validating daily plant inventory limits,
        measuring excess inventory, and supporting vehicle-splitting or
        merging heuristics.

    Complexity:
      - O(N), where N is the number of entries in state.s_ikt (approximately
        P * S * H).
    """
    plant_day_inventory = defaultdict(int)
    for (p, sku, d), inv in state.s_ikt.items():
        # Accumulate each (plant, sku, day) inventory entry into (plant, day).
        plant_day_inventory[(p, d)] += inv
    return plant_day_inventory


def precompute_dealer_shipments(state):
    """
    Precompute dealer shipment and demand aggregates.

    Returns:
      - shipments_by_dealer_sku_day: dict((dealer, sku, day) -> qty)
          * quantity shipped for each dealer, SKU, and departure day
      - shipments_by_dealer_total: dict(dealer -> total_shipped_qty)
          * total shipped quantity for each dealer in the current solution
      - dealer_total_demand: dict(dealer -> total_demand)
          * total input-data demand for each dealer, used to compare shipped
            quantity with demand

    Motivation:
      - Removal operators such as worst_removal and surplus_removal frequently
        aggregate vehicle shipments and demand by dealer. Precomputing these
        values turns repeated O(V*S) scans into one O(V + D*S) pass, reducing
        constant-factor runtime overhead.

    Usage notes:
      - Returned structures use defaultdict(int), which supports direct +=
        updates and missing keys with a default value of zero.
      - This function does not mutate state. Recompute it if the vehicle set
        changes after aggregation.

    Complexity:
      - O(V * L), where V is the number of vehicles and L is the average number
        of cargo entries per vehicle; dealer demand aggregation adds O(D*S).
    """
    shipments_by_dealer_sku_day = defaultdict(int)
    shipments_by_dealer_total = defaultdict(int)
    for veh in state.vehicles:
        dealer = veh.dealer_id
        for (sku, day), qty in veh.cargo.items():
            # Accumulate each vehicle cargo entry by (dealer, sku, day).
            shipments_by_dealer_sku_day[(dealer, sku, day)] += qty
            shipments_by_dealer_total[dealer] += qty

    dealer_total_demand = defaultdict(int)
    for (dealer, sku_id), dem in state.data.demands.items():
        # Aggregate input-data demand by dealer.
        dealer_total_demand[dealer] += dem

    return shipments_by_dealer_sku_day, shipments_by_dealer_total, dealer_total_demand
