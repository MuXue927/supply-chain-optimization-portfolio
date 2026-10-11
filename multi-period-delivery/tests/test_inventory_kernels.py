"""Focused tests for the multi-period inventory kernels and helper pools."""

from __future__ import annotations

import numpy as np
from multi_period_delivery.alns import inventory_numba
from multi_period_delivery.alns.alnsopt import (
    SolutionState,
    batch_update_inventory,
    veh_loading,
)
from multi_period_delivery.alns.InputDataALNS import DataALNS
from multi_period_delivery.alns.inventory_manager import UnifiedInventoryManager
from multi_period_delivery.alns.inventory_utils import (
    precompute_dealer_shipments,
    precompute_plant_day_inventory,
)
from multi_period_delivery.alns.repair_utils import (
    build_resource_pool as build_repair_resource_pool,
)
from multi_period_delivery.alns.vehicle import Vehicle


def _make_synthetic_data() -> DataALNS:
    """Build a two-plant, two-period data container without file I/O."""
    data = DataALNS("", "", "synthetic")
    data.plants = {"P1", "P2"}
    data.skus_plant = {"P1": {"A", "B"}, "P2": {"A"}}
    data.skus_dealer = {"D1": {"A"}}
    data.demands = {("D1", "A"): 12}
    data.horizons = 2
    data.historical_s_ikt = {
        ("P1", "A", 0): 10,
        ("P1", "B", 0): 7,
        ("P2", "A", 0): 5,
    }
    data.sku_prod_each_day = {
        ("P1", "A", 1): 3,
        ("P1", "A", 2): 2,
        ("P1", "B", 1): 1,
        ("P2", "A", 1): 4,
        ("P2", "A", 2): 1,
    }
    data.sku_sizes = {"A": 1.0, "B": 1.0}
    data.veh_type_cap = {"V": 100}
    return data


def _make_synthetic_state() -> SolutionState:
    """Create a state with one same-day shipment and one cross-day cargo item."""
    data = _make_synthetic_data()
    state = SolutionState(data)

    same_day = Vehicle("P1", "D1", "V", 1, data)
    same_day.cargo[("A", 1)] = 4

    later_day = Vehicle("P2", "D1", "V", 2, data)
    later_day.cargo[("A", 2)] = 1

    # This cargo must be ignored because it is keyed to day 2 while the
    # vehicle departs on day 1.
    cross_day = Vehicle("P1", "D1", "V", 1, data)
    cross_day.cargo[("A", 2)] = 9

    state.vehicles = [same_day, later_day, cross_day]
    return state


def test_inventory_kernels_match_reference_and_preserve_int_keys(
    monkeypatch,
) -> None:
    expected = {
        ("P1", "A", 0): 10,
        ("P1", "A", 1): 9,
        ("P1", "A", 2): 11,
        ("P1", "B", 0): 7,
        ("P2", "A", 0): 5,
        ("P2", "A", 1): 9,
        ("P2", "A", 2): 9,
    }

    numba_state = _make_synthetic_state()
    assert inventory_numba.compute_inventory_numba_safe(numba_state)
    numba_result = dict(numba_state.s_ikt)

    numpy_state = _make_synthetic_state()
    monkeypatch.setattr(inventory_numba, "NUMBA_AVAILABLE", False)
    assert inventory_numba.compute_inventory_numba_safe(numpy_state)
    numpy_result = dict(numpy_state.s_ikt)

    reference_state = _make_synthetic_state()
    monkeypatch.setattr(inventory_numba, "compute_inventory_fast", lambda _state: False)
    assert reference_state.compute_inventory()
    reference_result = dict(reference_state.s_ikt)

    assert numba_result == expected
    assert numpy_result == expected
    assert reference_result == expected
    assert set(numba_result) == set(expected)
    assert all(type(value) is int for value in numba_result.values())
    assert all(type(value) is int for value in numpy_result.values())


def test_inventory_cache_reuses_unchanged_shipments_and_detects_cargo_change(
    monkeypatch,
) -> None:
    monkeypatch.setattr(inventory_numba, "NUMBA_AVAILABLE", False)
    state = _make_synthetic_state()

    assert inventory_numba.compute_inventory_numba_safe(state)
    cache = state._inventory_cache
    shipments = cache["shipments_arr"]
    assert isinstance(shipments, np.ndarray)

    assert inventory_numba.compute_inventory_numba_safe(state)
    assert state._inventory_cache is cache
    assert state._inventory_cache["shipments_arr"] is shipments

    state.vehicles[0].cargo[("A", 1)] = 6
    assert inventory_numba.compute_inventory_numba_safe(state)

    assert state._inventory_cache is cache
    assert state._inventory_cache["shipments_arr"] is not shipments
    assert state.s_ikt[("P1", "A", 1)] == 7
    assert state.s_ikt[("P1", "A", 2)] == 9


def test_inventory_cache_is_invalidated_by_loading_and_batch_updates() -> None:
    state = _make_synthetic_state()
    state.vehicles = []
    assert state.compute_inventory()
    assert state._inventory_cache is not None

    loading_vehicle = Vehicle("P1", "D1", "V", 1, state.data)
    assert veh_loading(state, loading_vehicle, {"A": 2})
    assert state._inventory_cache is None

    assert state.compute_inventory()
    assert state._inventory_cache is not None
    batch_vehicle = Vehicle("P2", "D1", "V", 2, state.data)
    batch_vehicle.cargo[("A", 2)] = 1
    batch_update_inventory(state, [batch_vehicle], "repair")
    assert state._inventory_cache is None


def test_inventory_helpers_match_state_semantics() -> None:
    state = _make_synthetic_state()
    assert state.compute_inventory()

    manager = UnifiedInventoryManager()
    manager_pool = manager.build_resource_pool(state)
    repair_pool = dict(build_repair_resource_pool(state, state.data).items())
    assert manager_pool == repair_pool
    assert all(day >= 1 for (_plant, _sku, day) in manager_pool)

    plant_day_inventory = precompute_plant_day_inventory(state)
    assert plant_day_inventory[("P1", 0)] == 17
    assert plant_day_inventory[("P1", 1)] == 9
    assert plant_day_inventory[("P1", 2)] == 11

    shipments_by_dealer_sku_day, shipments_by_dealer, demand_by_dealer = (
        precompute_dealer_shipments(state)
    )
    assert shipments_by_dealer_sku_day[("D1", "A", 1)] == 4
    assert shipments_by_dealer_sku_day[("D1", "A", 2)] == 1
    assert shipments_by_dealer["D1"] == 5
    assert demand_by_dealer["D1"] == 12
