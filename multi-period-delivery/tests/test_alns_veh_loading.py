"""Regression tests for the shared multi-SKU vehicle-loading helper."""

from __future__ import annotations

from types import SimpleNamespace

from multi_period_delivery.alns.alnsopt import veh_loading
from multi_period_delivery.alns.vehicle import SiktDict, Vehicle


class _FakeData:
    veh_type_cap = {"T": 100}
    sku_sizes = {"A": 1.0, "B": 1.0, "C": 1.0}
    skus_plant = {"P": ["A", "B", "C"]}
    sku_prod_each_day = {
        ("P", "A", 1): 1000,
        ("P", "B", 1): 1000,
        ("P", "C", 1): 1000,
    }
    horizons = 2


def test_multi_sku_loading_appends_each_vehicle_once() -> None:
    """A vehicle kept across SKU iterations must not be appended repeatedly."""
    data = _FakeData()
    state = SimpleNamespace(
        data=data,
        vehicles=[],
        s_ikt=SiktDict(),
        s_initialized=True,
        mark_objective_dirty=lambda: None,
        _inventory_cache=None,
    )
    vehicle = Vehicle("P", "D", "T", 1, data)

    assert veh_loading(
        state,
        vehicle,
        {"A": 10, "B": 10, "C": 10},
    )

    assert len(state.vehicles) == 1
    assert len({id(item) for item in state.vehicles}) == 1
    assert state.vehicles[0].cargo == {
        ("A", 1): 10,
        ("B", 1): 10,
        ("C", 1): 10,
    }
    assert state.s_ikt[("P", "A", 1)] == -10
    assert state.s_ikt[("P", "B", 1)] == -10
    assert state.s_ikt[("P", "C", 1)] == -10
