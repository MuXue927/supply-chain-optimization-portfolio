"""The incremental resource-pool adjustment must equal a full rebuild.

A shipment of ``q`` on ``(plant, sku, day)`` lowers every later closing balance
by ``q``. Because earlier days' suffix minima also include those balances, the
refresh rescans the affected (plant, SKU) backwards rather than subtracting
from only ``day`` and later.
"""

from __future__ import annotations

from pathlib import Path

from multi_period_delivery.alns.alns_optimizer import ALNSOptimizer
from multi_period_delivery.alns.repair_utils import (
    build_resource_pool,
    refresh_suffix_resource_pool,
    veh_loading,
)
from multi_period_delivery.alns.vehicle import Vehicle

BASE = Path(__file__).resolve().parents[1]
# Test scratch data lives under the repository's ignored ``tmp/`` directory.
_TEST_OUTPUTS = BASE.parent / "tmp" / "test-runs"


def _state():
    optimizer = ALNSOptimizer(
        input_file_loc=str(BASE / "datasets" / "multiple-periods" / "small"),
        output_file_loc=str(_TEST_OUTPUTS / "_resource_pool_probe"),
    )
    optimizer.load_data("dataset_1")
    return optimizer.create_initial_solution()


def _flatten(pool) -> dict:
    return {
        (plant, sku, day): value
        for plant, skus in pool._data.items()
        for sku, days in skus.items()
        for day, value in days.items()
    }


def test_resource_pool_uses_the_suffix_minimum_balance() -> None:
    """Day availability must be bounded by every later closing balance."""
    state = _state()
    state.compute_inventory()
    pool = build_resource_pool(state, state.data)

    checked = 0
    for plant in state.data.plants:
        for sku in state.data.skus_plant.get(plant, []):
            for day in range(1, state.data.horizons + 1):
                expected = max(
                    0.0,
                    min(
                        float(state.s_ikt.get((plant, sku, d), 0.0))
                        for d in range(day, state.data.horizons + 1)
                    ),
                )
                assert pool[(plant, sku, day)] == expected
                checked += 1
    assert checked > 0


def test_incremental_pool_matches_a_rebuild_after_one_commit() -> None:
    state = _state()
    state.compute_inventory()
    pool = build_resource_pool(state, state.data)

    plant, sku, day = next(k for k, v in _flatten(pool).items() if v > 0)
    dealer_id = next(iter(state.data.demands))[0]
    veh_type = next(iter(state.data.all_veh_types))

    before = state.s_ikt.get((plant, sku, day), 0)
    veh_loading(
        state,
        Vehicle(plant, dealer_id, veh_type, day, state.data),
        {sku: 1},
        commit=True,
    )
    shipped = before - state.s_ikt.get((plant, sku, day), 0)
    assert shipped > 0, "the commit must actually have shipped something"

    refresh_suffix_resource_pool(state, pool, plant, sku)

    fresh = build_resource_pool(state, state.data)
    assert _flatten(pool) == _flatten(fresh)
