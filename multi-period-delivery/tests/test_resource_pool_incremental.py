"""The incremental resource-pool adjustment must equal a full rebuild.

``regret_based_repair`` used to rebuild the whole ``ResourcePool`` at the top of
every while iteration (~14 ms on the medium instance). It now applies the exact
delta of the shipment it just committed:

    a shipment of ``q`` on (plant, sku, day)
        lowers availability by q for that day (through `used`) and
        for every later day (through ``s_ikt``).

Clipping is idempotent under subtraction, so ``max(0, old - q)`` matches the
rebuild. A full-run probe checks this on every cache hit; this pins the algebra
on one concrete commit.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from multi_period_delivery.alns.alns_optimizer import ALNSOptimizer
from multi_period_delivery.alns.repair_utils import build_resource_pool, veh_loading
from multi_period_delivery.alns.vehicle import Vehicle

BASE = Path(__file__).resolve().parents[1]


def _state():
    optimizer = ALNSOptimizer(
        input_file_loc=str(BASE / "datasets" / "multiple-periods" / "small"),
        output_file_loc=str(Path(tempfile.gettempdir()) / "_resource_pool_probe"),
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

    for d in range(day, state.data.horizons + 1):
        key = (plant, sku, d)
        pool[key] = max(0, pool.get(key, 0) - shipped)

    fresh = build_resource_pool(state, state.data)
    assert _flatten(pool) == _flatten(fresh)
