"""``compute_shipped`` must cache on vehicle changes, not on ``objective_dirty``.

The two are different questions. ``compute_shipped`` depends only on the vehicles'
cargo, so it is exactly as current as the vehicles are. ``objective_dirty`` says
"the objective cache is stale", and candidate evaluation calls
``mark_objective_dirty`` far more often than it actually changes a load.

Using ``objective_dirty`` here meant the cache missed almost every time. Measured
on the medium instance *before* the fix: 8,458 calls, 40% hit. Measured cost per
miss: **916 us** against **1.2 us** for a hit - an 800x difference - so those
misses were ~4.6 s of an 86 s run, and profiling attributed ~9.6 s of a 120 s run
to this function.

The cache now validates with the same O(1) mutation stamp as the inventory cache:
cargo changes and declared mutations both move it, and the vehicle list's length
and last id catch structural changes.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from multi_period_delivery.alns.alns_optimizer import ALNSOptimizer

BASE = Path(__file__).resolve().parents[1]


def _state():
    optimizer = ALNSOptimizer(
        input_file_loc=str(BASE / "datasets" / "multiple-periods" / "small"),
        output_file_loc=str(Path(tempfile.gettempdir()) / "_shipped_cache_probe"),
    )
    optimizer.load_data("dataset_1")
    return optimizer.create_initial_solution()


# ---------------------------------------------------------------------------
# The cache hits when nothing that matters changed
# ---------------------------------------------------------------------------


def test_repeated_calls_reuse_the_cached_result() -> None:
    state = _state()

    first = state.compute_shipped()
    second = state.compute_shipped()

    assert first is second, "an unchanged state must reuse the cached dict"


def test_a_declared_mutation_does_invalidate_shipped() -> None:
    """``mark_objective_dirty`` also marks structural changes.

    It is called right after ``state.vehicles[i] = ...``, which the cargo counter
    cannot see, so treating it as "the vehicles may have changed" is correct - and
    it is exactly what the old ``objective_dirty`` test also concluded. The gain
    from the new stamp is elsewhere: it *also* stays valid across the many
    ``mark_objective_dirty`` calls that are followed by the objective being
    recomputed, where ``objective_dirty`` was left true and forced a rebuild.
    """
    state = _state()
    first = state.compute_shipped()

    state.mark_objective_dirty()

    result = state.compute_shipped()
    assert result == first, "the values are unchanged"
    assert result is not first, "but a declared mutation forces a rebuild"


def test_shipped_values_are_correct() -> None:
    """Reuse must not be achieved by returning something wrong."""
    state = _state()

    shipped = state.compute_shipped()

    expected: dict = {}
    for vehicle in state.vehicles:
        for (sku_id, _day), qty in vehicle.cargo.items():
            key = (vehicle.dealer_id, sku_id)
            expected[key] = expected.get(key, 0) + qty
    assert shipped == expected


# ---------------------------------------------------------------------------
# ...and misses as soon as a load changes
# ---------------------------------------------------------------------------


def test_changing_cargo_invalidates_the_cache() -> None:
    state = _state()
    before = dict(state.compute_shipped())

    vehicle = next(v for v in state.vehicles if v.cargo)
    key = next(iter(vehicle.cargo))
    vehicle.cargo[key] = vehicle.cargo[key] + 7

    after = state.compute_shipped()

    # The changed load must be reflected.
    assert after != before
    expected_key = (vehicle.dealer_id, key[0])
    assert after[expected_key] == before.get(expected_key, 0) + 7


def test_a_structural_vehicle_change_invalidates_the_cache() -> None:
    """An added or removed vehicle moves the list shape, not just cargo."""
    state = _state()
    first = state.compute_shipped()

    state.vehicles.append(state.vehicles[0])

    assert state.compute_shipped() is not first


def test_clearing_shipped_still_forces_a_rebuild() -> None:
    """``mark_objective_dirty(clear_shipped=True)`` must keep working."""
    state = _state()
    first = state.compute_shipped()

    state.mark_objective_dirty(clear_shipped=True)

    result = state.compute_shipped()
    assert result == first, "the values are unchanged"
    assert result is not first, "but the cache was dropped, so it was rebuilt"
