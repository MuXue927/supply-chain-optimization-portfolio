"""``validate()`` must not force a full inventory recompute needlessly.

``VALIDATE_FORCE_RECOMPUTE_INVENTORY`` (and ``STRICT_FEASIBILITY_MODE``) default
to true, so ``validate()`` used to call ``compute_inventory()`` on *every* call.
On the medium instance that cost ~10-25 ms per call and ran thousands of times
per iteration, which is what held a 150 s run to 23 iterations.

The fast inventory path already records the vehicle signature its cached
shipments were built from and rebuilds whenever that signature changes, so when
the recorded signature still matches, recomputing cannot change ``s_ikt``.
``SolutionState._inventory_is_fresh`` exposes exactly that fact.

These tests pin the two things that make reusing the cached inventory safe:

* freshness is **provable** only when the recorded signature matches, and
* it becomes **false** as soon as anything feeding shipments changes.
"""

from __future__ import annotations

from pathlib import Path

from multi_period_delivery.alns.alns_config import default_config as ALNSConfig
from multi_period_delivery.alns.alns_optimizer import ALNSOptimizer
from multi_period_delivery.alns.alnsopt import SolutionState
from multi_period_delivery.alns.vehicle import inventory_stamp

BASE = Path(__file__).resolve().parents[1]
# Test scratch data lives under the repository's ignored ``tmp/`` directory.
_TEST_OUTPUTS = BASE.parent / "tmp" / "test-runs"


def _state():
    optimizer = ALNSOptimizer(
        input_file_loc=str(BASE / "datasets" / "multiple-periods" / "small"),
        output_file_loc=str(_TEST_OUTPUTS / "_inventory_freshness_probe"),
    )
    optimizer.load_data("dataset_1")
    return optimizer.create_initial_solution()


# ---------------------------------------------------------------------------
# Conservative by default
# ---------------------------------------------------------------------------


def test_state_without_any_cache_is_not_fresh() -> None:
    """Without a cache there is nothing to prove freshness from."""
    state = _state()
    state.__dict__.pop("_inventory_cache", None)

    assert state._inventory_is_fresh() is False


def test_cache_without_a_recorded_stamp_is_not_fresh() -> None:
    """A cache that does not record what it was built from cannot be trusted."""
    state = _state()
    state.compute_inventory()
    state._inventory_cache.pop("stamp", None)

    assert state._inventory_is_fresh() is False


def test_cache_without_shipments_is_not_fresh() -> None:
    state = _state()
    state.compute_inventory()
    state._inventory_cache.pop("shipments_arr", None)

    assert state._inventory_is_fresh() is False


# ---------------------------------------------------------------------------
# Fresh after a compute, stale after a change
# ---------------------------------------------------------------------------


def test_fresh_immediately_after_compute() -> None:
    state = _state()
    state.compute_inventory()

    assert state._inventory_is_fresh() is True


def test_not_fresh_after_a_vehicle_with_cargo_is_added() -> None:
    """The vehicle set is exactly what the cached shipments describe."""
    state = _state()
    state.compute_inventory()
    assert state._inventory_is_fresh() is True

    state.vehicles.append(state.vehicles[0])

    assert state._inventory_is_fresh() is False


def test_not_fresh_after_cargo_changes() -> None:
    """A changed load must invalidate the cached shipments."""
    state = _state()
    state.compute_inventory()
    assert state._inventory_is_fresh() is True

    vehicle = next(v for v in state.vehicles if v.cargo)
    key = next(iter(vehicle.cargo))
    vehicle.cargo[key] = vehicle.cargo[key] + 1

    assert state._inventory_is_fresh() is False


def test_setdefault_and_union_invalidate_freshness() -> None:
    """Every CargoDict mutation path must move the inventory stamp."""
    state = _state()
    state.compute_inventory()
    vehicle = next(v for v in state.vehicles if v.cargo)
    day = int(vehicle.day)

    vehicle.cargo.setdefault(("__freshness_audit__", day), 1)
    assert state._inventory_is_fresh() is False

    state.compute_inventory()
    assert state._inventory_is_fresh() is True
    vehicle.cargo |= {("__freshness_audit_2__", day): 1}
    assert state._inventory_is_fresh() is False


def test_pop_missing_key_does_not_invalidate_freshness() -> None:
    state = _state()
    state.compute_inventory()
    vehicle = next(v for v in state.vehicles if v.cargo)

    vehicle.cargo.pop(("__missing__", int(vehicle.day)), 5)

    assert state._inventory_is_fresh() is True


def test_fresh_again_after_a_cargo_change_is_recomputed() -> None:
    """The property must recover, or the fast path would never re-engage."""
    state = _state()
    state.compute_inventory()
    vehicle = next(v for v in state.vehicles if v.cargo)
    key = next(iter(vehicle.cargo))
    vehicle.cargo[key] = vehicle.cargo[key] + 1
    assert state._inventory_is_fresh() is False

    state.compute_inventory()

    assert state._inventory_is_fresh() is True


# ---------------------------------------------------------------------------
# validate() honours the property
# ---------------------------------------------------------------------------


def test_validate_does_not_recompute_when_fresh(monkeypatch) -> None:
    """A fresh inventory must not be rebuilt just because strict mode is on."""
    state = _state()
    state.compute_inventory()
    assert state._inventory_is_fresh() is True

    # Must patch the class: ``validate`` calls ``self.compute_inventory()``, which
    # resolves through the type, so an instance attribute would never be seen.
    calls = {"n": 0}
    original = SolutionState.compute_inventory

    def counting_compute_inventory(self):
        calls["n"] += 1
        return original(self)

    monkeypatch.setattr(SolutionState, "compute_inventory", counting_compute_inventory)

    state.validate()

    assert calls["n"] == 0, "a provably fresh inventory must not be recomputed"


def test_validate_recomputes_when_freshness_cannot_be_proven(monkeypatch) -> None:
    """When freshness is unprovable, validate() must still recompute."""
    state = _state()

    calls = {"n": 0}
    original = SolutionState.compute_inventory

    def counting_compute_inventory(self):
        calls["n"] += 1
        return original(self)

    monkeypatch.setattr(SolutionState, "compute_inventory", counting_compute_inventory)
    monkeypatch.setattr(SolutionState, "_inventory_is_fresh", lambda self: False)

    state.validate()

    assert calls["n"] == 1


def test_validate_still_reports_correctly_after_a_cargo_change() -> None:
    """Honouring a *fresh* cache must never hide a real change.

    A changed load makes freshness unprovable, so ``validate()`` recomputes and
    the reported feasibility reflects the new load.
    """
    state = _state()
    state.compute_inventory()
    before = dict(state.s_ikt)

    # Add a large load that must move inventory.
    vehicle = next(v for v in state.vehicles if v.cargo)
    key = next(iter(vehicle.cargo))
    vehicle.cargo[key] = vehicle.cargo[key] + 1000
    assert state._inventory_is_fresh() is False

    feasible, violations = state.validate()

    assert isinstance(feasible, bool)
    # The change was picked up: either the table moved or the load was reported.
    moved = dict(state.s_ikt) != before
    reported = bool(violations["veh_over_load"])
    assert moved or reported, "the modified load must be reflected somewhere"


def test_defaults_are_the_strict_ones() -> None:
    """These tests are only meaningful while the strict defaults are in force."""
    assert ALNSConfig.VALIDATE_FORCE_RECOMPUTE_INVENTORY is True
    assert ALNSConfig.STRICT_FEASIBILITY_MODE is True


# ---------------------------------------------------------------------------
# The O(1) stamp itself
# ---------------------------------------------------------------------------


def test_stamp_is_stable_when_nothing_changes() -> None:
    state = _state()

    first = inventory_stamp(state.vehicles)
    second = inventory_stamp(state.vehicles)

    assert first == second


def test_stamp_moves_when_cargo_changes() -> None:
    state = _state()
    before = inventory_stamp(state.vehicles)

    vehicle = next(v for v in state.vehicles if v.cargo)
    key = next(iter(vehicle.cargo))
    vehicle.cargo[key] = vehicle.cargo[key] + 1

    assert inventory_stamp(state.vehicles) != before


def test_assigning_the_same_cargo_value_does_not_move_the_stamp() -> None:
    """Re-writing an unchanged value must not invalidate the cache needlessly."""
    state = _state()
    vehicle = next(v for v in state.vehicles if v.cargo)
    key = next(iter(vehicle.cargo))
    before = inventory_stamp(state.vehicles)

    vehicle.cargo[key] = vehicle.cargo[key]

    assert inventory_stamp(state.vehicles) == before


def test_stamp_moves_when_a_declared_mutation_happens() -> None:
    state = _state()
    before = inventory_stamp(state.vehicles)

    state.mark_objective_dirty()

    assert inventory_stamp(state.vehicles) != before


def test_stamp_moves_when_the_vehicle_list_changes() -> None:
    """Structural list changes are invisible to the cargo counter."""
    state = _state()
    before = inventory_stamp(state.vehicles)

    state.vehicles.append(state.vehicles[0])

    assert inventory_stamp(state.vehicles) != before


def test_stamp_handles_a_non_sized_vehicle_container() -> None:
    """A container that cannot be measured must not raise; it must differ."""
    assert inventory_stamp(None) != inventory_stamp([])
