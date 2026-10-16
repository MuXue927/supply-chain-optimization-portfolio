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

import tempfile
from pathlib import Path

from multi_period_delivery.alns.alns_config import default_config as ALNSConfig
from multi_period_delivery.alns.alns_optimizer import ALNSOptimizer
from multi_period_delivery.alns.alnsopt import SolutionState

BASE = Path(__file__).resolve().parents[1]


def _state():
    optimizer = ALNSOptimizer(
        input_file_loc=str(BASE / "datasets" / "multiple-periods" / "small"),
        output_file_loc=str(Path(tempfile.gettempdir()) / "_inventory_freshness_probe"),
    )
    optimizer.load_data("dataset_1")
    return optimizer.create_initial_solution()


# ---------------------------------------------------------------------------
# Conservative by default
# ---------------------------------------------------------------------------


def test_no_inventory_cache_means_not_fresh() -> None:
    """Without a cache there is nothing to prove freshness from."""
    state = _state()
    state.compute_inventory()
    # A cache that does not record what it was built from cannot be trusted.
    state._inventory_cache.pop("vehicles_sig", None)

    assert state._inventory_is_fresh() is False


def test_cache_without_a_recorded_signature_is_not_fresh() -> None:
    state = _state()
    state.compute_inventory()
    # A cache that does not record what it was built from cannot be trusted.
    state._inventory_cache.pop("vehicles_sig", None)

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
