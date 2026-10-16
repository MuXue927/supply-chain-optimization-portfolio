"""Tests for the extracted inventory-cache helpers.

``compute_inventory_numba_safe`` wraps its whole pipeline in one ``try`` that
falls back to the pure-Python implementation, so its body was ~166 lines of
preparation and computation mixed together. Two pieces of *preparation* are now
separate functions with their own contracts:

* :func:`_vehicle_signature` / :func:`_vehicles_signature` - detect whether the
  cached shipments array still describes the current load;
* :func:`_resolve_inventory_cache` - reuse the cached arrays when they belong to
  the current index maps and have the right shapes.

The signature used to be an inner function whose fallback path dropped the cargo
count and total quantity, and whose per-entry loop silently skipped an
unparseable quantity - either could make a genuinely different load look
unchanged, so the stale shipments array would be reused.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from multi_period_delivery.alns.inventory_numba import (
    _resolve_inventory_cache,
    _vehicle_signature,
    _vehicles_signature,
)


def _vehicle(*, cargo=None, day=1, veh_type="T1", fact_id="P1"):
    return SimpleNamespace(
        fact_id=fact_id,
        type=veh_type,
        day=day,
        cargo={} if cargo is None else cargo,
    )


# ---------------------------------------------------------------------------
# _vehicle_signature
# ---------------------------------------------------------------------------


def test_signature_is_deterministic() -> None:
    vehicle = _vehicle(cargo={("S1", 1): 3, ("S2", 2): 5})

    assert _vehicle_signature(vehicle) == _vehicle_signature(vehicle)


def test_signature_covers_every_part_of_the_load() -> None:
    base = _vehicle(cargo={("S1", 1): 3, ("S2", 2): 5})

    different_qty = _vehicle(cargo={("S1", 1): 4, ("S2", 2): 5})
    different_day = _vehicle(cargo={("S1", 1): 3, ("S2", 2): 5}, day=2)
    different_type = _vehicle(cargo={("S1", 1): 3, ("S2", 2): 5}, veh_type="T2")
    fewer_entries = _vehicle(cargo={("S1", 1): 3})

    for other in (different_qty, different_day, different_type, fewer_entries):
        assert _vehicle_signature(base) != _vehicle_signature(other)


def test_signature_invalidates_on_an_unparseable_quantity() -> None:
    """The old signature could miss this: it skipped the entry from the total."""
    numeric = _vehicle(cargo={("S1", 1): 3})
    opaque = _vehicle(cargo={("S1", 1): object()})

    assert _vehicle_signature(numeric) != _vehicle_signature(opaque)


def test_signature_survives_an_empty_load() -> None:
    signature = _vehicle_signature(_vehicle())

    assert len(signature) == 6
    assert signature[3] == 0  # cargo entry count
    assert signature[4] == 0.0  # total quantity


# ---------------------------------------------------------------------------
# _vehicles_signature
# ---------------------------------------------------------------------------


def test_vehicles_signature_covers_the_list() -> None:
    vehicles = [_vehicle(cargo={("S1", 1): 1}), _vehicle(cargo={("S2", 1): 2})]
    other = [_vehicle(cargo={("S1", 1): 1})]

    assert _vehicles_signature(vehicles) == _vehicles_signature(vehicles)
    assert _vehicles_signature(vehicles) != _vehicles_signature(other)


def test_a_vehicle_with_unreadable_cargo_still_gets_a_signature() -> None:
    """One bad vehicle must not disable the cache for the whole list."""

    class _Exploding:
        fact_id = "P1"
        type = "T1"
        day = 1

        @property
        def cargo(self):
            raise RuntimeError("cargo is not readable")

    signature = _vehicle_signature(_Exploding())

    assert len(signature) == 6
    assert signature[0] == "P1"
    assert signature[3] == 0  # counted as "no entries"


def test_vehicles_signature_returns_none_for_a_non_iterable_list() -> None:
    """None means "cannot tell" so the caller rebuilds instead of reusing."""
    assert _vehicles_signature(None) is None


# ---------------------------------------------------------------------------
# _resolve_inventory_cache
# ---------------------------------------------------------------------------


class _State:
    def __init__(self, vehicles=None, cache=None) -> None:
        self.data = SimpleNamespace()
        self.vehicles = vehicles or []
        if cache is not None:
            self._inventory_cache = cache


def _maps(n_plants=1, n_skus=1, horizons=2):
    return SimpleNamespace(n_plants=n_plants, n_skus=n_skus, horizons=horizons)


def test_cache_is_rebuilt_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    maps = _maps()
    state = _State()
    built = {"called": 0}

    def fake_state_to_arrays(_state, _maps, _mod=None):
        built["called"] += 1
        return np.zeros((1, 1, 3)), None, np.ones((1, 1, 3))

    monkeypatch.setattr(
        "multi_period_delivery.alns.inventory_numba.state_to_inventory_arrays",
        fake_state_to_arrays,
    )

    cache, prod_arr, baseline0, shipments_arr = _resolve_inventory_cache(
        state, maps, 1, 1, 2
    )

    assert built["called"] == 1
    assert prod_arr is not None and baseline0 is not None
    assert shipments_arr is None
    assert cache["maps"] is maps
    assert state._inventory_cache is cache


def test_cache_is_reused_when_the_maps_and_shapes_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    maps = _maps()
    cached = {
        "maps": maps,
        "prod_arr": np.ones((1, 1, 3)),
        "baseline0": np.zeros((1, 1)),
        "shipments_arr": np.full((1, 1, 3), 7.0),
    }
    state = _State(cache=cached)

    def explode(*_args, **_kwargs):  # pragma: no cover - must not be called
        raise AssertionError("the cache should have been reused")

    monkeypatch.setattr(
        "multi_period_delivery.alns.inventory_numba.state_to_inventory_arrays", explode
    )

    _, prod_arr, baseline0, shipments_arr = _resolve_inventory_cache(
        state, maps, 1, 1, 2
    )

    assert prod_arr is cached["prod_arr"]
    assert baseline0 is cached["baseline0"]
    assert shipments_arr is cached["shipments_arr"]


def test_cache_from_other_maps_is_rebuilt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Arrays built for different index maps describe a different model."""
    state = _State(cache={"maps": _maps(), "prod_arr": np.ones((1, 1, 3))})
    other_maps = _maps()
    calls = {"n": 0}

    def fake_state_to_arrays(_state, _maps, _mod=None):
        calls["n"] += 1
        return np.zeros((1, 1, 3)), None, np.ones((1, 1, 3))

    monkeypatch.setattr(
        "multi_period_delivery.alns.inventory_numba.state_to_inventory_arrays",
        fake_state_to_arrays,
    )

    _resolve_inventory_cache(state, other_maps, 1, 1, 2)

    assert calls["n"] == 1


def test_wrong_shape_invalidates_the_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    maps = _maps()
    state = _State(
        cache={
            "maps": maps,
            "prod_arr": np.ones((9, 9, 9)),  # wrong shape for (1, 1, 3)
            "baseline0": np.zeros((1, 1)),
        }
    )
    calls = {"n": 0}

    def fake_state_to_arrays(_state, _maps, _mod=None):
        calls["n"] += 1
        return np.zeros((1, 1, 3)), None, np.ones((1, 1, 3))

    monkeypatch.setattr(
        "multi_period_delivery.alns.inventory_numba.state_to_inventory_arrays",
        fake_state_to_arrays,
    )

    _resolve_inventory_cache(state, maps, 1, 1, 2)

    assert calls["n"] == 1


def test_unattachable_cache_does_not_fail_the_computation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failing to attach the cache only costs the next call a rebuild."""

    class _Frozen:
        __slots__ = ()

        def __setattr__(self, *_args):
            raise AttributeError("read-only state")

        def __getattr__(self, name):
            raise AttributeError(name)

    maps = _maps()
    state = _Frozen()

    monkeypatch.setattr(
        "multi_period_delivery.alns.inventory_numba.state_to_inventory_arrays",
        lambda *_a, **_k: (np.zeros((1, 1, 3)), None, np.ones((1, 1, 3))),
    )

    _cache, prod_arr, baseline0, _shipments = _resolve_inventory_cache(
        state, maps, 1, 1, 2
    )

    assert prod_arr is not None
    assert baseline0 is not None
