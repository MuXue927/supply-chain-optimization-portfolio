"""Tests for the extracted inventory-cache preparation helper.

``compute_inventory_numba_safe`` wraps its whole pipeline in one ``try`` that
falls back to the pure-Python implementation, so its body was ~166 lines of
preparation and computation mixed together. The preparation is now a separate
function with its own contract:

* :func:`_resolve_inventory_cache` - reuse the cached arrays when they belong to
  the current index maps and have the right shapes.

Cache *validity* ("has anything feeding shipments changed?") used to be decided
here by per-vehicle signatures. It is now the O(1) mutation stamp in
``vehicle.inventory_stamp``, covered by ``test_inventory_freshness.py``; the
signature helpers were deleted once nothing called them.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from multi_period_delivery.alns.inventory_numba import (
    _resolve_inventory_cache,
    state_to_inventory_arrays,
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
    # plant_to_idx/sku_to_idx are required by _production_array.
    return SimpleNamespace(
        n_plants=n_plants,
        n_skus=n_skus,
        horizons=horizons,
        plant_to_idx={},
        sku_to_idx={},
    )


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


# ---------------------------------------------------------------------------
# state_to_inventory_arrays no longer builds shipments by default
# ---------------------------------------------------------------------------


def test_state_to_arrays_skips_shipments_by_default(monkeypatch) -> None:
    """Every caller discarded it, so building it was pure waste.

    Profiling the medium instance showed this function spending ~70% of its time
    on an array nobody used - 9,713 of 12,162 shipments builds in a 91 s run.
    """
    calls = {"n": 0}

    def counting_build(*args, **kwargs):
        calls["n"] += 1
        return np.zeros((1, 1, 3))

    monkeypatch.setattr(
        "multi_period_delivery.alns.inventory_numba._build_shipments_from_vehicles",
        counting_build,
    )
    state = SimpleNamespace(s_ikt={}, data=SimpleNamespace(sku_prod_each_day={}))

    _s, shipments, _prod = state_to_inventory_arrays(state, _maps())

    assert calls["n"] == 0, "shipments must not be built unless requested"
    assert shipments is None


def test_state_to_arrays_builds_shipments_when_asked(monkeypatch) -> None:
    """The opt-in path must still work for any future caller that needs it."""
    calls = {"n": 0}

    def counting_build(*args, **kwargs):
        calls["n"] += 1
        return np.full((1, 1, 3), 5.0)

    monkeypatch.setattr(
        "multi_period_delivery.alns.inventory_numba._build_shipments_from_vehicles",
        counting_build,
    )
    state = SimpleNamespace(s_ikt={}, data=SimpleNamespace(sku_prod_each_day={}))

    _s, shipments, _prod = state_to_inventory_arrays(
        state, _maps(), build_shipments=True
    )

    assert calls["n"] == 1
    assert shipments is not None
    assert np.all(shipments == 5.0)


# ---------------------------------------------------------------------------
# the production array is solution-independent and cached
# ---------------------------------------------------------------------------


def _prod_state(plants, skus, horizons, prod):
    return SimpleNamespace(
        s_ikt={},
        data=SimpleNamespace(sku_prod_each_day=prod),
    )


def test_production_is_built_once_and_reused() -> None:
    """It depends only on the data, so rebuilding per call is recomputation.

    The medium instance spent 318 us of a 795 us call on this fill (~40%), and the
    function runs ~18,000 times per 120 s.
    """
    maps = _maps()
    maps.plant_to_idx = {"P": 0}
    maps.sku_to_idx = {"S": 0}
    state = _prod_state(1, 1, 2, {("P", "S", 1): 7.0})

    first = state_to_inventory_arrays(state, maps)[2]
    second = state_to_inventory_arrays(state, maps)[2]

    assert first is second, "the cached array must be reused, not rebuilt"
    assert first[0, 0, 1] == 7.0


def test_production_cache_invalidates_when_the_mapping_is_replaced() -> None:
    """A replaced production dict must not be served the old array."""
    maps = _maps()
    maps.plant_to_idx = {"P": 0}
    maps.sku_to_idx = {"S": 0}
    state = _prod_state(1, 1, 2, {("P", "S", 1): 7.0})
    first = state_to_inventory_arrays(state, maps)[2]
    assert first[0, 0, 1] == 7.0

    state.data.sku_prod_each_day = {("P", "S", 1): 3.0}
    second = state_to_inventory_arrays(state, maps)[2]

    assert second[0, 0, 1] == 3.0, "the replaced mapping must be picked up"


def test_production_cache_invalidates_for_other_maps() -> None:
    """A different model (other index maps) must not reuse the old array."""
    first_maps = _maps()
    first_maps.plant_to_idx = {"P": 0}
    first_maps.sku_to_idx = {"S": 0}
    state = _prod_state(1, 1, 2, {("P", "S", 1): 7.0})
    state_to_inventory_arrays(state, first_maps)
    first_key = state.data._sku_prod_arr_cache[0]

    other_maps = _maps()
    other_maps.plant_to_idx = {"P": 0}
    other_maps.sku_to_idx = {"S": 0}
    state_to_inventory_arrays(state, other_maps)

    # A distinct maps object means the cached array cannot be proven to apply, so
    # the cache must have been re-keyed to the new maps.
    assert state.data._sku_prod_arr_cache[0] is other_maps
    assert first_key is not other_maps
