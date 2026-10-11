"""Focused regression tests for ALNS state and repair invariants."""

from __future__ import annotations

from pathlib import Path

from multi_period_delivery.alns.InputDataALNS import DataALNS
from multi_period_delivery.alns.alnsopt import SolutionState
from multi_period_delivery.alns.repair_utils import get_unmet_demand


def _load_state() -> SolutionState:
    base = Path(__file__).resolve().parents[1]
    data = DataALNS(
        str(base / "datasets" / "multiple-periods" / "small"),
        str(base / "tmp"),
        "dataset_1",
    )
    data.load()
    return SolutionState(data)


def test_mark_objective_dirty_clears_unmet_cache() -> None:
    state = _load_state()
    state._unmet_cache = ((1, None), {("D", "S"): 1})

    state.mark_objective_dirty()

    assert not hasattr(state, "_unmet_cache")


def test_get_unmet_demand_does_not_expose_cached_mapping() -> None:
    state = _load_state()
    first = get_unmet_demand(state)
    assert first

    first.clear()
    second = get_unmet_demand(state)

    assert second
    assert second != first
