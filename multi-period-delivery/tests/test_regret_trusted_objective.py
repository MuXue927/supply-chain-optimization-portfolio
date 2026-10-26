"""M66: the search objective may skip a redundant inventory re-derivation.

``veh_loading(commit=True)`` already brings ``s_ikt`` up to date, so the
``validate()`` -> ``compute_inventory()`` pass at the start of
``SolutionState.objective`` repeats work. The trusted path keeps the formula
**identical** and only skips that traversal; if it ever disagreed, candidate
ranking (which uses the baseline) would change.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from multi_period_delivery.alns.alns_config import default_config as ALNSConfig
from multi_period_delivery.alns.alns_optimizer import ALNSOptimizer
from multi_period_delivery.alns.alnsopt import SolutionState


def _feasible_state() -> SolutionState:
    """A state that already passes ``validate()``.

    ``trust_inventory_cache`` is documented as *requiring* a feasible state:
    it skips the hard-violation check along with the redundant traversal, so an
    infeasible state must go through the plain path.
    """
    base = Path(__file__).resolve().parents[1]
    optimizer = ALNSOptimizer(
        input_file_loc=str(base / "datasets" / "multiple-periods" / "small"),
        output_file_loc=str(base / "tmp" / "m66-test"),
    )
    assert optimizer.load_data("dataset_1")
    state = optimizer.create_initial_solution()
    assert state is not None
    return state


def test_trusted_objective_matches_the_full_one() -> None:
    state = _feasible_state()
    full = state.objective()
    assert full != float("inf")

    # Invalidate the cache so the trusted call actually computes something.
    state.mark_objective_dirty()
    trusted = state.objective(trust_inventory_cache=True)

    assert trusted == pytest.approx(full, rel=0, abs=0)


def test_full_path_still_rejects_an_infeasible_state() -> None:
    """The skipped check belongs to the trusted path only.

    A caller that wants a feasibility verdict must not use
    ``trust_inventory_cache``; the default path keeps rejecting hard violations.
    """
    state = _feasible_state()
    state.objective()
    # Overload one vehicle: the full path must reject it.
    vehicle = next(v for v in state.vehicles if v.cargo)
    key = next(iter(vehicle.cargo))
    vehicle.cargo[key] = 10**6
    state.mark_objective_dirty()

    assert state.objective() == float("inf")


def test_flag_defaults_to_on_and_can_be_disabled(monkeypatch) -> None:
    assert ALNSConfig.REGRET_TRUST_CACHED_OBJECTIVE is True
    monkeypatch.setattr(ALNSConfig, "REGRET_TRUST_CACHED_OBJECTIVE", False)
    assert ALNSConfig.REGRET_TRUST_CACHED_OBJECTIVE is False


def test_trusted_call_leaves_the_cache_in_the_same_state() -> None:
    """A cache-state mismatch would silently change the *next* caller's cost."""
    state = _feasible_state()
    state.objective()
    state.mark_objective_dirty()
    trusted = state.objective(trust_inventory_cache=True)

    assert state.objective_dirty is False
    assert state.last_objective == trusted
