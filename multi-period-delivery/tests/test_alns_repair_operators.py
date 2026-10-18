"""Focused tests for the ALNS repair operators and their helpers.

A repair operator receives a partially destroyed solution and puts cargo back
without ever producing a hard-infeasible state: no negative inventory, no
vehicle overload, and no cargo attributed to the wrong day. Those are the
invariants asserted here for every operator, on a real small-dataset solution.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from multi_period_delivery.alns import destroy_operators as destroy
from multi_period_delivery.alns import repair_operators as repair
from multi_period_delivery.alns.alnsopt import SolutionState
from multi_period_delivery.alns.initial_solution_utils import improved_initial_solution
from multi_period_delivery.alns.InputDataALNS import DataALNS
from multi_period_delivery.alns.repair_utils import get_unmet_demand


def _load_data() -> DataALNS:
    base = Path(__file__).resolve().parents[1]
    data = DataALNS(
        str(base / "datasets" / "multiple-periods" / "small"),
        str(base / "tmp"),
        "dataset_1",
    )
    data.load()
    return data


@pytest.fixture(scope="module")
def feasible_solution() -> SolutionState:
    """A real feasible solution, built once and copied by each test."""
    state = SolutionState(_load_data())
    return improved_initial_solution(state, np.random.default_rng(7))


def _assert_still_consistent(state: SolutionState) -> None:
    """``s_ikt`` must match a fresh recomputation; hard violations must be empty."""
    expected = state.copy()
    expected.compute_inventory()
    assert state.s_ikt == expected.s_ikt

    _feasible, violations = state.validate()
    assert not violations["negative_inventory"]
    assert not violations["veh_over_load"]
    assert not violations["cargo_day_mismatch"]


def _destroyed(state: SolutionState) -> SolutionState:
    """Return a copy with roughly a quarter of the fleet removed.

    Destroy operators build their result on a copy and return it, so the
    returned state is the partial solution, not the state passed in.
    """
    return destroy.random_removal(
        state.copy(), np.random.default_rng(3), degree=0.25
    )


# Every operator takes ``(state, rng, ...)`` and returns the (mutated) state.
REPAIR_OPERATORS = {
    "greedy_repair": lambda s, r: repair.greedy_repair(s, r),
    "fill_minimum_load_repair": lambda s, r: repair.fill_minimum_load_repair(s, r),
    "cross_period_consolidation_repair": (
        lambda s, r: repair.cross_period_consolidation_repair(s, r)
    ),
    "local_search_repair": lambda s, r: repair.local_search_repair(s, r, max_iter=20),
    "inventory_balance_repair": lambda s, r: repair.inventory_balance_repair(s, r),
    "infeasible_repair": lambda s, r: repair.infeasible_repair(s, r),
    "smart_batch_repair": lambda s, r: repair.smart_batch_repair(s, r, max_iter=20),
    "regret_based_repair": lambda s, r: repair.regret_based_repair(s, r, k=2, topN=6),
    "fleet_consolidation_repair": (
        lambda s, r: repair.fleet_consolidation_repair(s, r)
    ),
    "sequential_repair": lambda s, r: repair.sequential_repair(s, r),
    "lookahead_repair": (
        lambda s, r: repair.lookahead_repair(s, r, lookahead_horizon=1)
    ),
}


@pytest.mark.parametrize("op_name", sorted(REPAIR_OPERATORS))
def test_repair_operator_keeps_the_solution_hard_feasible(
    feasible_solution: SolutionState, op_name: str
) -> None:
    state = _destroyed(feasible_solution)

    result = REPAIR_OPERATORS[op_name](state, np.random.default_rng(13))

    assert isinstance(result, SolutionState)
    _assert_still_consistent(result)


def test_unmet_demand_equals_total_demand_when_the_fleet_is_empty(
    feasible_solution: SolutionState,
) -> None:
    state = feasible_solution.copy()
    state.vehicles = []
    state.mark_objective_dirty()

    unmet = get_unmet_demand(state)

    assert sum(unmet.values()) == sum(state.data.demands.values())
    assert all(qty > 0 for qty in unmet.values())


def test_unmet_demand_is_empty_for_a_feasible_solution(
    feasible_solution: SolutionState,
) -> None:
    assert get_unmet_demand(feasible_solution.copy()) == {}


def test_greedy_repair_recovers_demand_after_a_full_removal(
    feasible_solution: SolutionState,
) -> None:
    state = feasible_solution.copy()
    state.vehicles = []
    state.mark_objective_dirty()
    before = sum(get_unmet_demand(state).values())
    assert before > 0

    repair.greedy_repair(state, np.random.default_rng(5))

    after = sum(get_unmet_demand(state).values())
    assert after < before
    _assert_still_consistent(state)


def test_greedy_repair_does_not_rescan_the_global_vehicle_list(
    feasible_solution: SolutionState, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _destroyed(feasible_solution)

    def global_scan(*args, **kwargs):
        raise AssertionError("greedy_repair should use its local route index")

    monkeypatch.setattr(repair, "find_existing_route_vehicles", global_scan)

    repair.greedy_repair(state, np.random.default_rng(13))

    _assert_still_consistent(state)


def test_cross_period_consolidation_never_loses_demand_coverage(
    feasible_solution: SolutionState,
) -> None:
    state = _destroyed(feasible_solution)
    before = sum(state.compute_shipped().values())

    repair.cross_period_consolidation_repair(state, np.random.default_rng(3))

    after = sum(state.compute_shipped().values())
    # Consolidation moves cargo between periods; it must not drop coverage.
    assert after >= before
    _assert_still_consistent(state)


def test_fill_minimum_load_repair_does_not_add_underloaded_vehicles(
    feasible_solution: SolutionState,
) -> None:
    state = _destroyed(feasible_solution)
    min_load = state.data.veh_type_min_load

    def underloaded(current: SolutionState) -> int:
        return sum(
            1
            for veh in current.iter_used_vehicles()
            if current.compute_veh_load(veh) + 1e-9 < min_load.get(veh.type, 0.0)
        )

    before = underloaded(state)
    repair.fill_minimum_load_repair(state, np.random.default_rng(2))

    assert underloaded(state) <= before
    _assert_still_consistent(state)


def test_regret_repair_does_not_copy_when_fast_estimates_reject_all_candidates(
    feasible_solution: SolutionState, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _destroyed(feasible_solution)
    copy_calls = {"clone_false": 0}
    original_copy = SolutionState.copy

    def counted_copy(self: SolutionState, clone_vehicles: bool = True):
        if not clone_vehicles:
            copy_calls["clone_false"] += 1
        return original_copy(self, clone_vehicles=clone_vehicles)

    monkeypatch.setattr(SolutionState, "copy", counted_copy)
    monkeypatch.setattr(
        repair,
        "compute_improvement_fast_add_vehicle",
        lambda *args, **kwargs: (0.0, 0.0, False),
    )

    repair.regret_based_repair(
        state,
        np.random.default_rng(13),
        k=2,
        topN=6,
        time_limit=1.0,
    )

    assert copy_calls["clone_false"] == 0
