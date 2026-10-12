"""Focused tests for the ALNS destroy operators and their helpers.

Every destroy operator shares one contract: given a feasible solution it
removes vehicles (never adds any), never drives inventory negative, and leaves
``s_ikt`` equal to what a fresh recomputation from the surviving fleet would
produce. The operators do not need a solver, so CI can exercise all of them on
a real small-dataset solution.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from multi_period_delivery.alns import destroy_operators as destroy
from multi_period_delivery.alns.alnsopt import SolutionState
from multi_period_delivery.alns.destroy_utils import (
    post_destroy,
    resolve_degree,
    summarize_removed_cargo,
)
from multi_period_delivery.alns.initial_solution_utils import improved_initial_solution
from multi_period_delivery.alns.InputDataALNS import DataALNS


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
    """A real feasible solution, built once and copied by each test.

    Constructing it costs roughly a second, so the module shares one instance
    and hands every test its own copy.
    """
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


# Every operator takes ``(state, rng, ...)`` and returns the (mutated) state.
# Explicit degrees keep the removals small so the tests stay fast.
DESTROY_OPERATORS = {
    "random_removal": lambda s, r: destroy.random_removal(s, r, degree=0.25),
    "worst_removal": lambda s, r: destroy.worst_removal(s, r, degree=0.25),
    "infeasible_removal": lambda s, r: destroy.infeasible_removal(s, r),
    "surplus_inventory_removal": (
        lambda s, r: destroy.surplus_inventory_removal(s, r, degree=0.25)
    ),
    "shaw_removal": lambda s, r: destroy.shaw_removal(s, r, degree=0.3),
    "periodic_shaw_removal": (
        lambda s, r: destroy.periodic_shaw_removal(s, r, degree=0.3, k_cluster=3)
    ),
    "cross_period_dealer_removal": (
        lambda s, r: destroy.cross_period_dealer_removal(s, r, degree=0.3)
    ),
    "period_range_removal": (
        lambda s, r: destroy.period_range_removal(s, r, degree=0.3)
    ),
    "path_removal": lambda s, r: destroy.path_removal(s, r, degree=0.3),
    "inventory_level_removal": (
        lambda s, r: destroy.inventory_level_removal(s, r, degree=0.3)
    ),
}


@pytest.mark.parametrize("op_name", sorted(DESTROY_OPERATORS))
def test_destroy_operator_never_grows_the_fleet(
    feasible_solution: SolutionState, op_name: str
) -> None:
    state = feasible_solution.copy()
    before = state.count_used_vehicles()

    result = DESTROY_OPERATORS[op_name](state, np.random.default_rng(11))

    assert isinstance(result, SolutionState)
    assert result.count_used_vehicles() <= before
    _assert_still_consistent(result)


@pytest.mark.parametrize("op_name", sorted(DESTROY_OPERATORS))
def test_destroy_operator_is_deterministic_for_a_fixed_seed(
    feasible_solution: SolutionState, op_name: str
) -> None:
    first = DESTROY_OPERATORS[op_name](feasible_solution.copy(), np.random.default_rng(23))
    second = DESTROY_OPERATORS[op_name](
        feasible_solution.copy(), np.random.default_rng(23)
    )

    assert first.count_used_vehicles() == second.count_used_vehicles()
    assert first.s_ikt == second.s_ikt


def test_resolve_degree_clamps_to_the_unit_interval(
    feasible_solution: SolutionState,
) -> None:
    state = feasible_solution.copy()

    assert resolve_degree("random_removal", state, 5.0) == 1.0
    assert resolve_degree("random_removal", state, -2.0) == 0.0
    assert resolve_degree("random_removal", state, 0.3) == pytest.approx(0.3)


def test_resolve_degree_uses_the_configured_default(
    feasible_solution: SolutionState,
) -> None:
    state = feasible_solution.copy()

    # random_removal's published default degree is 0.25.
    assert resolve_degree("random_removal", state, None) == pytest.approx(0.25)


def test_summarize_removed_cargo_matches_the_vehicle_cargo(
    feasible_solution: SolutionState,
) -> None:
    state = feasible_solution.copy()
    vehicle = next(veh for veh in state.vehicles if veh.cargo)

    summary = summarize_removed_cargo([vehicle])

    assert sum(summary.values()) == sum(vehicle.cargo.values())
    assert all(qty > 0 for qty in summary.values())
    # Keys are (plant, sku, day) triples.
    assert all(len(key) == 3 for key in summary)


def test_summarize_removed_cargo_handles_an_empty_list() -> None:
    assert summarize_removed_cargo([]) == {}


def test_post_destroy_restores_cargo_into_inventory(
    feasible_solution: SolutionState,
) -> None:
    state = feasible_solution.copy()
    vehicle = next(veh for veh in state.vehicles if veh.cargo)
    state.vehicles.remove(vehicle)

    post_destroy(state, [vehicle], "random_removal", early=True, prev_obj=1.0)

    _assert_still_consistent(state)
