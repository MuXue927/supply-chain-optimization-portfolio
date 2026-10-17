"""Destroy -> repair compositions: which invariants hold, and what is broken.

ALNS applies one destroy and one repair per iteration. The individual operator
tests cover each side alone; this file covers the composition, which is what the
search actually runs.

Measured on the small instance (2026-10-17):

* all 10 destroys leave a hard-feasible partial state;
* all 110 pairs keep ``s_ikt`` consistent, never lose coverage, and are
  hard-feasible.

Before the suffix-availability guard (2026-10-17) four pairs reproducibly
overshipped into negative inventory: ``worst_removal + smart_batch_repair``,
``periodic_shaw_removal + smart_batch_repair``,
``periodic_shaw_removal + regret_based_repair`` and
``periodic_shaw_removal + lookahead_repair``. Those are covered by the matrix
assertion here, so a regression fails with the pair name.
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

BASE = Path(__file__).resolve().parents[1]
SEED = 13

HARD_KEYS = (
    "negative_inventory",
    "veh_over_load",
    "plant_inv_exceed",
    "cargo_day_mismatch",
    "check_error",
)

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
    "regret_based_repair": (
        lambda s, r: repair.regret_based_repair(s, r, k=2, topN=6, time_limit=1.0)
    ),
    "fleet_consolidation_repair": lambda s, r: repair.fleet_consolidation_repair(s, r),
    "sequential_repair": lambda s, r: repair.sequential_repair(s, r),
    "lookahead_repair": (
        lambda s, r: repair.lookahead_repair(s, r, lookahead_horizon=1)
    ),
}

def _load_data() -> DataALNS:
    data = DataALNS(
        str(BASE / "datasets" / "multiple-periods" / "small"),
        str(BASE / "tmp"),
        "dataset_1",
    )
    data.load()
    return data


@pytest.fixture(scope="module")
def feasible_solution() -> SolutionState:
    state = SolutionState(_load_data())
    return improved_initial_solution(state, np.random.default_rng(7))


def _shipped(state: SolutionState) -> int:
    probe = state.copy()
    probe.mark_objective_dirty()
    return int(sum(probe.compute_shipped().values()))


def _hard_ok(state: SolutionState) -> bool:
    probe = state.copy()
    probe.mark_objective_dirty()
    _feasible, violations = probe.validate()
    return not any(violations.get(key) for key in HARD_KEYS)


def _inventory_consistent(state: SolutionState) -> bool:
    expected = state.copy()
    expected.compute_inventory()
    return dict(state.s_ikt) == dict(expected.s_ikt)


@pytest.mark.parametrize("destroy_name", sorted(DESTROY_OPERATORS))
@pytest.mark.parametrize("repair_name", sorted(REPAIR_OPERATORS))
def test_destroy_repair_keeps_coverage_and_inventory(
    feasible_solution: SolutionState, destroy_name: str, repair_name: str
) -> None:
    state = DESTROY_OPERATORS[destroy_name](
        feasible_solution.copy(), np.random.default_rng(SEED)
    )
    destroyed_shipped = _shipped(state)
    assert _hard_ok(state), f"{destroy_name} left a hard-infeasible partial state"

    result = REPAIR_OPERATORS[repair_name](state, np.random.default_rng(SEED))

    assert _shipped(result) >= destroyed_shipped, (
        f"{destroy_name} + {repair_name} lost coverage"
    )
    assert _inventory_consistent(result), (
        f"{destroy_name} + {repair_name} left s_ikt inconsistent"
    )
    assert _hard_ok(result), (
        f"{destroy_name} + {repair_name} produced a hard violation"
    )
