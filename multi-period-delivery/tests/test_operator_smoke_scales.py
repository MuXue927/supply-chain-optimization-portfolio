"""Slow smoke: every operator on medium and large.

The default suites exercise operators on the small instance only. This file
repeats the destroy/repair invariants on medium and large. It is marked
``slow`` and excluded from the default run (``pytest -m "not slow"``) because
building the large initial solution alone takes ~18 s and
``greedy_repair``/``local_search_repair`` take ~31-36 s each there.

Measured 2026-10-18:

* medium: all 10 destroys and all 11 repairs keep hard feasibility, ``s_ikt``
  consistency and non-decreasing coverage;
* large: all destroys and all repairs do too, **except**
  ``inventory_balance_repair``, which ends with negative inventory
  (4 entries at day 5). The negative balance appears between two loads (a
  refresh), not inside one ``veh_loading`` call, so it needs its own
  investigation; it is pinned as an ``xfail`` below.

Run explicitly with ``pytest -m slow multi-period-delivery/tests/test_operator_smoke_scales.py``.
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

pytestmark = pytest.mark.slow

BASE = Path(__file__).resolve().parents[1]
SEED = 13
SCALES = (("medium", "dataset_13"), ("large", "dataset_3"))
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
        lambda s, r: repair.regret_based_repair(s, r, k=2, topN=6, time_limit=2.0)
    ),
    "fleet_consolidation_repair": lambda s, r: repair.fleet_consolidation_repair(s, r),
    "sequential_repair": lambda s, r: repair.sequential_repair(s, r),
    "lookahead_repair": (
        lambda s, r: repair.lookahead_repair(s, r, lookahead_horizon=1)
    ),
}


def _load(scale: str, dataset: str) -> DataALNS:
    data = DataALNS(
        str(BASE / "datasets" / "multiple-periods" / scale),
        str(BASE / "tmp"),
        dataset,
    )
    data.load()
    return data


@pytest.fixture(scope="module", params=SCALES, ids=[s[0] for s in SCALES])
def base_solution(request) -> SolutionState:
    scale, dataset = request.param
    return improved_initial_solution(
        SolutionState(_load(scale, dataset)), np.random.default_rng(7)
    )


def _measure(state: SolutionState) -> tuple[int, bool, bool]:
    probe = state.copy()
    probe.mark_objective_dirty()
    shipped = int(sum(probe.compute_shipped().values()))
    _feasible, violations = probe.validate()
    hard_ok = not any(violations.get(key) for key in HARD_KEYS)
    expected = state.copy()
    expected.compute_inventory()
    return shipped, hard_ok, dict(state.s_ikt) == dict(expected.s_ikt)


@pytest.mark.parametrize("op_name", sorted(DESTROY_OPERATORS))
def test_destroy_operator_on_real_scale(base_solution: SolutionState, op_name: str) -> None:
    result = DESTROY_OPERATORS[op_name](
        base_solution.copy(), np.random.default_rng(SEED)
    )

    _shipped, hard_ok, sikt_ok = _measure(result)
    assert hard_ok, f"{op_name} left a hard-infeasible partial state"
    assert sikt_ok, f"{op_name} left s_ikt inconsistent"


@pytest.mark.parametrize("op_name", sorted(REPAIR_OPERATORS))
def test_repair_operator_on_real_scale(
    base_solution: SolutionState, op_name: str, request: pytest.FixtureRequest
) -> None:
    # Known large-scale defect (2026-10-18): it ends with negative inventory.
    if request.node.callspec.params["base_solution"][0] == "large" and (
        op_name == "inventory_balance_repair"
    ):
        pytest.xfail("known large-scale overship in inventory_balance_repair")

    destroyed = destroy.random_removal(
        base_solution.copy(), np.random.default_rng(3), degree=0.25
    )
    before_shipped, _hard, _ok = _measure(destroyed)

    result = REPAIR_OPERATORS[op_name](destroyed, np.random.default_rng(SEED))
    after_shipped, hard_ok, sikt_ok = _measure(result)

    assert hard_ok, f"{op_name} produced a hard violation"
    assert sikt_ok, f"{op_name} left s_ikt inconsistent"
    assert after_shipped >= before_shipped, f"{op_name} lost coverage"
