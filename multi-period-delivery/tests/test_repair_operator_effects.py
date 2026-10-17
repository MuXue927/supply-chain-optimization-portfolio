"""Repair operators must not only stay feasible - they must help.

The invariant tests in ``test_alns_repair_operators.py`` only assert that a
repair leaves the state hard-feasible. That is necessary but not sufficient: an
operator that silently does nothing would pass it.

Measured on the destroyed small-instance solution (three seeds, 2026-10-17):
covering nine demand-repairing operators shipped 225-914 more units and never
worsened the search objective; the two consolidation/infeasibility operators are
no-ops on this demand-only destruction (0 shipped, 0 objective change).

These tests pin the *direction*:

* coverage (shipped quantity) never decreases;
* the nine demand-repairing operators actually put demand back;
* the search objective never worsens;
* hard feasibility holds.
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
SEEDS = (11, 23, 37)

HARD_KEYS = (
    "negative_inventory",
    "veh_over_load",
    "plant_inv_exceed",
    "cargo_day_mismatch",
    "check_error",
)

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

# These two are not demand-repair operators: consolidation moves existing cargo,
# and infeasible_repair only fixes hard violations (none after a random removal).
NO_DEMAND_REPAIR = {
    "cross_period_consolidation_repair",
    "infeasible_repair",
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
def destroyed_solution() -> SolutionState:
    """One feasible solution, destroyed the same way for every operator."""
    state = SolutionState(_load_data())
    feasible = improved_initial_solution(state, np.random.default_rng(7))
    return destroy.random_removal(
        feasible, np.random.default_rng(3), degree=0.25
    )


def _metrics(state: SolutionState) -> tuple[int, float, bool]:
    """(shipped total, search objective, hard-feasible) on an independent copy."""
    probe = state.copy()
    probe.mark_objective_dirty()
    shipped = int(sum(probe.compute_shipped().values()))
    objective = float(probe.objective())
    _feasible, violations = probe.validate()
    hard_ok = not any(violations.get(key) for key in HARD_KEYS)
    return shipped, objective, hard_ok


@pytest.mark.parametrize("op_name", sorted(REPAIR_OPERATORS))
@pytest.mark.parametrize("seed", SEEDS)
def test_repair_operator_improves_or_preserves(
    destroyed_solution: SolutionState, op_name: str, seed: int
) -> None:
    state = destroyed_solution.copy()
    before_shipped, before_obj, _ = _metrics(state)

    result = REPAIR_OPERATORS[op_name](state, np.random.default_rng(seed))
    after_shipped, after_obj, hard_ok = _metrics(result)

    assert hard_ok, f"{op_name}/{seed}: repair produced a hard violation"
    assert after_shipped >= before_shipped, (
        f"{op_name}/{seed}: coverage dropped "
        f"({before_shipped} -> {after_shipped})"
    )
    assert after_obj <= before_obj + 1e-9, (
        f"{op_name}/{seed}: search objective worsened "
        f"({before_obj:.2f} -> {after_obj:.2f})"
    )
    if op_name not in NO_DEMAND_REPAIR:
        assert after_shipped > before_shipped, (
            f"{op_name}/{seed}: expected the operator to repair demand, "
            f"but shipped stayed at {before_shipped}"
        )
