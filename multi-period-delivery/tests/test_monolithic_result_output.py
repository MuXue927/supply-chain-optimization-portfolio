"""Tests for monolithic result extraction under non-optimal solve statuses.

Background: ``create_dataframe`` used to read variable values only when
``model.Status == GRB.OPTIMAL``. A run stopped by ``GRB.TIME_LIMIT`` therefore
wrote an empty ``result.csv`` - and ``result_state`` then raised ``KeyError`` on
the missing ``Index`` column - even though the solver held a feasible incumbent
(the search path already exports it as JSON for ``TIME_LIMIT``/``INTERRUPTED``).
Medium and large instances reach the time limit routinely.

These tests drive the real ``create_dataframe`` with a stand-in model object, so
they exercise the extraction logic without needing a Gurobi licence. Gurobi
itself cannot be solved on this machine (no valid licence file), so the solver
itself is out of scope here; what is covered is the acceptance logic that
decides whether the incumbent is written out.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from multi_period_delivery.monolithic.optLogicMultiple import (
    _has_feasible_solution,
    create_dataframe,
)


class _FakeVar:
    """Minimal stand-in for a gurobipy variable (only ``Xn`` is read)."""

    def __init__(self, value: object) -> None:
        self.Xn = value


class _FakeInnerModel:
    def __init__(self, status: int, sol_count: int, *, expose_sol_count: bool = True) -> None:
        self.Status = status
        if expose_sol_count:
            self.SolCount = sol_count


def _make_model(
    status: int,
    sol_count: int,
    *,
    expose_sol_count: bool = True,
    quantities: dict[tuple[str, str, str, int, int], object] | None = None,
) -> SimpleNamespace:
    quantities = quantities if quantities is not None else {
        ("P1", "D1", "S1", 1, 1): 5.0,
        ("P1", "D1", "S1", 1, 2): 0.0,   # zero -> must be skipped
    }
    return SimpleNamespace(
        model=_FakeInnerModel(status, sol_count, expose_sol_count=expose_sol_count),
        x_ijkn_t={key: _FakeVar(value) for key, value in quantities.items()},
        all_veh_types=["10.3"],
        slot_type=lambda plant, dealer, slot, day: "10.3",
    )


# Gurobi status codes, hard-coded so the test does not import gurobipy.
GRB_OPTIMAL = 2
GRB_TIME_LIMIT = 9
GRB_INFEASIBLE = 3
GRB_INF_OR_UNBD = 4


def test_optimal_status_extracts_positive_rows() -> None:
    frame = create_dataframe(_make_model(GRB_OPTIMAL, sol_count=1))

    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["qty"] == 5
    assert row["vehicle_id"] == 1
    assert row["vehicle_type"] == "10.3"
    assert list(frame.columns) == [
        "day",
        "plant_code",
        "client_code",
        "product_code",
        "vehicle_id",
        "vehicle_type",
        "qty",
    ]


def test_time_limit_with_incumbent_extracts_rows() -> None:
    """The regression: a time-limited run must still write its incumbent."""
    model = _make_model(GRB_TIME_LIMIT, sol_count=1)

    assert _has_feasible_solution(model) is True
    frame = create_dataframe(model)

    assert len(frame) == 1
    assert frame.iloc[0]["qty"] == 5


def test_time_limit_without_incumbent_writes_nothing() -> None:
    model = _make_model(GRB_TIME_LIMIT, sol_count=0)

    assert _has_feasible_solution(model) is False
    assert create_dataframe(model).empty


@pytest.mark.parametrize("status", [GRB_INFEASIBLE, GRB_INF_OR_UNBD])
def test_statuses_without_solution_write_nothing(status: int) -> None:
    model = _make_model(status, sol_count=0)

    assert _has_feasible_solution(model) is False
    assert create_dataframe(model).empty


def test_missing_solcount_is_treated_as_no_solution() -> None:
    """An object without ``SolCount`` must not raise, and must yield no rows."""
    model = _make_model(GRB_TIME_LIMIT, sol_count=0, expose_sol_count=False)

    assert _has_feasible_solution(model) is False
    assert create_dataframe(model).empty


@pytest.mark.parametrize("value", [float("nan"), float("inf"), None, "unavailable"])
def test_unreadable_variable_values_are_skipped(value: object) -> None:
    model = _make_model(
        GRB_TIME_LIMIT,
        sol_count=1,
        quantities={("P1", "D1", "S1", 1, 1): value},
    )

    assert create_dataframe(model).empty
