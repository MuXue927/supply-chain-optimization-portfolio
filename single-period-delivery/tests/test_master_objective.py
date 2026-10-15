"""Tests for the master-model objective helper.

Regression for the audit finding that ``print_sol_summary`` reported
``sum(var.X for var in self.d_xs.values())`` - a count of the columns weighted by
one, not the objective. The master objective prices every column
(``obj=obj_contribution`` in ``insert_pattern``) and also carries the artificial
penalty terms, so the coefficients have to be applied.
"""

from __future__ import annotations

from types import SimpleNamespace

from single_period_delivery.column_generation.masterModel import MasterModel


class _FakeVar:
    """Stand-in for a gurobipy variable exposing ``X`` and ``Obj``."""

    def __init__(self, value: float, coefficient: float) -> None:
        self.X = value
        self.Obj = coefficient


def _master_with(variables: list[_FakeVar]) -> MasterModel:
    master = MasterModel.__new__(MasterModel)
    master.model = SimpleNamespace(getVars=lambda: list(variables))
    return master


def test_objective_applies_the_coefficients() -> None:
    """Two columns of value 1 must not be reported as an objective of 2."""
    master = _master_with([_FakeVar(1.0, 5.0), _FakeVar(1.0, 7.0)])

    assert master.compute_objective_value() == 12.0


def test_objective_includes_the_penalty_terms() -> None:
    """The artificial penalty variables carry cost and must be counted."""
    master = _master_with([_FakeVar(3.0, 2.5), _FakeVar(0.0, 100.0)])

    assert master.compute_objective_value() == 7.5


def test_objective_is_zero_without_variables() -> None:
    master = _master_with([])

    assert master.compute_objective_value() == 0.0


def test_objective_ignores_unused_columns() -> None:
    """An unused column contributes nothing regardless of its coefficient."""
    master = _master_with([_FakeVar(0.0, 999.0), _FakeVar(4.0, 0.5)])

    assert master.compute_objective_value() == 2.0


def test_print_sol_summary_reports_the_priced_objective(capsys) -> None:
    """The summary line must show the priced objective, not the column count."""
    master = _master_with([_FakeVar(1.0, 5.0), _FakeVar(1.0, 7.0)])
    master.d_xs = {("a",): _FakeVar(1.0, 5.0)}
    master.get_alpha_size = lambda: 0
    master.get_beta_size = lambda: 0
    master.get_gamma_size = lambda: 0
    master.num_patterns = 2
    master.model.ObjVal = 12.0

    master.print_sol_summary()

    printed = capsys.readouterr().out
    assert "RealObj=12.0" in printed
    # The old implementation printed 2.0 here (the number of columns).
    assert "RealObj=2.0" not in printed
