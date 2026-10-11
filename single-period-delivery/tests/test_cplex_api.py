"""Tests for the official CPLEX Python API models."""

from __future__ import annotations

from pathlib import Path

import pytest

from single_period_delivery.cplex_api_support import CplexApiModel
from single_period_delivery.cplex_api_monolithic.model import (
    SinglePeriodCplexApiMonolithic,
)
from single_period_delivery.monolithic.InputDataSingle import InputDataSingle


def test_cplex_api_support_solves_a_small_model(tmp_path: Path) -> None:
    pytest.importorskip("cplex")
    model = CplexApiModel("api_test")
    x = model.add_variable("x", kind="integer")
    model.set_objective({x: 1.0})
    model.add_constraint("lower_bound", {x: 1.0}, ">=", 2.0)

    solution = model.solve(tmp_path, time_limit=10.0, random_seed=1)

    assert solution.status == "optimal"
    assert solution.objective == 2.0
    assert solution.variables["x"] == 2.0


def test_single_period_api_monolithic_builds_model() -> None:
    pytest.importorskip("cplex")
    base = Path(__file__).resolve().parents[1]
    data = InputDataSingle(
        str(base / "datasets" / "single-period" / "small"),
        str(base / "tmp"),
        "dataset_9",
    )
    data.load()

    model = SinglePeriodCplexApiMonolithic(data)

    assert model.model.variables
    assert model.model.constraints
