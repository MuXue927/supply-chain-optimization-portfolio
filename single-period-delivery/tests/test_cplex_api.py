"""Tests for the official CPLEX Python API models."""

from __future__ import annotations

from pathlib import Path

import pytest

from single_period_delivery.column_generation.InputDataCG import InputDataCG
from single_period_delivery.cplex_api_column_generation.model import (
    CplexApiColumnGeneration,
)
from single_period_delivery.cplex_api_monolithic.model import (
    SinglePeriodCplexApiMonolithic,
)
from single_period_delivery.cplex_api_support import CplexApiModel
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


def test_cplex_api_column_generation_reads_alpha_from_dataset() -> None:
    """The minimum-load coefficient must come from the dataset, not a constant."""
    pytest.importorskip("cplex")
    base = Path(__file__).resolve().parents[1]
    input_dir = base / "datasets" / "single-period" / "small"

    data = InputDataCG(str(input_dir), str(base / "tmp"), "dataset_1")
    data.load()

    # model_config.json declares alpha = 2; the column-generation model has to
    # inherit that value instead of falling back to a hard-coded 2.0.
    assert data.param_alpha == pytest.approx(2.0)

    algorithm = CplexApiColumnGeneration(data)
    assert algorithm.alpha == pytest.approx(data.param_alpha)

    # An explicit override still wins, for callers that want another value.
    overridden = CplexApiColumnGeneration(data, alpha=3.5)
    assert overridden.alpha == pytest.approx(3.5)


def _small_single_period_dir() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "datasets"
        / "single-period"
        / "small"
    )


def test_fleet_composition_is_reproducible_across_loads(tmp_path: Path) -> None:
    """Repeated loads must assign the same vehicle types.

    The per-vehicle type draw is seeded inside ``load()``: before that, the
    composition depended on whatever the caller had last done with the global
    ``random`` module, so model construction changed between runs.
    """
    compositions = []
    for _ in range(3):
        data = InputDataSingle(
            str(_small_single_period_dir()), str(tmp_path), "dataset_1"
        )
        data.load()
        compositions.append(dict(data.veh_type))
        assert data.max_veh_num > 0
        assert len(data.veh_type) == data.max_veh_num

    assert compositions[0] == compositions[1] == compositions[2]


def test_fleet_composition_is_deterministic_after_an_unseeded_caller(
    tmp_path: Path,
) -> None:
    """A caller that dirties the global RNG must not change the model."""
    import random

    random.seed(12345)
    first = InputDataSingle(
        str(_small_single_period_dir()), str(tmp_path), "dataset_1"
    )
    first.load()

    random.seed(999)
    second = InputDataSingle(
        str(_small_single_period_dir()), str(tmp_path), "dataset_1"
    )
    second.load()

    assert first.veh_type == second.veh_type
