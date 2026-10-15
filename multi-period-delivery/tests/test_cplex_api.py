"""Tests for the official CPLEX Python API monolithic model."""

from __future__ import annotations

from pathlib import Path

import pytest

from multi_period_delivery.cplex_api_monolithic.model import (
    MultiPeriodCplexApiMonolithic,
)
from multi_period_delivery.monolithic.InputDataMultiple import InputDataMultiple


def test_multi_period_api_monolithic_builds_expected_model() -> None:
    pytest.importorskip("cplex")
    base = Path(__file__).resolve().parents[1]
    data = InputDataMultiple(
        str(base / "datasets" / "multiple-periods" / "small"),
        str(base / "tmp"),
        "dataset_1",
    )
    data.load()

    model = MultiPeriodCplexApiMonolithic(data)

    assert len(model.x) > 0
    assert len(model.s) > 0
    assert len(model._constraint_expressions) > 0


def _small_dir() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "datasets"
        / "multiple-periods"
        / "small"
    )


def test_fleet_composition_is_reproducible_across_loads(tmp_path: Path) -> None:
    """Repeated loads must assign the same vehicle types.

    The per-vehicle type draw is seeded inside ``load()``: before that, the
    composition depended on whatever the caller had last done with the global
    ``random`` module, so the constructed model changed between runs.
    """
    compositions = []
    for _ in range(3):
        data = InputDataMultiple(str(_small_dir()), str(tmp_path), "dataset_1")
        data.load()
        counts = {
            veh_type: sum(1 for v in data.veh_type.values() if v == veh_type)
            for veh_type in data.all_veh_types
        }
        compositions.append(counts)
        assert data.max_veh_num > 0
        # Every vehicle ID receives exactly one type.
        assert len(data.veh_type) == data.max_veh_num

    assert compositions[0] == compositions[1] == compositions[2]


def test_fleet_composition_is_deterministic_after_an_unseeded_caller(
    tmp_path: Path,
) -> None:
    """A caller that dirties the global RNG must not change the model."""
    import random

    random.seed(12345)
    first = InputDataMultiple(str(_small_dir()), str(tmp_path), "dataset_1")
    first.load()

    random.seed(999)
    second = InputDataMultiple(str(_small_dir()), str(tmp_path), "dataset_1")
    second.load()

    assert first.veh_type == second.veh_type


def test_fleet_composition_can_be_changed_by_the_seed(tmp_path: Path) -> None:
    """An explicit seed must still control the composition."""
    baseline = InputDataMultiple(str(_small_dir()), str(tmp_path), "dataset_1")
    baseline.load()

    alternate = InputDataMultiple(
        str(_small_dir()), str(tmp_path), "dataset_1", seed=1
    )
    alternate.load()

    assert alternate.seed == 1
    # The upper bound is data-derived and must not depend on the seed.
    assert alternate.max_veh_num == baseline.max_veh_num
    assert alternate.veh_type != baseline.veh_type
