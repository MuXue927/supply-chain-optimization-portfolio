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
