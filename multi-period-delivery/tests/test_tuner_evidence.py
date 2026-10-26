"""The tuner must not inject noise it cannot learn from.

Measured problem: random parameter exploration sampled new parameters for
35-50% of operator calls for the whole run, and those samples were never written
back, so they taught the tuner nothing. Because the directional update reads
``used - baseline``, a random sample also dragged the baseline in a random
direction. The tuning that was supposed to help was therefore mostly a source of
parameter churn.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from multi_period_delivery.alns.alns_config import default_config as ALNSConfig
from multi_period_delivery.alns.InputDataALNS import DataALNS
from multi_period_delivery.alns.param_tuner import ParamAutoTuner


def _load_data() -> DataALNS:
    base = Path(__file__).resolve().parents[1]
    data = DataALNS(
        str(base / "datasets" / "multiple-periods" / "small"),
        str(base / "tmp"),
        "dataset_1",
    )
    data.load()
    return data


def _tuner(monkeypatch) -> ParamAutoTuner:
    monkeypatch.setattr(ALNSConfig, "ENABLE_PARAM_TUNER", True)
    tuner = ParamAutoTuner(_load_data(), np.random.default_rng(3))
    tuner.small_instance = False
    return tuner


def _register(tuner: ParamAutoTuner) -> None:
    tuner.register_operator(
        "regret_based_repair",
        {"k": 3, "topN": 6},
        {"k": (1, 7), "topN": (2, 10)},
    )


def test_exploration_stops_after_the_seeding_phase(monkeypatch) -> None:
    monkeypatch.setattr(ALNSConfig, "TUNER_EXPLORATION_GATE_USES", 4)
    tuner = _tuner(monkeypatch)
    _register(tuner)
    op = tuner.operators["regret_based_repair"]
    op.usage_count = 10  # past the gate
    tuner.exploration_rate = 1.0  # it would explore on every read if allowed

    params = tuner.get_operator_params("regret_based_repair")

    assert params == op.params


def test_exploration_is_allowed_while_seeding(monkeypatch) -> None:
    monkeypatch.setattr(ALNSConfig, "TUNER_EXPLORATION_GATE_USES", 20)
    tuner = _tuner(monkeypatch)
    _register(tuner)
    tuner.exploration_rate = 1.0

    samples = [tuner.get_operator_params("regret_based_repair") for _ in range(20)]
    baseline = tuner.operators["regret_based_repair"].params

    assert any(sample != baseline for sample in samples)


def test_zero_gate_restores_unlimited_exploration(monkeypatch) -> None:
    monkeypatch.setattr(ALNSConfig, "TUNER_EXPLORATION_GATE_USES", 0)
    tuner = _tuner(monkeypatch)
    _register(tuner)
    op = tuner.operators["regret_based_repair"]
    op.usage_count = 500
    tuner.exploration_rate = 1.0

    sample = tuner.get_operator_params("regret_based_repair")

    assert sample != op.params


def test_adjustments_are_capped_per_operator(monkeypatch) -> None:
    """Without a cap the baseline chases noise for the entire run."""
    monkeypatch.setattr(ALNSConfig, "TUNER_MAX_ADJUSTMENTS", 2)
    monkeypatch.setattr(ALNSConfig, "TUNER_MIN_HISTORY_FOR_ADJUST", 1)
    monkeypatch.setattr(ALNSConfig, "TUNER_MAX_REL_STEP", 0.5)
    tuner = _tuner(monkeypatch)
    _register(tuner)
    tuner.exploration_rate = 0.0
    # The directional update reads ``used - baseline``; an all-zero direction
    # would never move anything, so give the operator an off-baseline snapshot.
    tuner._last_used["regret_based_repair"] = {"k": 7, "topN": 10}

    for step in range(40):
        tuner.get_operator_params("regret_based_repair")
        tuner._last_used["regret_based_repair"] = {"k": 7, "topN": 10}
        # Rising improvements so each one clears the running average.
        tuner.adjust_operator_params(
            "regret_based_repair", improvement=1.0 + 0.5 * step
        )

    assert tuner._adjust_counts["regret_based_repair"] <= 2


def test_zero_cap_keeps_adjusting(monkeypatch) -> None:
    monkeypatch.setattr(ALNSConfig, "TUNER_MAX_ADJUSTMENTS", 0)
    monkeypatch.setattr(ALNSConfig, "TUNER_MIN_HISTORY_FOR_ADJUST", 1)
    monkeypatch.setattr(ALNSConfig, "TUNER_MAX_REL_STEP", 0.5)
    tuner = _tuner(monkeypatch)
    _register(tuner)
    tuner.exploration_rate = 0.0
    tuner._last_used["regret_based_repair"] = {"k": 7, "topN": 10}

    for step in range(40):
        tuner.get_operator_params("regret_based_repair")
        tuner._last_used["regret_based_repair"] = {"k": 7, "topN": 10}
        tuner.adjust_operator_params(
            "regret_based_repair", improvement=1.0 + 0.5 * step
        )

    assert tuner._adjust_counts["regret_based_repair"] > 2
