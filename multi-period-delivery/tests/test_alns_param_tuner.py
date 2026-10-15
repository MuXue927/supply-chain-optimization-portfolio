"""Focused tests for the ALNS automatic parameter tuner.

``param_tuner`` keeps the operator-parameter baselines, the exploration /
exploitation / refinement schedule, the directional updates driven by observed
improvements, and the destroy-strength ``adaptive_degree``. It is pure logic
with no solver dependency, so it is tested directly rather than through a full
search run.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from multi_period_delivery.alns.alns_config import default_config as ALNSConfig
from multi_period_delivery.alns.InputDataALNS import DataALNS
from multi_period_delivery.alns.param_tuner import OperatorParams, ParamAutoTuner


def _load_data() -> DataALNS:
    base = Path(__file__).resolve().parents[1]
    data = DataALNS(
        str(base / "datasets" / "multiple-periods" / "small"),
        str(base / "tmp"),
        "dataset_1",
    )
    data.load()
    return data


def _make_tuner() -> ParamAutoTuner:
    tuner = ParamAutoTuner(_load_data(), np.random.default_rng(0))
    # Force the "large instance" branch so the schedule is not further capped.
    tuner.small_instance = False
    return tuner


def _advance_tuner_clock(tuner: ParamAutoTuner, fraction: float) -> None:
    """Place the tuner ``fraction`` of the way through the runtime budget.

    The tuner derives its progress from elapsed time, so a test that calls
    ``set_iteration`` immediately after construction sees ~0 elapsed time and no
    decay. Setting the origin explicitly keeps these tests deterministic.
    """
    budget = float(getattr(ALNSConfig, "MAX_RUNTIME", 600))
    tuner.start_time = time.time() - fraction * budget


# ---------------------------------------------------------------------------
# OperatorParams
# ---------------------------------------------------------------------------
def test_operator_params_caps_history_and_tracks_success_rate() -> None:
    op = OperatorParams(name="demo")

    for index in range(110):
        op.update_performance(1.0, success=index % 2 == 0)

    assert len(op.performance_history) == 100
    assert op.usage_count == 110
    assert op.success_count == 55
    assert op.get_success_rate() == 0.5


def test_operator_params_average_uses_recent_window() -> None:
    op = OperatorParams(name="demo")
    op.performance_history = [10.0, 1.0, 1.0, 1.0]

    assert op.get_avg_performance(window=3) == 1.0
    # No history at all -> a neutral 0.0 rather than an exception.
    assert OperatorParams(name="empty").get_avg_performance() == 0.0


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def test_tuner_registers_default_destroy_and_repair_operators() -> None:
    tuner = _make_tuner()

    assert "random_removal" in tuner.operators
    assert tuner.operators["random_removal"].params == {"degree": 0.25}
    assert tuner.operators["random_removal"].ranges


def test_register_operator_overrides_baseline() -> None:
    tuner = _make_tuner()

    tuner.register_operator("custom_repair", {"degree": 0.4}, {"degree": (0.1, 0.9)})

    assert tuner.operators["custom_repair"].params == {"degree": 0.4}
    assert tuner.operators["custom_repair"].ranges == {"degree": (0.1, 0.9)}


def test_registration_does_not_alias_the_shared_config_table() -> None:
    # The default table lives on the shared config object; a tuner must not be
    # able to mutate it, or one run's tuning would leak into the next run.
    tuner = _make_tuner()

    assert (
        tuner.operators["random_removal"].params
        is not ALNSConfig.OPERATOR_DEFAULTS["random_removal"]["params"]
    )


def test_tuning_does_not_mutate_the_shared_config_table() -> None:
    tuner = _make_tuner()
    name = "random_removal"
    op = tuner.operators[name]
    for _ in range(5):
        op.update_performance(1.0, success=True)
    tuner._last_used[name] = {"degree": 0.5}

    tuner.adjust_operator_params(name, improvement=5.0, success=True)

    assert op.params["degree"] != 0.25  # the live baseline did move ...
    # ... but the shared configuration table stayed at its published default.
    assert ALNSConfig.OPERATOR_DEFAULTS[name]["params"]["degree"] == 0.25


# ---------------------------------------------------------------------------
# Phase schedule
# ---------------------------------------------------------------------------
def test_set_iteration_switches_phase() -> None:
    tuner = _make_tuner()

    tuner.set_iteration(10, 100)
    assert tuner.phase == "exploration"
    assert tuner.exploration_rate == pytest.approx(0.45)

    tuner.set_iteration(50, 100)
    assert tuner.phase == "exploitation"
    assert tuner.exploration_rate == pytest.approx(0.25)

    tuner.set_iteration(80, 100)
    assert tuner.phase == "refinement"
    assert tuner.exploration_rate == pytest.approx(0.1)


def test_set_iteration_clips_exploration_rate_to_zero_at_the_end() -> None:
    tuner = _make_tuner()

    tuner.set_iteration(100, 100)

    assert tuner.exploration_rate == 0.0


def test_small_instance_caps_exploration_rate() -> None:
    tuner = _make_tuner()
    tuner.small_instance = True

    tuner.set_iteration(10, 100)  # uncapped schedule would be 0.45

    assert tuner.exploration_rate <= 0.18


# ---------------------------------------------------------------------------
# Parameter retrieval
# ---------------------------------------------------------------------------
def test_get_operator_params_warns_on_unregistered_operator(capsys) -> None:
    tuner = _make_tuner()

    params = tuner.get_operator_params("does_not_exist")

    assert params == {}
    assert "WARNING" in capsys.readouterr().out


def test_get_operator_params_exploration_stays_within_range() -> None:
    tuner = _make_tuner()
    tuner.exploration_rate = 1.0  # always explore

    params = tuner.get_operator_params("random_removal")

    low, high = tuner.operators["random_removal"].ranges["degree"]
    assert low <= params["degree"] <= high
    # Exploration samples are not written back to the baseline.
    assert tuner.operators["random_removal"].params == {"degree": 0.25}


def test_get_operator_params_exploitation_returns_baseline() -> None:
    tuner = _make_tuner()
    tuner.exploration_rate = 0.0  # never explore

    params = tuner.get_operator_params("random_removal")

    assert params == {"degree": 0.25}
    assert params is not tuner.operators["random_removal"].params


# ---------------------------------------------------------------------------
# Directional tuning
# ---------------------------------------------------------------------------
def test_adjust_operator_params_waits_for_min_history() -> None:
    tuner = _make_tuner()

    tuner.adjust_operator_params("random_removal", improvement=5.0, success=True)

    assert tuner.operators["random_removal"].params == {"degree": 0.25}


def test_adjust_operator_params_follows_the_used_direction() -> None:
    tuner = _make_tuner()
    name = "random_removal"
    op = tuner.operators[name]
    for _ in range(5):
        op.update_performance(1.0, success=True)
    # Pretend the last execution explored upwards from the baseline.
    tuner._last_used[name] = {"degree": 0.5}

    tuner.adjust_operator_params(name, improvement=5.0, success=True)

    assert 0.25 < op.params["degree"] <= 0.5


def test_adjust_operator_params_ignores_unsuccessful_use() -> None:
    tuner = _make_tuner()
    name = "random_removal"
    op = tuner.operators[name]
    for _ in range(5):
        op.update_performance(1.0, success=True)
    tuner._last_used[name] = {"degree": 0.5}

    tuner.adjust_operator_params(name, improvement=5.0, success=False)

    assert op.params["degree"] == 0.25


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------
def test_get_best_operators_requires_usage_history() -> None:
    tuner = _make_tuner()
    tuner.register_operator("proven_removal", {"degree": 0.3}, {"degree": (0.1, 0.5)})
    tuner.register_operator("untried_removal", {"degree": 0.3}, {"degree": (0.1, 0.5)})
    for _ in range(6):
        tuner.operators["proven_removal"].update_performance(2.0, success=True)

    best = tuner.get_best_operators("removal", top_k=10)

    assert "proven_removal" in best
    assert "untried_removal" not in best


# ---------------------------------------------------------------------------
# Adaptive destroy strength
# ---------------------------------------------------------------------------
def test_adaptive_degree_returns_base_before_the_search_starts() -> None:
    tuner = _make_tuner()

    assert tuner.adaptive_degree(0.4) == 0.4


def test_adaptive_degree_decays_and_stays_in_bounds() -> None:
    tuner = _make_tuner()
    tuner.set_iteration(100, 200)
    # Progress is measured against the runtime budget, so advance the clock too.
    _advance_tuner_clock(tuner, 0.5)

    value = tuner.adaptive_degree(0.5, min_degree=0.05, max_degree=0.5)

    assert 0.05 <= value < 0.5


def test_adaptive_degree_boosts_after_stagnation() -> None:
    tuner = _make_tuner()
    tuner.set_iteration(100, 200)
    _advance_tuner_clock(tuner, 0.5)
    baseline = tuner.adaptive_degree(0.4)

    tuner.no_improve_iters = 400  # far past the stagnation threshold
    boosted = tuner.adaptive_degree(0.4)

    assert boosted > baseline


def test_adaptive_degree_is_unchanged_early_in_the_run() -> None:
    """No decay while little of the runtime budget has been consumed."""
    tuner = _make_tuner()
    tuner.set_iteration(100, 200)  # half the *iteration* ceiling
    _advance_tuner_clock(tuner, 0.0)  # but the run has just started

    assert tuner.adaptive_degree(0.4) == pytest.approx(0.4)
