"""Focused regression tests for the ML operator-selection subsystem."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from multi_period_delivery.alns.alns_config import default_config as ALNSConfig
from multi_period_delivery.alns.alns_optimizer import ALNSOptimizer
from multi_period_delivery.alns.alnsopt import SolutionState
from multi_period_delivery.alns.InputDataALNS import DataALNS
from multi_period_delivery.alns.ml_destroy_selection import _DESTROY_OPERATORS
from multi_period_delivery.alns.ml_operator_selector import (
    DESTROY_OPERATOR_NAMES,
    REPAIR_OPERATOR_NAMES,
    MLOperatorSelector,
)
from multi_period_delivery.alns.ml_repair_selection import _REPAIR_OPERATORS


class _DummyTuner:
    """Minimal tuner double for selector tests."""

    def __init__(self) -> None:
        self.iteration = 0
        self.max_iterations = 100
        self.small_instance = False
        self.lookups: list[str] = []
        self.operators: dict[str, dict[str, Any]] = {}
        self.registered: list[str] = []

    def get_operator_params(self, op_name: str) -> dict[str, Any]:
        self.lookups.append(op_name)
        return {}

    def register_operator(
        self,
        op_name: str,
        params: dict[str, Any],
        ranges: dict[str, tuple[float, float]],
    ) -> None:
        self.operators[op_name] = {
            "params": dict(params),
            "ranges": dict(ranges),
        }
        self.registered.append(op_name)

    def set_iteration(
        self,
        current: int,
        maximum: int,
        no_improve_iters: int = 0,
    ) -> None:
        del no_improve_iters
        self.iteration = int(current)
        self.max_iterations = int(maximum)


class _IdentityScaler:
    """Pass-through scaler used to exercise model prediction paths."""

    def transform(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(features, dtype=float)


class _ZeroModel:
    """Model double returning a degenerate all-zero prediction vector."""

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.zeros(features.shape[0], dtype=float)


class _NaNModel:
    """Model double returning non-finite predictions."""

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(features.shape[0], np.nan, dtype=float)


def _load_state() -> SolutionState:
    base = Path(__file__).resolve().parents[1]
    data = DataALNS(
        str(base / "datasets" / "multiple-periods" / "small"),
        str(base / "tmp"),
        "dataset_1",
    )
    data.load()
    return SolutionState(data)


def _registered_operator_names(monkeypatch) -> tuple[list[str], list[str]]:
    monkeypatch.setattr(ALNSConfig, "ENABLE_ML_OPERATORS", False)
    optimizer = ALNSOptimizer()
    alns = optimizer.setup_alns()
    destroy_names = [name for name, _ in alns.destroy_operators]
    repair_names = [name for name, _ in alns.repair_operators]
    return destroy_names, repair_names


def test_ml_operator_maps_match_actual_registrations(monkeypatch) -> None:
    """The selectable ML maps must match ALNS's actual base operators."""
    registered_destroy, registered_repair = _registered_operator_names(monkeypatch)

    assert list(_DESTROY_OPERATORS) == registered_destroy
    assert list(_REPAIR_OPERATORS) == registered_repair
    assert list(DESTROY_OPERATOR_NAMES) == registered_destroy
    assert list(REPAIR_OPERATOR_NAMES) == registered_repair

    selector = MLOperatorSelector(_DummyTuner(), np.random.default_rng(123))
    assert list(selector.destroy_op_map) == registered_destroy
    assert list(selector.repair_op_map) == registered_repair

    assert "cross_period_consolidation_repair" in _REPAIR_OPERATORS
    assert "infeasible_repair" not in _REPAIR_OPERATORS


def test_state_features_are_finite_for_nonfinite_state_values(monkeypatch) -> None:
    """Feature extraction must survive NaN objectives and tracker statistics."""
    state = _load_state()
    selector = MLOperatorSelector(_DummyTuner(), np.random.default_rng(456))

    features = selector.extract_state_features(state)
    assert features.ndim == 1
    assert np.isfinite(features).all()

    monkeypatch.setattr(state, "objective", lambda: float("nan"))

    class _BadTracker:
        def get_statistics(self) -> dict[str, float]:
            return {
                "total_iterations": float("nan"),
                "max_iterations": 0.0,
                "recent_improvement": float("inf"),
            }

    state.tracker = _BadTracker()
    features = selector.extract_state_features(state)

    assert features.ndim == 1
    assert np.isfinite(features).all()


def test_reward_recording_sanitizes_nonfinite_values() -> None:
    """Non-finite operator feedback must not enter the training buffers."""
    state = _load_state()
    selector = MLOperatorSelector(_DummyTuner(), np.random.default_rng(789))

    selector.record_operator_performance(
        "random_removal",
        "destroy",
        state,
        float("nan"),
    )
    selector.record_operator_performance(
        "shaw_removal",
        "destroy",
        state,
        float("inf"),
    )
    selector.record_operator_performance(
        "path_removal",
        "destroy",
        state,
        float("-inf"),
    )

    assert all(math.isfinite(value) for value in selector.destroy_performances)
    assert selector.destroy_performances == [0.0, 1e6, -1e6]


def test_selector_caps_destroy_buffers_at_the_configured_window() -> None:
    """Destroy buffers must keep only the most recent ML_MAX_SAMPLES samples.

    Without the cap the buffers grow with every recorded operator execution and
    are never released, so memory and retraining cost scale with the total
    number of iterations. The tracker already applies the same window in
    ``ALNSTracker.update_ml_data``.
    """
    state = _load_state()
    selector = MLOperatorSelector(_DummyTuner(), np.random.default_rng(11))
    cap = int(getattr(ALNSConfig, "ML_MAX_SAMPLES", 1000))

    for index in range(cap + 5):
        selector.record_operator_performance(
            "random_removal",
            "destroy",
            state,
            float(index),
        )

    assert len(selector.destroy_features) == cap
    assert len(selector.destroy_performances) == cap
    # The window keeps the newest samples, not the oldest.
    assert selector.destroy_performances[-1] == float(cap + 4)
    # Performance labels stay aligned with their feature rows.
    assert all(
        feature.shape == selector.destroy_features[0].shape
        for feature in selector.destroy_features
    )


def test_selector_caps_repair_buffers_at_the_configured_window() -> None:
    """Repair buffers are bounded by the same window as the destroy buffers."""
    state = _load_state()
    selector = MLOperatorSelector(_DummyTuner(), np.random.default_rng(12))
    cap = int(getattr(ALNSConfig, "ML_MAX_SAMPLES", 1000))

    for index in range(cap + 3):
        selector.record_operator_performance(
            "greedy_repair",
            "repair",
            state,
            float(index),
        )

    assert len(selector.repair_features) == cap
    assert len(selector.repair_performances) == cap
    assert selector.repair_performances[-1] == float(cap + 2)
    # Untouched buffers stay empty.
    assert selector.destroy_features == []


def test_selector_window_honours_a_lower_configured_cap(monkeypatch) -> None:
    """A smaller ML_MAX_SAMPLES must take effect without code changes."""
    monkeypatch.setattr(ALNSConfig, "ML_MAX_SAMPLES", 4)
    state = _load_state()
    selector = MLOperatorSelector(_DummyTuner(), np.random.default_rng(13))

    for index in range(9):
        selector.record_operator_performance(
            "random_removal",
            "destroy",
            state,
            float(index),
        )
        selector.record_operator_performance(
            "greedy_repair",
            "repair",
            state,
            float(index),
        )

    assert selector.destroy_performances == [5.0, 6.0, 7.0, 8.0]
    assert selector.repair_performances == [5.0, 6.0, 7.0, 8.0]


def test_selector_falls_back_to_a_positive_cap(monkeypatch) -> None:
    """A non-positive or non-numeric cap must fall back instead of unbounded."""
    state = _load_state()
    selector = MLOperatorSelector(_DummyTuner(), np.random.default_rng(14))

    monkeypatch.setattr(ALNSConfig, "ML_MAX_SAMPLES", 0)
    assert MLOperatorSelector._max_samples() == 1000

    monkeypatch.setattr(ALNSConfig, "ML_MAX_SAMPLES", "not-a-number")
    assert MLOperatorSelector._max_samples() == 1000

    monkeypatch.setattr(ALNSConfig, "ML_MAX_SAMPLES", 3)
    for index in range(5):
        selector.record_operator_performance(
            "random_removal",
            "destroy",
            state,
            float(index),
        )
    assert len(selector.destroy_features) == 3


def test_selection_returns_valid_operator_without_training_or_candidates() -> None:
    """Cold-start and empty-candidate paths must always return a usable name."""
    state = _load_state()
    selector = MLOperatorSelector(_DummyTuner(), np.random.default_rng(101))

    selected, params = selector.select_best_operator(
        state,
        "destroy",
        list(_DESTROY_OPERATORS),
    )
    assert selected in _DESTROY_OPERATORS
    assert isinstance(params, dict)

    fallback, fallback_params = selector.select_best_operator(
        state,
        "destroy",
        [],
    )
    assert fallback == "random_removal"
    assert isinstance(fallback_params, dict)


def test_parameterless_repair_operator_skips_tuner_lookup() -> None:
    """A registered parameterless repair must not produce false tuner warnings."""
    state = _load_state()
    tuner = _DummyTuner()
    selector = MLOperatorSelector(tuner, np.random.default_rng(404))

    selected, params = selector.select_best_operator(
        state,
        "repair",
        ["cross_period_consolidation_repair"],
    )

    assert selected == "cross_period_consolidation_repair"
    assert params == {}
    assert tuner.lookups == []
    assert "cross_period_consolidation_repair" in tuner.operators


def test_selection_handles_degenerate_and_nonfinite_predictions() -> None:
    """A zero-weight or all-NaN model must not break operator selection."""
    state = _load_state()
    selector = MLOperatorSelector(_DummyTuner(), np.random.default_rng(202))
    candidates = list(_DESTROY_OPERATORS)

    selector.destroy_scaler = _IdentityScaler()
    selector.destroy_model = _ZeroModel()
    selected, _ = selector.select_best_operator(state, "destroy", candidates)
    assert selected in candidates

    selector.destroy_model = _NaNModel()
    selected, _ = selector.select_best_operator(state, "destroy", candidates)
    assert selected in candidates


def test_training_filters_nonfinite_samples() -> None:
    """Training buffers with invalid rows should still yield a usable state."""
    state = _load_state()
    selector = MLOperatorSelector(_DummyTuner(), np.random.default_rng(303))
    feature = selector.extract_state_features(state)
    operator_names = list(_DESTROY_OPERATORS)
    selector.destroy_features = [
        np.concatenate(
            [
                feature,
                selector.encode_operator(
                    operator_names[index % len(operator_names)],
                    "destroy",
                ),
            ]
        )
        for index in range(selector.min_samples + 2)
    ]
    selector.destroy_performances = [0.0] * len(selector.destroy_features)
    selector.destroy_features[0][0] = float("nan")
    selector.destroy_performances[-1] = float("inf")

    selector.train_models(force=True)

    assert selector.destroy_model is not None
    selected, _ = selector.select_best_operator(
        state,
        "destroy",
        list(_DESTROY_OPERATORS),
    )
    assert selected in _DESTROY_OPERATORS


# ---------------------------------------------------------------------------
# ML without a tuner: selection only, parameters from configured defaults
# ---------------------------------------------------------------------------
def test_selector_accepts_no_tuner_and_keeps_its_own_iteration_clock(
    monkeypatch,
) -> None:
    """``ENABLE_ML_OPERATORS=True`` with ``ENABLE_PARAM_TUNER=False``.

    The selector used to read the iteration from the tuner, so without one the
    iteration stayed at 0 and interval-triggered retraining never fired.
    """
    selector = MLOperatorSelector(None, np.random.default_rng(1))
    # Both knobs are read at construction time, so set them on the instance.
    selector.retrain_interval = 5
    selector.min_samples = 2

    feature = np.zeros(3)
    selector.destroy_features = [feature.copy() for _ in range(4)]
    selector.destroy_performances = [0.1] * 4

    # Force-less retraining must still be gated by the selector's own clock.
    selector.set_iteration(3, 100)
    selector.train_models()
    assert selector.destroy_model is None

    # Past the interval it must train, without any tuner present.
    selector.set_iteration(6, 100)
    selector.train_models()
    assert selector.destroy_model is not None
    assert selector.iteration == 6


def test_selector_without_tuner_uses_configured_default_params() -> None:
    selector = MLOperatorSelector(None, np.random.default_rng(2))

    assert selector._safe_operator_params("random_removal") == {}
    assert selector._safe_operator_params("greedy_repair") == {}
    # set_iteration must not fail when there is no tuner to synchronise.
    selector.set_iteration(10, 100, no_improve_iters=3)
    assert selector.max_iterations == 100


def test_ml_only_mode_builds_a_selector_without_a_tuner(
    monkeypatch, tmp_path
) -> None:
    """The optimizer must build ML-only mode with no ``ParamAutoTuner``."""
    monkeypatch.setattr(ALNSConfig, "ENABLE_PARAM_TUNER", False)
    monkeypatch.setattr(ALNSConfig, "ENABLE_ML_OPERATORS", True)
    monkeypatch.setattr(ALNSConfig, "MAX_ITERATIONS", 1)
    monkeypatch.setattr(ALNSConfig, "DYNAMIC_MAX_ITERATIONS_PER_SECOND", 0.0)
    monkeypatch.setattr(ALNSConfig, "DYNAMIC_MAX_ITERATION_BUFFER", 0)
    monkeypatch.setattr(ALNSConfig, "MAX_RUNTIME", 1.0)
    monkeypatch.setattr(ALNSConfig, "SEED", 5)

    optimizer = ALNSOptimizer(
        input_file_loc=str(
            Path(__file__).resolve().parents[1]
            / "datasets"
            / "multiple-periods"
            / "small"
        ),
        output_file_loc=str(tmp_path / "out"),
    )
    assert optimizer.load_data("dataset_1")
    state = optimizer.create_initial_solution()

    assert state is not None
    assert optimizer.param_tuner is None
    assert optimizer.ml_selector is not None
    assert state.param_tuner is None
    assert state.ml_selector is optimizer.ml_selector
