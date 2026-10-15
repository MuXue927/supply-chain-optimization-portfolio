"""Tests for the phase-progress measure that drives the adaptive ALNS policies.

The phase thresholds (``LIGHT_INTENSIFY_MIN_PROGRESS`` 0.20,
``LATE_FLEET_HARD_PRIORITY_START_PROGRESS`` 0.60,
``ACTUAL_PRIORITY_START_PROGRESS`` 0.70, the elite-relink window,
``ML_TAKEOVER_MIN_PROGRESS`` 0.30) read as fractions of the run's budget: 0.70
means "the last 30% of the run".

Those policies must activate at the same point of a run on any host. An
iteration-based measure could not deliver that: converting "iterations" into
"fraction of the run" needs an assumed throughput, and throughput is both
machine- and instance-specific. Measuring against elapsed time makes the
fraction independent of how fast the host is.

These tests pin that property, including the failure mode of the old measure
(a ceiling sized by a throughput constant).
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from multi_period_delivery.alns.accept import DemandConstraintAccept
from multi_period_delivery.alns.alns_config import default_config as ALNSConfig


class _Tracker:
    """Tracker stand-in exposing only what ``_resolve_progress`` reads."""

    def __init__(self, *, iteration: int = 0, max_iterations: int = 0, elapsed: float = 0.0) -> None:
        self.iteration = iteration
        self.max_iterations = max_iterations
        # Monotonic origin, exactly as ALNSTracker records it.
        self.budget_start_time = time.perf_counter() - elapsed


class _BaseAcceptor:
    """Minimal base acceptor; ``_resolve_progress`` never delegates to it."""

    def __call__(self, *args: object, **kwargs: object) -> bool:
        return True


def _accept() -> DemandConstraintAccept:
    # Only the progress helper is exercised here, so the base acceptor and the
    # problem data are irrelevant.
    return DemandConstraintAccept(_BaseAcceptor(), data=None, penalty_factor=1.0)


def _state(tracker: _Tracker) -> SimpleNamespace:
    return SimpleNamespace(tracker=tracker)


# ---------------------------------------------------------------------------
# Elapsed-time measure
# ---------------------------------------------------------------------------


def test_progress_tracks_the_share_of_the_runtime_budget() -> None:
    accept = _accept()
    runtime = float(ALNSConfig.MAX_RUNTIME)

    progress = accept._resolve_progress(_state(_Tracker(elapsed=runtime / 2)))

    assert progress == pytest.approx(0.5, abs=0.05)


def test_progress_is_clamped_to_the_unit_interval() -> None:
    accept = _accept()

    assert accept._resolve_progress(_state(_Tracker(elapsed=0.0))) < 1e-6
    assert accept._resolve_progress(_state(_Tracker(elapsed=10_000_000))) == 1.0


def test_progress_reaches_phase_thresholds_within_one_budget() -> None:
    """Every documented gate must be reachable before the budget expires."""
    accept = _accept()
    runtime = float(ALNSConfig.MAX_RUNTIME)
    thresholds = {
        "light_intensify": ALNSConfig.LIGHT_INTENSIFY_MIN_PROGRESS,
        "late_fleet": ALNSConfig.LATE_FLEET_HARD_PRIORITY_START_PROGRESS,
        "actual_priority": ALNSConfig.ACTUAL_PRIORITY_START_PROGRESS,
        "ml_takeover": ALNSConfig.ML_TAKEOVER_MIN_PROGRESS,
    }

    for name, threshold in thresholds.items():
        tracker = _Tracker(elapsed=threshold * runtime)
        progress = accept._resolve_progress(_state(tracker))
        assert progress >= threshold - 0.05, f"{name} gate not reached"
        assert progress < 1.0, f"{name} gate must be reached before the budget ends"


# ---------------------------------------------------------------------------
# The property that makes it machine-independent
# ---------------------------------------------------------------------------


def test_phase_schedule_does_not_depend_on_machine_speed() -> None:
    """A far faster host must not shift when the gates fire.

    Both trackers sit at the same point of the *budget* (70% elapsed) but have
    executed wildly different iteration counts, which is exactly what different
    hardware looks like. The measured progress must be the same.
    """
    accept = _accept()
    runtime = float(ALNSConfig.MAX_RUNTIME)
    elapsed = 0.70 * runtime

    slow_host = _Tracker(iteration=40, max_iterations=1_000, elapsed=elapsed)
    fast_host = _Tracker(iteration=4_000_000, max_iterations=900_200, elapsed=elapsed)

    slow_progress = accept._resolve_progress(_state(slow_host))
    fast_progress = accept._resolve_progress(_state(fast_host))

    assert slow_progress == pytest.approx(0.70, abs=0.02)
    assert fast_progress == pytest.approx(0.70, abs=0.02)
    # The iteration-based measure is what diverged: at the same instant the two
    # hosts report completely different fractions.
    slow_iteration_fraction = slow_host.iteration / slow_host.max_iterations
    fast_iteration_fraction = fast_host.iteration / fast_host.max_iterations
    assert slow_iteration_fraction != pytest.approx(fast_iteration_fraction, abs=0.1)


def test_old_iteration_measure_failure_mode_is_covered() -> None:
    """A ceiling inflated by a throughput constant used to disable every gate.

    This documents the failure the elapsed-time measure removes: with the
    historical ``DYNAMIC_MAX_ITERATIONS_PER_SECOND = 1000.0`` the ceiling was
    900,200 iterations, so a run executing a few thousand iterations reported a
    progress around 0.005 - below all five thresholds. Progress no longer reads
    that ceiling at all, so the inflated value cannot suppress the gates.
    """
    accept = _accept()
    inflated = _Tracker(iteration=4_700, max_iterations=900_200, elapsed=0.0)

    # Same tracker, but the run is genuinely half done: the gates follow the
    # clock, not the iteration ratio.
    assert accept._resolve_progress(_state(inflated)) < (
        ALNSConfig.LIGHT_INTENSIFY_MIN_PROGRESS
    )
    halfway = _Tracker(iteration=4_700, max_iterations=900_200, elapsed=0.5 * float(ALNSConfig.MAX_RUNTIME))
    assert accept._resolve_progress(_state(halfway)) == pytest.approx(0.5, abs=0.02)


# ---------------------------------------------------------------------------
# Missing tracker
# ---------------------------------------------------------------------------


def test_no_tracker_at_all_reports_zero_progress() -> None:
    accept = _accept()

    assert accept._resolve_progress(SimpleNamespace()) == 0.0
