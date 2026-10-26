"""Tests for the shared search-progress measure.

``alns.progress`` is the single source of truth for "how far through the run are
we?", consumed by:

* ``accept.DemandConstraintAccept`` - the phase gates for acceptance;
* ``ml_operator_selector.MLOperatorSelector`` - ML takeover and the
  epsilon-greedy exploration decay;
* ``param_tuner.ParamAutoTuner`` - the destroy-strength decay schedule;
* ``callbacks`` - the light-intensify window.

All of them compare against a fraction of the budget, so the measure must be
independent of the host's speed. The module has deliberately no fallback
measure: a tracker without a clock, or a non-positive budget, is an error.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from multi_period_delivery.alns.alns_config import default_config as ALNSConfig
from multi_period_delivery.alns.progress import (
    elapsed_progress,
    iteration_progress,
    progress_from_elapsed,
    resolve_progress,
    search_progress,
)


def _tracker(*, elapsed: float | None = None, with_clock: bool = False):
    """Tracker stand-in.

    ``with_clock`` models ``ALNSTracker``, which reports elapsed seconds itself;
    otherwise the monotonic origin is read directly.
    """
    data: dict[str, object] = {}
    if with_clock:
        data["elapsed_seconds"] = (lambda: elapsed or 0.0)
    else:
        data["budget_start_time"] = time.perf_counter() - (elapsed or 0.0)
    return SimpleNamespace(**data)


# ---------------------------------------------------------------------------
# The measure itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fraction", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_progress_follows_the_runtime_fraction(fraction: float) -> None:
    budget = float(ALNSConfig.MAX_RUNTIME)

    assert elapsed_progress(_tracker(elapsed=fraction * budget)) == pytest.approx(
        fraction, abs=0.02
    )


def test_progress_reads_the_trackers_own_clock() -> None:
    """A tracker that reports elapsed seconds is used directly."""
    budget = float(ALNSConfig.MAX_RUNTIME)
    tracker = _tracker(elapsed=0.5 * budget, with_clock=True)

    assert elapsed_progress(tracker) == pytest.approx(0.5, abs=0.01)


def test_progress_is_clamped() -> None:
    budget = float(ALNSConfig.MAX_RUNTIME)

    # Lower bound: time keeps moving between building the tracker and reading it,
    # so "just started" is a few nanoseconds of progress rather than exactly 0.
    started = elapsed_progress(_tracker(elapsed=0.0))
    assert 0.0 <= started < 1e-6

    # Upper bound: an overrun run reports exactly 1.0 and never beyond.
    assert elapsed_progress(_tracker(elapsed=10 * budget)) == 1.0


def test_progress_from_elapsed_is_the_single_mapping() -> None:
    budget = float(ALNSConfig.MAX_RUNTIME)

    assert progress_from_elapsed(0.0) == 0.0
    assert progress_from_elapsed(0.25 * budget) == pytest.approx(0.25)
    assert progress_from_elapsed(budget) == 1.0
    assert progress_from_elapsed(2 * budget) == 1.0
    assert progress_from_elapsed(-1.0) == 0.0


# ---------------------------------------------------------------------------
# Loud failure instead of a silent fallback
# ---------------------------------------------------------------------------


def test_non_positive_runtime_budget_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero/negative budget is a configuration error, not a mode switch."""
    for bad in (0, -1):
        monkeypatch.setattr(ALNSConfig, "MAX_RUNTIME", bad, raising=False)
        with pytest.raises(ValueError, match="MAX_RUNTIME"):
            progress_from_elapsed(1.0)


def test_tracker_without_any_clock_is_rejected() -> None:
    """A tracker with no elapsed source is a programming error, not 0.0."""
    with pytest.raises(AttributeError, match="budget_start_time"):
        elapsed_progress(SimpleNamespace())


# ---------------------------------------------------------------------------
# resolve_progress
# ---------------------------------------------------------------------------


def test_resolve_reads_the_state_tracker() -> None:
    budget = float(ALNSConfig.MAX_RUNTIME)
    state = SimpleNamespace(tracker=_tracker(elapsed=0.5 * budget))

    assert resolve_progress((state,)) == pytest.approx(0.5, abs=0.02)


def test_resolve_accepts_a_bare_tracker() -> None:
    budget = float(ALNSConfig.MAX_RUNTIME)

    assert resolve_progress(
        tracker=_tracker(elapsed=0.25 * budget)
    ) == pytest.approx(0.25, abs=0.02)


def test_resolve_returns_zero_before_the_search_starts() -> None:
    """No tracker means no budget has been consumed yet."""
    assert resolve_progress(()) == 0.0
    assert resolve_progress((SimpleNamespace(),)) == 0.0


# ---------------------------------------------------------------------------
# Iteration basis (fixed-iteration experiments only)
# ---------------------------------------------------------------------------


def test_iteration_basis_counts_iterations_not_seconds(monkeypatch) -> None:
    """Capping iterations alone still leaks wall clock; this removes the leak.

    Under the runtime basis two modes at the same iteration have different
    progress if their per-iteration cost differs, which moves the SA cooling,
    the phase gates and the ML takeover threshold. The iteration basis makes the
    schedule identical for the same iteration count.
    """
    monkeypatch.setattr(ALNSConfig, "PROGRESS_BASIS", "iteration")
    fast = SimpleNamespace(iteration=50, max_iterations=200)
    slow = SimpleNamespace(iteration=50, max_iterations=200)

    assert search_progress(fast) == pytest.approx(0.25)
    assert search_progress(slow) == pytest.approx(0.25)
    assert resolve_progress((SimpleNamespace(tracker=slow),)) == pytest.approx(0.25)

    monkeypatch.setattr(ALNSConfig, "PROGRESS_BASIS", "runtime")
    timed = _tracker(elapsed=float(ALNSConfig.MAX_RUNTIME) * 0.9)
    assert search_progress(timed) == pytest.approx(0.9, abs=0.02)
    assert iteration_progress(fast) == pytest.approx(0.25)


def test_iteration_basis_clamps_and_rejects_bad_budgets(monkeypatch) -> None:
    monkeypatch.setattr(ALNSConfig, "PROGRESS_BASIS", "iteration")

    assert iteration_progress(
        SimpleNamespace(iteration=500, max_iterations=200)
    ) == 1.0
    assert iteration_progress(
        SimpleNamespace(iteration=-3, max_iterations=200)
    ) == 0.0
    with pytest.raises(ValueError, match="positive planned iteration"):
        iteration_progress(SimpleNamespace(iteration=1, max_iterations=0))
    with pytest.raises(AttributeError, match="tracker.iteration"):
        iteration_progress(SimpleNamespace())


def test_unknown_progress_basis_is_a_configuration_error(monkeypatch) -> None:
    """Falling back would silently redefine every progress threshold."""
    monkeypatch.setattr(ALNSConfig, "PROGRESS_BASIS", "walltime")

    with pytest.raises(ValueError, match="PROGRESS_BASIS"):
        search_progress(SimpleNamespace(iteration=1, max_iterations=10))
