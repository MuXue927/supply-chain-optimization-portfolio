"""Tests for the shared search-progress measure.

``alns.progress`` is the single source of truth for "how far through the run are
we?", consumed by three policies:

* ``accept.DemandConstraintAccept`` - the phase gates for acceptance;
* ``ml_operator_selector.MLOperatorSelector`` - ML takeover and the
  epsilon-greedy exploration decay;
* ``param_tuner.ParamAutoTuner`` - the destroy-strength decay schedule.

All of them compare against a fraction of the budget, so the measure must be
independent of the host's speed. See the module docstring for why the
iteration-based measure cannot be.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from multi_period_delivery.alns.alns_config import default_config as ALNSConfig
from multi_period_delivery.alns.progress import (
    elapsed_progress,
    iteration_progress,
    resolve_progress,
)


def _tracker(*, elapsed: float | None = None, iteration: int = 0, ceiling: int = 0):
    data: dict[str, object] = {"iteration": iteration, "max_iterations": ceiling}
    if elapsed is not None:
        data["start_time"] = time.time() - elapsed
    return SimpleNamespace(**data)


# ---------------------------------------------------------------------------
# Elapsed-time measure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fraction", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_elapsed_progress_follows_the_runtime_fraction(fraction: float) -> None:
    budget = float(ALNSConfig.MAX_RUNTIME)

    assert elapsed_progress(_tracker(elapsed=fraction * budget)) == pytest.approx(
        fraction, abs=0.02
    )


def test_elapsed_progress_is_clamped() -> None:
    budget = float(ALNSConfig.MAX_RUNTIME)

    # Lower bound: time keeps moving between building the tracker and reading it,
    # so "just started" is a few nanoseconds of progress rather than exactly 0.
    # What matters is that a fresh run reports ~0 and never goes negative.
    started = elapsed_progress(_tracker(elapsed=0.0))
    assert 0.0 <= started < 1e-6

    # Upper bound: an overrun run reports exactly 1.0 and never beyond.
    assert elapsed_progress(_tracker(elapsed=10 * budget)) == 1.0
    assert elapsed_progress(_tracker(elapsed=budget)) <= 1.0


def test_elapsed_progress_is_none_without_a_start_time() -> None:
    assert elapsed_progress(_tracker()) is None


def test_elapsed_progress_is_none_without_a_time_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ALNSConfig, "MAX_RUNTIME", 0, raising=False)

    assert elapsed_progress(_tracker(elapsed=5.0)) is None


def test_elapsed_progress_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ALNSConfig, "DYNAMIC_PROGRESS_USE_ELAPSED_TIME", False, raising=False
    )

    assert elapsed_progress(_tracker(elapsed=5.0)) is None


# ---------------------------------------------------------------------------
# Iteration fallback
# ---------------------------------------------------------------------------


def test_iteration_progress_is_the_ratio() -> None:
    assert iteration_progress(25, 100) == pytest.approx(0.25)
    assert iteration_progress(200, 100) == 1.0
    assert iteration_progress(-5, 100) == 0.0


@pytest.mark.parametrize("ceiling", [0, None, "many"])
def test_iteration_progress_is_none_for_an_unusable_ceiling(ceiling: object) -> None:
    assert iteration_progress(10, ceiling) is None


# ---------------------------------------------------------------------------
# resolve_progress: precedence
# ---------------------------------------------------------------------------


def test_resolve_prefers_elapsed_time_over_iterations() -> None:
    """Same state, two very different iteration fractions, one answer."""
    budget = float(ALNSConfig.MAX_RUNTIME)
    state = SimpleNamespace(
        tracker=_tracker(elapsed=0.5 * budget, iteration=1, ceiling=1_000_000)
    )

    assert resolve_progress((state,)) == pytest.approx(0.5, abs=0.02)


def test_resolve_falls_back_to_iterations() -> None:
    state = SimpleNamespace(tracker=_tracker(iteration=80, ceiling=100))

    assert resolve_progress((state,)) == pytest.approx(0.8)


def test_resolve_returns_zero_without_any_information() -> None:
    assert resolve_progress(()) == 0.0
    assert resolve_progress((SimpleNamespace(),)) == 0.0


def test_resolve_accepts_a_bare_tracker() -> None:
    assert resolve_progress(tracker=_tracker(iteration=30, ceiling=60)) == pytest.approx(
        0.5
    )
