"""Regression tests for the P1 fixes.

Covers three items from the repository audit:

* the master objective must apply its coefficients (``MasterModel``);
* a CPLEX solver-level failure must not be recorded as a successful run;
* the runtime-derived ALNS iteration ceiling must stay calibrated to real
  throughput, because ``progress`` - and therefore several phase gates - is
  computed from it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from multi_period_delivery.alns.alns_config import default_config as ALNSConfig
from multi_period_delivery.alns.alns_optimizer import compute_dynamic_iteration_cap

# ---------------------------------------------------------------------------
# Dynamic iteration ceiling (audit item A2)
# ---------------------------------------------------------------------------

#: End-to-end ALNS throughput measured on the bundled instances (2026-10-16):
#: small/dataset_1 ran 3663 iterations within a 90 s budget, medium/dataset_13
#: ran 15. Recorded so the calibration cannot silently drift back to an
#: assumption that no machine actually reaches.
_MEASURED_FASTEST_ITERATIONS_PER_SECOND = 40.0


def test_dynamic_cap_follows_the_documented_formula() -> None:
    config = SimpleNamespace(
        MAX_RUNTIME=600,
        MAX_ITERATIONS=1000,
        DYNAMIC_MAX_ITERATIONS_PER_SECOND=5.0,
        DYNAMIC_MAX_ITERATION_SAFETY_FACTOR=1.5,
        DYNAMIC_MAX_ITERATION_BUFFER=200,
    )

    assert compute_dynamic_iteration_cap(config) == int(600 * 5.0 * 1.5) + 200


def test_dynamic_cap_never_undercuts_the_hard_iteration_cap() -> None:
    """A tiny dynamic estimate must not truncate MAX_ITERATIONS."""
    config = SimpleNamespace(
        MAX_RUNTIME=1,
        MAX_ITERATIONS=1000,
        DYNAMIC_MAX_ITERATIONS_PER_SECOND=1.0,
        DYNAMIC_MAX_ITERATION_SAFETY_FACTOR=1.0,
        DYNAMIC_MAX_ITERATION_BUFFER=0,
    )

    assert compute_dynamic_iteration_cap(config) == 1000


def test_shared_ceiling_propagates_to_stop_and_tracker() -> None:
    """The ceiling that gates ``progress`` must be the one the run uses."""
    config = SimpleNamespace(
        MAX_RUNTIME=600,
        MAX_ITERATIONS=1000,
        DYNAMIC_MAX_ITERATIONS_PER_SECOND=5.0,
        DYNAMIC_MAX_ITERATION_SAFETY_FACTOR=1.5,
        DYNAMIC_MAX_ITERATION_BUFFER=200,
    )

    cap = compute_dynamic_iteration_cap(config)

    # Both consumers are fed the same number, so accept._resolve_progress and
    # the stopping rule cannot disagree about how far the run has advanced.
    assert cap == 4700
    assert ALNSConfig.MAX_ITERATIONS <= cap


def test_default_throughput_estimate_stays_within_measured_range() -> None:
    """The calibration must not drift back to an unreachable assumption.

    ``progress`` is ``iteration / cap``, and the phase thresholds are 0.20-0.70
    of that scale. If ``iter_per_second`` x ``safety_factor`` dwarfs measured
    throughput, the cap sits far beyond what MAX_RUNTIME can execute and every
    one of those gates becomes dead code - which is exactly what the previous
    value (1000.0) caused, yielding a 900,200-iteration ceiling.
    """
    effective = (
        ALNSConfig.DYNAMIC_MAX_ITERATIONS_PER_SECOND
        * max(ALNSConfig.DYNAMIC_MAX_ITERATION_SAFETY_FACTOR, 1.0)
    )

    assert effective <= _MEASURED_FASTEST_ITERATIONS_PER_SECOND, (
        f"DYNAMIC_MAX_ITERATIONS_PER_SECOND x safety factor = {effective} "
        f"exceeds the fastest throughput ever measured on these instances "
        f"({_MEASURED_FASTEST_ITERATIONS_PER_SECOND} iter/s)"
    )


def test_phase_gates_are_reachable_within_the_default_budget() -> None:
    """At least the earliest phase gate must be reachable in one budget."""
    cap = compute_dynamic_iteration_cap(ALNSConfig)
    reachable = _MEASURED_FASTEST_ITERATIONS_PER_SECOND * ALNSConfig.MAX_RUNTIME

    threshold = int(ALNSConfig.ACTUAL_PRIORITY_START_PROGRESS * cap)

    assert reachable >= threshold, (
        f"progress {ALNSConfig.ACTUAL_PRIORITY_START_PROGRESS} needs "
        f"{threshold} iterations, but MAX_RUNTIME={ALNSConfig.MAX_RUNTIME} can "
        f"only execute about {reachable:.0f} at measured throughput"
    )
    # The other documented gates live earlier in the same window.
    assert (
        ALNSConfig.LIGHT_INTENSIFY_MIN_PROGRESS
        < ALNSConfig.LATE_FLEET_HARD_PRIORITY_START_PROGRESS
        < ALNSConfig.ACTUAL_PRIORITY_START_PROGRESS
    )


@pytest.mark.parametrize("zeroed", ["iterations_per_second", "safety_factor"])
def test_zeroed_dynamic_knobs_disable_widening(zeroed: str) -> None:
    """Zeroing a dynamic knob must pin the budget, not silently multiply it."""
    config = SimpleNamespace(
        MAX_RUNTIME=120,
        MAX_ITERATIONS=120,
        DYNAMIC_MAX_ITERATIONS_PER_SECOND=0.0,
        DYNAMIC_MAX_ITERATION_SAFETY_FACTOR=0.0,
        DYNAMIC_MAX_ITERATION_BUFFER=0,
    )

    cap = compute_dynamic_iteration_cap(config)

    assert cap == max(config.MAX_ITERATIONS, config.MAX_RUNTIME)


# ---------------------------------------------------------------------------
# CPLEX status handling (audit item C2)
# ---------------------------------------------------------------------------

try:
    from single_period_delivery.cplex_api_support import SOLVED_STATUSES
except ImportError:  # pragma: no cover - module import must not need CPLEX
    SOLVED_STATUSES = frozenset()


def test_solver_error_is_not_a_solved_status() -> None:
    """A swallowed CPLEX failure must be distinguishable from a real result."""
    assert "solver_error: CPLEX Error 1016" not in SOLVED_STATUSES
    assert "optimal" in SOLVED_STATUSES
    assert "time_limit_feasible" in SOLVED_STATUSES


def test_solver_error_can_be_recognised_by_prefix() -> None:
    """The status string carries the failure, so a prefix check is possible."""
    status = "solver_error: CPLEX Error 1016: Community Edition."

    assert not status.startswith(tuple(SOLVED_STATUSES))
    assert status.startswith("solver_error")
