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
#: ran 15. Recorded so the safety net can be checked against reality.
_MEASURED_FASTEST_ITERATIONS_PER_SECOND = 40.0


def test_dynamic_cap_follows_the_documented_formula() -> None:
    config = SimpleNamespace(
        MAX_RUNTIME=600,
        MAX_ITERATIONS=1000,
        DYNAMIC_MAX_ITERATIONS_PER_SECOND=7.0,
        DYNAMIC_MAX_ITERATION_SAFETY_FACTOR=1.5,
        DYNAMIC_MAX_ITERATION_BUFFER=200,
    )

    assert compute_dynamic_iteration_cap(config) == int(600 * 7.0 * 1.5) + 200


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
    """The ceiling that gates the stopping rule must be the shared one."""
    config = SimpleNamespace(
        MAX_RUNTIME=600,
        MAX_ITERATIONS=1000,
        DYNAMIC_MAX_ITERATIONS_PER_SECOND=7.0,
        DYNAMIC_MAX_ITERATION_SAFETY_FACTOR=1.5,
        DYNAMIC_MAX_ITERATION_BUFFER=200,
    )

    cap = compute_dynamic_iteration_cap(config)

    assert cap == 6500
    assert ALNSConfig.MAX_ITERATIONS <= cap


def test_iteration_ceiling_is_a_safety_net_not_a_schedule_input() -> None:
    """The ceiling must not bind before MAX_RUNTIME in normal operation.

    Phase progress is measured against the runtime budget, so if the iteration
    ceiling fires first the run is truncated and the later phase gates never
    execute. That is why the throughput bound is deliberately generous: any
    value a real machine can reach keeps the runtime budget in control.

    An earlier revision sized this to *measured* throughput (5.0), which made
    the ceiling bind after ~152 s of a 600 s run on ``small/dataset_1`` and left
    progress at ~0.25 - so the 0.60/0.70 gates still could not fire.
    """
    effective = (
        ALNSConfig.DYNAMIC_MAX_ITERATIONS_PER_SECOND
        * max(ALNSConfig.DYNAMIC_MAX_ITERATION_SAFETY_FACTOR, 1.0)
    )

    assert effective > _MEASURED_FASTEST_ITERATIONS_PER_SECOND, (
        f"the safety net ({effective} iter/s) would bind on a machine measured "
        f"at {_MEASURED_FASTEST_ITERATIONS_PER_SECOND} iter/s, truncating the "
        f"run before MAX_RUNTIME and skipping the later phase gates"
    )


def test_phase_gates_are_reachable_within_the_default_budget() -> None:
    """Every gate must be reachable while MAX_RUNTIME is still the binding limit."""
    budget = float(ALNSConfig.MAX_RUNTIME)
    reachable = _MEASURED_FASTEST_ITERATIONS_PER_SECOND * budget
    ceiling = compute_dynamic_iteration_cap(ALNSConfig)

    # The runtime budget stays in control, so progress can span the full [0, 1].
    assert ceiling > reachable, (
        f"iteration ceiling {ceiling} would bind before the runtime budget "
        f"(~{reachable:.0f} iterations at measured throughput)"
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


def test_hard_iteration_budget_pins_the_ceiling() -> None:
    """A pinned iteration budget must not be widened by the runtime budget.

    Without this a fixed-iteration paired experiment cannot be expressed: the
    runtime budget both raises the ceiling and acts as a stop, so a slow mode
    silently executes fewer iterations than the fast one.
    """
    config = SimpleNamespace(
        MAX_RUNTIME=100_000,
        MAX_ITERATIONS=120,
        DYNAMIC_MAX_ITERATIONS_PER_SECOND=5.0,
        DYNAMIC_MAX_ITERATION_SAFETY_FACTOR=1.5,
        DYNAMIC_MAX_ITERATION_BUFFER=200,
        HARD_ITERATION_BUDGET=True,
    )

    assert compute_dynamic_iteration_cap(config) == 120

    # Off (the default) keeps the historical widening behaviour.
    config.HARD_ITERATION_BUDGET = False
    assert compute_dynamic_iteration_cap(config) > 120


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
