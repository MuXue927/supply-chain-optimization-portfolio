"""
Search-progress measure shared by the adaptive ALNS policies.

Several policies read "how far through the run are we?" and switch behaviour at
a documented fraction of the budget:

======================  =======================================
Policy                  Threshold
======================  =======================================
light intensification   ``LIGHT_INTENSIFY_MIN_PROGRESS`` = 0.20
late fleet priority     ``LATE_FLEET_HARD_PRIORITY_START_PROGRESS`` = 0.60
actual-priority accept  ``ACTUAL_PRIORITY_START_PROGRESS`` = 0.70
elite relinking window  ``ELITE_RELINK_MIN/MAX_PROGRESS`` = 0.15 / 0.95
ML takeover             ``ML_TAKEOVER_MIN_PROGRESS`` = 0.30
parameter decay         ``ParamAutoTuner.adaptive_degree`` (first 70%)
epsilon-greedy decay    ``ML_EPSILON_BASE`` / ``..._EARLY_BOOST``
ML warm-up budget       ``ML_WARMUP_BUDGET_FRACTION`` = 0.15
======================  =======================================

A fraction of the budget must mean the same thing on every machine, so progress
is **elapsed seconds / MAX_RUNTIME**. A machine twice as fast executes twice the
iterations in the same fraction of the wall clock, so the thresholds fire at the
same point of the run.

An iteration-based measure was tried and removed: it needs an assumed
throughput, and throughput is both machine- and instance-specific (measured on
one host: 0.1-40 iterations/second), so no constant can be correct everywhere.

It is still available behind ``PROGRESS_BASIS = "iteration"`` for one purpose:
**fixed-iteration paired experiments**. Capping ``MAX_ITERATIONS`` alone does not
remove wall clock from the comparison, because the slower mode then has a higher
elapsed/``MAX_RUNTIME`` value at the same iteration. The switch is off by
default and must not be turned on for production runs.

**Every schedule in the table must dispatch through this module.** A schedule
that keeps its own clock is invisible in the table and reintroduces exactly the
dependency the switch exists to remove: ``ParamAutoTuner.adaptive_degree`` did
that until M70, and the same full/seed finished at 782.67 and 799.63 on a
fixed-iteration protocol. New phase gates or decay schedules should call
``search_progress`` / ``resolve_progress`` (or, where only iteration counts are
available, ``iteration_progress_from_counts``) rather than reading
``time`` directly.

**One clock, one origin.** ``tracker.start_time`` is the only origin, recorded
on the monotonic clock (``time.perf_counter``) so neither a wall-clock step nor
a second origin can shift the schedule. ``ALNSTracker`` sets it when the run
starts; every other participant reads elapsed time from the tracker.

**One failure mode, and it is loud.** A non-positive ``MAX_RUNTIME`` is a
configuration error, so this module raises instead of silently switching to a
different measure.
"""

from __future__ import annotations

import time
from typing import Any

from .alns_config import default_config as ALNSConfig

#: Attribute a progress tracker exposes: the run's monotonic origin.
#:
#: Deliberately separate from ``ALNSTracker.start_time``, which is a wall-clock
#: stamp used for log display (``time.time() - start_time``). Mixing the two
#: clocks is exactly the kind of drift this module exists to prevent.
BUDGET_START_ATTRIBUTE = "budget_start_time"


def _require_positive_runtime_budget() -> float:
    """Return ``MAX_RUNTIME``, or raise when it cannot describe a budget."""
    budget = float(ALNSConfig.MAX_RUNTIME)
    if not budget > 0:
        raise ValueError(
            "MAX_RUNTIME must be a positive number of seconds to measure search "
            f"progress; got {budget!r}. Fix the configuration (or the value set "
            "by --max-runtime) instead of relying on a fallback measure."
        )
    return budget


def progress_from_elapsed(elapsed_seconds: float) -> float:
    """Map elapsed seconds onto the [0, 1] progress fraction."""
    budget = _require_positive_runtime_budget()
    return max(0.0, min(1.0, float(elapsed_seconds) / budget))


def elapsed_progress(tracker: Any) -> float:
    """
    Progress of ``tracker`` as elapsed-seconds / ``MAX_RUNTIME``.

    Elapsed time comes from ``tracker.elapsed_seconds()`` when the tracker
    measures it (``ALNSTracker`` does, from the run's clock), otherwise from the
    ``budget_start_time`` monotonic origin that tracker exposes. Both are the
    monotonic clock from the same origin, so a missing attribute is a
    programming error and surfaces as an ``AttributeError`` rather than
    silently degrading the measure.
    """
    elapsed_seconds = getattr(tracker, "elapsed_seconds", None)
    if callable(elapsed_seconds):
        return progress_from_elapsed(float(elapsed_seconds()))
    start = float(getattr(tracker, BUDGET_START_ATTRIBUTE))
    return progress_from_elapsed(time.perf_counter() - start)


def iteration_progress(tracker: Any) -> float:
    """Progress of ``tracker`` as completed iterations / planned iterations.

    Only meaningful when the run has a real iteration budget: with a runtime
    stop the planned count is an upper bound, so this measure would never reach
    1.0. ``PROGRESS_BASIS = "iteration"`` exists for fixed-iteration paired
    experiments, where the alternative - elapsed / ``MAX_RUNTIME`` - gives the
    slower mode a higher progress value at the same iteration and lets wall
    clock back into a comparison meant to exclude it.
    """
    completed = getattr(tracker, "iteration", None)
    planned = getattr(tracker, "max_iterations", None)
    if completed is None or planned is None:
        # A silent 0.0 would switch every progress-gated policy off, which is
        # worse than failing: the run would look fine and behave differently.
        raise AttributeError(
            "PROGRESS_BASIS='iteration' needs tracker.iteration and "
            f"tracker.max_iterations; got {type(tracker).__name__}"
        )
    return iteration_progress_from_counts(int(completed), int(planned))


def iteration_progress_from_counts(completed: int, planned: int) -> float:
    """Iteration progress from bare counts, for callers without a tracker.

    ``ParamAutoTuner`` already holds the current and planned iteration counts,
    so it uses this entry point instead of keeping a second definition of the
    measure. Same clamping and same error surface as ``iteration_progress``.
    """
    planned_int = int(planned)
    if planned_int <= 0:
        raise ValueError(
            "PROGRESS_BASIS='iteration' needs a positive planned iteration "
            f"count; got {planned!r}"
        )
    return max(0.0, min(1.0, float(completed) / float(planned_int)))


def basis_name() -> str:
    """The configured progress basis, validated.

    An unknown value is a configuration error: falling back would silently
    change what every threshold in the table above means.
    """
    basis = str(getattr(ALNSConfig, "PROGRESS_BASIS", "runtime")).strip().lower()
    if basis not in {"runtime", "iteration"}:
        raise ValueError(
            "PROGRESS_BASIS must be 'runtime' or 'iteration'; got "
            f"{ALNSConfig.PROGRESS_BASIS!r}"
        )
    return basis


def search_progress(tracker: Any) -> float:
    """Progress of ``tracker`` under the configured basis."""
    if basis_name() == "iteration":
        return iteration_progress(tracker)
    return elapsed_progress(tracker)


def resolve_progress(states: Any = (), *, tracker: Any = None) -> float:
    """
    Progress for the current search state.

    ``states`` mirrors the call shape of the acceptance policy: the first object
    exposing a ``tracker`` wins. ``tracker`` accepts a tracker directly, for
    callers that already hold one.

    A state without a tracker, or with none supplied at all, yields 0.0: that
    happens before the search starts, when no budget has been consumed yet.
    """
    if tracker is not None:
        return search_progress(tracker)
    for state in states or ():
        found = getattr(state, "tracker", None)
        if found is not None:
            return search_progress(found)
    return 0.0
