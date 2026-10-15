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
parameter decay         ``ParamAutoTuner`` (normalized over the first 70%)
epsilon-greedy decay    ``ML_EPSILON_BASE`` / ``..._EARLY_BOOST``
======================  =======================================

A fraction of the budget must mean the same thing on every machine, so progress
is **elapsed seconds / MAX_RUNTIME**. A machine twice as fast executes twice the
iterations in the same fraction of the wall clock, so the thresholds fire at the
same point of the run.

An iteration-based measure was tried and removed: it needs an assumed
throughput, and throughput is both machine- and instance-specific (measured on
one host: 0.1-40 iterations/second), so no constant can be correct everywhere.

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
        return elapsed_progress(tracker)
    for state in states or ():
        found = getattr(state, "tracker", None)
        if found is not None:
            return elapsed_progress(found)
    return 0.0
