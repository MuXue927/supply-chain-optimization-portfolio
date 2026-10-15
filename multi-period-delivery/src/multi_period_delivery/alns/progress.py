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
parameter decay         ``ParamAutoTuner`` (normalized over [0, 0.7])
epsilon-greedy decay    ``ML_EPSILON_BASE`` / ``..._EARLY_BOOST``
======================  =======================================

A fraction of the budget must mean the same thing on every machine, so the
measure is **elapsed runtime / MAX_RUNTIME**. A machine twice as fast executes
twice the iterations in the same fraction of the wall clock, so the thresholds
fire at the same point of the run.

The alternative - iterations / iteration-ceiling - silently depends on an
assumed throughput, and throughput is both machine- and instance-specific
(measured on one host: 0.1-40 iterations/second across instance sizes). Sizing
the ceiling with such a constant made ``progress`` stall near zero and turned
every threshold above into dead code.

Two call sites are kept working:

* a *tracker* that records ``start_time`` (the normal case);
* an *iteration counter*, used as a fallback when no time budget or start time
  is available - and by ``ParamAutoTuner``, which tracks counts only.

The iteration fallback is intentional: it keeps every caller well-defined rather
than failing, but it is a degraded measure and the code paths that matter record
a start time.
"""

from __future__ import annotations

import time
from typing import Any

from .alns_config import default_config as ALNSConfig


def _positive_float(value: Any) -> float | None:
    """Return ``value`` as a positive float, or None when unusable."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def elapsed_progress(tracker: Any) -> float | None:
    """
    Progress of ``tracker`` as elapsed-runtime / ``MAX_RUNTIME``.

    Returns None when there is no usable time budget or the tracker does not
    record a start time, so callers can fall back to the iteration measure.
    """
    if not bool(getattr(ALNSConfig, "DYNAMIC_PROGRESS_USE_ELAPSED_TIME", True)):
        return None
    budget = _positive_float(getattr(ALNSConfig, "MAX_RUNTIME", 0))
    if budget is None or budget <= 0:
        return None
    start_time = _positive_float(getattr(tracker, "start_time", 0.0))
    if start_time is None or start_time <= 0:
        return None
    return max(0.0, min(1.0, (time.time() - start_time) / budget))


def iteration_progress(iteration: Any, max_iterations: Any) -> float | None:
    """Progress as iteration / ceiling, or None when the ceiling is unusable."""
    current = _positive_float(iteration)
    ceiling = _positive_float(max_iterations)
    if ceiling is None or ceiling <= 0:
        return None
    if current is None:
        current = 0.0
    return max(0.0, min(1.0, current / ceiling))


def resolve_progress(states: Any = (), *, tracker: Any = None) -> float:
    """
    Best available progress measure for the current search state.

    ``states`` mirrors the call shape of the acceptance policy: the first object
    exposing a ``tracker`` wins. ``tracker`` accepts a tracker directly, for
    callers that already hold one.
    """
    candidates: list[Any] = []
    if tracker is not None:
        candidates.append(tracker)
    for state in states or ():
        found = getattr(state, "tracker", None)
        if found is not None:
            candidates.append(found)

    for candidate in candidates:
        fraction = elapsed_progress(candidate)
        if fraction is not None:
            return fraction
    for candidate in candidates:
        fraction = iteration_progress(
            getattr(candidate, "iteration", 0),
            getattr(candidate, "max_iterations", 0),
        )
        if fraction is not None:
            return fraction
    return 0.0
