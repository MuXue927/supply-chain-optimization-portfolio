"""
Observability for deliberately swallowed exceptions.

Many handlers in this package exist to keep one local failure from aborting a
whole search (a single candidate, an optional dependency, a best-effort cache
refresh). That is a legitimate design, but a handler that only does ``pass``
turns a defect into a silent behaviour change: the search keeps running and the
result looks healthy while a capability has quietly switched off.

``record_swallowed`` keeps the degradation, but makes it visible:

* the first failure at each *site* is logged with a traceback, so the cause is
  recoverable from the logs;
* every failure is counted, so the end of a run can report how often each site
  degraded instead of the operator silently becoming a no-op;
* repeat failures at the same site are counted but not re-logged, which keeps
  hot paths quiet.

Sites are identified by a stable string chosen by the caller, not by line
number, so the counter survives edits.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_counts: dict[str, int] = {}
_logged: set[str] = set()

#: Guard against a pathological run producing an unbounded log volume.
_MAX_LOGGED_SITES = 200


def record_swallowed(site: str, exc: BaseException) -> None:
    """
    Count a swallowed exception and log its first occurrence at ``site``.

    Parameters
    ----------
    site :
        Stable identifier, for example ``"repair_operators.candidate_eval"``.
    exc :
        The exception being swallowed.
    """
    with _lock:
        seen = _counts.get(site, 0)
        _counts[site] = seen + 1
        first = site not in _logged
        if first:
            _logged.add(site)

    if first:
        if len(_logged) <= _MAX_LOGGED_SITES:
            logger.warning(
                "swallowed an exception at %s (further occurrences are counted "
                "only): %s",
                site,
                exc,
                exc_info=True,
            )
        else:
            logger.warning(
                "swallowed an exception at %s (logging budget for distinct "
                "sites exhausted): %s",
                site,
                exc,
            )
    else:
        logger.debug("swallowed again at %s: %s", site, exc)


def swallowed_summary() -> dict[str, int]:
    """Return a copy of ``{site: times swallowed}`` for reporting."""
    with _lock:
        return dict(_counts)


def reset_swallowed_counters() -> None:
    """Clear the counters. Intended for tests."""
    with _lock:
        _counts.clear()
        _logged.clear()


def report_swallowed(log_printer: Any | None = None) -> dict[str, int]:
    """
    Summarise swallowed exceptions, and return the counts.

    Prints through ``log_printer`` when one is supplied so the run's own output
    carries the information; otherwise the caller can log the returned mapping.
    """
    summary = swallowed_summary()
    if not summary:
        return summary
    total = sum(summary.values())
    lines = [
        f"Swallowed-exception summary: {total} occurrence(s) across "
        f"{len(summary)} site(s)."
    ]
    for site, count in sorted(summary.items(), key=lambda item: -item[1]):
        lines.append(f"  {count:>6}  {site}")
    message = "\n".join(lines)
    if log_printer is not None:
        log_printer.print(message)
    else:
        logger.info(message)
    return summary
