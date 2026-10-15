"""Tests for the swallowed-exception observability helper."""

from __future__ import annotations

import logging

import pytest

from multi_period_delivery.alns.swallowed import (
    record_swallowed,
    report_swallowed,
    reset_swallowed_counters,
    swallowed_summary,
)


@pytest.fixture(autouse=True)
def _clean_counters():
    reset_swallowed_counters()
    yield
    reset_swallowed_counters()


def test_records_and_counts_each_site() -> None:
    record_swallowed("op.alpha", ValueError("a"))
    record_swallowed("op.alpha", ValueError("b"))
    record_swallowed("op.beta", KeyError("c"))

    assert swallowed_summary() == {"op.alpha": 2, "op.beta": 1}


def test_logs_only_the_first_occurrence_per_site(caplog) -> None:
    """A hot-path site must not flood the log."""
    with caplog.at_level(logging.WARNING):
        for index in range(5):
            record_swallowed("op.noisy", ValueError(f"n{index}"))

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "op.noisy" in warnings[0].getMessage()
    # Still counted every time.
    assert swallowed_summary() == {"op.noisy": 5}


def test_distinct_sites_each_log_once(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        record_swallowed("op.one", ValueError("x"))
        record_swallowed("op.two", ValueError("y"))

    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 2


def test_summary_is_a_copy() -> None:
    record_swallowed("op.alpha", ValueError("a"))

    snapshot = swallowed_summary()
    snapshot["op.alpha"] = 999

    assert swallowed_summary() == {"op.alpha": 1}


def test_report_is_silent_when_nothing_was_swallowed(caplog) -> None:
    with caplog.at_level(logging.INFO):
        summary = report_swallowed()

    assert summary == {}
    assert caplog.records == []


def test_report_lists_sites_most_frequent_first(caplog) -> None:
    record_swallowed("op.rare", ValueError("a"))
    for _ in range(3):
        record_swallowed("op.common", ValueError("b"))

    # Without a log_printer the summary goes to the logger at INFO level.
    with caplog.at_level(logging.INFO):
        summary = report_swallowed()

    assert summary == {"op.rare": 1, "op.common": 3}
    message = next(
        r.getMessage()
        for r in caplog.records
        if "Swallowed-exception summary" in r.getMessage()
    )
    assert message.index("op.common") < message.index("op.rare")


def test_report_uses_the_run_log_printer_when_given() -> None:
    class _Printer:
        def __init__(self) -> None:
            self.messages: list[str] = []

        def print(self, message: str, **_: object) -> None:
            self.messages.append(message)

    printer = _Printer()
    record_swallowed("op.alpha", ValueError("boom"))

    report_swallowed(printer)

    assert len(printer.messages) == 1
    assert "op.alpha" in printer.messages[0]
