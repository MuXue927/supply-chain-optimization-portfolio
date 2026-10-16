"""``validate()`` must distinguish a failed check from a detected violation.

Before this, an exception raised while *measuring* a vehicle's load was recorded
under ``veh_over_load``:

    violations["veh_over_load"].append({"veh": veh, "error": err})

so a code defect was reported as "vehicle overloaded". Every consumer checks
``violations.get("veh_over_load")`` for truthiness, which meant the rollback path
in ``repair_utils.post_repair`` triggered with the wrong - and entirely
misleading - reason, and an audit of the violations pointed at the wrong cause.

The state must still be infeasible when a check cannot complete (an unverified
state is not a feasible one), but the reason is now its own category.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from multi_period_delivery.alns.alns_optimizer import ALNSOptimizer

BASE = Path(__file__).resolve().parents[1]


def _state() -> object:
    optimizer = ALNSOptimizer(
        input_file_loc=str(BASE / "datasets" / "multiple-periods" / "small"),
        output_file_loc=str(Path(tempfile.gettempdir()) / "_validate_keys_probe"),
    )
    optimizer.load_data("dataset_1")
    return optimizer.create_initial_solution()


# ---------------------------------------------------------------------------
# Healthy state
# ---------------------------------------------------------------------------


def test_feasible_state_reports_no_violations_of_any_kind() -> None:
    state = _state()

    feasible, violations = state.validate()

    assert feasible is True
    assert violations["veh_over_load"] == []
    assert violations["check_error"] == []


def test_check_error_key_always_exists() -> None:
    """Consumers read it directly, so it must not be missing on a clean state."""
    _, violations = _state().validate()

    assert "check_error" in violations


# ---------------------------------------------------------------------------
# A failed check is not an overload
# ---------------------------------------------------------------------------


def test_load_calculation_failure_is_recorded_as_a_check_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state()
    # ``compute_veh_load`` is called per vehicle while measuring the load. Making
    # it raise simulates the measurement itself failing.
    state.compute_veh_load = "not callable"

    feasible, violations = state.validate()

    # Not certified feasible: the load could not be measured.
    assert feasible is False
    # The new category carries it...
    assert violations["check_error"], "the failed check must be recorded"
    entry = violations["check_error"][0]
    assert entry["check"] == "vehicle_load"
    assert entry["veh_id"] is not None
    assert "TypeError" in entry["error"]
    # ...and the overload bucket stays empty, so nothing reports a false overload.
    assert violations["veh_over_load"] == []


def test_failed_check_does_not_look_like_an_overload_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Overload entries carry measurement data; a failed check must not mimic them."""
    state = _state()
    state.compute_veh_load = "not callable"

    _, violations = state.validate()

    for entry in violations["veh_over_load"]:
        assert "error" not in entry, (
            "an overload entry must describe a measured overload, not an error"
        )
