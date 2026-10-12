"""Focused tests for the ALNS callback helpers.

``callbacks`` wraps the third-party ALNS hooks and decides which candidate
becomes the reported best feasible solution. The helpers below are pure enough
to test directly: fleet profiling, candidate comparison, snapshotting, and the
acceptance of a candidate as the recorded best.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from multi_period_delivery.alns import callbacks as cb
from multi_period_delivery.alns.alns_config import default_config as ALNSConfig
from multi_period_delivery.alns.alnsopt import SolutionState
from multi_period_delivery.alns.initial_solution_utils import improved_initial_solution
from multi_period_delivery.alns.InputDataALNS import DataALNS


def _load_data() -> DataALNS:
    base = Path(__file__).resolve().parents[1]
    data = DataALNS(
        str(base / "datasets" / "multiple-periods" / "small"),
        str(base / "tmp"),
        "dataset_1",
    )
    data.load()
    return data


@pytest.fixture(scope="module")
def data() -> DataALNS:
    return _load_data()


@pytest.fixture(scope="module")
def feasible_solution(data: DataALNS) -> SolutionState:
    """A real feasible solution, built once and copied by each test."""
    return improved_initial_solution(SolutionState(data), np.random.default_rng(7))


class _DummyPrinter:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def print(self, message: str, **kwargs) -> None:
        self.messages.append(message)


class _DummyOptimizer:
    """Minimal stand-in exposing the attributes the callbacks touch."""

    def __init__(self) -> None:
        self.best_feasible_state: SolutionState | None = None
        self.best_feasible_obj: float = float("inf")
        self.tracker = None
        self.log_printer = _DummyPrinter()


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def test_callback_console_flag_is_a_bool() -> None:
    assert isinstance(cb._callback_console_enabled(), bool)


def test_safe_state_snapshot_copies_and_handles_none(
    feasible_solution: SolutionState,
) -> None:
    assert cb.safe_state_snapshot(None) is None

    snapshot = cb.safe_state_snapshot(feasible_solution)

    assert snapshot is not None
    assert snapshot is not feasible_solution
    assert snapshot.s_ikt == feasible_solution.s_ikt


def test_safe_float_falls_back_to_the_default() -> None:
    assert cb._safe_float(None) == float("inf")
    assert cb._safe_float("nope", default=7.0) == 7.0
    assert cb._safe_float("2.5") == 2.5


def test_get_capacity_threshold_returns_a_real_capacity(data: DataALNS) -> None:
    threshold = cb._get_capacity_threshold(data)

    assert 0.0 < threshold <= max(data.veh_type_cap.values())


def test_used_vehicle_count_matches_the_fleet(feasible_solution: SolutionState) -> None:
    expected = len(list(feasible_solution.iter_used_vehicles()))

    assert cb._used_vehicle_count(feasible_solution) == expected
    assert cb._used_vehicle_count(None) == 0


# ---------------------------------------------------------------------------
# Fleet profiling
# ---------------------------------------------------------------------------
def test_build_state_profile_reports_fleet_metrics(
    feasible_solution: SolutionState,
) -> None:
    profile = cb.build_state_profile(feasible_solution)

    assert set(profile) == {"vehicles", "load_ratio", "large_ratio", "fixed_cost"}
    assert profile["vehicles"] > 0
    assert profile["load_ratio"] >= 0.0
    assert 0.0 <= profile["large_ratio"] <= 1.0
    assert profile["fixed_cost"] > 0.0


def test_build_state_profile_is_zero_for_an_empty_state(data: DataALNS) -> None:
    profile = cb.build_state_profile(SolutionState(data))

    assert profile == {
        "vehicles": 0.0,
        "load_ratio": 0.0,
        "large_ratio": 0.0,
        "fixed_cost": 0.0,
    }


def test_profile_signature_is_deterministic(feasible_solution: SolutionState) -> None:
    profile = cb.build_state_profile(feasible_solution)

    first = cb._profile_signature(profile)
    second = cb._profile_signature(profile)

    assert first == second
    assert len(first) == 4


def test_vehicle_route_signature_is_deterministic(
    feasible_solution: SolutionState,
) -> None:
    vehicle = next(iter(feasible_solution.iter_used_vehicles()))

    assert cb._vehicle_route_signature(vehicle) == cb._vehicle_route_signature(vehicle)


# ---------------------------------------------------------------------------
# Candidate comparison
# ---------------------------------------------------------------------------
def test_is_candidate_better_prefers_a_lower_objective(
    feasible_solution: SolutionState,
) -> None:
    profile = cb.build_state_profile(feasible_solution)

    assert cb.is_candidate_better(1.0, profile, 2.0, profile) is True
    assert cb.is_candidate_better(3.0, profile, 2.0, profile) is False


def test_is_candidate_better_accepts_the_first_candidate(
    feasible_solution: SolutionState,
) -> None:
    profile = cb.build_state_profile(feasible_solution)

    assert cb.is_candidate_better(5.0, profile, None, None) is True


def test_is_candidate_better_fast_mode_uses_objective_only(
    feasible_solution: SolutionState, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ALNSConfig, "ENABLE_FAST_ALNS_MODE", True)
    profile = cb.build_state_profile(feasible_solution)

    assert cb.is_candidate_better(1.0, profile, 2.0, None) is True
    assert cb.is_candidate_better(2.0, profile, 1.0, None) is False


# ---------------------------------------------------------------------------
# Recording the best feasible solution
# ---------------------------------------------------------------------------
def test_is_best_feasible_solution_accepts_a_feasible_state(
    feasible_solution: SolutionState,
) -> None:
    assert cb._is_best_feasible_solution(feasible_solution.copy()) is True


def test_is_best_feasible_solution_rejects_an_infeasible_state(data: DataALNS) -> None:
    # An empty fleet cannot satisfy demand, so it is not a valid record.
    assert cb._is_best_feasible_solution(SolutionState(data)) is False


def test_record_best_feasible_returns_none_without_candidates(data: DataALNS) -> None:
    assert cb._record_best_feasible([]) is None
    assert cb._record_best_feasible([SolutionState(data)]) is None


def test_record_best_feasible_selects_the_feasible_candidate(
    data: DataALNS, feasible_solution: SolutionState
) -> None:
    best = cb._record_best_feasible([SolutionState(data), feasible_solution])

    assert best is feasible_solution


def test_record_best_feasible_updates_the_optimizer(
    feasible_solution: SolutionState,
) -> None:
    optimizer = _DummyOptimizer()

    cb.record_best_feasible(optimizer, feasible_solution.copy())

    assert optimizer.best_feasible_state is not None
    assert optimizer.best_feasible_obj < float("inf")
    assert optimizer.log_printer.messages


def test_record_best_feasible_ignores_an_infeasible_candidate(data: DataALNS) -> None:
    optimizer = _DummyOptimizer()

    cb.record_best_feasible(optimizer, SolutionState(data))

    assert optimizer.best_feasible_state is None


def test_dump_state_snapshot_ignores_none() -> None:
    # Must be a no-op rather than raising on a missing state.
    assert cb.dump_state_snapshot(_DummyOptimizer(), None, "empty") is None


def test_make_callback_on_best_records_and_survives_failures(
    feasible_solution: SolutionState,
) -> None:
    optimizer = _DummyOptimizer()
    callback = cb.make_callback_on_best(optimizer)

    callback(feasible_solution.copy(), np.random.default_rng(0))
    assert optimizer.best_feasible_state is not None

    # A state whose validation explodes must not propagate out of the callback.
    class _Broken:
        def compute_inventory(self):
            raise RuntimeError("boom")

        def validate(self):
            raise RuntimeError("boom")

    callback(_Broken(), np.random.default_rng(0))
