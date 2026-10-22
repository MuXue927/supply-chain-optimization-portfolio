"""Focused tests for the ALNS acceptance policy and progress tracker.

``accept.DemandConstraintAccept`` decides whether a candidate solution is
kept, and ``alnstrack.ALNSTracker`` records the search trajectory. Both are
pure logic (no solver), so they get direct unit tests instead of relying on a
full ALNS run to exercise them.
"""

from __future__ import annotations

import csv
import time

import numpy as np
import pytest

from multi_period_delivery.alns import accept as accept_module
from multi_period_delivery.alns import snapshot as snapshot_module
from multi_period_delivery.alns.accept import (
    DemandConstraintAccept,
    _CandidateProxy,
    _safe_float,
)
from multi_period_delivery.alns.alns_config import default_config as ALNSConfig
from multi_period_delivery.alns.alnstrack import ALNSTracker, calculate_gap


class _FakeVehicle:
    def __init__(self, veh_type: str = "A", load: float = 5.0, empty: bool = False):
        self.type = veh_type
        self.load = load
        self._empty = empty

    def is_empty(self) -> bool:
        return self._empty


class _FakeTracker:
    def __init__(self, iteration: int, max_iterations: int, elapsed: float = 0.0):
        self.iteration = iteration
        self.max_iterations = max_iterations
        # Phase progress is measured against the runtime budget, so the tracker
        # carries a monotonic origin; ``elapsed`` places the run in its budget.
        self.budget_start_time = time.perf_counter() - elapsed


class _FakeData:
    all_veh_types = ["A"]
    veh_type_cap = {"A": 10.0}
    veh_type_cost = {"A": 1.0}


class _FakeState:
    """Minimal stand-in exposing the interface accept/track actually consume."""

    def __init__(
        self,
        objective: float = 10.0,
        actual: float = 10.0,
        feasible: bool = True,
        violations: dict | None = None,
        vehicles: list | None = None,
    ):
        self._objective = float(objective)
        self._actual = float(actual)
        self._feasible = feasible
        self._violations = violations if violations is not None else {}
        self.vehicles = list(vehicles or [])
        self.data = _FakeData()

    def objective(self) -> float:
        return self._objective

    def calculate_cost(self) -> float:
        return self._actual

    def validate(self) -> tuple[bool, dict]:
        return self._feasible, self._violations

    def iter_used_vehicles(self):
        return iter([veh for veh in self.vehicles if not veh.is_empty()])

    def compute_veh_load(self, veh: _FakeVehicle) -> float:
        return veh.load

    def copy(self) -> _FakeState:
        clone = _FakeState(
            self._objective,
            self._actual,
            self._feasible,
            dict(self._violations),
            self.vehicles,
        )
        return clone


class _RecordingAcceptor:
    """Base acceptor that records the objective it was asked to judge."""

    def __init__(self, decision: bool = True):
        self.decision = decision
        self.seen_objectives: list[float] = []

    def __call__(self, rng, best_state, current_state, candidate_state) -> bool:
        self.seen_objectives.append(float(candidate_state.objective()))
        return self.decision


# ---------------------------------------------------------------------------
# _safe_float / _CandidateProxy
# ---------------------------------------------------------------------------
def test_safe_float_falls_back_to_default() -> None:
    assert _safe_float(None) == float("inf")
    assert _safe_float("not-a-number", default=42.0) == 42.0
    assert _safe_float("3.5") == 3.5


def test_candidate_proxy_adds_penalty_and_delegates() -> None:
    state = _FakeState(objective=10.0)

    proxy = _CandidateProxy(state, penalty=2.5)

    assert proxy.objective() == 12.5
    # Un-overridden attributes fall through to the wrapped object.
    assert proxy.calculate_cost() == state.calculate_cost()
    assert proxy.validate() == state.validate()
    # compute_shipped is absent on the fake, so the proxy returns {}.
    assert proxy.compute_shipped() == {}


def test_candidate_proxy_keeps_non_finite_objective() -> None:
    state = _FakeState(objective=float("inf"))

    proxy = _CandidateProxy(state, penalty=100.0)

    assert proxy.objective() == float("inf")


# ---------------------------------------------------------------------------
# DemandConstraintAccept
# ---------------------------------------------------------------------------
def test_demand_accept_delegates_feasible_candidate_to_base() -> None:
    base = _RecordingAcceptor(decision=True)
    acceptor = DemandConstraintAccept(base, data=None, penalty_factor=1000.0)

    accepted = acceptor(
        np.random.default_rng(0),
        _FakeState(),
        _FakeState(),
        _FakeState(objective=8.0),
    )

    assert accepted is True
    assert base.seen_objectives == [8.0]
    stats = acceptor.get_statistics()
    assert stats["calls"] == 1
    assert stats["accepted"] == 1
    assert stats["accept_rate"] == 1.0


def test_demand_accept_rejects_candidate_when_validation_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _RecordingAcceptor(decision=True)
    acceptor = DemandConstraintAccept(base, data=None)
    candidate = _FakeState()
    recorded: list[tuple[str, BaseException]] = []

    def raising_validate():
        raise RuntimeError("validator exploded")

    monkeypatch.setattr(candidate, "validate", raising_validate)
    monkeypatch.setattr(
        accept_module,
        "record_swallowed",
        lambda site, exc: recorded.append((site, exc)),
    )

    accepted = acceptor(
        np.random.default_rng(0),
        _FakeState(),
        _FakeState(),
        candidate,
    )

    assert accepted is False
    assert base.seen_objectives == []
    assert len(recorded) == 1
    assert recorded[0][0] == "accept.DemandConstraintAccept.validate"
    assert isinstance(recorded[0][1], RuntimeError)
    stats = acceptor.get_statistics()
    assert stats["calls"] == 1
    assert stats["accepted"] == 0
    assert stats["accept_rate"] == 0.0


def test_demand_accept_hard_rejects_non_demand_violations() -> None:
    base = _RecordingAcceptor(decision=True)
    acceptor = DemandConstraintAccept(base, data=None)
    candidate = _FakeState(
        feasible=False,
        violations={"negative_inventory": [{"plant": "P1", "inventory": -5}]},
    )

    accepted = acceptor(
        np.random.default_rng(0),
        _FakeState(),
        _FakeState(),
        candidate,
    )

    assert accepted is False
    assert base.seen_objectives == []
    stats = acceptor.get_statistics()
    assert stats["calls"] == 1
    assert stats["accepted"] == 0


def test_demand_accept_rejects_malformed_infeasible_violations() -> None:
    base = _RecordingAcceptor(decision=True)
    acceptor = DemandConstraintAccept(base, data=None)
    candidate = _FakeState(feasible=False)
    candidate._violations = None

    accepted = acceptor(
        np.random.default_rng(0),
        _FakeState(),
        _FakeState(),
        candidate,
    )

    assert accepted is False
    assert base.seen_objectives == []


def test_demand_accept_adds_penalty_for_unmet_demand() -> None:
    base = _RecordingAcceptor()
    acceptor = DemandConstraintAccept(base, data=None, penalty_factor=10.0)
    candidate = _FakeState(
        objective=5.0,
        feasible=False,
        violations={
            "unmet_demand": [
                {"dealer": "D1", "sku_id": "S1", "demand": 10, "shipped": 4},
                {"dealer": "D2", "sku_id": "S2", "demand": 6, "shipped": 6},
            ]
        },
    )

    acceptor(np.random.default_rng(0), _FakeState(), _FakeState(), candidate)

    # unmet total = (10 - 4) + (6 - 6) = 6 -> penalty = 10 * 6 = 60 -> 5 + 60.
    assert base.seen_objectives == [65.0]
    assert acceptor.get_statistics()["total_penalty"] == 60.0


def test_demand_accept_uses_actual_priority_late_in_search() -> None:
    base = _RecordingAcceptor()
    acceptor = DemandConstraintAccept(base, data=None)
    # Progress is elapsed-runtime / MAX_RUNTIME; 0.8 of the budget is past
    # ACTUAL_PRIORITY_START_PROGRESS (0.70).
    late_tracker = _FakeTracker(
        iteration=80, max_iterations=100, elapsed=0.8 * float(ALNSConfig.MAX_RUNTIME)
    )
    current = _FakeState(objective=100.0, actual=100.0)
    current.tracker = late_tracker
    candidate = _FakeState(objective=95.0, actual=90.0)
    candidate.tracker = late_tracker

    accepted = acceptor(np.random.default_rng(0), current, current, candidate)

    # Better actual objective wins directly, and the base acceptor is bypassed.
    assert accepted is True
    assert base.seen_objectives == []


def test_demand_accept_rejects_late_regression() -> None:
    base = _RecordingAcceptor()
    acceptor = DemandConstraintAccept(base, data=None)
    late_tracker = _FakeTracker(
        iteration=90, max_iterations=100, elapsed=0.9 * float(ALNSConfig.MAX_RUNTIME)
    )
    current = _FakeState(objective=100.0, actual=100.0)
    current.tracker = late_tracker
    candidate = _FakeState(objective=300.0, actual=300.0)
    candidate.tracker = late_tracker

    accepted = acceptor(np.random.default_rng(0), current, current, candidate)

    assert accepted is False
    assert base.seen_objectives == []


# ---------------------------------------------------------------------------
# ALNSTracker
# ---------------------------------------------------------------------------
def test_tracker_records_best_objective_and_history() -> None:
    tracker = ALNSTracker()

    tracker.on_iteration(_FakeState(objective=100.0, actual=100.0), np.random.default_rng(0))
    tracker.on_iteration(_FakeState(objective=80.0, actual=80.0), np.random.default_rng(0))
    tracker.on_iteration(_FakeState(objective=90.0, actual=90.0), np.random.default_rng(0))

    stats = tracker.get_statistics()
    assert stats["total_iterations"] == 3
    assert stats["best_objective"] == 80.0
    assert stats["current_objective"] == 90.0
    assert len(stats["objectives_history"]) == 3
    assert tracker.no_improve_iters == 1  # third iteration did not improve


def test_tracker_ignores_infeasible_improvements() -> None:
    tracker = ALNSTracker()
    tracker.on_iteration(_FakeState(objective=100.0, actual=100.0), np.random.default_rng(0))

    tracker.on_iteration(
        _FakeState(objective=1.0, actual=1.0, feasible=False),
        np.random.default_rng(0),
    )

    assert tracker.best_obj == 100.0


def test_tracker_clears_stale_snapshots_when_copy_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = ALNSTracker()
    tracker.best_solution = object()
    tracker.best_actual_solution = object()

    def raising_snapshot(_state):
        raise RuntimeError("snapshot failed")

    monkeypatch.setattr(snapshot_module, "safe_state_snapshot", raising_snapshot)

    tracker.on_iteration(
        _FakeState(objective=9.0, actual=9.0),
        np.random.default_rng(0),
    )

    assert tracker.best_obj == float("inf")
    assert tracker.best_actual_obj == float("inf")
    assert tracker.best_solution is None
    assert tracker.best_actual_solution is None


def test_tracker_should_stop_after_patience() -> None:
    tracker = ALNSTracker()
    tracker.early_stop_patience = 2
    state = _FakeState(objective=10.0)

    tracker.on_iteration(state, np.random.default_rng(0))  # first best -> reset
    assert not tracker.should_stop()
    tracker.on_iteration(state, np.random.default_rng(0))
    assert not tracker.should_stop()
    tracker.on_iteration(state, np.random.default_rng(0))
    assert tracker.should_stop()


def test_tracker_writes_csv_rows(tmp_path) -> None:
    log_file = tmp_path / "progress.csv"
    tracker = ALNSTracker(output_file=str(log_file))

    tracker.on_iteration(_FakeState(objective=12.0), np.random.default_rng(0))
    tracker.on_iteration(_FakeState(objective=11.0), np.random.default_rng(0))

    with log_file.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["Iteration", "Current_Obj", "Best_Obj", "GapVsInitial"]
    assert len(rows) == 3  # header + two iterations


def test_tracker_ml_cache_roundtrip() -> None:
    tracker = ALNSTracker()
    assert tracker.has_cached_model() is False

    tracker.update_ml_data([1.0, 2.0], 0.5)
    tracker.cache_ml_model(model=object(), scaler=object(), iteration=7)

    assert tracker.has_cached_model() is True
    model, scaler, iteration = tracker.get_cached_model()
    assert model is not None
    assert scaler is not None
    assert iteration == 7
    assert tracker.get_ml_cache_info()["data_samples"] == 1

    tracker.clear_ml_cache()
    assert tracker.has_cached_model() is False


def test_tracker_caps_ml_samples() -> None:
    tracker = ALNSTracker()

    for index in range(1005):
        tracker.update_ml_data([float(index)], 0.0)

    assert len(tracker.features) == 1000
    assert len(tracker.labels) == 1000


def test_record_elite_candidate_deduplicates_and_keeps_best() -> None:
    tracker = ALNSTracker()
    tracker.elite_pool_max_size = 2
    vehicles = [_FakeVehicle(load=5.0)]

    tracker.record_elite_candidate(_FakeState(objective=30.0, vehicles=vehicles))
    tracker.record_elite_candidate(_FakeState(objective=20.0, vehicles=vehicles))

    # Same profile signature -> one slot, updated to the better objective.
    assert len(tracker.elite_pool) == 1
    assert tracker.elite_pool[0]["obj"] == 20.0


def test_record_elite_candidate_rejects_infeasible() -> None:
    tracker = ALNSTracker()

    accepted = tracker.record_elite_candidate(_FakeState(feasible=False))

    assert accepted is False
    assert tracker.elite_pool == []


@pytest.mark.parametrize(
    ("current", "best", "expected"),
    [
        (0.0, 0.0, 0.0),
        (100.0, 80.0, 20.0),
        (50.0, 50.0, 0.0),
    ],
)
def test_calculate_gap_relative_difference(current, best, expected) -> None:
    assert calculate_gap(current, best) == pytest.approx(expected)


def test_calculate_gap_is_infinite_when_current_is_zero() -> None:
    assert calculate_gap(0.0, 5.0) == float("inf")
