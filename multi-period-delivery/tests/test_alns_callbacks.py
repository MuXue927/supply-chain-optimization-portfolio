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
from multi_period_delivery.alns import snapshot as snapshot_module
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


def test_record_best_feasible_clears_stale_state_when_snapshot_fails(
    feasible_solution: SolutionState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer = _DummyOptimizer()
    stale = feasible_solution.copy()
    monkeypatch.setattr(stale, "objective", lambda: 1e9)
    optimizer.best_feasible_state = stale
    optimizer.best_feasible_obj = 123.0
    recorded: list[tuple[str, BaseException]] = []

    def raising_snapshot(_state):
        raise RuntimeError("snapshot failed")

    monkeypatch.setattr(snapshot_module, "safe_state_snapshot", raising_snapshot)
    monkeypatch.setattr(
        snapshot_module,
        "record_swallowed",
        lambda site, exc: recorded.append((site, exc)),
    )

    cb.record_best_feasible(optimizer, feasible_solution.copy())

    assert optimizer.best_feasible_state is None
    assert optimizer.best_feasible_obj == 123.0
    assert recorded[0][0] == "callbacks.best_feasible"
    assert isinstance(recorded[0][1], RuntimeError)


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


# ---------------------------------------------------------------------------
# Fallback paths
# ---------------------------------------------------------------------------
def test_safe_state_snapshot_falls_back_to_deepcopy() -> None:
    class _CopyRaises:
        def copy(self):
            raise RuntimeError("no copy")

    original = _CopyRaises()

    snapshot = cb.safe_state_snapshot(original)

    assert snapshot is not original


def test_safe_state_snapshot_fails_closed_when_everything_fails() -> None:
    class _Uncopyable:
        def copy(self):
            raise RuntimeError("no copy")

        def __deepcopy__(self, memo):
            raise RuntimeError("no deepcopy")

    original = _Uncopyable()

    with pytest.raises(RuntimeError, match="independent state snapshot"):
        cb.safe_state_snapshot(original)


def test_safe_state_snapshot_rejects_copy_results_that_alias_the_original() -> None:
    class _SelfReturning:
        def copy(self):
            return self

        def __deepcopy__(self, memo):
            return self

    original = _SelfReturning()

    with pytest.raises(RuntimeError, match="returned the original or None"):
        cb.safe_state_snapshot(original)


def test_runtime_guard_rolls_back_to_verified_snapshot() -> None:
    class _Snapshot:
        def __init__(self, feasible: bool, vehicles: list[str]):
            self.feasible = feasible
            self.vehicles = list(vehicles)

        def copy(self):
            return _Snapshot(self.feasible, self.vehicles)

        def validate(self):
            return self.feasible, {}

    class _State:
        def __init__(self):
            self.vehicles: list[str] = []

        def replace_vehicles(self, vehicles):
            self.vehicles = list(vehicles)

        def compute_inventory(self):
            return True

        def validate(self):
            return True, {}

    class _Optimizer:
        def __init__(self, best, incumbent):
            self.best_feasible_state = best
            self._incumbent_state_snapshot = incumbent
            self.result = None

    state = _State()
    optimizer = _Optimizer(
        best=_Snapshot(False, ["invalid"]),
        incumbent=_Snapshot(True, ["verified"]),
    )

    assert cb._restore_verified_runtime_state(state, optimizer) is True
    assert state.vehicles == ["verified"]

    state = _State()
    optimizer = _Optimizer(
        best=_Snapshot(False, ["invalid"]),
        incumbent=_Snapshot(False, ["also-invalid"]),
    )

    assert cb._restore_verified_runtime_state(state, optimizer) is False
    assert state.vehicles == []


def test_incumbent_snapshot_failure_clears_stale_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer = _DummyOptimizer()
    optimizer._incumbent_state_snapshot = object()
    recorded: list[tuple[str, BaseException]] = []

    def raising_snapshot(_state):
        raise RuntimeError("snapshot failed")

    monkeypatch.setattr(snapshot_module, "safe_state_snapshot", raising_snapshot)
    monkeypatch.setattr(
        snapshot_module,
        "record_swallowed",
        lambda site, exc: recorded.append((site, exc)),
    )

    stored = cb._store_incumbent_snapshot(optimizer, object())

    assert stored is False
    assert optimizer._incumbent_state_snapshot is None
    assert recorded[0][0] == "callbacks.incumbent_snapshot"
    assert isinstance(recorded[0][1], RuntimeError)


def test_dump_state_snapshot_survives_a_broken_objective() -> None:
    class _Broken:
        def objective(self):
            raise RuntimeError("boom")

    assert cb.dump_state_snapshot(_DummyOptimizer(), _Broken(), "broken") is None


def test_get_capacity_threshold_is_zero_without_vehicle_types() -> None:
    class _NoTypes:
        all_veh_types: list[str] = []

    assert cb._get_capacity_threshold(_NoTypes()) == 0.0


def test_get_capacity_threshold_skips_unknown_vehicle_types() -> None:
    class _Partial:
        all_veh_types = ["A", "MISSING_FROM_CAPACITY_MAP"]
        veh_type_cap = {"A": 42.0}

    assert cb._get_capacity_threshold(_Partial()) == 42.0


def test_build_state_profile_without_iter_used_vehicles(
    feasible_solution: SolutionState,
) -> None:
    class _NoIterator:
        """Proxy that hides ``iter_used_vehicles`` to exercise the fallback."""

        def __init__(self, inner: SolutionState) -> None:
            self._inner = inner
            self.vehicles = inner.vehicles

        def __getattr__(self, name: str):
            if name == "iter_used_vehicles":
                raise AttributeError(name)
            return getattr(self._inner, name)

    profile = cb.build_state_profile(_NoIterator(feasible_solution))

    assert profile["vehicles"] == len(feasible_solution.vehicles)
    assert profile["fixed_cost"] > 0.0


def test_used_vehicle_count_falls_back_to_the_vehicles_attribute() -> None:
    class _Vehicle:
        def __init__(self, empty: bool) -> None:
            self._empty = empty

        def is_empty(self) -> bool:
            return self._empty

    class _NoIterator:
        def __init__(self) -> None:
            self.vehicles = [_Vehicle(False), _Vehicle(True), None]

    assert cb._used_vehicle_count(_NoIterator()) == 1


# ---------------------------------------------------------------------------
# Candidate comparison: phased lexicographic and vehicle priority
# ---------------------------------------------------------------------------
def _profile(
    vehicles: float,
    load: float = 0.5,
    cost: float = 10.0,
    large: float = 0.5,
) -> dict:
    return {
        "vehicles": vehicles,
        "load_ratio": load,
        "large_ratio": large,
        "fixed_cost": cost,
    }


def test_is_candidate_better_fast_mode_accepts_the_first_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ALNSConfig, "ENABLE_FAST_ALNS_MODE", True)

    assert cb.is_candidate_better(1.0, _profile(3.0), None, None) is True


def test_is_candidate_better_phased_lexicographic_prefers_a_smaller_fleet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ALNSConfig, "ENABLE_PHASED_LEXICOGRAPHIC_BEST", True)
    monkeypatch.setattr(ALNSConfig, "PHASED_LEXICOGRAPHIC_OBJ_ABS_TOL", 5.0)

    better = cb.is_candidate_better(102.0, _profile(3.0), 100.0, _profile(5.0), 0.5)
    worse = cb.is_candidate_better(120.0, _profile(3.0), 100.0, _profile(5.0), 0.5)
    larger = cb.is_candidate_better(100.0, _profile(6.0), 100.0, _profile(5.0), 0.5)

    assert better is True
    assert worse is False
    assert larger is False


def test_is_candidate_better_uses_a_stricter_late_search_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ALNSConfig, "ENABLE_VEHICLE_COUNT_PRIORITY", True)
    monkeypatch.setattr(ALNSConfig, "ENABLE_PHASED_LEXICOGRAPHIC_BEST", False)
    monkeypatch.setattr(ALNSConfig, "VEHICLE_PRIORITY_OBJ_ABS_TOL", 2.0)

    # The allowed objective worsening is max(abs_tol, rel_tol * |old_obj|) with
    # the phase factor applied to both: late halves it (8.0 * 0.5 = 4.0), early
    # multiplies it by 1.5 (8.0 * 1.5 = 12.0). A +6.0 objective fits only early.
    late = cb.is_candidate_better(106.0, _profile(3.0), 100.0, _profile(5.0), 0.9)
    early = cb.is_candidate_better(106.0, _profile(3.0), 100.0, _profile(5.0), 0.1)

    assert late is False
    assert early is True


@pytest.mark.parametrize(
    ("new_profile", "old_profile", "expected"),
    [
        # Same objective -> inside the near-equal band; the fleet tie-breaks
        # below decide the winner.
        (_profile(3.0), _profile(5.0), True),
        (_profile(6.0), _profile(5.0), False),
        (_profile(5.0, load=0.9), _profile(5.0, load=0.5), True),
        (_profile(5.0, load=0.2), _profile(5.0, load=0.5), False),
        (_profile(5.0, load=0.5, cost=5.0), _profile(5.0, load=0.5, cost=10.0), True),
        (_profile(5.0, load=0.5, cost=20.0), _profile(5.0, load=0.5, cost=10.0), False),
        (
            _profile(5.0, load=0.5, cost=10.0, large=0.2),
            _profile(5.0, load=0.5, cost=10.0, large=0.5),
            True,
        ),
        (
            _profile(5.0, load=0.5, cost=10.0, large=0.5),
            _profile(5.0, load=0.5, cost=10.0, large=0.5),
            False,
        ),
    ],
)
def test_is_candidate_better_tie_breaks_within_the_near_equal_band(
    monkeypatch: pytest.MonkeyPatch,
    new_profile: dict,
    old_profile: dict,
    expected: bool,
) -> None:
    monkeypatch.setattr(ALNSConfig, "ENABLE_PHASED_LEXICOGRAPHIC_BEST", False)
    monkeypatch.setattr(ALNSConfig, "ENABLE_FAST_ALNS_MODE", False)

    # Identical objectives put the comparison inside the near-equal band.
    assert (
        cb.is_candidate_better(100.0, new_profile, 100.0, old_profile) is expected
    )


# ---------------------------------------------------------------------------
# Path relinking between the incumbent and an elite solution
# ---------------------------------------------------------------------------
def test_build_path_relinked_vehicle_list_blends_the_target(
    feasible_solution: SolutionState,
) -> None:
    current = feasible_solution.copy()
    target = feasible_solution.copy()
    target_used = list(target.iter_used_vehicles())

    relinked = cb._build_path_relinked_vehicle_list(
        current, target, blend_ratio=0.5, max_target_veh=5
    )

    assert relinked
    # The injected vehicles come straight from the target solution.
    assert any(veh in target_used for veh in relinked)


def test_build_path_relinked_vehicle_list_without_a_target_keeps_current(
    data: DataALNS, feasible_solution: SolutionState,
) -> None:
    current = feasible_solution.copy()
    empty_target = SolutionState(data)

    relinked = cb._build_path_relinked_vehicle_list(current, empty_target)

    assert relinked == list(current.iter_used_vehicles())


def test_build_relink_progression_is_monotonic() -> None:
    schedule = cb._build_relink_progression(0.2, 4, steps=3)

    assert len(schedule) == 3
    ratios = [ratio for ratio, _ in schedule]
    targets = [target for _, target in schedule]
    assert ratios == sorted(ratios)
    assert targets == sorted(targets)
    assert all(0.0 <= ratio <= 1.0 for ratio in ratios)
    # The schedule never asks for fewer than one target vehicle.
    assert all(target >= 1 for target in targets)


def test_vehicle_elimination_repair_returns_a_state(
    feasible_solution: SolutionState,
) -> None:
    state = feasible_solution.copy()

    result = cb._vehicle_elimination_repair(state, np.random.default_rng(1))

    assert isinstance(result, SolutionState)


def test_is_relink_candidate_acceptable(
    data: DataALNS, feasible_solution: SolutionState,
) -> None:
    current = feasible_solution.copy()

    assert (
        cb._is_relink_candidate_acceptable(current, feasible_solution.copy()) is True
    )
    # A solution with unmet demand has an enormous objective and is rejected.
    assert cb._is_relink_candidate_acceptable(current, SolutionState(data)) is False


def test_choose_elite_relink_target_handles_an_empty_pool(
    feasible_solution: SolutionState,
) -> None:
    assert cb._choose_elite_relink_target(feasible_solution.copy(), []) is None


def test_choose_elite_relink_target_skips_identical_signatures(
    feasible_solution: SolutionState,
) -> None:
    current = feasible_solution.copy()
    profile = cb.build_state_profile(current)
    pool = [
        {
            "state": feasible_solution.copy(),
            "obj": float(current.objective()) - 1.0,
            "profile": profile,
            "signature": cb._profile_signature(profile),
        }
    ]

    assert cb._choose_elite_relink_target(current, pool) is None


def test_choose_elite_relink_target_returns_the_best_candidate(
    feasible_solution: SolutionState,
) -> None:
    current = feasible_solution.copy()
    pool = [
        {
            "state": feasible_solution.copy(),
            "obj": float(current.objective()) - 1.0,
            "profile": _profile(8.0, 0.9, 1.0),
            "signature": (8, 9, 9, 1),
        },
        {
            "state": feasible_solution.copy(),
            "obj": float(current.objective()) - 5.0,
            "profile": _profile(7.0, 0.9, 1.0),
            "signature": (7, 9, 9, 1),
        },
    ]

    chosen = cb._choose_elite_relink_target(current, pool)

    assert chosen is not None
    assert chosen["obj"] == float(current.objective()) - 5.0


# ---------------------------------------------------------------------------
# ML warmup
# ---------------------------------------------------------------------------
def test_ml_warmup_sample_is_a_noop_without_ml(
    feasible_solution: SolutionState,
) -> None:
    optimizer = _DummyOptimizer()
    optimizer.ml_selector = None
    optimizer.param_tuner = None

    assert cb.ml_warmup_sample(optimizer, feasible_solution, budget=2) is None


def test_ml_warmup_sample_copies_the_state_and_survives_failures(
    feasible_solution: SolutionState,
) -> None:
    # The stub components are not real selectors, so both the destroy and the
    # repair attempt fail internally - which is exactly the path that must not
    # disturb the main loop.
    optimizer = _DummyOptimizer()
    optimizer.ml_selector = object()
    optimizer.param_tuner = object()
    optimizer.rng = np.random.default_rng(1)
    state = feasible_solution.copy()
    before = state.s_ikt

    assert cb.ml_warmup_sample(optimizer, state, budget=2) is None

    # Warmup works on copies only, so the original state is untouched.
    assert state.s_ikt is before


def test_ml_warmup_sample_starts_no_trial_once_the_budget_is_spent(
    feasible_solution: SolutionState,
) -> None:
    """An exhausted time budget must stop sampling before another trial starts."""

    class _BudgetedSelector:
        warmup_stop_reason = None

        def __init__(self) -> None:
            self.calls = 0

        def warmup_scope(self):
            from contextlib import nullcontext

            return nullcontext()

        def warmup_budget_exhausted(self) -> bool:
            return True

        def note_warmup_seconds(self, seconds: float) -> None:
            self.calls += 1

    optimizer = _DummyOptimizer()
    optimizer.ml_selector = _BudgetedSelector()
    optimizer.rng = np.random.default_rng(2)

    assert cb.ml_warmup_sample(optimizer, feasible_solution, budget=3) is None

    assert optimizer.ml_selector.calls == 0
    assert optimizer.ml_selector.warmup_stop_reason == "budget"

    # A second call must bail out before doing any work as well.
    assert cb.ml_warmup_sample(optimizer, feasible_solution, budget=3) is None
    assert optimizer.ml_selector.calls == 0


# ---------------------------------------------------------------------------
# Iteration callback
# ---------------------------------------------------------------------------
def test_iteration_callback_keeps_the_state_hard_feasible(
    tmp_path, feasible_solution: SolutionState
) -> None:
    from multi_period_delivery.alns.alns_optimizer import ALNSOptimizer
    from multi_period_delivery.alns.alnstrack import ALNSTracker

    base = Path(__file__).resolve().parents[1]
    optimizer = ALNSOptimizer(
        input_file_loc=str(base / "datasets" / "multiple-periods" / "small"),
        output_file_loc=str(tmp_path / "out"),
    )
    optimizer.load_data("dataset_1")
    optimizer.tracker = ALNSTracker(log_printer=optimizer.log_printer)

    callback = cb.make_callback_on_iteration(optimizer, stop=None)
    state = feasible_solution.copy()
    for _ in range(3):
        callback(state, np.random.default_rng(4))
        optimizer.increment_iteration()

    _feasible, violations = state.validate()
    assert not violations["negative_inventory"]
    assert not violations["veh_over_load"]


def test_iteration_callback_handles_negative_inventory(
    tmp_path, feasible_solution: SolutionState
) -> None:
    from multi_period_delivery.alns.alns_optimizer import ALNSOptimizer
    from multi_period_delivery.alns.alnstrack import ALNSTracker

    base = Path(__file__).resolve().parents[1]
    optimizer = ALNSOptimizer(
        input_file_loc=str(base / "datasets" / "multiple-periods" / "small"),
        output_file_loc=str(tmp_path / "out"),
    )
    optimizer.load_data("dataset_1")
    optimizer.tracker = ALNSTracker(log_printer=optimizer.log_printer)

    state = feasible_solution.copy()
    # Inflate one shipment so the recomputed inventory goes negative, which is
    # what the callback's inventory check and runtime guard exist to catch.
    vehicle = state.vehicles[0]
    cargo_key = next(iter(vehicle.cargo))
    vehicle.cargo[cargo_key] += 10_000
    state.compute_inventory()
    assert state.s_ikt[(vehicle.fact_id, cargo_key[0], cargo_key[1])] < 0

    callback = cb.make_callback_on_iteration(optimizer, stop=None)
    callback(state, np.random.default_rng(6))

    # The callback must not raise, whatever repair it manages to apply.
    assert isinstance(state.s_ikt, dict)


def test_record_best_feasible_reads_progress_from_the_tracker(
    feasible_solution: SolutionState, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A real run always carries a tracker, and the tie-break tolerances are
    # progress-dependent, so the progress fraction must actually reach
    # is_candidate_better().
    class _Tracker:
        _iteration_count = 5
        _max_iterations = 10

    optimizer = _DummyOptimizer()
    optimizer.tracker = _Tracker()
    seen: dict = {}
    original = cb.is_candidate_better

    def _spy(new_obj, new_profile, old_obj, old_profile, progress=None):
        seen["progress"] = progress
        return original(new_obj, new_profile, old_obj, old_profile, progress)

    monkeypatch.setattr(cb, "is_candidate_better", _spy)

    cb.record_best_feasible(optimizer, feasible_solution.copy())

    assert seen["progress"] == pytest.approx(0.5)
    assert optimizer.best_feasible_state is not None


def test_iteration_callback_reports_a_temporarily_infeasible_candidate(
    tmp_path, feasible_solution: SolutionState, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Early in a run a candidate can be infeasible simply because demand is not
    # fulfilled yet and no incumbent exists. With the diagnostics switch on,
    # that path prints an explanation instead of failing silently.
    #
    # An empty fleet reports *two* violations (unmet demand plus the plant
    # inventory that was never shipped), so this covers the generic message.
    monkeypatch.setattr(ALNSConfig, "ENABLE_CALLBACK_DIAGNOSTICS_CONSOLE", True)

    from multi_period_delivery.alns.alns_optimizer import ALNSOptimizer
    from multi_period_delivery.alns.alnstrack import ALNSTracker

    base = Path(__file__).resolve().parents[1]
    optimizer = ALNSOptimizer(
        input_file_loc=str(base / "datasets" / "multiple-periods" / "small"),
        output_file_loc=str(tmp_path / "out"),
    )
    optimizer.load_data("dataset_1")
    optimizer.tracker = ALNSTracker(log_printer=optimizer.log_printer)
    optimizer.best_feasible_state = None

    callback = cb.make_callback_on_iteration(optimizer, stop=None)
    infeasible = SolutionState(optimizer.data)  # empty fleet -> unmet demand
    callback(infeasible, np.random.default_rng(9))

    # The callback must survive the infeasible candidate and keep going.
    assert isinstance(infeasible.s_ikt, dict)


def test_iteration_callback_reports_an_unmet_demand_only_candidate(
    tmp_path, feasible_solution: SolutionState, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The "only unmet demand" case gets its own, more reassuring message: the
    # candidate is temporarily infeasible because no incumbent exists yet.
    monkeypatch.setattr(ALNSConfig, "ENABLE_CALLBACK_DIAGNOSTICS_CONSOLE", True)

    from multi_period_delivery.alns.alns_optimizer import ALNSOptimizer
    from multi_period_delivery.alns.alnstrack import ALNSTracker

    base = Path(__file__).resolve().parents[1]
    optimizer = ALNSOptimizer(
        input_file_loc=str(base / "datasets" / "multiple-periods" / "small"),
        output_file_loc=str(tmp_path / "out"),
    )
    optimizer.load_data("dataset_1")
    optimizer.tracker = ALNSTracker(log_printer=optimizer.log_printer)
    optimizer.best_feasible_state = None

    state = feasible_solution.copy()
    monkeypatch.setattr(
        state,
        "validate",
        lambda: (
            False,
            {
                "negative_inventory": [],
                "veh_over_load": [],
                "plant_inv_exceed": [],
                "unmet_demand": [
                    {"dealer": "D1", "sku_id": "S1", "demand": 5, "shipped": 0}
                ],
                "cargo_day_mismatch": [],
            },
        ),
    )

    callback = cb.make_callback_on_iteration(optimizer, stop=None)
    callback(state, np.random.default_rng(9))

    assert isinstance(state.s_ikt, dict)
