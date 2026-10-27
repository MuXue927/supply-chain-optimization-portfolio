"""Tests for the ALNS high-level controller.

``ALNSOptimizer`` owns the whole pipeline: load data, build an initial
solution, register operators, run the search, then post-process the result.
The helpers are unit-tested directly; one short optimization run is shared by
several tests that then exercise the reporting steps of that same run.
"""

from __future__ import annotations

import builtins
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from multi_period_delivery.alns import alns_optimizer as optimizer_module
from multi_period_delivery.alns.alns_config import default_config as ALNSConfig
from multi_period_delivery.alns.alns_optimizer import (
    ALNSOptimizer,
    is_file_in_use,
    resolve_run_seed,
    safe_state_snapshot,
    suppress_noisy_operator_stdout,
)
from multi_period_delivery.alns.alnstrack import ALNSTracker
from multi_period_delivery.alns.param_tuner import ParamAutoTuner


def _small_dataset_dir() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "datasets"
        / "multiple-periods"
        / "small"
    )


def _optimizer(tmp_path: Path) -> ALNSOptimizer:
    return ALNSOptimizer(
        input_file_loc=str(_small_dataset_dir()),
        output_file_loc=str(tmp_path / "out"),
    )


def test_run_seed_is_read_from_the_live_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Import order must not freeze the seed of the whole search.

    The module used to snapshot ``SEED`` at import time, so an embedding that
    imported this module before updating the shared config silently ran every
    "seed" on the same RNG stream.
    """
    monkeypatch.setattr(ALNSConfig, "SEED", 42)
    assert resolve_run_seed() == 42
    monkeypatch.setattr(ALNSConfig, "SEED", 7)
    assert resolve_run_seed() == 7


def test_optimizers_created_after_a_seed_change_use_the_new_seed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(ALNSConfig, "SEED", 42)
    first = _optimizer(tmp_path / "a").rng.integers(0, 1_000_000, 4)
    monkeypatch.setattr(ALNSConfig, "SEED", 7)
    second = _optimizer(tmp_path / "b").rng.integers(0, 1_000_000, 4)

    assert list(first) != list(second)


@pytest.fixture(scope="module")
def short_run(tmp_path_factory: pytest.TempPathFactory) -> ALNSOptimizer:
    """Run one short optimization and hand the finished optimizer to the tests.

    The default budget (300 iterations / 900 s) would make CI unusably slow.
    ``run_optimization`` also widens the stopping rule's iteration cap to a
    runtime-derived floor (``runtime * iterations-per-second * safety factor``
    plus a buffer), which would otherwise let the number of executed iterations
    - and therefore the coverage this fixture contributes - depend on how fast
    the machine is. Zeroing the dynamic knobs pins that floor to the iteration
    budget, so the run stops after a fixed number of iterations.
    """
    output = tmp_path_factory.mktemp("alns-run") / "out"
    optimizer = ALNSOptimizer(
        input_file_loc=str(_small_dataset_dir()),
        output_file_loc=str(output),
    )
    pinned = {
        "MAX_ITERATIONS": 120,
        "MAX_RUNTIME": 120,
        "DYNAMIC_MAX_ITERATIONS_PER_SECOND": 0,
        "DYNAMIC_MAX_ITERATION_SAFETY_FACTOR": 0,
        "DYNAMIC_MAX_ITERATION_BUFFER": 0,
    }
    originals = {name: getattr(ALNSConfig, name) for name in pinned}
    for name, value in pinned.items():
        setattr(ALNSConfig, name, value)
    try:
        assert optimizer.run_optimization("dataset_1") is True
    finally:
        for name, value in originals.items():
            setattr(ALNSConfig, name, value)
    return optimizer


# ---------------------------------------------------------------------------
# Construction and data loading
# ---------------------------------------------------------------------------
def test_constructor_defaults_resolve_against_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The CLI, run_model and this class all document cwd-relative defaults:
    # datasets/ and OutPut-ALNS/ live in the project checkout, not inside the
    # installed package.
    scale = ALNSConfig.DATASET_TYPE
    expected_input = tmp_path / "datasets" / "multiple-periods" / scale
    expected_input.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    optimizer = ALNSOptimizer()

    assert optimizer.input_file_loc == expected_input
    assert optimizer.input_file_loc.is_dir()
    assert optimizer.output_file_loc == (
        tmp_path / "OutPut-ALNS" / "multiple-periods" / scale
    )


def test_load_data_succeeds(tmp_path: Path) -> None:
    optimizer = _optimizer(tmp_path)

    assert optimizer.load_data("dataset_1") is True
    assert optimizer.data is not None


def test_load_data_reports_failure_for_an_unknown_dataset(tmp_path: Path) -> None:
    optimizer = _optimizer(tmp_path)

    assert optimizer.load_data("does_not_exist") is False


def test_create_initial_solution_requires_loaded_data(tmp_path: Path) -> None:
    optimizer = _optimizer(tmp_path)

    assert optimizer.create_initial_solution() is None


def test_create_initial_solution_is_feasible(tmp_path: Path) -> None:
    optimizer = _optimizer(tmp_path)
    optimizer.load_data("dataset_1")

    state = optimizer.create_initial_solution()

    assert state is not None
    assert state.vehicles
    feasible, violations = state.validate()
    assert feasible
    assert not violations["negative_inventory"]


def test_setup_alns_registers_destroy_and_repair_operators(tmp_path: Path) -> None:
    optimizer = _optimizer(tmp_path)
    optimizer.load_data("dataset_1")

    alns = optimizer.setup_alns()

    assert alns is not None
    assert optimizer._num_registered_destroy_ops > 0
    assert optimizer._num_registered_repair_ops > 0


def test_disabled_destroy_operators_defaults_to_empty() -> None:
    assert ALNSConfig.DISABLED_DESTROY_OPERATORS == ()


def test_disabled_destroy_operators_are_not_registered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Disabling removes the operator instead of giving it zero weight."""
    monkeypatch.setattr(ALNSConfig, "ENABLE_ML_OPERATORS", False)
    monkeypatch.setattr(ALNSConfig, "DISABLED_DESTROY_OPERATORS", ())
    optimizer = _optimizer(tmp_path / "enabled")
    optimizer.load_data("dataset_1")
    enabled = optimizer.setup_alns()
    enabled_names = [name for name, _ in enabled.destroy_operators]
    enabled_count = optimizer._num_registered_destroy_ops

    assert "surplus_inventory_removal" in enabled_names

    monkeypatch.setattr(
        ALNSConfig,
        "DISABLED_DESTROY_OPERATORS",
        ("surplus_inventory_removal",),
    )
    optimizer = _optimizer(tmp_path / "disabled")
    optimizer.load_data("dataset_1")
    disabled = optimizer.setup_alns()
    disabled_names = [name for name, _ in disabled.destroy_operators]

    assert "surplus_inventory_removal" not in disabled_names
    assert disabled_names == [
        name for name in enabled_names if name != "surplus_inventory_removal"
    ]
    assert optimizer._num_registered_destroy_ops == enabled_count - 1


def _state_fingerprint(state) -> str:
    digest = hashlib.sha256()
    for vehicle in state.vehicles:
        digest.update(
            json.dumps(
                [
                    vehicle.fact_id,
                    vehicle.dealer_id,
                    vehicle.type,
                    int(vehicle.day),
                    sorted(
                        [sku, int(day), int(qty)]
                        for (sku, day), qty in vehicle.cargo.items()
                    ),
                ],
                separators=(",", ":"),
            ).encode("utf-8")
        )
    return digest.hexdigest()


def test_default_destroy_registration_preserves_the_legacy_trajectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty disable list must leave the archived small run bit-for-bit intact."""
    pinned = {
        "DATASET_TYPE": "small",
        "DATASET_IDX": 1,
        "MAX_ITERATIONS": 12,
        "HARD_ITERATION_BUDGET": True,
        "MAX_RUNTIME": 900.0,
        "DYNAMIC_MAX_ITERATIONS_PER_SECOND": 0.0,
        "DYNAMIC_MAX_ITERATION_BUFFER": 0,
        "DYNAMIC_MAX_ITERATION_SAFETY_FACTOR": 1.0,
        "PROGRESS_BASIS": "iteration",
        "SEED": 42,
        "ENABLE_PARAM_TUNER": False,
        "ENABLE_ML_OPERATORS": False,
        "ENABLE_REPORTS": False,
        "QUIET_OPERATOR_LOGS": True,
        "QUIET_INITIAL_SOLUTION_LOGS": True,
        "INVENTORY_HOLDING_COST": 0.0,
        "DISABLED_DESTROY_OPERATORS": (),
    }
    for name, value in pinned.items():
        monkeypatch.setattr(ALNSConfig, name, value)

    class _SilentPrinter:
        def print(self, *_args, **_kwargs):
            return None

        def print_title(self, *_args, **_kwargs):
            return None

    optimizer = ALNSOptimizer(
        input_file_loc=str(_small_dataset_dir()),
        output_file_loc=str(tmp_path / "legacy"),
    )
    optimizer.log_printer = _SilentPrinter()
    trace: list[float] = []
    original = ALNSTracker.on_iteration

    def recording(self, state, rng, **kwargs):
        result = original(self, state, rng, **kwargs)
        trace.append(round(float(self.best_actual_obj), 6))
        return result

    monkeypatch.setattr(ALNSTracker, "on_iteration", recording)
    assert optimizer.run_optimization("dataset_1") is True

    best = optimizer.best_solution
    assert best is not None
    assert round(float(best.calculate_cost()), 6) == 41.52
    assert _state_fingerprint(best) == (
        "e0377bfd03b99fc4c8f12ccfe7af937114bb1a26da8b94c47e652b43fee9a75d"
    )
    assert trace == [
        51.2,
        43.52,
        41.52,
        41.52,
        41.52,
        41.52,
        41.52,
        41.52,
        41.52,
        41.52,
        41.52,
        41.52,
    ]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------
def test_suppress_noisy_operator_stdout_filters_operator_tags(capsys) -> None:
    with suppress_noisy_operator_stdout(enabled=True):
        print("[OPLOG] hidden")
        print("[DIAG] hidden")
        print("[OPJSON] hidden")
        print("visible")

    captured = capsys.readouterr().out
    assert "visible" in captured
    assert "OPLOG" not in captured
    assert "DIAG" not in captured
    assert "OPJSON" not in captured


def test_suppress_noisy_operator_stdout_passes_everything_when_disabled(capsys) -> None:
    with suppress_noisy_operator_stdout(enabled=False):
        print("[OPLOG] still visible")

    assert "[OPLOG] still visible" in capsys.readouterr().out


def test_suppression_restores_print_after_an_exception() -> None:
    with pytest.raises(RuntimeError):
        with suppress_noisy_operator_stdout(enabled=True):
            raise RuntimeError("boom")

    # print must be the real builtin again.
    assert builtins.print.__module__ == "builtins"


def test_is_file_in_use_is_false_for_missing_and_writable_files(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"
    assert is_file_in_use(str(missing)) is False

    writable = tmp_path / "writable.txt"
    writable.write_text("data", encoding="utf-8")
    assert is_file_in_use(str(writable)) is False


def test_is_file_in_use_detects_a_locked_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    locked = tmp_path / "locked.txt"
    locked.write_text("data", encoding="utf-8")

    def _deny(*args, **kwargs):
        raise PermissionError("locked by another process")

    monkeypatch.setattr(builtins, "open", _deny)

    assert is_file_in_use(str(locked)) is True


def test_safe_state_snapshot_copies_the_state(tmp_path: Path) -> None:
    optimizer = _optimizer(tmp_path)
    optimizer.load_data("dataset_1")
    state = optimizer.create_initial_solution()
    assert state is not None

    snapshot = safe_state_snapshot(state)

    assert snapshot is not None
    assert snapshot is not state
    assert safe_state_snapshot(None) is None


def test_repair_wrapper_exposes_pre_repair_snapshot(tmp_path: Path) -> None:
    optimizer = _optimizer(tmp_path)
    optimizer.load_data("dataset_1")
    state = optimizer_module.SolutionState(optimizer.data)
    seen = {}

    def operator(current, _rng):
        seen["snapshot"] = getattr(current, "_pre_repair_snapshot", None)
        return current

    wrapped = optimizer_module._NoOpFlagPreservingRepair(operator)
    result = wrapped(state, object())

    assert seen["snapshot"] is not None
    assert seen["snapshot"] is not state
    assert seen["snapshot"]._pre_repair_snapshot is None
    assert result._pre_repair_snapshot is None


def test_destroy_wrapper_exposes_pre_destroy_snapshot(tmp_path: Path) -> None:
    optimizer = _optimizer(tmp_path)
    optimizer.load_data("dataset_1")
    state = optimizer_module.SolutionState(optimizer.data)
    seen = {}

    def operator(current, _rng):
        seen["snapshot"] = getattr(current, "_pre_destroy_snapshot", None)
        return current

    wrapped = optimizer_module._PreDestroySnapshot(operator)
    result = wrapped(state, object())

    assert seen["snapshot"] is not None
    assert seen["snapshot"] is not state
    assert seen["snapshot"]._pre_destroy_snapshot is None
    assert result._pre_destroy_snapshot is None


def test_process_results_restores_pre_cross_period_state_when_consolidation_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _State:
        def __init__(self, vehicles: list[str]) -> None:
            self.vehicles = list(vehicles)

        def copy(self):
            return _State(self.vehicles)

        def objective(self) -> float:
            return 10.0

        def calculate_cost(self) -> float:
            return 10.0

        def validate(self):
            return True, {}

    class _Result:
        best_state = _State(["original"])

    class _Tracker:
        initial_search_obj = 20.0
        initial_actual_obj = 20.0

        def get_statistics(self):
            return {
                "best_actual_solution": None,
                "final_gap": -50.0,
                "final_actual_gap": -50.0,
                "total_iterations": 1,
            }

        def _signed_gap_from_initial(self, value, initial):
            return (value - initial) / initial * 100.0

    class _Printer:
        def print(self, *_args, **_kwargs):
            return None

    optimizer = object.__new__(ALNSOptimizer)
    optimizer.result = _Result()
    optimizer.best_feasible_state = None
    optimizer.tracker = _Tracker()
    optimizer.rng = object()
    optimizer.log_printer = _Printer()
    for method_name in (
        "_update_demands",
        "_update_inventory",
        "_output_results",
        "_generate_statistics",
        "_write_repair_audit_report",
    ):
        monkeypatch.setattr(optimizer, method_name, lambda: None)
    monkeypatch.setattr(ALNSConfig, "PREFER_FINAL_BEST_ACTUAL_SOLUTION", False)
    monkeypatch.setattr(ALNSConfig, "FINAL_BEST_ENABLE_CONSOLIDATION", False)

    def _fail_after_mutating(state, *_args, **_kwargs):
        state.vehicles.append("mutated")
        raise RuntimeError("consolidation failed")

    monkeypatch.setattr(
        optimizer_module,
        "cross_period_consolidation_repair",
        _fail_after_mutating,
    )

    optimizer._process_results()

    assert optimizer.best_solution.vehicles == ["original"]


def test_process_results_restores_pre_light_consolidation_state_when_merge_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mid-way merge failure must not keep the half-consolidated incumbent."""

    class _State:
        def __init__(self, vehicles: list[str]) -> None:
            self.vehicles = list(vehicles)

        def copy(self):
            return _State(self.vehicles)

        def objective(self) -> float:
            return 10.0

        def calculate_cost(self) -> float:
            return 10.0

        def validate(self):
            return True, {}

    class _Result:
        best_state = _State(["original"])

    class _Tracker:
        initial_search_obj = 20.0
        initial_actual_obj = 20.0

        def get_statistics(self):
            return {
                "best_actual_solution": None,
                "final_gap": -50.0,
                "final_actual_gap": -50.0,
                "total_iterations": 1,
            }

        def _signed_gap_from_initial(self, value, initial):
            return (value - initial) / initial * 100.0

    class _Printer:
        def __init__(self) -> None:
            self.messages: list[str] = []

        def print(self, message="", **_kwargs):
            self.messages.append(str(message))

    optimizer = object.__new__(ALNSOptimizer)
    optimizer.result = _Result()
    optimizer.best_feasible_state = None
    optimizer.tracker = _Tracker()
    optimizer.rng = object()
    optimizer.log_printer = _Printer()
    for method_name in (
        "_update_demands",
        "_update_inventory",
        "_output_results",
        "_generate_statistics",
        "_write_repair_audit_report",
    ):
        monkeypatch.setattr(optimizer, method_name, lambda: None)
    monkeypatch.setattr(ALNSConfig, "PREFER_FINAL_BEST_ACTUAL_SOLUTION", False)
    monkeypatch.setattr(ALNSConfig, "FINAL_BEST_ENABLE_CONSOLIDATION", True)

    def _fail_after_mutating(state, *_args, **_kwargs):
        state.vehicles.append("half-merged")
        raise RuntimeError("light consolidation failed")

    monkeypatch.setattr(
        optimizer_module, "consolidate_route_vehicles", _fail_after_mutating
    )
    # The later cross-period pass is irrelevant here; keep it a no-op.
    monkeypatch.setattr(
        optimizer_module,
        "cross_period_consolidation_repair",
        lambda state, *_args, **_kwargs: state,
    )

    optimizer._process_results()

    assert optimizer.best_solution.vehicles == ["original"]
    assert any(
        "final light consolidation failed" in message
        for message in optimizer.log_printer.messages
    )


class _LightConsolidationState:
    """Minimal incumbent for the final light-consolidation helper."""

    def __init__(self, vehicles: list[str], feasible: bool = True, cost: float = 10.0):
        self.vehicles = list(vehicles)
        self._feasible = feasible
        self._cost = cost

    def copy(self):
        return _LightConsolidationState(self.vehicles, self._feasible, self._cost)

    def validate(self):
        return self._feasible, {}

    def calculate_cost(self) -> float:
        return self._cost


def _light_consolidation_optimizer(state, messages: list[str]):
    class _Printer:
        def print(self, message="", **_kwargs):
            messages.append(str(message))

    optimizer = object.__new__(ALNSOptimizer)
    optimizer.best_solution = state
    optimizer.log_printer = _Printer()
    return optimizer


def test_final_light_consolidation_rolls_back_when_merged_state_is_infeasible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[str] = []
    state = _LightConsolidationState(["a", "b"])
    optimizer = _light_consolidation_optimizer(state, messages)
    monkeypatch.setattr(ALNSConfig, "FINAL_BEST_ENABLE_CONSOLIDATION", True)

    def _merge_then_break(current_state, max_pairs_per_group=800):
        current_state.vehicles = ["merged"]
        current_state._feasible = False
        return 1

    monkeypatch.setattr(
        optimizer_module, "consolidate_route_vehicles", _merge_then_break
    )

    optimizer._apply_final_light_consolidation()

    assert optimizer.best_solution.vehicles == ["a", "b"]
    assert any("did not improve" in message for message in messages)


def test_final_light_consolidation_rolls_back_a_cost_regression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[str] = []
    state = _LightConsolidationState(["a", "b"], cost=10.0)
    optimizer = _light_consolidation_optimizer(state, messages)
    monkeypatch.setattr(ALNSConfig, "FINAL_BEST_ENABLE_CONSOLIDATION", True)

    def _merge_and_get_expensive(current_state, max_pairs_per_group=800):
        current_state.vehicles = ["merged"]
        current_state._cost = 25.0
        return 1

    monkeypatch.setattr(
        optimizer_module, "consolidate_route_vehicles", _merge_and_get_expensive
    )

    optimizer._apply_final_light_consolidation()

    assert optimizer.best_solution.vehicles == ["a", "b"]
    assert optimizer.best_solution._cost == 10.0


def test_final_light_consolidation_is_skipped_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[str] = []
    state = _LightConsolidationState(["a", "b"])
    optimizer = _light_consolidation_optimizer(state, messages)
    monkeypatch.setattr(ALNSConfig, "FINAL_BEST_ENABLE_CONSOLIDATION", False)

    called = {"merge": False}

    def _merge(current_state, max_pairs_per_group=800):
        called["merge"] = True
        return 1

    monkeypatch.setattr(optimizer_module, "consolidate_route_vehicles", _merge)

    optimizer._apply_final_light_consolidation()

    assert called["merge"] is False
    assert optimizer.best_solution.vehicles == ["a", "b"]


# ---------------------------------------------------------------------------
# The shared short run
# ---------------------------------------------------------------------------


def test_short_run_completes_the_whole_pipeline(short_run: ALNSOptimizer) -> None:
    assert short_run.result is not None
    assert short_run.data is not None
    assert short_run.tracker is not None
    assert short_run.best_solution is not None
    assert short_run.best_solution.validate()[0] is True


def test_short_run_writes_its_result_files(short_run: ALNSOptimizer) -> None:
    output = Path(short_run.data.output_file_loc)

    for name in ("opt_result.csv", "opt_details.csv", "opt_summary.csv"):
        produced = output / name
        assert produced.is_file(), name
        assert produced.stat().st_size > 0


def test_create_result_dataframe_describes_every_loaded_sku(
    short_run: ALNSOptimizer,
) -> None:
    frame = short_run._create_result_dataframe()

    assert list(frame.columns) == [
        "day",
        "plant_code",
        "client_code",
        "product_code",
        "vehicle_id",
        "vehicle_type",
        "qty",
    ]
    assert not frame.empty
    assert (frame["qty"] > 0).all()


def test_generate_reports_can_be_disabled(
    short_run: ALNSOptimizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ALNSConfig, "ENABLE_REPORTS", False)

    # Must be a silent no-op rather than raising or writing figures.
    assert short_run._generate_reports() is None


def test_generate_reports_is_a_noop_without_a_result(short_run: ALNSOptimizer) -> None:
    saved = short_run.result
    short_run.result = None
    try:
        assert short_run._generate_reports() is None
    finally:
        short_run.result = saved


def test_validate_capacity_constraints_accepts_a_feasible_summary(
    short_run: ALNSOptimizer,
) -> None:
    summary = pd.read_csv(Path(short_run.data.output_file_loc) / "opt_summary.csv")
    excess = Path(short_run.data.output_file_loc) / "extra_volume.csv"
    if excess.exists():
        excess.unlink()

    short_run._validate_capacity_constraints(summary)

    # Nothing exceeds capacity, so no excess report is written.
    assert not excess.exists()


def test_validate_capacity_constraints_reports_an_overload(
    short_run: ALNSOptimizer,
) -> None:
    summary = pd.read_csv(Path(short_run.data.output_file_loc) / "opt_summary.csv")
    summary.loc[0, "occupied_size"] = summary.loc[0, "carry_standard_size"] + 5.0

    short_run._validate_capacity_constraints(summary)

    excess = Path(short_run.data.output_file_loc) / "extra_volume.csv"
    assert excess.is_file()
    assert "exceeded_volume" in excess.read_text(encoding="utf-8")


def test_generate_param_tuning_report_writes_a_csv(short_run: ALNSOptimizer) -> None:
    tuner = ParamAutoTuner(short_run.data, short_run.rng)
    saved = short_run.param_tuner
    short_run.param_tuner = tuner
    try:
        short_run._generate_param_tuning_report()
    finally:
        short_run.param_tuner = saved

    report = (
        Path(short_run.output_file_loc)
        / "param_tuning_report"
        / "param_tuning_report.csv"
    )
    assert report.is_file()
    assert "Operator,Parameter,Final Value" in report.read_text(encoding="utf-8")


def test_generate_param_tuning_report_is_a_noop_without_a_tuner(
    short_run: ALNSOptimizer,
) -> None:
    saved = short_run.param_tuner
    short_run.param_tuner = None
    try:
        assert short_run._generate_param_tuning_report() is None
    finally:
        short_run.param_tuner = saved


def test_write_repair_audit_report_handles_a_missing_solution(
    short_run: ALNSOptimizer,
) -> None:
    saved = short_run.best_solution
    short_run.best_solution = None
    try:
        assert short_run._write_repair_audit_report() is None
    finally:
        short_run.best_solution = saved


def test_update_demands_reports_unmet_demand(short_run: ALNSOptimizer) -> None:
    saved = short_run.best_solution
    # An empty fleet leaves every demand unmet, which exercises the branch that
    # writes non_fulfill.csv (a feasible run never reaches it).
    from multi_period_delivery.alns.alnsopt import SolutionState

    short_run.best_solution = SolutionState(short_run.data)
    output = Path(short_run.data.output_file_loc)
    non_fulfill = output / "non_fulfill.csv"
    if non_fulfill.exists():
        non_fulfill.unlink()
    try:
        short_run._update_demands()
    finally:
        short_run.best_solution = saved

    assert non_fulfill.is_file()
    assert non_fulfill.stat().st_size > 0


def test_output_files_are_skipped_while_a_file_is_locked(
    short_run: ALNSOptimizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(optimizer_module, "is_file_in_use", lambda _path: True)

    # Each writer must bail out instead of appending to a locked file.
    assert short_run._update_demands() is None
    assert short_run._update_inventory() is None
    assert short_run._output_results() is None
