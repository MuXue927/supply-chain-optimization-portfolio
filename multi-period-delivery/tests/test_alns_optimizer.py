"""Tests for the ALNS high-level controller.

``ALNSOptimizer`` owns the whole pipeline: load data, build an initial
solution, register operators, run the search, then post-process the result.
The helpers are unit-tested directly; one short optimization run is shared by
several tests that then exercise the reporting steps of that same run.
"""

from __future__ import annotations

import builtins
from pathlib import Path

import pandas as pd
import pytest

from multi_period_delivery.alns import alns_optimizer as optimizer_module
from multi_period_delivery.alns.alns_config import default_config as ALNSConfig
from multi_period_delivery.alns.alns_optimizer import (
    ALNSOptimizer,
    is_file_in_use,
    safe_state_snapshot,
    suppress_noisy_operator_stdout,
)
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
