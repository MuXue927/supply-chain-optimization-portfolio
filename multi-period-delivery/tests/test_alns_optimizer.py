"""Tests for the ALNS high-level controller.

``ALNSOptimizer`` owns the whole pipeline: load data, build an initial
solution, register operators, run the search, then post-process the result.
The helpers are unit-tested directly, and one short end-to-end run exercises the
pipeline including the registered callbacks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from multi_period_delivery.alns.alns_config import default_config as ALNSConfig
from multi_period_delivery.alns.alns_optimizer import (
    ALNSOptimizer,
    is_file_in_use,
    safe_state_snapshot,
    suppress_noisy_operator_stdout,
)


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
    import builtins

    assert builtins.print.__module__ == "builtins"


def test_is_file_in_use_is_false_for_missing_and_writable_files(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"
    assert is_file_in_use(str(missing)) is False

    writable = tmp_path / "writable.txt"
    writable.write_text("data", encoding="utf-8")
    assert is_file_in_use(str(writable)) is False


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
# Short end-to-end run
# ---------------------------------------------------------------------------
def test_short_run_completes_the_whole_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The default budget (300 iterations / 900 s) would make CI unusably slow.
    #
    # ``run_optimization`` widens the stopping rule's iteration cap to a
    # runtime-derived floor (`runtime * iterations-per-second * safety factor`,
    # plus a buffer), which would otherwise let a 3-second runtime budget - and
    # therefore the number of iterations actually executed - depend on how fast
    # the machine is. Setting the dynamic knobs to zero pins the floor to the
    # iteration budget, so the run stops after a fixed number of iterations and
    # the coverage it contributes stays reproducible across machines.
    monkeypatch.setattr(ALNSConfig, "MAX_ITERATIONS", 120)
    monkeypatch.setattr(ALNSConfig, "MAX_RUNTIME", 120)
    monkeypatch.setattr(ALNSConfig, "DYNAMIC_MAX_ITERATIONS_PER_SECOND", 0)
    monkeypatch.setattr(ALNSConfig, "DYNAMIC_MAX_ITERATION_SAFETY_FACTOR", 0)
    monkeypatch.setattr(ALNSConfig, "DYNAMIC_MAX_ITERATION_BUFFER", 0)

    optimizer = _optimizer(tmp_path)

    assert optimizer.run_optimization("dataset_1") is True
    assert optimizer.result is not None
    assert optimizer.data is not None
    # The run must leave a record of the best solution it found.
    assert (
        optimizer.best_feasible_state is not None
        or optimizer.best_solution is not None
    )
