"""Every phase schedule must follow the configured progress basis.

M69 found that ``ParamAutoTuner.adaptive_degree`` kept reading
``elapsed / MAX_RUNTIME`` even when ``PROGRESS_BASIS = "iteration"``, so the
same seed produced 782.67 and 799.63 on a fixed-iteration protocol. These tests
pin the dispatch, and pin that the production (runtime) path is unchanged.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pytest

from multi_period_delivery.alns.alns_config import default_config as ALNSConfig
from multi_period_delivery.alns.alns_optimizer import ALNSOptimizer
from multi_period_delivery.alns.InputDataALNS import DataALNS
from multi_period_delivery.alns.param_tuner import ParamAutoTuner
from multi_period_delivery.alns.progress import progress_from_elapsed


def _tuner() -> ParamAutoTuner:
    base = Path(__file__).resolve().parents[1]
    data = DataALNS(
        str(base / "datasets" / "multiple-periods" / "small"),
        str(base / "tmp"),
        "dataset_1",
    )
    data.load()
    return ParamAutoTuner(data, np.random.default_rng(7))


def test_adaptive_degree_runtime_basis_keeps_the_elapsed_formula(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production path must stay bit-for-bit the old formula."""
    monkeypatch.setattr(ALNSConfig, "PROGRESS_BASIS", "runtime")
    tuner = _tuner()
    tuner.set_iteration(180, 360, no_improve_iters=0)
    tuner.set_elapsed_clock(lambda: 250.0)

    expected = 0.25 * (1.0 - 0.67 * min(1.0, progress_from_elapsed(250.0) / 0.7))

    assert tuner.adaptive_degree(0.25) == pytest.approx(expected, abs=1e-12)


def test_adaptive_degree_iteration_basis_ignores_the_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same iteration, wildly different clocks: the degree must not move."""
    monkeypatch.setattr(ALNSConfig, "PROGRESS_BASIS", "iteration")
    fast, slow = _tuner(), _tuner()
    for tuner in (fast, slow):
        tuner.set_iteration(90, 360, no_improve_iters=0)
    fast.set_elapsed_clock(lambda: 1.0)
    slow.set_elapsed_clock(lambda: 800.0)

    assert fast.adaptive_degree(0.25) == slow.adaptive_degree(0.25)

    # ... and it follows the iteration ratio, not the clock.
    expected = 0.25 * (1.0 - 0.67 * (90 / 360) / 0.7)
    assert fast.adaptive_degree(0.25) == pytest.approx(expected, abs=1e-12)


def test_warmup_budget_is_wall_clock_only_under_the_runtime_basis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from multi_period_delivery.alns.ml_operator_selector import MLOperatorSelector

    class _Tuner:
        operators: dict = {}

        def set_iteration(self, *_args, **_kwargs):
            return None

    selector = MLOperatorSelector(_Tuner(), np.random.default_rng(3))

    monkeypatch.setattr(ALNSConfig, "PROGRESS_BASIS", "runtime")
    monkeypatch.setattr(ALNSConfig, "MAX_RUNTIME", 900.0)
    monkeypatch.setattr(ALNSConfig, "ML_WARMUP_BUDGET_FRACTION", 0.15)
    selector.warmup_seconds = 0.0
    assert selector.warmup_budget_exhausted() is False
    selector.warmup_seconds = 135.0
    assert selector.warmup_budget_exhausted() is True

    # Under the iteration basis the gate must read iterations, so a slow host
    # cannot stop warm-up early on a fixed-iteration run.
    monkeypatch.setattr(ALNSConfig, "PROGRESS_BASIS", "iteration")
    selector.warmup_seconds = 10_000.0
    selector.set_iteration(10, 360)
    assert selector.warmup_budget_exhausted() is False
    selector.set_iteration(60, 360)
    assert selector.warmup_budget_exhausted() is True


def _fingerprint(state) -> str:
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


def _run_full_small(tmp_path: Path, delay_ms: float) -> tuple[float, str, list[float]]:
    """One short full-config run; optionally slow the objective down."""
    from multi_period_delivery.alns.alnsopt import SolutionState

    iterations = 12
    # ALNSConfig is a shared singleton: restore it or every later test in the
    # session inherits "iteration" and the runtime-basis tests fail.
    saved = {
        key: getattr(ALNSConfig, key)
        for key in (
            "DATASET_TYPE",
            "DATASET_IDX",
            "MAX_ITERATIONS",
            "HARD_ITERATION_BUDGET",
            "MAX_RUNTIME",
            "DYNAMIC_MAX_ITERATIONS_PER_SECOND",
            "DYNAMIC_MAX_ITERATION_BUFFER",
            "DYNAMIC_MAX_ITERATION_SAFETY_FACTOR",
            "PROGRESS_BASIS",
            "SEED",
            "ENABLE_PARAM_TUNER",
            "ENABLE_ML_OPERATORS",
            "ML_SELECTOR_KIND",
            "ENABLE_REPORTS",
            "QUIET_OPERATOR_LOGS",
            "QUIET_INITIAL_SOLUTION_LOGS",
        )
    }
    ALNSConfig.update_from_dict(
        {
            "DATASET_TYPE": "small",
            "DATASET_IDX": 1,
            "MAX_ITERATIONS": iterations,
            "HARD_ITERATION_BUDGET": True,
            "MAX_RUNTIME": 900.0,
            "DYNAMIC_MAX_ITERATIONS_PER_SECOND": 0.0,
            "DYNAMIC_MAX_ITERATION_BUFFER": 0,
            "DYNAMIC_MAX_ITERATION_SAFETY_FACTOR": 1.0,
            "PROGRESS_BASIS": "iteration",
            "SEED": 42,
            "ENABLE_PARAM_TUNER": True,
            "ENABLE_ML_OPERATORS": True,
            "ML_SELECTOR_KIND": "bandit",
            "ENABLE_REPORTS": False,
            "QUIET_OPERATOR_LOGS": True,
            "QUIET_INITIAL_SOLUTION_LOGS": True,
        }
    )

    degrees: list[float] = []
    original_degree = ParamAutoTuner.adaptive_degree
    original_objective = SolutionState.objective
    active = {"on": True}

    def recording(self, *args, **kwargs):
        value = original_degree(self, *args, **kwargs)
        degrees.append(round(float(value), 12))
        return value

    def delayed(self, *args, **kwargs):
        if active["on"] and delay_ms > 0:
            time.sleep(delay_ms / 1000.0)
        return original_objective(self, *args, **kwargs)

    ParamAutoTuner.adaptive_degree = recording
    if delay_ms > 0:
        SolutionState.objective = delayed
    optimizer = ALNSOptimizer(
        input_file_loc=str(
            Path(__file__).resolve().parents[1]
            / "datasets"
            / "multiple-periods"
            / "small"
        ),
        output_file_loc=str(tmp_path / f"delay{delay_ms}"),
    )
    try:
        assert optimizer.run_optimization("dataset_1") is True
    finally:
        active["on"] = False
        ParamAutoTuner.adaptive_degree = original_degree
        SolutionState.objective = original_objective
        for key, value in saved.items():
            setattr(ALNSConfig, key, value)

    best = optimizer.best_solution
    assert best is not None
    return float(best.calculate_cost()), _fingerprint(best), degrees


def test_iteration_basis_full_run_is_immune_to_an_objective_delay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slower objective must not change a single degree decision."""
    monkeypatch.setattr(ALNSConfig, "ML_BANDIT_MIN_RUNTIME_SECONDS", 0.0)
    clean = _run_full_small(tmp_path, 0.0)
    delayed = _run_full_small(tmp_path, 15.0)

    assert clean[2], "no adaptive_degree decisions were recorded"
    assert clean[2] == delayed[2]
    assert clean[0] == delayed[0]
    assert clean[1] == delayed[1]
