"""Tests for the optional inventory holding-cost term."""

from __future__ import annotations

from pathlib import Path

import pytest

from multi_period_delivery.alns.alns_config import default_config as ALNSConfig
from multi_period_delivery.alns.alns_optimizer import ALNSOptimizer
from multi_period_delivery.alns.alnsopt import SolutionState
from multi_period_delivery.alns.repair_utils import (
    compute_improvement_delta,
    compute_improvement_fast_add_vehicle,
)


def _state() -> SolutionState:
    base = Path(__file__).resolve().parents[1]
    optimizer = ALNSOptimizer(
        input_file_loc=str(base / "datasets" / "multiple-periods" / "small"),
        output_file_loc=str(base / "tmp" / "m67-holding-test"),
    )
    assert optimizer.load_data("dataset_1")
    state = optimizer.create_initial_solution()
    assert state is not None
    return state


def _inventory_volume_days(state: SolutionState) -> float:
    total = 0.0
    for (_plant, sku, day), quantity in state.s_ikt.items():
        if day > 0 and quantity > 0:
            total += float(quantity) * float(state.data.sku_sizes[sku])
    return total


def test_zero_rate_preserves_the_archived_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ALNSConfig, "INVENTORY_HOLDING_COST", 0.0)
    state = _state()

    # This is the recorded dataset_1 initial-solution objective before the
    # holding-cost field existed. Any drift here is a legacy-baseline break.
    assert state.actual_objective() == 43.52
    assert state.objective() == 43.52


def test_positive_rate_adds_volume_days(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ALNSConfig, "INVENTORY_HOLDING_COST", 0.0)
    state = _state()
    baseline = state.actual_objective()
    volume_days = _inventory_volume_days(state)

    monkeypatch.setattr(ALNSConfig, "INVENTORY_HOLDING_COST", 0.001)
    state.mark_objective_dirty()
    expected = baseline + 0.001 * volume_days

    assert state.actual_objective() == pytest.approx(expected, abs=1e-9)
    assert state.objective() == pytest.approx(expected, abs=1e-9)


def test_holding_cost_defaults_off() -> None:
    assert ALNSConfig.INVENTORY_HOLDING_COST == 0.0


def test_fast_and_delta_estimators_include_holding_saving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ALNSConfig, "INVENTORY_HOLDING_COST", 0.0)
    state = _state()
    state.compute_inventory()
    state.compute_shipped()

    dealer, sku = next(iter(state.data.demands))
    plant = next(
        plant
        for plant in state.data.plants
        if sku in state.data.skus_plant.get(plant, ())
    )
    veh_type = next(iter(state.data.veh_type_cap))
    day = 2
    qty = 5
    periods = state.data.horizons - day + 1
    sku_size = state.data.sku_sizes[sku]

    _, fast_zero, _ = compute_improvement_fast_add_vehicle(
        state, plant, dealer, veh_type, day, sku, qty
    )
    prev_zero = state.objective()
    _, delta_zero, _ = compute_improvement_delta(
        state, prev_zero, dealer, sku, qty, veh_type, day=day
    )

    rate = 0.01
    monkeypatch.setattr(ALNSConfig, "INVENTORY_HOLDING_COST", rate)
    state.mark_objective_dirty()
    prev_held = state.objective()
    _, fast_held, _ = compute_improvement_fast_add_vehicle(
        state, plant, dealer, veh_type, day, sku, qty
    )
    _, delta_held, _ = compute_improvement_delta(
        state, prev_held, dealer, sku, qty, veh_type, day=day
    )

    saving = rate * qty * sku_size * periods
    assert fast_held - fast_zero == pytest.approx(saving, abs=1e-9)
    assert delta_held - delta_zero == pytest.approx(saving, abs=1e-9)
