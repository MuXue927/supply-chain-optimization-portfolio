"""Focused tests for the ALNS initial-solution constructor.

These cover ``initial_solution_utils.improved_initial_solution`` and the pure
helper functions it composes. The constructor is the starting point of every
ALNS run, so the invariants tested here (integer inventory, no negative stock,
a populated fleet, an attached build-stats record) are the ones a broken
starting point would violate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from multi_period_delivery.alns import initial_solution_utils as isu
from multi_period_delivery.alns.alns_config import default_config as ALNSConfig
from multi_period_delivery.alns.alnsopt import SolutionState
from multi_period_delivery.alns.InputDataALNS import DataALNS


def _load_data(dataset: str = "dataset_1") -> DataALNS:
    base = Path(__file__).resolve().parents[1]
    data = DataALNS(
        str(base / "datasets" / "multiple-periods" / "small"),
        str(base / "tmp"),
        dataset,
    )
    data.load()
    return data


def _seed() -> np.random.Generator:
    # A fixed seed keeps allocation order (and therefore the fleet) stable.
    return np.random.default_rng(7)


def test_improved_initial_solution_is_feasible_and_integer() -> None:
    data = _load_data()
    state = SolutionState(data)

    sol = isu.improved_initial_solution(state, _seed())

    assert isinstance(sol, SolutionState)
    assert sol.vehicles, "the constructor should build a non-empty fleet"

    feasible, violations = sol.validate()
    assert not violations["negative_inventory"]
    assert not violations["veh_over_load"]
    assert not violations["cargo_day_mismatch"]
    assert feasible

    # Inventory is accumulated from integer production and integer loads, so it
    # must stay integral (no fractional drift).
    assert all(float(value).is_integer() for value in sol.s_ikt.values())


def test_improved_initial_solution_attaches_build_stats() -> None:
    data = _load_data()
    sol = isu.improved_initial_solution(SolutionState(data), _seed())

    stats = sol.initial_build_stats
    for key in (
        "resource_pool_time",
        "prioritize_time",
        "allocation_time",
        "repair_time",
        "parallel_used",
        "repair_iterations",
    ):
        assert key in stats
    assert stats["parallel_used"] is False
    assert isinstance(stats["repair_iterations"], int)
    assert stats["allocation_time"] is not None
    assert stats["allocation_time"] >= 0.0


def test_improved_initial_solution_survives_a_tiny_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A near-zero budget forces the timeout branches inside allocation/repair.
    # The constructor must still return a usable state instead of raising.
    monkeypatch.setattr(ALNSConfig, "MIN_INIT_TIMEOUT", 1)
    monkeypatch.setattr(ALNSConfig, "MAX_INIT_TIMEOUT", 1)
    monkeypatch.setattr(ALNSConfig, "INIT_TIMEOUT_PER_DEMAND", 0.0)

    sol = isu.improved_initial_solution(SolutionState(_load_data()), _seed())

    assert isinstance(sol, SolutionState)
    assert isinstance(sol.vehicles, list)
    assert "allocation_time" in sol.initial_build_stats


def test_prioritize_demands_is_positive_and_sorted() -> None:
    data = _load_data()

    prioritized = isu._prioritize_demands(data)

    assert prioritized
    assert all(demand > 0 for _key, demand, _score in prioritized)
    scores = [score for _key, _demand, score in prioritized]
    assert scores == sorted(scores, reverse=True)


def test_resource_pool_accumulates_production_over_days() -> None:
    data = _load_data()

    pool = isu._initialize_resource_pool(data)

    plant = next(iter(data.plants))
    sku = next(iter(data.skus_plant[plant]))
    # Day 0 is the initial inventory; later days add that day's production.
    assert pool[(plant, sku, 0)] == data.sku_initial_inv[(plant, sku)]
    for day in range(1, data.horizons + 1):
        expected = pool[(plant, sku, day - 1)] + data.sku_prod_each_day.get(
            (plant, sku, day), 0
        )
        assert pool[(plant, sku, day)] == expected


def test_select_best_vehicle_uses_smallest_fitting_type() -> None:
    data = _load_data()
    sku = next(iter(data.all_skus))

    # A single unit needs almost no volume: the smallest type that fits wins.
    small_type = isu._select_best_vehicle(1, sku, data)
    assert data.veh_type_cap[small_type] >= data.sku_sizes[sku]

    # A quantity that no single vehicle can carry falls back to the largest type.
    largest_type = max(data.veh_type_cap, key=lambda vt: data.veh_type_cap[vt])
    huge_qty = int(
        (max(data.veh_type_cap.values()) + 1) / max(data.sku_sizes[sku], 1e-9)
    )
    assert isu._select_best_vehicle(huge_qty, sku, data) == largest_type
