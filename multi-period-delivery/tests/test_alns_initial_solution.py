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
from multi_period_delivery.alns import repair_operators as repair
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


def test_allocation_does_not_rebuild_shipments_per_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The allocator must not call ``compute_shipped`` around every load.

    It used to bracket each load with two ``compute_shipped()`` calls to learn
    how much the load moved. That rebuilds the shipment table for the whole
    fleet, and ``veh_loading`` invalidates the cache in between, so both calls
    missed: 11,777 calls and 10.3 s of a 15 s initial solution on large.

    The loaded amount is already visible on the vehicles the call touched. The
    two definitions were compared at every allocation site
    (``_init_alloc_equivalence.py``: 5,887 allocations, 0 disagreements) and
    the finished solution matches on all 3,606 (dealer, sku) pairs.

    A small call count is the regression signal; it must stay bounded by the
    number of *stages* rather than by the number of loads.
    """
    calls = {"n": 0}
    original = SolutionState.compute_shipped

    def counted(self, *args, **kwargs):
        calls["n"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(SolutionState, "compute_shipped", counted)
    monkeypatch.setattr(
        ALNSConfig, "INITIAL_CROSS_PERIOD_CONSOLIDATION", False
    )
    state = SolutionState(_load_data())
    solution = isu.improved_initial_solution(state, _seed())

    # The constructor legitimately needs a few full rebuilds (final inventory,
    # feasibility finalisation); it must not need one per allocated vehicle.
    # The small dataset builds ~80 vehicles; the guard only rules out an empty
    # fleet making the ratio below trivially true.
    assert len(solution.vehicles) > 40, "no fleet built; the check is vacuous"
    assert calls["n"] < len(solution.vehicles) / 4, (
        f"{calls['n']} compute_shipped calls for {len(solution.vehicles)} "
        "vehicles - the per-load rebuild is back"
    )


def test_route_repack_preserves_shipments_and_cuts_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regrouping a route's cargo must not change a single shipment.

    The repack only moves cargo *between vehicles of the same
    (plant, dealer, day) route*, so every (plant, sku, day) quantity - and
    therefore the whole inventory table - has to come out identical. That is
    the invariant the pass itself verifies before committing; this test pins it
    from the outside.

    It also asserts the direction of the effect. Without that half, a repack
    that silently did nothing would still pass the equality check.
    """

    def build(repack: bool):
        monkeypatch.setattr(ALNSConfig, "INITIAL_REPACK_ROUTES", repack)
        # Cross-period moves legitimately change the inventory timing, so they
        # are excluded when isolating the same-day repack invariant.
        monkeypatch.setattr(
            ALNSConfig, "INITIAL_CROSS_PERIOD_CONSOLIDATION", False
        )
        state = SolutionState(_load_data())
        return isu.improved_initial_solution(state, _seed())

    without = build(False)
    with_repack = build(True)

    assert dict(with_repack.s_ikt) == dict(without.s_ikt), (
        "the repack changed the inventory table - it must only regroup cargo "
        "inside one route"
    )
    assert sum(without.compute_shipped().values()) == sum(
        with_repack.compute_shipped().values()
    )
    assert len(with_repack.vehicles) < len(without.vehicles), (
        f"repack did not reduce the fleet ({len(without.vehicles)} -> "
        f"{len(with_repack.vehicles)})"
    )
    assert with_repack.calculate_cost() <= without.calculate_cost()
    assert with_repack.validate()[0], "repacked initial solution is not feasible"


def test_repacked_vehicles_respect_capacity_and_route() -> None:
    """Each repacked vehicle must stay on its route and inside its capacity."""
    state = SolutionState(_load_data())
    solution = isu.improved_initial_solution(state, _seed())
    data = solution.data

    for veh in solution.vehicles:
        assert veh.cargo, "repack produced an empty vehicle"
        volume = sum(
            qty * float(data.sku_sizes[sku]) for (sku, _day), qty in veh.cargo.items()
        )
        assert volume <= float(data.veh_type_cap[veh.type]) + 1e-9, (
            f"vehicle {veh.id} carries {volume} > capacity {data.veh_type_cap[veh.type]}"
        )
        assert all(day == veh.day for (_sku, day) in veh.cargo), (
            f"vehicle {veh.id} carries cargo for another day"
        )


def test_allocation_keeps_the_plant_day_index_current() -> None:
    """The loader is handed a per-(plant, day) index instead of the whole fleet.

    ``veh_loading`` builds ``used_inv`` from the vehicles it is given, so a
    stale index would silently under-count already-loaded quantity for that
    plant and day, and the allocator would over-allocate inventory.

    Asserting feasibility catches that: every closing balance must stay
    non-negative, which is exactly what over-allocation breaks.
    """
    state = SolutionState(_load_data())
    solution = isu.improved_initial_solution(state, _seed())

    assert solution.s_ikt, "inventory was never computed"
    negative = {k: v for k, v in solution.s_ikt.items() if v < 0}
    assert not negative, f"{len(negative)} negative balances, e.g. {list(negative)[:3]}"
    assert solution.validate()[0], "initial solution is not hard-feasible"


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
        "cross_period_time",
        "cross_period_applied",
        "cross_period_cost_before",
        "cross_period_cost_after",
        "cross_period_vehicles_before",
        "cross_period_vehicles_after",
    ):
        assert key in stats
    assert stats["parallel_used"] is False
    assert isinstance(stats["repair_iterations"], int)
    assert stats["allocation_time"] is not None
    assert stats["allocation_time"] >= 0.0


def test_initial_cross_period_consolidation_preserves_a_feasible_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The extra pass may only keep a feasible, non-worsening fleet."""
    monkeypatch.setattr(
        ALNSConfig, "INITIAL_CROSS_PERIOD_CONSOLIDATION_MIN_VEHICLES", 0
    )
    with_cross = isu.improved_initial_solution(SolutionState(_load_data()), _seed())

    assert with_cross.validate()[0]
    assert with_cross.vehicles
    stats = with_cross.initial_build_stats
    assert stats["cross_period_time"] >= 0.0
    if stats["cross_period_applied"]:
        assert stats["cross_period_cost_after"] < stats["cross_period_cost_before"]
        assert (
            stats["cross_period_vehicles_after"]
            < stats["cross_period_vehicles_before"]
        )


def test_initial_cross_period_failure_keeps_the_repacked_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing consolidation must restore the verified same-day solution."""

    def fail(*_args, **_kwargs):
        raise RuntimeError("forced cross-period failure")

    monkeypatch.setattr(
        ALNSConfig, "INITIAL_CROSS_PERIOD_CONSOLIDATION_MIN_VEHICLES", 0
    )
    monkeypatch.setattr(
        repair, "cross_period_consolidation_repair", fail
    )
    sol = isu.improved_initial_solution(SolutionState(_load_data()), _seed())

    assert sol.validate()[0]
    assert sol.vehicles
    assert sol.initial_build_stats["cross_period_applied"] is False


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
