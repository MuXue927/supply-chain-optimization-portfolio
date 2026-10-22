"""Focused tests for destroy/repair post-processing invariants."""

from __future__ import annotations

import math
from pathlib import Path

from multi_period_delivery.alns import repair_utils
from multi_period_delivery.alns.alns_config import default_config as ALNSConfig
from multi_period_delivery.alns.alnsopt import SolutionState
from multi_period_delivery.alns.destroy_utils import (
    compute_improvement_and_report,
    post_destroy,
    summarize_removed_cargo,
)
from multi_period_delivery.alns.InputDataALNS import DataALNS
from multi_period_delivery.alns.repair_utils import (
    consolidate_route_vehicles,
    post_repair,
    veh_loading,
)
from multi_period_delivery.alns.snapshot import safe_state_snapshot
from multi_period_delivery.alns.vehicle import Vehicle


def _load_state() -> SolutionState:
    base = Path(__file__).resolve().parents[1]
    data = DataALNS(
        str(base / "datasets" / "multiple-periods" / "small"),
        str(base / "tmp"),
        "dataset_1",
    )
    data.load()
    return SolutionState(data)


def _assert_inventory_matches_fresh(state: SolutionState) -> None:
    expected = state.copy()
    expected.compute_inventory()
    assert state.s_ikt == expected.s_ikt
    assert all(float(value).is_integer() for value in state.s_ikt.values())


def _vehicle_with_load(
    state: SolutionState,
    *,
    day: int = 5,
    sku_id: str = "O7132564089",
    qty: int = 20,
) -> Vehicle:
    vehicle = Vehicle("8070", "H2854679", "10.3", day, state.data)
    vehicle.load(sku_id, qty)
    return vehicle


def test_destroy_repair_round_trip_matches_fresh_inventory() -> None:
    state = _load_state()
    vehicle = _vehicle_with_load(state)
    state.vehicles = [vehicle]
    state.mark_objective_dirty()
    state.compute_inventory()

    state.vehicles = []
    post_destroy(
        state,
        [vehicle],
        "random_removal",
        early=True,
        prev_obj=1.0,
    )
    _assert_inventory_matches_fresh(state)

    replacement = Vehicle("8070", "H2854679", "10.3", 5, state.data)
    veh_loading(
        state,
        replacement,
        {"O7132564089": 20},
        commit=True,
        skip_compute_inventory=True,
    )
    _assert_inventory_matches_fresh(state)

    post_repair(
        state,
        [replacement],
        op_name="round_trip_repair",
        prev_obj=1.0,
    )
    _assert_inventory_matches_fresh(state)


def test_post_destroy_does_not_double_restore_removed_cargo() -> None:
    state = _load_state()
    vehicle = _vehicle_with_load(state)
    state.vehicles = [vehicle]
    state.mark_objective_dirty()
    state.compute_inventory()

    removed_summary = summarize_removed_cargo([vehicle])
    duplicate_partial_updates = {
        (plant, sku_id, day): qty
        for (plant, sku_id, shipped_day), qty in removed_summary.items()
        for day in range(shipped_day, state.data.horizons + 1)
    }

    state.vehicles = []
    post_destroy(
        state,
        [vehicle],
        "periodic_shaw_removal",
        early=True,
        prev_obj=1.0,
        extra={"partial_updates": duplicate_partial_updates},
    )

    _assert_inventory_matches_fresh(state)


def test_post_destroy_preserves_distinct_partial_rollback_on_overlap() -> None:
    state = _load_state()
    removed = _vehicle_with_load(state, qty=10)
    remaining = _vehicle_with_load(state, qty=5)
    state.vehicles = [removed, remaining]
    state.mark_objective_dirty()
    state.compute_inventory()

    # Simulate a partial rollback on a vehicle that remains in the solution.
    remaining.cargo[("O7132564089", 5)] = 2
    state.vehicles = [remaining]
    post_destroy(
        state,
        [removed],
        "infeasible_removal",
        early=True,
        prev_obj=1.0,
        extra={
            "partial_updates": {
                ("8070", "O7132564089", 5): 3,
            }
        },
    )

    _assert_inventory_matches_fresh(state)


def test_physically_feasible_partial_repair_is_not_rolled_back() -> None:
    state = _load_state()
    vehicle = _vehicle_with_load(state)
    state.vehicles = [vehicle]

    post_repair(
        state,
        [vehicle],
        op_name="partial_repair",
        prev_obj=1.0,
    )

    assert vehicle in state.vehicles
    audit = state.repair_audit["partial_repair"]
    assert audit["infeasible"] == 1
    assert audit["rolled_back"] == 0
    assert not state.validate()[1]["negative_inventory"]


def test_negative_inventory_repair_is_rolled_back() -> None:
    state = _load_state()
    vehicle = Vehicle("8070", "H2854679", "10.3", 1, state.data)
    vehicle.cargo[("C3098216457", 1)] = 100
    state.vehicles = [vehicle]

    post_repair(
        state,
        [vehicle],
        op_name="invalid_repair",
        prev_obj=1.0,
    )

    assert vehicle not in state.vehicles
    audit = state.repair_audit["invalid_repair"]
    assert audit["rolled_back"] == 1
    assert audit["fail_negative_inventory"] == 1
    assert not state.validate()[1]["negative_inventory"]


def test_physical_rollback_is_checked_without_strict_demand(
    monkeypatch,
) -> None:
    monkeypatch.setattr(ALNSConfig, "STRICT_FEASIBILITY_MODE", False)
    monkeypatch.setattr(ALNSConfig, "VALIDATE_REQUIRE_FULL_DEMAND", False)

    state = _load_state()
    vehicle = Vehicle("8070", "H2854679", "10.3", 1, state.data)
    vehicle.cargo[("C3098216457", 1)] = 100
    state.vehicles = [vehicle]

    post_repair(
        state,
        [vehicle],
        op_name="non_strict_invalid_repair",
        prev_obj=1.0,
    )

    assert vehicle not in state.vehicles
    assert state.repair_audit["non_strict_invalid_repair"]["rolled_back"] == 1
    assert not state.validate()[1]["negative_inventory"]


def test_post_repair_marks_existing_only_violation_uncertified_without_snapshot(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        repair_utils,
        "consolidate_route_vehicles",
        lambda _state, max_pairs_per_group=300: 0,
    )
    monkeypatch.setattr(
        "multi_period_delivery.alns.repair_operators.fill_minimum_load_repair",
        lambda current, _rng: current,
    )

    state = _load_state()
    vehicle = _vehicle_with_load(state, qty=20)
    vehicle.cargo[("O7132564089", 5)] += 10_000
    state.vehicles = [vehicle]

    post_repair(
        state,
        [],
        op_name="existing_without_snapshot",
        prev_obj=1.0,
    )

    assert vehicle in state.vehicles
    assert state._post_repair_certified is False
    assert "originated in existing vehicles" in state._post_repair_certification_error
    audit = state.repair_audit["existing_without_snapshot"]
    assert audit["rolled_back"] == 0
    assert audit["post_repair_uncertified"] == 1


def test_post_repair_restores_existing_vehicle_changes_from_pre_repair_snapshot(
    monkeypatch,
) -> None:
    monkeypatch.setattr(ALNSConfig, "STRICT_FEASIBILITY_MODE", False)
    monkeypatch.setattr(ALNSConfig, "VALIDATE_REQUIRE_FULL_DEMAND", False)
    monkeypatch.setattr(
        repair_utils,
        "consolidate_route_vehicles",
        lambda _state, max_pairs_per_group=300: 0,
    )
    monkeypatch.setattr(
        "multi_period_delivery.alns.repair_operators.fill_minimum_load_repair",
        lambda current, _rng: current,
    )

    state = _load_state()
    vehicle = _vehicle_with_load(state, qty=20)
    state.vehicles = [vehicle]
    state.mark_objective_dirty()
    state.compute_inventory()
    original_cargo = dict(vehicle.cargo)
    state._pre_repair_snapshot = safe_state_snapshot(state)

    vehicle.cargo[("O7132564089", 5)] += 10_000
    post_repair(
        state,
        [],
        op_name="existing_vehicle_rollback",
        prev_obj=1.0,
    )

    assert state.vehicles[0].cargo == original_cargo
    assert state._post_repair_certified is True
    assert state._post_repair_certification_error is None
    assert state.repair_audit["existing_vehicle_rollback"]["rolled_back"] == 1
    assert (
        state.repair_audit["existing_vehicle_rollback"][
            "post_repair_uncertified"
        ]
        == 0
    )


def test_post_repair_marks_uncertified_when_rollback_inventory_rebuild_fails(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        repair_utils,
        "consolidate_route_vehicles",
        lambda _state, max_pairs_per_group=300: 0,
    )
    monkeypatch.setattr(
        "multi_period_delivery.alns.repair_operators.fill_minimum_load_repair",
        lambda current, _rng: current,
    )

    state = _load_state()
    vehicle = _vehicle_with_load(state, qty=20)
    state.vehicles = [vehicle]
    state.mark_objective_dirty()

    def raising_compute_inventory():
        raise ValueError("inventory rebuild failed")

    monkeypatch.setattr(state, "compute_inventory", raising_compute_inventory)

    post_repair(
        state,
        [vehicle],
        op_name="rollback_inventory_failure",
        prev_obj=1.0,
    )

    assert state.vehicles == []
    assert state._post_repair_certified is False
    assert "rollback inventory rebuild" in state._post_repair_certification_error
    audit = state.repair_audit["rollback_inventory_failure"]
    assert audit["rolled_back"] == 1
    assert audit["post_repair_uncertified"] == 1


def test_post_repair_marks_uncertified_when_post_rollback_validation_raises(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        repair_utils,
        "consolidate_route_vehicles",
        lambda _state, max_pairs_per_group=300: 0,
    )
    monkeypatch.setattr(
        "multi_period_delivery.alns.repair_operators.fill_minimum_load_repair",
        lambda current, _rng: current,
    )
    monkeypatch.setattr(
        repair_utils,
        "_check_inserted_feasibility",
        lambda _state, _inserted: (False, {"negative_inventory": ["forced"]}),
    )

    state = _load_state()
    vehicle = _vehicle_with_load(state, qty=20)
    state.vehicles = [vehicle]
    validate_calls = 0

    def fail_after_rollback():
        nonlocal validate_calls
        validate_calls += 1
        if validate_calls == 1:
            return False, {"negative_inventory": ["forced"]}
        raise RuntimeError("post-rollback validation failed")

    monkeypatch.setattr(state, "validate", fail_after_rollback)

    post_repair(
        state,
        [vehicle],
        op_name="rollback_validation_failure",
        prev_obj=1.0,
    )

    assert state.vehicles == []
    assert state._post_repair_certified is False
    assert "post-rollback validation" in state._post_repair_certification_error
    audit = state.repair_audit["rollback_validation_failure"]
    assert audit["rolled_back"] == 1
    assert audit["post_repair_uncertified"] == 1


def test_post_repair_fills_before_consolidation_and_preserves_capacity(
    monkeypatch,
) -> None:
    state = _load_state()
    vehicles = [_vehicle_with_load(state), _vehicle_with_load(state)]
    state.vehicles = vehicles
    observed_underload_counts: list[int] = []
    original_consolidate = repair_utils.consolidate_route_vehicles

    def _consolidation_spy(current_state, max_pairs_per_group=300):
        underload = sum(
            1
            for vehicle in current_state.vehicles
            if current_state.compute_veh_load(vehicle)
            + 1e-9
            < current_state.data.veh_type_min_load.get(vehicle.type, 0.0)
        )
        observed_underload_counts.append(underload)
        return original_consolidate(
            current_state, max_pairs_per_group=max_pairs_per_group
        )

    monkeypatch.setattr(
        repair_utils,
        "consolidate_route_vehicles",
        _consolidation_spy,
    )

    post_repair(
        state,
        vehicles,
        op_name="minimum_load_repair",
        prev_obj=1.0,
    )

    assert observed_underload_counts
    assert observed_underload_counts[0] == 0
    for vehicle in state.vehicles:
        loaded = state.compute_veh_load(vehicle)
        min_load = state.data.veh_type_min_load.get(vehicle.type, 0.0)
        capacity = state.data.veh_type_cap.get(vehicle.type, 0.0)
        assert loaded + 1e-9 >= min_load
        assert loaded <= capacity + 1e-9


def test_consolidation_upgrade_preserves_target_vehicle_identity() -> None:
    state = _load_state()
    target = _vehicle_with_load(state, qty=73)
    source = _vehicle_with_load(state, qty=2)
    state.vehicles = [target, source]
    source_cargo = source.cargo

    merged_count = consolidate_route_vehicles(state)

    assert merged_count == 1
    assert len(state.vehicles) == 1
    merged = state.vehicles[0]
    assert merged.id == target.id
    assert merged.type == "11.4"
    assert merged.capacity >= 0
    assert math.isclose(state.compute_veh_load(merged), 120.0, abs_tol=1e-9)
    assert merged.cargo is not source_cargo
    assert merged.cargo[("O7132564089", 5)] == 75


def test_operator_reporting_ignores_non_finite_objective() -> None:
    class _Tuner:
        def __init__(self) -> None:
            self.calls: list[tuple[str, float, bool]] = []

        def update_operator_performance(
            self, op_name: str, improvement: float, success: bool = False
        ) -> None:
            self.calls.append((op_name, improvement, success))

        def adjust_operator_params(
            self,
            op_name: str,
            improvement: float,
            success: bool = False,
            skip_update: bool = False,
        ) -> None:
            return None

    class _InfiniteState:
        def __init__(self) -> None:
            self.param_tuner = _Tuner()

        def objective(self) -> float:
            return float("inf")

        def calculate_cost(self) -> float:
            return float("inf")

    state = _InfiniteState()
    new_obj, improvement, success = compute_improvement_and_report(
        state, prev_obj=10.0, op_name="non_finite_operator"
    )

    assert new_obj == float("inf")
    assert math.isfinite(improvement)
    assert improvement == 0.0
    assert success is False
    assert state.param_tuner.calls == [("non_finite_operator", 0.0, False)]
