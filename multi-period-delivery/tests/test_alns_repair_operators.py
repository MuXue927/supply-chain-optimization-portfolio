"""Focused tests for the ALNS repair operators and their helpers.

A repair operator receives a partially destroyed solution and puts cargo back
without ever producing a hard-infeasible state: no negative inventory, no
vehicle overload, and no cargo attributed to the wrong day. Those are the
invariants asserted here for every operator, on a real small-dataset solution.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from multi_period_delivery.alns import destroy_operators as destroy
from multi_period_delivery.alns import repair_operators as repair
from multi_period_delivery.alns import repair_utils
from multi_period_delivery.alns.alns_config import default_config as ALNSConfig
from multi_period_delivery.alns.alnsopt import SolutionState
from multi_period_delivery.alns.initial_solution_utils import improved_initial_solution
from multi_period_delivery.alns.InputDataALNS import DataALNS
from multi_period_delivery.alns.repair_utils import ResourcePool, get_unmet_demand
from multi_period_delivery.alns.vehicle import Vehicle


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
def feasible_solution() -> SolutionState:
    """A real feasible solution, built once and copied by each test."""
    state = SolutionState(_load_data())
    return improved_initial_solution(state, np.random.default_rng(7))


def _assert_still_consistent(state: SolutionState) -> None:
    """``s_ikt`` must match a fresh recomputation; hard violations must be empty."""
    expected = state.copy()
    expected.compute_inventory()
    assert state.s_ikt == expected.s_ikt

    _feasible, violations = state.validate()
    assert not violations["negative_inventory"]
    assert not violations["veh_over_load"]
    assert not violations["cargo_day_mismatch"]


def test_fleet_consolidation_returns_the_original_when_post_repair_raises(
    feasible_solution: SolutionState, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An uncertified merge must never be handed back as a candidate.

    ``post_repair`` is the step that certifies the merged state. When it
    raises, ``_post_repair_certified`` stays unset, and the acceptor reads an
    unset flag as "no objection" - so returning the merge would let an
    unverified consolidation through.
    """
    state = feasible_solution.copy()
    layout = _vehicle_layout(state)
    recorded: list[str] = []

    def _explode(*_args, **_kwargs):
        raise RuntimeError("post_repair exploded")

    # ``fleet_consolidation_repair`` imports ``post_repair`` locally, so the
    # patch has to land on the defining module.
    monkeypatch.setattr(repair_utils, "post_repair", _explode)
    monkeypatch.setattr(
        repair, "record_swallowed", lambda site, exc: recorded.append(site)
    )

    result = repair.fleet_consolidation_repair(state, np.random.default_rng(3))

    assert _vehicle_layout(result) == layout
    assert result._repair_noop is True
    assert "repair_operators.fleet_consolidation_repair.post_repair" in recorded


def _vehicle_layout(state: SolutionState) -> list[tuple]:
    return sorted(
        (
            vehicle.fact_id,
            vehicle.dealer_id,
            vehicle.type,
            vehicle.day,
            tuple(sorted(vehicle.cargo.items())),
        )
        for vehicle in state.vehicles
    )


def _destroyed(state: SolutionState) -> SolutionState:
    """Return a copy with roughly a quarter of the fleet removed.

    Destroy operators build their result on a copy and return it, so the
    returned state is the partial solution, not the state passed in.
    """
    return destroy.random_removal(
        state.copy(), np.random.default_rng(3), degree=0.25
    )


# Every operator takes ``(state, rng, ...)`` and returns the (mutated) state.
REPAIR_OPERATORS = {
    "greedy_repair": lambda s, r: repair.greedy_repair(s, r),
    "fill_minimum_load_repair": lambda s, r: repair.fill_minimum_load_repair(s, r),
    "cross_period_consolidation_repair": (
        lambda s, r: repair.cross_period_consolidation_repair(s, r)
    ),
    "local_search_repair": lambda s, r: repair.local_search_repair(s, r, max_iter=20),
    "inventory_balance_repair": lambda s, r: repair.inventory_balance_repair(s, r),
    "infeasible_repair": lambda s, r: repair.infeasible_repair(s, r),
    "smart_batch_repair": lambda s, r: repair.smart_batch_repair(s, r, max_iter=20),
    "regret_based_repair": lambda s, r: repair.regret_based_repair(s, r, k=2, topN=6),
    "fleet_consolidation_repair": (
        lambda s, r: repair.fleet_consolidation_repair(s, r)
    ),
    "sequential_repair": lambda s, r: repair.sequential_repair(s, r),
    "lookahead_repair": (
        lambda s, r: repair.lookahead_repair(s, r, lookahead_horizon=1)
    ),
}


@pytest.mark.parametrize("op_name", sorted(REPAIR_OPERATORS))
def test_repair_operator_keeps_the_solution_hard_feasible(
    feasible_solution: SolutionState, op_name: str
) -> None:
    state = _destroyed(feasible_solution)

    result = REPAIR_OPERATORS[op_name](state, np.random.default_rng(13))

    assert isinstance(result, SolutionState)
    _assert_still_consistent(result)


def test_unmet_demand_equals_total_demand_when_the_fleet_is_empty(
    feasible_solution: SolutionState,
) -> None:
    state = feasible_solution.copy()
    state.vehicles = []
    state.mark_objective_dirty()

    unmet = get_unmet_demand(state)

    assert sum(unmet.values()) == sum(state.data.demands.values())
    assert all(qty > 0 for qty in unmet.values())


def test_unmet_demand_is_empty_for_a_feasible_solution(
    feasible_solution: SolutionState,
) -> None:
    assert get_unmet_demand(feasible_solution.copy()) == {}


def test_greedy_repair_recovers_demand_after_a_full_removal(
    feasible_solution: SolutionState,
) -> None:
    state = feasible_solution.copy()
    state.vehicles = []
    state.mark_objective_dirty()
    before = sum(get_unmet_demand(state).values())
    assert before > 0

    repair.greedy_repair(state, np.random.default_rng(5))

    after = sum(get_unmet_demand(state).values())
    assert after < before
    _assert_still_consistent(state)


def test_greedy_repair_does_not_rescan_the_global_vehicle_list(
    feasible_solution: SolutionState, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _destroyed(feasible_solution)

    def global_scan(*args, **kwargs):
        raise AssertionError("greedy_repair should use its local route index")

    monkeypatch.setattr(repair, "find_existing_route_vehicles", global_scan)

    repair.greedy_repair(state, np.random.default_rng(13))

    _assert_still_consistent(state)


def test_runtime_deadline_uses_tracker_without_affecting_direct_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ALNSConfig, "MAX_RUNTIME", 10.0)
    state = SimpleNamespace(
        tracker=SimpleNamespace(elapsed_seconds=lambda: 4.0),
    )
    deadline = repair._runtime_deadline(state)
    assert deadline is not None
    assert time.monotonic() <= deadline <= time.monotonic() + 7.0
    assert repair._runtime_deadline(SimpleNamespace()) is None


def test_cross_period_consolidation_never_loses_demand_coverage(
    feasible_solution: SolutionState,
) -> None:
    state = _destroyed(feasible_solution)
    before = sum(state.compute_shipped().values())

    repair.cross_period_consolidation_repair(state, np.random.default_rng(3))

    after = sum(state.compute_shipped().values())
    # Consolidation moves cargo between periods; it must not drop coverage.
    assert after >= before
    _assert_still_consistent(state)


def test_cross_period_consolidation_reuses_validation_for_candidate_cost(
    feasible_solution: SolutionState, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _destroyed(feasible_solution)
    active = False
    copy_calls = 0
    validate_calls = 0
    original_copy = SolutionState.copy
    original_validate_uncached = SolutionState._validate_uncached

    def counted_copy(self: SolutionState, *args, **kwargs):
        nonlocal copy_calls
        if active:
            copy_calls += 1
        return original_copy(self, *args, **kwargs)

    def counted_validate_uncached(self: SolutionState):
        nonlocal validate_calls
        if active:
            validate_calls += 1
        return original_validate_uncached(self)

    monkeypatch.setattr(SolutionState, "copy", counted_copy)
    monkeypatch.setattr(
        SolutionState, "_validate_uncached", counted_validate_uncached
    )
    active = True
    try:
        repair.cross_period_consolidation_repair(
            state, np.random.default_rng(3), time_limit=5.0, max_moves=10
        )
    finally:
        active = False

    # One validation for the initial cost, then exactly one per candidate copy.
    # calculate_cost() must reuse the candidate's first validation from the memo.
    assert copy_calls > 0
    assert validate_calls == copy_calls


def test_fill_minimum_load_repair_does_not_add_underloaded_vehicles(
    feasible_solution: SolutionState,
) -> None:
    state = _destroyed(feasible_solution)
    min_load = state.data.veh_type_min_load

    def underloaded(current: SolutionState) -> int:
        return sum(
            1
            for veh in current.iter_used_vehicles()
            if current.compute_veh_load(veh) + 1e-9 < min_load.get(veh.type, 0.0)
        )

    before = underloaded(state)
    repair.fill_minimum_load_repair(state, np.random.default_rng(2))

    assert underloaded(state) <= before
    _assert_still_consistent(state)


def test_fill_minimum_load_repair_trusts_incremental_inventory_after_commit(
    feasible_solution: SolutionState, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _destroyed(feasible_solution)
    inventory_calls = 0
    original_compute_inventory = SolutionState.compute_inventory

    def counted_compute_inventory(self: SolutionState):
        nonlocal inventory_calls
        inventory_calls += 1
        return original_compute_inventory(self)

    monkeypatch.setattr(SolutionState, "compute_inventory", counted_compute_inventory)
    before_volume = sum(state.compute_veh_load(veh) for veh in state.vehicles)

    repair.fill_minimum_load_repair(state, np.random.default_rng(2))

    after_volume = sum(state.compute_veh_load(veh) for veh in state.vehicles)
    assert after_volume > before_volume
    assert inventory_calls == 1
    _assert_still_consistent(state)


def test_inventory_rollback_fails_closed_when_recompute_raises(
    feasible_solution: SolutionState, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = feasible_solution.copy()
    vehicle = next(veh for veh in state.vehicles if not veh.is_empty())
    key = next(iter(vehicle.cargo))
    before_len = len(state.vehicles)
    before_cargo = dict(vehicle.cargo)
    before_s_ikt = dict(state.s_ikt)
    recorded: list[tuple[str, BaseException]] = []

    vehicle.cargo[key] = vehicle.cargo[key] + 1
    state.vehicles.append(
        Vehicle(vehicle.fact_id, vehicle.dealer_id, vehicle.type, vehicle.day, state.data)
    )

    def raising_compute_inventory():
        raise RuntimeError("recompute failed")

    monkeypatch.setattr(state, "compute_inventory", raising_compute_inventory)
    monkeypatch.setattr(
        repair,
        "record_swallowed",
        lambda site, exc: recorded.append((site, exc)),
    )

    kept = repair._rollback_load_if_inventory_negative(
        state,
        vehicle,
        before_len,
        before_cargo,
        before_s_ikt,
    )

    assert kept is False
    assert len(state.vehicles) == before_len
    assert vehicle.cargo == before_cargo
    assert state.s_ikt == before_s_ikt
    assert recorded[0][0] == (
        "repair_operators.inventory_balance_repair.compute_inventory"
    )
    assert isinstance(recorded[0][1], RuntimeError)


def test_inventory_balance_retries_only_after_partial_load_is_restored(
    feasible_solution: SolutionState, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = _destroyed(feasible_solution)
    repaired_baseline = repair.inventory_balance_repair(
        baseline.copy(), np.random.default_rng(7)
    )
    flaky = _destroyed(feasible_solution)
    original_veh_loading = repair.veh_loading
    calls = {"n": 0}

    def flaky_veh_loading(state, vehicle, orders, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            original_veh_loading(state, vehicle, orders, **kwargs)
            raise RuntimeError("failure after partial commit")
        return original_veh_loading(state, vehicle, orders, **kwargs)

    monkeypatch.setattr(repair, "veh_loading", flaky_veh_loading)

    repaired_flaky = repair.inventory_balance_repair(
        flaky, np.random.default_rng(7)
    )

    assert calls["n"] > 1
    assert _vehicle_layout(repaired_flaky) == _vehicle_layout(repaired_baseline)
    assert repaired_flaky.s_ikt == repaired_baseline.s_ikt
    assert get_unmet_demand(repaired_flaky) == get_unmet_demand(repaired_baseline)
    _assert_still_consistent(repaired_flaky)


def test_inventory_balance_does_not_spend_demand_when_load_is_rolled_back(
    feasible_solution: SolutionState, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _destroyed(feasible_solution)
    before_layout = _vehicle_layout(state)
    before_inventory = dict(state.s_ikt)
    before_unmet = get_unmet_demand(state)

    def reject_and_restore(
        current: SolutionState,
        vehicle: Vehicle,
        before_len: int,
        before_cargo: dict,
        before_s_ikt: dict,
    ) -> bool:
        assert repair._restore_load_checkpoint(
            current,
            vehicle,
            before_len,
            before_cargo,
            before_s_ikt,
        )
        return False

    monkeypatch.setattr(
        repair,
        "_rollback_load_if_inventory_negative",
        reject_and_restore,
    )
    monkeypatch.setattr(repair, "post_repair", lambda *args, **kwargs: None)

    repair.inventory_balance_repair(state, np.random.default_rng(7))

    assert _vehicle_layout(state) == before_layout
    assert state.s_ikt == before_inventory
    assert get_unmet_demand(state) == before_unmet


def test_regret_repair_never_commits_a_vehicle_the_loader_rejects(
    feasible_solution: SolutionState,
) -> None:
    """Every vehicle regret adds must carry cargo.

    ``veh_loading(suffix_safe=True)`` refuses a load when any later period has
    no slack, while the resource pool only knows the load's own day. Before the
    availability cap the two disagreed: regret appended a vehicle that shipped
    nothing, then re-picked the same demand until its time budget ran out
    (measured on medium/dataset_13: 59-88% of commits shipped zero).

    A vehicle appended with empty cargo is the observable signature of that
    mismatch, so it is asserted directly.
    """
    state = _destroyed(feasible_solution)
    before_total = len(state.vehicles)

    repaired = repair.regret_based_repair(
        state,
        np.random.default_rng(13),
        k=2,
        topN=6,
        time_limit=3.0,
    )

    appended = repaired.vehicles[before_total:]
    # Guard against a vacuous pass: the operator must have inserted something
    # for the assertion below to mean anything.
    assert appended, "regret inserted nothing; the invariant was not exercised"
    empty = [veh for veh in appended if not (veh.cargo or {})]
    assert not empty, f"{len(empty)} of {len(appended)} appended vehicles carry no cargo"


def _fleet_layout(state: SolutionState) -> list[tuple]:
    return [
        (
            veh.fact_id,
            veh.dealer_id,
            veh.type,
            veh.day,
            tuple(sorted((veh.cargo or {}).items())),
        )
        for veh in state.vehicles
    ]


def test_shortlist_cache_reuses_the_scan_without_changing_the_result(
    feasible_solution: SolutionState, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reusing a shortlist must not change a single commit.

    A pass commits one vehicle and then re-enumerates every unmet demand. The
    cache only skips the ``plant x day x vehicle type`` scan for demands whose
    unmet quantity and SKU availability are unchanged, and the improvement
    scores are still recomputed, so the outcome has to be *identical* rather
    than merely close.

    ``ResourcePool.get`` is counted as the observable of the scan: a reuse that
    never fires would leave the count unchanged and this test would still pass
    its equality half, so the ratio is asserted first.
    """
    scans = {"count": 0}
    original_get = ResourcePool.get

    def counted_get(self, key, default=0):
        scans["count"] += 1
        return original_get(self, key, default)

    monkeypatch.setattr(ResourcePool, "get", counted_get)

    def run(cache_on: bool) -> tuple[SolutionState, int]:
        monkeypatch.setattr(ALNSConfig, "REGRET_SHORTLIST_CACHE", cache_on)
        state = _destroyed(feasible_solution)
        before = scans["count"]
        repaired = repair.regret_based_repair(
            state,
            np.random.default_rng(13),
            k=2,
            topN=6,
            time_limit=3.0,
        )
        return repaired, scans["count"] - before

    without, scans_without = run(False)
    with_cache, scans_with = run(True)

    assert scans_without > 0, "the scan never ran; the comparison is vacuous"
    assert scans_with < scans_without, (
        f"cache reused nothing: {scans_with} scans vs {scans_without} without it"
    )
    assert _fleet_layout(with_cache) == _fleet_layout(without)
    assert with_cache.s_ikt == without.s_ikt
    assert with_cache.calculate_cost() == without.calculate_cost()


def test_regret_repair_does_not_copy_when_fast_estimates_reject_all_candidates(
    feasible_solution: SolutionState, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _destroyed(feasible_solution)
    copy_calls = {"clone_false": 0}
    original_copy = SolutionState.copy

    def counted_copy(self: SolutionState, clone_vehicles: bool = True):
        if not clone_vehicles:
            copy_calls["clone_false"] += 1
        return original_copy(self, clone_vehicles=clone_vehicles)

    monkeypatch.setattr(SolutionState, "copy", counted_copy)
    monkeypatch.setattr(
        repair,
        "compute_improvement_fast_add_vehicle",
        lambda *args, **kwargs: (0.0, 0.0, False),
    )

    repair.regret_based_repair(
        state,
        np.random.default_rng(13),
        k=2,
        topN=6,
        time_limit=1.0,
    )

    assert copy_calls["clone_false"] == 0
