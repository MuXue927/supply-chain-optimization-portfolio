"""Focused tests for the standalone solution validator.

``check_solution`` re-checks an ALNS result against the four hard constraints
of the model (demand satisfaction, plant inventory capacity, vehicle capacity,
inventory flow). It needs no solver, so each check is exercised directly on
hand-built inputs, and the module entry point is driven end to end once.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from multi_period_delivery.alns import check_solution as cs
from multi_period_delivery.alns.InputDataALNS import DataALNS
from multi_period_delivery.alns.OutputDataALNS import OutPutData


def _scale_dir() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "datasets"
        / "multiple-periods"
        / "small"
    )


@pytest.fixture(scope="module")
def data() -> DataALNS:
    instance = DataALNS(str(_scale_dir()), str(_scale_dir().parent / "tmp"), "dataset_1")
    instance.load()
    return instance


def _empty_output() -> OutPutData:
    """An output wrapper with no records at all."""
    return OutPutData("unused", "dataset_1")


def _fill_consistent_inventory(data: DataALNS, output: OutPutData) -> None:
    """Record the ending inventory implied by an all-zero shipment plan."""
    for plant_id in data.plants:
        for sku_id in data.all_skus:
            current = data.sku_initial_inv.get((plant_id, sku_id), 0)
            for day in range(1, data.horizons + 1):
                current += data.sku_prod_each_day.get((plant_id, sku_id, day), 0)
                output.sku_inv_left[(plant_id, sku_id, day)] = current


def _ship_every_demand(data: DataALNS, output: OutPutData) -> str:
    """Record shipping exactly the demanded quantity from one plant."""
    plant_id = sorted(data.plants)[0]
    for (dealer_id, sku_id), demand in data.demands.items():
        output.order_fulfill[(1, plant_id, dealer_id, sku_id)] = demand
    return plant_id


# ---------------------------------------------------------------------------
# Demand satisfaction
# ---------------------------------------------------------------------------
def test_demand_satisfaction_flags_a_complete_shortage(data: DataALNS) -> None:
    result = cs.SolutionValidator(data, _empty_output()).check_demand_satisfaction()

    assert result.is_satisfied is False
    assert result.total_violations == len(data.demands)
    assert result.details["undersupplied_demands"] == len(data.demands)
    assert result.details["total_shortage"] == sum(data.demands.values())


def test_demand_satisfaction_passes_when_every_demand_is_met(data: DataALNS) -> None:
    output = _empty_output()
    _ship_every_demand(data, output)

    result = cs.SolutionValidator(data, output).check_demand_satisfaction()

    assert result.is_satisfied is True
    assert result.total_violations == 0
    assert result.details["satisfied_demands"] == len(data.demands)


def test_demand_satisfaction_records_oversupply_without_failing(data: DataALNS) -> None:
    output = _empty_output()
    plant_id = _ship_every_demand(data, output)
    dealer_id, sku_id = next(iter(data.demands))
    output.order_fulfill[(1, plant_id, dealer_id, sku_id)] += 3

    result = cs.SolutionValidator(data, output).check_demand_satisfaction()

    assert result.is_satisfied is True
    assert result.details["oversupplied_demands"] == 1
    assert result.details["total_oversupply"] == 3


# ---------------------------------------------------------------------------
# Plant inventory capacity
# ---------------------------------------------------------------------------
def test_plant_capacity_passes_and_counts_every_day(data: DataALNS) -> None:
    result = cs.SolutionValidator(data, _empty_output()).check_plant_capacity()

    assert result.is_satisfied is True
    assert result.details["total_checks"] == len(data.plants) * data.horizons


def test_plant_capacity_flags_an_excess(data: DataALNS) -> None:
    output = _empty_output()
    plant_id = sorted(data.plants)[0]
    sku_id = sorted(data.all_skus)[0]
    limit = data.plant_inv_limit[plant_id]
    output.sku_inv_left[(plant_id, sku_id, 1)] = limit + 7

    result = cs.SolutionValidator(data, output).check_plant_capacity()

    assert result.is_satisfied is False
    assert result.details["violations_count"] == 1
    assert result.details["total_excess"] == 7
    assert result.details["max_violation"] == 7


# ---------------------------------------------------------------------------
# Vehicle capacity
# ---------------------------------------------------------------------------
def test_vehicle_capacity_flags_an_overload(data: DataALNS) -> None:
    output = _empty_output()
    veh_type = sorted(data.all_veh_types)[0]
    plant_id = sorted(data.plants)[0]
    dealer_id = sorted(data.dealers)[0]
    output.vehicle_load[(1, plant_id, dealer_id, 1, veh_type)] = (
        data.veh_type_cap[veh_type] + 5
    )

    result = cs.SolutionValidator(data, output).check_vehicle_capacity()

    assert result.is_satisfied is False
    assert result.details["capacity_violations"] == 1
    assert result.details["max_capacity_violation"] == 5


def test_vehicle_capacity_records_min_load_warnings(data: DataALNS) -> None:
    # Minimum load is a soft constraint: it is reported, not failed.
    veh_type = max(data.all_veh_types, key=lambda vt: data.veh_type_min_load[vt])
    min_load = data.veh_type_min_load[veh_type]
    if min_load <= 1:
        pytest.skip("the fixture has no vehicle type with a meaningful min load")
    output = _empty_output()
    output.vehicle_load[
        (1, sorted(data.plants)[0], sorted(data.dealers)[0], 1, veh_type)
    ] = min_load - 1

    result = cs.SolutionValidator(data, output).check_vehicle_capacity()

    assert result.is_satisfied is True
    assert result.details["min_load_warnings"] == 1


# ---------------------------------------------------------------------------
# Inventory flow
# ---------------------------------------------------------------------------
def test_inventory_flow_passes_when_records_match(data: DataALNS) -> None:
    output = _empty_output()
    _fill_consistent_inventory(data, output)

    result = cs.SolutionValidator(data, output).check_inventory_flow()

    assert result.is_satisfied is True
    assert result.details["negative_inventory_violations"] == 0
    assert result.details["record_inconsistencies"] == 0
    assert result.details["total_flow_checks"] == len(data.plants) * len(data.all_skus) * data.horizons


def test_inventory_flow_flags_negative_inventory(data: DataALNS) -> None:
    output = _empty_output()
    plant_id = sorted(data.plants)[0]
    output.order_fulfill[
        (1, plant_id, sorted(data.dealers)[0], sorted(data.all_skus)[0])
    ] = 10**6

    result = cs.SolutionValidator(data, output).check_inventory_flow()

    assert result.is_satisfied is False
    assert result.details["negative_inventory_violations"] >= 1
    assert result.details["min_negative_inventory"] < 0


def test_inventory_flow_flags_a_recorded_inconsistency(data: DataALNS) -> None:
    output = _empty_output()
    _fill_consistent_inventory(data, output)
    plant_id = sorted(data.plants)[0]
    sku_id = sorted(data.all_skus)[0]
    output.sku_inv_left[(plant_id, sku_id, 1)] += 5

    result = cs.SolutionValidator(data, output).check_inventory_flow()

    assert result.is_satisfied is False
    assert result.details["record_inconsistencies"] >= 1
    assert result.details["max_inconsistency"] >= 5


# ---------------------------------------------------------------------------
# Aggregate validation
# ---------------------------------------------------------------------------
def test_validate_all_aggregates_every_check(data: DataALNS) -> None:
    validator = cs.SolutionValidator(data, _empty_output())

    feasible, results = validator.validate_all()

    assert feasible is False
    assert len(results) == 4
    assert validator.validation_results is results
    assert {result.constraint_name for result in results} == {
        "Demand satisfaction constraint",
        "Plant inventory capacity constraint",
        "Vehicle capacity constraint",
        "Inventory flow constraint",
    }


def test_validate_all_reports_a_check_that_raises(
    data: DataALNS, monkeypatch: pytest.MonkeyPatch
) -> None:
    validator = cs.SolutionValidator(data, _empty_output())

    def _boom():
        raise RuntimeError("checker exploded")

    monkeypatch.setattr(validator, "check_vehicle_capacity", _boom)

    feasible, results = validator.validate_all()

    assert feasible is False
    failed = [result for result in results if result.constraint_name == "_boom"]
    assert len(failed) == 1
    assert failed[0].violations == ["Validation process error: checker exploded"]
    assert failed[0].details == {}


def test_print_validation_summary_lists_violations(
    data: DataALNS, capsys
) -> None:
    validator = cs.SolutionValidator(data, _empty_output())
    validator.validate_all()

    validator.print_validation_summary()

    printed = capsys.readouterr().out
    assert "Solution validation summary" in printed
    assert "[FAIL]" in printed
    assert "additional violations" in printed
    assert "Solution status: infeasible" in printed


def test_print_validation_summary_reports_a_feasible_solution(
    data: DataALNS, capsys
) -> None:
    validator = cs.SolutionValidator(data, _empty_output())
    validator.validation_results = [
        cs.ValidationResult(
            constraint_name="Demand satisfaction constraint",
            is_satisfied=True,
            violations=[],
            total_violations=0,
            details={},
        )
    ]

    validator.print_validation_summary()

    printed = capsys.readouterr().out
    assert "[PASS]" in printed
    assert "Total violations: 0" in printed
    assert "Solution status: feasible" in printed


# ---------------------------------------------------------------------------
# Module entry point
# ---------------------------------------------------------------------------
def test_check_solution_returns_false_for_missing_input() -> None:
    assert cs.check_solution("no-such-input", "no-such-output") is False


def test_check_solution_runs_on_real_data_without_output(tmp_path) -> None:
    # Valid input data, but no output files: everything is unfulfilled, so the
    # validator must report the solution as infeasible rather than crashing.
    assert cs.check_solution(str(_scale_dir()), str(tmp_path / "out")) is False


def test_check_solution_returns_true_when_every_check_passes(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The individual checks are covered above; here we only verify that the
    # entry point propagates a passing result.
    monkeypatch.setattr(
        cs.SolutionValidator, "validate_all", lambda self: (True, [])
    )

    assert cs.check_solution(str(_scale_dir()), str(tmp_path / "out")) is True
