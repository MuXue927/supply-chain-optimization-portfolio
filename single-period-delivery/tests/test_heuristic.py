"""Focused tests for the single-period two-stage heuristic.

The heuristic is the one single-period path that runs without a solver
licence, so it is the part of this package CI can genuinely exercise. The
tests cover the data loader, the ``Vehicle`` container, the stage-1/stage-2
model pipeline, the cost model (including the minimum-load penalty), and one
end-to-end ``run_model`` call that writes its output files.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from single_period_delivery.heuristic import main as heuristic_main
from single_period_delivery.heuristic.HeuristicModel import HeuristicModel
from single_period_delivery.heuristic.InputDataHeuristic import InputDataHeuristic
from single_period_delivery.heuristic.vehicle import Vehicle


class _MiniData:
    """Tiny hand-built dataset so the model tests do not depend on a fixture."""

    def __init__(self) -> None:
        self.plants = {"P1"}
        self.dealers = {"D1"}
        self.all_skus = {"S1"}
        self.all_veh_types = {"V1"}
        self.sku_sizes = {"S1": 1.0}
        self.sku_inv = {("P1", "S1"): 100}
        self.skus_plant = {"P1": {"S1"}}
        self.demands = {("D1", "S1"): 10}
        self.skus_dealer = {"D1": {"S1"}}
        self.veh_type_cost = {"V1": 100.0}
        self.veh_type_cap = {"V1": 50}
        self.veh_type_min_load = {"V1": 20}
        self.param_alpha = 2.0

    def construct_supply_chain(self) -> dict:
        return {("P1", "D1"): {"S1"}}


def _small_dataset_dir() -> str:
    base = Path(__file__).resolve().parents[1]
    return str(base / "datasets" / "single-period" / "small")


def _load_data(dataset: str = "dataset_1") -> InputDataHeuristic:
    data = InputDataHeuristic(_small_dataset_dir(), str(Path(_small_dataset_dir()).parent), dataset)
    data.load()
    return data


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def test_input_data_loads_expected_interfaces() -> None:
    data = _load_data()

    assert data.horizons == 1
    assert data.plants
    assert data.dealers
    assert data.all_skus
    assert data.all_veh_types
    assert all(size > 0 for size in data.sku_sizes.values())
    assert all(cap > 0 for cap in data.veh_type_cap.values())
    assert sum(data.demands.values()) > 0
    assert data.param_alpha >= 0.0

    plant = next(iter(data.plants))
    dealer = next(iter(data.dealers))
    assert data.available_skus_in_plant(plant) == data.skus_plant[plant]
    assert isinstance(data.available_skus_to_dealer(plant, dealer), set)
    assert data.construct_supply_chain()


# ---------------------------------------------------------------------------
# Vehicle
# ---------------------------------------------------------------------------
def test_vehicle_load_respects_capacity_and_reports_remainder() -> None:
    data = _MiniData()
    vehicle = Vehicle("P1", "D1", "V1", data)
    assert vehicle.is_empty()

    # 10 units fit easily: nothing is left unloaded.
    assert vehicle.load("S1", 10) == 0
    assert vehicle.cargo["S1"] == 10
    assert vehicle.get_avail_space() == pytest.approx(40.0)

    # Asking for 100 more fits only the remaining 40 units of capacity.
    assert vehicle.load("S1", 100) == 60
    assert vehicle.cargo["S1"] == 50
    assert vehicle.get_avail_space() == pytest.approx(0.0)

    vehicle.clear()
    assert vehicle.is_empty()
    assert vehicle.get_avail_space() == pytest.approx(50.0)


def test_vehicle_load_returns_full_request_when_no_space() -> None:
    data = _MiniData()
    vehicle = Vehicle("P1", "D1", "V1", data)
    vehicle.load("S1", 50)  # fills the vehicle

    assert vehicle.load("S1", 7) == 7


# ---------------------------------------------------------------------------
# Two-stage model
# ---------------------------------------------------------------------------
def test_two_stage_pipeline_satisfies_demand() -> None:
    random.seed(0)
    data = _MiniData()
    model = HeuristicModel(data)

    model.optimize_stage_1()
    model.cal_supply()
    model.update_X_ij()
    model.optimize_stage_2()
    model.cal_vehicle_type_nums()
    model.cal_vehicle_type_sets()
    solution = model.calculate_solution()

    assert model.check_solution() is True
    assert solution[("P1", "D1", "S1", "V1", 1)] == 10
    assert model.calculate_vehicle_nums() == 1
    assert model.N_f["V1"] == 1
    assert model.Veh_type_sets["V1"] == {1}


def test_stage_one_records_shipped_quantities() -> None:
    data = _MiniData()
    model = HeuristicModel(data)

    model.optimize_stage_1()
    supply = model.cal_supply()
    fulfill = model.check_fulfill()

    assert supply[("P1", "D1", "S1")] == 10
    assert fulfill[("D1", "S1")] == 10


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------
def test_cost_is_fixed_cost_per_used_vehicle() -> None:
    data = _MiniData()
    model = HeuristicModel(data)
    model.N_f = {"V1": 1}
    vehicle = Vehicle("P1", "D1", "V1", data)
    vehicle.load("S1", 50)  # at/above the minimum load -> no penalty
    model.Vehicles = [vehicle]

    assert model.calculate_cost() == pytest.approx(100.0)


def test_cost_penalises_underloaded_vehicles() -> None:
    data = _MiniData()
    model = HeuristicModel(data)
    model.N_f = {"V1": 1}
    vehicle = Vehicle("P1", "D1", "V1", data)
    vehicle.load("S1", 5)  # below the minimum load of 20
    model.Vehicles = [vehicle]

    # fixed cost + alpha * fixed cost
    assert model.calculate_cost() == pytest.approx(300.0)


def test_cost_ignores_empty_vehicles() -> None:
    data = _MiniData()
    model = HeuristicModel(data)
    model.N_f = {"V1": 1}
    model.Vehicles = [Vehicle("P1", "D1", "V1", data)]

    assert model.calculate_cost() == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# End-to-end run
# ---------------------------------------------------------------------------
def test_run_model_writes_its_output_files(tmp_path) -> None:
    output_dir = tmp_path / "heuristic-out"

    heuristic_main.run_model(
        input_file_loc=_small_dataset_dir(),
        output_file_loc=str(output_dir),
        exp_nums=1,
        dataset_name="dataset_1",
    )

    produced = output_dir / "dataset_1"
    for name in (
        "opt_result.csv",
        "opt_details.csv",
        "opt_summary.csv",
        "fulfill.csv",
    ):
        assert (produced / name).is_file(), name

    # The summary file should describe at least one dispatched vehicle.
    summary_lines = (produced / "opt_summary.csv").read_text(encoding="utf-8").splitlines()
    assert len(summary_lines) >= 2
