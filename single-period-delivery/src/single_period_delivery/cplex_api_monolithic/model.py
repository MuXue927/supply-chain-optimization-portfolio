"""CPLEX Python API monolithic model for the single-period delivery problem."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from ..cplex_api_support import CplexApiModel, CplexApiSolution
from ..monolithic.InputDataSingle import InputDataSingle


class SinglePeriodCplexApiMonolithic:
    """Build and solve the single-period monolithic model with CPLEX Python."""

    def __init__(self, data: InputDataSingle):
        self.data = data
        self.model = CplexApiModel("single_period_cplex_api_monolithic")
        self.z: Dict[int, str] = {}
        self.zz: Dict[int, str] = {}
        self.p: Dict[int, str] = {}
        self.y: Dict[Tuple[str, str, int], str] = {}
        self.x: Dict[Tuple[str, str, str, int], str] = {}
        self._build_variables()
        self._build_objective()
        self._build_constraints()

    def _build_variables(self) -> None:
        """Create all variables in the original monolithic formulation."""
        data = self.data
        for vehicle in data.all_vehicles:
            self.z[vehicle] = self.model.add_variable(
                f"z_{vehicle}", kind="binary"
            )
            self.zz[vehicle] = self.model.add_variable(
                f"zz_{vehicle}", kind="binary"
            )
            self.p[vehicle] = self.model.add_variable(
                f"p_{vehicle}", kind="binary"
            )

        for plant, dealer in data.supply_chain:
            for vehicle in data.all_vehicles:
                self.y[plant, dealer, vehicle] = self.model.add_variable(
                    f"y_{plant}_{dealer}_{vehicle}",
                    kind="binary",
                )

        for plant, dealer in data.supply_chain:
            for sku in data.supply_chain[plant, dealer]:
                for vehicle in data.all_vehicles:
                    self.x[plant, dealer, sku, vehicle] = (
                        self.model.add_variable(
                            f"x_{plant}_{dealer}_{sku}_{vehicle}",
                            kind="integer",
                        )
                    )

    def _build_objective(self) -> None:
        """Minimize fixed cost plus minimum-load penalties."""
        terms: Dict[str, float] = {}
        for vehicle in self.z:
            terms[self.z[vehicle]] = float(self.data.veh_cost[vehicle])
            terms[self.p[vehicle]] = (
                float(self.data.param_alpha)
                * float(self.data.veh_cost[vehicle])
            )
        self.model.set_objective(terms)

    def _build_constraints(self) -> None:
        """Add all constraints from the original single-period model."""
        data = self.data
        for vehicle in data.all_vehicles:
            terms = {
                self.y[plant, dealer, vehicle]: 1.0
                for plant, dealer in data.supply_chain
            }
            terms[self.z[vehicle]] = -1.0
            self.model.add_constraint(
                f"vehicle_use_{vehicle}",
                terms,
                "=",
                0.0,
            )

        for plant, dealer in data.supply_chain:
            for vehicle in data.all_vehicles:
                terms = {
                    self.x[plant, dealer, sku, vehicle]: 1.0
                    for sku in data.supply_chain[plant, dealer]
                }
                terms[self.y[plant, dealer, vehicle]] = (
                    -float(data.param_BIG_NUM)
                )
                self.model.add_constraint(
                    f"link_x_y_{plant}_{dealer}_{vehicle}",
                    terms,
                    "<=",
                    0.0,
                )

        for vehicle in data.all_vehicles:
            load = {
                self.x[plant, dealer, sku, vehicle]: float(
                    data.sku_sizes[sku]
                )
                for plant, dealer in data.supply_chain
                for sku in data.supply_chain[plant, dealer]
            }
            load[self.zz[vehicle]] = float(data.param_BIG_NUM)
            self.model.add_constraint(
                f"minimum_load_{vehicle}",
                load,
                ">=",
                float(data.veh_min_load[vehicle]),
            )
            self.model.add_constraint(
                f"penalty_link_{vehicle}",
                {
                    self.z[vehicle]: 1.0,
                    self.zz[vehicle]: 1.0,
                    self.p[vehicle]: -1.0,
                },
                "<=",
                1.0,
            )
            capacity = {
                self.x[plant, dealer, sku, vehicle]: float(
                    data.sku_sizes[sku]
                )
                for plant, dealer in data.supply_chain
                for sku in data.supply_chain[plant, dealer]
            }
            self.model.add_constraint(
                f"vehicle_capacity_{vehicle}",
                capacity,
                "<=",
                float(data.veh_cap[vehicle]),
            )

        for dealer, sku in data.demands:
            terms = {
                self.x[plant, dealer, sku, vehicle]: 1.0
                for plant in data.plants
                for vehicle in data.all_vehicles
                if (plant, dealer, sku, vehicle) in self.x
            }
            self.model.add_constraint(
                f"demand_{dealer}_{sku}",
                terms,
                ">=",
                float(data.demands[dealer, sku]),
            )

        for plant, sku in data.sku_inv:
            terms = {
                self.x[plant, dealer, sku, vehicle]: 1.0
                for dealer in data.dealers
                for vehicle in data.all_vehicles
                if (plant, dealer, sku, vehicle) in self.x
            }
            self.model.add_constraint(
                f"supply_{plant}_{sku}",
                terms,
                "<=",
                float(data.sku_inv[plant, sku]),
            )

    def solve(
        self,
        work_dir: str | Path,
        *,
        time_limit: float | None = None,
        mip_gap: float | None = None,
        random_seed: int | None = None,
    ) -> CplexApiSolution:
        """Solve with CPLEX and return the parsed result."""
        return self.model.solve(
            work_dir,
            time_limit=time_limit,
            mip_gap=mip_gap,
            random_seed=random_seed,
            parameter_overrides={
                "mip.strategy.heuristicfreq": 1,
                "mip.strategy.fpheur": 1,
                "mip.strategy.rinsheur": 1,
                "mip.strategy.heuristiceffort": 3,
            },
            write_model_path=Path(work_dir) / "cplex_api_model.lp",
        )

    def write_outputs(
        self,
        solution: CplexApiSolution,
        output_dir: str | Path,
    ) -> Dict[str, object]:
        """Write result files matching the original single-period schema."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        rows = []
        for key, variable in self.x.items():
            value = solution.variables.get(variable, 0.0)
            if value <= 1e-7:
                continue
            plant, dealer, sku, vehicle = key
            rows.append(
                {
                    "Index": len(rows) + 1,
                    "day": 1,
                    "plant_code": plant,
                    "client_code": dealer,
                    "product_code": sku,
                    "vehicle_id": vehicle,
                    "vehicle_type": self.data.veh_type[vehicle],
                    "qty": int(round(value)),
                }
            )
        result = pd.DataFrame(
            rows,
            columns=[
                "Index",
                "day",
                "plant_code",
                "client_code",
                "product_code",
                "vehicle_id",
                "vehicle_type",
                "qty",
            ],
        )
        result.to_csv(output_path / "result.csv", index=False)

        details = result.drop(columns=["Index"]).copy()
        if not details.empty:
            details["standard_size"] = details["product_code"].map(
                self.data.sku_sizes
            )
            details["carry_standard_size"] = details["vehicle_id"].map(
                self.data.veh_cap
            )
            details["min_standard_size"] = details["vehicle_id"].map(
                self.data.veh_min_load
            )
            details["occupied_size"] = (
                details["qty"] * details["standard_size"]
            )
        details.to_csv(output_path / "opt_details.csv", index=False)

        summary = (
            details.groupby(
                ["day", "plant_code", "client_code", "vehicle_id"],
                as_index=False,
            ).agg(
                vehicle_type=("vehicle_type", "first"),
                qty=("qty", "sum"),
                carry_standard_size=("carry_standard_size", "first"),
                min_standard_size=("min_standard_size", "first"),
                occupied_size=("occupied_size", "sum"),
            )
            if not details.empty
            else pd.DataFrame()
        )
        if not summary.empty:
            summary["below_min_load"] = (
                summary["occupied_size"] + 1e-9
                < summary["min_standard_size"]
            )
            summary["fixed_cost"] = summary["vehicle_id"].map(
                self.data.veh_cost
            )
            summary["minimum_load_penalty"] = (
                summary["below_min_load"].astype(float)
                * float(self.data.param_alpha)
                * summary["fixed_cost"]
            )
            summary["total_cost"] = (
                summary["fixed_cost"] + summary["minimum_load_penalty"]
            )
        summary.to_csv(output_path / "opt_summary.csv", index=False)

        shipped = (
            result.groupby(["plant_code", "product_code"], as_index=False)[
                "qty"
            ].sum()
            if not result.empty
            else pd.DataFrame(columns=["plant_code", "product_code", "qty"])
        )
        inventory_rows = []
        for (plant, sku), initial in self.data.sku_inv.items():
            sent = 0
            match = shipped[
                (shipped["plant_code"] == plant)
                & (shipped["product_code"] == sku)
            ]
            if not match.empty:
                sent = int(match.iloc[0]["qty"])
            inventory_rows.append(
                {
                    "plant_code": plant,
                    "product_code": sku,
                    "volume": int(initial - sent),
                }
            )
        pd.DataFrame(inventory_rows).to_csv(
            output_path / "sku_inv_left.csv",
            index=False,
        )

        return {
            "status": solution.status,
            "objective": solution.objective,
            "deliveries": int(len(result)),
            "vehicles": int(summary["vehicle_id"].nunique())
            if not summary.empty
            else 0,
        }
