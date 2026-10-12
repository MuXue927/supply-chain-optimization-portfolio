"""CPLEX Python API monolithic model for the single-period delivery problem."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..cplex_api_support import CplexApiModel, CplexApiSolution
from ..monolithic.InputDataSingle import InputDataSingle


class SinglePeriodCplexApiMonolithic:
    """Build and solve the single-period monolithic model with CPLEX Python."""

    def __init__(self, data: InputDataSingle):
        self.data = data
        self.model = CplexApiModel("single_period_cplex_api_monolithic")
        # Route-local slots: a slot belongs to one (plant, dealer) route and its
        # vehicle type is chosen by the model, so there is no global fleet index
        # and no `y` assignment variable.
        self.z: dict[tuple[str, str, str, int], str] = {}
        self.zz: dict[tuple[str, str, int], str] = {}
        self.p: dict[tuple[str, str, str, int], str] = {}
        self.x: dict[tuple[str, str, str, int], str] = {}
        self._build_variables()
        self._build_objective()
        self._build_constraints()

    def _build_variables(self) -> None:
        """Create the route-local slot variables.

        For every route and every slot on it:
          z_ijfn  - slot n of route (i, j) uses vehicle type f
          zz_ijn  - slot n is allowed to fall below its minimum load
          p_ijfn  - slot n is used with type f *and* below the minimum load
          x_ijkn  - units of SKU k loaded on slot n of route (i, j)
        """
        data = self.data
        for (plant, dealer), slots in data.route_slots.items():
            for slot in slots:
                self.zz[plant, dealer, slot] = self.model.add_variable(
                    f"zz_{plant}_{dealer}_{slot}", kind="binary"
                )
                for sku in data.supply_chain[plant, dealer]:
                    self.x[plant, dealer, sku, slot] = self.model.add_variable(
                        f"x_{plant}_{dealer}_{sku}_{slot}", kind="integer"
                    )
                for v_type in data.all_veh_types:
                    self.z[plant, dealer, v_type, slot] = self.model.add_variable(
                        f"z_{plant}_{dealer}_{v_type}_{slot}", kind="binary"
                    )
                    self.p[plant, dealer, v_type, slot] = self.model.add_variable(
                        f"p_{plant}_{dealer}_{v_type}_{slot}", kind="binary"
                    )

    def _build_objective(self) -> None:
        """Minimize each activated slot's type cost plus its minimum-load penalty."""
        terms: dict[str, float] = {}
        for (_plant, _dealer, v_type, _slot), variable in self.z.items():
            cost = float(self.data.veh_type_cost[v_type])
            terms[variable] = cost
            terms[self.p[_plant, _dealer, v_type, _slot]] = (
                float(self.data.param_alpha) * cost
            )
        self.model.set_objective(terms)

    def _build_constraints(self) -> None:
        """Add the route-local constraints.

        Constraint set (per route (i, j) and slot n):

          one type per slot   sum_f z_ijfn <= 1
          capacity            sum_k size_k x_ijkn <= sum_f cap_f z_ijfn
          minimum load        sum_k size_k x_ijkn + BIG zz_ijn >= sum_f min_f z_ijfn
          penalty link        z_ijfn + zz_ijn - 1 <= p_ijfn

        plus the two unchanged global rows (demand coverage and plant supply).
        The capacity row also replaces the old x-to-y linking constraint: an
        unused slot has sum_f z_ijfn = 0, which forces every x_ijkn to zero.
        """
        data = self.data

        for (plant, dealer), slots in data.route_slots.items():
            skus = sorted(data.supply_chain[plant, dealer])
            for slot in slots:
                self.model.add_constraint(
                    f"one_type_{plant}_{dealer}_{slot}",
                    {
                        self.z[plant, dealer, v_type, slot]: 1.0
                        for v_type in data.all_veh_types
                    },
                    "<=",
                    1.0,
                )

                capacity = {
                    self.x[plant, dealer, sku, slot]: float(data.sku_sizes[sku])
                    for sku in skus
                }
                for v_type in data.all_veh_types:
                    capacity[self.z[plant, dealer, v_type, slot]] = -float(
                        data.veh_type_cap[v_type]
                    )
                self.model.add_constraint(
                    f"capacity_{plant}_{dealer}_{slot}", capacity, "<=", 0.0
                )

                minimum = {
                    self.x[plant, dealer, sku, slot]: float(data.sku_sizes[sku])
                    for sku in skus
                }
                minimum[self.zz[plant, dealer, slot]] = float(data.param_BIG_NUM)
                for v_type in data.all_veh_types:
                    minimum[self.z[plant, dealer, v_type, slot]] = -float(
                        data.veh_type_min_load[v_type]
                    )
                self.model.add_constraint(
                    f"minimum_load_{plant}_{dealer}_{slot}",
                    minimum,
                    ">=",
                    0.0,
                )

                for v_type in data.all_veh_types:
                    self.model.add_constraint(
                        f"penalty_link_{plant}_{dealer}_{v_type}_{slot}",
                        {
                            self.z[plant, dealer, v_type, slot]: 1.0,
                            self.zz[plant, dealer, slot]: 1.0,
                            self.p[plant, dealer, v_type, slot]: -1.0,
                        },
                        "<=",
                        1.0,
                    )

        for dealer, sku in data.demands:
            terms = {
                self.x[plant, dealer, sku, slot]: 1.0
                for plant in data.plants
                for slot in data.route_slots.get((plant, dealer), [])
                if (plant, dealer, sku, slot) in self.x
            }
            self.model.add_constraint(
                f"demand_{dealer}_{sku}",
                terms,
                ">=",
                float(data.demands[dealer, sku]),
            )

        for plant, sku in data.sku_inv:
            terms = {
                self.x[plant, dealer, sku, slot]: 1.0
                for dealer in data.dealers
                for slot in data.route_slots.get((plant, dealer), [])
                if (plant, dealer, sku, slot) in self.x
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
    ) -> dict[str, object]:
        """Write result files matching the original single-period schema."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        rows = []
        for key, variable in self.x.items():
            value = solution.variables.get(variable, 0.0)
            if value <= 1e-7:
                continue
            plant, dealer, sku, slot = key
            rows.append(
                {
                    "Index": len(rows) + 1,
                    "day": 1,
                    "plant_code": plant,
                    "client_code": dealer,
                    "product_code": sku,
                    "vehicle_id": slot,
                    "vehicle_type": self._slot_type(solution, plant, dealer, slot),
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
            details["carry_standard_size"] = details["vehicle_type"].map(
                self.data.veh_type_cap
            )
            details["min_standard_size"] = details["vehicle_type"].map(
                self.data.veh_type_min_load
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
            summary["fixed_cost"] = summary["vehicle_type"].map(
                self.data.veh_type_cost
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

    def _slot_type(
        self, solution: CplexApiSolution, plant: str, dealer: str, slot: int
    ) -> str:
        """Return the vehicle type a used slot ended up choosing."""
        best_type = self.data.all_veh_types[0]
        best_value = 0.0
        for v_type in self.data.all_veh_types:
            value = solution.variables.get(
                self.z[plant, dealer, v_type, slot], 0.0
            )
            if value > best_value:
                best_value = value
                best_type = v_type
        return best_type
