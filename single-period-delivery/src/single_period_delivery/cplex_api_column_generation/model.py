"""CPLEX Python API column generation for the single-period delivery problem.

The implementation keeps the original idea of column generation:

- the master problem selects vehicle load patterns;
- the pricing subproblems generate patterns with negative reduced cost;
- the final master problem is solved as an integer program.

The official CPLEX Python API is used directly. Master LP dual prices and
subproblem solutions are read through the API's solution objects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from ..cplex_api_support import CplexApiModel, CplexApiSolution
from ..column_generation.InputDataCG import InputDataCG


@dataclass(frozen=True)
class Pattern:
    """One feasible load pattern for a plant-dealer vehicle type."""

    plant: str
    dealer: str
    vehicle_type: str
    load: Dict[str, int]
    penalty_indicator: int
    cost: float


class CplexApiColumnGeneration:
    """Column-generation solver using the official CPLEX Python API."""

    def __init__(
        self,
        data: InputDataCG,
        *,
        max_iterations: int = 100,
        reduced_cost_tolerance: float = 1e-6,
        alpha: float | None = None,
        random_seed: int = 0,
    ):
        self.data = data
        self.max_iterations = max_iterations
        self.reduced_cost_tolerance = reduced_cost_tolerance
        # The monolithic model uses alpha * vehicle cost as the minimum-load
        # penalty. Keep the same coefficient in the master and pricing models.
        # Default to the coefficient declared by the dataset
        # (InputDataCG.param_alpha) rather than a hard-coded 2.0, so a dataset
        # with a different alpha is priced correctly.
        self.alpha = float(data.param_alpha if alpha is None else alpha)
        self.random_seed = int(random_seed)
        self.patterns: List[Pattern] = []
        self.iteration_log: List[Dict[str, float]] = []
        self.master_lp_objective: float | None = None
        self.final_objective: float | None = None
        self.final_status = "not_solved"

    @property
    def supply_pairs(self) -> List[Tuple[str, str]]:
        """Return all plant-dealer pairs with at least one shareable SKU."""
        pairs = []
        for plant in self.data.plants:
            for dealer in self.data.dealers:
                if any(
                    (dealer, sku) in self.data.demands
                    and (plant, sku) in self.data.sku_inv
                    for sku in self.data.all_skus
                ):
                    pairs.append((plant, dealer))
        return pairs

    def _available_skus(self, plant: str, dealer: str) -> List[str]:
        return [
            sku
            for sku in self.data.all_skus
            if (plant, sku) in self.data.sku_inv
            and (dealer, sku) in self.data.demands
        ]

    def _pattern_cost(
        self,
        vehicle_type: str,
        load: Dict[str, int],
    ) -> Tuple[float, int]:
        volume = sum(
            self.data.sku_sizes[sku] * qty for sku, qty in load.items()
        )
        below_min = volume < self.data.veh_type_min_load[vehicle_type]
        fixed_cost = float(self.data.veh_type_cost[vehicle_type])
        penalty = self.alpha * fixed_cost if below_min else 0.0
        return fixed_cost + penalty, int(below_min)

    def initialize_patterns(self) -> None:
        """Create a simple single-SKU pattern for every valid combination."""
        self.patterns.clear()
        for plant, dealer in self.supply_pairs:
            for sku in self._available_skus(plant, dealer):
                demand = int(self.data.demands[dealer, sku])
                inventory = int(self.data.sku_inv[plant, sku])
                for vehicle_type in self.data.all_veh_types:
                    capacity = float(self.data.veh_type_cap[vehicle_type])
                    volume = float(self.data.sku_sizes[sku])
                    max_units = int(math.floor(capacity / volume))
                    quantity = min(demand, inventory, max_units)
                    if quantity <= 0:
                        continue
                    load = {sku: quantity}
                    cost, below = self._pattern_cost(vehicle_type, load)
                    pattern = Pattern(
                        plant,
                        dealer,
                        vehicle_type,
                        load,
                        below,
                        cost,
                    )
                    if pattern not in self.patterns:
                        self.patterns.append(pattern)

    def seed_patterns_from_result_csv(self, result_csv: str | Path) -> int:
        """Add every vehicle load from an existing feasible result as a column.

        This is an optional validation aid. It does not replace pricing; it
        gives the restricted master a known feasible starting solution.
        """
        path = Path(result_csv)
        if not path.exists():
            raise FileNotFoundError(f"Seed result file not found: {path}")
        frame = pd.read_csv(path)
        required = {
            "plant_code",
            "client_code",
            "product_code",
            "vehicle_id",
            "vehicle_type",
            "qty",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(
                f"Seed result file is missing columns: {sorted(missing)}"
            )

        added = 0
        # A result row is keyed by vehicle_id. Grouping only by route and
        # vehicle type would merge several vehicles into one oversized
        # pattern, so keep vehicle_id in the grouping key.
        for (plant, dealer, vehicle_type, _vehicle_id), group in frame.groupby(
            ["plant_code", "client_code", "vehicle_type", "vehicle_id"]
        ):
            load = (
                group.groupby("product_code")["qty"].sum().astype(int).to_dict()
            )
            cost, below = self._pattern_cost(str(vehicle_type), load)
            pattern = Pattern(
                str(plant),
                str(dealer),
                str(vehicle_type),
                load,
                below,
                cost,
            )
            if pattern not in self.patterns:
                self.patterns.append(pattern)
                added += 1
        return added

    def _build_master(
        self,
        *,
        integer_patterns: bool,
        artificial_penalty: float = 1e6,
    ) -> Tuple[CplexApiModel, Dict[int, str], Dict[Tuple[str, str], str]]:
        """Build the restricted master problem."""
        model = CplexApiModel(
            (
                "single_period_master_mip"
                if integer_patterns
                else "single_period_master_lp"
            ),
        )
        pattern_vars: Dict[int, str] = {}
        artificial_vars: Dict[Tuple[str, str], str] = {}

        for index, pattern in enumerate(self.patterns):
            pattern_vars[index] = model.add_variable(
                f"lambda_{index}",
                kind="integer" if integer_patterns else "continuous",
            )

        for dealer, sku in self.data.demands:
            artificial_vars[dealer, sku] = model.add_variable(
                f"art_{dealer}_{sku}",
                kind="continuous",
            )

        objective: Dict[str, float] = {}
        for index, pattern in enumerate(self.patterns):
            objective[pattern_vars[index]] = pattern.cost
        for variable in artificial_vars.values():
            objective[variable] = artificial_penalty
        model.set_objective(objective)

        for dealer, sku in self.data.demands:
            terms = {
                pattern_vars[index]: float(pattern.load.get(sku, 0))
                for index, pattern in enumerate(self.patterns)
                if pattern.dealer == dealer and pattern.load.get(sku, 0)
            }
            terms[artificial_vars[dealer, sku]] = 1.0
            model.add_constraint(
                f"demand_{dealer}_{sku}",
                terms,
                "=",
                float(self.data.demands[dealer, sku]),
            )

        for plant, sku in self.data.sku_inv:
            terms = {
                pattern_vars[index]: float(pattern.load.get(sku, 0))
                for index, pattern in enumerate(self.patterns)
                if pattern.plant == plant and pattern.load.get(sku, 0)
            }
            if terms:
                model.add_constraint(
                    f"supply_{plant}_{sku}",
                    terms,
                    "<=",
                    float(self.data.sku_inv[plant, sku]),
                )

        return model, pattern_vars, artificial_vars

    def solve_master_lp(
        self,
        work_dir: str | Path,
    ) -> Tuple[CplexApiSolution, Dict[int, str], Dict[Tuple[str, str], str]]:
        """Solve one master LP relaxation and return its dual solution."""
        model, pattern_vars, artificial_vars = self._build_master(
            integer_patterns=False
        )
        solution = model.solve(
            work_dir,
            random_seed=self.random_seed,
        )
        self.master_lp_objective = solution.objective
        return solution, pattern_vars, artificial_vars

    def solve_pricing_subproblem(
        self,
        plant: str,
        dealer: str,
        vehicle_type: str,
        demand_duals: Dict[Tuple[str, str], float],
        supply_duals: Dict[Tuple[str, str], float],
        work_dir: str | Path,
    ) -> Pattern | None:
        """Solve one pricing subproblem and return a negative-cost pattern."""
        model = CplexApiModel(
            f"pricing_{plant}_{dealer}_{vehicle_type}",
            sense="max",
        )
        skus = self._available_skus(plant, dealer)
        if not skus:
            return None

        x = {
            sku: model.add_variable(
                f"x_{sku}",
                ub=min(
                    float(self.data.demands[dealer, sku]),
                    float(self.data.sku_inv[plant, sku]),
                ),
                kind="integer",
            )
            for sku in skus
        }
        penalty = model.add_variable("penalty", kind="binary")
        fixed_cost = float(self.data.veh_type_cost[vehicle_type])
        objective = {
            x[sku]: float(demand_duals.get((dealer, sku), 0.0))
            + float(supply_duals.get((plant, sku), 0.0))
            for sku in skus
        }
        objective[penalty] = -self.alpha * fixed_cost
        model.set_objective(objective)

        model.add_constraint(
            "capacity",
            {
                x[sku]: float(self.data.sku_sizes[sku])
                for sku in skus
            },
            "<=",
            float(self.data.veh_type_cap[vehicle_type]),
        )
        big_m = float(self.data.veh_type_cap[vehicle_type])
        min_load_terms = {
            x[sku]: float(self.data.sku_sizes[sku]) for sku in skus
        }
        min_load_terms[penalty] = big_m
        model.add_constraint(
            "minimum_load",
            min_load_terms,
            ">=",
            float(self.data.veh_type_min_load[vehicle_type]),
        )
        model.add_constraint(
            "nonempty",
            {x[sku]: 1.0 for sku in skus},
            ">=",
            1.0,
        )
        solution = model.solve(work_dir, random_seed=self.random_seed)
        if solution.status not in {"optimal", "time_limit_feasible"}:
            return None
        if solution.objective is None:
            return None
        if solution.objective <= self.reduced_cost_tolerance:
            return None

        load = {
            sku: int(round(solution.variables.get(x[sku], 0.0)))
            for sku in skus
            if solution.variables.get(x[sku], 0.0) > 1e-7
        }
        if not load:
            return None
        below = int(
            sum(self.data.sku_sizes[sku] * qty for sku, qty in load.items())
            < self.data.veh_type_min_load[vehicle_type]
        )
        cost, _ = self._pattern_cost(vehicle_type, load)
        return Pattern(
            plant,
            dealer,
            vehicle_type,
            load,
            below,
            cost,
        )

    def run(
        self,
        work_dir: str | Path,
        *,
        final_time_limit: float = 60.0,
    ) -> Dict[str, object]:
        """Run column generation and then solve the integer restricted master."""
        work_path = Path(work_dir)
        work_path.mkdir(parents=True, exist_ok=True)
        # The runner may already have loaded seed columns from a feasible
        # solution. Preserve those columns instead of clearing them here.
        if not self.patterns:
            self.initialize_patterns()

        for iteration in range(1, self.max_iterations + 1):
            solution, _, _ = self.solve_master_lp(
                work_path / f"iter_{iteration}",
            )
            if solution.status != "optimal" or solution.objective is None:
                break

            demand_duals = {
                key: value
                for key, value in solution.duals.items()
                if key.startswith("demand_")
            }
            supply_duals = {
                key: value
                for key, value in solution.duals.items()
                if key.startswith("supply_")
            }
            # CPLEX removes constraint prefixes only when names collide; the
            # canonical names are preserved in the LP file, so reconstruct the
            # tuple keys from the parsed names.
            demand_lookup = {
                (dealer, sku): demand_duals.get(f"demand_{dealer}_{sku}", 0.0)
                for dealer, sku in self.data.demands
            }
            supply_lookup = {
                (plant, sku): supply_duals.get(f"supply_{plant}_{sku}", 0.0)
                for plant, sku in self.data.sku_inv
            }

            new_patterns = []
            for plant, dealer in self.supply_pairs:
                for vehicle_type in self.data.all_veh_types:
                    pattern = self.solve_pricing_subproblem(
                        plant,
                        dealer,
                        vehicle_type,
                        demand_lookup,
                        supply_lookup,
                        work_path / f"iter_{iteration}" / "pricing",
                    )
                    if pattern is not None and pattern not in self.patterns:
                        new_patterns.append(pattern)

            self.iteration_log.append(
                {
                    "iteration": float(iteration),
                    "master_objective": float(solution.objective),
                    "new_patterns": float(len(new_patterns)),
                    "total_patterns": float(len(self.patterns)),
                }
            )
            if not new_patterns:
                break
            self.patterns.extend(new_patterns)

        master_model, pattern_vars, artificial_vars = self._build_master(
            integer_patterns=True
        )
        final_solution = master_model.solve(
            work_path / "final_integer_master",
            time_limit=final_time_limit,
            mip_gap=0.0,
            random_seed=self.random_seed,
            parameter_overrides={"mip.strategy.heuristicfreq": 1},
        )
        self.final_status = final_solution.status
        self.final_objective = final_solution.objective
        artificial_usage = sum(
            final_solution.variables.get(variable, 0.0)
            for variable in artificial_vars.values()
        )
        return {
            "status": final_solution.status,
            "objective": final_solution.objective,
            "patterns": len(self.patterns),
            "iterations": len(self.iteration_log),
            "artificial_usage": artificial_usage,
            "final_solution": final_solution,
            "pattern_vars": pattern_vars,
            "artificial_vars": artificial_vars,
        }

    def write_outputs(
        self,
        result: Dict[str, object],
        output_dir: str | Path,
    ) -> Dict[str, object]:
        """Write the final pattern solution in the project CSV schema."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        solution: CplexApiSolution = result["final_solution"]  # type: ignore[assignment]
        pattern_vars: Dict[int, str] = result["pattern_vars"]  # type: ignore[assignment]

        rows = []
        vehicle_id = 0
        for index, pattern in enumerate(self.patterns):
            value = solution.variables.get(pattern_vars[index], 0.0)
            copies = int(round(value))
            for _ in range(max(0, copies)):
                vehicle_id += 1
                for sku, qty in pattern.load.items():
                    rows.append(
                        {
                            "Index": len(rows) + 1,
                            "day": 1,
                            "plant_code": pattern.plant,
                            "client_code": pattern.dealer,
                            "product_code": sku,
                            "vehicle_id": vehicle_id,
                            "vehicle_type": pattern.vehicle_type,
                            "qty": int(qty),
                        }
                    )
        frame = pd.DataFrame(
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
        frame.to_csv(output_path / "result.csv", index=False)

        details = frame.drop(columns=["Index"]).copy()
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

        pd.DataFrame(self.iteration_log).to_csv(
            output_path / "column_generation_iterations.csv",
            index=False,
        )
        return {
            "status": self.final_status,
            "objective": self.final_objective,
            "patterns": len(self.patterns),
            "deliveries": int(len(frame)),
            "vehicles": int(frame["vehicle_id"].nunique())
            if not frame.empty
            else 0,
            "artificial_usage": float(result.get("artificial_usage", 0.0)),
        }
