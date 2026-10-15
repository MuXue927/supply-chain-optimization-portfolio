"""Official CPLEX Python API monolithic model for the multi-period problem.

This module is intentionally parallel to ``monolithic``. It implements the same
route-local slot reformulation with the official ``cplex`` Python API.

The CPLEX Python package is imported only when the model is constructed. This
keeps the module importable in environments that do not have IBM CPLEX Python
installed and produces a clear dependency error at model-build time.
"""

from __future__ import annotations

import io
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..monolithic.InputDataMultiple import InputDataMultiple

IndexRouteSlotPeriod = tuple[str, str, int, int]
IndexRouteTypeSlotPeriod = tuple[str, str, str, int, int]
IndexRouteSkuSlotPeriod = tuple[str, str, str, int, int]
IndexPlantSkuPeriod = tuple[str, str, int]

#: Statuses that carry a usable primal solution. Solver-level failures surface
#: as other status names, so callers can tell a real result from a failed run.
SOLVED_STATUSES: frozenset[str] = frozenset(
    {"optimal", "time_limit_feasible", "feasible"}
)


def _load_cplex() -> Any:
    """Import the official CPLEX Python package with an actionable error."""
    try:
        import cplex
    except ImportError as exc:
        raise ImportError(
            "The official IBM CPLEX Python API is required for the "
            "cplex_api_monolithic model. Install the CPLEX Python package "
            "with `uv add cplex` or make the CPLEX Python bindings available "
            "in the active virtual environment."
        ) from exc
    return cplex


def _safe_name(name: str) -> str:
    """Convert an internal identifier to a CPLEX-safe symbol name."""
    safe = "".join(
        character if character.isalnum() or character == "_" else "_"
        for character in str(name)
    )
    if not safe or not (safe[0].isalpha() or safe[0] == "_"):
        safe = f"v_{safe}"
    return safe


@dataclass
class CplexApiSolution:
    """Solution and solver metadata returned by the CPLEX Python API."""

    status: str
    objective: float | None
    variables: dict[str, float] = field(default_factory=dict)
    best_bound: float | None = None
    mip_gap: float | None = None
    solver_status: str = ""
    solver_status_code: int | None = None
    stdout: str = ""


class MultiPeriodCplexApiMonolithic:
    """Build and solve the multi-period monolithic model with CPLEX Python."""

    def __init__(self, data: InputDataMultiple):
        self.data = data
        self.cplex = _load_cplex()
        self.model = self.cplex.Cplex()
        self.model.objective.set_sense(self.model.objective.sense.minimize)

        self.periods = list(range(1, int(data.horizons) + 1))

        # Variable dictionaries map mathematical indices to CPLEX column
        # indices. The index-to-name mapping is retained for solution export.
        # Route-local slots: a slot belongs to one route and its vehicle type is
        # a decision variable in every period, so neither a global fleet index
        # nor a y assignment variable is required.
        self.z: dict[IndexRouteTypeSlotPeriod, int] = {}
        self.zz: dict[IndexRouteSlotPeriod, int] = {}
        self.p: dict[IndexRouteTypeSlotPeriod, int] = {}
        self.x: dict[IndexRouteSkuSlotPeriod, int] = {}
        self.s: dict[IndexPlantSkuPeriod, int] = {}

        self._variable_names: list[str] = []
        self._variable_lower_bounds: list[float] = []
        self._variable_upper_bounds: list[float] = []
        self._variable_types: list[str] = []
        self._objective_coefficients: list[float] = []

        self._constraint_expressions: list[Any] = []
        self._constraint_senses: list[str] = []
        self._constraint_rhs: list[float] = []
        self._constraint_names: list[str] = []

        self._build_variables()
        self._build_objective()
        self._build_constraints()

    def _add_variable(
        self,
        name: str,
        *,
        kind: str,
        lower_bound: float = 0.0,
        upper_bound: float | None = None,
        objective: float = 0.0,
        target: MutableMapping[Any, int] | None = None,
        key: Any = None,
    ) -> int:
        """Register one CPLEX variable and return its column index."""
        if upper_bound is None:
            upper_bound = float(self.cplex.infinity)
        if kind == "binary":
            lower_bound, upper_bound = 0.0, 1.0

        type_by_kind = {
            "continuous": "C",
            "binary": "B",
            "integer": "I",
        }
        if kind not in type_by_kind:
            raise ValueError(f"Unsupported CPLEX variable type: {kind!r}")

        index = len(self._variable_names)
        self._variable_names.append(_safe_name(name))
        self._variable_lower_bounds.append(float(lower_bound))
        self._variable_upper_bounds.append(float(upper_bound))
        self._variable_types.append(type_by_kind[kind])
        self._objective_coefficients.append(float(objective))
        if target is not None:
            target[key] = index
        return index

    def _build_variables(self) -> None:
        """Create route-local z, zz, p, x variables and plant inventory s."""
        data = self.data

        for plant, dealer in sorted(data.route_slots):
            for slot in data.route_slots[plant, dealer]:
                for period in self.periods:
                    self._add_variable(
                        f"zz_{plant}_{dealer}_{slot}_{period}",
                        kind="binary",
                        target=self.zz,
                        key=(plant, dealer, slot, period),
                    )
                    for sku in sorted(data.supply_chain[plant, dealer]):
                        self._add_variable(
                            f"x_{plant}_{dealer}_{sku}_{slot}_{period}",
                            kind="integer",
                            target=self.x,
                            key=(plant, dealer, sku, slot, period),
                        )
                    for v_type in sorted(data.all_veh_types):
                        self._add_variable(
                            f"z_{plant}_{dealer}_{v_type}_{slot}_{period}",
                            kind="binary",
                            target=self.z,
                            key=(plant, dealer, v_type, slot, period),
                        )
                        self._add_variable(
                            f"p_{plant}_{dealer}_{v_type}_{slot}_{period}",
                            kind="binary",
                            target=self.p,
                            key=(plant, dealer, v_type, slot, period),
                        )

        for plant, skus in sorted(data.skus_plant.items()):
            for sku in sorted(skus):
                for period in range(int(data.horizons) + 1):
                    initial_inventory = float(data.sku_initial_inv.get((plant, sku), 0))
                    is_initial = period == 0
                    self._add_variable(
                        f"s_{plant}_{sku}_{period}",
                        kind="integer",
                        lower_bound=initial_inventory if is_initial else 0.0,
                        upper_bound=initial_inventory if is_initial else None,
                        target=self.s,
                        key=(plant, sku, period),
                    )

        self.model.variables.add(
            lb=self._variable_lower_bounds,
            ub=self._variable_upper_bounds,
            types=self._variable_types,
            names=self._variable_names,
        )

    def _build_objective(self) -> None:
        """Set fixed vehicle cost plus the alpha-scaled minimum-load penalty."""
        for (plant, dealer, v_type, slot, period), index in self.z.items():
            self._objective_coefficients[index] = float(
                self.data.veh_type_cost[v_type]
            )
        for (plant, dealer, v_type, slot, period), index in self.p.items():
            self._objective_coefficients[index] = float(self.data.param_alpha) * float(
                self.data.veh_type_cost[v_type]
            )

        self.model.objective.set_linear(
            [
                (index, coefficient)
                for index, coefficient in enumerate(self._objective_coefficients)
                if abs(coefficient) > 1e-15
            ]
        )

    def _add_constraint(
        self,
        name: str,
        terms: Mapping[int, float],
        sense: str,
        rhs: float,
    ) -> None:
        """Queue one linear constraint for a single bulk CPLEX API call."""
        indices: list[int] = []
        coefficients: list[float] = []
        for index, coefficient in terms.items():
            coefficient = float(coefficient)
            if abs(coefficient) <= 1e-15:
                continue
            indices.append(int(index))
            coefficients.append(coefficient)

        self._constraint_expressions.append(
            self.cplex.SparsePair(ind=indices, val=coefficients)
        )
        self._constraint_senses.append(sense)
        self._constraint_rhs.append(float(rhs))
        self._constraint_names.append(_safe_name(name))

    def _build_constraints(self) -> None:
        """Add the route-local constraints for every route slot and period."""
        data = self.data

        for plant, dealer in sorted(data.route_slots):
            skus = sorted(data.supply_chain[plant, dealer])
            for slot in data.route_slots[plant, dealer]:
                for period in self.periods:
                    self._add_constraint(
                        f"one_type_{plant}_{dealer}_{slot}_{period}",
                        {
                            self.z[plant, dealer, v_type, slot, period]: 1.0
                            for v_type in data.all_veh_types
                        },
                        "L",
                        1.0,
                    )

                    capacity_terms: dict[int, float] = {
                        self.x[plant, dealer, sku, slot, period]: float(
                            data.sku_sizes[sku]
                        )
                        for sku in skus
                    }
                    for v_type in data.all_veh_types:
                        capacity_terms[
                            self.z[plant, dealer, v_type, slot, period]
                        ] = -float(data.veh_type_cap[v_type])
                    self._add_constraint(
                        f"capacity_{plant}_{dealer}_{slot}_{period}",
                        capacity_terms,
                        "L",
                        0.0,
                    )

                    load_terms: dict[int, float] = {
                        self.x[plant, dealer, sku, slot, period]: float(
                            data.sku_sizes[sku]
                        )
                        for sku in skus
                    }
                    load_terms[self.zz[plant, dealer, slot, period]] = float(
                        data.param_BIG_NUM
                    )
                    for v_type in data.all_veh_types:
                        load_terms[
                            self.z[plant, dealer, v_type, slot, period]
                        ] = -float(data.veh_type_min_load[v_type])
                    self._add_constraint(
                        f"minimum_load_{plant}_{dealer}_{slot}_{period}",
                        load_terms,
                        "G",
                        0.0,
                    )

                    for v_type in data.all_veh_types:
                        self._add_constraint(
                            f"penalty_link_{plant}_{dealer}_{v_type}_{slot}_{period}",
                            {
                                self.z[
                                    plant, dealer, v_type, slot, period
                                ]: 1.0,
                                self.zz[plant, dealer, slot, period]: 1.0,
                                self.p[
                                    plant, dealer, v_type, slot, period
                                ]: -1.0,
                            },
                            "L",
                            1.0,
                        )

        # Constraint 3: minimum load, penalty linking, and capacity were added
        # above for each route-local slot. The remaining rows below are the
        # unchanged aggregate demand, inventory balance, and plant capacity.
        for dealer, sku in sorted(data.demands):
            terms = {
                self.x[plant, dealer, sku, slot, period]: 1.0
                for plant in data.plants
                for slot in data.route_slots.get((plant, dealer), [])
                for period in self.periods
                if (plant, dealer, sku, slot, period) in self.x
            }
            self._add_constraint(
                f"demand_{dealer}_{sku}",
                terms,
                "G",
                float(data.demands[dealer, sku]),
            )

        inventory_keys = {
            (plant, sku) for plant, skus in data.skus_plant.items() for sku in skus
        }
        for plant, sku in sorted(inventory_keys):
            for period in self.periods:
                terms = {
                    self.x[plant, dealer, sku, slot, period]: 1.0
                    for dealer in data.dealers
                    for slot in data.route_slots.get((plant, dealer), [])
                    if (plant, dealer, sku, slot, period) in self.x
                }
                terms[self.s[plant, sku, period]] = 1.0
                terms[self.s[plant, sku, period - 1]] = -1.0
                self._add_constraint(
                    f"inventory_balance_{plant}_{sku}_{period}",
                    terms,
                    "E",
                    float(data.sku_prod_each_day.get((plant, sku, period), 0)),
                )

        for plant in sorted(data.plants):
            for period in self.periods:
                terms = {
                    self.s[plant, sku, period]: 1.0 for sku in data.skus_plant[plant]
                }
                self._add_constraint(
                    f"plant_capacity_{plant}_{period}",
                    terms,
                    "L",
                    float(data.plant_inv_limit[plant]),
                )

        if self._constraint_expressions:
            self.model.linear_constraints.add(
                lin_expr=self._constraint_expressions,
                senses=self._constraint_senses,
                rhs=self._constraint_rhs,
                names=self._constraint_names,
            )

    @staticmethod
    def _status_name(
        model: Any,
        status_code: int,
        status_text: str,
    ) -> str:
        """Translate a CPLEX status code into the portfolio status vocabulary."""
        status = model.solution.status

        def constant(name: str) -> int | None:
            return getattr(status, name, None)

        if status_code in {
            constant("MIP_optimal"),
            constant("optimal"),
        }:
            return "optimal"
        if status_code == constant("MIP_time_limit_feasible"):
            return "time_limit_feasible"
        if status_code == constant("MIP_time_limit_infeasible"):
            return "time_limit_infeasible"
        if status_code in {
            constant("MIP_infeasible"),
            constant("infeasible"),
        }:
            return "infeasible"
        if status_code in {
            constant("MIP_unbounded"),
            constant("unbounded"),
        }:
            return "unbounded"

        lowered = status_text.lower()
        if "time limit" in lowered:
            try:
                if model.solution.is_primal_feasible():
                    return "time_limit_feasible"
            except Exception:
                pass
            return "time_limit_infeasible"
        if "integer infeasible" in lowered:
            return "integer_infeasible"
        if "infeasible" in lowered:
            return "infeasible"
        if "unbounded" in lowered:
            return "unbounded"
        return lowered or "unknown"

    def solve(
        self,
        *,
        time_limit: float | None = None,
        mip_gap: float | None = None,
        random_seed: int | None = None,
        write_lp: str | Path | None = None,
    ) -> CplexApiSolution:
        """Solve the model through the official CPLEX Python API."""
        if time_limit is not None:
            self.model.parameters.timelimit.set(float(time_limit))
        if mip_gap is not None:
            self.model.parameters.mip.tolerances.mipgap.set(float(mip_gap))
        if random_seed is not None:
            self.model.parameters.randomseed.set(int(random_seed))

        # Keep the CLI and API variants aligned by enabling periodic MIP
        # heuristics. CPLEX still controls the heuristic details internally.
        try:
            self.model.parameters.mip.strategy.heuristicfreq.set(1)
        except Exception:
            pass

        log_stream = io.StringIO()
        self.model.set_log_stream(log_stream)
        self.model.set_results_stream(log_stream)

        self.model.solve()

        status_code = int(self.model.solution.get_status())
        try:
            status_text = str(self.model.solution.get_status_string())
        except Exception:
            status_text = str(self.model.solution.status[status_code])
        status_name = self._status_name(
            self.model,
            status_code,
            status_text,
        )

        objective: float | None = None
        variables: dict[str, float] = {}
        try:
            feasible = bool(self.model.solution.is_primal_feasible())
        except Exception:
            feasible = False

        if feasible:
            try:
                objective = float(self.model.solution.get_objective_value())
            except Exception:
                objective = None
            try:
                raw_values = self.model.solution.get_values()
                variables = {
                    name: float(value)
                    for name, value in zip(
                        self._variable_names,
                        raw_values,
                    )
                }
            except Exception:
                variables = {}

        best_bound: float | None = None
        try:
            best_bound = float(self.model.solution.get_best_objective_value())
        except Exception:
            best_bound = None

        relative_gap: float | None = None
        try:
            relative_gap = float(self.model.solution.get_mip_relative_gap())
        except Exception:
            relative_gap = None

        if write_lp is not None:
            lp_path = Path(write_lp)
            lp_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.write(str(lp_path))

        solver_log = log_stream.getvalue()
        if not solver_log:
            solver_log = (
                f"CPLEX status code: {status_code}\nCPLEX status: {status_text}\n"
            )

        solution = CplexApiSolution(
            status=status_name,
            objective=objective,
            variables=variables,
            best_bound=best_bound,
            mip_gap=relative_gap,
            solver_status=status_text,
            solver_status_code=status_code,
            stdout=solver_log,
        )
        # Release the native CPLEX model. Without this every solve leaks a
        # solver handle, which accumulates across the batch experiments (and can
        # exhaust CPLEX's concurrent-licence pool). Everything the caller needs
        # has already been captured above; `write_outputs` reads only this
        # solution object.
        try:
            self.model.end()
        except Exception:
            pass
        return solution

    def write_outputs(
        self,
        solution: CplexApiSolution,
        output_dir: str | Path,
    ) -> dict[str, object]:
        """Write result files compatible with the existing experiment schema."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        rows = []
        for key, index in self.x.items():
            value = solution.variables.get(self._variable_names[index], 0.0)
            if value <= 1e-7:
                continue
            plant, dealer, sku, slot, period = key
            rows.append(
                {
                    "day": period,
                    "plant_code": plant,
                    "client_code": dealer,
                    "product_code": sku,
                    "vehicle_id": slot,
                    "vehicle_type": self._slot_type(
                        solution, plant, dealer, slot, period
                    ),
                    "qty": int(round(value)),
                }
            )

        result = pd.DataFrame(
            rows,
            columns=[
                "day",
                "plant_code",
                "client_code",
                "product_code",
                "vehicle_id",
                "vehicle_type",
                "qty",
            ],
        )
        result.to_csv(output_path / "opt_result.csv", index=False)

        details = result.copy()
        if not details.empty:
            details["standard_size"] = details["product_code"].map(self.data.sku_sizes)
            details["carry_standard_size"] = details["vehicle_type"].map(
                self.data.veh_type_cap
            )
            details["min_standard_size"] = details["vehicle_type"].map(
                self.data.veh_type_min_load
            )
            details["occupied_size"] = details["qty"] * details["standard_size"]
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
                summary["occupied_size"] + 1e-9 < summary["min_standard_size"]
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

        inventory_rows = []
        for key, index in self.s.items():
            plant, sku, period = key
            if period == 0:
                continue
            inventory_rows.append(
                {
                    "plant_code": plant,
                    "product_code": sku,
                    "day": period,
                    "volume": int(
                        round(
                            solution.variables.get(
                                self._variable_names[index],
                                0.0,
                            )
                        )
                    ),
                }
            )
        pd.DataFrame(inventory_rows).to_csv(
            output_path / "sku_inv_left.csv",
            index=False,
        )

        pd.DataFrame(columns=["client_code", "product_code", "volume"]).to_csv(
            output_path / "non_fulfill.csv", index=False
        )

        return {
            "status": solution.status,
            # A solver-level failure must not be recorded as a successful run
            # that merely happens to have no objective.
            "solved": solution.status in SOLVED_STATUSES,
            "objective": solution.objective,
            "deliveries": int(len(result)),
            "vehicles": int(
                round(
                    sum(
                        solution.variables.get(self._variable_names[index], 0.0)
                        for index in self.z.values()
                    )
                )
            ),
        }

    def _slot_type(
        self,
        solution: CplexApiSolution,
        plant: str,
        dealer: str,
        slot: int,
        period: int,
    ) -> str:
        """Return the vehicle type a used route slot selected in a period."""
        best_type = self.data.all_veh_types[0]
        best_value = 0.0
        for v_type in self.data.all_veh_types:
            value = solution.variables.get(
                self._variable_names[
                    self.z[plant, dealer, v_type, slot, period]
                ],
                0.0,
            )
            if value > best_value:
                best_value = value
                best_type = v_type
        return best_type
