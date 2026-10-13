# Gurobi implementation of the single-period mathematical model.
# The model assumes a single period, SKU inventory greater than or equal to
# demand, unlimited storage capacity at the production plants, multiple plants
# and dealers, multiple vehicle types and SKUs, and overlapping SKU portfolios
# across plants.
import os
import time
from dataclasses import dataclass, field

import gurobipy as gp
from gurobipy import GRB

from .InputDataSingle import InputDataSingle
from .optutility import LogPrinter

# Create the log printer instance.
log_printer = LogPrinter(time.time())
# # Setting force_plain=True forces plain-text output, which is useful when redirecting to a file.
# # To retain formatted text such as colors, comment out this line and disable output redirection.
# # This also requires enabling the commented code in run_model in optLogicSingle.py and commenting out the corresponding active block.
# log_printer.set_output_mode(force_plain=True)


@dataclass
class MonolithicModelSingle:
    data: InputDataSingle
    # The single-period problem uses a planning horizon of one period.
    horizons = 1
    # Planning starts on the first day by default.
    day = 1

    # Environment parameters.
    # Penalty coefficient alpha used by the model.
    param_alpha: float = field(init=False)
    # Big-M parameter used by the model.
    param_BIG_NUM: int = field(init=False)
    # MIP optimality gap at which the solver stops; the default is 0.
    param_gap_limit: float = field(init=False)
    # Time limit for the MIP solve; the default is 60 seconds.
    param_time_limit: float = field(init=False)
    # Whether to print Gurobi solver output.
    param_output_flag: int = field(init=False)

    # Data interface aligned with the notation in the mathematical model.
    # Set of production plants.
    plants: set[str] = field(init=False)
    # Set of dealers.
    dealers: set[str] = field(init=False)
    # Set of all SKUs.
    all_skus: set[str] = field(init=False)
    # Unit volume of each SKU: {sku: size}.
    sku_sizes: dict[str, float] = field(init=False)
    # Current SKU inventory at each production plant: {(plant, sku): qty}.
    sku_inv: dict[tuple[str, str], int] = field(init=False)
    # Total inventory of each SKU: {sku: total_qty}.
    sku_total_inv: dict[str, int] = field(init=False)
    # Dealer demand for each SKU: {(dealer, sku): qty}.
    demands: dict[tuple[str, str], int] = field(init=False)
    # Supply relationships between production plants and dealers: {(plant, dealer): {sku}}.
    supply_chain: dict[tuple[str, str], set[str]] = field(init=False)
    # All vehicle types.
    all_veh_types: list[str] = field(init=False)
    # Vehicle-type attributes used by the route-local formulation.
    veh_type_cap: dict[str, int] = field(init=False)
    veh_type_min_load: dict[str, int] = field(init=False)
    veh_type_cost: dict[str, float] = field(init=False)
    # Route-local slots: {(plant, dealer): [slot, ...]}.
    route_slots: dict[tuple[str, str], list[int]] = field(init=False)

    # Gurobi model.
    model: gp.Model = field(init=False)
    # Objective function.
    obj: gp.LinExpr = field(init=False)

    # Auxiliary variables, for every route (i, j), slot n, and vehicle type f:
    # z_ijfn  - slot n of route (i, j) uses vehicle type f
    # zz_ijn  - slot n is allowed to fall below its minimum load
    # p_ijfn  - slot n is used with type f *and* below the minimum load
    z_ijfn: gp.tupledict[tuple[str, str, str, int], gp.Var] = field(init=False)
    zz_ijn: gp.tupledict[tuple[str, str, int], gp.Var] = field(init=False)
    p_ijfn: gp.tupledict[tuple[str, str, str, int], gp.Var] = field(init=False)
    # Quantity of SKU k loaded on slot n of route (i, j):
    # {(plant, dealer, sku_id, slot): gp.Var}.
    x_ijkn: gp.tupledict[tuple[str, str, str, int], gp.Var] = field(init=False)

    # Decision-variable indices.
    z_indices: list[tuple[str, str, str, int]] = field(init=False)
    zz_indices: list[tuple[str, str, int]] = field(init=False)
    x_indices: list[tuple[str, str, str, int]] = field(init=False)
    new_duet_ik: set[tuple[str, str]] = field(init=False)

    # Constraints added to the model.
    # Constraint 1: at most one vehicle type is assigned to a slot.
    cons_one_type: gp.tupledict[tuple[str, str, int], gp.Constr] = field(init=False)
    # Constraint 2: the load on a slot cannot exceed the capacity of its type.
    cons_veh_cap_limit: gp.tupledict[tuple[str, str, int], gp.Constr] = field(
        init=False
    )
    # Constraint 3: the load must satisfy the selected type's minimum load
    # unless the binary slack variable zz is activated.
    cons_veh_min_load: gp.tupledict[tuple[str, str, int], gp.Constr] = field(
        init=False
    )
    # Constraint 4: link the minimum-load penalty to z and zz.
    cons_auxi_relation: gp.tupledict[
        tuple[str, str, str, int], gp.Constr
    ] = field(init=False)
    # Constraint 5: total shipments to each dealer meet demand.
    cons_dealer_demands: gp.tupledict[tuple[str, str], gp.Constr] = field(init=False)
    # Constraint 6: total shipments from each plant do not exceed inventory.
    cons_sku_supply: gp.tupledict[tuple[str, str], gp.Constr] = field(init=False)
    # Constraint 7: optional symmetry-breaking cuts are not added.
    cons_symmetry_break: gp.tupledict[tuple[str, str, int], gp.Constr] = field(
        init=False
    )

    def __post_init__(self):
        self.param_alpha = self.data.param_alpha
        self.param_BIG_NUM = self.data.param_BIG_NUM
        self.param_gap_limit = self.data.param_gap_limit
        self.param_time_limit = self.data.param_time_limit
        self.param_output_flag = self.data.param_output_flag

        self.plants = self.data.plants
        self.dealers = self.data.dealers
        self.all_skus = self.data.all_skus
        self.sku_sizes = self.data.sku_sizes
        self.sku_inv = self.data.sku_inv
        self.sku_total_inv = self.data.sku_total_inv
        self.demands = self.data.demands
        self.supply_chain = self.data.supply_chain
        self.all_veh_types = self.data.all_veh_types
        self.veh_type_cap = self.data.veh_type_cap
        self.veh_type_min_load = self.data.veh_type_min_load
        self.veh_type_cost = self.data.veh_type_cost
        self.route_slots = self.data.route_slots

        self.model = gp.Model()
        self.obj = gp.LinExpr()

        # Build the route-local slot variables. A slot belongs to one route and
        # its type is a model decision, so no global fleet index or y variable
        # is needed.
        self.zz_indices = [
            (plant, dealer, slot)
            for (plant, dealer), slots in self.route_slots.items()
            for slot in slots
        ]
        self.z_indices = [
            (plant, dealer, v_type, slot)
            for (plant, dealer), slots in self.route_slots.items()
            for slot in slots
            for v_type in self.all_veh_types
        ]
        self.x_indices = [
            (plant, dealer, sku_id, slot)
            for (plant, dealer), slots in self.route_slots.items()
            for slot in slots
            for sku_id in self.supply_chain[plant, dealer]
        ]

        self.zz_ijn = self.model.addVars(
            self.zz_indices, lb=0, ub=1, vtype=GRB.BINARY, name="zz"
        )
        self.z_ijfn = self.model.addVars(
            self.z_indices, lb=0, ub=1, vtype=GRB.BINARY, name="z"
        )
        self.p_ijfn = self.model.addVars(
            self.z_indices, lb=0, ub=1, vtype=GRB.BINARY, name="p"
        )
        self.x_ijkn = self.model.addVars(
            self.x_indices, lb=0, vtype=GRB.INTEGER, name="x"
        )

        # Build the objective function.
        self.obj = gp.quicksum(
            self.veh_type_cost[v_type] * self.z_ijfn[plant, dealer, v_type, slot]
            for (plant, dealer, v_type, slot) in self.z_indices
        ) + gp.quicksum(
            self.param_alpha
            * self.veh_type_cost[v_type]
            * self.p_ijfn[plant, dealer, v_type, slot]
            for (plant, dealer, v_type, slot) in self.z_indices
        )
        self.model.setObjective(self.obj, sense=GRB.MINIMIZE)

        # Constraint 1: at most one type per slot.
        self.cons_one_type = self.model.addConstrs(
            (
                gp.quicksum(
                    self.z_ijfn[plant, dealer, v_type, slot]
                    for v_type in self.all_veh_types
                )
                <= 1
                for plant, dealer, slot in self.zz_indices
            ),
            name="cons_one_type",
        )

        # Constraint 2: the load on a slot is bounded by the selected type.
        self.cons_veh_cap_limit = self.model.addConstrs(
            (
                gp.quicksum(
                    self.x_ijkn[plant, dealer, sku_id, slot]
                    * self.sku_sizes[sku_id]
                    for sku_id in self.supply_chain[plant, dealer]
                )
                - gp.quicksum(
                    self.veh_type_cap[v_type]
                    * self.z_ijfn[plant, dealer, v_type, slot]
                    for v_type in self.all_veh_types
                )
                <= 0
                for plant, dealer, slot in self.zz_indices
            ),
            name="cons_veh_cap_limit",
        )

        # Constraint 3: minimum load, with a binary slack for penalized slots.
        self.cons_veh_min_load = self.model.addConstrs(
            (
                gp.quicksum(
                    self.x_ijkn[plant, dealer, sku_id, slot]
                    * self.sku_sizes[sku_id]
                    for sku_id in self.supply_chain[plant, dealer]
                )
                + self.zz_ijn[plant, dealer, slot] * self.param_BIG_NUM
                - gp.quicksum(
                    self.veh_type_min_load[v_type]
                    * self.z_ijfn[plant, dealer, v_type, slot]
                    for v_type in self.all_veh_types
                )
                >= 0
                for plant, dealer, slot in self.zz_indices
            ),
            name="cons_veh_min_load",
        )

        # Constraint 4: z + zz - 1 <= p for each possible type.
        self.cons_auxi_relation = self.model.addConstrs(
            (
                self.z_ijfn[plant, dealer, v_type, slot]
                + self.zz_ijn[plant, dealer, slot]
                - 1
                <= self.p_ijfn[plant, dealer, v_type, slot]
                for plant, dealer, v_type, slot in self.z_indices
            ),
            name="cons_auxi_relation",
        )

        # Constraint 5: dealer demand.
        self.cons_dealer_demands = self.model.addConstrs(
            (
                gp.quicksum(
                    self.x_ijkn[plant, dealer, sku_id, slot]
                    for plant in self.plants
                    for slot in self.route_slots.get((plant, dealer), [])
                    if (plant, dealer, sku_id, slot) in self.x_ijkn
                )
                >= self.demands[dealer, sku_id]
                for dealer, sku_id in self.demands
            ),
            name="cons_dealer_demands",
        )

        self.new_duet_ik = {
            (plant, sku_id)
            for plant, dealer in self.supply_chain
            for sku_id in self.supply_chain[plant, dealer]
        }

        # Constraint 6: plant supply.
        self.cons_sku_supply = self.model.addConstrs(
            (
                gp.quicksum(
                    self.x_ijkn[plant, dealer, sku_id, slot]
                    for dealer in self.dealers
                    for slot in self.route_slots.get((plant, dealer), [])
                    if (plant, dealer, sku_id, slot) in self.x_ijkn
                )
                <= self.sku_inv[plant, sku_id]
                for plant, sku_id in self.new_duet_ik
            ),
            name="cons_sku_supply",
        )

        # Enable solver output.
        self.model.setParam("OutputFlag", self.param_output_flag)
        # Set the MIP optimality gap.
        self.model.Params.MIPGap = self.param_gap_limit
        # Set the solve time limit.
        self.model.Params.TimeLimit = self.param_time_limit

    def slot_type(self, plant: str, dealer: str, slot: int) -> str:
        """Return the vehicle type selected by a used route slot."""
        best_type = self.all_veh_types[0]
        best_value = 0.0
        for v_type in self.all_veh_types:
            value = self.z_ijfn[plant, dealer, v_type, slot].Xn
            if value > best_value:
                best_value = value
                best_type = v_type
        return best_type

    def run(self):
        self.model.optimize()
        # Dump the LP into this run's output directory rather than the current
        # working directory, so generated model files never land next to the
        # source (see .gitignore: *.lp / *.ilp).
        os.makedirs(self.data.output_file_loc, exist_ok=True)
        self.model.write(
            os.path.join(self.data.output_file_loc, "MonolithicModelSingle.lp")
        )

        if self.model.Status in (GRB.INF_OR_UNBD, GRB.INFEASIBLE, GRB.UNBOUNDED):
            self.model.computeIIS()
            self.model.write(
                os.path.join(
                    self.data.output_file_loc, "Infeasible_MonolithicModelSingle.ilp"
                )
            )

            if self.model.IISMinimal:
                log_printer.print("IIS is minimal \n")
            else:
                log_printer.print("IIS is not minimal \n")
            log_printer.print("\n The following constraint (s) cannot be satisfied :")
            for constraint in self.model.getConstrs():
                if constraint.IISConstr:
                    log_printer.print(f"{constraint.ConstrName}")
            return False
        elif self.model.Status == GRB.TIME_LIMIT:
            log_printer.print("Time limit is reached! Solving process is stopped.")
            self.model.terminate()
            self.model.Params.JSONSolDetail = 1
            self.save_feasible_sol()
        elif self.model.Status == GRB.INTERRUPTED:
            log_printer.print("Optimization is interrupted by user!")
            return False
        else:
            self.print_sol_summary()
            return True

    def save_feasible_sol(self):
        sol_file_loc = os.path.join(self.data.output_file_loc, self.data.dataset_name)
        if not os.path.exists(sol_file_loc):
            os.makedirs(sol_file_loc)
        sol_file_name = "Feasible-Solution.json"
        sol_file = os.path.join(sol_file_loc, sol_file_name)
        self.model.write(sol_file)

    def print_sol_summary(self):
        if self.model.Status == GRB.OPTIMAL:
            log_printer.print(
                "Monolithic Model for single period is solved to be optimal."
            )
            log_printer.print(f"Total cost of optimal solution is {self.model.ObjVal}")

            total_veh_num = sum(var.Xn for var in self.z_ijfn.values())
            log_printer.print(f"Total number of vehicles used is {total_veh_num}")

            veh_punished_num = sum(var.Xn for var in self.p_ijfn.values())
            log_printer.print(
                f"Among used vehicles, there are {veh_punished_num} vehicle(s) "
                f"that don't satisfy veh_min_load constraint and need to be punished."
            )

            total_ship_num = sum(var.Xn for var in self.x_ijkn.values())
            log_printer.print(f"Total SKU ship amount is {total_ship_num}")
