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
    # Set of all vehicle identifiers.
    all_vehicles: set[int] = field(init=False)
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
    # Cost of using each vehicle: {v_id: cost}.
    veh_cost: dict[int, float] = field(init=False)
    # Capacity of each vehicle: {v_id: capacity}.
    veh_cap: dict[int, int] = field(init=False)
    # Minimum load for each vehicle: {v_id: min_load}.
    veh_min_load: dict[int, int] = field(init=False)
    # All vehicle types.
    all_veh_types: list[str] = field(init=False)
    # Vehicle identifiers must be linked to their vehicle types.
    # Given a vehicle identifier, its type determines all vehicle attributes,
    # including cost, capacity, and minimum load.
    # {v_id: v_type}
    veh_type: dict[int, str] = field(init=False)
    # Similarly, each vehicle type maps to the identifiers assigned to that type: {v_type: {v_id}}.
    veh_ids: dict[str, set[int]] = field(init=False)
    # All data interfaces required by the single-period optimization problem are now declared.

    # Next, define the mathematical model, objective function, auxiliary variables, and decision variables.
    # Constraints are defined afterwards.
    # Gurobi model.
    model: gp.Model = field(init=False)
    # Objective function.
    obj: gp.LinExpr = field(init=False)
    # Auxiliary variables.
    # Whether vehicle l is used: {v_id: gp.Var}, where 1 means used and 0 means unused.
    z_l: gp.tupledict[int, gp.Var] = field(init=False)
    # Whether the actual load of vehicle l is below its minimum load: {v_id: gp.Var}, where 1 means below and 0 means not below.
    zz_l: gp.tupledict[int, gp.Var] = field(init=False)
    # Penalty indicator for vehicle l's minimum-load requirement: {v_id: gp.Var}, where 1 means penalized and 0 means not penalized.
    # A used vehicle is penalized only when its actual load is below the minimum load.
    # An unused vehicle is not penalized.
    p_l: gp.tupledict[int, gp.Var] = field(init=False)
    # Decision variables.
    # Whether vehicle l delivers goods from production plant i to dealer j: 1 for delivery, 0 otherwise: {(plant, dealer, v_id): gp.Var}.
    y_ijl: gp.tupledict[tuple[str, str, int], gp.Var] = field(init=False)
    # Quantity of SKU k loaded on vehicle l when delivering from plant i to dealer j: {(plant, dealer, sku_id, v_id): gp.Var}.
    x_ijkl: gp.tupledict[tuple[str, str, str, int], gp.Var] = field(init=False)

    # Decision-variable indices.
    y_indices: list[tuple[str, str, int]] = field(init=False)
    x_indices: list[tuple[str, str, str, int]] = field(init=False)
    new_duet_ik: set[tuple[str, str]] = field(init=False)

    # Constraints to add to the model.
    # Constraint 1: for every ordered pair (i, j) of production plant i and dealer j,
    # if plant i uses vehicle l to deliver goods to dealer j, then vehicle l is considered used.
    cons_veh_use: gp.tupledict[int, gp.Constr] = field(init=False)                      # {v_id: gp.Constr}
    # Constraint 2: goods can be loaded onto vehicle l only when vehicle l is used.
    cons_sku_load: gp.tupledict[int, gp.Constr] = field(init=False)                     # {v_id: gp.Constr}
    # Constraint 3: the actual load of vehicle l must be at least its minimum load.
    cons_veh_min_load: gp.tupledict[int, gp.Constr] = field(init=False)                 # {v_id: gp.Constr}
    # Constraint 4: relationship among the minimum-load penalty indicator p_l,
    # the vehicle usage state z_l, and the minimum-load satisfaction state zz_l.
    cons_auxi_relation: gp.tupledict[int, gp.Constr] = field(init=False)                # {v_id: gp.Constr}
    # Constraint 5: the actual load of vehicle l must not exceed its capacity.
    cons_veh_cap_limit: gp.tupledict[int, gp.Constr] = field(init=False)                # {v_id: gp.Constr}
    # Constraint 6: the total quantity of SKU k delivered to dealer j by all plants and vehicles must satisfy dealer j's demand.
    cons_dealer_demands: gp.tupledict[tuple[str, str], gp.Constr] = field(init=False)   # {(dealer, sku_id): gp.Constr}
    # Constraint 7: the quantity of SKU k shipped from plant i must not exceed its current inventory at plant i.
    cons_sku_supply: gp.tupledict[tuple[str, str], gp.Constr] = field(init=False)       # {(plant, sku_id): gp.Constr}
    # # Constraint 8: the quantity of SKU k shipped from all plants must not exceed total SKU k inventory.                       # {sku_id: gp.Constr}
    # cons_sku_total_supply: gp.tupledict[str, gp.Constr] = field(init=False)

    # Constraint 9: symmetry-breaking cuts reduce symmetry and shrink the solution space.
    # {(v_type, v_id, v_di_1): gp.Constr}
    cons_symmetry_break: gp.tupledict[tuple[str, int, int], gp.Constr] = field(init=False)
    # All attributes of the single-period optimization model are now declared.
    # The next step is to instantiate them.

    def __post_init__(self):
        self.param_alpha = self.data.param_alpha
        self.param_BIG_NUM = self.data.param_BIG_NUM
        self.param_gap_limit = self.data.param_gap_limit
        self.param_time_limit = self.data.param_time_limit
        self.param_output_flag = self.data.param_output_flag

        self.plants = self.data.plants
        self.dealers = self.data.dealers
        self.all_skus = self.data.all_skus
        self.all_vehicles = self.data.all_vehicles
        self.sku_sizes = self.data.sku_sizes
        self.sku_inv = self.data.sku_inv
        self.skus_plant = self.data.skus_plant
        self.sku_total_inv = self.data.sku_total_inv
        self.demands = self.data.demands
        self.skus_dealer = self.data.skus_dealer
        self.supply_chain = self.data.supply_chain
        self.veh_cost = self.data.veh_cost
        self.veh_cap = self.data.veh_cap
        self.veh_min_load = self.data.veh_min_load
        self.all_veh_types = self.data.all_veh_types
        self.veh_type = self.data.veh_type
        self.veh_ids = self.data.veh_ids

        self.model = gp.Model()
        self.obj = gp.LinExpr()

        # Add auxiliary variables.
        self.z_l = self.model.addVars(self.all_vehicles, lb=0, ub=1, vtype=GRB.BINARY, name="z_l")
        self.zz_l = self.model.addVars(self.all_vehicles, lb=0, ub=1, vtype=GRB.BINARY, name="zz_l")
        self.p_l = self.model.addVars(self.all_vehicles, lb=0, ub=1, vtype=GRB.BINARY, name="p_l")

        # Add decision variables.
        self.y_indices = [
            (plant, dealer, v_id) for (plant, dealer) in self.supply_chain 
            for v_id in self.all_vehicles
        ]
        self.y_ijl = self.model.addVars(self.y_indices, lb=0, ub=1, vtype=GRB.BINARY, name="y")

        # Keys in self.demands, (dealer, sku_id), indicate that dealer j requires SKU k.
        # Build the set of (plant, dealer, sku_id) triples for which plant i can supply SKU k to dealer j.
        # For each dealer j, retain only feasible supply triples.
        # A set is used to guarantee unique variable indices and remove duplicates.
        triple_ijk = {
            (plant, dealer1, sku_id) for (plant, dealer1) in self.supply_chain
            for (dealer2, sku_id) in self.demands if dealer1 == dealer2 if (plant, sku_id) in self.sku_inv
        }
        self.new_duet_ik = {(plant, sku_id) for (plant, dealer, sku_id) in triple_ijk}
        
        self.x_indices = [
            (plant, dealer, sku_id, v_id) for (plant, dealer, sku_id) in triple_ijk 
            for v_id in self.all_vehicles
        ]
        self.x_ijkl = self.model.addVars(self.x_indices ,lb=0, vtype=GRB.INTEGER, name="x")

        # # Build the objective function.
        self.obj = (gp.quicksum(self.veh_cost[v_id] * self.z_l[v_id] for v_id in self.all_vehicles) +
                    self.param_alpha * gp.quicksum(self.veh_cost[v_id] * self.p_l[v_id] for v_id in self.all_vehicles))
        # Set the optimization direction.
        self.model.setObjective(self.obj, sense=GRB.MINIMIZE)

        # Add constraints.
        # Constraint 1: for every ordered pair (i, j) of production plant i and dealer j,
        # if plant i uses vehicle l to deliver goods to dealer j, then vehicle l is considered used.
        self.cons_veh_use = self.model.addConstrs((
            self.y_ijl.sum('*', '*', v_id) == self.z_l[v_id]
            for v_id in self.all_vehicles), name="cons_veh_use")
        # Constraint 2: goods can be loaded onto vehicle l only when vehicle l is used.
        self.cons_sku_load = self.model.addConstrs((
            self.x_ijkl.sum(plant, dealer, '*', v_id) <= self.param_BIG_NUM * self.y_ijl[plant, dealer, v_id]
            for (plant, dealer, v_id) in self.y_indices), name="cons_sku_load")
        # Constraint 3: the actual load of vehicle l must be at least its minimum load.
        self.cons_veh_min_load = self.model.addConstrs((
            gp.quicksum(
                self.x_ijkl.sum('*', '*', sku_id, v_id) * self.sku_sizes[sku_id]
                for sku_id in self.all_skus)
            + self.zz_l[v_id] * self.param_BIG_NUM >= self.veh_min_load[v_id]
            for v_id in self.all_vehicles),
            name="cons_veh_min_load")
        # Constraint 4: relationship among the minimum-load penalty indicator p_l,
        # the vehicle usage state z_l, and the minimum-load satisfaction state zz_l.
        self.cons_auxi_relation = self.model.addConstrs((
            self.z_l[v_id] + self.zz_l[v_id] - 1 <= self.p_l[v_id]
            for v_id in self.all_vehicles), name="cons_auxi_relation")
        # Constraint 5: the actual load of vehicle l must not exceed its capacity.
        self.cons_veh_cap_limit = self.model.addConstrs((
            gp.quicksum(
                self.x_ijkl.sum('*', '*', sku_id, v_id) * self.sku_sizes[sku_id]
                for sku_id in self.all_skus)
            <= self.veh_cap[v_id]
            for v_id in self.all_vehicles), name="cons_veh_cap_limit")
        # Constraint 6: the total quantity of SKU k delivered to dealer j by all plants and vehicles must satisfy dealer j's demand.
        # Gurobi uses >= here; the mathematical model states equality to express the requirement more strictly.
        self.cons_dealer_demands = self.model.addConstrs((
            self.x_ijkl.sum('*', dealer, sku_id, '*') >= self.demands[dealer, sku_id]
            for (dealer, sku_id) in self.demands), name="cons_dealer_demands")
        # Constraint 7: the quantity of SKU k shipped from plant i must not exceed its current inventory at plant i.
        self.cons_sku_supply = self.model.addConstrs((
            self.x_ijkl.sum(plant, '*', sku_id, '*') <= self.sku_inv[plant, sku_id]
            for (plant, sku_id) in self.new_duet_ik), name="cons_sku_supply")

        # Constraint 8 (optional): cap total shipments of SKU k at its total
        # inventory across all plants, that is
        # sum(x_ijkl[:, :, k, :]) <= sku_total_inv[k]. The per-plant supply
        # constraint above already implies it for the instances used here, so
        # the aggregated cut is not added.

        # Constraint 9 (optional): symmetry-breaking cut z_l[v_id] <=
        # z_l[v_id_1] over consecutive vehicle ids of the same type. Gurobi
        # handles symmetry internally, so this cut is not added by default.

        # Enable solver output.
        self.model.setParam('OutputFlag', self.param_output_flag)
        # Set the MIP optimality gap.
        self.model.Params.MIPGap = self.param_gap_limit
        # Set the solve time limit.
        self.model.Params.TimeLimit = self.param_time_limit
        

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

            if self.model.IISMinimal:  # Check whether computeIIS() returned a minimal irreducible inconsistent subsystem.
                log_printer.print('IIS is minimal \n')
            else:
                log_printer.print('IIS is not minimal \n')
            log_printer.print('\n The following constraint (s) cannot be satisfied :')
            for c in self.model.getConstrs():
                if c.IISConstr:
                    log_printer.print(f'{c.ConstrName}')
            return False
        elif self.model.Status == GRB.TIME_LIMIT:
            log_printer.print("Time limit is reached! Solving process is stopped.")
            self.model.terminate()
            # Set JSONSolDetail = 1 so the saved solution file contains more detailed information.
            self.model.Params.JSONSolDetail = 1
            # Save the best feasible solution found so far to a JSON file.
            self.save_feasible_sol()
        elif self.model.Status == GRB.INTERRUPTED:
            log_printer.print("Optimization is interrupted by user!")
            return False
        else:  # model is optimal
            self.print_sol_summary()
            return True

    def save_feasible_sol(self):
        sol_file_loc = os.path.join(self.data.output_file_loc, self.data.dataset_name)
        if not os.path.exists(sol_file_loc):
            os.makedirs(sol_file_loc)
        sol_file_name = 'Feasible-Solution.json'
        sol_file = os.path.join(sol_file_loc, sol_file_name)
        self.model.write(sol_file)

    # Display a summary of the solution.
    def print_sol_summary(self):
        if self.model.Status == GRB.OPTIMAL:
            log_printer.print("Monolithic Model for single period is solved to be optimal.")
            log_printer.print(f"Total cost of optimal solution is {self.model.ObjVal}")

            total_veh_num = sum(var.Xn for var in self.z_l.values())
            log_printer.print(f"Total number of vehicles used is {total_veh_num}")

            veh_punished_num = sum(var.Xn for var in self.p_l.values())
            log_printer.print(f"Among used vehicles, there are {veh_punished_num} vehicle(s) "
                              f"that don't satisfy veh_min_load constraint and need to be punished.")

            total_ship_num = sum(var.Xn for var in self.x_ijkl.values())
            log_printer.print(f"Total SKU ship amount is {total_ship_num}")
