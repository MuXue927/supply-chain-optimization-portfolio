# Master model for the column-generation algorithm.
# This is the decomposition approach for the single-period problem: the master
# selects vehicle-route patterns and subModel.py prices new ones.
# It assumes a single period, SKU inventory greater than demand, and unlimited
# storage capacity at the production plants.
# It supports multiple production plants, dealers, vehicle types, and SKUs.
# Production plants may supply overlapping SKU sets.
import os
import random
import time
from dataclasses import dataclass, field

import gurobipy as gp
from gurobipy import GRB

from .InputDataCG import InputDataCG
from .optutility import POSITIVEEPS, LogPrinter
from .vehicle import Vehicle

# Create a shared LogPrinter instance.
log_printer = LogPrinter(time.time())
# Set force_plain=True to force plain-text output, for example when redirecting to a file.
# log_printer.set_output_mode(force_plain=True)

@dataclass
class MasterModel:
    data: InputDataCG
    # Single-period problems use a one-day planning horizon by default.
    horizons = 1
    # Planning starts on day 1 by default.
    day = 1

    # Environment parameters.
    # MIP gap tolerance. Optimization stops once the configured gap is reached; the default is 0.
    param_gap_limit: float = field(init=False)
    # MIP time limit in seconds; the default is 60.
    param_time_limit: float = field(init=False)
    # Whether to print the Gurobi solver log.
    param_output_flag: int = field(init=False)

    # Load-pattern identifier. It only needs to be unique and is used as part of a key.
    pattern_id = 0
    # Records each load pattern added to the master model. The key is the pattern name used by d_xs,
    # and the value is the concrete SKU loading pattern and its quantities.
    pattern_content = {}
    # Number of load patterns currently in the model.
    num_patterns = 0
    # Number of load patterns removed per shrink operation; currently disabled.
    remove_batch = 100

    # Data interfaces corresponding to the symbols in the mathematical model.
    # Set of production plants.
    plants: set[str] = field(init=False)
    # Set of dealers.
    dealers: set[str] = field(init=False)
    # Set of all SKUs.
    all_skus: set[str] = field(init=False)
    # Set of all vehicle types.
    all_veh_types: set[str] = field(init=False)

    # Volume per unit for each SKU: {sku: size}.
    sku_sizes: dict[str, float] = field(init=False)
    # Current SKU inventory at production plants: {(plant, sku): qty}.
    sku_inv: dict[tuple[str, str], int] = field(init=False)
    # Total inventory of each SKU: {sku: total_qty}.
    sku_total_inv: dict[str, int] = field(init=False)
    # SKU set that each production plant can supply.
    skus_plant: dict[str, set[str]] = field(init=False)
    # Dealer demand by SKU: {(dealer, sku): qty}.
    demands: dict[tuple[str, str], int] = field(init=False)
    # SKU set requested by each dealer.
    skus_dealer: dict[str, set[str]] = field(init=False)

    # Define the mathematical model, objective function, artificial variables, and decision variables.
    # Add the constraints after defining those components.
    # Master model.
    model: gp.Model = field(init=False)
    # Objective function.
    obj: gp.LinExpr = field(init=False)
    # Artificial variable alpha.
    d_alpha: gp.tupledict[tuple[str, str], gp.Var] = field(init=False)    # {(dealer, sku): gp.Var}
    # Artificial variable beta.
    d_beta: gp.tupledict[str, gp.Var] = field(init=False)     # {sku: gp.Var}
    # Artificial variable gamma.
    d_gamma: gp.tupledict[tuple[str, str], gp.Var] = field(init=False)  # {(plant, sku): gp.Var}
    # Decision variable x_ijf.
    # Number of vehicles of type f whose loading pattern is used from production plant i to dealer j.
    # {(plant, dealer, v_type, load_pattern_id): gp.Var}
    d_xs: gp.tupledict[tuple[str, str, str, int], gp.Var] = field(init=False)

    # Constraints added to the model.
    # Total quantity of SKU k shipped to dealer j from all plants i using all vehicle types f
    # must equal dealer j's demand for SKU k.
    cons_dealer_demands: gp.tupledict[tuple[str, str], gp.Constr] = field(init=False)   # {(dealer, sku_id): gp.Constr}
    # Total shipped quantity of SKU k must not exceed its total inventory.
    cons_sku_supply: gp.tupledict[str, gp.Constr] = field(init=False)  # {sku_id: gp.Constr}
    # Quantity of SKU k shipped from plant i must not exceed plant i's inventory of SKU k.
    cons_fact_sku_inv: gp.tupledict[tuple[str, str], gp.Constr] = field(init=False)    # {(plant, sku_id): gp.Constr}

    def __post_init__(self):
        self.param_gap_limit = self.data.param_gap_limit
        self.param_time_limit = self.data.param_time_limit
        self.param_output_flag = self.data.param_output_flag

        self.plants = self.data.plants
        self.dealers = self.data.dealers
        self.all_skus = self.data.all_skus
        self.all_veh_types = self.data.all_veh_types
        self.sku_sizes = self.data.sku_sizes
        self.sku_inv = self.data.sku_inv
        self.sku_total_inv = self.data.sku_total_inv
        self.skus_plant = self.data.skus_plant
        self.demands = self.data.demands
        self.skus_dealer = self.data.skus_dealer

        self.model = gp.Model()
        self.obj = gp.LinExpr()

        duet_dealer_sku = {(dealer, sku_id) for (dealer, sku_id) in self.demands}
        self.d_alpha = self.model.addVars(duet_dealer_sku, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="d_alpha")

        self.d_beta = self.model.addVars(self.all_skus, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="d_beta")

        duet_plant_sku = {(plant, sku_id) for (plant, sku_id) in self.sku_inv}
        self.d_gamma = self.model.addVars(duet_plant_sku, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="d_gamma")

        self.d_xs = gp.tupledict()

        self.obj = (self.data.param_pun_factor1 * self.d_alpha.sum('*', '*') +
                    self.data.param_pun_factor2 * self.d_beta.sum('*') +
                    self.data.param_pun_factor3 * self.d_gamma.sum('*', '*'))
        self.model.setObjective(self.obj, sense=GRB.MINIMIZE)

        self.cons_dealer_demands = self.model.addConstrs((
            self.d_alpha[dealer, sku_id] >= self.demands[dealer, sku_id]
            for (dealer, sku_id) in duet_dealer_sku
        ), name="cons_dealer_demands")

        self.cons_sku_supply = self.model.addConstrs((
            self.d_beta[sku_id] >= -self.sku_total_inv[sku_id]
            for sku_id in self.all_skus
        ), name="cons_sku_supply")

        self.cons_fact_sku_inv = self.model.addConstrs((
            self.d_gamma[plant, sku_id] >= -self.sku_inv[plant, sku_id]
            for (plant, sku_id) in duet_plant_sku
        ), name="cons_fact_sku_inv")

        # Enable the solver log.
        self.model.setParam('OutputFlag', self.param_output_flag)
        # Set the MIP gap tolerance.
        self.model.Params.MIPGap = self.param_gap_limit
        # Set the solver time limit.
        self.model.Params.TimeLimit = self.param_time_limit

    def get_alpha_size(self):
        non_zero_alpha = {(dealer, sku_id) for (dealer, sku_id), var in self.d_alpha.items() if var.X > POSITIVEEPS}
        return len(non_zero_alpha)

    def get_beta_size(self):
        non_zero_beta = {sku_id for sku_id, var in self.d_beta.items() if var.X > POSITIVEEPS}
        return len(non_zero_beta)

    def get_gamma_size(self):
        non_zero_gamma = {(plant, sku_id) for (plant, sku_id), var in self.d_gamma.items() if var.X > POSITIVEEPS}
        return len(non_zero_gamma)

    def deactivate_alpha(self):
        for (dealer, sku_id) in self.d_alpha:
            self.d_alpha[dealer, sku_id].setAttr('UB', 0)

    def deactivate_beta(self):
        for sku_id in self.d_beta:
            self.d_beta[sku_id].setAttr('UB', 0)

    def deactivate_gamma(self):
        for (plant, sku_id) in self.d_gamma:
            self.d_gamma[plant, sku_id].setAttr('UB', 0)

    def deactivate_artificial_vars(self):
        self.deactivate_alpha()
        self.deactivate_beta()
        self.deactivate_gamma()

    def activate_alpha(self):
        for (dealer, sku_id) in self.d_alpha:
            self.d_alpha[dealer, sku_id].setAttr('UB', GRB.INFINITY)

    def activate_beta(self):
        for sku_id in self.d_beta:
            self.d_beta[sku_id].setAttr('UB', GRB.INFINITY)

    def activate_gamma(self):
        for (plant, sku_id) in self.d_gamma:
            self.d_gamma[plant, sku_id].setAttr('UB', GRB.INFINITY)

    def activate_artificial_vars(self):
        self.activate_alpha()
        self.activate_beta()
        self.activate_gamma()

    def change_var_type(self, var_type=GRB.INTEGER):
        log_printer.print("Set variables in the master model to integer type")
        for var in self.model.getVars():
            # if var.VarName.startswith("d_xs_"):
            var.setAttr('VType', var_type)
        self.model.Params.MIPGap = self.param_gap_limit
        self.model.Params.TimeLimit = self.param_time_limit
        self.model.setParam('OutputFlag', self.param_output_flag)

    def remove_partial_artificial_var(self):
        """
        remove artificial variables whose value equals zero and update model.
        """
        remove_alpha = {(dealer, sku_id): var for (dealer, sku_id), var in self.d_alpha.items() if var.X <= POSITIVEEPS}
        reserve_alpha = {(dealer, sku_id): var for (dealer, sku_id), var in self.d_alpha.items() if var.X > POSITIVEEPS}
        self.d_alpha = gp.tupledict(reserve_alpha)

        remove_beta = {sku_id: var for sku_id, var in self.d_beta.items() if var.X <= POSITIVEEPS}
        reserve_beta = {sku_id: var for sku_id, var in self.d_beta.items() if var.X > POSITIVEEPS}
        self.d_beta = gp.tupledict(reserve_beta)

        remove_gamma = {(plant, sku_id): var for (plant, sku_id), var in self.d_gamma.items() if var.X <= POSITIVEEPS}
        reserve_gamma = {(plant, sku_id): var for (plant, sku_id), var in self.d_gamma.items() if var.X > POSITIVEEPS}
        self.d_gamma = gp.tupledict(reserve_gamma)

        self.model.remove(remove_alpha)
        self.model.remove(remove_beta)
        self.model.remove(remove_gamma)

        self.model.update()
        self.obj = self.model.getObjective()  # type: ignore
        return self.obj.size()

    def insert_pattern(self, fact_id: str, dealer_id: str, vehicle_type: str,
                        obj_contribution: float, load_pattern: dict[str, int]):
        """
        insert load patterns submitted by sub models into master model

        :param fact_id: a specific plant
        :param dealer_id: a specific dealer
        :param vehicle_type: a specific vehicle type
        :param obj_contribution: the profit contribution of a specific load pattern
        :param load_pattern: a specific load pattern
        :return: None
        """
        pattern_column = gp.Column()

        # A leading * unpacks a list into separate positional arguments.
        # zip(*iterables, strict=False)
        # Iterate over several iterables in parallel, producing tuples with an item from each one.
        # dict.items() returns a view of (key, value) pairs.
        skus_loaded, load_num = zip(*list(load_pattern.items()))
        skus_loaded = list(skus_loaded)
        load_num = list(load_num)

        cons_dealer_demands_impacted = [self.cons_dealer_demands[dealer_id, sku_id] for sku_id in skus_loaded]
        pattern_column.addTerms(load_num, cons_dealer_demands_impacted)

        # neg_load_num = [-qty for qty in load_num]

        cons_sku_supply_impacted = [self.cons_sku_supply[sku_id] for sku_id in skus_loaded]
        pattern_column.addTerms(load_num, cons_sku_supply_impacted)

        cons_fact_sku_inv_impacted = [self.cons_fact_sku_inv[fact_id, sku_id] for sku_id in skus_loaded]
        pattern_column.addTerms(load_num, cons_fact_sku_inv_impacted)

        d_xs_name = f"d_xs_{fact_id}_{dealer_id}_{vehicle_type}_{self.pattern_id}"
        self.d_xs[fact_id, dealer_id, vehicle_type, self.pattern_id] = (
            self.model.addVar(lb=0, obj=obj_contribution, vtype=GRB.CONTINUOUS, column=pattern_column, name=d_xs_name))

        self.pattern_content[d_xs_name] = load_pattern

        self.pattern_id += 1
        self.num_patterns += 1
        self.model.update()

    def shrink(self):
        """
        remove load patterns when the number of load patterns is greater than predefined value.
        """
        log_printer.print("Start to shrink master model...")
        remove_pattern_keys = [key for key in self.d_xs if self.d_xs[key].X == 0]
        remove_pattern_nums = len(remove_pattern_keys)
        random.shuffle(remove_pattern_keys)
        while remove_pattern_nums > 0 and self.num_patterns > self.data.param_pattern_num_shrink_to:
            remove_key = remove_pattern_keys.pop()
            self.model.remove(self.d_xs[remove_key])
            self.d_xs.pop(remove_key)
            remove_pattern_nums -= 1
            self.num_patterns -= 1
        self.model.update()
        log_printer.print("Shrink Done!")

    def initialize_solution(self):
        for (plant, dealer) in self.data.construct_supply_chain():
            # Select the vehicle type with the largest capacity.
            max_veh_type = max(self.all_veh_types, key=float)
            selected_vehicle = Vehicle(plant, dealer, max_veh_type, self.data)
            # SKU types and quantities requested by the current dealer.
            orders = {sku_id: qty for (dealer_id, sku_id), qty in self.demands.items() if dealer_id == dealer}
            for sku_id, qty in orders.items():
                # The production plant can supply the SKU requested by the dealer.
                if sku_id in self.skus_plant[plant]:
                    while qty > 0:
                        qty = selected_vehicle.load(sku_id, qty)  # type: ignore
                        if qty > 0:
                            self.insert_pattern(plant, dealer, max_veh_type,
                                                self.data.veh_type_cost[max_veh_type], selected_vehicle.cargo)
                            selected_vehicle.clear()
            
            if not selected_vehicle.is_empty():
                self.insert_pattern(plant, dealer, max_veh_type,
                                    self.data.veh_type_cost[max_veh_type], selected_vehicle.cargo)
                selected_vehicle.clear()
        pass

    def run(self):
        # log_printer.print("Optimizing master model...")
        self.model.optimize()
        # Dump the LP into this run's output directory rather than the current
        # working directory, so generated model files never land next to the
        # source (see .gitignore: *.lp / *.ilp).
        os.makedirs(self.data.output_file_loc, exist_ok=True)
        self.model.write(os.path.join(self.data.output_file_loc, "MasterModel.lp"))

        if self.model.Status in (GRB.INF_OR_UNBD, GRB.INFEASIBLE, GRB.UNBOUNDED):
            log_printer.print("Exception: Master Model Is Infeasible or Unbounded!")
            self.model.computeIIS()
            self.model.write(
                os.path.join(
                    self.data.output_file_loc, "Infeasible_MasterModel.ilp"
                )
            )

            if self.model.IISMinimal:  # Check whether computeIIS() returned a minimal irreducible infeasible subsystem.
                log_printer.print('IIS is minimal \n')
            else:
                log_printer.print('IIS is not minimal \n')
            log_printer.print('\n The following constraint (s) cannot be satisfied: ')
            for c in self.model.getConstrs():
                if c.IISConstr:
                    log_printer.print(f'{c.ConstrName}')
            return False
        else:
            self.print_sol_summary()
            return True

    def print_sol_summary(self):
        real_obj = sum(var.X for var in self.d_xs.values())
        log_printer.print(
            f"\tRealObj={real_obj:.1f}"
            f"\tNo. Alpha={self.get_alpha_size()}"
            f"\tNo. Beta={self.get_beta_size()}"
            f"\tNo. Gamma={self.get_gamma_size()}"
            f"\tNo. Pattern={self.num_patterns}"
            f"\tTotalObj={self.model.ObjVal:.2f}"
        )
