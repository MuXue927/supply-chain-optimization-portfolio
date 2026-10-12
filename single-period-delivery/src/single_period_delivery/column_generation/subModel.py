# Subproblem for the column-generation algorithm.
# This is the pricing problem of the decomposition approach for the
# single-period problem; it generates vehicle-route patterns for the master.
# It assumes a single period, SKU inventory greater than demand, and unlimited
# storage capacity at the production plants.
# It supports multiple production plants, dealers, vehicle types, and SKUs.
# Production plants may supply overlapping SKU sets.
import math
import os
from dataclasses import dataclass, field
from typing import Dict, Hashable, Set, Tuple

import gurobipy as gp
from gurobipy import GRB

from .InputDataCG import InputDataCG


@dataclass
class SubModel:
    fact_id: str
    dealer_id: str
    vehicle_type: str
    data: InputDataCG

    # Environment parameters.
    # MIP gap tolerance. Optimization stops once the configured gap is reached; the default is 0.
    param_gap_limit: float = field(init=False)
    # MIP time limit in seconds; the default is 60.
    param_time_limit: float = field(init=False)
    # Whether to print the Gurobi solver log.
    param_output_flag: int = field(init=False)

    # Set of all SKUs.
    all_skus: Set[str] = field(init=False)
    # SKUs required by the current dealer and available from the current production plant.
    require_skus: Set[str] = field(init=False)
    # Volume per unit for each SKU: {sku: size}.
    sku_sizes: Dict[str, float] = field(init=False)
    # SKU quantity available at production plants: {(plant, sku): qty}.
    sku_inv: Dict[Tuple[str, str], int] = field(init=False)
    # Dealer demand by SKU: {(dealer, sku): qty}.
    demands: Dict[Tuple[str, str], int] = field(init=False)
    # SKU set requested by each dealer.
    skus_dealer: Dict[str, Set[str]] = field(init=False)

    # Usage cost for each vehicle type: {v_type: cost}.
    veh_type_cost: Dict[str, float] = field(init=False)
    # Capacity limit for each vehicle type: {v_type: capacity}.
    veh_type_cap: Dict[str, int] = field(init=False)
    # Minimum shipment quantity for each vehicle type: {v_type: min_load}.
    veh_type_min_load: Dict[str, int] = field(init=False)

    # Define the mathematical model, objective function, and decision variables.
    # Add the constraints after defining those components.
    # Subproblem model.
    model: gp.Model = field(init=False)
    # Objective function.
    obj: gp.LinExpr = field(init=False)
    # Decision variables.
    # Quantity of SKU k loaded in a vehicle of type f traveling from production plant i to dealer j.
    # Production plant i, dealer j, and vehicle type f are fixed in each subproblem.
    d_xs: gp.tupledict[str, gp.Var] = field(init=False)
    # Auxiliary variable: 1 if the loaded volume for vehicle type f is below the minimum shipment
    # quantity, and 0 otherwise.
    p_f: gp.Var = field(init=False)

    # Constraints added to the model.
    # Capacity constraint for vehicle type f.
    cons_veh_cap: gp.Constr = field(init=False)
    # Minimum shipment quantity constraint for vehicle type f.
    cons_veh_min_load: gp.Constr = field(init=False)
    # Upper-bound constraints for the decision variables.
    cons_xs_bounds1: gp.tupledict[Hashable, gp.Constr] = field(init=False)
    cons_xs_bounds2: gp.tupledict[Hashable, gp.Constr] = field(init=False)

    def __post_init__(self):
        self.param_gap_limit = self.data.param_gap_limit
        self.param_time_limit = self.data.param_time_limit
        self.param_output_flag = self.data.param_output_flag

        self.all_skus = self.data.all_skus
        self.sku_sizes = self.data.sku_sizes
        self.sku_inv = self.data.sku_inv
        self.demands = self.data.demands
        self.skus_dealer = self.data.skus_dealer
        self.veh_type_cost = self.data.veh_type_cost
        self.veh_type_cap = self.data.veh_type_cap
        self.veh_type_min_load = self.data.veh_type_min_load

        # The subproblem objective is built only after dual prices for the master constraints are available.
        self.model = gp.Model()
        self.obj = gp.LinExpr()

        self.require_skus = self.data.available_skus_to_dealer(self.fact_id, self.dealer_id)
        self.d_xs = self.model.addVars(self.require_skus, lb=0, vtype=GRB.INTEGER, name="d_xs")
        self.p_f = self.model.addVar(lb=0, vtype=GRB.BINARY, name="p_f")

        # Compute the total volume of all SKU types for the Big-M term in the minimum shipment constraint.
        # BigM = sum(self.demands[dealer_id, sku_id] * self.sku_sizes[sku_id] for (dealer_id, sku_id) in self.demands)

        # BigM = sum(self.demands[self.dealer_id, sku_id] * self.sku_sizes[sku_id] for sku_id in self.require_skus)

        # delta_k = {sku_id: min(self.demands[self.dealer_id, sku_id], self.sku_inv[self.fact_id, sku_id])
        #            for sku_id in self.require_skus}
        delta_k = {sku_id: max(self.demands[self.dealer_id, sku_id], self.sku_inv[self.fact_id, sku_id])
                   for sku_id in self.require_skus}
        BigM = sum(delta_k[sku_id] * self.sku_sizes[sku_id] for sku_id in self.require_skus)
        BigM = math.ceil(BigM)

        # v1 = sum(self.demands[self.dealer_id, sku_id] * self.sku_sizes[sku_id] for sku_id in self.require_skus)
        # v2 = sum(self.sku_inv[self.fact_id, sku_id] * self.sku_sizes[sku_id] for sku_id in self.require_skus)
        # BigM = min(v1, v2)
        # BigM = max(v1, v2)

        # BigM = BigNum

        # Add constraints.
        self.cons_veh_cap = self.model.addConstr((
            gp.quicksum(
                self.d_xs[sku_id] * self.sku_sizes[sku_id] for sku_id in self.require_skus
            ) <= self.veh_type_cap[self.vehicle_type]
        ), name="cons_veh_cap")

        self.cons_veh_min_load = self.model.addConstr((
            gp.quicksum(
                self.d_xs[sku_id] * self.sku_sizes[sku_id] for sku_id in self.require_skus
            ) + BigM * self.p_f >= self.veh_type_min_load[self.vehicle_type]
        ), name="cons_veh_min_load")

        self.cons_xs_bounds = self.model.addConstrs((
            self.d_xs[sku_id] <= min(self.demands[self.dealer_id, sku_id], self.sku_inv[self.fact_id, sku_id])
            for sku_id in self.require_skus
        ), name="cons_xs_bounds")

        self.model.setParam(GRB.Param.OutputFlag, 0)

    def run(self):
        # log_printer.print("Optimizing sub model...")
        self.model.setObjective(self.obj, sense=GRB.MAXIMIZE)
        self.model.Params.FeasibilityTol = 1e-9
        self.model.optimize()
        # Dump the LP into this run's output directory rather than the current
        # working directory, so generated model files never land next to the
        # source (see .gitignore: *.lp / *.ilp).
        os.makedirs(self.data.output_file_loc, exist_ok=True)
        self.model.write(os.path.join(self.data.output_file_loc, "SubModel.lp"))

        if self.model.Status in (GRB.INF_OR_UNBD, GRB.INFEASIBLE, GRB.UNBOUNDED):
            return False
        return True

    def get_load_pattern(self):
        """
        retrieve load patterns and prepare to submit them to master model.
        """
        return {sku_id: var.Xn for sku_id, var in self.d_xs.items() if var.Xn > 0}
