# Gurobi formulation of the multi-period mathematical model.
# This module implements the model defined as Model 2.
# Model 2 assumes multiple periods, SKU supply = initial inventory + production >=
# demand, and a finite storage capacity at each production plant.
# It supports multiple plants, dealers, vehicle types, and SKUs.
# Each plant can supply the same SKUs, and different plants' SKU sets may overlap.
import os
import time
from gurobipy import GRB
import gurobipy as gp
from typing import Dict, Tuple, Set, List
from dataclasses import dataclass, field
from .optutility import LogPrinter
from .InputDataMultiple import InputDataMultiple


# Create a LogPrinter instance.
log_printer = LogPrinter(time.time())
# Set force_plain=True to force plain-text output, for example when redirecting to a file.
# To display formatted text such as colored output, comment out this line and disable redirection.
# You must also update run_model in optLogicMultiple.py by enabling the commented block
# and disabling the corresponding active block.
# log_printer.set_output_mode(force_plain=True)


@dataclass
class MonolithicModelMultiple:
    data: InputDataMultiple
    # Default planning horizon for multi-period instances.
    # The actual value is the maximum produce_date in warehouse_production.csv.
    horizons = 1
    # Current planning period; the default is 1.
    # In particular, day = 0 represents the initial state, so day = 0, 1, ..., horizons.
    # day = 1

    # Environment parameters.
    # Penalty coefficient Alpha used in the model.
    param_alpha: float = field(init=False)
    # Big-M constant used in the model.
    param_BIG_NUM: int = field(init=False)
    # MIP optimality gap. The solver stops when this gap is reached; the default is 0.
    param_gap_limit: float = field(init=False)
    # MIP solver time limit; the default is 60 seconds.
    param_time_limit: float = field(init=False)
    # Whether to output Gurobi solver logs.
    param_output_flag: int = field(init=False)

    # Data interfaces aligned with the notation in the mathematical model.
    # Set of production plants.
    plants: Set[str] = field(init=False)
    # Set of dealers.
    dealers: Set[str] = field(init=False)
    # Set of all SKUs.
    all_skus: Set[str] = field(init=False)
    # Set of all vehicle IDs.
    all_vehicles: Set[int] = field(init=False)
    # Set of planning periods t.
    periods: Set[int] = field(init=False)

    # Volume per SKU {sku: size}.
    sku_sizes: Dict[str, float] = field(init=False)
    # Planned SKU production at each plant in period t {(plant, sku, day): qty}.
    sku_prod_each_day: Dict[Tuple[str, str, int], int] = field(init=False)
    # Initial SKU inventory at each plant {(plant, sku): qty}.
    sku_initial_inv: Dict[Tuple[str, str], int] = field(init=False)
    # SKUs that each plant can supply.
    skus_plant: Dict[str, Set[str]] = field(init=False)

    # Dealer demand for each SKU {(dealer, sku): qty}.
    demands: Dict[Tuple[str, str], int] = field(init=False)
    # SKUs required by each dealer.
    skus_dealer: Dict[str, Set[str]] = field(init=False)

    # Use cost for each vehicle {v_id: cost}.
    veh_cost: Dict[int, float] = field(init=False)
    # Capacity upper bound for each vehicle {v_id: capacity}.
    veh_cap: Dict[int, float] = field(init=False)
    # Minimum load for each vehicle {v_id: min_load}.
    veh_min_load: Dict[int, float] = field(init=False)
    # Inventory capacity upper bound for each plant {plant: max_cap}.
    plant_inv_limit: Dict[str, float] = field(init=False)

    # All vehicle types.
    all_veh_types: List[str] = field(init=False)
    # Vehicle IDs must be linked to vehicle types.
    # Given a vehicle ID, its type identifies all relevant vehicle attributes,
    # including use cost, capacity, and minimum load.
    # {v_id: v_type}
    veh_type: Dict[int, str] = field(init=False)
    # Conversely, each vehicle type identifies its vehicle IDs {v_type: {v_id}}.
    veh_ids: Dict[str, Set[int]] = field(init=False)

    # Supply relationships between plants and dealers {(plant, dealer): {sku}}.
    supply_chain: Dict[Tuple[str, str], Set[str]] = field(init=False)
    # All data interfaces required by the multi-period optimization problem are defined.

    # Next, define the mathematical model, objective function, auxiliary variables,
    # decision variables, and finally the constraints.
    # Gurobi model.
    model: gp.Model = field(init=False)
    # Objective function.
    obj: gp.LinExpr = field(init=False)

    # Auxiliary variables.
    # Whether vehicle l is used in period t {(v_id, day): gp.Var}; 1 = used, 0 = unused.
    z_lt: gp.tupledict[Tuple[int, int], gp.Var] = field(init=False)
    # Whether vehicle l's actual load in period t is below its minimum load
    # {(v_id, day): gp.Var}; 1 = below, 0 = not below.
    zz_lt: gp.tupledict[Tuple[int, int], gp.Var] = field(init=False)
    # Penalty indicator for vehicle l's minimum-load violation in period t
    # {(v_id, day): gp.Var}; 1 = penalized, 0 = not penalized.
    # A used vehicle is penalized only when its actual load is below its minimum load.
    # An unused vehicle is not penalized.
    p_lt: gp.tupledict[Tuple[int, int], gp.Var] = field(init=False)

    # Decision variables.
    # Whether vehicle l delivers goods from plant i to dealer j in period t
    # {(plant, dealer, v_id, day): gp.Var}; 1 = used, 0 = unused.
    y_ijlt: gp.tupledict[Tuple[str, str, int, int], gp.Var] = field(init=False)
    # Inventory of SKU k at plant i carried into period t, where t = 0, 1, ..., T-1.
    # In particular, s_ik0 is the initial inventory of SKU k at plant i.
    # {(plant, sku_id, day): gp.Var}
    s_ikt: gp.tupledict[Tuple[str, str, int], gp.Var] = field(init=False)
    # Quantity of SKU k loaded onto vehicle l for shipment from plant i to dealer j in period t
    # {(plant, dealer, sku_id, v_id, day): gp.Var}
    x_ijklt: gp.tupledict[Tuple[str, str, str, int, int], gp.Var] = field(init=False)

    # Variable index sets.
    auxi_indices: List[Tuple[int, int]] = field(init=False)
    y_indices: List[Tuple[str, str, int, int]] = field(init=False)
    s_indices: List[Tuple[str, str, int]] = field(init=False)
    x_indices: List[Tuple[str, str, str, int, int]] = field(init=False)
    new_triple_ikt: Set[Tuple[str, str, int]] = field(init=False)

    # Constraints added to the model.
    # Constraint 1: for every ordered pair (i, j) of plant i and dealer j,
    # if plant i uses vehicle l to deliver goods to dealer j in period t,
    # then vehicle l is considered used.
    cons_veh_use: gp.tupledict[Tuple[int, int], gp.Constr] = field(
        init=False
    )  # {(v_id, day): gp.Constr}
    # Constraint 2: in period t, goods can be loaded onto vehicle l only if it is used.
    cons_sku_load: gp.tupledict[Tuple[str, str, int, int], gp.Constr] = field(
        init=False
    )  # {(plant, dealer, v_id, day): gp.Constr}
    # Constraint 3: in period t, vehicle l's actual load must be at least its minimum load.
    cons_veh_min_load: gp.tupledict[Tuple[int, int], gp.Constr] = field(
        init=False
    )  # {(v_id, day): gp.Constr}
    # Constraint 4: relationship among the minimum-load penalty indicator p_lt,
    # the vehicle-use variable z_lt, and the minimum-load satisfaction variable zz_lt.
    cons_auxi_relation: gp.tupledict[Tuple[int, int], gp.Constr] = field(
        init=False
    )  # {(v_id, day): gp.Constr}
    # Constraint 5: in period t, vehicle l's actual load must not exceed its capacity.
    cons_veh_cap_limit: gp.tupledict[Tuple[int, int], gp.Constr] = field(
        init=False
    )  # {(v_id, day): gp.Constr}
    # Constraint 6: across all periods t, the total quantity of SKU k shipped to dealer j
    # from all plants i using all vehicles l must equal dealer j's demand for SKU k.
    cons_dealer_demands: gp.tupledict[Tuple[str, str], gp.Constr] = field(
        init=False
    )  # {(dealer, sku_id): gp.Constr}
    # Constraint 7: in period t, the total quantity of SKU k shipped from plant i to all
    # dealers j equals the inventory carried from period t-1 plus production in period t
    # minus the inventory carried into period t.
    # {(plant, sku_id, day): gp.Constr}
    cons_sku_inv_transfer: gp.tupledict[Tuple[str, str, int], gp.Constr] = field(
        init=False
    )
    # Constraint 8: in period t, the total remaining inventory at plant i must not
    # exceed the plant's maximum storage capacity.
    cons_plant_inv_limit: gp.tupledict[Tuple[str, int], gp.Constr] = field(
        init=False
    )  # {(plant, day): gp.Constr}

    # Constraint 9: add symmetry-breaking cuts to reduce model symmetry and solution space.
    # {(v_type, v_id, v_di_1, day): gp.Constr}
    # cons_symmetry_break: gp.tupledict[Tuple[str, int, int, int], gp.Constr] = field(init=False)

    # All attributes of the multi-period optimization model are defined.
    # The next step is to instantiate them.

    def __post_init__(self):
        self.horizons = self.data.horizons

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
        self.sku_prod_each_day = self.data.sku_prod_each_day
        self.sku_initial_inv = self.data.sku_initial_inv
        self.skus_plant = self.data.skus_plant
        self.demands = self.data.demands
        self.skus_dealer = self.data.skus_dealer
        self.supply_chain = self.data.supply_chain
        self.veh_cost = self.data.veh_cost
        self.veh_cap = self.data.veh_cap
        self.veh_min_load = self.data.veh_min_load
        self.plant_inv_limit = self.data.plant_inv_limit
        self.all_veh_types = self.data.all_veh_types
        self.veh_type = self.data.veh_type
        self.veh_ids = self.data.veh_ids

        self.model = gp.Model()
        self.obj = gp.LinExpr()

        # Add auxiliary variables.
        self.periods = {t for t in range(1, self.horizons + 1)}
        self.auxi_indices = [
            (v_id, t) for v_id in self.all_vehicles for t in self.periods
        ]
        self.z_lt = self.model.addVars(
            self.auxi_indices, lb=0, ub=1, vtype=GRB.BINARY, name="z_lt"
        )
        self.zz_lt = self.model.addVars(
            self.auxi_indices, lb=0, ub=1, vtype=GRB.BINARY, name="zz_lt"
        )
        self.p_lt = self.model.addVars(
            self.auxi_indices, lb=0, ub=1, vtype=GRB.BINARY, name="p_lt"
        )

        # Add decision variables.
        self.y_indices = [
            (plant, dealer, v_id, t)
            for (plant, dealer) in self.supply_chain
            for (v_id, t) in self.auxi_indices
        ]
        self.y_ijlt = self.model.addVars(
            self.y_indices, lb=0, ub=1, vtype=GRB.BINARY, name="y"
        )

        # Build the set of (plant, dealer, sku_id) triples where plant i can supply SKU k to dealer j.
        # Fix a dealer j.
        # A set is used to guarantee unique variable indices by removing duplicates.
        # First derive the SKUs available from each plant, including inventory and production,
        # from skus_plant. The source data is stored as a dictionary, so convert it to the required form.
        duet_ik = {
            (plant, sku_id)
            for plant, skus in self.skus_plant.items()
            for sku_id in skus
        }
        triple_ijk = {
            (plant, dealer1, sku_id)
            for (plant, dealer1) in self.supply_chain
            for (dealer2, sku_id) in self.demands
            if dealer1 == dealer2
            if (plant, sku_id) in duet_ik
        }
        self.new_triple_ikt = {
            (plant, sku_id, t)
            for (plant, dealer, sku_id) in triple_ijk
            for t in self.periods
        }

        self.x_indices = [
            (plant, dealer, sku_id, v_id, t)
            for (plant, dealer, sku_id) in triple_ijk
            for (v_id, t) in self.auxi_indices
        ]
        self.x_ijklt = self.model.addVars(
            self.x_indices, lb=0, vtype=GRB.INTEGER, name="x"
        )

        self.s_indices = [
            (plant, sku_id, t)
            for (plant, sku_id) in duet_ik
            for t in range(self.horizons + 1)
        ]
        self.s_ikt = self.model.addVars(
            self.s_indices, lb=0, vtype=GRB.INTEGER, name="s"
        )
        # Since s_ik0 represents the initial inventory of SKU k at plant i, fix this variable.
        # Set both its lower and upper bounds to the same constant value.
        for plant, sku_id, t in self.s_indices:
            if t == 0 and (plant, sku_id) in self.sku_initial_inv:
                self.s_ikt[plant, sku_id, t].setAttr(
                    "LB", self.sku_initial_inv[plant, sku_id]
                )
                self.s_ikt[plant, sku_id, t].setAttr(
                    "UB", self.sku_initial_inv[plant, sku_id]
                )

        # Build the objective function.
        self.obj = gp.quicksum(
            self.veh_cost[v_id] * self.z_lt[v_id, t] for (v_id, t) in self.auxi_indices
        ) + gp.quicksum(
            self.param_alpha * self.veh_cost[v_id] * self.p_lt[v_id, t]
            for (v_id, t) in self.auxi_indices
        )
        # Set the optimization direction of the objective function.
        self.model.setObjective(self.obj, sense=GRB.MINIMIZE)

        # Add constraints.
        # Constraint 1: for every ordered pair (i, j) of plant i and dealer j,
        # if plant i uses vehicle l to deliver goods to dealer j in period t,
        # then vehicle l is considered used.
        # Create cons_veh_use with explicit unique names to avoid duplicate-name warnings
        _cons_veh_use_tmp = {}
        for v_id, t in self.auxi_indices:
            cname = f"cons_veh_use_{v_id}_{t}"
            _cons_veh_use_tmp[v_id, t] = self.model.addConstr(
                self.y_ijlt.sum("*", "*", v_id, t) == self.z_lt[v_id, t], name=cname
            )
        # convert to Gurobi tupledict for compatibility with later code
        self.cons_veh_use = gp.tupledict(_cons_veh_use_tmp)

        def _sanitize_part(p):
            return (
                str(p)
                .replace(" ", "_")
                .replace(":", "_")
                .replace(".", "_")
                .replace("-", "_")
            )

        # Constraint 2: in period t, goods can be loaded onto vehicle l only if it is used.
        _cons_sku_load_tmp = {}
        for plant, dealer, v_id, t in self.y_indices:
            cname = f"cons_sku_load_{_sanitize_part(plant)}_{_sanitize_part(dealer)}_{v_id}_{t}"
            _cons_sku_load_tmp[plant, dealer, v_id, t] = self.model.addConstr(
                self.x_ijklt.sum(plant, dealer, "*", v_id, t)
                <= self.param_BIG_NUM * self.y_ijlt[plant, dealer, v_id, t],
                name=cname,
            )
        self.cons_sku_load = gp.tupledict(_cons_sku_load_tmp)

        # Constraint 3: in period t, vehicle l's actual load must be at least its minimum load.
        _cons_veh_min_load_tmp = {}
        for v_id, t in self.auxi_indices:
            cname = f"cons_veh_min_load_{v_id}_{t}"
            _cons_veh_min_load_tmp[v_id, t] = self.model.addConstr(
                gp.quicksum(
                    self.x_ijklt.sum("*", "*", sku_id, v_id, t) * self.sku_sizes[sku_id]
                    for sku_id in self.all_skus
                )
                + self.zz_lt[v_id, t] * self.param_BIG_NUM
                >= self.veh_min_load[v_id],
                name=cname,
            )
        self.cons_veh_min_load = gp.tupledict(_cons_veh_min_load_tmp)

        # Constraint 4: relationship among the minimum-load penalty indicator p_lt,
        # the vehicle-use variable z_lt, and the minimum-load satisfaction variable zz_lt.
        _cons_auxi_relation_tmp = {}
        for v_id, t in self.auxi_indices:
            cname = f"cons_auxi_relation_{v_id}_{t}"
            _cons_auxi_relation_tmp[v_id, t] = self.model.addConstr(
                self.z_lt[v_id, t] + self.zz_lt[v_id, t] - 1 <= self.p_lt[v_id, t],
                name=cname,
            )
        self.cons_auxi_relation = gp.tupledict(_cons_auxi_relation_tmp)

        # Constraint 5: in period t, vehicle l's actual load must not exceed its capacity.
        _cons_veh_cap_limit_tmp = {}
        for v_id, t in self.auxi_indices:
            cname = f"cons_veh_cap_limit_{v_id}_{t}"
            _cons_veh_cap_limit_tmp[v_id, t] = self.model.addConstr(
                gp.quicksum(
                    self.x_ijklt.sum("*", "*", sku_id, v_id, t) * self.sku_sizes[sku_id]
                    for sku_id in self.all_skus
                )
                <= self.veh_cap[v_id],
                name=cname,
            )
        self.cons_veh_cap_limit = gp.tupledict(_cons_veh_cap_limit_tmp)

        # Constraint 6: across all periods t, the total quantity of SKU k shipped to dealer j
        # from all plants i using all vehicles l must equal dealer j's demand for SKU k.
        _cons_dealer_demands_tmp = {}
        for dealer, sku_id in self.demands:
            cname = (
                f"cons_dealer_demands_{_sanitize_part(dealer)}_{_sanitize_part(sku_id)}"
            )
            _cons_dealer_demands_tmp[dealer, sku_id] = self.model.addConstr(
                self.x_ijklt.sum("*", dealer, sku_id, "*", "*")
                >= self.demands[dealer, sku_id],
                name=cname,
            )
        self.cons_dealer_demands = gp.tupledict(_cons_dealer_demands_tmp)

        # Constraint 7: in period t, the total quantity of SKU k shipped from plant i to all
        # dealers j equals the inventory carried from period t-1 plus production in period t
        # minus the inventory carried into period t.
        _cons_sku_inv_transfer_tmp = {}
        for plant, sku_id, t in self.new_triple_ikt:
            cname = f"cons_sku_inv_transfer_{_sanitize_part(plant)}_{_sanitize_part(sku_id)}_{t}"
            _cons_sku_inv_transfer_tmp[plant, sku_id, t] = self.model.addConstr(
                self.x_ijklt.sum(plant, "*", sku_id, "*", t)
                == self.sku_prod_each_day.get((plant, sku_id, t), 0)
                + self.s_ikt[plant, sku_id, t - 1]
                - self.s_ikt[plant, sku_id, t],
                name=cname,
            )
        self.cons_sku_inv_transfer = gp.tupledict(_cons_sku_inv_transfer_tmp)

        # Constraint 8: in period t, the total remaining inventory at plant i must not
        # exceed the plant's maximum storage capacity.
        _cons_plant_inv_limit_tmp = {}
        for plant in self.plants:
            for t in self.periods:
                cname = f"cons_plant_inv_limit_{_sanitize_part(plant)}_{t}"
                _cons_plant_inv_limit_tmp[plant, t] = self.model.addConstr(
                    self.s_ikt.sum(plant, "*", t) <= self.plant_inv_limit[plant],
                    name=cname,
                )
        self.cons_plant_inv_limit = gp.tupledict(_cons_plant_inv_limit_tmp)

        # Constraint 9: add symmetry-breaking cuts to reduce model symmetry.
        # Gurobi detects and handles symmetry automatically, so these cuts are optional.
        # If added, evaluate their effect on solver efficiency.
        # self.cons_symmetry_break = self.model.addConstrs((
        #     self.z_lt[v_id, t] <= self.z_lt[v_id_1, t]
        #     for v_type in self.all_veh_types
        #     for v_id, v_id_1 in zip(list(self.veh_ids[v_type]), list(self.veh_ids[v_type])[1:])
        #     for t in self.periods
        # ), name="cons_symmetry_break")

        # Enable solver logs.
        self.model.setParam("OutputFlag", self.param_output_flag)
        # Set the MIP optimality gap.
        self.model.Params.MIPGap = self.param_gap_limit
        # Set the solver time limit.
        self.model.Params.TimeLimit = self.param_time_limit

    def run(self):
        # If BestBd improves slowly or not at all during optimization, try the following
        # parameter settings. For algorithm comparison experiments, it is generally better
        # to set only TimeLimit and MIPGap.
        # self.model.Params.MIPFocus = 3    # Focus on improving the bound.
        # self.model.Params.MIPFocus = 2    # Focus on proving optimality.
        # self.model.Params.Cuts = 2        # Generate more cutting planes.

        # Collect solver status information.
        meta = {}

        # Helper to read variable value (compatible with .X / .Xn)
        def _var_value(v):
            try:
                return float(v.X)
            except Exception:
                try:
                    return float(v.Xn)
                except Exception:
                    try:
                        return float(v)
                    except Exception:
                        return 0.0

        # Export current/best solution to JSON for offline analysis / mapping.
        # value_getter defaults to model solution values; callback path can pass cbGetSolution getter.
        def _export_solution_json(status_tag: str, value_getter=None, obj_value=None):
            try:
                sol_file_loc = os.path.join(
                    self.data.output_file_loc, self.data.dataset_name
                )
                if not os.path.exists(sol_file_loc):
                    os.makedirs(sol_file_loc)

                sol_file_name = (
                    f"Feasible-Solution-{status_tag}-{int(time.time())}.json"
                )
                sol_file = os.path.join(sol_file_loc, sol_file_name)

                sol = {
                    "z": {},
                    "y": {},
                    "x": {},
                    "s": {},
                    "veh_type_by_id": {},
                    "gurobi_obj": None,
                }

                # veh_type_by_id mapping
                try:
                    for vid, vtype in self.veh_type.items():
                        sol["veh_type_by_id"][str(vid)] = str(vtype)
                except Exception:
                    pass

                # objective value if available
                try:
                    if obj_value is not None:
                        sol["gurobi_obj"] = float(obj_value)
                    else:
                        sol["gurobi_obj"] = float(self.model.ObjVal)
                except Exception:
                    # if no ObjVal yet, keep None
                    sol["gurobi_obj"] = None

                get_value = value_getter if callable(value_getter) else _var_value

                # z variables
                try:
                    for k, var in self.z_lt.items():
                        v_id, t = k
                        sol["z"][f"{v_id}|{t}"] = get_value(var)
                except Exception:
                    pass

                # y variables
                try:
                    for k, var in self.y_ijlt.items():
                        plant, dealer, v_id, t = k
                        sol["y"][f"{plant}|{dealer}|{v_id}|{t}"] = get_value(var)
                except Exception:
                    pass

                # x variables
                try:
                    for k, var in self.x_ijklt.items():
                        plant, dealer, sku, v_id, t = k
                        sol["x"][f"{plant}|{dealer}|{sku}|{v_id}|{t}"] = int(
                            round(get_value(var))
                        )
                except Exception:
                    pass

                # s variables
                try:
                    for k, var in self.s_ikt.items():
                        plant, sku, t = k
                        sol["s"][f"{plant}|{sku}|{t}"] = int(round(get_value(var)))
                except Exception:
                    pass

                # write json
                with open(sol_file, "w", encoding="utf-8") as f:
                    import json as _json

                    _json.dump(sol, f, indent=2, ensure_ascii=False)
                log_printer.print(f"Feasible solution exported to {sol_file}")
            except Exception as e:
                log_printer.print(f"Failed to export feasible solution JSON: {e}")

        self.model.optimize()
        self.model.write("MonolithicModelMultiple.lp")

        if self.model.Status in (GRB.INF_OR_UNBD, GRB.INFEASIBLE, GRB.UNBOUNDED):
            self.model.computeIIS()
            self.model.write("Infeasible_MonolithicModelMultiple.ilp")

            if (
                self.model.IISMinimal
            ):  # Check whether computeIIS() returned a minimal irreducible infeasible subsystem.
                log_printer.print("IIS is minimal \n")
            else:
                log_printer.print("IIS is not minimal \n")
            log_printer.print("\n The following constraint (s) cannot be satisfied :")
            for c in self.model.getConstrs():
                if c.IISConstr:
                    log_printer.print("%s" % c.ConstrName)

            if self.model.Status == GRB.INFEASIBLE:
                meta["status"] = "infeasible"
            elif self.model.Status == GRB.UNBOUNDED:
                meta["status"] = "unbounded"
            elif self.model.Status == GRB.INF_OR_UNBD:
                meta["status"] = "inf_or_unbd"

            # Attempt to export any available solution (unlikely), tag as infeasible
            _export_solution_json("infeasible")
            return False, meta

        elif self.model.Status == GRB.TIME_LIMIT:
            log_printer.print("Time limit is reached! Solving process is stopped.")
            # Export the incumbent solution if present
            _export_solution_json("time_limit")
            try:
                self.model.terminate()
            except Exception:
                pass
            meta["status"] = "time_limit"
            return False, meta

        elif self.model.Status == GRB.INTERRUPTED:
            log_printer.print("Optimization is interrupted by user!")
            # Export current solution if any
            _export_solution_json("interrupted")
            meta["status"] = "interrupted"
            return False, meta
        else:  # model is optimal
            self.print_sol_summary()
            # export final optimal solution
            _export_solution_json("optimal")
            meta["status"] = "optimal"
            return True, meta

    def save_feasible_sol(self):
        """
        Save the feasible solution found by Gurobi to a JSON file.
        This function is called when the solver status is TIME_LIMIT, but it can be
        very time-consuming and is not recommended for batch runs.
        """
        sol_file_loc = os.path.join(self.data.output_file_loc, self.data.dataset_name)
        if not os.path.exists(sol_file_loc):
            os.makedirs(sol_file_loc)
        sol_file_name = "Feasible-Solution.json"
        sol_file = os.path.join(sol_file_loc, sol_file_name)
        self.model.write(sol_file)

    def print_sol_summary(self):
        if self.model.Status == GRB.OPTIMAL:
            log_printer.print(
                "Monolithic Model for multiple periods is solved to be optimal."
            )
            log_printer.print(
                f"Total cost of optimal solution is {self.model.ObjVal:.2f}"
            )

            total_veh_num = sum(var.Xn for var in self.z_lt.values())
            log_printer.print(f"Total number of vehicles used is {total_veh_num}")

            veh_punished_num = sum(var.Xn for var in self.p_lt.values())
            log_printer.print(
                f"Among used vehicles, there are {veh_punished_num} vehicle(s) "
                f"that don't satisfy veh_min_load constraint and need to be punished."
            )

            total_ship_num = sum(var.Xn for var in self.x_ijklt.values())
            log_printer.print(f"Total SKU ship amount is {total_ship_num}")
