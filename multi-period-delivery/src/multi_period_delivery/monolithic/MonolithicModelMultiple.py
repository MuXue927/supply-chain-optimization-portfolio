# Gurobi formulation of the multi-period mathematical model.
# This module implements the model defined as Model 2.
# Model 2 assumes multiple periods, SKU supply = initial inventory + production >=
# demand, and a finite storage capacity at each production plant.
# It supports multiple plants, dealers, vehicle types, and SKUs.
# Each plant can supply the same SKUs, and different plants' SKU sets may overlap.
import os
import time
from dataclasses import dataclass, field

import gurobipy as gp
from gurobipy import GRB

from .InputDataMultiple import InputDataMultiple
from .optutility import LogPrinter

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
    plants: set[str] = field(init=False)
    # Set of dealers.
    dealers: set[str] = field(init=False)
    # Set of all SKUs.
    all_skus: set[str] = field(init=False)
    # Set of all vehicle IDs.
    all_vehicles: set[int] = field(init=False)
    # Set of planning periods t.
    periods: set[int] = field(init=False)

    # Volume per SKU {sku: size}.
    sku_sizes: dict[str, float] = field(init=False)
    # Planned SKU production at each plant in period t {(plant, sku, day): qty}.
    sku_prod_each_day: dict[tuple[str, str, int], int] = field(init=False)
    # Initial SKU inventory at each plant {(plant, sku): qty}.
    sku_initial_inv: dict[tuple[str, str], int] = field(init=False)
    # SKUs that each plant can supply.
    skus_plant: dict[str, set[str]] = field(init=False)

    # Dealer demand for each SKU {(dealer, sku): qty}.
    demands: dict[tuple[str, str], int] = field(init=False)
    # SKUs required by each dealer.
    skus_dealer: dict[str, set[str]] = field(init=False)

    # Use cost for each vehicle {v_id: cost}.
    veh_cost: dict[int, float] = field(init=False)
    # Capacity upper bound for each vehicle {v_id: capacity}.
    veh_cap: dict[int, float] = field(init=False)
    # Minimum load for each vehicle {v_id: min_load}.
    veh_min_load: dict[int, float] = field(init=False)
    # Inventory capacity upper bound for each plant {plant: max_cap}.
    plant_inv_limit: dict[str, float] = field(init=False)

    # All vehicle types.
    all_veh_types: list[str] = field(init=False)
    # Vehicle IDs must be linked to vehicle types.
    # Given a vehicle ID, its type identifies all relevant vehicle attributes,
    # including use cost, capacity, and minimum load.
    # {v_id: v_type}
    veh_type: dict[int, str] = field(init=False)
    # Conversely, each vehicle type identifies its vehicle IDs {v_type: {v_id}}.
    veh_ids: dict[str, set[int]] = field(init=False)
    # Vehicle-type attributes used by the route-local formulation.
    veh_type_cap: dict[str, int] = field(init=False)
    veh_type_min_load: dict[str, int] = field(init=False)
    veh_type_cost: dict[str, float] = field(init=False)

    # Supply relationships between plants and dealers {(plant, dealer): {sku}}.
    supply_chain: dict[tuple[str, str], set[str]] = field(init=False)
    # Route-local slots: {(plant, dealer): [slot, ...]}.
    route_slots: dict[tuple[str, str], list[int]] = field(init=False)
    # All data interfaces required by the multi-period optimization problem are defined.

    # Next, define the mathematical model, objective function, auxiliary variables,
    # decision variables, and finally the constraints.
    # Gurobi model.
    model: gp.Model = field(init=False)
    # Objective function.
    obj: gp.LinExpr = field(init=False)

    # Auxiliary variables, for every route (i, j), slot n, type f, period t:
    # z_ijfn_t  - slot n of route (i, j) uses type f in period t
    # zz_ijn_t  - slot n is allowed to fall below its minimum load in period t
    # p_ijfn_t  - slot n is used with type f and below its minimum load
    z_ijfn_t: gp.tupledict[tuple[str, str, str, int, int], gp.Var] = field(
        init=False
    )
    zz_ijn_t: gp.tupledict[tuple[str, str, int, int], gp.Var] = field(init=False)
    p_ijfn_t: gp.tupledict[tuple[str, str, str, int, int], gp.Var] = field(
        init=False
    )

    # Decision variables.
    # Inventory of SKU k at plant i carried into period t, where t = 0, 1, ..., T-1.
    # In particular, s_ik0 is the initial inventory of SKU k at plant i.
    # {(plant, sku_id, day): gp.Var}
    s_ikt: gp.tupledict[tuple[str, str, int], gp.Var] = field(init=False)
    # Quantity of SKU k loaded onto slot n of route (i, j) in period t:
    # {(plant, dealer, sku_id, slot, day): gp.Var}
    x_ijkn_t: gp.tupledict[tuple[str, str, str, int, int], gp.Var] = field(
        init=False
    )

    # Variable index sets.
    zz_indices: list[tuple[str, str, int, int]] = field(init=False)
    z_indices: list[tuple[str, str, str, int, int]] = field(init=False)
    s_indices: list[tuple[str, str, int]] = field(init=False)
    x_indices: list[tuple[str, str, str, int, int]] = field(init=False)
    new_triple_ikt: set[tuple[str, str, int]] = field(init=False)

    # Constraints added to the model.
    # Constraint 1: at most one vehicle type is assigned to each route slot.
    cons_one_type: gp.tupledict[tuple[str, str, int, int], gp.Constr] = field(
        init=False
    )
    # Constraint 2: in period t, a slot's load must not exceed its type capacity.
    cons_veh_cap_limit: gp.tupledict[tuple[str, str, int, int], gp.Constr] = field(
        init=False
    )
    # Constraint 3: in period t, a slot's actual load must be at least its
    # selected type's minimum load unless the zz slack is activated.
    cons_veh_min_load: gp.tupledict[tuple[str, str, int, int], gp.Constr] = field(
        init=False
    )
    # Constraint 4: relationship among the minimum-load penalty indicator p,
    # the vehicle-use variable z, and the minimum-load satisfaction variable zz.
    cons_auxi_relation: gp.tupledict[
        tuple[str, str, str, int, int], gp.Constr
    ] = field(init=False)
    # Constraint 5: across all periods t, the total quantity of SKU k shipped to dealer j
    # from all plants i using all vehicles l must equal dealer j's demand for SKU k.
    cons_dealer_demands: gp.tupledict[tuple[str, str], gp.Constr] = field(
        init=False
    )  # {(dealer, sku_id): gp.Constr}
    # Constraint 6: in period t, the total quantity of SKU k shipped from plant i to all
    # dealers j equals the inventory carried from period t-1 plus production in period t
    # minus the inventory carried into period t.
    # {(plant, sku_id, day): gp.Constr}
    cons_sku_inv_transfer: gp.tupledict[tuple[str, str, int], gp.Constr] = field(
        init=False
    )
    # Constraint 7: in period t, the total remaining inventory at plant i must not
    # exceed the plant's maximum storage capacity.
    cons_plant_inv_limit: gp.tupledict[tuple[str, int], gp.Constr] = field(
        init=False
    )  # {(plant, day): gp.Constr}

    # Constraint 8: add symmetry-breaking cuts to reduce model symmetry and solution space.
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
        self.veh_type_cap = self.data.veh_type_cap
        self.veh_type_min_load = self.data.veh_type_min_load
        self.veh_type_cost = self.data.veh_type_cost
        self.route_slots = self.data.route_slots

        self.model = gp.Model()
        self.obj = gp.LinExpr()

        # Add route-local variables. A slot belongs to one route; its type is
        # selected independently in each period, so no global fleet index or
        # y assignment variable is required.
        self.periods = {t for t in range(1, self.horizons + 1)}
        self.zz_indices = [
            (plant, dealer, slot, t)
            for (plant, dealer), slots in self.route_slots.items()
            for slot in slots
            for t in self.periods
        ]
        self.z_indices = [
            (plant, dealer, v_type, slot, t)
            for (plant, dealer), slots in self.route_slots.items()
            for slot in slots
            for v_type in self.all_veh_types
            for t in self.periods
        ]
        self.x_indices = [
            (plant, dealer, sku_id, slot, t)
            for (plant, dealer), slots in self.route_slots.items()
            for slot in slots
            for sku_id in self.supply_chain[plant, dealer]
            for t in self.periods
        ]

        self.zz_ijn_t = self.model.addVars(
            self.zz_indices, lb=0, ub=1, vtype=GRB.BINARY, name="zz"
        )
        self.z_ijfn_t = self.model.addVars(
            self.z_indices, lb=0, ub=1, vtype=GRB.BINARY, name="z"
        )
        self.p_ijfn_t = self.model.addVars(
            self.z_indices, lb=0, ub=1, vtype=GRB.BINARY, name="p"
        )
        self.x_ijkn_t = self.model.addVars(
            self.x_indices, lb=0, vtype=GRB.INTEGER, name="x"
        )

        # Inventory is carried per plant, SKU, and period.
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

        self.s_indices = [
            (plant, sku_id, t)
            for (plant, sku_id) in duet_ik
            for t in range(self.horizons + 1)
        ]
        self.s_ikt = self.model.addVars(
            self.s_indices, lb=0, vtype=GRB.INTEGER, name="s"
        )
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
            self.veh_type_cost[v_type] * self.z_ijfn_t[plant, dealer, v_type, slot, t]
            for plant, dealer, v_type, slot, t in self.z_indices
        ) + gp.quicksum(
            self.param_alpha
            * self.veh_type_cost[v_type]
            * self.p_ijfn_t[plant, dealer, v_type, slot, t]
            for plant, dealer, v_type, slot, t in self.z_indices
        )
        self.model.setObjective(self.obj, sense=GRB.MINIMIZE)

        def _sanitize_part(p):
            return (
                str(p)
                .replace(" ", "_")
                .replace(":", "_")
                .replace(".", "_")
                .replace("-", "_")
            )

        # Constraint 1: at most one vehicle type per route slot and period.
        _cons_one_type_tmp = {}
        for plant, dealer, slot, t in self.zz_indices:
            cname = (
                f"cons_one_type_{_sanitize_part(plant)}_{_sanitize_part(dealer)}_"
                f"{slot}_{t}"
            )
            _cons_one_type_tmp[plant, dealer, slot, t] = self.model.addConstr(
                gp.quicksum(
                    self.z_ijfn_t[plant, dealer, v_type, slot, t]
                    for v_type in self.all_veh_types
                )
                <= 1,
                name=cname,
            )
        self.cons_one_type = gp.tupledict(_cons_one_type_tmp)

        # Constraint 2: load cannot exceed the capacity of the selected type.
        _cons_veh_cap_limit_tmp = {}
        for plant, dealer, slot, t in self.zz_indices:
            cname = (
                f"cons_veh_cap_limit_{_sanitize_part(plant)}_"
                f"{_sanitize_part(dealer)}_{slot}_{t}"
            )
            _cons_veh_cap_limit_tmp[plant, dealer, slot, t] = self.model.addConstr(
                gp.quicksum(
                    self.x_ijkn_t[plant, dealer, sku_id, slot, t]
                    * self.sku_sizes[sku_id]
                    for sku_id in self.supply_chain[plant, dealer]
                )
                - gp.quicksum(
                    self.veh_type_cap[v_type]
                    * self.z_ijfn_t[plant, dealer, v_type, slot, t]
                    for v_type in self.all_veh_types
                )
                <= 0,
                name=cname,
            )
        self.cons_veh_cap_limit = gp.tupledict(_cons_veh_cap_limit_tmp)

        # Constraint 3: minimum load, with a binary slack for penalized slots.
        _cons_veh_min_load_tmp = {}
        for plant, dealer, slot, t in self.zz_indices:
            cname = (
                f"cons_veh_min_load_{_sanitize_part(plant)}_"
                f"{_sanitize_part(dealer)}_{slot}_{t}"
            )
            _cons_veh_min_load_tmp[plant, dealer, slot, t] = self.model.addConstr(
                gp.quicksum(
                    self.x_ijkn_t[plant, dealer, sku_id, slot, t]
                    * self.sku_sizes[sku_id]
                    for sku_id in self.supply_chain[plant, dealer]
                )
                + self.zz_ijn_t[plant, dealer, slot, t] * self.param_BIG_NUM
                - gp.quicksum(
                    self.veh_type_min_load[v_type]
                    * self.z_ijfn_t[plant, dealer, v_type, slot, t]
                    for v_type in self.all_veh_types
                )
                >= 0,
                name=cname,
            )
        self.cons_veh_min_load = gp.tupledict(_cons_veh_min_load_tmp)

        # Constraint 4: z + zz - 1 <= p for every possible type.
        _cons_auxi_relation_tmp = {}
        for plant, dealer, v_type, slot, t in self.z_indices:
            cname = (
                f"cons_auxi_relation_{_sanitize_part(plant)}_"
                f"{_sanitize_part(dealer)}_{_sanitize_part(v_type)}_{slot}_{t}"
            )
            _cons_auxi_relation_tmp[plant, dealer, v_type, slot, t] = (
                self.model.addConstr(
                    self.z_ijfn_t[plant, dealer, v_type, slot, t]
                    + self.zz_ijn_t[plant, dealer, slot, t]
                    - 1
                    <= self.p_ijfn_t[plant, dealer, v_type, slot, t],
                    name=cname,
                )
            )
        self.cons_auxi_relation = gp.tupledict(_cons_auxi_relation_tmp)

        # Constraint 5: aggregate demand across all periods and route slots.
        _cons_dealer_demands_tmp = {}
        for dealer, sku_id in self.demands:
            cname = (
                f"cons_dealer_demands_{_sanitize_part(dealer)}_{_sanitize_part(sku_id)}"
            )
            _cons_dealer_demands_tmp[dealer, sku_id] = self.model.addConstr(
                gp.quicksum(
                    self.x_ijkn_t[plant, dealer, sku_id, slot, t]
                    for plant in self.plants
                    for slot in self.route_slots.get((plant, dealer), [])
                    for t in self.periods
                    if (plant, dealer, sku_id, slot, t) in self.x_ijkn_t
                )
                >= self.demands[dealer, sku_id],
                name=cname,
            )
        self.cons_dealer_demands = gp.tupledict(_cons_dealer_demands_tmp)

        # Constraint 6: inventory transfer.
        _cons_sku_inv_transfer_tmp = {}
        for plant, sku_id, t in self.new_triple_ikt:
            cname = f"cons_sku_inv_transfer_{_sanitize_part(plant)}_{_sanitize_part(sku_id)}_{t}"
            _cons_sku_inv_transfer_tmp[plant, sku_id, t] = self.model.addConstr(
                gp.quicksum(
                    self.x_ijkn_t[plant, dealer, sku_id, slot, t]
                    for dealer in self.dealers
                    for slot in self.route_slots.get((plant, dealer), [])
                    if (plant, dealer, sku_id, slot, t) in self.x_ijkn_t
                )
                == self.sku_prod_each_day.get((plant, sku_id, t), 0)
                + self.s_ikt[plant, sku_id, t - 1]
                - self.s_ikt[plant, sku_id, t],
                name=cname,
            )
        self.cons_sku_inv_transfer = gp.tupledict(_cons_sku_inv_transfer_tmp)

        # Constraint 7: plant storage capacity.
        _cons_plant_inv_limit_tmp = {}
        for plant in self.plants:
            for t in self.periods:
                cname = f"cons_plant_inv_limit_{_sanitize_part(plant)}_{t}"
                _cons_plant_inv_limit_tmp[plant, t] = self.model.addConstr(
                    self.s_ikt.sum(plant, "*", t) <= self.plant_inv_limit[plant],
                    name=cname,
                )
        self.cons_plant_inv_limit = gp.tupledict(_cons_plant_inv_limit_tmp)

        # Enable solver logs.
        self.model.setParam("OutputFlag", self.param_output_flag)
        # Set the MIP optimality gap.
        self.model.Params.MIPGap = self.param_gap_limit
        # Set the solver time limit.
        self.model.Params.TimeLimit = self.param_time_limit

    def run(self):
        # The Gurobi settings that helped most during development when the best
        # bound improved slowly were MIPFocus = 3 (focus on improving the
        # bound), MIPFocus = 2 (focus on proving optimality), and Cuts = 2
        # (generate more cutting planes). The reported comparisons set only
        # TimeLimit and MIPGap so that the runs stay comparable.

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
                    except (TypeError, ValueError):
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
                    "x": {},
                    "s": {},
                    "gurobi_obj": None,
                }

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
                    for k, var in self.z_ijfn_t.items():
                        plant, dealer, v_type, slot, t = k
                        sol["z"][
                            f"{plant}|{dealer}|{v_type}|{slot}|{t}"
                        ] = get_value(var)
                except Exception:
                    pass

                # x variables
                try:
                    for k, var in self.x_ijkn_t.items():
                        plant, dealer, sku, slot, t = k
                        sol["x"][f"{plant}|{dealer}|{sku}|{slot}|{t}"] = int(
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
        # Dump the LP into this run's output directory rather than the current
        # working directory, so generated model files never land next to the
        # source (see .gitignore: *.lp / *.ilp).
        os.makedirs(self.data.output_file_loc, exist_ok=True)
        self.model.write(
            os.path.join(self.data.output_file_loc, "MonolithicModelMultiple.lp")
        )

        if self.model.Status in (GRB.INF_OR_UNBD, GRB.INFEASIBLE, GRB.UNBOUNDED):
            self.model.computeIIS()
            self.model.write(
                os.path.join(
                    self.data.output_file_loc, "Infeasible_MonolithicModelMultiple.ilp"
                )
            )

            if (
                self.model.IISMinimal
            ):  # Check whether computeIIS() returned a minimal irreducible infeasible subsystem.
                log_printer.print("IIS is minimal \n")
            else:
                log_printer.print("IIS is not minimal \n")
            log_printer.print("\n The following constraint (s) cannot be satisfied :")
            for c in self.model.getConstrs():
                if c.IISConstr:
                    log_printer.print(f"{c.ConstrName}")

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

    def slot_type(self, plant: str, dealer: str, slot: int, day: int) -> str:
        """Return the vehicle type selected by a route slot in one period."""
        best_type = self.all_veh_types[0]
        best_value = 0.0
        for v_type in self.all_veh_types:
            value = self.z_ijfn_t[plant, dealer, v_type, slot, day].Xn
            if value > best_value:
                best_value = value
                best_type = v_type
        return best_type

    def print_sol_summary(self):
        if self.model.Status == GRB.OPTIMAL:
            log_printer.print(
                "Monolithic Model for multiple periods is solved to be optimal."
            )
            log_printer.print(
                f"Total cost of optimal solution is {self.model.ObjVal:.2f}"
            )

            total_veh_num = sum(var.Xn for var in self.z_ijfn_t.values())
            log_printer.print(f"Total number of vehicles used is {total_veh_num}")

            veh_punished_num = sum(var.Xn for var in self.p_ijfn_t.values())
            log_printer.print(
                f"Among used vehicles, there are {veh_punished_num} vehicle(s) "
                f"that don't satisfy veh_min_load constraint and need to be punished."
            )

            total_ship_num = sum(var.Xn for var in self.x_ijkn_t.values())
            log_printer.print(f"Total SKU ship amount is {total_ship_num}")
