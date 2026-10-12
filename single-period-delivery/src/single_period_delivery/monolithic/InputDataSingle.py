# Data interface for the single-period optimization problem.
# Loads and normalizes the data stored in datasets/single-period.
# Exposes each dataset in the representation required by the optimization model.
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class InputDataSingle:
    input_file_loc: str
    output_file_loc: str
    # specify which dataset to load
    dataset_name: str

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
    # SKUs that can be supplied by each production plant.
    skus_plant: dict[str, set[str]] = field(init=False)
    # Total inventory of each SKU: {sku: total_qty}.
    sku_total_inv: dict[str, int] = field(init=False)

    # Dealer demand for each SKU: {(dealer, sku): qty}.
    demands: dict[tuple[str, str], int] = field(init=False)
    # SKUs required by each dealer.
    skus_dealer: dict[str, set[str]] = field(init=False)

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

    supply_chain: dict[tuple[str, str], set[str]] = field(init=False)
    # # Maximum number of available vehicles for each type.
    # MAX_VEH_NUM: int = field(init=False)
    # Upper bound on the number of vehicles that can be used.
    max_veh_num: int = field(init=False)

    # All data interfaces required by the single-period optimization problem are now declared.
    # The next step is to load and normalize the input data.

    # Names with a double leading underscore support data encapsulation by marking attributes or methods as private.
    # Python name mangling rewrites such names to "_ClassName__name"; this is an implementation convention rather than strict privacy.
    # It also prevents subclasses from accidentally overriding or changing the method implementation.
    # This helper renames DataFrame columns according to the supplied title map.
    def __reorganize_dataframe(self, df_input: pd.DataFrame, title_map):
        ordered_cols = list(title_map.values())
        df_input.rename(columns=title_map, inplace=True)
        df_input.reindex(columns=ordered_cols)

    def load(self):
        input_file_loc = self.input_file_loc
        # reading param setting from json file
        with open(os.path.join(input_file_loc, 'model_config.json')) as fp:
            param = json.load(fp)

        self.param_alpha = param['alpha']
        self.param_BIG_NUM = param['BIG_NUM']
        self.param_gap_limit = param['gap_limit']
        self.param_time_limit = param['time_limit']
        self.param_output_flag = param['output_flag']

        # reading data from each dataset
        input_file_loc = os.path.join(input_file_loc, self.dataset_name)

        input_file1 = os.path.join(input_file_loc, "order.csv")
        input_file2 = os.path.join(input_file_loc, "product_size.csv")
        input_file3 = os.path.join(input_file_loc, "vehicle.csv")
        input_file4 = os.path.join(input_file_loc, "warehouse_storage.csv")

        # Dealer demand for SKUs.
        df_order = pd.read_csv(input_file1, header=0)
        # SKU volume information.
        df_sku_size = pd.read_csv(input_file2, header=0)
        # Vehicle type information, including capacity, minimum load, and usage cost.
        df_all_veh = pd.read_csv(input_file3, header=0)
        # SKU inventory at production plants.
        df_sku_inv = pd.read_csv(input_file4, header=0)

        # Normalize the data in order.csv.
        title_map = {'client_code': 'dealer_id', 'product_code': 'sku_id', 'volume': 'order_qty'}
        self.__reorganize_dataframe(df_order, title_map)
        df_order['dealer_id'] = df_order['dealer_id'].astype(str)
        df_order['sku_id'] = df_order['sku_id'].astype(str)
        df_order['order_qty'] = df_order['order_qty'].astype(int)

        # Normalize the data in product_size.csv.
        title_map = {'product_code': 'sku_id', 'standard_size': 'size'}
        self.__reorganize_dataframe(df_sku_size, title_map)
        df_sku_size['sku_id'] = df_sku_size['sku_id'].astype(str)
        df_sku_size['size'] = df_sku_size['size'].astype(float)

        # Normalize the data in vehicle.csv.
        title_map = {'vehicle_type': 'vehicle_type', 'carry_standard_size': 'capacity',
                     'min_standard_size': 'min_load', 'cost_to_use': 'cost'}
        self.__reorganize_dataframe(df_all_veh, title_map)
        df_all_veh['vehicle_type'] = df_all_veh['vehicle_type'].astype(str)
        df_all_veh['capacity'] = df_all_veh['capacity'].astype(int)
        df_all_veh['min_load'] = df_all_veh['min_load'].astype(int)
        df_all_veh['cost'] = df_all_veh['cost'].astype(float)

        # Normalize the data in warehouse_storage.csv.
        title_map = {'plant_code': 'fact_id', 'product_code': 'sku_id', 'volume': 'inv'}
        self.__reorganize_dataframe(df_sku_inv, title_map)
        df_sku_inv['fact_id'] = df_sku_inv['fact_id'].astype(str)
        df_sku_inv['sku_id'] = df_sku_inv['sku_id'].astype(str)
        df_sku_inv['inv'] = df_sku_inv['inv'].astype(int)

        # After normalization, derive the data interfaces declared above.
        plants = list(df_sku_inv['fact_id'].unique())
        self.plants = {plant for plant in plants}

        dealers = list(df_order['dealer_id'].unique())
        self.dealers = {dealer for dealer in dealers}

        all_skus = list(df_sku_size['sku_id'].unique())
        self.all_skus = {sku for sku in all_skus}
        
        # # ============================================================
        # # The model assumes that each vehicle type has an unlimited fleet, but every vehicle still needs an identifier.
        # # One approach is to assign the same sufficiently large upper bound to every vehicle type.
        # # This bound must not be excessively large because it can substantially increase model construction time
        # # or even exhaust memory.
        # # In repeated runs, this parameter also has a strong effect on feasibility and should be tuned for each dataset.
        # # Intuitively, too few vehicles cannot carry all SKUs required by the dealers.
        # # This approach cannot provide a tight bound and is therefore not recommended.
        # # ============================================================
        # # Identify all vehicle types present in the dataset.
        # self.all_veh_types = list(df_all_veh['vehicle_type'].unique())
        # # Assign sequential identifiers to all vehicles, starting from 1.
        # ub = self.MAX_VEH_NUM * len(self.all_veh_types) + 1
        # self.all_vehicles = {v_id for v_id in range(1, ub)}
        # {v_id: v_type}
        # # Assume that vehicles are numbered by type from largest to smallest.
        # # This keeps identifiers for the same type contiguous and makes it easy to retrieve a type's vehicle identifiers.
        # sorted_veh_types = sorted(self.all_veh_types, key=float, reverse=True)
        # # For each vehicle identifier v_id, (v_id - 1) // max_vehicles_per_type determines its vehicle type.
        # # Subtracting 1 accounts for identifiers starting at 1 rather than 0.
        # # Integer division // produces an index suitable for the sorted type list.
        # self.veh_type = {v_id: sorted_veh_types[(v_id - 1) // self.MAX_VEH_NUM] for v_id in self.all_vehicles}
        #
        # # Compute the set of vehicle identifiers associated with each vehicle type.
        # self.veh_ids = {}
        # for i, v_type in enumerate(sorted_veh_types):
        #     # Compute the first and last identifier for the current vehicle type.
        #     start_id = i * self.MAX_VEH_NUM + 1
        #     end_id = (i + 1) * self.MAX_VEH_NUM
        #     # Create the set of vehicle identifiers for this type.
        #     self.veh_ids[v_type] = set(range(start_id, end_id + 1))
        # # ============================================================
        
        # {sku_id: size}
        self.sku_sizes = dict(zip(df_sku_size['sku_id'], df_sku_size['size']))

        # {(plant, sku): inv}
        self.sku_inv = df_sku_inv.set_index(['fact_id', 'sku_id'])['inv'].to_dict()
        # {plant: {sku}}
        self.skus_plant = {}
        for (plant, sku) in self.sku_inv:
            if plant not in self.skus_plant:
                self.skus_plant[plant] = set()
            self.skus_plant[plant].add(sku)

        # {sku: total_qty}
        self.sku_total_inv = defaultdict(int)
        for (plant, sku), qty in self.sku_inv.items():
            self.sku_total_inv[sku] += qty
        self.sku_total_inv = dict(self.sku_total_inv)

        # {(dealer, sku): demand}
        self.demands = df_order.set_index(['dealer_id', 'sku_id'])['order_qty'].to_dict()
        # {dealer: {sku}}
        self.skus_dealer = {}
        for (dealer, sku) in self.demands:
            if dealer not in self.skus_dealer:
                self.skus_dealer[dealer] = set()
            self.skus_dealer[dealer].add(sku)
        # ============================================================
        # Method 2 computes an upper bound on the required number of vehicles.
        # Identify all vehicle types present in the dataset.
        self.all_veh_types = list(df_all_veh['vehicle_type'].unique())
        
        # Compute the total volume of all SKUs held by all production plants.
        total_volume1 = sum(self.sku_inv[plant, sku] * self.sku_sizes[sku] for (plant, sku) in self.sku_inv)
        # Compute the total volume of all SKUs required by all dealers.
        total_volume2 = sum(self.demands[dealer, sku] * self.sku_sizes[sku] for (dealer, sku) in self.demands)
        
        # Sort vehicle types by size in ascending order.
        sorted_veh_types = sorted(self.all_veh_types, key=float)
        # Identify the smallest vehicle type.
        min_veh_type = sorted_veh_types[0]
        # Compute the capacity of each vehicle type: {v_type: cap}.
        each_type_cap = dict(zip(df_all_veh['vehicle_type'], df_all_veh['capacity']))
        # Capacity of the smallest vehicle type.
        min_veh_cap = each_type_cap[min_veh_type]
        
        # Compute upper bounds on the required vehicle count; veh_num_ub2 is clearly smaller than veh_num_ub1.
        veh_num_ub1 = math.floor(total_volume1 / min_veh_cap)
        veh_num_ub2 = math.floor(total_volume2 / min_veh_cap)
        print(f"The first upper bound on the number of vehicles used is {veh_num_ub1}")
        print(f"The second upper bound on the number of vehicles used is {veh_num_ub2}")
        plant_nums = len(self.plants)
        dealer_nums = len(self.dealers)
        veh_num_ub3 = plant_nums * dealer_nums
        print(f"The third upper bound on the number of vehicles used is {veh_num_ub3}")
        self.max_veh_num = max(veh_num_ub1, veh_num_ub2, veh_num_ub3)
        # Taking the largest of the three bounds keeps the model conservative:
        # the number of variables and constraints grows with the number of
        # available vehicles, so a looser bound costs build and solve time. The
        # generated instances keep the three bounds close, so the extra vehicles
        # have limited impact.
        
        # Assign sequential identifiers to all vehicles.
        self.all_vehicles = {v_id for v_id in range(1, self.max_veh_num + 1)}
        # Assign a vehicle type to each vehicle: {v_id: v_type}.
        self.veh_type = {}
        for v_id in self.all_vehicles:
            # Randomly select a vehicle type.
            v_type = random.sample(self.all_veh_types, k=1)
            self.veh_type[v_id] = v_type[0]
        # Group vehicle identifiers by vehicle type.
        self.veh_ids = {v_type: set() for v_type in self.all_veh_types}
        for v_type in self.veh_ids.keys():
            for v_id in self.veh_type.keys():
                if self.veh_type[v_id] == v_type:
                    self.veh_ids[v_type].add(v_id)
        # Derive the number of vehicles assigned to each vehicle type.
        each_type_nums = {}
        for v_type, veh_set in self.veh_ids.items():
            each_type_nums[v_type] = len(veh_set)
        # ============================================================

        self.supply_chain = defaultdict(set)
        for plant, skus_supply in self.skus_plant.items():
            for dealer, skus_needed in self.skus_dealer.items():
                for sku in skus_needed:
                    if sku in skus_supply:
                        self.supply_chain[plant, dealer].add(sku)
        self.supply_chain = dict(self.supply_chain)


        # Usage cost of each vehicle type: {v_type: cost}.
        each_type_cost = dict(zip(df_all_veh['vehicle_type'], df_all_veh['cost']))
        # {v_id: cost}
        self.veh_cost = {v_id: each_type_cost[self.veh_type[v_id]] for v_id in self.all_vehicles}
        # Capacity of each vehicle type: {v_type: cap}.
        each_type_cap = dict(zip(df_all_veh['vehicle_type'], df_all_veh['capacity']))
        # {v_id: capacity}
        self.veh_cap = {v_id: each_type_cap[self.veh_type[v_id]] for v_id in self.all_vehicles}
        # Minimum load of each vehicle type: {v_type: min_load}.
        each_type_min_load = dict(zip(df_all_veh['vehicle_type'], df_all_veh['min_load']))
        # {v_id: min_load}
        self.veh_min_load = {v_id: each_type_min_load[self.veh_type[v_id]] for v_id in self.all_vehicles}
