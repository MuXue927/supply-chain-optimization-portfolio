# Define the data interface for the multi-period optimization problem.
# Organize the data stored under datasets/multiple-periods.
# Present each dataset in the form required by the optimization model.
import os
import math
import json
import random
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set

@dataclass
class InputDataMultiple:
    input_file_loc: str
    output_file_loc: str
    # specify which dataset to load
    dataset_name: str

    # Default planning horizon for multi-period instances.
    # The actual value is the maximum produce_date in warehouse_production.csv.
    horizons = 1

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
    
    # Volume per SKU {sku: size}.
    sku_sizes: Dict[str, float] = field(init=False)

    # Planned SKU production at each plant in each period {(plant, sku, day): qty}.
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
    # 
    veh_type: Dict[int, str] = field(init=False)
    # Conversely, each vehicle type identifies its vehicle IDs {v_type: {v_id}}.
    veh_ids: Dict[str, Set[int]] = field(init=False)

    # Supply relationships between plants and dealers {(plant, dealer): {sku}}.
    supply_chain: Dict[Tuple[str, str], Set[str]] = field(init=False)
    
    # Upper bound on the number of vehicles that can be used.
    max_veh_num: int = field(init=False)

    # All data interfaces required by the multi-period optimization problem are defined.
    # The next step is to read and organize the input data.

    # Names beginning with double underscores trigger Python name mangling and are
    # conventionally used to mark internal attributes or methods. Python rewrites
    # such names to _ClassName__name. This is not strict access control, but it
    # discourages accidental external access and redefinition in subclasses.
    # Rename the columns of a DataFrame.
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
        input_file2 = os.path.join(input_file_loc, "plant_size_upper_limit.csv")
        input_file3 = os.path.join(input_file_loc, "product_size.csv")
        input_file4 = os.path.join(input_file_loc, "vehicle.csv")
        input_file5 = os.path.join(input_file_loc, "warehouse_production.csv")
        input_file6 = os.path.join(input_file_loc, "warehouse_storage.csv")

        # Dealer demand for each SKU.
        df_order = pd.read_csv(input_file1, header=0)
        # Inventory capacity upper bound for each plant.
        df_plant_cap = pd.read_csv(input_file2, header=0)
        # Volume information for each SKU.
        df_sku_size = pd.read_csv(input_file3, header=0)
        # Vehicle type information, including capacity, minimum load, and use cost.
        df_all_veh = pd.read_csv(input_file4, header=0)
        # SKU production quantity at each plant in each period.
        df_sku_prod = pd.read_csv(input_file5, header=0)
        # Initial SKU inventory at each plant.
        df_sku_inv = pd.read_csv(input_file6, header=0)

        # Organize data from order.csv.
        title_map = {'client_code': 'dealer_id', 'product_code': 'sku_id', 'volume': 'order_qty'}
        self.__reorganize_dataframe(df_order, title_map)
        df_order['dealer_id'] = df_order['dealer_id'].astype(str)
        df_order['sku_id'] = df_order['sku_id'].astype(str)
        df_order['order_qty'] = df_order['order_qty'].astype(int)

        # Organize data from plant_size_upper_limit.csv.
        title_map = {'plant_code': 'fact_id', 'max_volume': 'max_cap'}
        self.__reorganize_dataframe(df_plant_cap, title_map)
        df_plant_cap['fact_id'] = df_plant_cap['fact_id'].astype(str)
        df_plant_cap['max_cap'] = df_plant_cap['max_cap'].astype(int)

        # Organize data from product_size.csv.
        title_map = {'product_code': 'sku_id', 'standard_size': 'size'}
        self.__reorganize_dataframe(df_sku_size, title_map)
        df_sku_size['sku_id'] = df_sku_size['sku_id'].astype(str)
        df_sku_size['size'] = df_sku_size['size'].astype(float)

        # Organize data from vehicle.csv.
        title_map = {'vehicle_type': 'vehicle_type', 'carry_standard_size': 'capacity',
                     'min_standard_size': 'min_load', 'cost_to_use': 'cost'}
        self.__reorganize_dataframe(df_all_veh, title_map)
        df_all_veh['vehicle_type'] = df_all_veh['vehicle_type'].astype(str)
        df_all_veh['capacity'] = df_all_veh['capacity'].astype(int)
        df_all_veh['min_load'] = df_all_veh['min_load'].astype(int)

        # Organize data from warehouse_production.csv.
        title_map = {'plant_code': 'fact_id', 'product_code': 'sku_id', 'produce_date': 'day', 'volume': 'prod_qty'}
        self.__reorganize_dataframe(df_sku_prod, title_map)
        df_sku_prod['fact_id'] = df_sku_prod['fact_id'].astype(str)
        df_sku_prod['sku_id'] = df_sku_prod['sku_id'].astype(str)
        df_sku_prod['day'] = df_sku_prod['day'].astype(int)
        df_sku_prod['prod_qty'] = df_sku_prod['prod_qty'].astype(int)

        # Organize data from warehouse_storage.csv.
        title_map = {'plant_code': 'fact_id', 'product_code': 'sku_id', 'volume': 'inv'}
        self.__reorganize_dataframe(df_sku_inv, title_map)
        df_sku_inv['fact_id'] = df_sku_inv['fact_id'].astype(str)
        df_sku_inv['sku_id'] = df_sku_inv['sku_id'].astype(str)
        df_sku_inv['inv'] = df_sku_inv['inv'].astype(int)

        # After organizing the input files, derive the data interfaces defined above.
        # Determine the maximum planning horizon in the dataset.
        if len(df_sku_prod.index) > 0:
            self.horizons = df_sku_prod['day'].max()
        plants = list(df_plant_cap['fact_id'].unique())
        self.plants = {plant for plant in plants}

        dealers = list(df_order['dealer_id'].unique())
        self.dealers = {dealer for dealer in dealers}

        all_skus = list(df_sku_size['sku_id'].unique())
        self.all_skus = {sku for sku in all_skus}

        # {sku_id: size}
        self.sku_sizes = dict(zip(df_sku_size['sku_id'], df_sku_size['size']))
        # {(plant, sku, day): prod_qty}
        self.sku_prod_each_day = df_sku_prod.set_index(['fact_id', 'sku_id', 'day'])['prod_qty'].to_dict()
        # {(plant, sku): inv}
        self.sku_initial_inv = df_sku_inv.set_index(['fact_id', 'sku_id'])['inv'].to_dict()
        # {plant: {sku}}
        # When deriving the SKUs available from each plant, include both initial
        # inventory and planned production.
        self.skus_plant = {}
        # First add SKUs available from initial inventory.
        for (plant, sku) in self.sku_initial_inv:
            if plant not in self.skus_plant:
                self.skus_plant[plant] = set()
            self.skus_plant[plant].add(sku)
        # Then add SKUs available from planned production.
        # Every plant has already been inserted above, so no key existence check is needed.
        # Adding a duplicate SKU to a set has no effect.
        for (plant, sku, day) in self.sku_prod_each_day:
            self.skus_plant[plant].add(sku)

        # {(dealer, sku): demand}
        self.demands = df_order.set_index(['dealer_id', 'sku_id'])['order_qty'].to_dict()
        # {dealer: {sku}}
        self.skus_dealer = {}
        for (dealer, sku) in self.demands:
            if dealer not in self.skus_dealer:
                self.skus_dealer[dealer] = set()
            self.skus_dealer[dealer].add(sku)
        # =========================================================
        # Derive an upper bound on the required number of vehicles.
        # Identify all vehicle types present in the dataset.
        self.all_veh_types = list(df_all_veh['vehicle_type'].unique())
        # Compute the total volume of initial inventory across all plants and SKUs.
        total_volume1 = sum(self.sku_initial_inv[plant, sku] * self.sku_sizes[sku]
                            for (plant, sku) in self.sku_initial_inv)
        # Compute the total volume required by all dealers across all SKUs.
        total_volume2 = sum(self.demands[dealer, sku] * self.sku_sizes[sku] for (dealer, sku) in self.demands)
        # Sort vehicle types in ascending order by capacity.
        sorted_veh_types = sorted(self.all_veh_types, key=float)
        # Identify the smallest vehicle type.
        min_veh_type = sorted_veh_types[0]
        # Map each vehicle type to its capacity upper bound {v_type: cap}.
        each_type_cap = dict(zip(df_all_veh['vehicle_type'], df_all_veh['capacity']))
        # Capacity of the smallest vehicle type.
        min_veh_cap = each_type_cap[min_veh_type]
        # Compute upper bounds on the required number of vehicles.
        veh_num_ub1 = math.floor(total_volume1 / min_veh_cap)
        veh_num_ub2 = math.floor(total_volume2 / min_veh_cap)
        print(f"The first upper bound on the number of vehicles used is {veh_num_ub1}")
        print(f"The second upper bound on the number of vehicles used is {veh_num_ub2}")

        plant_nums = len(self.plants)
        dealer_nums = len(self.dealers)
        veh_num_ub3 = plant_nums * dealer_nums
        print(f"The third upper bound on the number of vehicles used is {veh_num_ub3}")
        self.max_veh_num = max(veh_num_ub1, veh_num_ub2, veh_num_ub3)

        # Assign a unified ID to every vehicle.
        self.all_vehicles = {v_id for v_id in range(1, self.max_veh_num + 1)}
        # Assign a vehicle type to each vehicle {v_id: v_type}.
        self.veh_type = {}
        for v_id in self.all_vehicles:
            # Randomly select a vehicle type.
            v_type = random.sample(self.all_veh_types, k=1)
            self.veh_type[v_id] = v_type[0]
        # Group vehicle IDs by vehicle type.
        self.veh_ids = {v_type: set() for v_type in self.all_veh_types}
        for v_type in self.veh_ids.keys():
            for v_id in self.veh_type.keys():
                if self.veh_type[v_id] == v_type:
                    self.veh_ids[v_type].add(v_id)
        # Count the number of vehicles assigned to each vehicle type.
        each_type_nums = {}
        for v_type, veh_set in self.veh_ids.items():
            each_type_nums[v_type] = len(veh_set)
        # =========================================================

        # Use cost for each vehicle type {v_type: cost}.
        each_type_cost = dict(zip(df_all_veh['vehicle_type'], df_all_veh['cost']))
        # {v_id: cost}
        self.veh_cost = {v_id: each_type_cost[self.veh_type[v_id]] for v_id in self.all_vehicles}
        # Capacity upper bound for each vehicle type {v_type: cap}.
        each_type_cap = dict(zip(df_all_veh['vehicle_type'], df_all_veh['capacity']))
        # {v_id: capacity}
        self.veh_cap = {v_id: each_type_cap[self.veh_type[v_id]] for v_id in self.all_vehicles}
        # Minimum load for each vehicle type {v_type: min_load}.
        each_type_min_load = dict(zip(df_all_veh['vehicle_type'], df_all_veh['min_load']))
        # {v_id: min_load}
        self.veh_min_load = {v_id: each_type_min_load[self.veh_type[v_id]] for v_id in self.all_vehicles}
        # {plant: max_cap}
        self.plant_inv_limit = dict(zip(df_plant_cap['fact_id'], df_plant_cap['max_cap']))

        self.supply_chain = defaultdict(set)
        for plant, skus_supply in self.skus_plant.items():
            for dealer, skus_needed in self.skus_dealer.items():
                for sku in skus_needed:
                    if sku in skus_supply:
                        self.supply_chain[plant, dealer].add(sku)
        self.supply_chain = dict(self.supply_chain)
