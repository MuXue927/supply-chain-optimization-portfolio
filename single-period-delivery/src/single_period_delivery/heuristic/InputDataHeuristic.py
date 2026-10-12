# Data interface for the heuristic algorithm.
# Loads and normalizes the data stored in datasets/single-period.
# Exposes each dataset in the representation required by the heuristic.
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Set, Tuple

import pandas as pd


@dataclass
class InputDataHeuristic:
    input_file_loc: str
    output_file_loc: str
    # specify which dataset to load
    dataset_name: str
    
    # The single-period problem uses a planning horizon of one period.
    horizons = 1
    # Planning starts on the first day by default.
    day = 1
    
    # Required data interfaces.
    # Set of production plants.
    plants: Set[str] = field(init=False)
    # Set of dealers.
    dealers: Set[str] = field(init=False)
    # Set of all SKUs.
    all_skus: Set[str] = field(init=False)
    # Set of all vehicle types.
    all_veh_types: Set[str] = field(init=False)
    
    # Unit volume of each SKU: {sku: size}.
    sku_sizes: Dict[str, float] = field(init=False)  
    
    # Current SKU inventory at each production plant: {(plant, sku): qty}.
    sku_inv: Dict[Tuple[str, str], int] = field(init=False)
    # SKUs that can be supplied by each production plant.
    skus_plant: Dict[str, Set[str]] = field(init=False)
    
    # Dealer demand for each SKU: {(dealer, sku): qty}.
    demands: Dict[Tuple[str, str], int] = field(init=False)
    # SKUs required by each dealer.
    skus_dealer: Dict[str, Set[str]] = field(init=False)
    
    # Usage cost of each vehicle type: {v_type: cost}.
    veh_type_cost: Dict[str, float] = field(init=False)
    # Capacity of each vehicle type: {v_type: capacity}.
    veh_type_cap: Dict[str, int] = field(init=False)
    # Minimum load of each vehicle type: {v_type: min_load}.
    veh_type_min_load: Dict[str, int] = field(init=False)
    # Penalty coefficient applied to underloaded vehicles.
    param_alpha: float = field(init=False)
    
    df_order: pd.DataFrame = field(default_factory=pd.DataFrame)
    df_sku_inv: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    
    # Load and normalize the input data.
    def __reorganize_dataframe(self, df_input: pd.DataFrame, title_map):
        ordered_cols = list(title_map.values())
        df_input.rename(columns=title_map, inplace=True)
        df_input.reindex(columns=ordered_cols)
        
    def load(self):
        input_file_loc = self.input_file_loc
        with open(os.path.join(input_file_loc, "model_config.json")) as fp:
            model_config = json.load(fp)
        self.param_alpha = float(model_config.get("alpha", 2.0))

        # reading data from each dataset
        input_file_loc = os.path.join(input_file_loc, self.dataset_name)
        
        input_file1 = os.path.join(input_file_loc, "order.csv")
        input_file2 = os.path.join(input_file_loc, "product_size.csv")
        input_file3 = os.path.join(input_file_loc, "vehicle.csv")
        input_file4 = os.path.join(input_file_loc, "warehouse_storage.csv")
        
        # Dealer demand for SKUs.
        self.df_order = pd.read_csv(input_file1, header=0)
        # SKU volume information.
        df_sku_size = pd.read_csv(input_file2, header=0)
        # Vehicle type information, including capacity, minimum load, and usage cost.
        df_all_veh = pd.read_csv(input_file3, header=0)
        # SKU inventory at production plants.
        self.df_sku_inv = pd.read_csv(input_file4, header=0)
        
        # Normalize the data in order.csv.
        title_map = {'client_code': 'dealer_id', 'product_code': 'sku_id', 'volume': 'order_qty'}
        self.__reorganize_dataframe(self.df_order, title_map)
        self.df_order['dealer_id'] = self.df_order['dealer_id'].astype(str)
        self.df_order['sku_id'] = self.df_order['sku_id'].astype(str)
        self.df_order['order_qty'] = self.df_order['order_qty'].astype(int)
        
        # Normalize the data in product_size.csv.
        title_map = {'product_code': 'sku_id', 'standard_size': 'size'}
        self.__reorganize_dataframe(df_sku_size, title_map)
        df_sku_size['sku_id'] = df_sku_size['sku_id'].astype(str)
        df_sku_size['size'] = df_sku_size['size'].astype(float)
        
        # Normalize the data in vehicle.csv.
        title_map = {
            'vehicle_type': 'vehicle_type', 'carry_standard_size': 'capacity',
            'min_standard_size': 'min_load', 'cost_to_use': 'cost'
        }
        self.__reorganize_dataframe(df_all_veh, title_map)
        df_all_veh['vehicle_type'] = df_all_veh['vehicle_type'].astype(str)
        df_all_veh['capacity'] = df_all_veh['capacity'].astype(int)
        df_all_veh['min_load'] = df_all_veh['min_load'].astype(int)
        df_all_veh['cost'] = df_all_veh['cost'].astype(float)
        
        # Normalize the data in warehouse_storage.csv.
        title_map = {'plant_code': 'fact_id', 'product_code': 'sku_id', 'volume': 'inv'}
        self.__reorganize_dataframe(self.df_sku_inv, title_map)
        self.df_sku_inv['fact_id'] = self.df_sku_inv['fact_id'].astype(str)
        self.df_sku_inv['sku_id'] = self.df_sku_inv['sku_id'].astype(str)
        self.df_sku_inv['inv'] = self.df_sku_inv['inv'].astype(int)
        
        # After normalization, derive the data interfaces declared above.
        plants = list(self.df_sku_inv['fact_id'].unique())
        self.plants = {plant for plant in plants}

        dealers = list(self.df_order['dealer_id'].unique())
        self.dealers = {dealer for dealer in dealers}

        all_skus = list(df_sku_size['sku_id'].unique())
        self.all_skus = {sku for sku in all_skus}

        all_veh_types = list(df_all_veh['vehicle_type'].unique())
        self.all_veh_types = {v_type for v_type in all_veh_types}
        
        # {sku_id: size}
        self.sku_sizes = dict(zip(df_sku_size['sku_id'], df_sku_size['size']))
        # {(plant, sku): inv}
        self.sku_inv = self.df_sku_inv.set_index(['fact_id', 'sku_id'])['inv'].to_dict()
        
        # {plant: {sku}}
        self.skus_plant = {}
        for (plant, sku) in self.sku_inv:
            if plant not in self.skus_plant:
                self.skus_plant[plant] = set()
            self.skus_plant[plant].add(sku)
            
        # {(dealer, sku): demand}
        self.demands = self.df_order.set_index(['dealer_id', 'sku_id'])['order_qty'].to_dict()
        # {dealer: {sku}}
        self.skus_dealer = {}
        for (dealer, sku) in self.demands:
            if dealer not in self.skus_dealer:
                self.skus_dealer[dealer] = set()
            self.skus_dealer[dealer].add(sku)
        
        # {v_type: cost}
        self.veh_type_cost = dict(zip(df_all_veh['vehicle_type'], df_all_veh['cost']))
        # {v_type: cap}
        self.veh_type_cap = dict(zip(df_all_veh['vehicle_type'], df_all_veh['capacity']))
        # {v_type: min_load}
        self.veh_type_min_load = dict(zip(df_all_veh['vehicle_type'], df_all_veh['min_load']))
        
    
    def available_skus_in_plant(self, plant_id: str):
        """
        find skus available in a specific plant
        """
        skus = {sku_id for fact_id, skus in self.skus_plant.items() for sku_id in skus if fact_id == plant_id}
        return skus
    
    def plan_dealer_skus(self, plant_id: str):
        """
        find dealer and sku tuples needed to be satisfied by a specific plant
        """
        # duet_dealer_sku = {
        #     (dealer, sku) for (dealer, sku) in self.demands for (plant, sku_id) in self.sku_inv
        #     if plant == plant_id and sku == sku_id
        # }
        duet_dealer_sku = {
            (dealer, sku) for (plant, dealer), skus_available in self.construct_supply_chain().items()
            for sku in skus_available if plant == plant_id
        }
        return duet_dealer_sku
    
    def available_skus_to_dealer(self, plant_id: str, dealer_id: str):
        """
        find skus available to a specific dealer in a specific plant
        """
        require_skus = {sku_id for (dealer, sku_id) in self.plan_dealer_skus(plant_id) if dealer == dealer_id}
        return require_skus
    
    def construct_supply_chain(self):
        """
        construct supply chain between plants and dealers
        """
        supply_chain = defaultdict(set)
        for plant, skus_supply in self.skus_plant.items():
            for dealer, skus_needed in self.skus_dealer.items():
                for sku in skus_needed:
                    if sku in skus_supply:
                        supply_chain[plant, dealer].add(sku)
        supply_chain = dict(supply_chain)
        return supply_chain
