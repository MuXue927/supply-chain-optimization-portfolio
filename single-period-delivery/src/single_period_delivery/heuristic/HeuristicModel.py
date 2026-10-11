import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set
from .InputDataHeuristic import InputDataHeuristic
from .vehicle import Vehicle
from .optutility import POSITIVEEPS

@dataclass
class HeuristicModel:
    data: InputDataHeuristic
    
    # Set X_ij of SKU types and quantities supplied from production plant i to dealer j.
    X_ij: Dict[Tuple[str, str], Set[Tuple[str, int]]] = field(default_factory=dict)
    
    # Quantity of SKU k supplied from production plant i to dealer j: {(i, j, k): qty}.
    supply: Dict[Tuple[str, str, str], int] = field(default_factory=dict)
    
    # SKUs already considered for each dealer j.
    SKUs_considered: Dict[str, Set[str]] = field(default_factory=dict)
    
    # Set L_f of vehicle identifiers l assigned to vehicle type f.
    Veh_type_sets: Dict[str, Set[int]] = field(default_factory=dict)
    
    # Number N_f of vehicles used for vehicle type f.
    N_f: Dict[str, int] = field(default_factory=dict)
    
    # Number of vehicles of type f used for each production plant i and dealer j.
    Nums_veh_used: Dict[Tuple[str, str, str], int] = field(default_factory=dict)
    
    # Records for the feasible solution.
    # Quantity x_k of SKU k supplied from production plant i to dealer j by vehicle l of type f.
    Vehicles: List[Vehicle] = field(default_factory=list)
    X_full: Dict[Tuple[str, str, str, str, int], int] = field(default_factory=dict)
    
    # SKUs already loaded for the route between production plant i and dealer j.
    SKUs_loaded: Dict[Tuple[str, str], Set[str]] = field(default_factory=dict)
    
    def optimize_stage_1(self):
        
        # Deep-copy demand and inventory because both are updated during the heuristic.
        demands = deepcopy(self.data.demands)
        sku_inv = deepcopy(self.data.sku_inv)
        
        # Initialize X_ij.
        for (plant, dealer) in self.data.construct_supply_chain().keys():
            self.X_ij[plant, dealer] = set()
        
        # =====================================================================================
        # Initialize SKUs_considered.
        for dealer in self.data.dealers:
            self.SKUs_considered[dealer] = set()
        
        for (plant, dealer), skus_set in self.data.construct_supply_chain().items():
            for sku_id in skus_set:
                # Skip SKU k if it has already been considered for dealer j.
                if sku_id in self.SKUs_considered[dealer]:
                    continue
                
                # Retrieve dealer j's demand for SKU k.
                demand = demands[dealer, sku_id]
                
                # Iterate over production plants i until the demand is satisfied.
                while demand > 0:
                    for fact_id in self.data.plants:
                        if sku_id in self.data.skus_plant[fact_id]:
                            inv = sku_inv[fact_id, sku_id]
                            if inv > 0:
                                demand, inv = self.get_partial_solution(fact_id, dealer, sku_id, demand, inv)
                                
                                # Update inventory.
                                sku_inv[fact_id, sku_id] = inv 
                                # Update demand.
                                demands[dealer, sku_id] = demand
                                
                                if demand == 0:
                                    self.SKUs_considered[dealer].add(sku_id)
                                    break
        # =====================================================================================
                     
        return self.X_ij
                
                    
    def get_partial_solution(self, plant, dealer, sku, demand, inv):
        
        while demand > 0 and inv > 0:
            
            qty = min(demand, inv)
            
            demand -= qty
            inv -= qty
            
            self.X_ij[plant, dealer].add((sku, qty))
            
        return demand, inv
        
    def select_vehicle_type(self):
        # Randomly select a vehicle type.
        veh_type = random.choice(list(self.data.all_veh_types))
        return veh_type
                    
    def cal_supply(self):
        for (plant, dealer), duet_sku_qty in self.X_ij.items():
            for (sku, qty) in duet_sku_qty:
                self.supply[plant, dealer, sku] = self.supply.get((plant, dealer, sku), 0) + qty
        return self.supply
    
    def check_fulfill(self):
        # Check whether the partial solution from Stage 1 satisfies each dealer's demand.
        
        fulfill = {}
        for (dealer, sku) in self.data.demands.keys():
            fulfill[dealer, sku] = 0
        
        for (plant, dealer, sku), qty in self.supply.items():
            fulfill[dealer, sku] += qty
        
        for (dealer, sku), qty in fulfill.items():
            demand = self.data.demands[dealer, sku]
            if qty < demand:
                print(f"Dealer {dealer} still needs {demand - qty} of SKU {sku}.")
        
        return fulfill
    
    
    def update_X_ij(self):
        # Clear X_ij.
        self.X_ij.clear()
        # Initialize X_ij.
        for (plant, dealer, sku_id), qty in self.supply.items():
            self.X_ij[plant, dealer] = set()
        
        for (plant, dealer, sku_id), qty in self.supply.items():
            self.X_ij[plant, dealer].add((sku_id, qty))
        
        return self.X_ij
    
    def optimize_stage_2(self):
        
        for (plant, dealer) in self.X_ij.keys():
            # Initialize the loaded-SKU set for production plant i and dealer j.
            self.SKUs_loaded[(plant, dealer)] = set()
        
        # Create a dictionary that tracks the remaining unloaded quantity for each SKU.
        remaining_to_load = deepcopy(self.supply)
        
        for (plant, dealer) in self.X_ij.keys():
            
            # Randomly select vehicle type f_rnd for production plant i and dealer j.
            veh_type_rnd = self.select_vehicle_type()
            
            # Use a greedy loading strategy: create vehicles as needed rather than precomputing their count.
            vehicles_for_route = []  # Vehicles used on the current route.
            
            # Keep loading until every remaining SKU has been assigned.
            while True:
                # Check whether any SKU remains to be loaded.
                has_remaining = False
                for (fact_id, dealer_id, sku_id), qty in remaining_to_load.items():
                    if fact_id == plant and dealer_id == dealer and qty > 0:
                        has_remaining = True
                        break
                
                if not has_remaining:
                    break
                
                # Create a new vehicle.
                vehicle = Vehicle(fact_id=plant, dealer_id=dealer, type=veh_type_rnd, data=self.data)
                vehicles_for_route.append(vehicle)
                self.Vehicles.append(vehicle)
                
                # Track whether the current vehicle receives any cargo.
                vehicle_loaded = False
                
                # Try to load all remaining SKUs onto the current vehicle.
                for (fact_id, dealer_id, sku_id), qty in list(remaining_to_load.items()):
                    if fact_id == plant and dealer_id == dealer and qty > 0:
                        
                        original_qty = qty
                        qty = vehicle.load(sku_id, qty)
                        
                        # Update the remaining quantity.
                        remaining_to_load[fact_id, dealer_id, sku_id] = qty
                        
                        if qty == 0:
                            # Record the SKU once it has been fully loaded.
                            self.SKUs_loaded[plant, dealer].add(sku_id)
                        
                        if original_qty > qty:  # At least part of the SKU was loaded.
                            vehicle_loaded = True
                
                # If the current vehicle receives no SKU,
                if not vehicle_loaded:
                    # Remove the empty vehicle.
                    vehicles_for_route.pop()
                    self.Vehicles.pop()
                    break  # Exit the loop and finish loading for the current route.
            
            # Record the number of vehicles used on the current route.
            veh_nums = len(vehicles_for_route)
            if veh_nums > 0:
                self.Nums_veh_used[plant, dealer, veh_type_rnd] = veh_nums

        return self.Vehicles
                
        
    def cal_vehicle_type_nums(self):
        # Compute the number of vehicles for each vehicle type f.
        for veh_type in self.data.all_veh_types:
            self.N_f[veh_type] = 0
        
        # for veh_type in self.data.all_veh_types:
        #     for (i, j, k, f, l) in self.X_full.keys():
        #         if veh_type == f:
        #             self.N_f[veh_type] += 1
        
        for vehicle in self.Vehicles:
            veh_type = vehicle.type
            self.N_f[veh_type] += 1
        
        return self.N_f
    
    def cal_vehicle_type_sets(self):
        for veh_type in self.data.all_veh_types:
            self.Veh_type_sets[veh_type] = set()
            for veh_id in range(1, self.N_f[veh_type] + 1):
                self.Veh_type_sets[veh_type].add(veh_id)
        return self.Veh_type_sets
    
    def calculate_vehicle_nums(self):
        total_veh_nums = 0
        for veh_type, veh_nums in self.N_f.items():
            total_veh_nums += veh_nums
        return total_veh_nums
    
    def calculate_cost(self):
        total_cost = 0
        for veh_type, veh_nums in self.N_f.items():
            total_cost += self.data.veh_type_cost[veh_type] * veh_nums

        for vehicle in self.Vehicles:
            if not vehicle.cargo:
                continue
            loaded_volume = sum(
                self.data.sku_sizes[sku_id] * qty
                for sku_id, qty in vehicle.cargo.items()
            )
            min_load = self.data.veh_type_min_load[vehicle.type]
            if loaded_volume + POSITIVEEPS < min_load:
                total_cost += (
                    self.data.param_alpha
                    * self.data.veh_type_cost[vehicle.type]
                )
        return total_cost
        
    
    def calculate_solution(self):
        # Compute the complete feasible solution X_full.
        # Maintain a separate vehicle-ID counter for each vehicle type.
        vehicle_id_counters = {veh_type: 1 for veh_type in self.data.all_veh_types}
        
        # Iterate over all vehicles and assign identifiers directly.
        for vehicle in self.Vehicles:
            if vehicle.cargo:  # Process only vehicles carrying cargo.
                veh_type = vehicle.type
                plant = vehicle.fact_id
                dealer = vehicle.dealer_id
                
                # Assign the vehicle identifier.
                veh_id = vehicle_id_counters[veh_type]
                vehicle_id_counters[veh_type] += 1
                
                # Iterate over the SKUs loaded on the vehicle.
                for sku_id, qty in vehicle.cargo.items():
                    # Add the vehicle identifier and loaded SKU quantity to X_full.
                    self.X_full[(plant, dealer, sku_id, veh_type, veh_id)] = qty
     
        return self.X_full
    
    def check_solution(self):
        # Check whether the complete solution from Stage 2 satisfies each dealer's demand.
        order_fulfill = {}
        for (dealer, sku_id), qty in self.data.demands.items():
            order_fulfill[dealer, sku_id] = 0
        
        for (plant, dealer, sku_id, veh_type, veh_id), qty in self.X_full.items():
            order_fulfill[dealer, sku_id] += qty
        
        
        is_satisfied = True
        for (dealer, sku_id), demand in self.data.demands.items():
            fulfill = order_fulfill.get((dealer, sku_id), 0)
            
            if abs(demand - fulfill) > POSITIVEEPS:
                print(f"Dealer {dealer} demand of SKU {sku_id} is {demand}, fulfill is {fulfill}.")
            
            if demand > fulfill:
                print(f"Dealer {dealer} still needs {demand - fulfill} of SKU {sku_id}.")
                is_satisfied = False
            
            if demand < fulfill:
                print(f"Extra fulfill of dealer {dealer} for SKU {sku_id} is {fulfill - demand}.")
                is_satisfied = True

        return is_satisfied
