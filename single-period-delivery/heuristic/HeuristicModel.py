import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set
from InputDataHeuristic import InputDataHeuristic
from vehicle import Vehicle
from optutility import POSITIVEEPS

@dataclass
class HeuristicModel:
    data: InputDataHeuristic
    
    # 生产基地 i 向经销商 j 供给的SKU种类及数量的集合 X_ij
    X_ij: Dict[Tuple[str, str], Set[Tuple[str, int]]] = field(default_factory=dict)
    
    # 生产基地 i 向经销商 j 供给的SKU k 的数量 {(i, j, k): qty}
    supply: Dict[Tuple[str, str, str], int] = field(default_factory=dict)
    
    # 记录已经考虑过的经销商 j 需要的SKU集合
    SKUs_considered: Dict[str, Set[str]] = field(default_factory=dict)
    
    # 车型 f 的车辆 l 的集合 L_f
    Veh_type_sets: Dict[str, Set[int]] = field(default_factory=dict)
    
    # 车型 f 使用的车辆数量 N_f
    N_f: Dict[str, int] = field(default_factory=dict)
    
    # 记录每个生产基地 i 和经销商 j 使用车型 f 的车辆数量
    Nums_veh_used: Dict[Tuple[str, str, str], int] = field(default_factory=dict)
    
    # 用于记录可行解
    # 生产基地 i 使用车型 f 的车辆 l 向经销商 j 供给 SKU k 的数量 x_k
    Vehicles: List[Vehicle] = field(default_factory=list)
    X_full: Dict[Tuple[str, str, str, str, int], int] = field(default_factory=dict)
    
    # 记录生产基地 i 和经销商 j 之间已经装载的SKU集合
    SKUs_loaded: Dict[Tuple[str, str], Set[str]] = field(default_factory=dict)
    
    def optimize_stage_1(self):
        
        # 由于需要更新需求和库存，所以需要深拷贝
        demands = deepcopy(self.data.demands)
        sku_inv = deepcopy(self.data.sku_inv)
        
        # 初始化 X_ij
        for (plant, dealer) in self.data.construct_supply_chain().keys():
            self.X_ij[plant, dealer] = set()
        
        # =====================================================================================
        # 初始化 SKUs_considered
        for dealer in self.data.dealers:
            self.SKUs_considered[dealer] = set()
        
        for (plant, dealer), skus_set in self.data.construct_supply_chain().items():
            for sku_id in skus_set:
                # 如果经销商 j 需要的SKU k 已经被考虑过，则跳过
                if sku_id in self.SKUs_considered[dealer]:
                    continue
                
                # 获取经销商 j 对SKU k 的需求
                demand = demands[dealer, sku_id]
                
                # 遍历所有生产基地 i，直到满足需求
                while demand > 0:
                    for fact_id in self.data.plants:
                        if sku_id in self.data.skus_plant[fact_id]:
                            inv = sku_inv[fact_id, sku_id]
                            if inv > 0:
                                demand, inv = self.get_partial_solution(fact_id, dealer, sku_id, demand, inv)
                                
                                # 更新库存
                                sku_inv[fact_id, sku_id] = inv 
                                # 更新需求
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
        # 随机选择车型
        veh_type = random.choice(list(self.data.all_veh_types))
        return veh_type
                    
    def cal_supply(self):
        for (plant, dealer), duet_sku_qty in self.X_ij.items():
            for (sku, qty) in duet_sku_qty:
                self.supply[plant, dealer, sku] = self.supply.get((plant, dealer, sku), 0) + qty
        return self.supply
    
    def check_fulfill(self):
        # 检查第一阶段获得的部分解, 是否满足每个经销商的需求
        
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
        # 清空 X_ij
        self.X_ij.clear()
        # 初始化 X_ij
        for (plant, dealer, sku_id), qty in self.supply.items():
            self.X_ij[plant, dealer] = set()
        
        for (plant, dealer, sku_id), qty in self.supply.items():
            self.X_ij[plant, dealer].add((sku_id, qty))
        
        return self.X_ij
    
    def optimize_stage_2(self):
        
        for (plant, dealer) in self.X_ij.keys():
            # 初始化生产基地 i 和经销商 j 的已装载SKU集合
            self.SKUs_loaded[(plant, dealer)] = set()
        
        # 创建剩余装载量字典，追踪每个SKU的剩余未装载数量
        remaining_to_load = deepcopy(self.supply)
        
        for (plant, dealer) in self.X_ij.keys():
            
            # 为生产基地 i 和经销商 j 随机选择车型 f_rnd
            veh_type_rnd = self.select_vehicle_type()
            
            # 使用贪心装载策略：不预先计算车辆数量，而是按需创建车辆
            vehicles_for_route = []  # 当前路线使用的车辆列表
            
            # 持续装载直到所有SKU都被装载完毕
            while True:
                # 检查是否还有SKU需要装载
                has_remaining = False
                for (fact_id, dealer_id, sku_id), qty in remaining_to_load.items():
                    if fact_id == plant and dealer_id == dealer and qty > 0:
                        has_remaining = True
                        break
                
                if not has_remaining:
                    break
                
                # 创建新车辆
                vehicle = Vehicle(fact_id=plant, dealer_id=dealer, type=veh_type_rnd, data=self.data)
                vehicles_for_route.append(vehicle)
                self.Vehicles.append(vehicle)
                
                # 标记当前车辆是否装载了货物
                vehicle_loaded = False
                
                # 尝试将所有剩余的SKU装载到当前车辆
                for (fact_id, dealer_id, sku_id), qty in list(remaining_to_load.items()):
                    if fact_id == plant and dealer_id == dealer and qty > 0:
                        
                        original_qty = qty
                        qty = vehicle.load(sku_id, qty)
                        
                        # 更新剩余装载量
                        remaining_to_load[fact_id, dealer_id, sku_id] = qty
                        
                        if qty == 0:
                            # 如果当前SKU已经装载完毕，记录已经装载的SKU
                            self.SKUs_loaded[plant, dealer].add(sku_id)
                        
                        if original_qty > qty:  # 如果有SKU被装载
                            vehicle_loaded = True
                
                # 如果当前车辆没有装载任何SKU
                if not vehicle_loaded:
                    # 移除这个空车辆
                    vehicles_for_route.pop()
                    self.Vehicles.pop()
                    break  # 退出循环，结束当前路线的装载
            
            # 记录当前路线使用的车辆数量
            veh_nums = len(vehicles_for_route)
            if veh_nums > 0:
                self.Nums_veh_used[plant, dealer, veh_type_rnd] = veh_nums

        return self.Vehicles
                
        
    def cal_vehicle_type_nums(self):
        # 计算每种车型 f 的车辆数量
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
        return total_cost
        
    
    def calculate_solution(self):
        # 计算完整的可行解 X_full
        # 为每种车型维护一个车辆编号计数器
        vehicle_id_counters = {veh_type: 1 for veh_type in self.data.all_veh_types}
        
        # 直接遍历所有车辆，为每辆车分配编号
        for vehicle in self.Vehicles:
            if vehicle.cargo:  # 只处理有装载货物的车辆
                veh_type = vehicle.type
                plant = vehicle.fact_id
                dealer = vehicle.dealer_id
                
                # 分配车辆编号
                veh_id = vehicle_id_counters[veh_type]
                vehicle_id_counters[veh_type] += 1
                
                # 遍历车辆上装载的SKU
                for sku_id, qty in vehicle.cargo.items():
                    # 将车辆编号和SKU装载数量添加到完整解 X_full 中
                    self.X_full[(plant, dealer, sku_id, veh_type, veh_id)] = qty
     
        return self.X_full
    
    def check_solution(self):
        # 检查第二阶段获得的完整可行解, 是否满足每个经销商的需求
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