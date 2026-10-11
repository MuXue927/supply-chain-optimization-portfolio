# 定义多周期优化问题的数据接口
# 整理保存在datasets//multiple-periods文件夹中的数据
# 将每个数据集中提供的数据，按照你需要的方式呈现
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

    # 对于多周期问题，默认规划周期长度为 1
    # 具体数值是warehouse_production.csv文件中produce_date这一列的最大值
    horizons = 1

    # 设定环境参数
    # 模型中的惩罚系数 Alpha
    param_alpha: float = field(init=False)
    # 模型中的 M
    param_BIG_NUM: int = field(init=False)
    # MIP模型的求解gap，当达到设定的gap后，模型停止，默认值为0
    param_gap_limit: float = field(init=False)
    # MIP模型求解的时间限制，默认值为60
    param_time_limit: float = field(init=False)
    # 是否输出gurobi的求解日志
    param_output_flag: int = field(init=False)

    # 定义需要的数据接口，看数学模型中的符号定义
    # 生产基地的集合
    plants: Set[str] = field(init=False)
    # 经销商的集合
    dealers: Set[str] = field(init=False)
    # 所有SKU的集合
    all_skus: Set[str] = field(init=False)
    # 所有车辆编号的集合
    all_vehicles: Set[int] = field(init=False)
    
    # 每种SKU的体积量化数值 {sku: size}
    sku_sizes: Dict[str, float] = field(init=False)

    # 在每个周期内，生产基地计划生产的SKU数量 {(plant, sku, day): qty}
    sku_prod_each_day: Dict[Tuple[str, str, int], int] = field(init=False)
    # 生产基地中SKU的期初库存数量 {(plant, sku): qty}
    sku_initial_inv: Dict[Tuple[str, str], int] = field(init=False)
    
    # 生产基地可以提供的SKU集合
    skus_plant: Dict[str, Set[str]] = field(init=False)

    # 经销商对SKU的需求 {(dealer, sku): qty}
    demands: Dict[Tuple[str, str], int] = field(init=False)
    # 经销商需要的SKU集合
    skus_dealer: Dict[str, Set[str]] = field(init=False)

    # 每一辆车的使用成本 {v_id: cost}
    veh_cost: Dict[int, float] = field(init=False)
    
    # 每一辆车的容量上限 {v_id: capacity}
    veh_cap: Dict[int, float] = field(init=False)
    # 每一辆车的最小起运量 {v_id: min_load}
    veh_min_load: Dict[int, float] = field(init=False)
    
    # 每个生产基地的库存上限 {plant: max_cap}
    plant_inv_limit: Dict[str, float] = field(init=False)
    
    # 所有车辆类型
    all_veh_types: List[str] = field(init=False)
    # 有了车辆编号，还需要在车辆编号和车辆类型之间建立联系
    # 告诉我任意一辆车的编号，通过查看这辆车的类型，就能知道
    # 这辆车的所有信息，包括使用成本、容量上限、最小起运量
    # 
    veh_type: Dict[int, str] = field(init=False)
    # 同理，告诉我任意一种车辆的类型，我能知道这种车辆类型拥有哪些编号的车辆 {v_type: {v_id}}
    veh_ids: Dict[str, Set[int]] = field(init=False)

    # 生产基地和经销商之间的供需关系映射 {(plant, dealer): {sku}}
    supply_chain: Dict[Tuple[str, str], Set[str]] = field(init=False)# 可以使用的车辆数量上限
    
    # 可以使用的车辆数量上限
    max_veh_num: int = field(init=False)

    # 到这里，多周期优化问题需要的所有数据接口，已经定义完毕
    # 接下来，要做的事情是读取并整理数据

    # 开头带双下划线，用于对象的数据封装，以此命名的属性或者方法为类的私有属性或者私有方法，用于隐藏数据
    # 但这种机制并不是很严格，类中所有以双下划线开头的名称都会自动变为"_类名__name"的新名称
    # 这种机制可以阻止继承类重新定义或者更改方法的实现
    # 这个函数用于给dataframe中每一列重新命名
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

        # multiple-factor for sku size and vehicle capacity
        multiple_factor = 100
        # reading data from each dataset
        input_file_loc = os.path.join(input_file_loc, self.dataset_name)

        input_file1 = os.path.join(input_file_loc, "order.csv")
        input_file2 = os.path.join(input_file_loc, "plant_size_upper_limit.csv")
        input_file3 = os.path.join(input_file_loc, "product_size.csv")
        input_file4 = os.path.join(input_file_loc, "vehicle.csv")
        input_file5 = os.path.join(input_file_loc, "warehouse_production.csv")
        input_file6 = os.path.join(input_file_loc, "warehouse_storage.csv")

        # 经销商对SKU的需求信息
        df_order = pd.read_csv(input_file1, header=0)
        # 生产基地的库存容量上限
        df_plant_cap = pd.read_csv(input_file2, header=0)
        # SKU的体积信息
        df_sku_size = pd.read_csv(input_file3, header=0)
        # 所有车辆类型信息，包括车辆容量、最小起运量、使用成本
        df_all_veh = pd.read_csv(input_file4, header=0)
        # 生产基地在每个周期内生产的SKU数量
        df_sku_prod = pd.read_csv(input_file5, header=0)
        # 生产基地中SKU的期初库存
        df_sku_inv = pd.read_csv(input_file6, header=0)

        # 整理order.csv里面的数据
        title_map = {'client_code': 'dealer_id', 'product_code': 'sku_id', 'volume': 'order_qty'}
        self.__reorganize_dataframe(df_order, title_map)
        df_order['dealer_id'] = df_order['dealer_id'].astype(str)
        df_order['sku_id'] = df_order['sku_id'].astype(str)
        df_order['order_qty'] = df_order['order_qty'].astype(int)

        # 整理plant_size_upper_limit.csv里面的数据
        title_map = {'plant_code': 'fact_id', 'max_volume': 'max_cap'}
        self.__reorganize_dataframe(df_plant_cap, title_map)
        df_plant_cap['fact_id'] = df_plant_cap['fact_id'].astype(str)
        df_plant_cap['max_cap'] = df_plant_cap['max_cap'].astype(int)

        # 整理product_size.csv里面的数据
        title_map = {'product_code': 'sku_id', 'standard_size': 'size'}
        self.__reorganize_dataframe(df_sku_size, title_map)
        df_sku_size['sku_id'] = df_sku_size['sku_id'].astype(str)
        df_sku_size['size'] = df_sku_size['size'].astype(float)

        # 整理vehicle.csv里面的数据
        title_map = {'vehicle_type': 'vehicle_type', 'carry_standard_size': 'capacity',
                     'min_standard_size': 'min_load', 'cost_to_use': 'cost'}
        self.__reorganize_dataframe(df_all_veh, title_map)
        df_all_veh['vehicle_type'] = df_all_veh['vehicle_type'].astype(str)
        df_all_veh['capacity'] = df_all_veh['capacity'].astype(int)
        df_all_veh['min_load'] = df_all_veh['min_load'].astype(int)

        # 整理warehouse_production.csv里面的数据
        title_map = {'plant_code': 'fact_id', 'product_code': 'sku_id', 'produce_date': 'day', 'volume': 'prod_qty'}
        self.__reorganize_dataframe(df_sku_prod, title_map)
        df_sku_prod['fact_id'] = df_sku_prod['fact_id'].astype(str)
        df_sku_prod['sku_id'] = df_sku_prod['sku_id'].astype(str)
        df_sku_prod['day'] = df_sku_prod['day'].astype(int)
        df_sku_prod['prod_qty'] = df_sku_prod['prod_qty'].astype(int)

        # 整理warehouse_storage.csv里面的数据
        title_map = {'plant_code': 'fact_id', 'product_code': 'sku_id', 'volume': 'inv'}
        self.__reorganize_dataframe(df_sku_inv, title_map)
        df_sku_inv['fact_id'] = df_sku_inv['fact_id'].astype(str)
        df_sku_inv['sku_id'] = df_sku_inv['sku_id'].astype(str)
        df_sku_inv['inv'] = df_sku_inv['inv'].astype(int)

        # 在整理好数据之后，接下来需要计算出之前定义好的数据接口
        # 计算数据集中的最大规划周期
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
        # 这里需要注意的是，在计算生产基地可以提供的SKU集合时，需要同时检查期初库存和计划生产库存
        self.skus_plant = {}
        # 先添加期初库存对应的SKU
        for (plant, sku) in self.sku_initial_inv:
            if plant not in self.skus_plant:
                self.skus_plant[plant] = set()
            self.skus_plant[plant].add(sku)
        # 再添加计划生产库存对应的SKU
        # 由于上面已经添加了所有生产基地，所以不需要提前检查键是否存在
        # 其次，如果SKU已经在集合中，那么再添加相同的SKU对集合没有影响
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
        # 计算所有经销商对所有种类SKU的需求总量
        sku_total_demands = sum(self.demands.values())

        # =========================================================
        # 用于计算所需车辆数量的上界
        # 识别出数据集中存在的所有车辆类型
        self.all_veh_types = list(df_all_veh['vehicle_type'].unique())
        # 计算 所有生产基地 在期初 拥有的 所有种类SKU的体积之和
        total_volume1 = sum(self.sku_initial_inv[plant, sku] * self.sku_sizes[sku]
                            for (plant, sku) in self.sku_initial_inv)
        # 计算 所有经销商需要的所有种类SKU的体积之和
        total_volume2 = sum(self.demands[dealer, sku] * self.sku_sizes[sku] for (dealer, sku) in self.demands)
        # 将所有车辆类型按照车辆类型大小，升序排列
        sorted_veh_types = sorted(self.all_veh_types, key=float)
        # 找出最小车型
        min_veh_type = sorted_veh_types[0]
        # 计算每种车型的容量上限 {v_type: cap}
        each_type_cap = dict(zip(df_all_veh['vehicle_type'], df_all_veh['capacity']))
        # 最小车型对应的车辆容量
        min_veh_cap = each_type_cap[min_veh_type]
        # 计算所需车辆数量的上界
        veh_num_ub1 = math.floor(total_volume1 / min_veh_cap)
        veh_num_ub2 = math.floor(total_volume2 / min_veh_cap)
        print(f"The first upper bound on the number of vehicles used is {veh_num_ub1}")
        print(f"The second upper bound on the number of vehicles used is {veh_num_ub2}")

        plant_nums = len(self.plants)
        dealer_nums = len(self.dealers)
        veh_num_ub3 = plant_nums * dealer_nums
        print(f"The third upper bound on the number of vehicles used is {veh_num_ub3}")
        self.max_veh_num = max(veh_num_ub1, veh_num_ub2, veh_num_ub3)

        # 为所有车辆进行统一编号
        self.all_vehicles = {v_id for v_id in range(1, self.max_veh_num + 1)}
        # 接下来需要为每辆车指定一种车辆类型 {v_id: v_type}
        self.veh_type = {}
        for v_id in self.all_vehicles:
            # 随机选择一种车辆类型
            v_type = random.sample(self.all_veh_types, k=1)
            self.veh_type[v_id] = v_type[0]
        # 统计每一种车辆类型对应的车辆编号
        self.veh_ids = {v_type: set() for v_type in self.all_veh_types}
        for v_type in self.veh_ids.keys():
            for v_id in self.veh_type.keys():
                if self.veh_type[v_id] == v_type:
                    self.veh_ids[v_type].add(v_id)
        # 同时也可以计算出每一种车辆类型对应的车辆数量
        each_type_nums = {}
        for v_type, veh_set in self.veh_ids.items():
            each_type_nums[v_type] = len(veh_set)
        # =========================================================

        # 每种车型的使用成本 {v_type: cost}
        each_type_cost = dict(zip(df_all_veh['vehicle_type'], df_all_veh['cost']))
        # {v_id: cost}
        self.veh_cost = {v_id: each_type_cost[self.veh_type[v_id]] for v_id in self.all_vehicles}
        # 每种车型的容量上限 {v_type: cap}
        each_type_cap = dict(zip(df_all_veh['vehicle_type'], df_all_veh['capacity']))
        # {v_id: capacity}
        self.veh_cap = {v_id: each_type_cap[self.veh_type[v_id]] for v_id in self.all_vehicles}
        # 每种车型的最小起运量 {v_type: min_load}
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