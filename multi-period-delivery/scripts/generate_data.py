# 不依赖任何原始数据的输入, 生成全新的数据集
# 要求该程序对外开放一些接口, 表示生成数据集需要的关键参数
# 需要的关键参数包括：规划周期、数据集规模的大小、运行一次程序可以生成的数据集的数量
# 规划周期 -- horizons 默认值为1
# 数据集规模的大小 -- dataset_size [0, 1, 2] 0表示小规模  1表示中规模  2表示大规模
# 运行一次程序可以生成的数据集的数量 -- dataset_nums 默认值为1
# 上述这些参数, 要求可以在通过命令行调用py文件时, 跟随在脚本文件名称的后面
# 使用命令示例如下：
# python scripts/generate_data.py 1 0 3
# 上述命令表示, 生成3个单周期的小规模数据集

# 对于单周期的问题, 需要生成的csv文件包括：
# 1)order.csv文件
# 该文件包含三列, 分别是client_code、product_code、volume
# 2)product_size.csv文件
# 该文件包含两列, 分别是product_code、standard_size
# 3)vehicle.csv文件
# 该文件需要包含四列, 分别是vehicle_type、carry_standard_size、min_standard_size、cost_to_use
# 4)warehouse_storage.csv文件
# 该文件需要包含三列, 分别是plant_code、product_code、volume
# 注意, 需要让每家生产基地可以提供相同的SKU

# 对于多周期的问题, 需要生成的csv文件包括：
# 1)order.csv文件
# 该文件包含三列, 分别是client_code、product_code、volume
# 2)product_size.csv文件
# 该文件包含两列, 分别是product_code、standard_size
# 3)vehicle.csv文件
# 该文件需要包含四列, 分别是vehicle_type、carry_standard_size、min_standard_size、cost_to_use
# 4)warehouse_storage.csv文件
# 该文件需要包含三列, 分别是plant_code、product_code、volume
# 注意, 需要让每家生产基地可以提供相同的SKU
# 5)plant_size_upper_limit.csv文件
# 该文件包含两列, 分别是plant_code、max_volume
# 6)warehouse_production.csv文件
# 该文件包含四列, 分别是plant_code、product_code、produce_date、volume
# 同理, 在该文件中, 也需要让每家生产基地可以提供相同的SKU

# 另外, 还需要生成一个instance_statistics.csv文件
# 用于统计每个数据集的规模大小, 需要记录的信息包括：
# 生产基地数量、经销商数量、SKU数量、车辆类型数量、规划周期

import os
import json
import argparse
import random
import string
import time
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

# 确定脚本所在目录 (无论从何处运行, 基于脚本位置构建默认输出)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

parser = argparse.ArgumentParser(description="The arguments needed to generate dataset")
# 规划周期
parser.add_argument("horizons", type=int, default=1, help="The planning horizon needed to be considered")
# 数据集的规模, 0 -- 小规模, 1 -- 中规模, 2 -- 大规模
parser.add_argument("dataset_size", type=int, choices=[0, 1, 2], help="The size of dataset")
# 生成数据集的数量
parser.add_argument("dataset_nums", type=int, default=1,
                    help="The quantities of different datasets needed to be generated")
# 新增：可选的输出目录参数, 优先级高于默认的基于脚本的位置
parser.add_argument("--output-dir", type=str, default=None, help="Optional output directory. If omitted, defaults to datasets next to the script.")
args = parser.parse_args()
# 设置日志
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def generate_dataset():
    horizons = args.horizons
    dataset_size = args.dataset_size
    dataset_nums = args.dataset_nums

    logger.info(f"Start generating datasets: horizons={horizons}, dataset_size={dataset_size}, dataset_nums={dataset_nums}")

    # 使用 Path 对象构建路径, 优先使用用户提供的 --output-dir
    if args.output_dir:
        # 将用户提供的路径解析为绝对路径, 确保无论脚本被移动到何处, 输出仍写入期望位置
        base_file_loc = Path(args.output_dir).resolve()
    else:
        base_file_loc = PROJECT_ROOT / "datasets"

    if horizons == 1:
        print("Generating datasets for single period.")
        base_file_loc = base_file_loc / "single-period"
    else:
        print("Generating datasets for multiple periods.")
        base_file_loc = base_file_loc / "multiple-periods"

    if dataset_size == 0:
        base_file_loc = base_file_loc / "small"
        # base_file_loc = os.path.join(base_file_loc, "small-test")
    elif dataset_size == 1:
        base_file_loc = base_file_loc / "medium"
    else:
        base_file_loc = base_file_loc / "large"

    # 创建一个列表来存储每个数据集的统计信息
    dataset_statistics = []

    for i in range(1, dataset_nums + 1):
        output_file_loc = base_file_loc / f"dataset_{i}"
        # 创建数据集的保存路径 (使用 Path.mkdir, exist_ok=True)
        output_file_loc.mkdir(parents=True, exist_ok=True)
        logger.info(f"Preparing dataset {i}/{dataset_nums} at {output_file_loc}")
        start_time = time.time()

        if dataset_size == 0:  # 小规模
            plant_nums = random.randint(2, 3)
            dealer_nums = random.randint(2, 30)
            sku_nums = random.randint(10, 40)
            veh_type_nums = random.randint(2, 3)
            # plant_nums = 2
            # dealer_nums = 2
            # sku_nums = 5
            # veh_type_nums = 1
        elif dataset_size == 1:  # 中规模
            plant_nums = random.randint(3, 4)
            dealer_nums = random.randint(30, 50)
            sku_nums = random.randint(40, 60)
            veh_type_nums = random.randint(2, 4)
        else:  # 大规模
            plant_nums = random.randint(4, 6)
            dealer_nums = random.randint(50, 90)
            sku_nums = random.randint(60, 100)
            veh_type_nums = random.randint(2, 4)

        # 生成 生产基地、经销商、SKU、车辆类型的编号
        plant_ids = generate_ids(length=4, nums=plant_nums, id_type='factory')
        dealer_ids = generate_ids(length=8, nums=dealer_nums, id_type='dealer')
        sku_ids = generate_ids(length=11, nums=sku_nums, id_type='sku')
        veh_types = generate_ids(length=3, nums=veh_type_nums, id_type='vehicle')
        logger.info(f"Generated ids: plants={len(plant_ids)}, dealers={len(dealer_ids)}, skus={len(sku_ids)}, veh_types={len(veh_types)}")

        # ==============================================================================
        # 下面这些是单周期和多周期问题共有的文件
        # 生成订单信息 {(dealer, sku): qty}
        orders = generate_order(dealer_ids, sku_ids)
        # 生成每种SKU的体积信息 {sku: size}
        sku_sizes = generate_sku_size(sku_ids)
        # 生成每种车型的信息 {v_type: capacity}  {v_type: min_load}  {v_type: cost}
        veh_cap, veh_min_load, veh_cost = generate_vehicle(veh_types)

        # 将上述计算得到的所有信息保存到指定的csv文件
        order_to_csv(orders, file_loc=output_file_loc)
        logger.info(f"Saved order.csv to {output_file_loc / 'order.csv'}")
        sku_size_to_csv(sku_sizes, file_loc=output_file_loc)
        logger.info(f"Saved product_size.csv to {output_file_loc / 'product_size.csv'}")
        vehicle_to_csv(veh_cap, veh_min_load, veh_cost, file_loc=output_file_loc)
        logger.info(f"Saved vehicle.csv to {output_file_loc / 'vehicle.csv'}")
        # ==============================================================================

        if horizons == 1:  # 单周期
            # 单独计算生成SKU的库存信息 {plant: {sku: qty}}
            storage = generate_storage(plant_ids, sku_ids, orders)
            storage_to_csv(storage, file_loc=output_file_loc)
        else:  # 多周期
            # 在每个周期内, 每种SKU的总需求量
            sku_demands = aggregate_order(orders)
            
            # 计算所有SKU的总需求量
            total_sku_qty = sum(sku_demands.values())  # 所有经销商的总需求
            # 为确保基地有足够的空间, 可以增加10%的余量
            total_sku_qty = int(total_sku_qty * 1.1)
            # 每个生产基地的库存容量上限  {plant: capacity}
            # 使用更合理的计算方法：将总需求按生产基地分配 (允许一定不均衡), 
            # 并只加入较小的缓冲 (5%~25%), 确保总容量不会被设置得过于宽松。
            # 同时考虑规划期 (horizons), 以估计单周期/峰值生产带来的库存压力。
            plant_capacity = generate_plant_inv_limit(plant_ids, total_sku_qty, horizons)
            plant_capacity_to_csv(plant_capacity, file_loc=output_file_loc)
            
            # SKU的库存信息和计划生产信息  {plant: {sku: qty}}  {plant: {(sku, day): qty}}
            storage, production = generate_storage_and_production(plant_ids, sku_ids, orders, plant_capacity, horizons)
            storage_to_csv(storage, file_loc=output_file_loc)
            production_to_csv(production, file_loc=output_file_loc)
        # 收集数据集的统计信息 (包括供需汇总, 用于后续诊断)
        # 计算总需求 (所有经销商对所有SKU在所有周期内的需求总和)
        sku_demands = aggregate_order(orders)
        total_demand_qty = int(sum(sku_demands.values())) if sku_demands else 0

        # 计算期初库存总和和生产总和 (多周期场景下 production 存在)
        total_initial_inv = 0
        total_production_qty = 0
        # 单周期：storage 是 {plant: {sku: qty}}
        # 多周期：storage 同样存在, production 为 {plant: {(sku, day): qty}}
        try:
            # storage 在单/多周期均会被创建
            for plant, sku_map in storage.items():
                for sku, qty in sku_map.items():
                    total_initial_inv += int(qty)
        except NameError:
            total_initial_inv = 0

        try:
            for plant, prod_map in production.items():
                for _, qty in prod_map.items():
                    total_production_qty += int(qty)
        except NameError:
            total_production_qty = 0

        total_supply_qty = int(total_initial_inv + total_production_qty)
        supply_demand_ratio = round(total_supply_qty / total_demand_qty, 3) if total_demand_qty > 0 else None

        dataset_statistics.append({
            "dataset_name": f"dataset_{i}",
            "plant_nums": plant_nums,
            "dealer_nums": dealer_nums,
            "sku_nums": sku_nums,
            "vehicle_type_nums": veh_type_nums,
            "horizons": horizons,
            "total_demand_qty": total_demand_qty,
            "total_initial_inv": total_initial_inv,
            "total_production_qty": total_production_qty,
            "total_supply_qty": total_supply_qty,
            "supply_demand_ratio": supply_demand_ratio
        })
        elapsed = time.time() - start_time
        logger.info(f"Finished dataset {i}/{dataset_nums} in {elapsed:.2f}s")
    # 在所有数据集都生成完毕后, 保存统计信息
    save_dataset_statistics(dataset_statistics, file_loc=base_file_loc)
    print(f"All datasets have been saved in the location {base_file_loc}")
    print("Success!")

def save_dataset_statistics(dataset_statistics, file_loc):
    """
    统计每个数据集的规模大小

    :param dataset_statistics: 包含所有数据集统计信息的列表
    :param file_loc: 文件的保存位置
    :return: None
    """
    df_statistics = pd.DataFrame(dataset_statistics)
    file_loc = Path(file_loc)
    file = file_loc / "instance_statistics.csv"
    df_statistics.to_csv(file, index=False)

def order_to_csv(orders, file_loc):
    """
    将订单信息保存到指定的csv文件

    :param orders: 订单信息
    :param file_loc: 文件保存位置
    :return: None
    """
    rows = []
    if args.horizons == 1:  # 单周期
        for (dealer, sku), qty in orders.items():
            rows.append({
                'client_code': dealer,
                'product_code': sku,
                'volume': qty
            })
    else:  # 多周期, 这里记录的是经销商对 SKU k 在所有周期内的总需求
        for (dealer, sku), qty in orders.items():
            rows.append({
                'client_code': dealer,
                'product_code': sku,
                'volume': qty,
            })
    df_order = pd.DataFrame(rows)
    file_loc = Path(file_loc)
    file = file_loc / "order.csv"
    df_order.to_csv(file, index=False)  # 不保存索引

def sku_size_to_csv(sku_sizes, file_loc):
    """
    将SKU体积信息保存到指定的csv文件

    :param sku_sizes: 每种SKU的体积
    :param file_loc: 文件的保存位置
    :return: None
    """
    rows = []
    for sku, size in sku_sizes.items():
        rows.append({
            'product_code': sku,
            'standard_size': size
        })
    df_sku_size = pd.DataFrame(rows)
    file_loc = Path(file_loc)
    file = file_loc / "product_size.csv"
    df_sku_size.to_csv(file, index=False)

def vehicle_to_csv(veh_cap, veh_min_load, veh_cost, file_loc):
    """
    将车辆类型信息保存到指定的csv文件

    :param veh_cap: 车型容量
    :param veh_min_load: 车型最小起运量
    :param veh_cost: 车型使用成本
    :param file_loc: 文件的保存位置
    :return: None
    """
    if set(veh_cap.keys()) != set(veh_min_load.keys()) or set(veh_cap.keys()) != set(veh_cost.keys()):
        raise ValueError("Input dictionaries must have the same keys (vehicle types)")

    rows = [{
        'vehicle_type': v_type,
        'carry_standard_size': veh_cap[v_type],
        'min_standard_size': veh_min_load[v_type],
        'cost_to_use': veh_cost[v_type]
    } for v_type in veh_cap]
    df_vehicle = pd.DataFrame(rows)
    file_loc = Path(file_loc)
    file = file_loc / "vehicle.csv"
    df_vehicle.to_csv(file, index=False)

def storage_to_csv(storage, file_loc):
    """
    将SKU的库存信息保存到指定的csv文件

    :param storage: SKU的库存信息
    :param file_loc: 文件的保存位置
    :return: None
    """
    rows = []
    for plant, skus_supply in storage.items():
        for sku, qty in skus_supply.items():
            rows.append({
                'plant_code': plant,
                'product_code': sku,
                'volume': qty
            })
    df_storage = pd.DataFrame(rows)
    # 对于大型的dataframe, 可以使用布尔索引, 方便地删除多行数据
    # 指定要检查的列名
    col_name = 'volume'
    # 删除volume这一列取值为0对应的行, 并且使得删除后的dataframe的索引连续
    df_storage = df_storage[df_storage[col_name] != 0].reset_index(drop=True)
    file_loc = Path(file_loc)
    file = file_loc / "warehouse_storage.csv"
    df_storage.to_csv(file, index=False)

def plant_capacity_to_csv(plant_capacity, file_loc):
    """
    将生产基地的库存容量上限信息保存到指定的csv文件

    :param plant_capacity: 生产基地的库存容量上限
    :param file_loc: 文件的保存位置
    :return: None
    """
    rows = []
    for plant, cap in plant_capacity.items():
        rows.append({
            'plant_code': plant,
            'max_volume': cap
        })
        # print(f"DEBUG: plant = {plant}, max_volume = {cap}")
    df_plant_cap = pd.DataFrame(rows)
    file_loc = Path(file_loc)
    file = file_loc / "plant_size_upper_limit.csv"
    df_plant_cap.to_csv(file, index=False)

def production_to_csv(production, file_loc):
    """
    将SKU的计划生产数量信息保存到指定的csv文件

    :param production: SKU的计划生产数量
    :param file_loc: 文件的保存位置
    :return: None
    """
    rows = []
    for plant, prod_data in production.items():
        for (sku, day), qty in prod_data.items():
            rows.append({
                'plant_code': plant,
                'product_code': sku,
                'produce_date': day,
                'volume': qty
            })
    df_production = pd.DataFrame(rows)
    col_name = 'volume'
    df_production = df_production[df_production[col_name] != 0].reset_index(drop=True)
    file_loc = Path(file_loc)
    file = file_loc / "warehouse_production.csv"
    df_production.to_csv(file, index=False)

def generate_ids(length, nums, id_type='default'):
    """
    生成指定长度和数量的编号

    :param length: 编号的固定长度
    :param nums: 需要生成的编号数量
    :param id_type: 编号类型, 可选值：'factory', 'vehicle', 'dealer', 'sku'
    :return: 生成的编号列表
    """
    ids = set()  # 使用集合去重

    for _ in range(nums):
        if id_type == 'factory':
            first = str(random.randint(1, 9))
            rest = ''.join(random.choices(string.digits, k=length - 1))
            gen_id = first + rest
        elif id_type == 'vehicle':
            v_type = generate_float_num(lb=4.0, ub=20.0, precision=1)
            gen_id = str(v_type)
        elif id_type == 'dealer':
            first = random.choice(string.ascii_uppercase)
            rest = ''.join(random.sample(string.digits, k=length-1))
            gen_id = first + rest
        elif id_type == 'sku':
            first = random.choice(string.ascii_uppercase)
            rest = ''.join(random.sample(string.digits, k=length-1))
            gen_id = first + rest
        else:
            gen_id = ''.join(random.sample(string.ascii_uppercase + string.digits, k=length))
        ids.add(gen_id)

    return list(ids)

def generate_order(dealers, skus):
    """
    生成订单信息

    :param dealers: 经销商的编号
    :param skus: SKU的编号
    :return: 生成订单信息的字典
    """
    # 每个经销商需要的SKU集合
    dealer_sku_map = {dealer: set() for dealer in dealers}
    sku_dealer_map = {sku: set() for sku in skus}
    
    # 确保每个经销商至少对一种SKU有需求
    for dealer in dealers:
        nums = random.randint(1, len(skus))
        selected_skus = random.sample(skus, nums)
        for sku in selected_skus:
            dealer_sku_map[dealer].add(sku)
            sku_dealer_map[sku].add(dealer)

    # 确保每种SKU至少被一个经销商需求
    for sku in skus:
        if not sku_dealer_map[sku]:  # 如果没有经销商对该SKU有需求
            dealer = random.choice(dealers)
            dealer_sku_map[dealer].add(sku)
            sku_dealer_map[sku].add(dealer)

    orders = {}
    if args.horizons == 1:  # 单周期
        for dealer, skus_need in dealer_sku_map.items():
            for sku in skus_need:
                orders[dealer, sku] = int(round(random.uniform(5, 40)))
    
    else:  # 多周期, 这里记录的是经销商对 SKU k 在所有周期内的总需求
        for dealer, skus_need in dealer_sku_map.items():
            for sku in skus_need:
                orders[dealer, sku] = int(round(random.uniform(5, 40)))
    return orders

def generate_float_num(lb, ub, precision=1):
    """
    生成指定精度的浮点数

    :param lb: 生成浮点数的下限
    :param ub: 生成浮点数的上限
    :param precision: 浮点数的保留位数
    """
    return round(random.uniform(lb, ub), precision)

def generate_sku_size(skus):
    """
    生成每种SKU的体积信息

    :param skus: SKU的编号
    :return: 生成SKU体积的字典
    """
    sku_sizes = dict.fromkeys(skus)
    for sku in skus:
        sku_sizes[sku] = generate_float_num(lb=1.0, ub=3.0, precision=1)
    return sku_sizes

def generate_vehicle(veh_types):
    """
    生成每种类型车辆的容量信息、最小起运量信息、使用成本信息

    :param veh_types: 车辆类型
    :return: 生成车型容量的字典、车型最小起运量的字典、使用成本的字典
    """
    veh_cap = {}
    veh_min_load = {}
    veh_cost = {}
    # 首先需要对所有车型进行排序, 按照升序排列
    sorted_veh_types = sorted(veh_types, key=float)
    for i, v_type in enumerate(sorted_veh_types):
        type_value = float(v_type)
        # 根据车型大小计算基础容量
        base_cap = int(10 * type_value)
        # 在基础容量的80%到120%之间随机选择实际容量
        lb = round(base_cap * 0.8)
        ub = round(base_cap * 1.2)
        if i == 0:
            veh_cap[v_type] = random.randint(lb, ub)
        else:
            # 上一种车辆类型
            pre_type = sorted_veh_types[i - 1]
            pre_type_value = float(pre_type)
            # 上一种车型的容量
            pre_veh_cap = veh_cap[pre_type]
            if type_value - pre_type_value <= 1:  # 两种车型相近
                # 确保当前车型的容量不会小于上一种车型容量
                lb = pre_veh_cap + random.randint(1, 5)
                ub = pre_veh_cap + random.randint(6, 10)
                veh_cap[v_type] = random.randint(lb, ub)
            else:
                lb = pre_veh_cap + random.randint(10, 30)
                ub = pre_veh_cap + random.randint(40, 60)
                veh_cap[v_type] = random.randint(lb, ub)
        # 当前车型的容量
        capacity = veh_cap[v_type]
        # 最小起运量在 30%~50% 车型容量之间
        lb = round(0.3 * capacity)
        ub = round(0.5 * capacity)
        veh_min_load[v_type] = random.randint(lb, ub)
        # 每一种车型的使用成本
        # 大车的成本一定大于小车
        # 由于规模经济效应, 使用一辆大车的成本一定小于2辆小车
        min_cost = 1
        if i == 0:
            veh_cost[v_type] = min_cost
        else:
            # 上一种车辆类型
            pre_type = sorted_veh_types[i - 1]
            # 上一种车型的使用成本
            pre_veh_cost = veh_cost[pre_type]
            # 车型的使用成本与容量成正比关系, 但系数较小
            # 两种车型的容量差异
            cap_diff = veh_cap[v_type] - veh_cap[pre_type]
            cost_increase = round(cap_diff * 0.8) / veh_cap[pre_type]
            cost_increase = round(cost_increase, 2)
            veh_cost[v_type] = pre_veh_cost + cost_increase

    return veh_cap, veh_min_load, veh_cost

def aggregate_order(orders):
    """
    汇总经销商对某种SKU的需求

    :param orders: 经销商对SKU的需求信息
    :return: 生成SKU需求数量的字典
    """
    sku_demands = defaultdict(int)
    if args.horizons == 1:  # 单周期
        for (dealer_id, sku_id), qty in orders.items():
            sku_demands[sku_id] += qty
    else:  # 多周期
        for (dealer_id, sku_id), qty in orders.items():
            sku_demands[sku_id] += qty
    return dict(sku_demands)

def create_plant_sku_mapping(plants, skus):
    """
    创建映射关系, 表示每个生产基地可以提供的SKU集合

    :param plants: 生产基地编号
    :param skus: SKU编号
    :return: 生成表示生产基地可以提供的SKU集合的字典
    """
    # 每家生产基地可以提供的SKU集合
    plant_sku_map = {plant: set() for plant in plants}
    sku_plant_map = {sku: set() for sku in skus}
    
    # 确保每个生产基地至少提供一种SKU
    for plant in plants:
        nums = random.randint(1, len(skus))
        selected_skus = random.sample(skus, nums)
        for sku in selected_skus:
            plant_sku_map[plant].add(sku)
            sku_plant_map[sku].add(plant)
    
    # 确保每种SKU至少被一个生产基地提供
    for sku in skus:
        if not sku_plant_map[sku]:  # 如果没有生产基地提供这种SKU
            plant = random.choice(plants)
            plant_sku_map[plant].add(sku)
            sku_plant_map[sku].add(plant)

    return plant_sku_map

def generate_storage(plants, skus, orders):
    """
    对于单周期问题, 生成SKU的库存信息, 确保每种SKU的库存总量 > 所有经销商对这种SKU的需求

    :param plants: 生产基地的编号
    :param skus: SKU的编号
    :param orders: 经销商对SKU的需求信息
    :return: 生成SKU库存的字典
    """
    plant_sku_map = create_plant_sku_mapping(plants, skus)
    storage = {}  # 嵌套字典 {plant: {sku: qty}}
    for plant, skus_supply in plant_sku_map.items():
        storage[plant] = {}
        for sku in skus_supply:
            # 先设置库存数量为 1
            storage[plant][sku] = 1

    # 每种SKU的总需求量
    sku_demands = aggregate_order(orders)
    for sku, demand in sku_demands.items():
        # SKU的已有库存
        total_inv = sum(storage[plant].get(sku, 0) for plant in plant_sku_map)
        # 要确保SKU的库存量 > 需求量, 可以提前加上一个不为0的随机正整数
        demand += random.randint(10, 50)
        remaining_demand = int(demand - total_inv)
        if remaining_demand > 0:
            # 找出可以提供这种SKU的所有生产基地
            available_plants = [plant for plant in plant_sku_map if sku in plant_sku_map[plant]]
            # 在这些生产基地中分配剩余的需求量
            while remaining_demand >0 and available_plants:
                plant = random.choice(available_plants)
                allocation = random.randint(1, remaining_demand)
                storage[plant][sku] += allocation
                remaining_demand -= allocation
                if remaining_demand == 0:
                    break
                # 如果这个基地分配完毕, 就从可用的基地列表中移除
                available_plants.remove(plant)

            # 如果还有剩余需求, 就平均分配到所有可用基地
            if remaining_demand > 0:
                available_plants = [plant for plant in plant_sku_map if sku in plant_sku_map[plant]]
                per_plant_qty = remaining_demand // len(available_plants) + 1
                for plant in available_plants:
                    storage[plant][sku] += per_plant_qty
                    remaining_demand -= per_plant_qty
                    if remaining_demand <= 0:
                        break
    return storage

def total_production(production, sku):
    """
    计算在每个周期内, 所有生产基地生产某种SKU的总产量

    :param production: 在每个生产基地中, SKU在每个周期的计划生产数量
    :param sku: 某种特定的SKU
    :return: SKU在某个生产基地中在所有周期内的总产量
    """
    sku_production = defaultdict(int)
    for plant in production:
        for (sku_id, day), qty in production[plant].items():
            if sku_id == sku:
                sku_production[sku_id, day] += qty
    return dict(sku_production)

def weighted_choice(available_plants, weights):
    """
    使用加权法选择生产基地

    :param available_plants: 需要选择的生产基地
    :param weights: 权重
    """
    # print(f"DEBUG: weights = {weights}, sum(weights) = {np.sum(weights)}")  # 调试信息
    weights = [abs(w) for w in weights]
    if sum(weights) != 0:
        normalized_weights = np.array(weights) / sum(weights)
        return np.random.choice(available_plants, p=normalized_weights)
    else:
        return random.choice(available_plants)

def generate_plant_inv_limit(plants, total_sku_qty, horizons=1):
    """
    计算每个生产基地的库存容量上限 (改进版)

    思路：
    - 将总需求按生产基地分配为基础份额, 但允许小幅不均衡 (随机因子)以模拟不同厂的规模差异；
    - 在每个厂的基础份额上只加入较小的缓冲 (5%~25%), 避免将容量设置得过于宽松；
    - 考虑规划期 (horizons)用于估计单周期/峰值库存压力, 至少保证容量能容纳某一周期内的生产峰值；
    - 若分配后总容量仍小于总需求, 则按比例放大以保证可行性 (总容量 >= 总需求)。

    :param plants: 生产基地的编号 (list)
    :param total_sku_qty: 所有SKU的总量 (整型)
    :param horizons: 规划期 (用于估计单周期生产带来的峰值)
    :return: 生成每个生产基地库存容量的字典
    """
    if not plants:
        return {}

    n_plants = len(plants)
    # 基础份额：将总量均分到每个工厂 (浮点数)
    avg_share = float(total_sku_qty) / n_plants if n_plants > 0 else float(total_sku_qty)

    plant_inv_limit = {}
    for plant in plants:
        # 引入不均衡因子, 模拟工厂间规模差异 (0.6 ~ 1.4)
        imbalance = random.uniform(0.6, 1.4)
        base = max(1, int(avg_share * imbalance))

        # 估计每厂在单周期内可能产生或需要容纳的产量峰值
        # 保守估计：将总需求按horizons分摊, 再按厂数分配为单周期预期
        per_period_expected = max(1, int(round(total_sku_qty / max(1, horizons) / n_plants)))

        # 缓冲比例较小 (5%~25%), 避免容量过于宽松
        buffer = random.uniform(0.05, 0.25)

        # 最终容量至少为 base * (1 + buffer), 并保证能容纳一定的单周期峰值
        cap = int(max(base * (1.0 + buffer), per_period_expected * 2))
        plant_inv_limit[plant] = cap

    # 若总容量小于总需求, 则按比例放大每个厂的容量以保证可行性
    total_cap = sum(plant_inv_limit.values())
    if total_cap < total_sku_qty:
        if total_cap == 0:
            # 极端保护, 平均分配
            for plant in plant_inv_limit:
                plant_inv_limit[plant] = max(1, int(total_sku_qty // n_plants))
        else:
            scale = float(total_sku_qty) / total_cap
            for plant in plant_inv_limit:
                plant_inv_limit[plant] = max(1, int(round(plant_inv_limit[plant] * scale)))

    return plant_inv_limit

def distribute_prod(prod, selected_periods):
    """
    将生产数量prod分配到selected_periods指定的周期中

    :param prod: 要分配的总生产数量
    :param selected_periods: 要分配到的周期列表
    :return: 每个周期分配到的数量的字典
    """
    if not selected_periods:
        return {}

    n = len(selected_periods)
    # 计算每个周期至少分配到的基础数量
    base_amount = prod // n
    # 计算剩余需要分配的数量
    remaining = prod % n
    distribution = {}
    for i, period in enumerate(selected_periods):
        if i < remaining:
            distribution[period] = base_amount + 1
        else:
            distribution[period] = base_amount
    return distribution

def generate_storage_and_production(plants, skus, orders, plant_capacity, horizons):
    """
    对于多周期问题, 确保在所有周期内, SKU的总供给量 > 所有经销商对这种SKU的需求
    只需要确保在所有周期内, 所有生产基地生产SKU的总量 + 所有生产基地的期初库存 > 所有经销商对这种SKU的需求

    :param plants: 生产基地的编号
    :param skus: SKU的编号
    :param orders: 经销商对SKU的需求信息
    :param plant_capacity: 每个生产基地的容量上限
    :param horizons: 规划周期
    :return: 生成SKU库存的字典、SKU在每个周期内计划生产数量的字典
    """
    # 1. 统计每个SKU的总需求, 加10~50的余量
    sku_demands = aggregate_order(orders)
    sku_total_supply = {sku: int(np.ceil(sku_demands.get(sku, 0) + random.randint(10, 50))) for sku in skus}

    # --- 新增保护：确保全局供给不超过 plant 总容量，同时保证每个 SKU 的供给 >= demand ---
    plant_remain_cap = plant_capacity.copy()
    try:
        total_supply_desired = sum(int(v) for v in sku_total_supply.values())
        total_cap = sum(int(v) for v in plant_capacity.values()) if plant_capacity else 0
        total_demand = sum(int(v) for v in sku_demands.values()) if sku_demands else 0
    except Exception:
        total_supply_desired = sum(sku_total_supply.values())
        total_cap = sum(plant_capacity.values()) if plant_capacity else 0
        total_demand = sum(sku_demands.values()) if sku_demands else 0

    if total_cap <= 0:
        # 无有效产能约束，记录警告但保持原有分配逻辑
        logger.warning("Total plant capacity is zero or undefined; generated per-SKU supply may exceed capacity.")
    else:
        # 允许分配到 extras 的总额（capacity - minimum total demand）
        allowed_extra = max(0, total_cap - total_demand)
        current_extra = max(0, total_supply_desired - total_demand)

        if current_extra > 0 and current_extra > allowed_extra:
            # 需要缩放每个 SKU 的额外部分（仅缩放 demand 以上的部分），按比例降到 allowed_extra
            scale = allowed_extra / float(current_extra)
            for sku in list(sku_total_supply.keys()):
                base = int(sku_demands.get(sku, 0))
                extra = max(0, int(sku_total_supply[sku]) - base)
                new_extra = int(round(extra * scale))
                sku_total_supply[sku] = base + new_extra
            logger.info(
                "Scaled down SKU extras to fit plant capacity: desired=%d, cap=%d, scaled_total=%d",
                total_supply_desired, total_cap, sum(sku_total_supply.values())
            )
    # -----------------------------------------------------------------------
    plant_sku_supply = {plant: {sku: 0 for sku in skus} for plant in plants}
    for sku in skus:
        remaining = sku_total_supply[sku]
        # 使用向量化分配以避免逐单位循环带来的性能问题和潜在无限循环
        n_plants = len(plants)
        plant_indices = list(range(n_plants))
        # 构造 numpy 数组表示每个厂的剩余容量 (整数)
        plant_remain_arr = np.array([int(plant_remain_cap.get(p, 0)) for p in plants], dtype=np.int64)

        total_cap = int(plant_remain_arr.sum())
        if total_cap <= 0:
            # 无可用容量：为了保证每种 SKU 的供给 >= 需求，强制在所有工厂上平均分配（忽略当前剩余容量）
            logger.warning(f"No remaining capacity detected when allocating SKU {sku}; forcing equal distribution to guarantee supply >= demand.")
            # 将剩余量平分到所有工厂（至少分配 1 到部分工厂以覆盖 remainder）
            n_plants = len(plants) if plants else 0
            if n_plants > 0:
                per = remaining // n_plants
                extra = remaining % n_plants
                for idx, plant in enumerate(plants):
                    add = per + (1 if idx < extra else 0)
                    if add > 0:
                        plant_sku_supply[plant][sku] += int(add)
                        # 尽管 plant_remain_cap 可能为0，也更新 plant_remain_arr 以反映分配
                        plant_remain_arr[idx] = max(0, plant_remain_arr[idx] - int(add))
                # 写回剩余容量（虽然可能为负，这里保持为0下限）
                for idx, plant in enumerate(plants):
                    plant_remain_cap[plant] = int(max(0, plant_remain_arr[idx]))
                # 已经强制分配完成，跳过后续按比例分配逻辑
                continue
            else:
                logger.warning(f"No plants available to force-distribute SKU {sku!s}; leaving unassigned.")
                continue

        # 按容量比例初次分配 (向下取整)
        alloc = (remaining * plant_remain_arr) // total_cap
        # 不超过各自的剩余容量
        alloc = np.minimum(alloc, plant_remain_arr).astype(np.int64)

        # 应用初次分配
        for idx, plant in enumerate(plants):
            qty = int(alloc[idx])
            plant_sku_supply[plant][sku] = qty
            plant_remain_arr[idx] -= qty
        remaining_after_alloc = int(remaining - int(alloc.sum()))

        # 如果仍有剩余, 通过一次贪婪填充 (按剩余容量从大到小填满)
        if remaining_after_alloc > 0:
            # 找到按剩余容量从大到小排序的索引
            order = np.argsort(-plant_remain_arr)
            rem = remaining_after_alloc
            for idx in order:
                if rem <= 0:
                    break
                cap_here = int(plant_remain_arr[idx])
                if cap_here <= 0:
                    continue
                take = min(cap_here, rem)
                plant_name = plants[int(idx)]
                plant_sku_supply[plant_name][sku] += int(take)
                plant_remain_arr[idx] -= int(take)
                rem -= int(take)
            if rem > 0:
                # 强制填充剩余需求到可用工厂, 忽略当前 plant_remain_cap (保证供给 >= 需求)
                available_plants = [plant for plant in plants if sku in plant_sku_supply[plant]]
                if available_plants:
                    per = rem // len(available_plants)
                    extra = rem % len(available_plants)
                    for p in available_plants:
                        add = per + (1 if extra > 0 else 0)
                        if add > 0:
                            plant_sku_supply[p][sku] += int(add)
                            rem -= int(add)
                        if extra > 0:
                            extra -= 1
                    # 若仍有残留, 追加到第一个可用工厂
                    if rem > 0:
                        plant_sku_supply[available_plants[0]][sku] += int(rem)
                        rem = 0
                    logger.warning(f"Forced fill: assigned remaining units of SKU {sku!s} to plants to guarantee supply >= demand.")
                else:
                    logger.warning(f"No available plants to force-fill SKU {sku!s}; {rem} units remain unassigned.")
        # 将更新后的剩余容量写回 plant_remain_cap 字典
        for idx, plant in enumerate(plants):
            plant_remain_cap[plant] = int(plant_remain_arr[idx])

    storage = {plant: {sku: 0 for sku in skus} for plant in plants}
    production = {plant: {(sku, day): 0 for sku in skus for day in range(1, horizons + 1)} for plant in plants}
    for plant in plants:
        for sku in skus:
            total = plant_sku_supply[plant][sku]
            if total == 0:
                continue
            # 期初库存分配10%~30%, 其余分配到生产
            inv = max(1, int(total * random.uniform(0.1, 0.3)))
            prod_total = int(max(0, total - inv))
            storage[plant][sku] = inv
            # 将生产总量按周期分配, 确保生产总和等于 prod_total
            days = list(range(1, horizons + 1))
            prod_dist = distribute_prod(prod_total, days)
            for day in days:
                production[plant][(sku, day)] = int(prod_dist.get(day, 0))

    return storage, production

# 调用生成数据集的函数
if __name__ == '__main__':
    generate_dataset()
