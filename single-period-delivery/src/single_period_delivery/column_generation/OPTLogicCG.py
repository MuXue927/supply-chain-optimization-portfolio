# 定义列生成算法的执行逻辑
# 并将求解结果保存到csv文件
import os
import csv
import time
import datetime
import random
import pandas as pd
from typing import Dict, Tuple
from dataclasses import field
import gurobipy as gp
from gurobipy import GRB

from .masterModel import MasterModel, log_printer
from .subModel import SubModel
from .InputDataCG import InputDataCG
from .vehicle import Vehicle
from .optutility import POSITIVEEPS

data: InputDataCG
# fact_id
masterM: MasterModel
# (fact_id, dealer_id, vehicle_type)
subM: Dict[Tuple[str, str, str], SubModel] = field(default_factory=dict)

project_root = os.getcwd()
input_loc = os.path.join(project_root, 'datasets', 'single-period', 'large')
output_loc = os.path.join(project_root, 'OutPut-CG', 'single-period', 'large')


def run_model(
    input_file_loc=input_loc,
    output_file_loc=output_loc,
    dataset_name='dataset_30',
):
    medium_path = dataset_name
    output_file_loc = os.path.join(output_file_loc, medium_path)
    global data, subM

    log_printer.print("Loading Data...")
    start_time = time.time()
    data = InputDataCG(input_file_loc, output_file_loc, medium_path)
    data.load()
    end_time = time.time()
    log_printer.print(f"Elapsed time: {end_time - start_time: .2f}s")

    clear_output_file(output_file_loc)
    # =========================================================
    for day_id in range(1, data.horizons + 1):
        # 初始化主模型
        masterM = MasterModel(data)
        # 初始化子模型
        subM = {}
        triple_plant_dealer_veh = find_triple_plant_dealer_veh(masterM)
        for (plant, dealer, veh_type) in triple_plant_dealer_veh:
            subM[plant, dealer, veh_type] = SubModel(plant, dealer, veh_type, data)

        # 执行优化过程
        log_printer.print(f"Start Solving! Day {day_id} / Now {datetime.datetime.now()}")
        # 使用列生成算法，获取主模型松弛问题的可行解
        return_value = opt_master_model(model=masterM)
        if not return_value:
            return False
        # 更改主模型的决策变量类型，重新优化，拿到主模型的整数解
        return_value = construct_int_solution(model=masterM)
        if not return_value:
            return False
    pass
    log_printer.print_title(f'The Vehicle Loading Plan is Done! {datetime.datetime.now()}')


def find_triple_plant_dealer_veh(model: MasterModel):
    global data
    target_dealers = {dealer_id for dealer_id, sku_id in model.cons_dealer_demands}
    duet_dealer_veh = [
        (dealer_id, veh_type) for dealer_id in target_dealers for veh_type in data.all_veh_types
    ]
    triple_plant_dealer_veh = [
        (plant, dealer, veh_type) for (plant, dealer) in data.construct_supply_chain()
        for (dealer_id, veh_type) in duet_dealer_veh if dealer == dealer_id
    ]
    return triple_plant_dealer_veh

def opt_master_model(model: MasterModel):
    global data, subM
    log_printer.print("Constructing load patterns...")
    while True:
        model.model.setParam('OutPutFlag', 0)
        if not model.run():
            return False

        PIs_1 = {key: model.cons_dealer_demands[key].Pi for key in model.cons_dealer_demands}
        PIs_2 = {key: model.cons_sku_supply[key].Pi for key in model.cons_sku_supply}
        PIs_3 = {key: model.cons_fact_sku_inv[key].Pi for key in model.cons_fact_sku_inv}

        # 主模型中存在的装载模式的总数量
        num_patterns = 0

        triple_plant_dealer_veh = find_triple_plant_dealer_veh(model)
        random.shuffle(triple_plant_dealer_veh)

        for (plant, dealer, veh_type) in triple_plant_dealer_veh:
            sub_model = subM[plant, dealer, veh_type]
            pis1 = gp.tupledict({sku: PIs_1[dealer, sku] for sku in sub_model.require_skus})
            pis2 = gp.tupledict({sku: PIs_2[sku] for sku in sub_model.require_skus})
            pis3 = gp.tupledict({sku: PIs_3[plant, sku] for sku in sub_model.require_skus})
            # 子模型的目标函数
            sub_model.obj =  (sub_model.d_xs.prod(pis1) + sub_model.d_xs.prod(pis2) + sub_model.d_xs.prod(pis3) -
                              data.veh_type_cost[veh_type] - sub_model.p_f)
            if not sub_model.run():
                continue
            if sub_model.model.ObjVal >= POSITIVEEPS:
                load_pattern = sub_model.get_load_pattern()
                model.insert_pattern(plant, dealer, veh_type, data.veh_type_cost[veh_type], load_pattern)  # type: ignore
                num_patterns += 1
                continue
            if num_patterns > data.param_pattern_batch:
                break

        if num_patterns == 0:
            break
        if model.num_patterns > data.param_trigger_shrink_level:
            model.run()
            if model.get_alpha_size() == 0 and model.get_beta_size() == 0 and model.get_gamma_size() == 0:
                return True
            model.shrink()
    return True

def construct_int_solution(model: MasterModel):
    global data, subM
    log_printer.print("Initializing Feasible Solution...")
    # 关闭主模型中的人工变量 alpha
    model.deactivate_alpha()

    complete_pattern(model)
    model.model.setParam(GRB.Param.TimeLimit, data.param_time_limit)
    log_printer.print("Optimizing restricted master model...")
    model.model.Params.OutputFlag = 1
    model.run()

    # 更改主模型中决策变量的类型，拿到整数解
    model.change_var_type()
    model.run()
    if model.model.Status in (GRB.INF_OR_UNBD, GRB.INFEASIBLE, GRB.UNBOUNDED):
        log_printer.print("Exception: Master Model Is Infeasible or Unbounded!")
        return False

    # 如果模型中还存在人工变量，需要通过查看取值非0的人工变量，判断哪些约束不能被满足，进一步诊断原因
    if model.get_beta_size() != 0 or model.get_gamma_size() != 0:
        log_printer.print("Artificial variables exists in master model.")
        # 查看人工变量的取值
        for sku_id, var in model.d_beta.items():
            if var.X > 0:
                log_printer.print(f"d_beta{[sku_id]} = {var.X}")
        for (plant, sku_id), var in model.d_gamma.items():
            if var.X > 0:
                log_printer.print(f"d_gamma{[plant, sku_id]} = {var.X}")

    update_inv_order(rmp=model)
    # 保存计算结果
    output_result(model)
    result_state(r_data=data)
    return True

def complete_pattern(model: MasterModel):
    global data, subM
    return_value = 0
    for (plant, dealer) in data.construct_supply_chain():
        # 某个生产基地可以提供的SKU集合
        sku_set1 = {sku_id for sku_id in data.skus_plant[plant]}
        # 某个经销商需要的SKU集合
        sku_set2 = {sku_id for sku_id in data.skus_dealer[dealer]}
        sku_set = sku_set1 & sku_set2

        # 在主模型中存在向特定经销商供货的约束，但是约束条件左侧没有决策变量的sku集合，称为没有决策变量支撑的sku
        complement_sku = {}
        for sku in sku_set:
            cons = model.cons_dealer_demands[dealer, sku]
            LHS = model.model.getRow(cons)
            if LHS.size() > 1:
                continue
            complement_sku[sku] = data.demands[dealer, sku]

        if len(complement_sku) > 0:
            # 取得最大容量的车型
            max_veh_type = max(data.all_veh_types, key=float)
            selected_vehicle = Vehicle(plant, dealer, max_veh_type, data)

            for sku_id, qty in complement_sku.items():
                while qty > 0:
                    qty = selected_vehicle.load(sku_id, qty)
                    if qty > 0:
                        model.insert_pattern(plant, dealer, max_veh_type,
                                            data.veh_type_cost[max_veh_type], selected_vehicle.cargo)
                        return_value += 1
                        selected_vehicle.clear()

            if not selected_vehicle.is_empty():
                model.insert_pattern(plant, dealer, max_veh_type,
                                    data.veh_type_cost[max_veh_type], selected_vehicle.cargo)
                return_value += 1
                selected_vehicle.clear()

    return return_value

def clear_output_file(output_file_loc=output_loc):
    """清理输出文件，在每个数据集的文件夹中创建新的结果文件"""
    if not os.path.exists(output_file_loc):
        os.makedirs(output_file_loc)
    output_file_name = 'opt_result.csv'
    header = ['Index', 'day', 'plant_code', 'client_code', 'product_code', 'vehicle_id', 'vehicle_type', 'qty']
    with open(os.path.join(output_file_loc, output_file_name), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

def output_result(model: MasterModel):
    """在对应数据集的文件夹中保存结果"""
    output_file = os.path.join(data.output_file_loc, 'opt_result.csv')

    df_result = create_dataframe(model)
    # header=False表示不将dataframe的列名写入文件
    df_result.to_csv(output_file, mode='a', header=False)

def update_inv_order(rmp:MasterModel):
    global data, subM
    for (plant, dealer, veh_type, pattern_id), var1 in rmp.d_xs.items():
        if var1.X > 0:
            pattern_name = var1.VarName
            load_pattern = rmp.pattern_content[pattern_name]
            for sku_id, qty in load_pattern.items():
                data.sku_inv[plant, sku_id] -= round(var1.X * qty)
                data.demands[dealer, sku_id] -= round(var1.X * qty)

    # extra_fulfill是超额满足的经销商需求，qty=0表示恰好满足经销商需求
    # 为了便于在文件中查找，所以保留qty=0的那些行
    extra_fulfill = {(dealer, sku_id): -qty for (dealer, sku_id), qty in data.demands.items() if qty <= 0}
    data.demands = {(dealer, sku_id): qty for (dealer, sku_id), qty in data.demands.items() if qty > 0}

    # sku_inv_left是生产基地中剩余的库存，qty=0表示没有剩余库存
    # 为了便于在文件中查找，所以保留qty=0的那些行
    sku_inv_left = {(plant, sku_id): qty for (plant, sku_id), qty in data.sku_inv.items() if qty >= 0}
    data.sku_inv = {(plant, sku_id): qty for (plant, sku_id), qty in data.sku_inv.items() if qty > 0}

    data.df_order = pd.DataFrame({
        (dealer, sku_id, qty) for (dealer, sku_id), qty in data.demands.items()
    }, columns=['dealer_id', 'sku_id', 'order_qty'])
    data.df_sku_inv = pd.DataFrame({
        (plant, sku_id, qty) for (plant, sku_id), qty in data.sku_inv.items()
    }, columns=['fact_id', 'sku_id', 'inv'])

    df_extra_fulfill = pd.DataFrame({
        (dealer, sku_id, qty) for (dealer, sku_id), qty in extra_fulfill.items()
    }, columns=['client_code', 'product_code', 'volume'])
    df_sku_inv_left = pd.DataFrame({
        (plant, sku_id, qty) for (plant, sku_id), qty in sku_inv_left.items()
    }, columns=['plant_code', 'product_code', 'volume'])

    output_file1 = os.path.join(data.output_file_loc, 'extra_fulfill.csv')
    output_file2 = os.path.join(data.output_file_loc, 'sku_inv_left.csv')
    df_extra_fulfill.to_csv(output_file1, index=False)
    df_sku_inv_left.to_csv(output_file2, index=False)


def create_dataframe(rmp: MasterModel):
    rows = []
    # 构建每一种装载模式对应的车辆编号
    pattern_veh_id_map = create_pt_veh_id_map(rmp)
    for (plant, dealer, veh_type, pattern_id), var1 in rmp.d_xs.items():
        if var1.X > 0:
            pattern_name = var1.VarName
            if len(pattern_veh_id_map[pattern_name]) != 0:
                veh_id = pattern_veh_id_map[pattern_name].pop()
                load_pattern = rmp.pattern_content[pattern_name]
                for sku_id, qty in load_pattern.items():
                    rows.append({
                        'day': rmp.day,
                        'plant_code': plant,
                        'client_code': dealer,
                        'product_code': sku_id,
                        'vehicle_id': veh_id,
                        'vehicle_type': veh_type,
                        'qty': qty
                    })
    return pd.DataFrame(rows)

def create_pt_veh_id_map(rmp: MasterModel):
    """
    创建每一种装载模式对应的车辆编号的字典
    :param rmp: 主模型
    """
    # pattern_nums用于记录每一种装载模式的使用数量
    pattern_nums = {}
    for key, var in rmp.d_xs.items():
        if var.X > 0:
            pattern_nums[var.VarName] = var.X

    # 主模型计算出的使用的车辆总数
    num_veh_used = sum(var.X for key, var in rmp.d_xs.items() if var.X > 0)
    # 根据车辆总数，构建车辆编号列表
    veh_ids = [veh_id for veh_id in range(1, round(num_veh_used) + 1)]

    cur_veh_idx = 0  # 车辆编号在veh_ids中的索引
    # 为所有装载模式初始化一个空的车辆编号集合
    pt_veh_id_map = {pt: set() for pt in pattern_nums.keys()}

    for pt, pt_num in pattern_nums.items():
        # 当前装载模式对应的车辆编号集合
        pt_vehicles = set()
        if cur_veh_idx < len(veh_ids):
            pt_num = int(pt_num)
            for _ in range(pt_num):
                # 优先使用较小的车辆编号
                veh_id = veh_ids[cur_veh_idx]
                pt_vehicles.add(veh_id)
                cur_veh_idx += 1
        pt_veh_id_map[pt] = pt_vehicles
    return pt_veh_id_map


def result_state(r_data: InputDataCG):
    """处理并保存统计结果到对应的数据集文件夹"""
    dataset = r_data.dataset_name
    output_location = r_data.output_file_loc
    input_location = os.path.join(r_data.input_file_loc, dataset)

    df_result = pd.read_csv(os.path.join(output_location, 'opt_result.csv'))
    del df_result['Index']

    df_result.to_csv(os.path.join(output_location, 'opt_result.csv'), index=False)

    df_sku_size = pd.read_csv(os.path.join(input_location, 'product_size.csv'))

    # after merge, df_result has the 'standard_size' column, that means sku_size
    df_result = pd.merge(df_result, df_sku_size, on=['product_code'])

    df_all_veh = pd.read_csv(os.path.join(input_location, 'vehicle.csv'))
    # after merge, df_result has the 'carry_standard_size' column, 'min_standard_size' column
    # and 'cost_to_use' column
    df_result = pd.merge(df_result, df_all_veh, on=['vehicle_type'])
    del df_result['cost_to_use']

    df_result['occupied_size'] = df_result['qty'] * df_result['standard_size']
    df_result.to_csv(os.path.join(output_location, 'opt_details.csv'), index=False)

    df_stat1 = df_result.copy()
    df_stat1 = df_stat1.drop(['product_code', 'standard_size'], axis=1)

    df_tmp = df_stat1.groupby(['plant_code', 'day', 'client_code', 'vehicle_id']).agg(
        {'vehicle_type': 'first', 'qty': 'sum', 'carry_standard_size': 'first',
         'min_standard_size': 'first', 'occupied_size': 'sum'})

    df_tmp.to_csv(os.path.join(output_location, 'opt_summary.csv'))

    df_tmp = df_tmp.reset_index()
    # 判断是否所有车辆中装载的SKU体积满足车辆容量约束
    is_cap_exceed = pd.Series(df_tmp['occupied_size'] <= df_tmp['carry_standard_size'])
    if is_cap_exceed.all():  # 所有值都为True，则返回True
        log_printer.print("Congratulations! "
                          "The volume of SKUs loaded in each vehicle is equal or smaller than vehicle's capacity.")
    else:
        log_printer.print("Warning! The volume of SKUs loaded in some vehicles has exceeded their capacities.")
        df_exceed = df_tmp[df_tmp["occupied_size"] > df_tmp["carry_standard_size"]]

        df_exceed_info = df_exceed.loc[:, ['vehicle_id', 'vehicle_type', 'carry_standard_size', 'occupied_size']]
        df_exceed_info['exceeded_volume'] = df_exceed_info['occupied_size'] - df_exceed_info['carry_standard_size']

        df_exceed_info.to_csv(os.path.join(output_location, "extra_volume.csv"), index=False)
        log_printer.print("The relevant information of this warning has been recorded.")

