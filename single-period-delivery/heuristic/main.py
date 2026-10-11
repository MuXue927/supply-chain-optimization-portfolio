import os
import csv
import time
import random
import datetime
import pandas as pd
from InputDataHeuristic import InputDataHeuristic
from HeuristicModel import HeuristicModel
from optutility import LogPrinter

random.seed(1234)

data: InputDataHeuristic
# 获取当前.py文件所在的目录，这样可以确保路径正确，即使文件被移动到其他位置也不会影响
current_dir = os.path.dirname(__file__)
# 当前.py文件所在目录的上一级目录
par_path = os.path.dirname(current_dir)
input_loc = os.path.join(par_path, 'datasets', 'single-period', 'large')
output_loc = os.path.join(par_path, 'OutPut-Heuristic', 'single-period', 'large')

log_printer = LogPrinter(time.time())

def run_model(input_file_loc=input_loc, output_file_loc=output_loc, exp_nums=100):
    medium_path = 'dataset_30'
    output_file_loc = os.path.join(output_file_loc, medium_path)
    
    global data
    
    log_printer.print_title('Loading data...')
    start_time = time.time()
    data = InputDataHeuristic(input_file_loc, output_file_loc, medium_path)
    data.load()
    end_time = time.time()
    log_printer.print(f'Loading data time: {end_time - start_time:.2f}s')
    
    
    log_printer.print(f"Start optimizing! / Now {datetime.datetime.now()}")
    
    
    objectives = {}  # 记录每次实验的配送成本
    times = {}       # 记录每次实验的运行时间
    solutions = {}   # 记录每次实验获取的可行解
    fulfills = {}    # 记录每次实验满足的经销商需求量
    
    
    for i in range(exp_nums):
        log_printer.print(f"Experiment No. {i+1}\n")
        print('-' * 100)
        
        start_time = time.time()
        
        # 创建模型
        model = HeuristicModel(data)
        
        # =================================================================
        # 优化阶段1
        log_printer.print_title('Stage 1: Optimize the supply quantity')
        
        model.optimize_stage_1()
        model.cal_supply()
    
        fulfill = model.check_fulfill()
        
        model.update_X_ij()
        # =================================================================
    
    
        # =================================================================
        # 优化阶段2
        log_printer.print_title('Stage 2: Optimize the loading process')
        
        model.optimize_stage_2()
        
        # 计算每种车型 f 的使用数量
        veh_type_nums = model.cal_vehicle_type_nums()
        
        # 计算每种车型 f 的车辆 l 的集合
        model.cal_vehicle_type_sets()
        
        # 计算第二阶段获取的完整解 X_full
        X_full = model.calculate_solution()
        
        # 检查获得的完整可行解, 是否满足所有经销商的需求
        is_fulfill = model.check_solution()
        log_printer.print(f"Have the demands of all dealers been satisfied? {is_fulfill}.")
        
        # 计算总成本
        total_cost = model.calculate_cost()
        
        # 计算使用的车辆总数
        total_veh_nums = model.calculate_vehicle_nums()
        
        for veh_type, veh_num in veh_type_nums.items():
            print(f'Vehicle type {veh_type} is used {veh_num} times.')
        # =================================================================
        
        end_time = time.time()
        log_printer.print(f'Elapsed time: {end_time - start_time:.2f}s')
        
        log_printer.print(f'Number of vehicles used: {total_veh_nums}')
        log_printer.print(f'Total cost: {total_cost: .2f}\n')
            
        # 记录可行解
        solutions[i+1, total_cost] = X_full
        # 记录目标函数值
        objectives[i+1] = total_cost
        # 记录运行时间
        times[i+1] = end_time - start_time
        # 记录满足的经销商需求量
        fulfills[i+1] = fulfill

    # 计算平均目标函数值
    avg_obj = sum(objectives.values()) / exp_nums
    
    # 计算平均运行时间
    avg_time = sum(times.values()) / exp_nums
    
    # 计算最优目标函数值
    best_obj = min(list(objectives.values()))
    
    for (exp_no, cost), sol in solutions.items():
        if cost == best_obj:  
            print(f"The best objective function value is found in experiment No. {exp_no}, with cost: {cost: .2f}")
            X_full = sol
            # 只输出最优目标函数值对应的解
            clear_output_file(output_file_loc)
            
            clear_fulfill_file(output_file_loc)
            save_fulfill(fulfills[exp_no])
            
            output_result(X_full)
            result_state(r_data=data)
            update_inv_order(X_full)
            break
    
    log_printer.print(f"Average cost: {avg_obj: .2f}, Average run time: {avg_time: .2f} s, Best Objective: {best_obj: .2f}")
    log_printer.print_title(f'The Vehicle Loading Plan is Done! {datetime.datetime.now()}')


def update_inv_order(solution):
    global data
    
    # 更新库存和需求
    for (plant, dealer, sku_id, veh_type, veh_id), qty in solution.items():
        # 更新库存
        data.sku_inv[(plant, sku_id)] -= qty
        # 更新需求
        data.demands[(dealer, sku_id)] -= qty
    
    # extra_fulfill是超额满足的经销商需求，qty=0表示恰好满足经销商需求
    # 为了便于在文件中查找，所以保留qty=0的那些行
    extra_fulfill = {(dealer, sku_id): -qty for (dealer, sku_id), qty in data.demands.items() if qty <= 0}
    
    # sku_inv_left是生产基地中剩余的库存，qty=0表示没有剩余库存
    # 为了便于在文件中查找，所以保留qty=0的那些行
    sku_inv_left = {(plant, sku_id): qty for (plant, sku_id), qty in data.sku_inv.items() if qty >= 0}
    

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


def clear_fulfill_file(output_file_loc=output_loc):
    """清理满足经销商需求的文件，在每个数据集的文件夹中创建新的结果文件"""
    if not os.path.exists(output_file_loc):
        os.makedirs(output_file_loc)
    output_file_name = 'fulfill.csv'
    header = ['client_code', 'product_code', 'volume']
    with open(os.path.join(output_file_loc, output_file_name), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

def clear_output_file(output_file_loc=output_loc):
    """清理输出文件，在每个数据集的文件夹中创建新的结果文件"""
    if not os.path.exists(output_file_loc):
        os.makedirs(output_file_loc)
    output_file_name = 'opt_result.csv'
    header = ['Index', 'day', 'plant_code', 'client_code', 'product_code', 'vehicle_id', 'vehicle_type', 'qty']
    with open(os.path.join(output_file_loc, output_file_name), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        
def save_fulfill(fulfill):
    """保存满足经销商需求的结果"""
    if not os.path.exists(data.output_file_loc):
        os.makedirs(data.output_file_loc)
    
    output_file = os.path.join(data.output_file_loc, 'fulfill.csv')
    df_fulfill = pd.DataFrame({
        (dealer, sku_id, qty) for (dealer, sku_id), qty in fulfill.items()
    }, columns=['client_code', 'product_code', 'volume'])
    df_fulfill.to_csv(output_file, index=False)        


def output_result(solution):
    """在对应数据集的文件夹中保存结果"""
    output_file = os.path.join(data.output_file_loc, 'opt_result.csv')
    df_result = create_dataframe(solution)
    # header=False表示不将dataframe的列名写入文件
    df_result.to_csv(output_file, mode='a', header=False)

def create_dataframe(solution):
    rows = []
    
    for key, qty in solution.items():
        plant, dealer, sku, veh_type, veh_id = key
        rows.append({
            'day': data.day,
            'plant_code': plant,
            'client_code': dealer,
            'product_code': sku,
            'vehicle_id': veh_id,
            'vehicle_type': veh_type,
            'qty': qty
        })
    return pd.DataFrame(rows)


    
def result_state(r_data: InputDataHeuristic):
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
        log_printer.print("Warning! The volume of SKUs loaded in some vehicles has exceeded their capacities.", color='Bold red')
        df_exceed = df_tmp[df_tmp["occupied_size"] > df_tmp["carry_standard_size"]]

        df_exceed_info = df_exceed.loc[:, ['vehicle_id', 'vehicle_type', 'carry_standard_size', 'occupied_size']]
        df_exceed_info['exceeded_volume'] = df_exceed_info['occupied_size'] - df_exceed_info['carry_standard_size']

        df_exceed_info.to_csv(os.path.join(output_location, "extra_volume.csv"), index=False)
        log_printer.print("The relevant information of this warning has been recorded.")
    

if __name__ == '__main__':
    
    run_model()
