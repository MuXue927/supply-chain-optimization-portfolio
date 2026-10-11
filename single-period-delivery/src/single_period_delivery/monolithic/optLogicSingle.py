# 定义单周期优化模型的执行逻辑
# 并将模型的求解结果保存到csv文件
import os
import csv
import time
import datetime
import pandas as pd
from gurobipy import GRB

from .MonolithicModelSingle import MonolithicModelSingle, log_printer
from .InputDataSingle import InputDataSingle

data: InputDataSingle

project_root = os.getcwd()
input_loc = os.path.join(project_root, 'datasets', 'single-period', 'large')
output_loc = os.path.join(project_root, 'OutPut', 'single-period', 'large')

def run_model(
    input_file_loc=input_loc,
    output_file_loc=output_loc,
    dataset_name='dataset_1',
):
    medium_path = dataset_name
    output_file_loc = os.path.join(output_file_loc, medium_path)
    global data

    log_printer.print("Loading Data...")
    start_time = time.time()
    data = InputDataSingle(input_file_loc, output_file_loc, medium_path)
    data.load()
    end_time = time.time()
    log_printer.print(f"Elapsed time: {end_time - start_time: .2f}s")
    log_printer.print(f"The maximum number of vehicles available is {data.max_veh_num}")

    clear_output_file(output_file_loc)
    # ========================================================================================
    log_printer.print("Constructing model...")
    start_time = time.time()
    model = MonolithicModelSingle(data=data)
    end_time = time.time()
    log_printer.print(f"Elapsed time: {end_time - start_time: .2f}s")
    
    flag = model.run()
    if not flag:
        return False
    output_result(model)
    result_state(r_data=data)
    log_printer.print_title(f'The Vehicle Loading Plan is Done! {datetime.datetime.now()}')
    # ========================================================================================

    # # 创建gurobi求解日志文件的保存路径
    # log_path = os.path.join('solve-logs', 'single-period', 'small', 'dataset_3')
    # # 如果路径不存在，需要先创建路径
    # if not os.path.exists(log_path):
    #     os.makedirs(log_path)
    # # 指定要保存的日志文件
    # log_file = os.path.join(log_path, 'Windows PowerShell.txt')
    # # 保存原始的标准输出和错误输出
    # origin_stdout = sys.stdout
    # origin_stderr = sys.stderr
    # try:
    #     # 重定向标准输出和错误输出到文件
    #     with open(log_file, 'w') as file:
    #         sys.stdout = sys.stderr = file
    #         model = MonolithicModelSingle(data=data)
    #         flag = model.run()
    #         output_result(model)
    #         if not flag:
    #             return False
    #         result_state(r_data=data)
    #         log_printer.print_title(f'The Vehicle Loading Plan is Done! {datetime.datetime.now()}')
    #         # 强制将缓冲区里面的内容写入到文件
    #         sys.stdout.flush()
    #         sys.stderr.flush()
    # finally:  # 定义清理行为，无论是否发生异常，finally块中的代码都会被执行
    #     # 恢复原始的标准输出和错误输出
    #     sys.stdout = origin_stdout
    #     sys.stderr = origin_stderr
    # # 打印日志文件的保存位置
    # log_printer.print(f'Log file has been saved to the location: {log_file}')
    # # 在finally块中的return语句会覆盖try中的return语句
    # return flag


def clear_output_file(output_file_loc=output_loc):
    output_file_loc = os.path.join(output_file_loc)
    if not os.path.exists(output_file_loc):
        os.makedirs(output_file_loc)
    output_file_name = 'result.csv'
    header = ['Index', 'day', 'plant_code', 'client_code', 'product_code', 'vehicle_id', 'vehicle_type', 'qty']
    with open(os.path.join(output_file_loc, output_file_name), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

def output_result(model: MonolithicModelSingle):
    output_file = os.path.join(data.output_file_loc, 'result.csv')

    df_result = create_dataframe(model)
    # header=False表示不将dataframe的列名写入文件
    df_result.to_csv(output_file, mode='a', header=False)

def create_dataframe(model: MonolithicModelSingle):
    rows = []
    if model.model.Status == GRB.OPTIMAL:  # 确保只有在模型达到最优时，才访问变量的Xn属性，获得最优解
        for (plant, dealer, sku_id, v_id), var in model.x_ijkl.items():
            # 只有在使用车辆 l 且从生产基地 i 到经销商 j 的车辆 l 上装载的SKU k的数量大于0时，才保存结果
            if var.Xn > 0 and model.z_l[v_id].Xn > 0 and model.y_ijl[plant, dealer, v_id].Xn > 0:
                rows.append({
                    'day': model.day,
                    'plant_code': plant,
                    'client_code': dealer,
                    'product_code': sku_id,
                    'vehicle_id': int(v_id),
                    'vehicle_type': model.veh_type[int(v_id)],
                    'qty': round(var.Xn)
                })
    return pd.DataFrame(rows)

def result_state(r_data: InputDataSingle):
    dataset = r_data.dataset_name
    output_location = r_data.output_file_loc
    input_location = os.path.join(r_data.input_file_loc, dataset)

    df_result = pd.read_csv(os.path.join(output_location, 'result.csv'))
    del df_result['Index']

    df_sku_size = pd.read_csv(os.path.join(input_location, 'product_size.csv'))

    # after merge, df_result has the 'standard_size' column, that means sku_size
    df_result = pd.merge(df_result, df_sku_size, on=['product_code'])

    df_all_veh = pd.read_csv(os.path.join(input_location, 'vehicle.csv'))
    # after merge, df_result has the 'carry_standard_size' column, 'min_standard_size' column
    # and 'cost_to_use' column
    df_result = pd.merge(df_result, df_all_veh, on=['vehicle_type'])
    del df_result['cost_to_use']

    df_result['occupied_size'] = df_result['qty'] * df_result['standard_size']
    df_result.to_csv(os.path.join(output_location, 'stat0.csv'))

    df_stat1 = df_result
    del df_stat1['product_code']
    del df_stat1['standard_size']

    df_tmp = df_stat1.groupby(['plant_code', 'client_code', 'vehicle_id']).agg(
        {'vehicle_type': 'first', 'qty': 'sum', 'carry_standard_size': 'first',
         'min_standard_size': 'first', 'occupied_size': 'sum'})

    df_tmp.to_csv(os.path.join(output_location, 'stat1.csv'))

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
