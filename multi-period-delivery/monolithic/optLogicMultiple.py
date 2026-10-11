# 定义多周期优化模型的执行逻辑
# 并将模型的求解结果保存到csv文件
import os
import sys
import csv
import datetime
import pandas as pd
from .MonolithicModelMultiple import *

data: InputDataMultiple

# 获取当前.py文件所在的目录，这样可以确保路径正确，即使文件被移动到其他位置也不会影响
current_dir = os.path.dirname(__file__)
# 当前.py文件所在目录的上一级目录
par_path = os.path.dirname(current_dir)
input_loc = os.path.join(par_path, 'datasets', 'multiple-periods', 'small')
output_loc = os.path.join(par_path, 'outputs', 'multiple-periods', 'small')

    

def run_model(data: InputDataMultiple, model: MonolithicModelMultiple, 
              time_limit: int = None, out_dir: str = None, write_outputs: bool = True) -> dict:  # type: ignore
    """
    Run a prepared Monolithic model using already-constructed InputDataMultiple (data)
    and MonolithicModelMultiple (model). Return a metadata dict suitable for
    run_experiments.run_single to merge into run_metadata.json.

    Parameters
    - data: InputDataMultiple (already loaded)
    - model: MonolithicModelMultiple (constructed with data)
    - time_limit: optional wall-clock time limit in seconds (will be applied to Gurobi params where possible)
    - out_dir: optional output directory to write result CSVs (if None, uses data.output_file_loc)
    - write_outputs: if True, will call clear_output_file, output_result, result_state to produce CSVs

    Returns:
    - meta (dict): keys include status, wall_seconds, final_obj, best_bound, gap_pct,
                   gurobi_runtime, iterations, output_files, warnings, error (if any)
    """
    import time
    import os

    t0 = time.time()
    meta = {}
    warnings = []
    out_files = {}

    # apply time limit if provided
    if time_limit is not None:
        try:
            model.model.Params.TimeLimit = int(time_limit)
        except Exception:
            try:
                # some wrappers expose param_time_limit on data
                data.param_time_limit = int(time_limit)
            except Exception:
                pass

    # run model
    try:
        ok, meta = model.run()
        status = meta.get("status", "unknown")
        meta["status"] = status
    except Exception as e:
        meta["status"] = "runtime_error"
        meta["error"] = str(e)
        meta["wall_seconds"] = time.time() - t0
        return meta

    meta["wall_seconds"] = time.time() - t0

    # try to extract useful solver metrics in a defensive way
    final_obj = None
    best_bound = None
    gurobi_runtime = None
    iterations = None
    mip_gap = None
    node_count = None
    vehicle_count = None

    try:
        # ObjVal
        if hasattr(model.model, "ObjVal"):
            try:
                final_obj = float(model.model.ObjVal)
            except Exception:
                final_obj = None
    except Exception:
        final_obj = None

    try:
        # ObjBound via getAttr or attribute
        try:
            best_bound = model.model.getAttr("ObjBound")
        except Exception:
            best_bound = getattr(model.model, "ObjBound", None)
        if best_bound is not None:
            try:
                best_bound = float(best_bound)
            except Exception:
                pass
    except Exception:
        best_bound = None

    try:
        gurobi_runtime = getattr(model.model, "Runtime", None)   # 求解器运行时间
        if gurobi_runtime is not None:
            try:
                gurobi_runtime = float(gurobi_runtime)
            except Exception:
                pass
    except Exception:
        gurobi_runtime = None

    try:
        simplex_iters = getattr(model.model, "IterCount", None)     # gurobi求解模型调用的单纯形迭代次数
        barrier_iters = getattr(model.model, "BarIterCount", None)  # gurobi求解模型调用的内点法迭代次数
        
        iterations = 0  # 总迭代次数
        if simplex_iters is not None:
            iterations += int(simplex_iters)
        if barrier_iters is not None:
            iterations += int(barrier_iters)
        
        if iterations is not None:
            try:
                iterations = int(iterations)
            except Exception:
                pass
    except Exception:
        iterations = None

    try:
        mip_gap = getattr(model.model, "MIPGap", None)  # 对于 MIP模型, 该属性表示当前的 MIP gap
    except Exception:
        mip_gap = None

    try:
        node_count = getattr(model.model, "NodeCount", None)  # 分支定界法中处理的节点数
        if node_count is not None:
            try:
                node_count = int(node_count)
            except Exception:
                pass
    except Exception:
        node_count = None

    try:
        # 与模型口径保持一致: 车辆数 = sum_{v,t} z[v,t]
        z_vars = getattr(model, "z_lt", None)
        if z_vars is not None:
            vehicle_count = int(
                round(
                    sum(
                        float(getattr(var, "X", getattr(var, "Xn", 0.0)))
                        for var in z_vars.values()
                    )
                )
            )
    except Exception:
        vehicle_count = None

    if final_obj is not None:
        meta["final_obj"] = final_obj
    if best_bound is not None:
        meta["best_bound"] = best_bound
    if gurobi_runtime is not None:
        meta["gurobi_runtime"] = gurobi_runtime
    if iterations is not None:
        meta["iterations"] = iterations
    if mip_gap is not None:
        meta["mip_gap"] = mip_gap
        # compute gap_pct for minimization problems when possible
        meta["gap_pct"] = 100.0 * mip_gap
    if node_count is not None:
        meta["node_count"] = node_count
    if vehicle_count is not None:
        meta["vehicle_count"] = int(vehicle_count)


    # write outputs if requested
    try:
        if write_outputs:
            target_out = out_dir or getattr(data, "output_file_loc", None)
            if target_out is None:
                warnings.append("no_output_dir_provided")
            else:
                os.makedirs(target_out, exist_ok=True)
                try:
                    clear_output_file(target_out)
                except Exception as e:
                    # try legacy signature: clear_output_file() without args
                    try:
                        clear_output_file()
                    except Exception:
                        warnings.append(f"clear_output_failed:{e}")
                try:
                    # output_result expects the model object
                    output_result(model)
                except Exception as e:
                    warnings.append(f"output_result_failed:{e}")
                try:
                    result_state(r_data=data)
                except Exception as e:
                    warnings.append(f"result_state_failed:{e}")

                out_files["result_csv"] = os.path.join(target_out, "result.csv")
                out_files["stat0_csv"] = os.path.join(target_out, "stat0.csv")
                out_files["stat1_csv"] = os.path.join(target_out, "stat1.csv")
    except Exception as e:
        warnings.append(f"write_outputs_error:{e}")

    if out_files:
        meta["output_files"] = out_files
    if warnings:
        meta["warnings"] = warnings

    return meta


def run_model_from_paths(input_file_loc=input_loc, output_file_loc=output_loc, dataset_name="dataset_1",
                         time_limit: int = None, write_outputs: bool = True) -> dict:  # type: ignore
    """
    Backwards-compatible wrapper that loads InputDataMultiple and constructs
    MonolithicModelMultiple, then calls run_model(data, model, ...).
    """
    log_printer.print("Loading Data...")
    t0 = time.time()
    data = InputDataMultiple(input_file_loc, output_file_loc, dataset_name)
    data.load()
    log_printer.print(f"Elapsed time: {time.time() - t0: .2f}s")
    log_printer.print(f"The maximum number of vehicles available is {getattr(data, 'max_veh_num', 'unknown')}")

    log_printer.print("Constructing model...")
    t0 = time.time()
    model = MonolithicModelMultiple(data=data)
    log_printer.print(f"Elapsed time: {time.time() - t0: .2f}s")

    # call general runner
    return run_model(data, model, time_limit=time_limit, out_dir=os.path.join(output_file_loc, dataset_name),
                     write_outputs=write_outputs)

    # # 创建gurobi求解日志文件的保存路径
    # log_path = os.path.join('solve-logs', 'multiple-periods', 'small', 'dataset_1')
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
    #         model = MonolithicModelMultiple(data=data)
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
    #     # 打印日志文件的保存位置
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

def output_result(model: MonolithicModelMultiple):
    output_file = os.path.join(model.data.output_file_loc, 'result.csv')

    df_result = create_dataframe(model)
    # header=False表示不将dataframe的列名写入文件
    df_result.to_csv(output_file, mode='a', header=False)

def create_dataframe(model: MonolithicModelMultiple):
    rows = []
    if model.model.Status == GRB.OPTIMAL:  # 确保只有在模型达到最优时，才访问变量的Xn属性，获得最优解
        for (plant, dealer, sku_id, v_id, t), var in model.x_ijklt.items():
            # 只有在使用车辆 l 且从生产基地 i 到经销商 j 的车辆 l 上装载的SKU k的数量大于0时，才保存结果
            if var.Xn > 0 and model.z_lt[v_id, t].Xn > 0 and model.y_ijlt[plant, dealer, v_id, t].Xn > 0:
                rows.append({
                    'day': t,
                    'plant_code': plant,
                    'client_code': dealer,
                    'product_code': sku_id,
                    'vehicle_id': int(v_id),
                    'vehicle_type': model.veh_type[int(v_id)],
                    'qty': round(var.Xn)
                })
    return pd.DataFrame(rows)

def result_state(r_data: InputDataMultiple):
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
