import csv
import datetime
import os
import random
import time

import numpy as np
import pandas as pd

from .HeuristicModel import HeuristicModel
from .InputDataHeuristic import InputDataHeuristic
from .optutility import POSITIVEEPS, LogPrinter

random.seed(1234)

data: InputDataHeuristic
project_root = os.getcwd()
input_loc = os.path.join(project_root, 'datasets', 'single-period', 'large')
output_loc = os.path.join(project_root, 'OutPut-Heuristic', 'single-period', 'large')

log_printer = LogPrinter(time.time())

def run_model(
    input_file_loc=input_loc,
    output_file_loc=output_loc,
    exp_nums=100,
    dataset_name='dataset_30',
):
    medium_path = dataset_name
    output_file_loc = os.path.join(output_file_loc, medium_path)
    
    global data
    
    log_printer.print_title('Loading data...')
    start_time = time.time()
    data = InputDataHeuristic(input_file_loc, output_file_loc, medium_path)
    data.load()
    end_time = time.time()
    log_printer.print(f'Loading data time: {end_time - start_time:.2f}s')
    
    
    log_printer.print(f"Start optimizing! / Now {datetime.datetime.now()}")
    
    
    objectives = {}  # Delivery cost recorded for each experiment.
    times = {}       # Runtime recorded for each experiment.
    solutions = {}   # Feasible solution obtained in each experiment.
    fulfills = {}    # Dealer demand satisfied in each experiment.
    
    
    for i in range(exp_nums):
        log_printer.print(f"Experiment No. {i+1}\n")
        print('-' * 100)
        
        start_time = time.time()
        
        # Create the model.
        model = HeuristicModel(data)
        
        # =================================================================
        # Optimization Stage 1.
        log_printer.print_title('Stage 1: Optimize the supply quantity')
        
        model.optimize_stage_1()
        model.cal_supply()
    
        fulfill = model.check_fulfill()
        
        model.update_X_ij()
        # =================================================================
    
    
        # =================================================================
        # Optimization Stage 2.
        log_printer.print_title('Stage 2: Optimize the loading process')
        
        model.optimize_stage_2()
        
        # Compute the number of vehicles used for each vehicle type f.
        veh_type_nums = model.cal_vehicle_type_nums()
        
        # Compute the set of vehicle identifiers l for each vehicle type f.
        model.cal_vehicle_type_sets()
        
        # Compute the complete feasible solution X_full from Stage 2.
        X_full = model.calculate_solution()
        
        # Check whether the complete feasible solution satisfies all dealer demand.
        is_fulfill = model.check_solution()
        log_printer.print(f"Have the demands of all dealers been satisfied? {is_fulfill}.")
        
        # Compute the total cost.
        total_cost = model.calculate_cost()
        
        # Compute the total number of vehicles used.
        total_veh_nums = model.calculate_vehicle_nums()
        
        for veh_type, veh_num in veh_type_nums.items():
            print(f'Vehicle type {veh_type} is used {veh_num} times.')
        # =================================================================
        
        end_time = time.time()
        log_printer.print(f'Elapsed time: {end_time - start_time:.2f}s')
        
        log_printer.print(f'Number of vehicles used: {total_veh_nums}')
        log_printer.print(f'Total cost: {total_cost: .2f}\n')
            
        # Record the feasible solution.
        solutions[i+1, total_cost] = X_full
        # Record the objective value.
        objectives[i+1] = total_cost
        # Record the runtime.
        times[i+1] = end_time - start_time
        # Record satisfied dealer demand.
        fulfills[i+1] = fulfill

    # Compute the average objective value.
    avg_obj = sum(objectives.values()) / exp_nums
    
    # Compute the average runtime.
    avg_time = sum(times.values()) / exp_nums
    
    # Compute the best objective value.
    best_obj = min(list(objectives.values()))
    
    for (exp_no, cost), sol in solutions.items():
        if cost == best_obj:  
            print(f"The best objective function value is found in experiment No. {exp_no}, with cost: {cost: .2f}")
            X_full = sol
            # Output only the solution associated with the best objective value.
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
    
    # Update inventory and demand.
    for (plant, dealer, sku_id, veh_type, veh_id), qty in solution.items():
        # Update inventory.
        data.sku_inv[(plant, sku_id)] -= qty
        # Update demand.
        data.demands[(dealer, sku_id)] -= qty
    
    # extra_fulfill stores demand fulfilled in excess; qty=0 means the demand was satisfied exactly.
    # Rows with qty=0 are retained to make the output file easier to search.
    extra_fulfill = {(dealer, sku_id): -qty for (dealer, sku_id), qty in data.demands.items() if qty <= 0}
    
    # sku_inv_left stores remaining plant inventory; qty=0 means no inventory remains.
    # Rows with qty=0 are retained to make the output file easier to search.
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
    """Reset the dealer-fulfillment output file in the dataset folder."""
    if not os.path.exists(output_file_loc):
        os.makedirs(output_file_loc)
    output_file_name = 'fulfill.csv'
    header = ['client_code', 'product_code', 'volume']
    with open(os.path.join(output_file_loc, output_file_name), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

def clear_output_file(output_file_loc=output_loc):
    """Reset the optimization output file in the dataset folder."""
    if not os.path.exists(output_file_loc):
        os.makedirs(output_file_loc)
    output_file_name = 'opt_result.csv'
    header = ['Index', 'day', 'plant_code', 'client_code', 'product_code', 'vehicle_id', 'vehicle_type', 'qty']
    with open(os.path.join(output_file_loc, output_file_name), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        
def save_fulfill(fulfill):
    """Save dealer-fulfillment results."""
    if not os.path.exists(data.output_file_loc):
        os.makedirs(data.output_file_loc)
    
    output_file = os.path.join(data.output_file_loc, 'fulfill.csv')
    df_fulfill = pd.DataFrame({
        (dealer, sku_id, qty) for (dealer, sku_id), qty in fulfill.items()
    }, columns=['client_code', 'product_code', 'volume'])
    df_fulfill.to_csv(output_file, index=False)        


def output_result(solution):
    """Save the optimization result in the corresponding dataset folder."""
    output_file = os.path.join(data.output_file_loc, 'opt_result.csv')
    df_result = create_dataframe(solution)
    # header=False prevents DataFrame column names from being written to the file.
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
    """Process and save summary statistics in the corresponding dataset folder."""
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

    df_tmp = df_tmp.reset_index()
    df_tmp['below_min_load'] = (
        df_tmp['occupied_size'] + POSITIVEEPS < df_tmp['min_standard_size']
    )
    df_tmp['fixed_cost'] = (
        df_tmp['vehicle_type'].astype(str).map(r_data.veh_type_cost)
    )
    df_tmp['minimum_load_penalty'] = np.where(
        df_tmp['below_min_load'],
        r_data.param_alpha * df_tmp['fixed_cost'],
        0.0,
    )
    df_tmp['total_cost'] = (
        df_tmp['fixed_cost'] + df_tmp['minimum_load_penalty']
    )
    df_tmp.to_csv(os.path.join(output_location, 'opt_summary.csv'), index=False)

    # Check whether every vehicle's loaded SKU volume satisfies its capacity constraint.
    is_cap_exceed = pd.Series(df_tmp['occupied_size'] <= df_tmp['carry_standard_size'])
    if is_cap_exceed.all():  # Return True when every value is True.
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
