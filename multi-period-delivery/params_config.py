# 将模型需要的参数写入json文件
# 并保存到对应文件夹中
# 只需要修改medium_dir
import os
import json

data = {
    "solver_id": "GUROBI",
    "alpha": 2,
    "BIG_NUM": 99999,
    "gap_limit": 0,
    "time_limit": 900,
    "output_flag": 1
}

data_cg = {
    "solver_id": "GUROBI",
    "pun_factor1": 5000,
    "pun_factor2": 5000,
    "pun_factor3": 5000,
    "pun_factor4": 5000,
    "pun_factor5": 5000,
    "pattern_batch":100,
    "pattern_num_shrink_to":15000,
    "trigger_shrink_level":20000,
    "gap_limit": 0,
    "time_limit": 6000,
    "output_flag": 1
}

data_alns = {
    "pun_factor1": 20_000,  #  unmet demand penalty
    "pun_factor2": 5_000,   #  negative inventory penalty
    "pun_factor3": 1_000,   #  exceeding vehicle capacity penalty
    "pun_factor4": 10_000,  #  vehicle minimum load penalty
    "alpha": 2              #  when computing real objective, penalty for minimum load constraint is scaled down by this factor
}

# 指定要保存的文件夹路径
parent_dir = 'datasets'
# 多周期问题的保存路径
medium_dir = os.path.join('multiple-periods', 'large')

output_file_loc = os.path.join(parent_dir, medium_dir)
# 如果不存在该文件夹，则创建
if not os.path.exists(output_file_loc):
    os.makedirs(output_file_loc)
else:
    pass
# 指定保存的文件名
# file_name = 'model_config_cg.json'
# file_name = 'model_config.json'
file_name = 'model_config_alns.json'
# 组合完整的文件路径
file_path = os.path.join(output_file_loc, file_name)
# 将数据写入json文件
with open(file_path, "w", encoding="utf-8") as fp:
    # json.dump(data_cg, fp, indent=4)
    # json.dump(data, fp, indent=4)
    json.dump(data_alns, fp, indent=4)
print(f"JSON file has been written into location {file_path}")
