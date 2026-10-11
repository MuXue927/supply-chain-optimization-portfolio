# 单周期数学模型的gurobi代码
# 需要实现模型 1 中定义的数学模型
# 模型 1 的假设包括：单周期、SKU库存量 ≥ 需求量、生产基地的存储能力无上限
# 多生产基地、多经销商、多车型、多SKU
# 每家生产基地可以提供相同的SKU，不同生产基地提供的SKU之间存在交集
import os
import time
from gurobipy import GRB
import gurobipy as gp
from typing import Dict, Tuple, Set, List
from dataclasses import dataclass, field
from .InputDataSingle import InputDataSingle
from .optutility import LogPrinter

# 创建LogPrinter的实例
log_printer = LogPrinter(time.time())
# # 通过设置force_plain=True，强制化使用纯文本输出，例如在重定向到文件时
# # 如果需要显示格式化文本，例如字体颜色等，可以注释掉这行，并且关闭重定向
# # 需要修改optLogicSingle.py文件中的run_model方法，启用注释部分的代码，并注释掉相对应部分的现有代码
# log_printer.set_output_mode(force_plain=True)

@dataclass
class MonolithicModelSingle:
    data: InputDataSingle
    # 对于单周期问题，默认规划周期长度为 1
    horizons = 1
    # 默认从第一天开始规划
    day = 1

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
    # 生产基地中SKU的当前库存 {(plant, sku): qty}
    sku_inv: Dict[Tuple[str, str], int] = field(init=False)
    # 某种SKU的总库存数量 {sku: total_qty}
    sku_total_inv: Dict[str, int] = field(init=False)
    # 经销商对SKU的需求 {(dealer, sku): qty}
    demands: Dict[Tuple[str, str], int] = field(init=False)
    # 生产基地和经销商之间的供需关系 {(plant, dealer): {sku}}
    supply_chain: Dict[Tuple[str, str], Set[str]] = field(init=False)
    # 每一辆车的使用成本 {v_id: cost}
    veh_cost: Dict[int, float] = field(init=False)
    # 每一辆车的容量上限 {v_id: capacity}
    veh_cap: Dict[int, int] = field(init=False)
    # 每一辆车的最小起运量 {v_id: min_load}
    veh_min_load: Dict[int, int] = field(init=False)
    # 所有车辆类型
    all_veh_types: List[str] = field(init=False)
    # 有了车辆编号，还需要在车辆编号和车辆类型之间建立联系
    # 告诉我任意一辆车的编号，通过查看这辆车的类型，就能知道
    # 这辆车的所有信息，包括使用成本、容量上限、最小起运量
    # {v_id: v_type}
    veh_type: Dict[int, str] = field(init=False)
    # 同理，告诉我任意一种车辆的类型，我能知道这种车辆类型拥有哪些编号的车辆 {v_type: {v_id}}
    veh_ids: Dict[str, Set[int]] = field(init=False)
    # 到这里，单周期优化问题需要的所有数据接口，已经定义完毕

    # 接下来，需要定义数学模型、目标函数、各种辅助变量和决策变量
    # 最后再定义约束条件
    # 定义gurobi模型
    model: gp.Model = field(init=False)
    # 定义目标函数
    obj: gp.LinExpr = field(init=False)
    # 定义辅助变量
    # 是否使用车辆 l {v_id: gp.Var}，使用 = 1，不使用 = 0
    z_l: gp.tupledict[int, gp.Var] = field(init=False)
    # 车辆 l 的实际装载体积是否低于最小起运量， {v_id: gp.Var}，低于 = 1， 不低于 = 0
    zz_l: gp.tupledict[int, gp.Var] = field(init=False)
    # 车辆 l 最小起运量的惩罚标识变量，{v_id: gp.Var}，惩罚 = 1，不惩罚 = 0
    # 只有当使用车辆 l 时，如果车辆 l 的实际装载体积是否低于最小起运量，才惩罚，否则不惩罚
    # 如果没有使用车辆 l，则不惩罚
    p_l: gp.tupledict[int, gp.Var] = field(init=False)
    # 定义决策变量
    # 是否使用车辆 l 配送从生产基地 i 到经销商 j 的商品，配送 = 1， 不配送 = 0 {(plant, dealer, v_id): gp.Var}
    y_ijl: gp.tupledict[Tuple[str, str, int], gp.Var] = field(init=False)
    # 从生产基地 i 运往经销商 j 的车辆 l 中装载SKU k 的数量  {(plant, dealer, sku_id, v_id): gp.Var}
    x_ijkl: gp.tupledict[Tuple[str, str, str, int], gp.Var] = field(init=False)

    # 决策变量的下标
    y_indices: List[Tuple[str, str, int]] = field(init=False)
    x_indices: List[Tuple[str, str, str, int]] = field(init=False)
    new_duet_ik: Set[Tuple[str, str]] = field(init=False)

    # 定义需要向模型中添加的约束条件
    # 约束 1 -- 对于所有生产基地 i 和经销商 j 构成的有序对 (i, j)
    # 只要生产基地 i 使用车辆 l 为经销商 j 配送商品，那么就代表车辆 l 已经被使用
    cons_veh_use: gp.tupledict[int, gp.Constr] = field(init=False)                      # {v_id: gp.Constr}
    # 约束2 -- 只有在使用车辆 l 时，才能将商品装载到车辆 l 上
    cons_sku_load: gp.tupledict[int, gp.Constr] = field(init=False)                     # {v_id: gp.Constr}
    # 约束3 -- 车辆 l 的实际装载体积不小于该车辆的最小起运量
    cons_veh_min_load: gp.tupledict[int, gp.Constr] = field(init=False)                 # {v_id: gp.Constr}
    # 约束4 -- 车辆 l 最小起运量的惩罚标识变量 p_l，与车辆 l 的使用状态变量 z_l
    # 以及车辆 l是否满足最小起运量的状态变量 zz_l，三者之间应该满足的关系
    cons_auxi_relation: gp.tupledict[int, gp.Constr] = field(init=False)                # {v_id: gp.Constr}
    # 约束5 -- 车辆 l 的实际装载体积不超过该车辆的最大装载能力
    cons_veh_cap_limit: gp.tupledict[int, gp.Constr] = field(init=False)                # {v_id: gp.Constr}
    # 约束6 -- 所有生产基地 i 使用所有车辆 l，运往某个经销商 j 的SKU k 的总量，等于经销商 j 对SKU k 的需求
    cons_dealer_demands: gp.tupledict[Tuple[str, str], gp.Constr] = field(init=False)   # {(dealer, sku_id): gp.Constr}
    # 约束7 -- 从生产基地 i 实际运出去的SKU k 的数量不超过生产基地 i 中SKU k 的当前库存
    cons_sku_supply: gp.tupledict[Tuple[str, str], gp.Constr] = field(init=False)       # {(plant, sku_id): gp.Constr}
    # # 约束8 -- 从所有生产基地实际运出去的SKU k 的数量不超过SKU k的库存总量                       # {sku_id: gp.Constr}
    # cons_sku_total_supply: gp.tupledict[str, gp.Constr] = field(init=False)

    # 约束9 -- 添加对称性割，用于破除模型中存在的对称性，缩减解空间
    # {(v_type, v_id, v_di_1): gp.Constr}
    cons_symmetry_break: gp.tupledict[Tuple[str, int, int], gp.Constr] = field(init=False)
    # 至此，单周期优化模型的所有相关属性都已经定义完毕
    # 接下来，要做的事情是实例化这些属性

    def __post_init__(self):
        self.param_alpha = self.data.param_alpha
        self.param_BIG_NUM = self.data.param_BIG_NUM
        self.param_gap_limit = self.data.param_gap_limit
        self.param_time_limit = self.data.param_time_limit
        self.param_output_flag = self.data.param_output_flag

        self.plants = self.data.plants
        self.dealers = self.data.dealers
        self.all_skus = self.data.all_skus
        self.all_vehicles = self.data.all_vehicles
        self.sku_sizes = self.data.sku_sizes
        self.sku_inv = self.data.sku_inv
        self.skus_plant = self.data.skus_plant
        self.sku_total_inv = self.data.sku_total_inv
        self.demands = self.data.demands
        self.skus_dealer = self.data.skus_dealer
        self.supply_chain = self.data.supply_chain
        self.veh_cost = self.data.veh_cost
        self.veh_cap = self.data.veh_cap
        self.veh_min_load = self.data.veh_min_load
        self.all_veh_types = self.data.all_veh_types
        self.veh_type = self.data.veh_type
        self.veh_ids = self.data.veh_ids

        self.model = gp.Model()
        self.obj = gp.LinExpr()

        # 添加辅助变量
        self.z_l = self.model.addVars(self.all_vehicles, lb=0, ub=1, vtype=GRB.BINARY, name="z_l")
        self.zz_l = self.model.addVars(self.all_vehicles, lb=0, ub=1, vtype=GRB.BINARY, name="zz_l")
        self.p_l = self.model.addVars(self.all_vehicles, lb=0, ub=1, vtype=GRB.BINARY, name="p_l")

        # 添加决策变量
        self.y_indices = [
            (plant, dealer, v_id) for (plant, dealer) in self.supply_chain 
            for v_id in self.all_vehicles
        ]
        self.y_ijl = self.model.addVars(self.y_indices, lb=0, ub=1, vtype=GRB.BINARY, name="y")

        # self.demands的 key -- (dealer, sku_id) 体现的是经销商 j 对SKU k 存在需求
        # 构建 (plant, dealer, sku_id) 的集合，代表生产基地 i 可以向经销商 j 提供 SKU k
        # 锁定某个经销商 j
        # 为什么要用集合？因为必须保证每个变量的下标是唯一确定的，所以要强制去除重复元素，用集合最保险
        triple_ijk = {
            (plant, dealer1, sku_id) for (plant, dealer1) in self.supply_chain
            for (dealer2, sku_id) in self.demands if dealer1 == dealer2 if (plant, sku_id) in self.sku_inv
        }
        self.new_duet_ik = {(plant, sku_id) for (plant, dealer, sku_id) in triple_ijk}
        
        self.x_indices = [
            (plant, dealer, sku_id, v_id) for (plant, dealer, sku_id) in triple_ijk 
            for v_id in self.all_vehicles
        ]
        self.x_ijkl = self.model.addVars(self.x_indices ,lb=0, vtype=GRB.INTEGER, name="x")

        # # 构建目标函数
        self.obj = (gp.quicksum(self.veh_cost[v_id] * self.z_l[v_id] for v_id in self.all_vehicles) +
                    self.param_alpha * gp.quicksum(self.veh_cost[v_id] * self.p_l[v_id] for v_id in self.all_vehicles))
        # 设定目标函数的优化方向
        self.model.setObjective(self.obj, sense=GRB.MINIMIZE)

        # 添加约束条件
        # 约束 1 -- 对于所有生产基地 i 和经销商 j 构成的有序对 (i, j)
        # 只要生产基地 i 使用车辆 l 为经销商 j 配送商品，那么就代表车辆 l 已经被使用
        self.cons_veh_use = self.model.addConstrs((
            self.y_ijl.sum('*', '*', v_id) == self.z_l[v_id]
            for v_id in self.all_vehicles), name="cons_veh_use")
        # 约束2 -- 只有在使用车辆 l 时，才能将商品装载到车辆 l 上
        self.cons_sku_load = self.model.addConstrs((
            self.x_ijkl.sum(plant, dealer, '*', v_id) <= self.param_BIG_NUM * self.y_ijl[plant, dealer, v_id]
            for (plant, dealer, v_id) in self.y_indices), name="cons_sku_load")
        # 约束3 -- 车辆 l 的实际装载体积不小于该车辆的最小起运量
        self.cons_veh_min_load = self.model.addConstrs((
            gp.quicksum(
                self.x_ijkl.sum('*', '*', sku_id, v_id) * self.sku_sizes[sku_id]
                for sku_id in self.all_skus)
            + self.zz_l[v_id] * self.param_BIG_NUM >= self.veh_min_load[v_id]
            for v_id in self.all_vehicles),
            name="cons_veh_min_load")
        # 约束4 -- 车辆 l 最小起运量的惩罚标识变量 p_l，与车辆 l 的使用状态变量 z_l
        # 以及车辆 l是否满足最小起运量的状态变量 zz_l，三者之间应该满足的关系
        self.cons_auxi_relation = self.model.addConstrs((
            self.z_l[v_id] + self.zz_l[v_id] - 1 <= self.p_l[v_id]
            for v_id in self.all_vehicles), name="cons_auxi_relation")
        # 约束5 -- 车辆 l 的实际装载体积不超过该车辆的最大装载能力
        self.cons_veh_cap_limit = self.model.addConstrs((
            gp.quicksum(
                self.x_ijkl.sum('*', '*', sku_id, v_id) * self.sku_sizes[sku_id]
                for sku_id in self.all_skus)
            <= self.veh_cap[v_id]
            for v_id in self.all_vehicles), name="cons_veh_cap_limit")
        # 约束6 -- 所有生产基地 i 使用所有车辆 l，运往某个经销商 j 的SKU k 的总量，等于经销商 j 对SKU k 的需求
        # 在gurobi中可以写成 >= 数学模型中是为了更严谨地表达
        self.cons_dealer_demands = self.model.addConstrs((
            self.x_ijkl.sum('*', dealer, sku_id, '*') >= self.demands[dealer, sku_id]
            for (dealer, sku_id) in self.demands), name="cons_dealer_demands")
        # 约束7 -- 从生产基地 i 实际运出去的SKU j 的数量不超过生产基地 i 中SKU k 的当前库存
        self.cons_sku_supply = self.model.addConstrs((
            self.x_ijkl.sum(plant, '*', sku_id, '*') <= self.sku_inv[plant, sku_id]
            for (plant, sku_id) in self.new_duet_ik), name="cons_sku_supply")

        # # 约束8 -- 从所有生产基地实际运出去的SKU k 的数量不超过SKU k的库存总量
        # self.cons_sku_total_supply = self.model.addConstrs((
        #     self.x_ijkl.sum('*', '*', sku_id, '*') <= self.sku_total_inv[sku_id]
        #     for sku_id in self.sku_total_inv), name="cons_sku_total_inv")

        # # # 约束9 -- 添加的对称性割，用于破除模型中存在的对称性
        # self.cons_symmetry_break = self.model.addConstrs((
        #     self.z_l[v_id] <= self.z_l[v_id_1]
        #     for v_type in self.all_veh_types
        #     for v_id, v_id_1 in zip(list(self.veh_ids[v_type]), list(self.veh_ids[v_type])[1:])
        # ), name="cons_symmetry_break")

        # 启用求解日志
        self.model.setParam('OutputFlag', self.param_output_flag)
        # 设置模型求解gap
        self.model.Params.MIPGap = self.param_gap_limit
        # 设置求解时间
        self.model.Params.TimeLimit = self.param_time_limit
        

    def run(self):
        self.model.optimize()
        self.model.write("MonolithicModelSingle.lp")

        if self.model.Status in (GRB.INF_OR_UNBD, GRB.INFEASIBLE, GRB.UNBOUNDED):
            self.model.computeIIS()
            self.model.write("Infeasible_MonolithicModelSingle.ilp")

            if self.model.IISMinimal:  # 判断 model.computeIIS() 返回的是否是包含不可行约束条件数目最少的IIS
                log_printer.print('IIS is minimal \n')
            else:
                log_printer.print('IIS is not minimal \n')
            log_printer.print('\n The following constraint (s) cannot be satisfied :')
            for c in self.model.getConstrs():
                if c.IISConstr:
                    log_printer.print('%s' % c.ConstrName)
            return False
        elif self.model.Status == GRB.TIME_LIMIT:
            log_printer.print("Time limit is reached! Solving process is stopped.")
            self.model.terminate()
            # 设置模型参数JSONSolDetail = 1, 让模型在保存的solution文件中输出更为详细的信息
            self.model.Params.JSONSolDetail = 1
            # 将模型目前找到的可行解以json格式保存到文件中
            self.save_feasible_sol()
        elif self.model.Status == GRB.INTERRUPTED:
            log_printer.print("Optimization is interrupted by user!")
            return False
        else:  # model is optimal
            self.print_sol_summary()
            return True

    def save_feasible_sol(self):
        sol_file_loc = os.path.join(self.data.output_file_loc, self.data.dataset_name)
        if not os.path.exists(sol_file_loc):
            os.makedirs(sol_file_loc)
        sol_file_name = 'Feasible-Solution.json'
        sol_file = os.path.join(sol_file_loc, sol_file_name)
        self.model.write(sol_file)

    # 显示求解结果摘要信息
    def print_sol_summary(self):
        if self.model.Status == GRB.OPTIMAL:
            log_printer.print("Monolithic Model for single period is solved to be optimal.")
            log_printer.print(f"Total cost of optimal solution is {self.model.ObjVal}")

            total_veh_num = sum(var.Xn for var in self.z_l.values())
            log_printer.print(f"Total number of vehicles used is {total_veh_num}")

            veh_punished_num = sum(var.Xn for var in self.p_l.values())
            log_printer.print(f"Among used vehicles, there are {veh_punished_num} vehicle(s) "
                              f"that don't satisfy veh_min_load constraint and need to be punished.")

            total_ship_num = sum(var.Xn for var in self.x_ijkl.values())
            log_printer.print(f"Total SKU ship amount is {total_ship_num}")
