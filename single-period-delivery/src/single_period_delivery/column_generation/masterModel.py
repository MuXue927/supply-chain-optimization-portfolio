# 列生成算法的主模型
# 适用于模型1——分解方法
# 模型 1 的假设包括：单周期、SKU库存量 > 需求量、生产基地的存储能力无上限
# 多生产基地、多经销商、多车型、多SKU
# 每家生产基地可以提供相同的SKU, 不同生产基地提供的SKU之间存在交集
import time
import random
from gurobipy import GRB
import gurobipy as gp
from .vehicle import Vehicle
from typing import Dict, Tuple, Set
from dataclasses import dataclass, field
from .optutility import LogPrinter, POSITIVEEPS
from .InputDataCG import InputDataCG

# 创建LogPrinter的实例
log_printer = LogPrinter(time.time())
# # 通过设置force_plain=True, 强制化使用纯文本输出, 例如在重定向到文件时
# log_printer.set_output_mode(force_plain=True)

@dataclass
class MasterModel:
    data: InputDataCG
    # 对于单周期问题, 默认规划周期长度为 1
    horizons = 1
    # 默认从第一天开始规划
    day = 1

    # 设定环境参数
    # MIP模型的求解gap, 当达到设定的gap后, 模型停止, 默认值为0
    param_gap_limit: float = field(init=False)
    # MIP模型求解的时间限制, 默认值为60
    param_time_limit: float = field(init=False)
    # 是否输出gurobi的求解日志
    param_output_flag: int = field(init=False)

    # 装载模式的编号, 只要不重复就好, 用作key的一部分
    pattern_id = 0
    # 记录进入主模型中的每一种装载模式, key是装载模式的名称, 用d_xs的变量名表示
    # value是一种具体的装载模式, 包括装载的SKU及其数量
    pattern_content = {}
    # 当前模型中存在多少个装载模式
    num_patterns = 0
    # 裁剪模型的时候, 每次减少多少个装载模式, 当前没有启动
    remove_batch = 100

    # 定义需要的数据接口, 看数学模型中的符号定义
    # 生产基地的集合
    plants: Set[str] = field(init=False)
    # 经销商的集合
    dealers: Set[str] = field(init=False)
    # 所有SKU的集合
    all_skus: Set[str] = field(init=False)
    # 所有车辆类型集合
    all_veh_types: Set[str] = field(init=False)

    # 每种SKU的体积量化数值 {sku: size}
    sku_sizes: Dict[str, float] = field(init=False)
    # 生产基地中SKU的当前库存 {(plant, sku): qty}
    sku_inv: Dict[Tuple[str, str], int] = field(init=False)
    # 某种SKU的总库存数量 {sku: total_qty}
    sku_total_inv: Dict[str, int] = field(init=False)
    # 生产基地可以提供的SKU集合
    skus_plant: Dict[str, Set[str]] = field(init=False)
    # 经销商对SKU的需求 {(dealer, sku): qty}
    demands: Dict[Tuple[str, str], int] = field(init=False)
    # 经销商需要的SKU集合
    skus_dealer: Dict[str, Set[str]] = field(init=False)

    # 接下来, 需要定义数学模型、目标函数、人工变量和决策变量
    # 最后再定义约束条件
    # 定义主模型
    model: gp.Model = field(init=False)
    # 定义目标函数
    obj: gp.LinExpr = field(init=False)
    # 人工变量 alpha
    d_alpha: gp.tupledict[Tuple[str, str], gp.Var] = field(init=False)    # {(dealer, sku): gp.Var}
    # 人工变量 beta
    d_beta: gp.tupledict[str, gp.Var] = field(init=False)     # {sku: gp.Var}
    # 人工变量 gamma
    d_gamma: gp.tupledict[Tuple[str, str], gp.Var] = field(init=False)  # {(plant, sku): gp.Var}
    # 定义决策变量 x_ijf
    # 主模型的决策变量, 表示从生产基地 i 到经销商 j 的车辆类型 f 对应的一辆车辆的装载方案的使用数量
    # {(plant, dealer, v_type, load_pattern_id): gp.Var}
    d_xs: gp.tupledict[Tuple[str, str, str, int], gp.Var] = field(init=False)

    # 定义需要向模型中添加的约束条件
    # 所有生产基地 i 使用所有车型 f, 运往某个经销商 j 的SKU k 的总量, 等于经销商 j 对SKU k 的需求
    cons_dealer_demands: gp.tupledict[Tuple[str, str], gp.Constr] = field(init=False)   # {(dealer, sku_id): gp.Constr}
    # 实际运出去的SKU k 的数量不超过SKU k 的 **总库存数量**
    cons_sku_supply: gp.tupledict[str, gp.Constr] = field(init=False)  # {sku_id: gp.Constr}
    # 从生产基地 i 运出去的SKU k 的数量, 不超过生产基地 i 拥有的SKU k 的库存
    cons_fact_sku_inv: gp.tupledict[Tuple[str, str], gp.Constr] = field(init=False)    # {(plant, sku_id): gp.Constr}

    def __post_init__(self):
        self.param_gap_limit = self.data.param_gap_limit
        self.param_time_limit = self.data.param_time_limit
        self.param_output_flag = self.data.param_output_flag

        self.plants = self.data.plants
        self.dealers = self.data.dealers
        self.all_skus = self.data.all_skus
        self.all_veh_types = self.data.all_veh_types
        self.sku_sizes = self.data.sku_sizes
        self.sku_inv = self.data.sku_inv
        self.sku_total_inv = self.data.sku_total_inv
        self.skus_plant = self.data.skus_plant
        self.demands = self.data.demands
        self.skus_dealer = self.data.skus_dealer

        self.model = gp.Model()
        self.obj = gp.LinExpr()

        duet_dealer_sku = {(dealer, sku_id) for (dealer, sku_id) in self.demands}
        self.d_alpha = self.model.addVars(duet_dealer_sku, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="d_alpha")

        self.d_beta = self.model.addVars(self.all_skus, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="d_beta")

        duet_plant_sku = {(plant, sku_id) for (plant, sku_id) in self.sku_inv}
        self.d_gamma = self.model.addVars(duet_plant_sku, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="d_gamma")

        self.d_xs = gp.tupledict()

        self.obj = (self.data.param_pun_factor1 * self.d_alpha.sum('*', '*') +
                    self.data.param_pun_factor2 * self.d_beta.sum('*') +
                    self.data.param_pun_factor3 * self.d_gamma.sum('*', '*'))
        self.model.setObjective(self.obj, sense=GRB.MINIMIZE)

        self.cons_dealer_demands = self.model.addConstrs((
            self.d_alpha[dealer, sku_id] >= self.demands[dealer, sku_id]
            for (dealer, sku_id) in duet_dealer_sku
        ), name="cons_dealer_demands")

        self.cons_sku_supply = self.model.addConstrs((
            self.d_beta[sku_id] >= -self.sku_total_inv[sku_id]
            for sku_id in self.all_skus
        ), name="cons_sku_supply")

        self.cons_fact_sku_inv = self.model.addConstrs((
            self.d_gamma[plant, sku_id] >= -self.sku_inv[plant, sku_id]
            for (plant, sku_id) in duet_plant_sku
        ), name="cons_fact_sku_inv")

        # 启用求解日志
        self.model.setParam('OutputFlag', self.param_output_flag)
        # 设置模型求解gap
        self.model.Params.MIPGap = self.param_gap_limit
        # 设置求解时间
        self.model.Params.TimeLimit = self.param_time_limit

    def get_alpha_size(self):
        non_zero_alpha = {(dealer, sku_id) for (dealer, sku_id), var in self.d_alpha.items() if var.X > POSITIVEEPS}
        return len(non_zero_alpha)

    def get_beta_size(self):
        non_zero_beta = {sku_id for sku_id, var in self.d_beta.items() if var.X > POSITIVEEPS}
        return len(non_zero_beta)

    def get_gamma_size(self):
        non_zero_gamma = {(plant, sku_id) for (plant, sku_id), var in self.d_gamma.items() if var.X > POSITIVEEPS}
        return len(non_zero_gamma)

    def deactivate_alpha(self):
        for (dealer, sku_id) in self.d_alpha:
            self.d_alpha[dealer, sku_id].setAttr('UB', 0)

    def deactivate_beta(self):
        for sku_id in self.d_beta:
            self.d_beta[sku_id].setAttr('UB', 0)

    def deactivate_gamma(self):
        for (plant, sku_id) in self.d_gamma:
            self.d_gamma[plant, sku_id].setAttr('UB', 0)

    def deactivate_artificial_vars(self):
        self.deactivate_alpha()
        self.deactivate_beta()
        self.deactivate_gamma()

    def activate_alpha(self):
        for (dealer, sku_id) in self.d_alpha:
            self.d_alpha[dealer, sku_id].setAttr('UB', GRB.INFINITY)

    def activate_beta(self):
        for sku_id in self.d_beta:
            self.d_beta[sku_id].setAttr('UB', GRB.INFINITY)

    def activate_gamma(self):
        for (plant, sku_id) in self.d_gamma:
            self.d_gamma[plant, sku_id].setAttr('UB', GRB.INFINITY)

    def activate_artificial_vars(self):
        self.activate_alpha()
        self.activate_beta()
        self.activate_gamma()

    def change_var_type(self, var_type=GRB.INTEGER):
        log_printer.print("Set variables in the master model to integer type")
        for var in self.model.getVars():
            # if var.VarName.startswith("d_xs_"):
            var.setAttr('VType', var_type)
        self.model.Params.MIPGap = self.param_gap_limit
        self.model.Params.TimeLimit = self.param_time_limit
        self.model.setParam('OutputFlag', self.param_output_flag)

    def remove_partial_artificial_var(self):
        """
        remove artificial variables whose value equals zero and update model.
        """
        remove_alpha = {(dealer, sku_id): var for (dealer, sku_id), var in self.d_alpha.items() if var.X <= POSITIVEEPS}
        reserve_alpha = {(dealer, sku_id): var for (dealer, sku_id), var in self.d_alpha.items() if var.X > POSITIVEEPS}
        self.d_alpha = gp.tupledict(reserve_alpha)

        remove_beta = {sku_id: var for sku_id, var in self.d_beta.items() if var.X <= POSITIVEEPS}
        reserve_beta = {sku_id: var for sku_id, var in self.d_beta.items() if var.X > POSITIVEEPS}
        self.d_beta = gp.tupledict(reserve_beta)

        remove_gamma = {(plant, sku_id): var for (plant, sku_id), var in self.d_gamma.items() if var.X <= POSITIVEEPS}
        reserve_gamma = {(plant, sku_id): var for (plant, sku_id), var in self.d_gamma.items() if var.X > POSITIVEEPS}
        self.d_gamma = gp.tupledict(reserve_gamma)

        self.model.remove(remove_alpha)
        self.model.remove(remove_beta)
        self.model.remove(remove_gamma)

        self.model.update()
        self.obj = self.model.getObjective()  # type: ignore
        return self.obj.size()

    def insert_pattern(self, fact_id: str, dealer_id: str, vehicle_type: str,
                        obj_contribution: float, load_pattern: Dict[str, int]):
        """
        insert load patterns submitted by sub models into master model

        :param fact_id: a specific plant
        :param dealer_id: a specific dealer
        :param vehicle_type: a specific vehicle type
        :param obj_contribution: the profit contribution of a specific load pattern
        :param load_pattern: a specific load pattern
        :return: None
        """
        pattern_column = gp.Column()

        # list前面加*, 表示将list unpack成多个独立的参数传入函数
        # zip(*iterables, strict=False)
        # Iterate over several iterables in parallel, producing tuples with an item from each one.
        # dict.items() 返回(键, 值)组成的一个新视图
        skus_loaded, load_num = zip(*list(load_pattern.items()))
        skus_loaded = list(skus_loaded)
        load_num = list(load_num)

        cons_dealer_demands_impacted = [self.cons_dealer_demands[dealer_id, sku_id] for sku_id in skus_loaded]
        pattern_column.addTerms(load_num, cons_dealer_demands_impacted)

        # neg_load_num = [-qty for qty in load_num]

        cons_sku_supply_impacted = [self.cons_sku_supply[sku_id] for sku_id in skus_loaded]
        pattern_column.addTerms(load_num, cons_sku_supply_impacted)

        cons_fact_sku_inv_impacted = [self.cons_fact_sku_inv[fact_id, sku_id] for sku_id in skus_loaded]
        pattern_column.addTerms(load_num, cons_fact_sku_inv_impacted)

        d_xs_name = f"d_xs_{fact_id}_{dealer_id}_{vehicle_type}_{self.pattern_id}"
        self.d_xs[fact_id, dealer_id, vehicle_type, self.pattern_id] = (
            self.model.addVar(lb=0, obj=obj_contribution, vtype=GRB.CONTINUOUS, column=pattern_column, name=d_xs_name))

        self.pattern_content[d_xs_name] = load_pattern

        self.pattern_id += 1
        self.num_patterns += 1
        self.model.update()

    def shrink(self):
        """
        remove load patterns when the number of load patterns is greater than predefined value.
        """
        log_printer.print("Start to shrink master model...")
        remove_pattern_keys = [key for key in self.d_xs if self.d_xs[key].X == 0]
        remove_pattern_nums = len(remove_pattern_keys)
        random.shuffle(remove_pattern_keys)
        while remove_pattern_nums > 0 and self.num_patterns > self.data.param_pattern_num_shrink_to:
            remove_key = remove_pattern_keys.pop()
            self.model.remove(self.d_xs[remove_key])
            self.d_xs.pop(remove_key)
            remove_pattern_nums -= 1
            self.num_patterns -= 1
        self.model.update()
        log_printer.print("Shrink Done!")

    def initialize_solution(self):
        for (plant, dealer) in self.data.construct_supply_chain():
            # 取得最大容量的车型
            max_veh_type = max(self.all_veh_types, key=float)
            selected_vehicle = Vehicle(plant, dealer, max_veh_type, self.data)
            # 当前经销商需要的SKU种类和数量
            orders = {sku_id: qty for (dealer_id, sku_id), qty in self.demands.items() if dealer_id == dealer}
            for sku_id, qty in orders.items():
                # 生产基地可以提供经销商需要的SKU
                if sku_id in self.skus_plant[plant]:
                    while qty > 0:
                        qty = selected_vehicle.load(sku_id, qty)  # type: ignore
                        if qty > 0:
                            self.insert_pattern(plant, dealer, max_veh_type,
                                                self.data.veh_type_cost[max_veh_type], selected_vehicle.cargo)
                            selected_vehicle.clear()
            
            if not selected_vehicle.is_empty():
                self.insert_pattern(plant, dealer, max_veh_type,
                                    self.data.veh_type_cost[max_veh_type], selected_vehicle.cargo)
                selected_vehicle.clear()
        pass

    def run(self):
        # log_printer.print("Optimizing master model...")
        self.model.optimize()
        self.model.write("MasterModel.lp")

        if self.model.Status in (GRB.INF_OR_UNBD, GRB.INFEASIBLE, GRB.UNBOUNDED):
            log_printer.print("Exception: Master Model Is Infeasible or Unbounded!")
            self.model.computeIIS()
            self.model.write("Infeasible_MasterModel.ilp")

            if self.model.IISMinimal:  # 判断 model.computeIIS() 返回的是否是包含不可行约束条件数目最少的IIS
                log_printer.print('IIS is minimal \n')
            else:
                log_printer.print('IIS is not minimal \n')
            log_printer.print('\n The following constraint (s) cannot be satisfied: ')
            for c in self.model.getConstrs():
                if c.IISConstr:
                    log_printer.print('%s' % c.ConstrName)
            return False
        else:
            self.print_sol_summary()
            return True

    def print_sol_summary(self):
        real_obj = sum(var.X for var in self.d_xs.values())
        log_printer.print(
            f"\tRealObj={real_obj:.1f}"
            f"\tNo. Alpha={self.get_alpha_size()}"
            f"\tNo. Beta={self.get_beta_size()}"
            f"\tNo. Gamma={self.get_gamma_size()}"
            f"\tNo. Pattern={self.num_patterns}"
            f"\tTotalObj={self.model.ObjVal:.2f}"
        )
