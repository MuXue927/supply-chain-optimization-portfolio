# 列生成算法的子模型
# 适用于模型1——分解方法
# 模型 1 的假设包括：单周期、SKU库存量 > 需求量、生产基地的存储能力无上限
# 多生产基地、多经销商、多车型、多SKU
# 每家生产基地可以提供相同的SKU，不同生产基地提供的SKU之间存在交集
import math
from gurobipy import GRB
import gurobipy as gp
from typing import Dict, Tuple, Set, Hashable
from dataclasses import dataclass, field
from masterModel import log_printer
from InputDataCG import InputDataCG

@dataclass
class SubModel:
    fact_id: str
    dealer_id: str
    vehicle_type: str
    data: InputDataCG

    # 设定环境参数
    # MIP模型的求解gap，当达到设定的gap后，模型停止，默认值为0
    param_gap_limit: float = field(init=False)
    # MIP模型求解的时间限制，默认值为60
    param_time_limit: float = field(init=False)
    # 是否输出gurobi的求解日志
    param_output_flag: int = field(init=False)

    # 所有SKU的集合
    all_skus: Set[str] = field(init=False)
    # 当前经销商需要的且当前生产基地能够提供的SKU集合
    require_skus: Set[str] = field(init=False)
    # 每种SKU的体积量化数值 {sku: size}
    sku_sizes: Dict[str, float] = field(init=False)
    # 生产基地可以提供的SKU数量 {(plant, sku): qty}
    sku_inv: Dict[Tuple[str, str], int] = field(init=False)
    # 经销商对SKU的需求 {(dealer, sku): qty}
    demands: Dict[Tuple[str, str], int] = field(init=False)
    # 经销商需要的SKU集合
    skus_dealer: Dict[str, Set[str]] = field(init=False)

    # 每一种车辆类型的使用成本 {v_type: cost}
    veh_type_cost: Dict[str, float] = field(init=False)
    # 每一种车辆类型的容量上限 {v_type: capacity}
    veh_type_cap: Dict[str, int] = field(init=False)
    # 每一种车辆类型的最小起运量 {v_type: min_load}
    veh_type_min_load: Dict[str, int] = field(init=False)

    # 接下来，需要定义数学模型、目标函数和决策变量
    # 最后再定义约束条件
    # 定义子模型
    model: gp.Model = field(init=False)
    # 定义目标函数
    obj: gp.LinExpr = field(init=False)
    # 定义决策变量
    # 子模型的决策变量，表示从生产基地 i 到经销商 j 的车辆类型 f 对应的一辆车辆中装载SKU k 的商品数量
    # 其中，生产基地 i 、经销商 j 和车辆类型 f都是已知的
    d_xs: gp.tupledict[str, gp.Var] = field(init=False)
    # 辅助变量：车辆类型 f 中装载的SKU体积是否低于最小起运量，如果是取1，否则取0
    p_f: gp.Var = field(init=False)

    # 定义需要向模型中添加的约束条件
    # 车辆类型 f 的容量约束
    cons_veh_cap: gp.Constr = field(init=False)
    # 车辆类型 f 的最小起运量约束
    cons_veh_min_load: gp.Constr = field(init=False)
    # 决策变量的取值上限约束
    cons_xs_bounds1: gp.tupledict[Hashable, gp.Constr] = field(init=False)
    cons_xs_bounds2: gp.tupledict[Hashable, gp.Constr] = field(init=False)

    def __post_init__(self):
        self.param_gap_limit = self.data.param_gap_limit
        self.param_time_limit = self.data.param_time_limit
        self.param_output_flag = self.data.param_output_flag

        self.all_skus = self.data.all_skus
        self.sku_sizes = self.data.sku_sizes
        self.sku_inv = self.data.sku_inv
        self.demands = self.data.demands
        self.skus_dealer = self.data.skus_dealer
        self.veh_type_cost = self.data.veh_type_cost
        self.veh_type_cap = self.data.veh_type_cap
        self.veh_type_min_load = self.data.veh_type_min_load

        # 子模型的目标函数，只有在获取到主模型约束条件的对偶价格之后，才会构建
        self.model = gp.Model()
        self.obj = gp.LinExpr()

        self.require_skus = self.data.available_skus_to_dealer(self.fact_id, self.dealer_id)
        self.d_xs = self.model.addVars(self.require_skus, lb=0, vtype=GRB.INTEGER, name="d_xs")
        self.p_f = self.model.addVar(lb=0, vtype=GRB.BINARY, name="p_f")

        # 计算所有种类SKU体积之和，作为最小起运量约束中的 Big Number
        # BigM = sum(self.demands[dealer_id, sku_id] * self.sku_sizes[sku_id] for (dealer_id, sku_id) in self.demands)

        # BigM = sum(self.demands[self.dealer_id, sku_id] * self.sku_sizes[sku_id] for sku_id in self.require_skus)

        # delta_k = {sku_id: min(self.demands[self.dealer_id, sku_id], self.sku_inv[self.fact_id, sku_id])
        #            for sku_id in self.require_skus}
        delta_k = {sku_id: max(self.demands[self.dealer_id, sku_id], self.sku_inv[self.fact_id, sku_id])
                   for sku_id in self.require_skus}
        BigM = sum(delta_k[sku_id] * self.sku_sizes[sku_id] for sku_id in self.require_skus)
        BigM = math.ceil(BigM)

        # v1 = sum(self.demands[self.dealer_id, sku_id] * self.sku_sizes[sku_id] for sku_id in self.require_skus)
        # v2 = sum(self.sku_inv[self.fact_id, sku_id] * self.sku_sizes[sku_id] for sku_id in self.require_skus)
        # BigM = min(v1, v2)
        # BigM = max(v1, v2)

        # BigM = BigNum

        # 添加约束条件
        self.cons_veh_cap = self.model.addConstr((
            gp.quicksum(
                self.d_xs[sku_id] * self.sku_sizes[sku_id] for sku_id in self.require_skus
            ) <= self.veh_type_cap[self.vehicle_type]
        ), name="cons_veh_cap")

        self.cons_veh_min_load = self.model.addConstr((
            gp.quicksum(
                self.d_xs[sku_id] * self.sku_sizes[sku_id] for sku_id in self.require_skus
            ) + BigM * self.p_f >= self.veh_type_min_load[self.vehicle_type]
        ), name="cons_veh_min_load")

        self.cons_xs_bounds = self.model.addConstrs((
            self.d_xs[sku_id] <= min(self.demands[self.dealer_id, sku_id], self.sku_inv[self.fact_id, sku_id])
            for sku_id in self.require_skus
        ), name="cons_xs_bounds")

        self.model.setParam(GRB.Param.OutputFlag, 0)

    def run(self):
        # log_printer.print("Optimizing sub model...")
        self.model.setObjective(self.obj, sense=GRB.MAXIMIZE)
        self.model.Params.FeasibilityTol = 1e-9
        self.model.optimize()
        self.model.write("SubModel.lp")

        if self.model.Status in (GRB.INF_OR_UNBD, GRB.INFEASIBLE, GRB.UNBOUNDED):
            # log_printer.print("Exception: Sub Model Is Infeasible or Unbounded!")
            # self.model.computeIIS()
            # self.model.write("Infeasible_SubModel.ilp")
            # if self.model.IISMinimal:  # 判断 model.computeIIS() 返回的是否是包含不可行约束条件数目最少的IIS
            #     log_printer.print('IIS is minimal \n')
            # else:
            #     log_printer.print('IIS is not minimal \n')
            # log_printer.print('\n The following constraint (s) cannot be satisfied: ')
            # for c in self.model.getConstrs():
            #     if c.IISConstr:
            #         log_printer.print('%s' % c.ConstrName)
            return False
        return True

    def get_load_pattern(self):
        """
        retrieve load patterns and prepare to submit them to master model.
        """
        return {sku_id: var.Xn for sku_id, var in self.d_xs.items() if var.Xn > 0}
