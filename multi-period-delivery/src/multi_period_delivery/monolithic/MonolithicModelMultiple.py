# 多周期数学模型的gurobi代码
# 需要实现模型 2 中定义的数学模型
# 模型 2 的假设包括：多周期、SKU供给量 = 期初库存 + 生产数量 ≥ 需求量、生产基地的存储能力有上限
# 多生产基地、多经销商、多车型、多SKU
# 每家生产基地可以提供相同的SKU，不同生产基地提供的SKU之间存在交集
import os
import time
from gurobipy import GRB
import gurobipy as gp
from typing import Dict, Tuple, Set, List
from dataclasses import dataclass, field
from .optutility import LogPrinter
from .InputDataMultiple import InputDataMultiple


# 创建LogPrinter的实例
log_printer = LogPrinter(time.time())
# # 通过设置force_plain=True，强制化使用纯文本输出，例如在重定向到文件时
# # 如果需要显示格式化文本，例如字体颜色等，可以注释掉这行，并且关闭重定向
# # 需要修改optLogicMultiple.py文件中的run_model方法，启用注释部分的代码，并注释掉相对应部分的现有代码
# log_printer.set_output_mode(force_plain=True)


@dataclass
class MonolithicModelMultiple:
    data: InputDataMultiple
    # 对于多周期问题，默认规划周期长度为 1
    # 具体数值是warehouse_production.csv文件中produce_date这一列的最大值
    horizons = 1
    # # 当前的规划周期，默认值为 1
    # # 需要特别注意， day = 0 表示期初
    # # 即 day = 0, 1, ..., horizons
    # day = 1

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
    # 规划周期 t 的集合
    periods: Set[int] = field(init=False)

    # 每种SKU的体积量化数值 {sku: size}
    sku_sizes: Dict[str, float] = field(init=False)
    # 在每个周期 t 内，生产基地计划生产的SKU数量 {(plant, sku, day): qty}
    sku_prod_each_day: Dict[Tuple[str, str, int], int] = field(init=False)
    # 生产基地中SKU的期初库存数量 {(plant, sku): qty}
    sku_initial_inv: Dict[Tuple[str, str], int] = field(init=False)
    # 生产基地可以提供的SKU集合
    skus_plant: Dict[str, Set[str]] = field(init=False)

    # 经销商对SKU的需求 {(dealer, sku): qty}
    demands: Dict[Tuple[str, str], int] = field(init=False)
    # 经销商需要的SKU集合
    skus_dealer: Dict[str, Set[str]] = field(init=False)

    # 每一辆车的使用成本 {v_id: cost}
    veh_cost: Dict[int, float] = field(init=False)
    # 每一辆车的容量上限 {v_id: capacity}
    veh_cap: Dict[int, float] = field(init=False)
    # 每一辆车的最小起运量 {v_id: min_load}
    veh_min_load: Dict[int, float] = field(init=False)
    # 每个生产基地的库存上限 {plant: max_cap}
    plant_inv_limit: Dict[str, float] = field(init=False)

    # 所有车辆类型
    all_veh_types: List[str] = field(init=False)
    # 有了车辆编号，还需要在车辆编号和车辆类型之间建立联系
    # 告诉我任意一辆车的编号，通过查看这辆车的类型，就能知道
    # 这辆车的所有信息，包括使用成本、容量上限、最小起运量
    # {v_id: v_type}
    veh_type: Dict[int, str] = field(init=False)
    # 同理，告诉我任意一种车辆的类型，我能知道这种车辆类型拥有哪些编号的车辆 {v_type: {v_id}}
    veh_ids: Dict[str, Set[int]] = field(init=False)

    # 生产基地和经销商之间的供需关系映射 {(plant, dealer): {sku}}
    supply_chain: Dict[Tuple[str, str], Set[str]] = field(init=False)
    # 到这里，多周期优化问题需要的所有数据接口，已经定义完毕

    # 接下来，需要定义数学模型、目标函数、各种辅助变量和决策变量
    # 最后再定义约束条件
    # 定义gurobi模型
    model: gp.Model = field(init=False)
    # 定义目标函数
    obj: gp.LinExpr = field(init=False)

    # 定义辅助变量
    # 在周期 t 内是否使用车辆 l {(v_id, day): gp.Var}，使用 = 1，不使用 = 0
    z_lt: gp.tupledict[Tuple[int, int], gp.Var] = field(init=False)
    # 在周期 t 内车辆 l 的实际装载体积是否低于最小起运量， {(v_id, day): gp.Var}，低于 = 1， 不低于 = 0
    zz_lt: gp.tupledict[Tuple[int, int], gp.Var] = field(init=False)
    # 在周期 t 内车辆 l 最小起运量的惩罚标识变量，{(v_id, day): gp.Var}，惩罚 = 1，不惩罚 = 0
    # 只有当使用车辆 l 时，如果车辆 l 的实际装载体积是否低于最小起运量，才惩罚，否则不惩罚
    # 如果没有使用车辆 l，则不惩罚
    p_lt: gp.tupledict[Tuple[int, int], gp.Var] = field(init=False)

    # 定义决策变量
    # 在周期 t 内是否使用车辆 l 配送从生产基地 i 到经销商 j 的商品，配送 = 1， 不配送 = 0 {(plant, dealer, v_id, day): gp.Var}
    y_ijlt: gp.tupledict[Tuple[str, str, int, int], gp.Var] = field(init=False)
    # 生产基地 i 中SKU k 的库存转移到周期 t 的数量，其中 t = 0, 1, ..., T-1。
    # 特别地，s_ik0表示生产基地 i 中SKU k 的期初库存。 {(plant, sku_id, day): gp.Var}
    s_ikt: gp.tupledict[Tuple[str, str, int], gp.Var] = field(init=False)
    # 在周期 t 内从生产基地 i 运往经销商 j 的车辆 l 中装载SKU k 的数量  {(plant, dealer, sku_id, v_id, day): gp.Var}
    x_ijklt: gp.tupledict[Tuple[str, str, str, int, int], gp.Var] = field(init=False)

    # 变量的下标
    auxi_indices: List[Tuple[int, int]] = field(init=False)
    y_indices: List[Tuple[str, str, int, int]] = field(init=False)
    s_indices: List[Tuple[str, str, int]] = field(init=False)
    x_indices: List[Tuple[str, str, str, int, int]] = field(init=False)
    new_triple_ikt: Set[Tuple[str, str, int]] = field(init=False)

    # 定义需要向模型中添加的约束条件
    # 约束 1 -- 对于所有生产基地 i 和经销商 j 构成的有序对 (i, j)
    # 只要生产基地 i 在周期 t 内使用车辆 l 为经销商 j 配送商品，那么就代表车辆 l 已经被使用
    cons_veh_use: gp.tupledict[Tuple[int, int], gp.Constr] = field(
        init=False
    )  # {(v_id, day): gp.Constr}
    # 约束2 -- 在周期 t 内只有在使用车辆 l 时，才能将商品装载到车辆 l 上
    cons_sku_load: gp.tupledict[Tuple[str, str, int, int], gp.Constr] = field(
        init=False
    )  # {(plant, dealer, v_id, day): gp.Constr}
    # 约束3 -- 在周期 t 内车辆 l 的实际装载体积不小于该车辆的最小起运量
    cons_veh_min_load: gp.tupledict[Tuple[int, int], gp.Constr] = field(
        init=False
    )  # {(v_id, day): gp.Constr}
    # 约束4 -- 在周期 t 内车辆 l 最小起运量的惩罚标识变量 p_lt，与车辆 l 的使用状态变量 z_lt
    # 以及车辆 l 是否满足最小起运量的状态变量 zz_lt，三者之间应该满足的关系
    cons_auxi_relation: gp.tupledict[Tuple[int, int], gp.Constr] = field(
        init=False
    )  # {(v_id, day): gp.Constr}
    # 约束5 -- 在周期 t 内车辆 l 的实际装载体积不超过该车辆的最大装载能力
    cons_veh_cap_limit: gp.tupledict[Tuple[int, int], gp.Constr] = field(
        init=False
    )  # {(v_id, day): gp.Constr}
    # 约束6 -- 在搜有周期 t 内所有生产基地 i 使用所有车辆 l，运往某个经销商 j 的SKU k 的总量，等于经销商 j 对SKU k 的需求
    cons_dealer_demands: gp.tupledict[Tuple[str, str], gp.Constr] = field(
        init=False
    )  # {(dealer, sku_id): gp.Constr}
    # 约束7 -- 在周期 t 内，从某个生产基地 i 运往所有经销商 j 的SKU k 的总量，等于生产基地 i 中SKU k 的库存转移到周期 t-1 的数量，
    # 加上在周期 t 内，生产基地 i 的SKU k 的生产数量，再减去生产基地 i 中SKU k 的库存转移到周期 t 的数量。
    # {(plant, sku_id, day): gp.Constr}
    cons_sku_inv_transfer: gp.tupledict[Tuple[str, str, int], gp.Constr] = field(
        init=False
    )
    # 约束8 -- 在周期 t 内，生产基地 i 中剩余的所有商品数量不超过该基地的最大存储能力。
    cons_plant_inv_limit: gp.tupledict[Tuple[str, int], gp.Constr] = field(
        init=False
    )  # {(plant, day): gp.Constr}

    # 约束9 -- 添加对称性割，用于破除模型中存在的对称性，缩减解空间
    # {(v_type, v_id, v_di_1, day): gp.Constr}
    # cons_symmetry_break: gp.tupledict[Tuple[str, int, int, int], gp.Constr] = field(init=False)

    # 至此，多周期优化模型的所有相关属性都已经定义完毕
    # 接下来，要做的事情是实例化这些属性

    def __post_init__(self):
        self.horizons = self.data.horizons

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
        self.sku_prod_each_day = self.data.sku_prod_each_day
        self.sku_initial_inv = self.data.sku_initial_inv
        self.skus_plant = self.data.skus_plant
        self.demands = self.data.demands
        self.skus_dealer = self.data.skus_dealer
        self.supply_chain = self.data.supply_chain
        self.veh_cost = self.data.veh_cost
        self.veh_cap = self.data.veh_cap
        self.veh_min_load = self.data.veh_min_load
        self.plant_inv_limit = self.data.plant_inv_limit
        self.all_veh_types = self.data.all_veh_types
        self.veh_type = self.data.veh_type
        self.veh_ids = self.data.veh_ids

        self.model = gp.Model()
        self.obj = gp.LinExpr()

        # 添加辅助变量
        self.periods = {t for t in range(1, self.horizons + 1)}
        self.auxi_indices = [
            (v_id, t) for v_id in self.all_vehicles for t in self.periods
        ]
        self.z_lt = self.model.addVars(
            self.auxi_indices, lb=0, ub=1, vtype=GRB.BINARY, name="z_lt"
        )
        self.zz_lt = self.model.addVars(
            self.auxi_indices, lb=0, ub=1, vtype=GRB.BINARY, name="zz_lt"
        )
        self.p_lt = self.model.addVars(
            self.auxi_indices, lb=0, ub=1, vtype=GRB.BINARY, name="p_lt"
        )

        # 添加决策变量
        self.y_indices = [
            (plant, dealer, v_id, t)
            for (plant, dealer) in self.supply_chain
            for (v_id, t) in self.auxi_indices
        ]
        self.y_ijlt = self.model.addVars(
            self.y_indices, lb=0, ub=1, vtype=GRB.BINARY, name="y"
        )

        # 构建 (plant, dealer, sku_id) 的集合，代表生产基地 i 可以向经销商 j 提供 SKU k
        # 锁定某个经销商 j
        # 为什么要用集合？因为必须保证每个变量的下标是唯一确定的，所以要强制去除重复元素，用集合最保险
        # 在此之前，需要先计算出生产基地 i 可以提供的SKU k 的集合，包括 库存 + 生产，这个信息在skus_plant中
        # 但是是以字典保存的，需要转换为我们需要的形式
        duet_ik = {
            (plant, sku_id)
            for plant, skus in self.skus_plant.items()
            for sku_id in skus
        }
        triple_ijk = {
            (plant, dealer1, sku_id)
            for (plant, dealer1) in self.supply_chain
            for (dealer2, sku_id) in self.demands
            if dealer1 == dealer2
            if (plant, sku_id) in duet_ik
        }
        self.new_triple_ikt = {
            (plant, sku_id, t)
            for (plant, dealer, sku_id) in triple_ijk
            for t in self.periods
        }

        self.x_indices = [
            (plant, dealer, sku_id, v_id, t)
            for (plant, dealer, sku_id) in triple_ijk
            for (v_id, t) in self.auxi_indices
        ]
        self.x_ijklt = self.model.addVars(
            self.x_indices, lb=0, vtype=GRB.INTEGER, name="x"
        )

        self.s_indices = [
            (plant, sku_id, t)
            for (plant, sku_id) in duet_ik
            for t in range(self.horizons + 1)
        ]
        self.s_ikt = self.model.addVars(
            self.s_indices, lb=0, vtype=GRB.INTEGER, name="s"
        )
        # 由于s_ik0代表的是生产基地 i 中拥有的SKU k 的期初库存，所以需要 **固定** 这部分变量的取值
        # 如何实现？同时设定这部分变量的上界和下界为某个常数
        for plant, sku_id, t in self.s_indices:
            if t == 0 and (plant, sku_id) in self.sku_initial_inv:
                self.s_ikt[plant, sku_id, t].setAttr(
                    "LB", self.sku_initial_inv[plant, sku_id]
                )
                self.s_ikt[plant, sku_id, t].setAttr(
                    "UB", self.sku_initial_inv[plant, sku_id]
                )

        # 构建目标函数
        self.obj = gp.quicksum(
            self.veh_cost[v_id] * self.z_lt[v_id, t] for (v_id, t) in self.auxi_indices
        ) + gp.quicksum(
            self.param_alpha * self.veh_cost[v_id] * self.p_lt[v_id, t]
            for (v_id, t) in self.auxi_indices
        )
        # 设定目标函数的优化方向
        self.model.setObjective(self.obj, sense=GRB.MINIMIZE)

        # 添加约束条件
        # 约束 1 -- 对于所有生产基地 i 和经销商 j 构成的有序对 (i, j)
        # 只要生产基地 i 在周期 t 内使用车辆 l 为经销商 j 配送商品，那么就代表车辆 l 已经被使用
        # Create cons_veh_use with explicit unique names to avoid duplicate-name warnings
        _cons_veh_use_tmp = {}
        for v_id, t in self.auxi_indices:
            cname = f"cons_veh_use_{v_id}_{t}"
            _cons_veh_use_tmp[v_id, t] = self.model.addConstr(
                self.y_ijlt.sum("*", "*", v_id, t) == self.z_lt[v_id, t], name=cname
            )
        # convert to Gurobi tupledict for compatibility with later code
        self.cons_veh_use = gp.tupledict(_cons_veh_use_tmp)

        def _sanitize_part(p):
            return (
                str(p)
                .replace(" ", "_")
                .replace(":", "_")
                .replace(".", "_")
                .replace("-", "_")
            )

        # 约束2 -- 在周期 t 内只有在使用车辆 l 时，才能将商品装载到车辆 l 上
        _cons_sku_load_tmp = {}
        for plant, dealer, v_id, t in self.y_indices:
            cname = f"cons_sku_load_{_sanitize_part(plant)}_{_sanitize_part(dealer)}_{v_id}_{t}"
            _cons_sku_load_tmp[plant, dealer, v_id, t] = self.model.addConstr(
                self.x_ijklt.sum(plant, dealer, "*", v_id, t)
                <= self.param_BIG_NUM * self.y_ijlt[plant, dealer, v_id, t],
                name=cname,
            )
        self.cons_sku_load = gp.tupledict(_cons_sku_load_tmp)

        # 约束3 -- 在周期 t 内车辆 l 的实际装载体积不小于该车辆的最小起运量
        _cons_veh_min_load_tmp = {}
        for v_id, t in self.auxi_indices:
            cname = f"cons_veh_min_load_{v_id}_{t}"
            _cons_veh_min_load_tmp[v_id, t] = self.model.addConstr(
                gp.quicksum(
                    self.x_ijklt.sum("*", "*", sku_id, v_id, t) * self.sku_sizes[sku_id]
                    for sku_id in self.all_skus
                )
                + self.zz_lt[v_id, t] * self.param_BIG_NUM
                >= self.veh_min_load[v_id],
                name=cname,
            )
        self.cons_veh_min_load = gp.tupledict(_cons_veh_min_load_tmp)

        # 约束4 -- 在周期 t 内车辆 l 最小起运量的惩罚标识变量 p_lt，与车辆 l 的使用状态变量 z_lt
        # 以及车辆 l 是否满足最小起运量的状态变量 zz_lt，三者之间应该满足的关系
        _cons_auxi_relation_tmp = {}
        for v_id, t in self.auxi_indices:
            cname = f"cons_auxi_relation_{v_id}_{t}"
            _cons_auxi_relation_tmp[v_id, t] = self.model.addConstr(
                self.z_lt[v_id, t] + self.zz_lt[v_id, t] - 1 <= self.p_lt[v_id, t],
                name=cname,
            )
        self.cons_auxi_relation = gp.tupledict(_cons_auxi_relation_tmp)

        # 约束5 -- 在周期 t 内车辆 l 的实际装载体积不超过该车辆的最大装载能力
        _cons_veh_cap_limit_tmp = {}
        for v_id, t in self.auxi_indices:
            cname = f"cons_veh_cap_limit_{v_id}_{t}"
            _cons_veh_cap_limit_tmp[v_id, t] = self.model.addConstr(
                gp.quicksum(
                    self.x_ijklt.sum("*", "*", sku_id, v_id, t) * self.sku_sizes[sku_id]
                    for sku_id in self.all_skus
                )
                <= self.veh_cap[v_id],
                name=cname,
            )
        self.cons_veh_cap_limit = gp.tupledict(_cons_veh_cap_limit_tmp)

        # 约束6 -- 在搜有周期 t 内所有生产基地 i 使用所有车辆 l，运往某个经销商 j 的SKU k 的总量，等于经销商 j 对SKU k 的需求
        _cons_dealer_demands_tmp = {}
        for dealer, sku_id in self.demands:
            cname = (
                f"cons_dealer_demands_{_sanitize_part(dealer)}_{_sanitize_part(sku_id)}"
            )
            _cons_dealer_demands_tmp[dealer, sku_id] = self.model.addConstr(
                self.x_ijklt.sum("*", dealer, sku_id, "*", "*")
                >= self.demands[dealer, sku_id],
                name=cname,
            )
        self.cons_dealer_demands = gp.tupledict(_cons_dealer_demands_tmp)

        # 约束7 -- 在周期 t 内，从某个生产基地 i 运往所有经销商 j 的SKU k 的总量，等于生产基地 i 中SKU k 的库存转移到周期 t-1 的数量，
        # 加上在周期 t 内，生产基地 i 的SKU k 的生产数量，再减去生产基地 i 中SKU k 的库存转移到周期 t 的数量。
        _cons_sku_inv_transfer_tmp = {}
        for plant, sku_id, t in self.new_triple_ikt:
            cname = f"cons_sku_inv_transfer_{_sanitize_part(plant)}_{_sanitize_part(sku_id)}_{t}"
            _cons_sku_inv_transfer_tmp[plant, sku_id, t] = self.model.addConstr(
                self.x_ijklt.sum(plant, "*", sku_id, "*", t)
                == self.sku_prod_each_day.get((plant, sku_id, t), 0)
                + self.s_ikt[plant, sku_id, t - 1]
                - self.s_ikt[plant, sku_id, t],
                name=cname,
            )
        self.cons_sku_inv_transfer = gp.tupledict(_cons_sku_inv_transfer_tmp)

        # 约束8 -- 在周期 t 内，生产基地 i 中剩余的所有商品数量不超过该基地的最大存储能力。
        _cons_plant_inv_limit_tmp = {}
        for plant in self.plants:
            for t in self.periods:
                cname = f"cons_plant_inv_limit_{_sanitize_part(plant)}_{t}"
                _cons_plant_inv_limit_tmp[plant, t] = self.model.addConstr(
                    self.s_ikt.sum(plant, "*", t) <= self.plant_inv_limit[plant],
                    name=cname,
                )
        self.cons_plant_inv_limit = gp.tupledict(_cons_plant_inv_limit_tmp)

        # 约束9 -- 添加的对称性割，用于破除模型中存在的对称性
        # Gurobi会自动检测模型中的对称性, 并进行处理, 所以这里的对称性割可以不添加
        # 在添加时, 需要考虑对求解效率的影响
        # self.cons_symmetry_break = self.model.addConstrs((
        #     self.z_lt[v_id, t] <= self.z_lt[v_id_1, t]
        #     for v_type in self.all_veh_types
        #     for v_id, v_id_1 in zip(list(self.veh_ids[v_type]), list(self.veh_ids[v_type])[1:])
        #     for t in self.periods
        # ), name="cons_symmetry_break")

        # 启用求解日志
        self.model.setParam("OutputFlag", self.param_output_flag)
        # 设置模型求解gap
        self.model.Params.MIPGap = self.param_gap_limit
        # 设置求解时间
        self.model.Params.TimeLimit = self.param_time_limit

    def run(self):
        # 如果发现，模型在求解过程中BestBd移动缓慢，甚至一点也不变化，做法如下
        # 如果要用于算法对比实验, 不建议启用下面的参数设置, 只设置TimeLimit和MIPGap即可
        # # self.model.Params.MIPFocus = 3    # 专注于bound提升
        # self.model.Params.MIPFocus = 2    # 专注于证明最优
        # self.model.Params.Cuts = 2        # 生成更多的切平面

        # 收集模型求解的状态信息
        meta = {}

        # Helper to read variable value (compatible with .X / .Xn)
        def _var_value(v):
            try:
                return float(v.X)
            except Exception:
                try:
                    return float(v.Xn)
                except Exception:
                    try:
                        return float(v)
                    except Exception:
                        return 0.0

        # Export current/best solution to JSON for offline analysis / mapping.
        # value_getter defaults to model solution values; callback path can pass cbGetSolution getter.
        def _export_solution_json(status_tag: str, value_getter=None, obj_value=None):
            try:
                sol_file_loc = os.path.join(
                    self.data.output_file_loc, self.data.dataset_name
                )
                if not os.path.exists(sol_file_loc):
                    os.makedirs(sol_file_loc)

                sol_file_name = (
                    f"Feasible-Solution-{status_tag}-{int(time.time())}.json"
                )
                sol_file = os.path.join(sol_file_loc, sol_file_name)

                sol = {
                    "z": {},
                    "y": {},
                    "x": {},
                    "s": {},
                    "veh_type_by_id": {},
                    "gurobi_obj": None,
                }

                # veh_type_by_id mapping
                try:
                    for vid, vtype in self.veh_type.items():
                        sol["veh_type_by_id"][str(vid)] = str(vtype)
                except Exception:
                    pass

                # objective value if available
                try:
                    if obj_value is not None:
                        sol["gurobi_obj"] = float(obj_value)
                    else:
                        sol["gurobi_obj"] = float(self.model.ObjVal)
                except Exception:
                    # if no ObjVal yet, keep None
                    sol["gurobi_obj"] = None

                get_value = value_getter if callable(value_getter) else _var_value

                # z variables
                try:
                    for k, var in self.z_lt.items():
                        v_id, t = k
                        sol["z"][f"{v_id}|{t}"] = get_value(var)
                except Exception:
                    pass

                # y variables
                try:
                    for k, var in self.y_ijlt.items():
                        plant, dealer, v_id, t = k
                        sol["y"][f"{plant}|{dealer}|{v_id}|{t}"] = get_value(var)
                except Exception:
                    pass

                # x variables
                try:
                    for k, var in self.x_ijklt.items():
                        plant, dealer, sku, v_id, t = k
                        sol["x"][f"{plant}|{dealer}|{sku}|{v_id}|{t}"] = int(
                            round(get_value(var))
                        )
                except Exception:
                    pass

                # s variables
                try:
                    for k, var in self.s_ikt.items():
                        plant, sku, t = k
                        sol["s"][f"{plant}|{sku}|{t}"] = int(round(get_value(var)))
                except Exception:
                    pass

                # write json
                with open(sol_file, "w", encoding="utf-8") as f:
                    import json as _json

                    _json.dump(sol, f, indent=2, ensure_ascii=False)
                log_printer.print(f"Feasible solution exported to {sol_file}")
            except Exception as e:
                log_printer.print(f"Failed to export feasible solution JSON: {e}")

        self.model.optimize()
        self.model.write("MonolithicModelMultiple.lp")

        if self.model.Status in (GRB.INF_OR_UNBD, GRB.INFEASIBLE, GRB.UNBOUNDED):
            self.model.computeIIS()
            self.model.write("Infeasible_MonolithicModelMultiple.ilp")

            if (
                self.model.IISMinimal
            ):  # 判断 model.computeIIS() 返回的是否是包含不可行约束条件数最少的IIS
                log_printer.print("IIS is minimal \n")
            else:
                log_printer.print("IIS is not minimal \n")
            log_printer.print("\n The following constraint (s) cannot be satisfied :")
            for c in self.model.getConstrs():
                if c.IISConstr:
                    log_printer.print("%s" % c.ConstrName)

            if self.model.Status == GRB.INFEASIBLE:
                meta["status"] = "infeasible"
            elif self.model.Status == GRB.UNBOUNDED:
                meta["status"] = "unbounded"
            elif self.model.Status == GRB.INF_OR_UNBD:
                meta["status"] = "inf_or_unbd"

            # Attempt to export any available solution (unlikely), tag as infeasible
            _export_solution_json("infeasible")
            return False, meta

        elif self.model.Status == GRB.TIME_LIMIT:
            log_printer.print("Time limit is reached! Solving process is stopped.")
            # Export the incumbent solution if present
            _export_solution_json("time_limit")
            try:
                self.model.terminate()
            except Exception:
                pass
            meta["status"] = "time_limit"
            return False, meta

        elif self.model.Status == GRB.INTERRUPTED:
            log_printer.print("Optimization is interrupted by user!")
            # Export current solution if any
            _export_solution_json("interrupted")
            meta["status"] = "interrupted"
            return False, meta
        else:  # model is optimal
            self.print_sol_summary()
            # export final optimal solution
            _export_solution_json("optimal")
            meta["status"] = "optimal"
            return True, meta

    def save_feasible_sol(self):
        """
        将Gurobi找到的可行解以json格式保存到文件中,
        该函数在模型求解状态为TIME_LIMIT时调用, 但可能会非常耗时
        批量运行时不建议启用该函数
        """
        sol_file_loc = os.path.join(self.data.output_file_loc, self.data.dataset_name)
        if not os.path.exists(sol_file_loc):
            os.makedirs(sol_file_loc)
        sol_file_name = "Feasible-Solution.json"
        sol_file = os.path.join(sol_file_loc, sol_file_name)
        self.model.write(sol_file)

    def print_sol_summary(self):
        if self.model.Status == GRB.OPTIMAL:
            log_printer.print(
                "Monolithic Model for multiple periods is solved to be optimal."
            )
            log_printer.print(
                f"Total cost of optimal solution is {self.model.ObjVal:.2f}"
            )

            total_veh_num = sum(var.Xn for var in self.z_lt.values())
            log_printer.print(f"Total number of vehicles used is {total_veh_num}")

            veh_punished_num = sum(var.Xn for var in self.p_lt.values())
            log_printer.print(
                f"Among used vehicles, there are {veh_punished_num} vehicle(s) "
                f"that don't satisfy veh_min_load constraint and need to be punished."
            )

            total_ship_num = sum(var.Xn for var in self.x_ijklt.values())
            log_printer.print(f"Total SKU ship amount is {total_ship_num}")
