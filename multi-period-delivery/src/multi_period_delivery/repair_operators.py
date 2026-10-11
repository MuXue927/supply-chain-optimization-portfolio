"""
repair_operators.py

模块说明（中文详解）
- 本模块实现并暴露一组 ALNS 修复算子（repair operators），供主流程注册使用。
- 设计原则：
  1. 不修改 multi_period_delivery 其他模块的行为（只读依赖），尽量复用 multi_period_delivery 中已有的工具函数。
  2. 保持兼容性：保留对外算子名称与签名，避免主流程导入失败。
  3. 将复杂/模拟行为通过 state.copy() 在副本上执行，以避免修改核心状态或要求修改 alnsopt 接口。
  4. 对可能有较高开销的模拟（例如 regret 算子）进行显式注释并提出优化建议。

暴露接口（对外使用时请保持这些名称不变）
- greedy_repair(partial, rng)
- local_search_repair(partial, rng, max_iter=10, time_limit=5.0)
- inventory_balance_repair(partial, rng)
- smart_batch_repair(partial, rng)
- infeasible_repair(partial, rng)  # 可行性纠偏 wrapper，实际调用 infeasible_removal
- regret_based_repair(partial, rng, k=2, topN=6, time_limit=10.0)
- RegretBasedRepairOperator (可直接传入 alns.add_repair_operator)

说明：若按“核心修复算子”统计，本模块对应 8 类；infeasible_repair 仅作为
兼容性包装器存在，不单独计入核心修复算子类别。
"""

# -------------------------
# 标准库与类型导入（已整理）
# -------------------------
import time
from typing import Optional

# -------------------------
# 第三方库导入（明确用途）
# -------------------------
import numpy as np
from numpy import random as rnd
import logging

# -------------------------
# 包内模块导入（以相对导入保证与 multi_period_delivery 包兼容）
# -------------------------
from .alnsopt import SolutionState
from .vehicle import Vehicle
from .destroy_operators import (
    infeasible_removal,
)  # 用于调用 infeasible_removal 外部实现
from .inventory_utils import precompute_plant_day_inventory
from .alns_config import default_config as ALNSConfig

# helper functions moved to repair_utils
from .repair_utils import (
    _construct_resource_availability,
    _construct_resource_pool,
    get_unmet_demand,
    veh_loading,
    precompute_total_available,
    build_resource_pool,
    get_adaptive_parameters,
    post_repair,
    compute_improvement,
    compute_improvement_fast_add_vehicle,
    compute_improvement_delta,
    _get_unsatisfied_demands,
    _create_smart_batches,
    _process_demand_batch,
    _force_load_remaining_demands,
    select_vehicle_type_for_volume,
    find_existing_route_vehicles,
    commit_load_with_tracking,
)

# module logger
logger = logging.getLogger(__name__)


def _coerce_int(value, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except Exception:
        try:
            return int(np.array(value).flat[0])
        except Exception:
            return int(default)


def _coerce_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        try:
            return float(np.array(value).flat[0])
        except Exception:
            return float(default)


def _safe_prev_obj(state) -> float:
    try:
        value = state.objective()
    except Exception:
        return 0.0
    return float(value if value is not None else 0.0)


# ---------------------------
# 修复算子实现（对外调用）
# 说明：所有算子均以 partial/state 为输入，并返回修复后的 state（有些算子会 copy 后返回副本）
# ---------------------------


# Wrapper & factory utilities for repair operators
class RepairOperatorWrapper:
    """
    Wrapper that binds fixed keyword arguments to a repair operator function or callable.
    Provides a stable __name__ for ALNS logging and is callable with (state, rng, **kwargs).
    """

    def __init__(self, func_or_callable, name: Optional[str] = None, **fixed_kwargs):
        self.func = func_or_callable
        self.fixed_kwargs = fixed_kwargs or {}
        # Ensure ALNS logging can read a friendly name
        if name is None:
            self.__name__ = getattr(func_or_callable, "__name__", "repair_op")
        else:
            self.__name__ = name

    def __call__(self, state, rng, **kwargs):
        merged = {}
        merged.update(self.fixed_kwargs)
        if kwargs:
            merged.update(kwargs)
        return self.func(state, rng, **merged)

    def __repr__(self):
        return f"<RepairOperatorWrapper {self.__name__} fixed={self.fixed_kwargs}>"


def wrap_repair_no_args(func):
    """Wrap a repair function that does not require extra fixed kwargs."""
    func_name = getattr(func, "__name__", None)
    return RepairOperatorWrapper(
        func, name=func_name if isinstance(func_name, str) else "repair_op"
    )


def create_local_search_repair(max_iter: Optional[int] = None):
    """Factory that returns a wrapper for local_search_repair with max_iter bound."""
    defaults = ALNSConfig.get_operator_default("local_search_repair").get("params", {})
    max_iter = max_iter if max_iter is not None else defaults.get("max_iter", 10)
    name = "local_search_repair"
    return RepairOperatorWrapper(local_search_repair, name=name, max_iter=max_iter)


def create_smart_batch_repair(
    max_iter: Optional[int] = None,
    batch_size: Optional[int] = None,
    timeout: Optional[float] = None,
):
    """Factory that returns a wrapper for smart_batch_repair with bound parameters."""
    defaults = ALNSConfig.get_operator_default("smart_batch_repair").get("params", {})
    max_iter = max_iter if max_iter is not None else defaults.get("max_iter", 10)
    batch_size = (
        batch_size if batch_size is not None else defaults.get("batch_size", 10)
    )
    timeout = timeout if timeout is not None else defaults.get("timeout", None)
    name = "smart_batch_repair"
    return RepairOperatorWrapper(
        smart_batch_repair,
        name=name,
        max_iter=max_iter,
        batch_size=batch_size,
        timeout=timeout,
    )


def create_regret_repair(
    k: Optional[int] = None,
    topN: Optional[int] = None,
    time_limit: Optional[float] = None,
):
    """
    Return an instance of RegretBasedRepairOperator with readable __name__.
    RegretBasedRepairOperator is already a callable class, so instantiate and set name.
    """
    defaults = ALNSConfig.get_operator_default("regret_based_repair").get("params", {})
    k = _coerce_int(k if k is not None else defaults.get("k", 2), 2)
    topN = _coerce_int(topN if topN is not None else defaults.get("topN", 6), 6)
    time_limit = _coerce_float(
        time_limit if time_limit is not None else defaults.get("time_limit", 10.0), 10.0
    )
    inst = RegretBasedRepairOperator(k=k, topN=topN, time_limit=time_limit)
    inst.__name__ = "regret_based_repair"
    return inst


def greedy_repair(
    partial: SolutionState,
    rng: rnd.Generator,
    demand_weight: Optional[float] = None,
    stock_weight: Optional[float] = None,
) -> SolutionState:
    """
    重写后的 greedy_repair:
    - 预计算 total_available 与 resource_pool
    - 在内循环中只操作 resource_pool 并调用 veh_loading(..., skip_compute_inventory=True)
    - 整体插入结束后统一调用 post_repair 进行一次性 compute_inventory 与上报
    """
    t0 = time.time()
    logger.debug("[OPLOG] 开始执行 greedy_repair 算子")
    state = partial
    data = state.data

    prev_obj = None
    try:
        prev_obj = state.objective()
    except Exception:
        prev_obj = None

    shipped = state.compute_shipped()
    supply_chain = data.construct_supply_chain()

    # 预计算以减少重复遍历
    total_available_map = precompute_total_available(state, data, supply_chain)
    resource_pool = build_resource_pool(state, data)

    defaults = ALNSConfig.get_operator_default("greedy_repair").get("params", {})

    # Scheme A: prefer kwargs -> tuner -> defaults; clamp to [0,1]
    params = {}
    if demand_weight is None or stock_weight is None:
        param_tuner = getattr(state, "param_tuner", None)
        if param_tuner is not None:
            try:
                params = (
                    get_adaptive_parameters("greedy_repair", param_tuner=param_tuner)
                    or {}
                )
            except Exception:
                params = {}
    if demand_weight is None:
        try:
            demand_weight = float(
                params.get("demand_weight", defaults.get("demand_weight", 0.8))
            )
        except Exception:
            demand_weight = float(defaults.get("demand_weight", 0.8))
    else:
        demand_weight = float(demand_weight)
    if stock_weight is None:
        try:
            stock_weight = float(
                params.get("stock_weight", defaults.get("stock_weight", 0.2))
            )
        except Exception:
            stock_weight = float(defaults.get("stock_weight", 0.2))
    else:
        stock_weight = float(stock_weight)
    demand_weight = max(0.0, min(1.0, demand_weight))
    stock_weight = max(0.0, min(1.0, stock_weight))

    # 1) 构建未满足需求及优先级
    unsatisfied = []
    for (dealer, sku_id), demand in data.demands.items():
        shipped_qty = shipped.get((dealer, sku_id), 0)
        if demand > 0 and shipped_qty < demand:
            unmet = demand - shipped_qty
            demand_ratio = unmet / demand
            total_available = total_available_map.get((dealer, sku_id), 0)
            stock_urgency = (
                1.0 if total_available == 0 else min(1.0, unmet / total_available)
            )
            priority = demand_weight * demand_ratio + stock_weight * stock_urgency
            unsatisfied.append(((dealer, sku_id), unmet, priority))

    unsatisfied.sort(key=lambda x: x[2], reverse=True)

    # 2) 按优先级分配：使用 resource_pool 做本地库存检查并递减
    inserted_vehicles = []
    for (dealer, sku_id), remain_demand, _ in unsatisfied:
        if remain_demand <= 0:
            continue
        plants = [
            plant
            for (plant, dealer_key), skus in supply_chain.items()
            if dealer_key == dealer and sku_id in skus
        ]
        for plant in plants:
            if remain_demand <= 0:
                break
            for day in range(1, data.horizons + 1):
                available = resource_pool.get((plant, sku_id, day), 0)
                if available <= 0 or remain_demand <= 0:
                    continue

                # 先填充同线路已有车辆，减少新开车数量
                for existing in find_existing_route_vehicles(state, plant, dealer, day):
                    if remain_demand <= 0 or available <= 0:
                        break
                    request_qty = min(remain_demand, available)
                    try:
                        actual_loaded, new_vs = commit_load_with_tracking(
                            state,
                            existing,
                            sku_id,
                            request_qty,
                            day,
                            skip_compute_inventory=True,
                        )
                    except Exception:
                        continue
                    if actual_loaded <= 0:
                        continue
                    if new_vs:
                        inserted_vehicles.extend(new_vs)
                    available = max(0, available - actual_loaded)
                    resource_pool[(plant, sku_id, day)] = available
                    remain_demand -= actual_loaded

                if available <= 0 or remain_demand <= 0:
                    continue

                # 尝试不同车型（按容量降序）
                sku_size = data.sku_sizes.get(sku_id, 0)
                if sku_size <= 0:
                    sku_size = 1.0
                request_qty = min(remain_demand, available)
                if request_qty <= 0:
                    continue
                req_volume = float(request_qty) * float(sku_size)
                veh_type = select_vehicle_type_for_volume(req_volume, data)
                vehicle = Vehicle(plant, dealer, veh_type, day, data)
                try:
                    actual_loaded, new_vs = commit_load_with_tracking(
                        state,
                        vehicle,
                        sku_id,
                        request_qty,
                        day,
                        skip_compute_inventory=True,
                    )
                except Exception:
                    try:
                        actual_loaded, new_vs = commit_load_with_tracking(
                            state,
                            vehicle,
                            sku_id,
                            request_qty,
                            day,
                            skip_compute_inventory=False,
                        )
                    except Exception:
                        continue

                if actual_loaded > 0:
                    if new_vs:
                        inserted_vehicles.extend(new_vs)
                    else:
                        inserted_vehicles.append(vehicle)
                    resource_pool[(plant, sku_id, day)] = max(
                        0,
                        resource_pool.get((plant, sku_id, day), 0) - actual_loaded,
                    )
                    remain_demand -= actual_loaded
                if remain_demand <= 0:
                    break
            if remain_demand <= 0:
                break

    # 3) 批量后处理：一次性 compute_inventory 与 上报
    prev_obj_value: float = float(prev_obj if prev_obj is not None else 0.0)
    post_repair(
        state,
        inserted_vehicles,
        op_name="greedy_repair",
        prev_obj=prev_obj_value,
        t0=t0,
    )

    elapsed = time.time() - t0
    try:
        logger.debug(
            f"[OPLOG] greedy_repair: 插入 {len(inserted_vehicles)} 辆车 ({elapsed:.4f}s)"
        )
    except Exception:
        pass

    return state


def local_search_repair(
    partial: SolutionState, rng: rnd.Generator, max_iter: Optional[int] = None
) -> SolutionState:
    """
    局部搜索修复, Local Search Repair:
    - 在 greedy_repair 的基础上进行局部改进尝试, 例如替换车型以节省成本或提高装载效率
    设计思路:
    - 采用贪心+替换的启发式：遍历车辆，尝试使用更小容量车型（只要满足装载约束）替换，若目标函数变好则保留
    - 返回一个新的 state, 使用 partial.copy() 并在副本上操作
    注意: max_iter 参数控制最大迭代次数, 避免过长运行, 属于保护机制, 不应该纳入参数调优的范围
    """
    t0 = time.time()
    logger.debug("[OPLOG] 开始执行 local_search_repair 算子")
    # 记录进入局部搜索前的目标值（用于 post_repair 的改进计算与上报）
    prev_obj_saved: float = _safe_prev_obj(partial)
    state = partial.copy()
    state = greedy_repair(state, rng)
    data = state.data
    improved = True
    iter_count = 0

    num_demands = len(data.demands)
    # 动态调整最大迭代次数，仅当未显式传入时
    if max_iter is None:
        max_iter = min(50, max(5, num_demands // 20))
    else:
        try:
            max_iter = int(max_iter)
        except Exception:
            max_iter = min(50, max(5, num_demands // 20))
        max_iter = max(1, max_iter)

    while improved and iter_count < max_iter:
        iter_count += 1
        improved = False

        old_obj = state.objective()
        # 遍历车辆并尝试车型替换
        for i, veh in enumerate(list(state.vehicles)):
            current_load = state.compute_veh_load(veh)  # 计算当前车辆的总装载量
            cost = data.veh_type_cost[veh.type]  # 当前车型的成本

            for veh_type in list(
                data.all_veh_types
            ):  # 尝试所有车型, 但不包括当前车型, 进行替换
                if veh_type == veh.type:
                    continue

                cond1 = (
                    data.veh_type_cap[veh_type] >= current_load
                )  # 条件1: 新车型容量足够
                cond2 = (
                    current_load >= data.veh_type_min_load[veh_type]
                )  # 条件2: 新车型满足最小装载要求
                cond3 = data.veh_type_cost[veh_type] < cost  # 条件3: 新车型成本更低

                if cond1 and cond2 and cond3:
                    # 构造新车辆并尝试替换
                    new_veh = Vehicle(
                        veh.fact_id, veh.dealer_id, veh_type, veh.day, data
                    )
                    new_veh.cargo = veh.cargo.copy()
                    # 同步已装载体积与剩余容量，避免后续重复遍历 cargo 计算
                    try:
                        loaded = 0
                        for (sku_k, d_k), q_k in new_veh.cargo.items():
                            loaded += q_k * data.sku_sizes.get(sku_k, 0)
                        new_veh._loaded_volume = loaded
                        new_veh.capacity = max(0, new_veh.max_capacity - loaded)
                    except Exception:
                        # 任何异常下保持来自原车辆的缓存值以保证兼容性
                        new_veh._loaded_volume = getattr(veh, "_loaded_volume", 0)
                        new_veh.capacity = getattr(
                            veh, "capacity", new_veh.max_capacity
                        )
                    old_veh = state.vehicles[i]
                    state.vehicles[i] = new_veh
                    try:
                        new_obj = state.objective()
                        if new_obj < old_obj - 1e-12:
                            improved = True
                            old_obj = new_obj
                        else:
                            # 回退
                            state.vehicles[i] = old_veh
                    except Exception:
                        # 在异常情况下确保回退到旧车辆并继续
                        state.vehicles[i] = old_veh
                        continue

    # 由于该算子实际并没有插入新车辆, 所以 inserted 为空列表
    post_repair(
        state, [], op_name="local_search_repair", prev_obj=prev_obj_saved, t0=t0
    )

    elapsed = time.time() - t0
    try:
        logger.debug(
            f"[OPLOG] local_search_repair: 只尝试进行车型替换, 未插入新车辆, 完成 ({elapsed:.4f}s)"
        )
    except Exception:
        pass

    return state


def inventory_balance_repair(
    partial: SolutionState, rng: rnd.Generator
) -> SolutionState:
    """
    基于库存平衡的修复, Inventory Balance Repair:
    - 使用 inventory_utils.precompute_plant_day_inventory 预计算每个 (plant, day) 的总库存
    - 按库存多->少的顺序从附近（同厂）工厂分配车辆来满足需求，倾向于优先使用库存充足的工厂
    - 该算子直接在传入 state 上提交修改, commit=True
    """
    t0 = time.time()
    logger.debug("[OPLOG] 开始执行 inventory_balance_repair 算子")
    state = partial
    data = state.data

    prev_obj_saved: float = _safe_prev_obj(state)

    unsatisfied = get_unmet_demand(state)

    # 预计算工厂-日库存（提高性能，避免多次遍历 s_ikt）
    plant_inventory = precompute_plant_day_inventory(state)
    # 仅保留 day >= 1 的条目, 问题域特定约定
    plant_inventory = {
        (plant, day): inv for (plant, day), inv in plant_inventory.items() if day >= 1
    }

    # 按库存降序遍历，每个工厂按其覆盖的经销商分配车辆
    sorted_plant_days = sorted(
        plant_inventory.items(), key=lambda x: x[1], reverse=True
    )
    supply_chain = data.construct_supply_chain()

    # 记录插入的车辆, 用于 post_repair
    inserted_vehicles = []

    for (plant, day), _ in sorted_plant_days:
        dealers = [dealer for (p, dealer), skus in supply_chain.items() if p == plant]
        for dealer in dealers:
            # 构建受限 orders：每个 SKU 限制为未满足量与该 (plant,sku,day) 的可用量
            candidate_orders = {}
            total_volume = 0.0
            for (dealer_id, sku_id), qty in unsatisfied.items():
                if dealer_id != dealer:
                    continue
                avail = _construct_resource_availability(state, plant, sku_id, day)
                take = min(qty, avail)
                if take <= 0:
                    continue
                candidate_orders[sku_id] = take
                total_volume += take * data.sku_sizes.get(sku_id, 0)
            if not candidate_orders:
                continue
            # 使用确定性 best-fit 车型，避免随机车型导致车辆碎片化
            veh_type = select_vehicle_type_for_volume(total_volume, data)
            # 若总货量超出车辆容量，按比例缩放到车辆容量内（保守策略）
            cap = data.veh_type_cap[veh_type]
            if total_volume > cap and total_volume > 0:
                scale = cap / total_volume
                for sku in list(candidate_orders.keys()):
                    candidate_orders[sku] = max(0, int(candidate_orders[sku] * scale))
                candidate_orders = {k: v for k, v in candidate_orders.items() if v > 0}
                if not candidate_orders:
                    continue
            vehicle = Vehicle(plant, dealer, veh_type, day, data)
            try:
                before_len = len(state.vehicles)
                veh_loading(
                    state,
                    vehicle,
                    candidate_orders,
                    commit=True,
                    skip_compute_inventory=True,
                )
                # 捕获 veh_loading 实际加入到 state.vehicles 的所有新车辆
                new_vs = state.vehicles[before_len:]
                if new_vs:
                    inserted_vehicles.extend(new_vs)
                    # 使用新增车辆的实际装载量更新 unsatisfied
                    for sku_id in list(candidate_orders.keys()):
                        loaded = sum(v.cargo.get((sku_id, day), 0) for v in new_vs)
                        unsatisfied[(dealer, sku_id)] = max(
                            0, unsatisfied[(dealer, sku_id)] - loaded
                        )
            except Exception:
                # 回退到不抑制 compute_inventory 的尝试
                try:
                    before_len = len(state.vehicles)
                    veh_loading(
                        state,
                        vehicle,
                        candidate_orders,
                        commit=True,
                        skip_compute_inventory=False,
                    )
                    new_vs = state.vehicles[before_len:]
                    if new_vs:
                        inserted_vehicles.extend(new_vs)
                        for sku_id in list(candidate_orders.keys()):
                            loaded = sum(v.cargo.get((sku_id, day), 0) for v in new_vs)
                            unsatisfied[(dealer, sku_id)] = max(
                                0, unsatisfied[(dealer, sku_id)] - loaded
                            )
                except Exception:
                    continue

    post_repair(
        state,
        inserted_vehicles,
        op_name="inventory_balance_repair",
        prev_obj=prev_obj_saved,
        t0=t0,
    )

    elapsed = time.time() - t0
    try:
        logger.debug(
            f"[OPLOG] inventory_balance_repair: 插入 {len(inserted_vehicles)} 辆车, 完成 ({elapsed:.4f}s)"
        )
    except Exception:
        pass

    return state


def infeasible_repair(partial: SolutionState, rng: rnd.Generator) -> SolutionState:
    """
    不可行解快速修复的兼容 wrapper:
    - 直接调用 multi_period_delivery.destroy_operators.infeasible_removal, 外部已实现的移除/修复算子
    - 本 wrapper 在入口与返回处增加 OPLOG 日志，便于跟踪修复流程
    """
    t0 = time.time()
    logger.debug(
        "[OPLOG] 开始执行 infeasible_repair(wrapper) 算子 -> 调用 infeasible_removal"
    )
    result = infeasible_removal(partial, rng)
    elapsed = time.time() - t0
    try:
        logger.debug(f"[OPLOG] infeasible_repair(wrapper): 返回 ({elapsed:.4f}s)")
    except Exception:
        pass
    return result


def smart_batch_repair(
    partial: SolutionState,
    rng: rnd.Generator,
    max_iter: Optional[int] = None,
    batch_size: Optional[int] = None,
    timeout: Optional[float] = None,
) -> SolutionState:
    """
    Smart Batch Repair:
    - 将未满足需求划分为若干批次, 即 batch, 使用资源池 resource_pool 跟踪 (plant,sku,day) 的可用量
    - 对每个批次使用启发式规则 _find_best_allocation + greedy 装载 veh_loading
    - 适用于大量需求场景，通过批次与资源池减少重复计算
    注意: max_iter, timeout 参数控制最大迭代次数与时间, 避免过长运行, 属于保护机制, 不应该纳入参数调优的范围
    """
    t0 = time.time()
    logger.debug("[OPLOG] 开始执行 smart_batch_repair 算子")
    state = partial
    data = state.data

    prev_obj_saved: float = _safe_prev_obj(state)

    num_demands = len(data.demands)

    # 动态调整时间限制, 避免过长运行
    defaults = ALNSConfig.get_operator_default("smart_batch_repair").get("params", {})
    max_iter = max(
        1,
        _coerce_int(
            max_iter if max_iter is not None else defaults.get("max_iter", 10), 10
        ),
    )
    batch_size = max(
        1,
        _coerce_int(
            batch_size if batch_size is not None else defaults.get("batch_size", 10), 10
        ),
    )
    timeout = _coerce_float(
        timeout
        if timeout is not None
        else defaults.get("timeout", min(60.0, max(10.0, num_demands * 0.1))),
        min(60.0, max(10.0, num_demands * 0.1)),
    )

    resource_pool = _construct_resource_pool(state, data)

    iterations = 0

    total_inserted = []  # 记录所有插入的车辆, 用于 post_repair

    while iterations < max_iter and (timeout is None or time.time() - t0 < timeout):
        iterations += 1
        unsatisfied = _get_unsatisfied_demands(state, data)
        if not unsatisfied:
            break
        batches = _create_smart_batches(unsatisfied, data, batch_size)
        progress = False
        for batch in batches:
            if timeout is not None and time.time() - t0 > timeout:
                break
            ok, inserted_vehicles = _process_demand_batch(
                state, batch, resource_pool, data, rng
            )
            if inserted_vehicles:
                total_inserted.extend(inserted_vehicles)
            if ok:
                progress = True
        if not progress:
            break

    # 最后尝试填充剩余需求
    if timeout is None or time.time() - t0 < timeout:
        inserted_vehicles = _force_load_remaining_demands(state, data)
        if inserted_vehicles:
            total_inserted.extend(inserted_vehicles)

    post_repair(
        state,
        total_inserted,
        op_name="smart_batch_repair",
        prev_obj=prev_obj_saved,
        t0=t0,
    )

    elapsed = time.time() - t0
    try:
        logger.debug(
            f"[OPLOG] smart_batch_repair: 插入 {len(total_inserted)} 辆车 ({elapsed:.4f}s)"
        )
    except Exception:
        pass

    return state


# ---------------------------
# Regret 算子（后悔值法）实现
# ---------------------------
def regret_based_repair(
    partial: SolutionState,
    rng: rnd.Generator,
    k: Optional[int] = None,
    topN: Optional[int] = None,
    time_limit: Optional[float] = None,
) -> SolutionState:
    """
    注意: time_limit 参数控制最大运行时间, 属于限时保护机制, 不应该纳入参数调优的范围
    """
    t0 = time.time()
    logger.debug("[OPLOG] 开始执行 regret_based_repair 算子")
    state = partial
    data = state.data

    # local cache of frequently accessed attributes to reduce hot-path lookups
    horizons = getattr(data, "horizons", 0)
    veh_type_cost = getattr(data, "veh_type_cost", {})
    sku_sizes = getattr(data, "sku_sizes", {})
    veh_types_list = list(getattr(data, "all_veh_types", []))
    try:
        veh_types_sorted_global = sorted(
            veh_types_list, key=lambda x: data.veh_type_cap[x], reverse=True
        )
    except Exception:
        veh_types_sorted_global = veh_types_list
    resource_pool_get = None

    prev_obj_saved = None
    try:
        prev_obj_saved = state.objective()
    except Exception:
        prev_obj_saved = None

    inserted_vehicles = []  # 记录插入的车辆, 用于日志或分析

    num_demands = len(data.demands)
    # 动态调整时间限制, 避免过长运行
    defaults = ALNSConfig.get_operator_default("regret_based_repair").get("params", {})
    k = max(1, _coerce_int(k if k is not None else defaults.get("k", 2), 2))
    topN = max(1, _coerce_int(topN if topN is not None else defaults.get("topN", 6), 6))
    time_limit = _coerce_float(
        time_limit
        if time_limit is not None
        else defaults.get("time_limit", min(60.0, max(5.0, num_demands * 0.2))),
        min(60.0, max(5.0, num_demands * 0.2)),
    )
    time_limit = min(60.0, max(5.0, time_limit))
    repair_baseline_obj = prev_obj_saved

    while time.time() - t0 < time_limit:
        # 获取当前未满足需求
        removal = get_unmet_demand(state)
        if not removal:
            break
        # evaluate candidates against the current repaired state snapshot
        try:
            prev_obj = state.objective()
        except Exception:
            prev_obj = None
        regret_list = []

        # 预先缓存供应链映射与一次性构建的资源池以避免内层循环中频繁计算可用量
        supply_chain = data.construct_supply_chain()
        resource_pool = _construct_resource_pool(state, data)
        # 本地缓存热点只读字典以减少属性查找开销
        _sku_sizes = getattr(data, "sku_sizes", {})
        _veh_type_cap = getattr(data, "veh_type_cap", {})
        try:
            _param_pun_factor1 = float(getattr(data, "param_pun_factor1", 1.0))
        except Exception:
            _param_pun_factor1 = 1.0

        # Local bindings to avoid repeated attribute lookups in hot loops
        # bind resource pool accessors; prefer direct day_map access to avoid tuple-key hashing
        resource_pool_get = resource_pool.get
        sku_sizes = _sku_sizes
        veh_type_cap = _veh_type_cap
        _data_param_pun = _param_pun_factor1

        # ensure shipped_cache exists to increase chance of delta-path usage
        try:
            if getattr(state, "shipped_cache", None) is None:
                state.compute_shipped()
        except Exception:
            pass

        # Stronger cross-demand memoization keyed by (dealer + candidate key) and current state snapshot.
        # This allows reuse of both fast estimates and exact simulation results across different
        # unmet-demand entries within the same regret iteration (state hasn't changed).
        # state_snapshot is intentionally simple (num vehicles, last_objective) to detect structural changes.
        global_memo = {}
        state_snapshot = (len(state.vehicles), getattr(state, "last_objective", None))

        # 针对每个未满足需求生成候选并在副本上模拟
        # 为避免在每个 demand 上扫描 supply_chain，提前构建 (dealer,sku)->plants 映射并复用
        plants_for_dealer_sku = {}
        for (plant_key, dealer_key), skus in supply_chain.items():
            for s in skus:
                plants_for_dealer_sku.setdefault((dealer_key, s), []).append(plant_key)
        # 预计算 days range 以避免在热循环中反复构造 range 对象
        days_range = range(1, horizons + 1)
        # cache day_map per (plant, sku) for the duration of this regret iteration to avoid repeated
        # ResourcePool.get_day_map calls which are hot-path and expensive due to tuple-key hashing.

        for (dealer, sku_id), remain_qty in list(removal.items()):
            candidates = []
            # use mapping to get available plants quickly (avoids repeated supply_chain iteration)
            available_plants = plants_for_dealer_sku.get((dealer, sku_id), [])

            # 本循环是性能热点：尽量减少昂贵的 state.copy()/veh_loading()/compute_improvement 调用
            # 1) 对 topN/k 做安全上限（来自全局配置）
            # 2) 使用更紧凑的 memo key（对 est_qty 做分箱）以提高缓存命中率
            # 3) 将 available 提前计算避免在 veh_type 内重复查询
            # reuse global memo across demands (includes dealer + state snapshot)
            max_topn_cap = getattr(ALNSConfig, "REGRET_MAX_TOPN", None)
            max_k_cap = getattr(ALNSConfig, "REGRET_MAX_K", None)
            if max_topn_cap is not None:
                topN = min(topN, int(max_topn_cap))
            if max_k_cap is not None:
                k = min(k, int(max_k_cap))

            # 评估车辆类型时优先尝试大容量车型以便更快找到高回报候选
            # reuse the global pre-sorted vehicle type list computed once per operator call
            veh_types_sorted = veh_types_sorted_global

            # Two-stage evaluation to avoid expensive state.copy() for every candidate:
            # 1) Fast heuristic score to filter promising candidates
            # 2) Full simulation (state.copy() + veh_loading + compute_improvement) for top-K heuristics
            # Use a bounded heap to keep only the top `max_sim` candidates by heuristic_score.
            # This avoids building a huge candidate_pool and sorting it fully.
            max_sim = int(
                getattr(ALNSConfig, "REGRET_SIM_MAX_SIMULATE", max(6, topN * 2))
            )
            # Maintain a bounded array (top_list) instead of heapq to avoid frequent small-object allocation.
            # top_list stores tuples: (score, key, plant, veh_type, day, est_qty)
            top_list = []
            _seen_local = set()
            # Precompute day_maps for all available_plants once per demand to avoid repeated get_day_map calls.
            # This moves the potentially-expensive resource_pool_get_day_map call out of the inner veh_type loop.
            for plant in available_plants:
                # obtain a day->available view for this (plant, sku) to avoid repeated tuple-key lookups
                # generator over days to avoid list allocations
                day_items = (
                    (d, resource_pool_get((plant, sku_id, d), 0)) for d in days_range
                )

                # pull sku_size once per (dealer,sku) candidate to avoid repeated dict lookups
                sku_size = sku_sizes.get(sku_id, 0)

                for day, available_val in day_items:
                    if available_val <= 0:
                        continue

                    for veh_type in veh_types_sorted:
                        cap = veh_type_cap.get(veh_type, 0)
                        max_qty_by_cap = int(cap // sku_size) if sku_size > 0 else 0
                        est_qty = min(remain_qty, available_val, max_qty_by_cap)
                        if est_qty <= 0:
                            continue

                        # bucketize estimated qty to improve memoization hit-rate
                        if est_qty <= 5:
                            qty_bucket = int(est_qty)
                        else:
                            bucket_size = 5
                            qty_bucket = int((est_qty // bucket_size) * bucket_size)

                        # include dealer in the key so memoization is safe across different demands
                        key = (dealer, plant, veh_type, day, sku_id, qty_bucket)
                        if key in _seen_local:
                            continue
                        _seen_local.add(key)

                        try:
                            benefit = est_qty * _data_param_pun
                        except Exception:
                            benefit = est_qty * 1.0
                        cost = float(veh_type_cost.get(veh_type, 0.0))
                        heuristic_score = benefit - cost

                        # maintain a bounded top_list of size up to max_sim with O(max_sim) insertion
                        if len(top_list) < max_sim:
                            top_list.append(
                                (heuristic_score, key, plant, veh_type, day, est_qty)
                            )
                        else:
                            # find current minimum score index
                            min_idx = 0
                            min_val = top_list[0][0]
                            for i in range(1, len(top_list)):
                                if top_list[i][0] < min_val:
                                    min_val = top_list[i][0]
                                    min_idx = i
                            # replace if current candidate is better than min
                            if heuristic_score > min_val:
                                top_list[min_idx] = (
                                    heuristic_score,
                                    key,
                                    plant,
                                    veh_type,
                                    day,
                                    est_qty,
                                )

            if not top_list:
                continue

            # extract selected candidates sorted descending by heuristic score
            selected = list(top_list)
            selected.sort(key=lambda x: x[0], reverse=True)
            # normalize format to (key, plant, veh_type, day, est_qty)
            selected = [
                (item[1], item[2], item[3], item[4], item[5]) for item in selected
            ]

            # 'selected' already contains the top heuristic candidates (from bounded heap).
            # Respect max_sim as an additional safety cap (may be lower than heap size).
            max_sim = int(
                getattr(ALNSConfig, "REGRET_SIM_MAX_SIMULATE", max(6, topN * 2))
            )
            if len(selected) > max_sim:
                selected = selected[:max_sim]

            # For selected promises, first use a fast approximate estimator to avoid
            # performing a full state.copy() + veh_loading for every candidate.
            # Only when the fast estimate is promising (above a configurable threshold)
            # we fall back to the exact simulation. This reduces the number of expensive
            # copies and objective evaluations while preserving correctness for top
            # candidates.
            # Create a single temporary simulation base and perform apply+rollback
            # on it for multiple candidates to avoid repeated heavy copies.
            try:
                tmp_base = state.copy(clone_vehicles=False)
                # 在临时基态上抑制 compute_inventory，以避免在多次精评中触发全表重算
                try:
                    tmp_base.suppress_compute_inventory()
                except Exception:
                    # 若临时对象不支持该接口则继续（向后兼容）
                    pass
            except Exception:
                tmp_base = None
            for key, plant, veh_type, day, est_qty in selected:
                # avoid re-evaluating an identical candidate for the same state snapshot
                combined_key = (key, state_snapshot)
                if combined_key in global_memo:
                    continue
                try:
                    # Fast approximate evaluation (cheap, does not copy state)
                    try:
                        _, fast_impr, fast_ok = compute_improvement_fast_add_vehicle(
                            state, plant, dealer, veh_type, day, sku_id, est_qty
                        )
                    except Exception:
                        fast_impr, fast_ok = 0.0, False

                    # Configurable threshold for doing a full exact simulation
                    full_sim_threshold = float(
                        getattr(ALNSConfig, "REGRET_FULL_SIM_THRESHOLD", 0.0) or 0.0
                    )

                    if fast_ok and fast_impr > full_sim_threshold:
                        # Perform full (exact) simulation only when the fast estimator
                        # indicates a promising candidate.
                        # Try a cheap delta computation path that avoids any copy/compute_inventory
                        # when state caches are valid (most beneficial optimization path).
                        try:
                            # compute_improvement_delta(state, prev_obj, dealer, sku_id, qty, veh_type)
                            _newobj_d, _impr_d, _ok_d = compute_improvement_delta(
                                state, prev_obj, dealer, sku_id, est_qty, veh_type
                            )
                            if bool(_ok_d):
                                # accept delta estimate and memoize result; skip expensive simulation
                                global_memo[combined_key] = (float(_impr_d), True)
                                continue
                        except Exception:
                            # fallthrough to full simulation path on any error
                            pass

                        # Use tmp_base (single copy) and apply+rollback to avoid repeated copies.
                        if tmp_base is None:
                            try:
                                tmp = state.copy()
                                try:
                                    tmp.suppress_compute_inventory()
                                except Exception:
                                    pass
                            except Exception:
                                tmp = None
                        else:
                            tmp = tmp_base
                        improvement, success = 0.0, False
                        if tmp is not None:
                            # Snapshot minimal info for rollback
                            try:
                                before_vehicles_len = len(tmp.vehicles)
                                before_s_ikt_vals = {}
                                # we only expect changes on (plant, sku_id, d) for d in day..horizons
                                for d in range(day, horizons + 1):
                                    k2 = (plant, sku_id, d)
                                    before_s_ikt_vals[k2] = tmp.s_ikt.get(k2, None)
                                before_last_obj = getattr(tmp, "last_objective", None)
                                before_shipped = getattr(tmp, "shipped_cache", None)
                                before_dirty = getattr(tmp, "objective_dirty", True)
                                before_suppress = getattr(
                                    tmp, "_suppress_compute_inventory", False
                                )
                            except Exception:
                                before_vehicles_len = len(tmp.vehicles)
                                before_s_ikt_vals = {}
                                before_last_obj = getattr(tmp, "last_objective", None)
                                before_shipped = getattr(tmp, "shipped_cache", None)
                                before_dirty = getattr(tmp, "objective_dirty", True)
                                before_suppress = getattr(
                                    tmp, "_suppress_compute_inventory", False
                                )

                            try:
                                # use a lightweight surrogate vehicle for fast simulations when possible
                                veh_sim = Vehicle(plant, dealer, veh_type, day, data)
                                try:
                                    veh_loading(
                                        tmp, veh_sim, {sku_id: est_qty}, commit=True
                                    )
                                    _, improvement, success = compute_improvement(
                                        tmp, prev_obj if prev_obj is not None else 0.0
                                    )
                                except Exception:
                                    improvement, success = 0.0, False
                            except Exception:
                                improvement, success = 0.0, False

                            # rollback: remove appended vehicles and restore s_ikt + caches
                            try:
                                # remove any newly appended vehicles by trimming list
                                if len(tmp.vehicles) > before_vehicles_len:
                                    # remove appended vehicles
                                    del tmp.vehicles[before_vehicles_len:]
                                # restore s_ikt keys we touched
                                for k2, val in before_s_ikt_vals.items():
                                    if val is None:
                                        if k2 in tmp.s_ikt:
                                            del tmp.s_ikt[k2]
                                    else:
                                        tmp.s_ikt[k2] = val
                                # restore cached objective state
                                tmp.last_objective = before_last_obj
                                tmp.shipped_cache = before_shipped
                                tmp.objective_dirty = before_dirty
                                # restore suppression flag for compute_inventory to previous state
                                try:
                                    setattr(
                                        tmp,
                                        "_suppress_compute_inventory",
                                        before_suppress,
                                    )
                                except Exception:
                                    # best-effort: if we cannot restore attribute, ignore
                                    pass
                            except Exception:
                                # best-effort rollback; if rollback fails, drop tmp_base to avoid reuse
                                tmp_base = None
                            # store exact-simulation result in global_memo for this state snapshot
                            try:
                                global_memo[combined_key] = (improvement, bool(success))
                            except Exception:
                                # ignore memo write failures (shouldn't happen)
                                pass
                    else:
                        # Use fast estimate as memoized score (may be negative/zero)
                        global_memo[combined_key] = (float(fast_impr), bool(fast_ok))
                except Exception:
                    # On any unexpected error, mark as non-promising
                    global_memo[combined_key] = (0.0, False)

            # Build final candidates list from memo results, keep best topN by true improvement
            true_candidates = []
            for key, plant, veh_type, day, est_qty in selected:
                combined = (key, state_snapshot)
                if combined in global_memo:
                    improvement, success = global_memo[combined]
                    if success:
                        true_candidates.append(
                            (improvement, (plant, veh_type, day, est_qty))
                        )
                # safety cap to avoid huge lists; we'll later pick topN
                if len(true_candidates) >= topN * 5:
                    break

            if not true_candidates:
                continue

            true_candidates.sort(key=lambda x: x[0], reverse=True)
            # keep only topN (best true improvements) as in original logic
            candidates = true_candidates[:topN]
            # normalize candidates format to match original code expectation
            candidates = [(imp, cand) for (imp, cand) in candidates]
            # proceed as before
            # 计算后悔值（top1 - topk）
            candidates.sort(key=lambda x: x[0], reverse=True)
            top1 = candidates[0][0]
            topk = candidates[min(k - 1, len(candidates) - 1)][0]
            regret = top1 - topk
            regret_list.append(((dealer, sku_id), regret, candidates[0][1]))

        if not regret_list:
            break
        # 选择最大后悔值的任务并提交最优候选
        regret_list.sort(key=lambda x: x[1], reverse=True)
        (dealer_sel, sku_sel), _, best_candidate = regret_list[0]

        plant_sel, veh_type_sel, day_sel, qty_sel = best_candidate
        # 在提交前重新检查可用量并按照当前可用量调整提交量（防止超发）
        # Prefer using the precomputed resource_pool snapshot to avoid expensive recomputation;
        # fall back to accurate _construct_resource_availability only when pool has no entry.
        avail_now = resource_pool_get((plant_sel, sku_sel, day_sel), None)
        if avail_now is None:
            avail_now = _construct_resource_availability(
                state, plant_sel, sku_sel, day_sel
            )
        qty_to_commit = min(qty_sel, avail_now)
        if qty_to_commit <= 0:
            # 无可用量，跳过本次循环以便重新评估其他任务
            continue
        vehicle = Vehicle(plant_sel, dealer_sel, veh_type_sel, day_sel, data)
        try:
            before_len = len(state.vehicles)
            commit_qty = _coerce_int(qty_to_commit, 0)
            if commit_qty <= 0:
                continue
            veh_loading(state, vehicle, {sku_sel: commit_qty}, commit=True)
            new_vs = state.vehicles[before_len:]
            if new_vs:
                inserted_vehicles.extend(new_vs)
                # state has been changed by a committed load -> invalidate best-effort caches only
                try:
                    # prefer a proper API if available
                    if hasattr(state, "mark_objective_dirty"):
                        try:
                            state.mark_objective_dirty()
                        except Exception:
                            # fallback to manual invalidation
                            try:
                                state.shipped_cache = None
                            except Exception:
                                pass
                            try:
                                state.last_objective = None
                            except Exception:
                                pass
                            try:
                                state.objective_dirty = True
                            except Exception:
                                pass
                    else:
                        try:
                            state.shipped_cache = None
                        except Exception:
                            pass
                        try:
                            state.last_objective = None
                        except Exception:
                            pass
                        try:
                            state.objective_dirty = True
                        except Exception:
                            pass
                except Exception:
                    # ignore any failures to avoid disrupting main loop
                    pass
            else:
                # 兼容性回退：若 veh_loading 未将 vehicle 对象加入 state.vehicles，则保留原 vehicle 记录
                actual_committed = vehicle.cargo.get((sku_sel, day_sel), 0)
                if actual_committed > 0:
                    inserted_vehicles.append(vehicle)
                # state has been changed by a committed load -> invalidate best-effort caches only
                try:
                    # prefer a proper API if available
                    if hasattr(state, "mark_objective_dirty"):
                        try:
                            state.mark_objective_dirty()
                        except Exception:
                            # fallback to manual invalidation
                            try:
                                state.shipped_cache = None
                            except Exception:
                                pass
                            try:
                                state.last_objective = None
                            except Exception:
                                pass
                            try:
                                state.objective_dirty = True
                            except Exception:
                                pass
                    else:
                        try:
                            state.shipped_cache = None
                        except Exception:
                            pass
                        try:
                            state.last_objective = None
                        except Exception:
                            pass
                        try:
                            state.objective_dirty = True
                        except Exception:
                            pass
                except Exception:
                    # ignore any failures to avoid disrupting main loop
                    pass
        except Exception:
            # 提交失败，跳过并继续（下次循环会重新评估 unmet demands）
            continue

    repair_baseline_value: float = float(
        repair_baseline_obj if repair_baseline_obj is not None else 0.0
    )
    post_repair(
        state,
        inserted_vehicles,
        op_name="regret_based_repair",
        prev_obj=repair_baseline_value,
        t0=t0,
    )

    elapsed = time.time() - t0
    try:
        logger.debug(
            f"[OPLOG] regret_based_repair: 插入 {len(inserted_vehicles)} 辆车 ({elapsed:.4f}s)"
        )
    except Exception:
        pass

    return state


class RegretBasedRepairOperator:
    """
    可作为 alns.add_repair_operator 接受的对象。
    使用示例: alns.add_repair_operator(RegretBasedRepairOperator(k=2, topN=6))
    """

    def __init__(self, k: int = 2, topN: int = 6, time_limit: Optional[float] = None):
        defaults = ALNSConfig.get_operator_default("regret_based_repair").get(
            "params", {}
        )
        self.k = max(1, _coerce_int(k if k is not None else defaults.get("k", 2), 2))
        self.topN = max(
            1, _coerce_int(topN if topN is not None else defaults.get("topN", 6), 6)
        )
        self.time_limit = _coerce_float(
            time_limit if time_limit is not None else defaults.get("time_limit", 10.0),
            10.0,
        )
        self.__name__ = "regret_based_repair"

    def __call__(self, current, rng: np.random.Generator):
        return regret_based_repair(current, rng, self.k, self.topN, self.time_limit)


def fleet_consolidation_repair(
    current: SolutionState, rng: rnd.Generator
) -> SolutionState:
    """
    专门的车队整合修复算子。

    此算子的主要目标是通过合并小型车辆到大型车辆来减少车队规模，
    同时保持可行性。这与其他修复算子(需要满足需求)不同，这里优先考虑的是
    车队优化和合并效率。

    工作流程：
    1. 按(plant, dealer, day)将车辆分组（相同的发运路线）
    2. 在每个组内，按体积对车辆排序（从小到大）
    3. 贪心地尝试将较小的车辆合并到较大的车辆中
    4. 从大到小尝试目标车辆，以最大化合并概率
    5. 调用 post_repair 完成清理

    Args:
        current: 当前解状态
        rng: 随机数生成器（未使用，但保持接口一致性）

    Returns:
        合并后的SolutionState（如果合并成功）或原始状态的复制版本
    """
    from .repair_utils import consolidate_route_vehicles, post_repair

    state = current.copy()

    # 保存原始车辆数用于日志记录
    before_count = len(state.vehicles)

    try:
        # 调用consolidation函数进行车队合并
        # 对于这个专门的算子，我们使用更激进的合并策略（高max_pairs）
        max_pairs = getattr(ALNSConfig, "REPAIR_POST_CONSOLIDATION_MAX_PAIRS", 300)
        # 对于这个专门的consolidation算子，增加最大合并对数以获得最大效果
        max_pairs = max_pairs * 2  # 增加到600对

        merged_count = consolidate_route_vehicles(state, max_pairs_per_group=max_pairs)

        after_count = len(state.vehicles)

        # 使用post_repair完成清理和验证
        inserted_veh = []
        try:
            post_repair(
                state,
                inserted_veh,
                op_name="fleet_consolidation_repair",
                prev_obj=float("inf"),
                t0=time.time(),
            )
        except Exception as e:
            try:
                logger.debug(
                    f"[OPLOG] fleet_consolidation_repair: post_repair异常: {e}"
                )
            except Exception:
                pass

        # 日志记录合并结果
        try:
            logger.debug(
                f"[OPLOG] fleet_consolidation_repair: 合并 {merged_count} 对 "
                f"({before_count} -> {after_count} 辆车)"
            )
        except Exception:
            pass

        return state

    except Exception as e:
        # 如果出现异常，返回原始状态的副本
        try:
            logger.warning(
                f"[OPLOG] fleet_consolidation_repair: 异常 {e}, 返回原始状态"
            )
        except Exception:
            pass
        return current.copy()


def sequential_repair(
    partial: SolutionState, rng: rnd.Generator, lookahead_horizon: Optional[int] = None
) -> SolutionState:
    """
    顺序修复 (Sequential Repair) - P0 优先级

    核心思想：
      按时间周期顺序进行修复，充分利用多期库存约束的递推结构。
      第t期修复后，期末库存自动成为第t+1期的初始库存，确保库存约束自动满足。

    算法步骤：
      1. 复制当前状态，获取期初库存
      2. 遍历时间周期 t=1 to H,对该期执行单期贪心修复并维护库存
      3. 返回修复后的状态，库存约束自动满足

    返回值：修复后的可行或接近可行的解

    特点：✓ 显式维护库存约束，递推式修复 ✓ 对多期库存平衡问题特别有效
    """
    t0 = time.time()
    logger.debug("[OPLOG] 开始执行 sequential_repair 算子")
    state = partial.copy()
    data = state.data
    lookahead_h = max(0, _coerce_int(lookahead_horizon, 0))
    day_delay_penalty = _coerce_float(
        getattr(ALNSConfig, "LOOKAHEAD_REPAIR_DELAY_PENALTY", 0.5), 0.5
    )

    # 安全性检查
    if not state.vehicles:
        logger.debug("[OPLOG] sequential_repair: 无车辆，直接返回")
        return state

    horizons = getattr(data, "horizons", 0)
    if horizons <= 0:
        return state

    # 初始化库存
    try:
        state.compute_inventory()
    except Exception:
        pass

    # 计算当前已发运的库存缓存
    shipped_cache = {}
    try:
        shipped_cache = state.compute_shipped()
    except Exception:
        shipped_cache = {}

    # 按周期顺序修复
    supply_chain = data.construct_supply_chain()
    for period in range(1, horizons + 1):
        unsatisfied_this_period = []

        # 获取本期未满足需求
        for (dealer, sku_id), total_demand in data.demands.items():
            already_sent = shipped_cache.get((dealer, sku_id), 0)
            unmet = total_demand - already_sent

            if unmet > 0:
                unsatisfied_this_period.append((dealer, sku_id, unmet))

        # 对该期执行贪心修复
        for dealer, sku_id, unmet_qty in unsatisfied_this_period:
            plants = [
                p
                for (p, d), skus in supply_chain.items()
                if d == dealer and sku_id in skus
            ]

            while unmet_qty > 0:
                best_choice = None

                # 在 [period, period + lookahead_h] 的时间窗口中择优插入
                day_upper = min(horizons, period + lookahead_h)
                candidate_days = range(period, day_upper + 1)

                for plant in plants:
                    for cand_day in candidate_days:
                        try:
                            available = state.s_ikt.get((plant, sku_id, cand_day), 0)
                            if available <= 0:
                                continue

                            load_qty = int(min(unmet_qty, available))
                            if load_qty <= 0:
                                continue

                            # 优先用已有路由车辆，否则创建新车辆
                            cand_veh = None
                            for veh in state.vehicles:
                                if (
                                    veh.fact_id == plant
                                    and veh.dealer_id == dealer
                                    and veh.day == cand_day
                                    and veh.capacity >= load_qty
                                ):
                                    cand_veh = veh
                                    break

                            if cand_veh is None:
                                required_volume = load_qty * data.sku_sizes.get(
                                    sku_id, 0
                                )
                                veh_type = select_vehicle_type_for_volume(
                                    required_volume, data
                                )
                                if veh_type is None:
                                    continue
                            else:
                                veh_type = cand_veh.type

                            # 使用快速估计打分 + 延迟惩罚，避免仅按当期贪心。
                            _, impro_fast, _ = compute_improvement_fast_add_vehicle(
                                state,
                                plant=plant,
                                dealer=dealer,
                                veh_type=veh_type,
                                day=cand_day,
                                sku_id=sku_id,
                                qty=load_qty,
                            )
                            score = float(impro_fast) - day_delay_penalty * max(
                                0, cand_day - period
                            )

                            if (best_choice is None) or (score > best_choice["score"]):
                                best_choice = {
                                    "plant": plant,
                                    "day": cand_day,
                                    "load_qty": load_qty,
                                    "veh": cand_veh,
                                    "veh_type": veh_type,
                                    "score": score,
                                }
                        except Exception:
                            continue

                if best_choice is None:
                    break

                try:
                    plant = best_choice["plant"]
                    day = int(best_choice["day"])
                    load_qty = int(best_choice["load_qty"])
                    best_veh = best_choice["veh"]
                    if best_veh is None:
                        best_veh = Vehicle(
                            plant,
                            dealer,
                            best_choice["veh_type"],
                            day,
                            data,
                        )

                    before_route_qty = 0
                    for v in state.vehicles:
                        if (
                            v.fact_id == plant
                            and v.dealer_id == dealer
                            and v.day == day
                        ):
                            before_route_qty += int(v.cargo.get((sku_id, day), 0))

                    veh_loading(
                        state,
                        best_veh,
                        {sku_id: int(load_qty)},
                        commit=True,
                        skip_compute_inventory=True,
                    )

                    after_route_qty = 0
                    for v in state.vehicles:
                        if (
                            v.fact_id == plant
                            and v.dealer_id == dealer
                            and v.day == day
                        ):
                            after_route_qty += int(v.cargo.get((sku_id, day), 0))

                    actual_loaded = max(0, after_route_qty - before_route_qty)
                    if actual_loaded <= 0:
                        break

                    shipped_cache[(dealer, sku_id)] = (
                        shipped_cache.get((dealer, sku_id), 0) + actual_loaded
                    )
                    unmet_qty -= actual_loaded
                except Exception as e:
                    logger.debug(f"[OPLOG] sequential_repair: 装载异常 {e}")
                    break

    # 最终处理
    try:
        state.compute_inventory()
        state.mark_objective_dirty()
    except Exception:
        pass

    logger.debug(f"[OPLOG] sequential_repair: 完成 ({time.time() - t0:.4f}s)")
    return state


def create_sequential_repair(lookahead_horizon: Optional[int] = None):
    """工厂函数：创建顺序修复算子包装器"""
    defaults = ALNSConfig.get_operator_default("sequential_repair").get("params", {})
    lookahead_horizon = lookahead_horizon or defaults.get("lookahead_horizon", None)
    name = "sequential_repair"
    return RepairOperatorWrapper(
        sequential_repair, name=name, lookahead_horizon=lookahead_horizon
    )


def lookahead_repair(
    partial: SolutionState,
    rng: rnd.Generator,
    lookahead_horizon: Optional[int] = None,
) -> SolutionState:
    """
    前瞻修复:
    - 在 sequential 风格的周期推进框架下, 对 [t, t+h] 窗口内的候选插入进行显式前瞻评分
    - 评分同时考虑:
        1) 快速改进估计
        2) 延后交付惩罚
        3) 未来若干天的库存安全压力
        4) 是否可以复用已有路线车辆(降低车队碎片化)
    - 目标不是单纯“看更远一天”, 而是避免当前插入把未来库存压穿或制造额外碎片车
    """
    t0 = time.time()
    logger.debug("[OPLOG] 开始执行 lookahead_repair 算子")
    state = partial.copy()
    data = state.data

    defaults = ALNSConfig.get_operator_default("lookahead_repair").get("params", {})
    h = lookahead_horizon
    if h is None:
        h = defaults.get("lookahead_horizon", 2)
    h = max(1, _coerce_int(h, 2))

    delay_penalty = _coerce_float(
        getattr(ALNSConfig, "LOOKAHEAD_REPAIR_DELAY_PENALTY", 0.5), 0.5
    )
    future_weight = _coerce_float(
        getattr(ALNSConfig, "LOOKAHEAD_REPAIR_FUTURE_WEIGHT", 0.35), 0.35
    )
    safety_floor_ratio = max(
        0.01,
        _coerce_float(
            getattr(ALNSConfig, "LOOKAHEAD_REPAIR_SAFETY_FLOOR_RATIO", 0.10), 0.10
        ),
    )
    reuse_bonus = _coerce_float(
        getattr(ALNSConfig, "LOOKAHEAD_REPAIR_REUSE_BONUS", 0.10), 0.10
    )

    if not state.vehicles:
        return state

    horizons = getattr(data, "horizons", 0)
    if horizons <= 0:
        return state

    try:
        state.compute_inventory()
    except Exception:
        pass

    try:
        shipped_cache = state.compute_shipped()
    except Exception:
        shipped_cache = {}

    supply_chain = data.construct_supply_chain()
    inserted_vehicles = []

    for period in range(1, horizons + 1):
        unsatisfied_this_period = []
        for (dealer, sku_id), total_demand in data.demands.items():
            already_sent = shipped_cache.get((dealer, sku_id), 0)
            unmet = total_demand - already_sent
            if unmet > 0:
                unsatisfied_this_period.append((dealer, sku_id, unmet))

        for dealer, sku_id, unmet_qty in unsatisfied_this_period:
            plants = [
                p
                for (p, d), skus in supply_chain.items()
                if d == dealer and sku_id in skus
            ]

            while unmet_qty > 0:
                best_choice = None
                day_upper = min(horizons, period + h)
                candidate_days = range(period, day_upper + 1)
                safety_floor = max(
                    1,
                    int(
                        max(
                            1.0,
                            float(data.demands.get((dealer, sku_id), 0))
                            * safety_floor_ratio,
                        )
                    ),
                )

                for plant in plants:
                    for cand_day in candidate_days:
                        try:
                            available = state.s_ikt.get((plant, sku_id, cand_day), 0)
                            if available <= 0:
                                continue

                            load_qty = int(min(unmet_qty, available))
                            if load_qty <= 0:
                                continue

                            cand_veh = None
                            for veh in state.vehicles:
                                if (
                                    veh.fact_id == plant
                                    and veh.dealer_id == dealer
                                    and veh.day == cand_day
                                    and veh.capacity >= load_qty
                                ):
                                    cand_veh = veh
                                    break

                            if cand_veh is None:
                                required_volume = load_qty * data.sku_sizes.get(
                                    sku_id, 0
                                )
                                veh_type = select_vehicle_type_for_volume(
                                    required_volume, data
                                )
                                if veh_type is None:
                                    continue
                            else:
                                veh_type = cand_veh.type

                            _, impro_fast, success = (
                                compute_improvement_fast_add_vehicle(
                                    state,
                                    plant=plant,
                                    dealer=dealer,
                                    veh_type=veh_type,
                                    day=cand_day,
                                    sku_id=sku_id,
                                    qty=load_qty,
                                )
                            )
                            if not success:
                                continue

                            # Evaluate the pressure the insertion creates over the future window.
                            future_end = min(horizons, cand_day + h)
                            future_days = range(cand_day, future_end + 1)
                            future_pressure = 0.0
                            future_count = 0
                            for future_day in future_days:
                                base_inv = float(
                                    state.s_ikt.get((plant, sku_id, future_day), 0)
                                )
                                projected_inv = max(0.0, base_inv - float(load_qty))
                                deficit = max(0.0, float(safety_floor) - projected_inv)
                                future_pressure += deficit / float(safety_floor)
                                future_count += 1
                            if future_count > 0:
                                future_pressure /= float(future_count)

                            route_bonus = 0.0
                            if cand_veh is not None:
                                route_bonus = reuse_bonus

                            score = (
                                float(impro_fast)
                                - delay_penalty * max(0, cand_day - period)
                                - future_weight * future_pressure
                                + route_bonus
                            )

                            if (best_choice is None) or (score > best_choice["score"]):
                                best_choice = {
                                    "plant": plant,
                                    "day": cand_day,
                                    "load_qty": load_qty,
                                    "veh": cand_veh,
                                    "veh_type": veh_type,
                                    "score": score,
                                }
                        except Exception:
                            continue

                if best_choice is None:
                    break

                try:
                    plant = best_choice["plant"]
                    day = int(best_choice["day"])
                    load_qty = int(best_choice["load_qty"])
                    best_veh = best_choice["veh"]
                    if best_veh is None:
                        best_veh = Vehicle(
                            plant, dealer, best_choice["veh_type"], day, data
                        )

                    before_len = len(state.vehicles)
                    before_route_qty = 0
                    for v in state.vehicles:
                        if (
                            v.fact_id == plant
                            and v.dealer_id == dealer
                            and v.day == day
                        ):
                            before_route_qty += int(v.cargo.get((sku_id, day), 0))

                    veh_loading(
                        state,
                        best_veh,
                        {sku_id: int(load_qty)},
                        commit=True,
                        skip_compute_inventory=True,
                    )

                    new_vs = state.vehicles[before_len:]

                    after_route_qty = 0
                    for v in state.vehicles:
                        if (
                            v.fact_id == plant
                            and v.dealer_id == dealer
                            and v.day == day
                        ):
                            after_route_qty += int(v.cargo.get((sku_id, day), 0))

                    actual_loaded = max(0, after_route_qty - before_route_qty)
                    if actual_loaded <= 0:
                        break

                    shipped_cache[(dealer, sku_id)] = (
                        shipped_cache.get((dealer, sku_id), 0) + actual_loaded
                    )
                    unmet_qty -= actual_loaded
                    if new_vs:
                        inserted_vehicles.extend(new_vs)
                except Exception as e:
                    logger.debug(f"[OPLOG] lookahead_repair: 装载异常 {e}")
                    break

    try:
        state.compute_inventory()
        state.mark_objective_dirty()
    except Exception:
        pass

    post_repair(
        state,
        inserted_vehicles,
        op_name="lookahead_repair",
        prev_obj=_safe_prev_obj(partial),
        t0=t0,
    )
    logger.debug(f"[OPLOG] lookahead_repair: 完成 ({time.time() - t0:.4f}s)")
    return state


def create_lookahead_repair(lookahead_horizon: Optional[int] = None):
    """工厂函数：创建前瞻修复算子包装器"""
    defaults = ALNSConfig.get_operator_default("lookahead_repair").get("params", {})
    lookahead_horizon = (
        lookahead_horizon
        if lookahead_horizon is not None
        else defaults.get("lookahead_horizon", 2)
    )
    return RepairOperatorWrapper(
        lookahead_repair,
        name="lookahead_repair",
        lookahead_horizon=lookahead_horizon,
    )


class FleetConsolidationRepairOperator:
    """
    可作为 alns.add_repair_operator 接受的对象。
    使用示例: alns.add_repair_operator(FleetConsolidationRepairOperator())

    此算子专门优化车队规模，通过消除碎片化的车辆分配来实现。
    """

    def __init__(self):
        self.__name__ = "fleet_consolidation_repair"

    def __call__(self, state, rng: np.random.Generator, **kwargs):
        return fleet_consolidation_repair(state, rng)
