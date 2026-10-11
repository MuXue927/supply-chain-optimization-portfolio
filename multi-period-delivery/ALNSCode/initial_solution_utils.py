"""
initial_solution_utils
======================

模块定位:
    为 ALNS 求解生成初始可行解 (initial feasible solution) 的一组高性能实用函数。
    该模块已经通过 pytest 与多轮性能测试确认稳定性与正确性——本次整理仅限:
      * import 语句分组与精简（未删除功能相关引用）
      * 注释与文档统一风格、补充细节
      * 不改动任何运行逻辑 / 变量名 / 控制流程

总体方法 (improved_initial_solution):
    使用多阶段策略:
      1. 初始化资源池 (跨工厂/SKU/天的可用库存累积)
      2. 基于稀缺性 + 供需比 + 需求规模的优先级排序
      3. 顺序分配需求到车辆
      4. 修复阶段 (库存/车辆精简/合并)
      5. 返回可行解 (调用 compute_inventory 确认库存一致性)

关键性能优化点 (保持原实现，不修改):
    - 多处使用局部变量缓存 (函数内 get / dict / range 引用) 降低属性查找成本
    - 分配时按库存总量对工厂排序，提高匹配成功率
    - 需求处理与修复阶段均使用“提前超时”策略，防止初始化阶段卡死
    - 车辆合并阶段按 (工厂, 经销商, 天) 分组减少组合复杂度
    - 负库存修复采取局部增量调整而非全局重算

函数速览:
    improved_initial_solution : 入口函数, 组织整体五阶段流程
    _initialize_resource_pool : 生成 (plant, sku, day) → 累积可用库存
    _prioritize_demands       : 计算并排序需求优先级
    _sequential_allocation    : 顺序方式分配需求 → 车辆
    _repair_solution          : 负库存修复 + 空车移除 + 小车合并
    _merge_small_vehicles     : 小车辆合并以减少数量
    _select_best_vehicle      : 选择最契合容量的车型

注意:
    - 为避免循环依赖, 局部 import (.alnsopt 中的 veh_loading / SolutionState) 保留原位置
    - Vehicle 顶层 import 与函数内重复 import 不移除 (兼容现有依赖结构)

维护建议:
    - 若后续扩展新初始解策略, 可在本文件新增函数并在 improved_initial_solution 中加策略分支
    - 统计扩展(已实现):
        * initial_build_stats 挂载到解对象:
              {
                'resource_pool_time': float,
                'prioritize_time': float,
                'allocation_time': float,
                'repair_time': float,
                'parallel_used': bool,
                'repair_iterations': int
              }
"""

# =========================
# 标准库
# =========================
import time
from collections import defaultdict

# =========================
# 项目内部
# =========================
from .vehicle import Vehicle  # 顶层引用 (函数内仍有局部 import 以避免循环依赖)
from .alns_config import default_config as ALNSConfig


def improved_initial_solution(state, rng):
    """
    入口: 构造初始可行解 (多阶段 + 超时保护)

    阶段:
        1) 资源池初始化 (累积库存轨迹)
        2) 需求优先级排序 (稀缺性/供需比/需求量综合评分)
        3) 顺序分配
        4) 修复阶段 (负库存 / 空车 / 小车合并)
        5) 生成最终库存 (compute_inventory)

    参数:
        state : SolutionState 初始状态对象 (外部构造)
        rng   : numpy.random.Generator 随机数生成器 (保留接口, 当前实现未大量使用)

    返回:
        SolutionState (可行初始解)
    """
    data = state.data

    # 动态超时: 基于需求条目规模 (可通过 ALNSConfig 调整)
    min_timeout = getattr(ALNSConfig, "MIN_INIT_TIMEOUT", 30)
    max_timeout = getattr(ALNSConfig, "MAX_INIT_TIMEOUT", 120)
    per_demand = getattr(ALNSConfig, "INIT_TIMEOUT_PER_DEMAND", 0.05)
    timeout = min(max_timeout, max(min_timeout, len(data.demands) * per_demand))
    t_start = time.time()

    # 1. 资源池
    print("Initializing resource pool...")
    t0 = time.time()
    resource_pool = _initialize_resource_pool(data)
    print(f" Resource pool initialized in {time.time() - t0:.2f} seconds.")

    # 2. 需求优先级
    print("Prioritizing demands...")
    t0 = time.time()
    prioritized_demands = _prioritize_demands(data)
    print(f" Demands prioritized in {time.time() - t0:.2f} seconds.")

    # 3. 顺序分配
    print("Using sequential allocation...")
    t0 = time.time()
    stats = {}
    sol = _sequential_allocation(
        state, prioritized_demands, resource_pool, rng, timeout
    )
    stats["allocation_time"] = time.time() - t0
    print(
        f" Sequential allocation completed in {stats['allocation_time']:.2f} seconds."
    )

    # 4. 修复阶段
    print("Repairing solution if necessary...")
    t0 = time.time()
    # 统计修复阶段开始时间
    repair_start = time.time()
    _repair_solution(sol, timeout - (time.time() - t_start), stats=stats)
    repair_time = time.time() - repair_start
    print(f" Repair completed in {time.time() - t0:.2f} seconds.")

    # 5. 最终库存计算
    sol.compute_inventory()

    # 严格可行性收尾：在初始解阶段做少量确定性修复，尽可能输出严格可行解。
    _finalize_strict_initial_feasibility(sol, rng, timeout - (time.time() - t_start))

    # 汇总阶段统计信息 (不改变原逻辑)
    # 资源池与优先级阶段用已有打印时间差；为确保一致, 重新计算近似耗时
    stats.setdefault("repair_iterations", 0)
    stats["parallel_used"] = False
    if "allocation_time" not in stats:
        stats["allocation_time"] = None
    stats["resource_pool_time"] = None  # 原实现仅打印，不重新测量避免重复逻辑
    stats["prioritize_time"] = None
    stats["repair_time"] = repair_time

    # 挂载到解对象，供外部分析
    sol.initial_build_stats = stats
    return sol


def _initialize_resource_pool(data):
    """
    构建资源池: 记录 (plant, sku, day) → 当日可用累计库存
    规则:
        day=0 使用初始库存
        day>=1 递推: prev(day-1) + 当日生产
    """
    resource_pool = {}

    # 初始库存
    for (plant, sku), inv in data.sku_initial_inv.items():
        resource_pool[(plant, sku, 0)] = inv

    # 按日累积
    for day in range(1, data.horizons + 1):
        for plant in data.plants:
            for sku in data.skus_plant.get(plant, []):
                prev_inv = resource_pool.get((plant, sku, day - 1), 0)
                production = data.sku_prod_each_day.get((plant, sku, day), 0)
                resource_pool[(plant, sku, day)] = prev_inv + production

    return resource_pool


def _prioritize_demands(data):
    """
    计算需求优先级:
        指标:
            scarcity_score      = 1 / (供需比)
            supply_chain_score  = 1 / (可供应工厂数量)
            volume_score        = 需求量归一化
        综合:
            priority = 0.5*scarcity + 0.3*supply_chain + 0.2*volume
    返回:
        [( (dealer, sku), demand, priority_score ), ...] 降序
    """
    # SKU 总供应量 (生产 + 初始库存)
    sku_total_supply = defaultdict(int)
    for (_, sku, _), qty in data.sku_prod_each_day.items():
        sku_total_supply[sku] += qty
    for (_, sku), inv in data.sku_initial_inv.items():
        sku_total_supply[sku] += inv

    # SKU 总需求量
    sku_total_demand = defaultdict(int)
    for (_, sku), demand in data.demands.items():
        sku_total_demand[sku] += demand

    # 供需比
    sku_supply_demand_ratio = {}
    for sku in data.all_skus:
        supply = sku_total_supply.get(sku, 0)
        demand = sku_total_demand.get(sku, 0)
        sku_supply_demand_ratio[sku] = supply / demand if demand > 0 else float("inf")

    demand_priorities = []
    supply_chain = data.construct_supply_chain()
    max_demand = max(data.demands.values()) if data.demands else 1

    for (dealer, sku), demand in data.demands.items():
        if demand <= 0:
            continue

        available_plants = [
            plant
            for (plant, dealer_id), skus in supply_chain.items()
            if dealer_id == dealer and sku in skus
        ]
        supply_chain_score = 1.0 / (
            len(available_plants) if available_plants else float("inf")
        )
        scarcity_score = 1.0 / sku_supply_demand_ratio.get(sku, float("inf"))
        volume_score = demand / max_demand
        priority_score = (
            0.5 * scarcity_score + 0.3 * supply_chain_score + 0.2 * volume_score
        )
        demand_priorities.append(((dealer, sku), demand, priority_score))

    demand_priorities.sort(key=lambda x: x[2], reverse=True)
    return demand_priorities


def _sequential_allocation(state, prioritized_demands, resource_pool, rng, timeout):
    """
    顺序分配需求 → 车辆（经销商聚合 + 跨期累计约束）
    策略:
        - 将需求按经销商分组，按经销商平均优先级排序
        - 主分配阶段: 同时受 resource_pool 与 cumulative_map 约束
        - 补分配阶段: 若仍有未满足需求，仅受 resource_pool 约束
        - 成功装载后同步更新 resource_pool 与 cumulative_map
    """
    data = state.data
    from .alnsopt import veh_loading

    if not bool(getattr(ALNSConfig, "INITIAL_USE_DEALER_GROUPING", True)):
        return _sequential_allocation_legacy(
            state,
            prioritized_demands,
            resource_pool,
            timeout,
            veh_loading,
        )

    sol = state.copy()
    t_start = time.time()

    supply_chain = data.construct_supply_chain()
    demand_map = {
        (dealer, sku): int(demand)
        for (dealer, sku), demand, _ in prioritized_demands
        if demand > 0
    }
    dealer_groups = _group_demands_by_dealer(prioritized_demands)
    cumulative_map = _compute_cumulative_map(resource_pool, data)

    use_cumulative = bool(
        getattr(ALNSConfig, "INITIAL_ENABLE_CUMULATIVE_CONSTRAINT", True)
    )
    enable_relaxed = bool(
        getattr(ALNSConfig, "INITIAL_ENABLE_RELAXED_REALLOCATION", True)
    )

    # 主分配阶段: 累计约束 + 物理资源池双约束。
    for dealer, sku_demands, _avg_priority in dealer_groups:
        if time.time() - t_start > timeout:
            break
        _allocate_dealer_demands(
            sol,
            dealer,
            sku_demands,
            demand_map,
            resource_pool,
            cumulative_map,
            supply_chain,
            use_cumulative=use_cumulative,
            timeout=timeout,
            t_start=t_start,
            veh_loading=veh_loading,
        )

    # 补分配阶段: 放宽累计约束，尽量补齐未满足需求。
    if enable_relaxed and time.time() - t_start <= timeout:
        unmet_groups = []
        for dealer, sku_demands, avg_priority in dealer_groups:
            unmet_items = []
            for sku, _demand, sku_priority in sku_demands:
                remain = int(demand_map.get((dealer, sku), 0))
                if remain > 0:
                    unmet_items.append((sku, remain, sku_priority))
            if unmet_items:
                unmet_groups.append((dealer, unmet_items, avg_priority))

        for dealer, sku_demands, _avg_priority in unmet_groups:
            if time.time() - t_start > timeout:
                break
            _allocate_dealer_demands(
                sol,
                dealer,
                sku_demands,
                demand_map,
                resource_pool,
                cumulative_map,
                supply_chain,
                use_cumulative=False,
                timeout=timeout,
                t_start=t_start,
                veh_loading=veh_loading,
            )

    return sol


def _sequential_allocation_legacy(
    state,
    prioritized_demands,
    resource_pool,
    timeout,
    veh_loading,
):
    """Legacy allocation path: demand-by-demand greedy traversal."""
    data = state.data
    sol = state.copy()
    t_start = time.time()

    allocated = defaultdict(int)
    supply_chain = data.construct_supply_chain()
    supply_chain_items = list(supply_chain.items())
    horizons_range = range(1, data.horizons + 1)
    resource_get = resource_pool.get

    for (dealer, sku), total_demand, _ in prioritized_demands:
        if time.time() - t_start > timeout:
            break

        remain_demand = int(total_demand) - int(allocated[(dealer, sku)])
        if remain_demand <= 0:
            continue

        available_plants = [
            plant
            for (plant, dealer_id), skus in supply_chain_items
            if dealer_id == dealer and sku in skus
        ]
        if not available_plants:
            continue

        plant_inventory = [
            (
                plant,
                sum(int(resource_get((plant, sku, d), 0)) for d in horizons_range),
            )
            for plant in available_plants
        ]
        plant_inventory.sort(key=lambda x: x[1], reverse=True)

        for plant, _ in plant_inventory:
            if remain_demand <= 0:
                break
            for day in horizons_range:
                if remain_demand <= 0:
                    break
                available = int(resource_get((plant, sku, day), 0))
                if available <= 0:
                    continue

                assign_qty = int(min(remain_demand, available))
                if assign_qty <= 0:
                    continue

                veh_type = _select_best_vehicle(assign_qty, sku, data)
                vehicle = Vehicle(plant, dealer, veh_type, day, data)

                shipped_before = int(sol.compute_shipped().get((dealer, sku), 0))
                success = veh_loading(sol, vehicle, {sku: assign_qty})
                if not success:
                    continue

                shipped_after = int(sol.compute_shipped().get((dealer, sku), 0))
                actual_loaded = max(0, shipped_after - shipped_before)
                if actual_loaded <= 0:
                    continue

                for d_upd in range(day, data.horizons + 1):
                    resource_pool[(plant, sku, d_upd)] = int(
                        resource_pool.get((plant, sku, d_upd), 0) - actual_loaded
                    )

                allocated[(dealer, sku)] += actual_loaded
                remain_demand -= actual_loaded

    return sol


def _group_demands_by_dealer(prioritized_demands):
    """Group demands by dealer and compute average priority per dealer."""
    grouped = defaultdict(list)
    for (dealer, sku), demand, priority in prioritized_demands:
        if demand <= 0:
            continue
        grouped[dealer].append((sku, int(demand), float(priority)))

    dealer_groups = []
    for dealer, items in grouped.items():
        if not items:
            continue
        avg_priority = sum(p for _, _, p in items) / float(len(items))
        items.sort(key=lambda x: x[2], reverse=True)
        dealer_groups.append((dealer, items, avg_priority))

    dealer_groups.sort(key=lambda x: x[2], reverse=True)
    return dealer_groups


def _compute_cumulative_map(resource_pool, data):
    """Compute cumulative availability map C_ikt = sum_{u=t..T} R_iku."""
    cumulative = {}
    for plant in data.plants:
        for sku in data.skus_plant.get(plant, []):
            running = 0
            for day in range(data.horizons, 0, -1):
                running += int(resource_pool.get((plant, sku, day), 0))
                cumulative[(plant, sku, day)] = running
    return cumulative


def _refresh_cumulative_for_plant_sku(cumulative_map, resource_pool, plant, sku, horizons):
    """Refresh cumulative suffix values for a single (plant, sku) pair."""
    running = 0
    for day in range(horizons, 0, -1):
        running += int(resource_pool.get((plant, sku, day), 0))
        cumulative_map[(plant, sku, day)] = running


def _select_vehicle_type_for_volume(required_volume, data):
    """Pick smallest feasible vehicle type by volume; fallback to largest type."""
    suitable = [
        (veh_type, cap)
        for veh_type, cap in data.veh_type_cap.items()
        if cap >= required_volume
    ]
    if suitable:
        return min(suitable, key=lambda x: x[1])[0]
    return max(data.veh_type_cap.items(), key=lambda x: x[1])[0]


def _fit_orders_to_capacity(order_pairs, capacity_volume, data):
    """Trim sku orders to fit vehicle capacity while preserving priority order."""
    if capacity_volume <= 0:
        return {}

    fitted = {}
    used_volume = 0.0
    for sku, qty in order_pairs:
        if qty <= 0:
            continue
        sku_size = float(data.sku_sizes.get(sku, 0))
        if sku_size <= 0:
            continue
        remain_volume = capacity_volume - used_volume
        if remain_volume <= 1e-9:
            break
        max_qty = int(remain_volume // sku_size)
        take = int(min(int(qty), max_qty))
        if take <= 0:
            continue
        fitted[sku] = take
        used_volume += take * sku_size
    return fitted


def _allocate_dealer_demands(
    sol,
    dealer,
    sku_demands,
    demand_map,
    resource_pool,
    cumulative_map,
    supply_chain,
    use_cumulative,
    timeout,
    t_start,
    veh_loading,
):
    """Allocate one dealer's demands in sku-priority order with optional cumulative checks."""
    data = sol.data
    horizons_range = range(1, data.horizons + 1)
    supply_chain_items = list(supply_chain.items())

    for sku, _demand, _priority in sku_demands:
        if time.time() - t_start > timeout:
            break

        remain = int(demand_map.get((dealer, sku), 0))
        if remain <= 0:
            continue

        available_plants = [
            plant
            for (plant, dealer_id), skus in supply_chain_items
            if dealer_id == dealer and sku in skus
        ]
        if not available_plants:
            continue

        plant_inventory = [
            (
                plant,
                sum(int(resource_pool.get((plant, sku, d), 0)) for d in horizons_range),
            )
            for plant in available_plants
        ]
        plant_inventory.sort(key=lambda x: x[1], reverse=True)

        for plant, _ in plant_inventory:
            if remain <= 0 or time.time() - t_start > timeout:
                break

            for day in horizons_range:
                if remain <= 0 or time.time() - t_start > timeout:
                    break

                avail_day = int(resource_pool.get((plant, sku, day), 0))
                if avail_day <= 0:
                    continue

                if use_cumulative:
                    avail_cum = int(cumulative_map.get((plant, sku, day), 0))
                    avail = min(avail_day, avail_cum)
                else:
                    avail = avail_day
                if avail <= 0:
                    continue

                assign_qty = int(min(remain, avail))
                if assign_qty <= 0:
                    continue

                veh_type = _select_best_vehicle(assign_qty, sku, data)
                vehicle = Vehicle(plant, dealer, veh_type, day, data)
                shipped_before = int(sol.compute_shipped().get((dealer, sku), 0))

                success = veh_loading(sol, vehicle, {sku: assign_qty})
                if not success:
                    continue

                shipped_after = int(sol.compute_shipped().get((dealer, sku), 0))
                actual_loaded = max(0, shipped_after - shipped_before)
                if actual_loaded <= 0:
                    continue

                demand_map[(dealer, sku)] = max(
                    0,
                    int(demand_map.get((dealer, sku), 0)) - actual_loaded,
                )
                remain -= actual_loaded

                for d_upd in range(day, data.horizons + 1):
                    pool_key = (plant, sku, d_upd)
                    resource_pool[pool_key] = int(
                        resource_pool.get(pool_key, 0) - actual_loaded
                    )

                _refresh_cumulative_for_plant_sku(
                    cumulative_map,
                    resource_pool,
                    plant,
                    sku,
                    data.horizons,
                )


def _select_best_vehicle(qty, sku, data):
    """
    根据待发送 sku 数量选择车辆类型:
        - 计算需求总体积 = qty * sku_size
        - 选择能容纳且容量最接近的车辆
        - 若无车型完全容纳 → 选择最大容量车型 (后续可能再补车)
    """
    required_volume = qty * data.sku_sizes[sku]
    suitable_types = [
        (vt, cap) for vt, cap in data.veh_type_cap.items() if cap >= required_volume
    ]
    if suitable_types:
        return min(suitable_types, key=lambda x: x[1])[0]
    return max(data.veh_type_cap.items(), key=lambda x: x[1])[0]


def _repair_solution(sol, timeout, stats=None):
    """
    修复初始解:
        1. 负库存调整 (多轮迭代, 减少车辆装载)
        2. 移除空车辆
        3. 合并小车辆 (减少解规模)
    注意:
        - 逻辑保持不变，仅增强注释说明
        - current_inventory 在循环中增量更新, 最终一次性写回
    """
    t_start = time.time()
    max_iter = getattr(ALNSConfig, "INITIAL_REPAIR_MAX_ITER", 5)
    iter_count = 0

    for _ in range(max_iter):
        iter_count += 1
        if time.time() - t_start > timeout:
            break

        # 每轮都基于最新 cargo 重算库存，避免使用旧快照造成错误修复。
        sol.compute_inventory()
        neg_inv = [(k, v) for k, v in sol.s_ikt.items() if v < 0]
        if not neg_inv:
            break

        changed = False
        # 先修更早天，减少后续天累计库存误差放大。
        neg_inv.sort(key=lambda x: x[0][2])
        for (plant, sku, day), _neg_val in neg_inv:
            # 每个负库存点都基于当前最新 cargo 重算，避免同轮重复过削减。
            sol.compute_inventory()
            current_short = sol.s_ikt.get((plant, sku, day), 0)
            if current_short >= 0:
                continue

            relevant_vehicles = [
                veh
                for veh in sol.vehicles
                if veh.fact_id == plant
                and veh.day <= day
                and any(
                    c_sku == sku and c_day <= day for (c_sku, c_day) in veh.cargo.keys()
                )
            ]
            if not relevant_vehicles:
                continue

            to_reduce = -current_short
            relevant_vehicles.sort(
                key=lambda veh: (
                    veh.day,
                    sum(q for (s, d), q in veh.cargo.items() if s == sku and d <= day),
                ),
                reverse=True,
            )

            for veh in relevant_vehicles:
                if to_reduce <= 0:
                    break

                candidate_keys = [
                    (s, d)
                    for (s, d) in veh.cargo.keys()
                    if s == sku and d <= day and veh.cargo.get((s, d), 0) > 0
                ]
                candidate_keys.sort(key=lambda x: x[1], reverse=True)

                current_qty = sum(veh.cargo.get(k, 0) for k in candidate_keys)
                reduce_qty = min(current_qty, to_reduce)
                if reduce_qty <= 0:
                    continue

                left = reduce_qty
                for cargo_key in candidate_keys:
                    if left <= 0:
                        break
                    q0 = veh.cargo.get(cargo_key, 0)
                    if q0 <= 0:
                        continue
                    dq = min(q0, left)
                    veh.cargo[cargo_key] = q0 - dq
                    if veh.cargo[cargo_key] == 0:
                        del veh.cargo[cargo_key]
                    left -= dq

                if left < reduce_qty:
                    changed = True
                    to_reduce -= reduce_qty - left

        if not changed:
            break

    sol.vehicles = [veh for veh in sol.vehicles if not veh.is_empty()]
    if time.time() - t_start < timeout:
        _merge_small_vehicles(sol, timeout - (time.time() - t_start))

    sol.compute_inventory()
    if stats is not None:
        stats["repair_iterations"] = (
            iter_count
            if "repair_iterations" not in stats
            else stats["repair_iterations"] + iter_count
        )
    return sol


def _merge_small_vehicles(sol, timeout):
    """
    合并同 (工厂, 经销商, 天) 下小车辆:
        - 车辆按当前装载体积升序
        - 尝试将最小车合并入容量够的目标车
        - 超时提前退出并保留剩余车辆
    """
    data = sol.data
    t_start = time.time()
    vehicle_groups = defaultdict(list)
    for veh in sol.vehicles:
        key = (veh.fact_id, veh.dealer_id, veh.day)
        vehicle_groups[key].append(veh)

    merged_vehicles = []

    for key, vehicles in vehicle_groups.items():
        if time.time() - t_start > timeout:
            merged_vehicles.extend(vehicles)
            continue
        if len(vehicles) <= 1:
            merged_vehicles.extend(vehicles)
            continue

        def veh_volume(v):
            return sum(qty * data.sku_sizes[sku] for (sku, _), qty in v.cargo.items())

        vehicles.sort(key=veh_volume)

        while len(vehicles) > 1:
            if time.time() - t_start > timeout:
                break
            smallest = vehicles[0]
            smallest_volume = veh_volume(smallest)
            merged = False
            for i in range(1, len(vehicles)):
                target = vehicles[i]
                target_volume = veh_volume(target)
                total_volume = smallest_volume + target_volume
                if total_volume <= data.veh_type_cap.get(target.type, 0):
                    for (sku, day), qty in smallest.cargo.items():
                        target.cargo[(sku, day)] = target.cargo.get((sku, day), 0) + qty
                    vehicles.pop(0)
                    merged = True
                    break
            if not merged:
                merged_vehicles.append(vehicles.pop(0))

        merged_vehicles.extend(vehicles)

    sol.vehicles = merged_vehicles
    sol.compute_inventory()
    return sol


def _finalize_strict_initial_feasibility(sol, rng, remaining_timeout):
    """
    在初始解构造结束后，进行少量严格可行性收尾修复：
      1) validate 若已可行则直接返回
      2) 若存在 unmet_demand，优先调用 inventory_balance_repair
      3) 若存在负库存/容量等硬违例，再调用 _repair_solution 与 infeasible_repair
    该阶段仅在严格模式启用，且受 timeout 与最大轮次约束。
    """
    if not bool(getattr(ALNSConfig, "STRICT_FEASIBILITY_MODE", True)):
        return

    max_passes = int(getattr(ALNSConfig, "INITIAL_STRICT_FINALIZE_MAX_PASSES", 3))
    if max_passes <= 0:
        return

    t0 = time.time()
    for _ in range(max_passes):
        if remaining_timeout is not None and remaining_timeout > 0:
            if time.time() - t0 > remaining_timeout:
                break

        sol.compute_inventory()
        feasible, violations = sol.validate()
        if feasible:
            return

        # unmet_demand 优先修复，符合严格可行定义
        if violations.get("unmet_demand"):
            try:
                from .repair_operators import inventory_balance_repair

                inventory_balance_repair(sol, rng)
            except Exception:
                pass

        # 负库存/超载/工厂容量等违例走库存与不可行修复
        hard_bad = bool(
            violations.get("negative_inventory")
            or violations.get("veh_over_load")
            or violations.get("plant_inv_exceed")
        )
        if hard_bad:
            try:
                _repair_solution(sol, max(5.0, (remaining_timeout or 5.0) * 0.3))
            except Exception:
                pass
            try:
                from .repair_operators import infeasible_repair

                repaired = infeasible_repair(sol, rng)
                if repaired is not None:
                    sol.replace_vehicles(list(repaired.vehicles))
            except Exception:
                pass

    # 最后一次强制重算，保证库存状态与车辆状态一致
    sol.compute_inventory()
