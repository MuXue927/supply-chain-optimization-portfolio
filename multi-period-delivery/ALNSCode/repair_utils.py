"""
repair_utils.py

提取自 ALNSCode/repair_operators.py 的辅助/工具函数模块。
供 repair_operators.py 中的算子调用，保持原实现与文档注释。
"""

from typing import Dict, List, Tuple, Set, Optional
import time
import math
import random
from dataclasses import dataclass

import numpy as np
import numpy.random as rnd

from .alnsopt import SolutionState
from .vehicle import Vehicle
from .InputDataALNS import DataALNS
from .param_tuner import ParamAutoTuner
from .alns_config import default_config as ALNSConfig
from .inventory_utils import precompute_plant_day_inventory, precompute_dealer_shipments


# MinimalVehicle: lightweight vehicle surrogate for fast simulation in regret operator
# Purpose:
# - Provide a minimal API compatible with alnsopt.veh_loading so regret simulations can
#   avoid the full Vehicle dataclass construction overhead in tight inner loops.
# - Keep implementation intentionally small and allocation-cheap.
# Notes:
# - Instances are NOT registered with global Vehicle._id_counter and thus ids are local-only.
# - Attributes/methods implemented: fact_id, dealer_id, type, day, data, id, max_capacity,
#   capacity, cargo (dict), _loaded_volume, load(), is_empty().
# - This surrogate is intended only for temporary simulation on copied states (tmp) where
#   strict global id uniqueness is not required. Do NOT use in persistent state.
class MinimalVehicle:
    _local_id_seq = 0

    def __init__(self, fact_id, dealer_id, veh_type, day, data):
        self.fact_id = fact_id
        self.dealer_id = dealer_id
        self.type = veh_type
        self.day = day
        self.data = data
        # lightweight id (unique per process but not interfering with Vehicle._id_counter)
        self.id = f"minveh_{MinimalVehicle._local_id_seq}"
        MinimalVehicle._local_id_seq += 1
        # capacities
        try:
            self.max_capacity = data.veh_type_cap[veh_type]
        except Exception:
            self.max_capacity = 0
        self.capacity = self.max_capacity
        self.cargo = {}
        self._loaded_volume = 0

    def load(self, sku_id: str, num: int) -> int:
        # simplified load similar to Vehicle.load but without heavy checks
        if num <= 0:
            return num
        sku_size = self.data.sku_sizes.get(sku_id, 0)
        if sku_size <= 0:
            raise ValueError(f"Invalid sku_size for {sku_id}")
        max_by_cap = self.capacity // sku_size
        num_loaded = min(num, max_by_cap)
        if num_loaded <= 0:
            return num
        prev = self.cargo.get((sku_id, self.day), 0)
        self.cargo[(sku_id, self.day)] = prev + num_loaded
        self.capacity -= num_loaded * sku_size
        self._loaded_volume = getattr(self, "_loaded_volume", 0) + num_loaded * sku_size
        return num - num_loaded

    def is_empty(self):
        return not bool(self.cargo)


def _get_actual_reference(state) -> Optional[float]:
    tracker = getattr(state, "tracker", None)
    if tracker is None:
        return None

    for attr in ("initial_actual_obj", "best_actual_obj"):
        try:
            ref = float(getattr(tracker, attr, float("inf")))
        except Exception:
            continue
        if np.isfinite(ref) and abs(ref) > 1e-12:
            return ref
    return None


def compute_mixed_operator_reward(
    state: SolutionState,
    prev_obj: Optional[float],
    new_obj: Optional[float] = None,
    actual_obj: Optional[float] = None,
):
    """Compute a mixed search+actual reward for operator feedback."""
    if prev_obj is None:
        return None, 0.0, False

    if new_obj is None:
        try:
            new_obj = state.objective()
        except Exception:
            return None, 0.0, False

    if actual_obj is None:
        try:
            actual_obj = float(state.calculate_cost())
        except Exception:
            actual_obj = None

    try:
        search_weight = float(getattr(ALNSConfig, "MIXED_REWARD_SEARCH_WEIGHT", 0.60))
        actual_weight = float(getattr(ALNSConfig, "MIXED_REWARD_ACTUAL_WEIGHT", 0.40))
        score_scale = float(getattr(ALNSConfig, "MIXED_REWARD_SCORE_SCALE", 100.0))
    except Exception:
        search_weight = 0.60
        actual_weight = 0.40
        score_scale = 100.0

    search_norm = 0.0
    if np.isfinite(prev_obj) and np.isfinite(new_obj):
        search_norm = float((prev_obj - new_obj) / max(1.0, abs(prev_obj)))

    actual_norm = 0.0
    actual_ref = _get_actual_reference(state)
    if actual_ref is not None and actual_obj is not None and np.isfinite(actual_obj):
        actual_norm = float((actual_ref - actual_obj) / max(1.0, abs(actual_ref)))

    mixed_reward = score_scale * (
        search_weight * search_norm + actual_weight * actual_norm
    )
    if not np.isfinite(mixed_reward):
        mixed_reward = 0.0

    success = mixed_reward > 0.0
    return new_obj, float(mixed_reward), bool(success)


# ---------------------------
# 基础资源/需求工具函数
# ---------------------------
def _construct_resource_availability(
    state: SolutionState, plant: str, sku_id: str, day: int
) -> float:
    """
    返回 (plant, sku_id, day) 的可用库存数量, 不修改 state

    优化说明:
      - 原实现对每次调用都会遍历 state.vehicles，导致在热点路径被反复调用时开销巨大。
      - 这里引入一个轻量缓存：state._vehicles_by_plant_day_map
        它是一个 dict[(plant, day)] -> dict[sku_id] -> used_qty，按需构建并在 state 结构性改变时被清除。
      - 该缓存能将多次 O(V) 扫描降为一次 O(V) 构建，随后查表为 O(1)。
    """
    # 基线库存和产量
    prev_inv = state.s_ikt.get((plant, sku_id, day - 1), 0)
    production = state.data.sku_prod_each_day.get((plant, sku_id, day), 0)

    # 尝试复用缓存的按 (plant, day) 聚合的已装载映射
    vbkey = "_vehicles_by_plant_day_map"
    vehicles_map = getattr(state, vbkey, None)
    if vehicles_map is None:
        # 构建一次并缓存到 state（由 mark_objective_dirty 在结构变化时清除）
        vehicles_map = {}
        for veh in state.vehicles:
            key = (veh.fact_id, veh.day)
            m = vehicles_map.get(key)
            if m is None:
                m = {}
                vehicles_map[key] = m
            for (sku_k, d), q in veh.cargo.items():
                if d == veh.day:
                    m[sku_k] = m.get(sku_k, 0) + q
        try:
            setattr(state, vbkey, vehicles_map)
        except Exception:
            # 若无法设置属性则继续不缓存（兼容只读对象）
            pass

    used_inv = vehicles_map.get((plant, day), {}).get(sku_id, 0)
    return max(0, prev_inv + production - used_inv)


def _construct_resource_pool(
    state: SolutionState, data: DataALNS
) -> Dict[Tuple[str, str, int], float]:
    """
    构建资源池, 预先计算 (plant, sku, day) 的可用库存

    优化说明:
      - 使用按-state缓存 (_resource_pool_cache) 避免在短时间内重复构建资源池。
      - 当 state 结构变化时会由 mark_objective_dirty 清除缓存。
    """
    # 尝试复用缓存
    cache = getattr(state, "_resource_pool_cache", None)
    if cache is not None:
        return cache

    pool = build_resource_pool(state, data)
    try:
        setattr(state, "_resource_pool_cache", pool)
    except Exception:
        # 如果无法设置属性（只读对象等），则跳过缓存
        pass
    return pool


def get_unmet_demand(state: SolutionState) -> Dict[Tuple[str, str], int]:
    """
    返回当前解state中所有未满足的需求字典:
      key: (dealer, sku_id)
      value: unmet_quantity

    优化说明:
      - 这是 regret 算子中的热点函数（在短时间内会被频繁调用）。
      - 引入基于轻量 state snapshot 的缓存（(len(vehicles), last_objective)）以避免
        在 state 未发生结构性变化时重复调用 compute_shipped() 与遍历 demands。
      - 缓存存储在 state._unmet_cache，且在 state.mark_objective_dirty() 被调用时应失效
       （mark_objective_dirty 已在其它修改中实现对相关缓存的清理；若不存在则本函数
        也会在检测到结构变化后重建缓存）。
    """
    try:
        # lightweight snapshot that changes when vehicles list or objective baseline changes
        snapshot = (
            len(getattr(state, "vehicles", [])),
            getattr(state, "last_objective", None),
        )
    except Exception:
        snapshot = None

    # try to reuse cached value if present and snapshot matches
    cache = getattr(state, "_unmet_cache", None)
    if cache is not None:
        try:
            cached_snapshot, cached_removal = cache
            if snapshot is not None and cached_snapshot == snapshot:
                return cached_removal
        except Exception:
            # fallthrough to recompute
            pass

    # compute shipped only once for this call
    shipped = state.compute_shipped()
    removal_list: Dict[Tuple[str, str], int] = {}
    # iterate demands directly (avoid extra allocations where possible)
    for key, total_demand in state.data.demands.items():
        shipped_qty = shipped.get(key, 0)
        unmet_qty = total_demand - shipped_qty
        if unmet_qty > 0:
            removal_list[key] = unmet_qty

    # store into cache (best-effort; ignore failures on read-only wrappers)
    try:
        setattr(state, "_unmet_cache", (snapshot, removal_list))
    except Exception:
        pass

    return removal_list


# ---------------------------
# veh_loading wrapper（复用 alnsopt.veh_loading）
# ---------------------------
def veh_loading(
    state: SolutionState,
    veh: Vehicle,
    orders: Dict[str, int],
    commit: bool = True,
    skip_compute_inventory: bool = True,
) -> bool:
    """
    Wrapper that delegates to ALNSCode.alnsopt.veh_loading.

    新增参数:
      skip_compute_inventory: 当 commit=True 时，若为 True 则临时抑制 state.compute_inventory 的调用，
        以便在批量场景下减少全表计算次数（调用方负责在批量结束后再调用一次 state.compute_inventory()）。
        该优化为可选，遇到异常会回退到原始行为。

    注意:
      - 不修改 alnsopt.veh_loading 的签名；通过在 wrapper 层临时替换 state.compute_inventory 来避免多次全表更新.
      - 在任何异常或不兼容情况下请务必恢复原 compute_inventory 以保证一致性。
    """
    try:
        from . import alnsopt as _alnsopt
    except Exception:
        raise RuntimeError("无法导入 ALNSCode.alnsopt, veh_loading wrapper 无法工作。")

    if not commit:
        # 在副本上模拟，不触及真实 state
        tmp = state.copy()
        _alnsopt.veh_loading(tmp, veh, orders)
        return True

    # commit == True: 尝试在允许的情况下抑制 compute_inventory 以批量更新
    if skip_compute_inventory and hasattr(state, "compute_inventory"):
        orig_compute = getattr(state, "compute_inventory")
        try:
            # 先建立库存基线，再进入抑制模式，避免在未初始化 s_ikt 上做增量扣减。
            if not getattr(state, "s_initialized", False):
                orig_compute()
            # 临时禁用 compute_inventory（简单替换为 no-op）
            state.compute_inventory = lambda: None
            _alnsopt.veh_loading(state, veh, orders)
        except Exception:
            # 若发生任何异常，保证恢复并向上抛出以触发回退逻辑
            try:
                state.compute_inventory = orig_compute
            except Exception:
                pass
            raise
        finally:
            # 恢复原始函数（保护性恢复）
            try:
                state.compute_inventory = orig_compute
            except Exception:
                pass
        return True
    else:
        # 原始行为（不抑制 compute_inventory）
        _alnsopt.veh_loading(state, veh, orders)
        return True


def select_vehicle_type_for_volume(required_volume: float, data: DataALNS) -> str:
    """Choose the smallest vehicle type that can carry required volume; fallback to the largest type."""
    suitable = []
    for vt, cap in data.veh_type_cap.items():
        if cap >= required_volume:
            suitable.append((vt, cap))
    if suitable:
        return min(suitable, key=lambda x: x[1])[0]
    return max(data.veh_type_cap.keys(), key=lambda x: data.veh_type_cap[x])


def find_existing_route_vehicles(
    state: SolutionState, plant: str, dealer: str, day: int
) -> List[Vehicle]:
    """Return non-full vehicles on the same (plant, dealer, day), sorted by remaining capacity desc."""
    candidates: List[Vehicle] = []
    for veh in state.vehicles:
        if veh.fact_id != plant or veh.dealer_id != dealer or veh.day != day:
            continue
        try:
            if veh.is_full():
                continue
        except Exception:
            pass
        candidates.append(veh)
    candidates.sort(key=lambda v: getattr(v, "capacity", 0), reverse=True)
    return candidates


def commit_load_with_tracking(
    state: SolutionState,
    veh: Vehicle,
    sku_id: str,
    qty: int,
    day: int,
    skip_compute_inventory: bool = True,
) -> Tuple[int, List[Vehicle]]:
    """
    Commit a single-SKU load and return (actual_loaded_qty, appended_new_vehicles).
    actual_loaded_qty is measured as route-level sku delta to robustly include overflow-created vehicles.
    """
    request_qty = int(max(0, qty))
    if request_qty <= 0:
        return 0, []

    before_len = len(state.vehicles)
    before_route_qty = 0
    for v in state.vehicles:
        if v.fact_id == veh.fact_id and v.day == day:
            before_route_qty += int(v.cargo.get((sku_id, day), 0))

    veh_loading(
        state,
        veh,
        {sku_id: request_qty},
        commit=True,
        skip_compute_inventory=skip_compute_inventory,
    )

    after_route_qty = 0
    for v in state.vehicles:
        if v.fact_id == veh.fact_id and v.day == day:
            after_route_qty += int(v.cargo.get((sku_id, day), 0))

    actual_loaded = max(0, int(after_route_qty - before_route_qty))
    new_vs = state.vehicles[before_len:]
    return actual_loaded, new_vs


# ---------------------------
# 预计算与资源池工具（greedy 支持）
# ---------------------------
def precompute_total_available(
    state: SolutionState, data: DataALNS, supply_chain: Dict[Tuple[str, str], Set[str]]
) -> Dict[Tuple[str, str], float]:
    """
    预计算每个 (dealer, sku_id) 的跨工厂、跨期总的理论可用量（期初库存 + 生产）
    返回 dict[(dealer, sku_id)] -> total_available (int/float)
    """
    total_avail = {}
    for (dealer, sku_id), demand in data.demands.items():
        s = 0
        # 只汇总能为该 dealer 提供该 sku 的工厂
        plants = [
            plant
            for (plant, dealer_key), skus in supply_chain.items()
            if dealer_key == dealer and sku_id in skus
        ]
        for plant in plants:
            for day in range(1, data.horizons + 1):
                s += state.s_ikt.get(
                    (plant, sku_id, day - 1), 0
                ) + data.sku_prod_each_day.get((plant, sku_id, day), 0)
        total_avail[(dealer, sku_id)] = s
    return total_avail


class ResourcePool:
    """
    Lightweight proxy that stores resource availability in a nested structure
    self._data[plant][sku_id][day] = available

    Provides a dict-like tuple-key interface so existing callers that use
    resource_pool.get((plant, sku, day), 0) or resource_pool[(plant, sku, day)]
    continue to work without change, but hot-path tuple lookups avoid building
    large flat dicts repeatedly.
    """

    def __init__(self):
        self._data = {}

    def _ensure(self, plant, sku):
        if plant not in self._data:
            self._data[plant] = {}
        if sku not in self._data[plant]:
            self._data[plant][sku] = {}

    def get(self, key, default=0):
        # Accept either tuple key (plant, sku, day) or explicit (plant, sku, day)
        try:
            plant, sku, day = key
        except Exception:
            return default
        return self._data.get(plant, {}).get(sku, {}).get(day, default)

    def get_day_map(self, key_or_tuple):
        """
        Return the internal day->available dict for (plant, sku).
        Accepts either (plant, sku) tuple or the same tuple wrapped as (plant, sku).
        Returns an empty dict if not present to allow safe .items()/get() usage in hot loops.
        This avoids tuple-key lookups for per-day access in performance-critical code paths.
        """
        try:
            plant, sku = key_or_tuple
        except Exception:
            return {}
        return self._data.get(plant, {}).get(sku, {})

    def __getitem__(self, key):
        try:
            plant, sku, day = key
        except Exception:
            raise KeyError(key)
        return self._data[plant][sku][day]

    def __setitem__(self, key, value):
        try:
            plant, sku, day = key
        except Exception:
            raise KeyError(key)
        self._ensure(plant, sku)
        self._data[plant][sku][day] = value

    def __contains__(self, key):
        try:
            plant, sku, day = key
        except Exception:
            return False
        return day in self._data.get(plant, {}).get(sku, {})

    def items(self):
        # iterate flat (plant, sku, day) -> value pairs
        for plant, sku_map in self._data.items():
            for sku, day_map in sku_map.items():
                for day, val in day_map.items():
                    yield (plant, sku, day), val

    def keys(self):
        for plant, sku_map in self._data.items():
            for sku, day_map in sku_map.items():
                for day in day_map.keys():
                    yield (plant, sku, day)

    def __repr__(self):
        return f"<ResourcePool plants={len(self._data)}>"


def build_resource_pool(state: SolutionState, data: DataALNS):
    """
    构建资源池 ResourcePool，返回 ResourcePool 对象。
    兼容旧调用方式（tuple-key get/set），但在内部使用嵌套结构降低 tuple-key 哈希开销。
    """
    pool = ResourcePool()
    # 采用本地汇总当前 vehicles 使用量以避免在循环中重复遍历 state.vehicles
    vehicles_by_plant_day = {}
    for veh in state.vehicles:
        key = (veh.fact_id, veh.day)
        vehicles_by_plant_day.setdefault(key, []).append(veh)

    for plant in data.plants:
        for sku_id in data.skus_plant.get(plant, []):
            for day in range(1, data.horizons + 1):
                prev_inv = state.s_ikt.get((plant, sku_id, day - 1), 0)
                prod = data.sku_prod_each_day.get((plant, sku_id, day), 0)
                used = 0
                for veh in vehicles_by_plant_day.get((plant, day), []):
                    used += veh.cargo.get((sku_id, day), 0)
                pool[(plant, sku_id, day)] = max(0, prev_inv + prod - used)
    return pool


# ---------------------------------------------------------------------
# 获取算子自适应参数 与 批量后处理
# ---------------------------------------------------------------------
def get_adaptive_parameters(op_name: str, param_tuner: ParamAutoTuner):
    """
    获取某算子当前动态调参结果(若 ParamAutoTuner 不存在/异常则返回空字典)。
    :param op_name: 算子名称 (例如 'random_removal')
    :return: dict
    """
    if not param_tuner:
        return {}
    try:
        params = param_tuner.get_operator_params(op_name)
        return params if params else {}
    except Exception as e:
        print(f"[WARNING] get_adaptive_parameters({op_name}) 异常: {e}")
        return {}


def _veh_volume(veh: Vehicle, data: DataALNS) -> float:
    # Use cargo-based recomputation here (instead of _loaded_volume cache) because
    # some operators may leave cached volume temporarily stale during complex edits.
    # Consolidation must be conservative to avoid accidental overload merges.
    total = 0.0
    for (sku_id, _day), qty in getattr(veh, "cargo", {}).items():
        total += float(qty) * float(data.sku_sizes.get(sku_id, 0.0))
    return total


def consolidate_route_vehicles(
    state: SolutionState, max_pairs_per_group: int = 300
) -> int:
    """
    Merge small vehicles into larger ones for the same (plant, dealer, day) route.
    Returns merged vehicle count.
    """
    data = state.data
    groups = {}
    for veh in state.vehicles:
        try:
            if veh.is_empty():
                continue
        except Exception:
            pass
        key = (veh.fact_id, veh.dealer_id, veh.day)
        groups.setdefault(key, []).append(veh)

    merged_total = 0
    new_vehicles = []

    for _key, vehicles in groups.items():
        if len(vehicles) <= 1:
            new_vehicles.extend(vehicles)
            continue

        # Small-first packing into larger-capacity candidates.
        vehicles.sort(key=lambda v: _veh_volume(v, data))
        pairs_used = 0

        while len(vehicles) > 1 and pairs_used < max_pairs_per_group:
            smallest = vehicles[0]
            small_vol = _veh_volume(smallest, data)
            merged = False

            for i in range(len(vehicles) - 1, 0, -1):
                target = vehicles[i]
                target_vol = _veh_volume(target, data)
                target_cap = float(data.veh_type_cap.get(target.type, 0.0))
                if target_vol + small_vol <= target_cap + 1e-9:
                    for (sku, d), qty in smallest.cargo.items():
                        target.cargo[(sku, d)] = target.cargo.get((sku, d), 0) + qty
                    vehicles.pop(0)
                    merged_total += 1
                    pairs_used += 1
                    merged = True
                    break

            if not merged:
                new_vehicles.append(vehicles.pop(0))

        new_vehicles.extend(vehicles)

    if merged_total > 0:
        state.vehicles = new_vehicles
        try:
            state.mark_objective_dirty()
        except Exception:
            pass
    return merged_total


def post_repair(
    state: SolutionState,
    inserted: List[Vehicle],
    op_name: str,
    prev_obj: float = None,
    t0: float = None,
    extra: dict = None,
):
    """
    批量修复后的后处理：
    - 执行一次 state.compute_inventory()
    - 若发现车辆超载等硬约束违规，尝试回滚本次插入的车辆（按 id 匹配）并再次复核
    - 计算改进并向 param_tuner / 日志上报，打印简短 JSON 风格日志
    """
    merged_after_repair = 0
    try:
        if getattr(ALNSConfig, "REPAIR_ENABLE_POST_CONSOLIDATION", True):
            max_pairs = int(
                getattr(ALNSConfig, "REPAIR_POST_CONSOLIDATION_MAX_PAIRS", 300)
            )
            merged_after_repair = consolidate_route_vehicles(
                state, max_pairs_per_group=max(1, max_pairs)
            )
    except Exception:
        merged_after_repair = 0

    try:
        # Lightweight inserted-only feasibility check to avoid a full validate/compute_inventory
        # in the common (no-violation) case. Conservative semantics:
        #  - Check inserted vehicles for obvious vehicle-overload violations locally.
        #  - If the local check fails or raises, fall back to suppressed validate() as before.
        feasible_tmp = True
        violations_tmp = {}
        try:
            if inserted:
                data = state.data
                veh_over_ids = []
                for v in inserted:
                    try:
                        # compute loaded volume for this vehicle (consider only cargo entries matching the vehicle's day)
                        loaded_vol = 0.0
                        for (sku_k, d), q in getattr(v, "cargo", {}).items():
                            if d == getattr(v, "day", None):
                                sku_size = float(data.sku_sizes.get(sku_k, 0.0))
                                loaded_vol += float(q) * sku_size
                        cap = float(
                            data.veh_type_cap.get(getattr(v, "type", None), 0.0)
                        )
                        # small epsilon to avoid floating point jitter
                        if loaded_vol > cap + 1e-9:
                            veh_over_ids.append(getattr(v, "id", None))
                    except Exception:
                        # any unexpected error -> mark as inconclusive so we fallback to validate()
                        veh_over_ids = ["__error__"]
                        break
                if veh_over_ids:
                    feasible_tmp = False
                    violations_tmp = {"veh_over_load": veh_over_ids}
                else:
                    feasible_tmp = True
                    violations_tmp = {}
            else:
                feasible_tmp = True
                violations_tmp = {}
        except Exception:
            # Fallback: try suppressed validate() (original safe path)
            try:
                with state.suppress_inventory_ctx():
                    feasible_tmp, violations_tmp = state.validate()
            except Exception:
                feasible_tmp, violations_tmp = True, {}
        # If either check detected potential overload or negative inventory, perform full compute to get an
        # accurate baseline before continuing (same conservative behavior as before).
        if (
            not feasible_tmp
            and violations_tmp
            and (
                violations_tmp.get("veh_over_load")
                or violations_tmp.get("negative_inventory")
            )
        ):
            try:
                state.compute_inventory()
            except Exception:
                # If compute_inventory fails, continue; later logic will handle consistency/rollback.
                pass
    except Exception:
        # Defensive: do not let post_repair fail due to inventory checks
        pass

    # 诊断插入车辆摘要（限制采样以避免大量字符串构建/IO 开销）
    try:
        total_inserted = len(inserted) if inserted else 0
        # 仅采样前 20 辆车辆作为诊断示例，避免在大量插入场景下构建完整列表
        inserted_summary = []
        if inserted:
            for v in inserted[:20]:
                try:
                    inserted_summary.append(
                        {
                            "veh_id": getattr(v, "id", None),
                            "fact_id": getattr(v, "fact_id", None),
                            "dealer_id": getattr(v, "dealer_id", None),
                            "type": getattr(v, "type", None),
                            "day": getattr(v, "day", None),
                            "cargo": dict(getattr(v, "cargo", {})),
                        }
                    )
                except Exception:
                    inserted_summary.append({"veh": str(v)})
        # 打印总数与样本摘要（样本最多 20 条）
        try:
            print(
                f"[DIAG] post_repair op={op_name} inserted_cnt={total_inserted} merged_after={merged_after_repair} sample_cnt={len(inserted_summary)}"
            )
        except Exception:
            pass
    except Exception:
        inserted_summary = []

    # 快速验证当前 state 是否可行；若存在 veh_over_load 则尝试回滚本次插入的车辆
    # 如果之前已经做过轻量 inserted-only 检查并得出可行 (feasible_tmp)，则复用其结果以避免重复调用
    try:
        strict_mode = bool(
            getattr(ALNSConfig, "STRICT_FEASIBILITY_MODE", True)
            or getattr(ALNSConfig, "VALIDATE_REQUIRE_FULL_DEMAND", True)
        )
        if ("feasible_tmp" in locals()) and (not strict_mode):
            # 非严格模式下允许复用轻量检查结果，减少重复 validate 开销。
            if feasible_tmp:
                feasible, violations = True, {}
            else:
                try:
                    with state.suppress_inventory_ctx():
                        feasible, violations = state.validate()
                except Exception:
                    feasible, violations = False, {}
        else:
            try:
                feasible, violations = state.validate()
            except Exception:
                feasible, violations = False, {}
    except Exception:
        feasible, violations = False, {}

    rolled_back = False
    hard_violation_exists = bool(
        violations
        and (
            violations.get("veh_over_load")
            or violations.get("negative_inventory")
            or violations.get("plant_inv_exceed")
            or (
                bool(getattr(ALNSConfig, "STRICT_FEASIBILITY_MODE", True))
                and violations.get("unmet_demand")
            )
        )
    )
    if not feasible and hard_violation_exists:
        try:
            # 收集要回滚的车辆 id
            to_rollback_ids = {
                getattr(v, "id", None)
                for v in (inserted or [])
                if getattr(v, "id", None) is not None
            }
            if to_rollback_ids:
                before_cnt = len(state.vehicles)
                # 保留非本次插入的车辆
                state.vehicles = [
                    v
                    for v in state.vehicles
                    if getattr(v, "id", None) not in to_rollback_ids
                ]
                rolled_back = True
                # 使缓存失效并重建基线库存
                try:
                    state.mark_objective_dirty()
                except Exception:
                    pass
                try:
                    state.compute_inventory()
                except Exception:
                    pass
                # 复核
                try:
                    feasible_after, violations_after = state.validate()
                except Exception:
                    feasible_after, violations_after = False, {}
                print(
                    f"[DIAG] post_repair op={op_name} rollback_applied removed={before_cnt - len(state.vehicles)} feasible_after={feasible_after} violations_after_count={sum(len(v) for v in violations_after.values() if v)}"
                )
            else:
                print(
                    f"[DIAG] post_repair op={op_name} detected veh_over_load but no inserted ids to rollback"
                )
        except Exception as e:
            try:
                print(f"[DIAG] post_repair op={op_name} rollback error: {e}")
            except Exception:
                pass

    # 计算改进并上报：优先使用本模块的增量/缓存友好 compute_improvement() 以避免不必要的全表重算。
    new_obj = None
    improvement = 0.0
    success_flag = False
    try:
        if prev_obj is not None:
            # compute_improvement will use state's cached last_objective / objective_dirty semantics
            # and avoid calling state.objective() when possible.
            new_obj, improvement, success_flag = compute_improvement(state, prev_obj)
        else:
            # No prev_obj provided: try to reuse cached objective if present without forcing compute.
            if getattr(state, "last_objective", None) is not None and not getattr(
                state, "objective_dirty", True
            ):
                new_obj = state.last_objective
                improvement = 0.0
                success_flag = False
            else:
                # last resort: attempt to call objective() which may trigger compute_inventory()
                try:
                    new_obj = state.objective()
                    improvement = 0.0
                    success_flag = False
                except Exception:
                    new_obj = None
                    improvement = 0.0
                    success_flag = False
    except Exception:
        new_obj = None
        improvement = 0.0
        success_flag = False

    # 简短日志输出（JSON 风格）
    elapsed = time.time() - t0 if t0 is not None else None
    try:
        print(
            f"[OPJSON] {{'tag':'repair','op':'{op_name}','inserted_cnt':{len(inserted) if inserted else 0},'rolled_back':{rolled_back},'improvement':{improvement},'elapsed_sec':{elapsed}}}"
        )
    except Exception:
        pass

    # 记录修复算子可行性闭环统计，便于离线分析可行率与失败类型。
    try:
        if hasattr(state, "record_repair_audit") and callable(
            getattr(state, "record_repair_audit")
        ):
            state.record_repair_audit(
                op_name=op_name,
                feasible=bool(feasible),
                violations=violations,
                rolled_back=bool(rolled_back),
                inserted_cnt=len(inserted) if inserted else 0,
                improvement=float(improvement),
            )
    except Exception:
        pass


# ---------------------------
# Smart batch helpers（批处理相关）
# ---------------------------
def _get_unsatisfied_demands(state: SolutionState, data: DataALNS):
    shipped = state.compute_shipped()
    unsatisfied = []
    for (dealer, sku_id), demand in data.demands.items():
        shipped_qty = shipped.get((dealer, sku_id), 0)
        if demand > shipped_qty:
            unsatisfied.append(
                {
                    "dealer": dealer,
                    "sku_id": sku_id,
                    "remain_demand": demand - shipped_qty,
                    "total_demand": demand,
                    "priority": demand - shipped_qty,
                }
            )
    return unsatisfied


def _create_smart_batches(unsatisfied_demands, data, batch_size: int):
    if not unsatisfied_demands:
        return []
    sorted_demands = sorted(
        unsatisfied_demands, key=lambda x: x["priority"], reverse=True
    )
    batches = []
    for i in range(0, len(sorted_demands), batch_size):
        batches.append(sorted_demands[i : i + batch_size])
    return batches


def _process_demand_batch(state, batch, resource_pool, data, rng):
    """
    对单个批次进行处理：找到最优分配并通过 veh_loading 提交装载
    返回: batch_progress (bool) 指示是否有进展（至少一笔装载）
    """
    batch_progress = False
    inserted_vehicles = []  # 记录本批次插入的车辆, 用于后续可能的日志或分析

    for demand_info in batch:
        dealer = demand_info["dealer"]
        sku_id = demand_info["sku_id"]
        remain_demand = demand_info["remain_demand"]
        if remain_demand <= 0:
            continue
        supply_chain = data.construct_supply_chain()
        available_plants = [
            plant
            for (plant, dealer_key), skus in supply_chain.items()
            if dealer_key == dealer and sku_id in skus
        ]
        if not available_plants:
            continue
        best = _find_best_allocation(
            remain_demand, sku_id, available_plants, resource_pool, data
        )
        if not best:
            continue
        plant, day, available_qty = best
        loaded_total = 0

        # 先尝试填充已存在车辆，避免无谓新增车辆
        for existing in find_existing_route_vehicles(state, plant, dealer, day):
            if remain_demand <= 0 or available_qty <= 0:
                break
            request_qty = min(remain_demand, available_qty)
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
            loaded_total += actual_loaded
            remain_demand -= actual_loaded
            available_qty -= actual_loaded
            if new_vs:
                inserted_vehicles.extend(new_vs)
            batch_progress = True

        # 再新建车辆补齐剩余需求
        if remain_demand > 0 and available_qty > 0:
            veh_type = _select_optimal_vehicle_type(remain_demand, sku_id, data)
            request_qty = min(remain_demand, available_qty)
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
                if actual_loaded > 0:
                    loaded_total += actual_loaded
                    if new_vs:
                        inserted_vehicles.extend(new_vs)
                    else:
                        inserted_vehicles.append(vehicle)
                    batch_progress = True
            except Exception:
                continue

        if loaded_total > 0:
            resource_pool[(plant, sku_id, day)] = max(
                0, resource_pool.get((plant, sku_id, day), 0) - loaded_total
            )
    return batch_progress, inserted_vehicles


def _find_best_allocation(remain_demand, sku_id, available_plants, resource_pool, data):
    best_score = -1
    best = None
    for plant in available_plants:
        for day in range(1, data.horizons + 1):
            available_qty = resource_pool.get((plant, sku_id, day), 0)  # 可以用的库存
            if available_qty <= 0:
                continue
            satisfaction_ratio = min(1.0, available_qty / remain_demand)  # 满足比例
            efficiency_score = available_qty / max(
                1, available_qty + remain_demand
            )  # 可用库存 占比
            score = satisfaction_ratio * 0.7 + efficiency_score * 0.3
            if score > best_score:
                best_score = score
                best = (plant, day, available_qty)
    return best


def _select_optimal_vehicle_type(demand_qty, sku_id, data):
    required_volume = demand_qty * data.sku_sizes[sku_id]
    return select_vehicle_type_for_volume(required_volume, data)


def _force_load_remaining_demands(state: SolutionState, data: DataALNS):
    """
    当主循环未能完全满足所有需求时，尝试以最保守的方式填充剩余需求, 例如最小车辆、首选工厂/日
    该函数主要作为兜底机制，避免算子超时导致较多遗漏
    """
    shipped = state.compute_shipped()
    inserted_vehicles = []  # 记录插入的车辆, 用于日志或分析
    for (dealer, sku_id), demand in data.demands.items():
        remain = demand - shipped.get((dealer, sku_id), 0)
        if remain <= 0:
            continue
        supply_chain = data.construct_supply_chain()
        available_plants = [
            plant
            for (plant, dealer_key), skus in supply_chain.items()
            if dealer_key == dealer and sku_id in skus
        ]
        if not available_plants:
            continue
        best = None
        best_avail = 0
        for plant in available_plants:
            for day in range(1, data.horizons + 1):
                avail = _construct_resource_availability(state, plant, sku_id, day)
                if avail > best_avail:
                    best_avail = avail
                    best = (plant, day)
        if best is None or best_avail <= 0:
            continue

        plant, day = best
        required_volume = remain * data.sku_sizes.get(sku_id, 0)
        veh_type = select_vehicle_type_for_volume(required_volume, data)

        remain_to_load = min(remain, int(best_avail))
        for existing in find_existing_route_vehicles(state, plant, dealer, day):
            if remain_to_load <= 0:
                break
            try:
                actual_loaded, new_vs = commit_load_with_tracking(
                    state,
                    existing,
                    sku_id,
                    remain_to_load,
                    day,
                    skip_compute_inventory=True,
                )
            except Exception:
                continue
            if actual_loaded <= 0:
                continue
            remain_to_load -= actual_loaded
            if new_vs:
                inserted_vehicles.extend(new_vs)

        if remain_to_load > 0:
            vehicle = Vehicle(plant, dealer, veh_type, day, data)
            try:
                actual_loaded, new_vs = commit_load_with_tracking(
                    state,
                    vehicle,
                    sku_id,
                    remain_to_load,
                    day,
                    skip_compute_inventory=True,
                )
                if actual_loaded > 0:
                    if new_vs:
                        inserted_vehicles.extend(new_vs)
                    else:
                        inserted_vehicles.append(vehicle)
            except Exception:
                continue

    return inserted_vehicles


# ---------------------------
# 改进计算（快速估算 + 精确评估）
# ---------------------------
def compute_improvement_fast_add_vehicle(
    state: SolutionState,
    plant: str,
    dealer: str,
    veh_type: str,
    day: int,
    sku_id: str,
    qty: int,
):
    """
    快速近似评估向解中加入一辆 vehicle (只包含单一 SKU 装载 qty) 对目标的影响。
    说明:
      - 该函数不修改 state，也不复制整个 SolutionState。
      - 使用目标函数的结构性信息进行近似估算：
          improvement_unscaled ≈ param_pun_factor1 * qty - veh_cost - min_load_penalty
      - 返回 (new_obj_est, improvement_scaled, success_flag) 三元组。
    重要优化:
      - 为避免在候选筛选阶段触发昂贵的 state.objective()/compute_inventory()，本实现
        不再调用 state.objective()。new_obj_est 返回 None，调用方仅使用 impro_scaled
        与 success_flag 作为过滤依据；在需要精评时再在副本上运行精确模拟。
    限制:
      - 该估算忽略库存溢出/约束导致的实际不可行性，仅作为 cheap filter 使用。
    """
    data = state.data
    try:
        scale = float(getattr(ALNSConfig, "SCALE_FACTOR", 1e-3))
    except Exception:
        scale = 1e-3

    # 基本参数
    veh_cost = float(data.veh_type_cost.get(veh_type, 0.0))
    try:
        benefit = float(data.param_pun_factor1) * float(qty)
    except Exception:
        benefit = float(qty)

    # 估算最小起运量惩罚（若装载体积不足以满足车辆 min_load）
    sku_size = float(data.sku_sizes.get(sku_id, 0.0))
    load_volume = sku_size * float(qty)
    min_load = float(data.veh_type_min_load.get(veh_type, 0.0))
    min_load_penalty = 0.0
    if load_volume < min_load:
        try:
            min_load_penalty = float(data.param_pun_factor3) * (min_load - load_volume)
        except Exception:
            min_load_penalty = 0.0

    # 估算未缩放的改进值（正表示改进）
    impro_unscaled = benefit - veh_cost - min_load_penalty

    # 转换到 objective 的缩放尺度
    impro_scaled = impro_unscaled * scale

    # 不在此处调用 state.objective() 以避免触发 compute_inventory 等昂贵操作。
    new_obj_est = None

    success = impro_scaled > 0.0
    return new_obj_est, float(impro_scaled), bool(success)


def compute_improvement_delta(
    state: SolutionState,
    prev_obj: Optional[float],
    dealer: str,
    sku_id: str,
    qty: int,
    veh_type: str,
):
    """
    Best-effort incremental delta computation for adding a single-SKU vehicle.

    Returns (new_obj_est, improvement, success).
    This function avoids calling state.objective() / compute_inventory() when possible by
    relying on state.shipped_cache and state.objective_dirty semantics. It is intentionally
    conservative: it only succeeds when a safe incremental estimate can be produced.

    Preconditions for success:
      - prev_obj is provided (scaled objective)
      - state.shipped_cache is present and state.objective_dirty is False (i.e., cached)
      - The candidate is a single-SKU load described by (dealer, sku_id, qty, veh_type)

    The incremental model:
      delta_unscaled = veh_cost + min_load_penalty_of_added_vehicle - demand_penalty_reduction
      new_obj = prev_obj + delta_unscaled * scale_factor
      improvement = prev_obj - new_obj

    If any required information is missing or non-finite, returns (None, 0.0, False).
    """
    try:
        if prev_obj is None:
            return None, 0.0, False

        # Only use the delta path when the state's cached shipped info appears valid.
        if getattr(state, "objective_dirty", True):
            return None, 0.0, False
        shipped_cache = getattr(state, "shipped_cache", None)
        if shipped_cache is None:
            return None, 0.0, False

        data = state.data
        try:
            scale_factor = float(getattr(ALNSConfig, "SCALE_FACTOR", 1e-3))
        except Exception:
            scale_factor = 1e-3

        # current shipped and demand
        current_shipped = int(shipped_cache.get((dealer, sku_id), 0))
        demand = int(data.demands.get((dealer, sku_id), 0))

        # reduction in unmet demand due to adding qty (cannot exceed current unmet)
        unmet_now = max(0, demand - current_shipped)
        reduction = min(int(qty), unmet_now)

        # unscaled components
        try:
            demand_penalty_factor = float(getattr(data, "param_pun_factor1", 1.0))
        except Exception:
            demand_penalty_factor = 1.0
        demand_penalty_reduction = demand_penalty_factor * reduction

        vehicle_cost = float(data.veh_type_cost.get(veh_type, 0.0))
        sku_size = float(data.sku_sizes.get(sku_id, 0.0))
        loaded_vol = float(qty) * sku_size
        min_load = float(data.veh_type_min_load.get(veh_type, 0.0))
        min_load_penalty = 0.0
        if loaded_vol < min_load:
            try:
                min_load_penalty = float(getattr(data, "param_pun_factor3", 0.0)) * (
                    min_load - loaded_vol
                )
            except Exception:
                min_load_penalty = 0.0

        delta_unscaled = vehicle_cost + min_load_penalty - demand_penalty_reduction
        delta_scaled = delta_unscaled * scale_factor

        new_obj = float(prev_obj + delta_scaled)
        improvement = float(np.clip(prev_obj - new_obj, -1e9, 1e9))
        success = improvement > 0.0
        return new_obj, improvement, success
    except Exception:
        return None, 0.0, False


def compute_improvement(state: SolutionState, prev_obj: Optional[float] = None):
    """
    计算改进量, 返回 (new_obj, improvement, success)

    优化说明:
      - 接受可选的 prev_obj，从调用方传入可以避免重复计算“基线”目标。
      - 使用 state 上的缓存 (state.last_objective, state.objective_dirty) 避免对同一 state 重复调用
        state.objective()。在第一次计算后会尝试缓存 new_obj 以供随后复用（低风险、安全回退）。
      - 行为兼容：如果 prev_obj 为 None，仍然返回 (new_obj, 0.0, False) 以保持现有调用约定。
    """
    try:
        # Prefer cached objective when available to avoid recomputation
        last_obj = getattr(state, "last_objective", None)
        dirty = getattr(state, "objective_dirty", True)
        if last_obj is not None and not dirty:
            new_obj = last_obj
        else:
            new_obj = state.objective()
            # best-effort cache the computed objective on the state for subsequent calls
            try:
                state.last_objective = new_obj
                state.objective_dirty = False
            except Exception:
                # ignore caching failures (e.g., read-only wrappers)
                pass

        _, improvement, success = compute_mixed_operator_reward(
            state,
            prev_obj,
            new_obj=new_obj,
        )
    except Exception:
        new_obj = None
        improvement = 0.0
        success = False
    return new_obj, improvement, success
