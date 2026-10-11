"""
模块: alnsopt
核心职责:
1. 定义 SolutionState:
   - 保存车辆集合、逐日逐SKU库存 s_ikt、索引集合、以及用于自适应/ML算子的外部引用
   - 提供目标函数 (带缓存) + 可行性校验 + 库存计算 + 车辆负载等基础查询
2. 目标函数缓存机制:
   - last_objective 保存上一次计算结果
   - objective_dirty 标记结构是否变动(车辆集合/车辆货物/库存表)
   - 结构性修改后调用 mark_objective_dirty(), 下次 objective() 触发重算
3. 库存两种更新路径:
   A. 全表重算 compute_inventory():
      - 用递推关系 s_t = s_{t-1} + prod_t - shipped_t
      - O(V + P*S*H) 复杂度 (V=车辆数, P=工厂数, S=SKU数, H=时间跨度)
      - 作为“基线”库存视图: 假设当前车辆集合的发运已固定
   B. 增量扣减 veh_loading():
      - 针对新增装载 (即 shipped 增加) 直接对 day..H 的 s_ikt 逐期减去 load_qty
      - 原因: 一条发运在递推式中对所有未来期具有线性链式影响
      - 避免每次装载后重复全表重算, 将多次局部装载成本降低到 O(Σ( H - day + 1 ))
      - 需确保 compute_inventory() 已先建立基线 (依赖 s_initialized 标志)
4. 批量库存更新 batch_update_inventory():
   - 用于 destroy / repair 后根据被移除/新增车辆一次性回补或扣减库存
   - 与 veh_loading 的增量策略一致(对 day..H 处理) 但方向相反
5. 重要不变量 / 一致性:
   - s_ikt[(plant, sku, 0)] = 期初库存 (来自 historical_s_ikt)
   - 对任意 (plant, sku, day>0):
       s_ikt[plant, sku, day] = s_ikt[plant, sku, day-1] + production(plant, sku, day) - shipped(plant, sku, day)
   - 增量装载保证上述不变量在"基线+增量扣减"组合下仍成立
6. 典型调用顺序:
   - 初始: __post_init__ 读入期初库存
   - 首次需要库存/目标时: compute_inventory() → s_initialized=True
   - 车辆装载: veh_loading() 若未初始化先全表重算 → 增量扣减库存
   - 销毁/修复: 直接修改 vehicles → batch_update_inventory 回补/扣减库存
   - 目标函数: objective() (利用缓存 + validate() 中触发 compute_inventory())

维护提示:
- 新增修改车辆/库存的算子后务必 mark_objective_dirty()
- 若要引入更细粒度(例如部分 SKU 回退) 需保持“对 day..H 逐日更新”一致性
"""

# =========================
# 标准库
# =========================
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Set, Optional
import traceback

# =========================
# 第三方库
# =========================
import numpy as np
import numpy.random as rnd

# =========================
# 项目内部模块
# =========================
from .alnstrack import ALNSTracker
from .vehicle import Vehicle
from .InputDataALNS import DataALNS
from .optutility import POSITIVEEPS
from .initial_solution_utils import improved_initial_solution
from .param_tuner import ParamAutoTuner
from .ml_operator_selector import MLOperatorSelector
from .alns_config import default_config as ALNSConfig
from .inventory_manager import UnifiedInventoryManager


@dataclass
class SolutionState:
    data: DataALNS
    vehicles: List[Vehicle] = field(default_factory=list)
    s_ikt: Dict[Tuple[str, str, int], int] = field(default_factory=dict)
    s_indices: Set[Tuple[str, str, int]] = field(default_factory=set)
    tracker: Optional["ALNSTracker"] = (
        None  # 引入追踪器, 便于访问迭代历史数据,用于ML训练
    )
    s_initialized: bool = field(default=False)  # 引入标志, 标记 s_ikt 是否已初始化
    param_tuner: Optional["ParamAutoTuner"] = None  # 参数自动调整器
    ml_selector: Optional["MLOperatorSelector"] = None  # ML算子选择器
    inventory_manager: Optional[Any] = field(default=None, repr=False, compare=False)
    _inventory_cache: Optional[Any] = field(default=None, repr=False, compare=False)
    _suppress_compute_inventory: bool = field(default=False, repr=False, compare=False)
    repair_audit: Dict[str, Dict[str, Any]] = field(
        default_factory=dict, repr=False, compare=False
    )
    # ---- 目标函数缓存相关 ----
    shipped_cache: Optional[Dict[Tuple[str, str], int]] = field(
        default=None
    )  # cache for compute_shipped
    last_objective: Optional[float] = field(default=None)  # 上一次计算的目标值
    objective_dirty: bool = field(
        default=True
    )  # 标记当前解结构是否发生变动(车辆/库存等)需要重新计算

    def __post_init__(self):
        self.vehicles = []
        self.s_ikt = {}
        self.s_indices = self.construct_indices()
        self._iteration_count = 0  # 迭代计数器, 用于ALNSTracker
        if self.inventory_manager is None:
            self.inventory_manager = UnifiedInventoryManager()

        # 对s_ikt进行初始化, s_ik0表示期初库存
        for (plant, sku_id, day), inv in self.data.historical_s_ikt.items():
            if day == 0:
                self.s_ikt[plant, sku_id, day] = inv

    def set_tracker(self, tracker: "ALNSTracker"):
        """
        设置tracker引用, 用于ML-based operators访问迭代历史数据
        """
        self.tracker = tracker

    def set_param_tuner(self, param_tuner: "ParamAutoTuner"):
        """
        设置参数自动调整器引用
        """
        self.param_tuner = param_tuner

    def set_ml_selector(self, ml_selector: "MLOperatorSelector"):
        """
        设置ML算子选择器引用
        """
        self.ml_selector = ml_selector

    def iter_used_vehicles(self):
        """Iterate over non-empty vehicles in the current state."""
        for veh in self.vehicles:
            try:
                if veh is None or veh.is_empty():
                    continue
            except Exception:
                continue
            yield veh

    def count_used_vehicles(self) -> int:
        """Return the number of non-empty vehicles currently in the state."""
        return sum(1 for _ in self.iter_used_vehicles())

    def replace_vehicles(self, vehicles: List[Vehicle]):
        """Replace the vehicle list and invalidate all derived caches."""
        self.vehicles = list(vehicles or [])
        self.s_initialized = False
        self.mark_objective_dirty()

    # ---------------- compute_inventory 抑制机制（用于批量/模拟场景以避免不必要全表重算） ----------------
    def suppress_compute_inventory(self):
        """
        启用抑制：后续对 validate()/objective() 的调用不会触发 compute_inventory()。
        调用者在批量修改结束后应显式调用 resume_compute_inventory() 或在退出上下文时恢复。
        """
        setattr(self, "_suppress_compute_inventory", True)

    def resume_compute_inventory(self):
        """
        取消抑制：恢复 validate()/objective() 中的正常 compute_inventory() 调用行为。
        """
        setattr(self, "_suppress_compute_inventory", False)

    def suppress_inventory_ctx(self):
        """
        返回一个简易上下文管理器，用于临时抑制 compute_inventory()。
        用法:
            with state.suppress_inventory_ctx():
                ... 批量修改 state, 不会在 validate/objective 中触发 compute_inventory ...
            # 退出上下文后自动恢复
        """

        class _Ctx:
            def __init__(self, state):
                self._state = state

            def __enter__(self):
                setattr(self._state, "_suppress_compute_inventory", True)
                return self._state

            def __exit__(self, exc_type, exc, tb):
                try:
                    setattr(self._state, "_suppress_compute_inventory", False)
                except Exception:
                    pass

        return _Ctx(self)

    # ---------------- 目标值缓存/失效控制 ----------------
    def mark_objective_dirty(self, clear_shipped: bool = True):
        """
        标记当前解的目标值缓存失效。

        参数:
          clear_shipped: bool
            当 True（默认）时同时清除 shipped_cache（保持向后兼容）。
            当 False 时只失效目标缓存/其他按-state 缓存，但保留 shipped_cache 以启用增量维护。
        """
        self.objective_dirty = True
        # 清除上一次计算结果（目标缓存）; shipped_cache 根据参数决定是否清除
        self.last_objective = None
        if clear_shipped:
            self.shipped_cache = None
        # 清除与资源/车辆聚合相关的按-state 缓存，保证下次重建数据一致性
        try:
            if hasattr(self, "_vehicles_by_plant_day_map"):
                delattr(self, "_vehicles_by_plant_day_map")
        except Exception:
            pass
        try:
            if hasattr(self, "_resource_pool_cache"):
                delattr(self, "_resource_pool_cache")
        except Exception:
            pass

    def record_repair_audit(
        self,
        op_name: str,
        feasible: bool,
        violations: Optional[Dict[str, Any]] = None,
        rolled_back: bool = False,
        inserted_cnt: int = 0,
        improvement: float = 0.0,
    ):
        """Aggregate repair feasibility diagnostics per operator."""
        name = str(op_name or "unknown_repair")
        entry = self.repair_audit.setdefault(
            name,
            {
                "attempts": 0,
                "feasible": 0,
                "infeasible": 0,
                "rolled_back": 0,
                "inserted_total": 0,
                "improvement_total": 0.0,
                "fail_negative_inventory": 0,
                "fail_veh_over_load": 0,
                "fail_plant_inv_exceed": 0,
                "fail_unmet_demand": 0,
            },
        )
        entry["attempts"] += 1
        if feasible:
            entry["feasible"] += 1
        else:
            entry["infeasible"] += 1
        if rolled_back:
            entry["rolled_back"] += 1
        entry["inserted_total"] += int(max(0, inserted_cnt))
        entry["improvement_total"] += float(improvement)

        viol = violations or {}
        if viol.get("negative_inventory"):
            entry["fail_negative_inventory"] += 1
        if viol.get("veh_over_load"):
            entry["fail_veh_over_load"] += 1
        if viol.get("plant_inv_exceed"):
            entry["fail_plant_inv_exceed"] += 1
        if viol.get("unmet_demand"):
            entry["fail_unmet_demand"] += 1

    def validate(self):
        """
        验证解是否为可行解, 无副作用。
        返回 (is_feasible, violations) 元组:
        - is_feasible: bool, 是否可行
        - violations: dict, 包含各类违反项的详细信息
        """
        # 在严格可行性模式下，validate始终重算库存，避免抑制标志导致的陈旧库存被误判为可行。
        force_recompute = bool(
            getattr(ALNSConfig, "VALIDATE_FORCE_RECOMPUTE_INVENTORY", True)
            or getattr(ALNSConfig, "STRICT_FEASIBILITY_MODE", True)
        )
        if force_recompute or (not getattr(self, "_suppress_compute_inventory", False)):
            self.compute_inventory()
        violations = {
            "negative_inventory": [],
            "veh_over_load": [],
            "plant_inv_exceed": [],
            "unmet_demand": [],
        }

        # 尝试使用 ndarray（NumPy/Numba）加速负库存与工厂日合计检查；失败时回退到 dict 迭代
        try:
            from . import inventory_numba

            maps = getattr(self, "_inventory_maps", None)
            if maps is None:
                maps = inventory_numba.build_index_maps(self.data)
            # 使用 state_to_inventory_arrays 生成 s_arr（shape P,S,H+1）
            s_arr, _, _ = inventory_numba.state_to_inventory_arrays(self, maps)
            # 负库存索引查找
            neg_idx = np.where(s_arr < 0)
            for pi, si, d in zip(*neg_idx):
                plant = maps.idx_to_plant[pi]
                sku = maps.idx_to_sku[si]
                inv = float(s_arr[pi, si, d])
                violations["negative_inventory"].append(((plant, sku, int(d)), inv))
        except Exception:
            # 回退到原有的 dict 遍历方式
            for key, inv in self.s_ikt.items():
                if inv < 0:
                    violations["negative_inventory"].append((key, inv))

        # 检查车辆容量是否超限
        for veh in self.vehicles:
            try:
                loaded = self.compute_veh_load(veh)
                cap = self.data.veh_type_cap[veh.type]
                if loaded - cap > POSITIVEEPS:
                    info = {
                        "veh_id": getattr(veh, "id", None),
                        "fact_id": getattr(veh, "fact_id", None),
                        "dealer_id": getattr(veh, "dealer_id", None),
                        "type": getattr(veh, "type", None),
                        "day": getattr(veh, "day", None),
                        "loaded": loaded,
                        "cap": cap,
                        "cargo": dict(getattr(veh, "cargo", {})),
                    }
                    violations["veh_over_load"].append(info)
                    # Overload is a solution-feasibility issue, not a code-error signal.
            except Exception:
                # Defensive: record exception during load calculation
                try:
                    err = traceback.format_exc()
                    violations["veh_over_load"].append({"veh": veh, "error": err})
                    # Keep silent by default; callers can inspect violations if needed.
                except Exception:
                    pass

        # 检查在每个周期内, 生产基地中的库存是否超过限制
        # 尝试使用 ndarray 聚合（按 plant 汇总 day），失败时回退到 dict 聚合
        try:
            from . import inventory_numba

            maps = getattr(self, "_inventory_maps", None)
            if maps is None:
                maps = inventory_numba.build_index_maps(self.data)
            s_arr, _, _ = inventory_numba.state_to_inventory_arrays(self, maps)
            # sum over sku axis -> shape (P, H+1)
            plant_day_tot = s_arr.sum(axis=1)
            for pi, plant in enumerate(maps.idx_to_plant):
                for d in range(plant_day_tot.shape[1]):
                    total_inv = float(plant_day_tot[pi, d])
                    max_cap = self.data.plant_inv_limit.get(plant, float("inf"))
                    if total_inv - max_cap > POSITIVEEPS:
                        violations["plant_inv_exceed"].append(
                            {
                                "plant": plant,
                                "day": int(d),
                                "total_inv": total_inv,
                                "max_cap": max_cap,
                            }
                        )
        except Exception:
            plant_day_inventory = defaultdict(int)
            for (plant, sku, day), inv in self.s_ikt.items():
                plant_day_inventory[(plant, day)] += inv

            for (plant, day), total_inv in plant_day_inventory.items():
                max_cap = self.data.plant_inv_limit[plant]
                if total_inv - max_cap > POSITIVEEPS:
                    violations["plant_inv_exceed"].append(
                        {
                            "plant": plant,
                            "day": day,
                            "total_inv": total_inv,
                            "max_cap": max_cap,
                        }
                    )

        # 严格模式下，将“完全满足需求”视为硬可行性条件。
        require_full_demand = bool(
            getattr(ALNSConfig, "VALIDATE_REQUIRE_FULL_DEMAND", True)
            or getattr(ALNSConfig, "STRICT_FEASIBILITY_MODE", True)
        )
        if require_full_demand:
            shipped = self.compute_shipped()
            for (dealer, sku_id), demand in self.data.demands.items():
                shipped_qty = shipped.get((dealer, sku_id), 0)
                if shipped_qty + POSITIVEEPS < demand:
                    violations["unmet_demand"].append(
                        {
                            "dealer": dealer,
                            "sku_id": sku_id,
                            "demand": demand,
                            "shipped": shipped_qty,
                        }
                    )

        # 综合判断解的可行性
        is_feasible = not (
            violations["negative_inventory"]
            or violations["veh_over_load"]
            or violations["plant_inv_exceed"]
            or violations["unmet_demand"]
        )

        return is_feasible, violations

    def objective(self) -> float:
        """
        搜索目标函数(带缓存)：
        - 用于 ALNS 搜索引导和接受准则。
        - 由“实际目标”作为基底，再叠加搜索阶段的引导惩罚（如缺货惩罚、车数惩罚、欠载差额惩罚）。
        - 注意：这不是对外评估口径；对外评估请使用 actual_objective()。
        """
        # 若缓存可用直接返回
        if (not self.objective_dirty) and (self.last_objective is not None):
            return self.last_objective

        scale_factor = float(getattr(ALNSConfig, "SCALE_FACTOR", 1e-3))

        # validate 内部会在必要时触发 compute_inventory()
        feasibility, _ = self.validate()
        if not feasibility:
            self.last_objective = float("inf")
            self.objective_dirty = False
            return self.last_objective

        data = self.data

        # ---- 基底：实际目标（与 Gurobi 口径对齐）----
        actual_base = self.actual_objective()

        # 根据进度做分阶段权重：前期更激进压缩车辆数，后期回归综合成本
        progress = 0.0
        if bool(getattr(ALNSConfig, "SEARCH_PHASED_VEHICLE_PRIORITY", True)):
            try:
                if self.tracker is not None:
                    cur_it = int(self.tracker.get_iteration())
                    max_it = int(self.tracker.get_max_iterations())
                    if max_it > 0:
                        progress = max(0.0, min(1.0, float(cur_it) / float(max_it)))
            except Exception:
                progress = 0.0

        phase1_end = float(getattr(ALNSConfig, "SEARCH_PHASE1_END_PROGRESS", 0.45))
        phase2_end = float(getattr(ALNSConfig, "SEARCH_PHASE2_END_PROGRESS", 0.80))
        if progress < phase1_end:
            demand_weight = float(
                getattr(ALNSConfig, "SEARCH_PHASE1_DEMAND_WEIGHT", 0.35)
            )
            underload_weight = float(
                getattr(ALNSConfig, "SEARCH_PHASE1_UNDERLOAD_WEIGHT", 0.80)
            )
            fleet_weight = float(
                getattr(ALNSConfig, "SEARCH_PHASE1_FLEET_WEIGHT", 4.00)
            )
        elif progress < phase2_end:
            demand_weight = float(
                getattr(ALNSConfig, "SEARCH_PHASE2_DEMAND_WEIGHT", 0.80)
            )
            underload_weight = float(
                getattr(ALNSConfig, "SEARCH_PHASE2_UNDERLOAD_WEIGHT", 1.00)
            )
            fleet_weight = float(
                getattr(ALNSConfig, "SEARCH_PHASE2_FLEET_WEIGHT", 2.00)
            )
        else:
            demand_weight = float(
                getattr(ALNSConfig, "SEARCH_PHASE3_DEMAND_WEIGHT", 1.00)
            )
            underload_weight = float(
                getattr(ALNSConfig, "SEARCH_PHASE3_UNDERLOAD_WEIGHT", 1.00)
            )
            fleet_weight = float(
                getattr(ALNSConfig, "SEARCH_PHASE3_FLEET_WEIGHT", 1.00)
            )

        # ---- 引导项1：未满足需求惩罚（用于搜索引导）----
        demand_penalty = 0.0
        unmet_total = 0.0
        demand_total = 0.0
        if bool(getattr(ALNSConfig, "SEARCH_USE_DEMAND_PENALTY", True)):
            shipped = self.compute_shipped()
            for (dealer, sku_id), demand in data.demands.items():
                demand_total += float(demand)
                shipped_qty = shipped.get((dealer, sku_id), 0)
                if shipped_qty < demand:
                    gap = float(demand - shipped_qty)
                    unmet_total += gap
                    demand_penalty += data.param_pun_factor1 * gap
            demand_penalty *= demand_weight

        # ---- 引导项2：最小起运量“差额”惩罚（用于搜索引导）----
        underload_gap_penalty = 0.0
        if bool(getattr(ALNSConfig, "SEARCH_USE_UNDERLOAD_GAP_PENALTY", True)):
            _, _, underload_gap_penalty = self._vehicle_cost_components()
            underload_gap_penalty *= underload_weight

        # ---- 引导项3：显式车辆数惩罚（用于引导更少车辆）----
        fleet_size_penalty = 0.0
        if bool(getattr(ALNSConfig, "ENABLE_FLEET_SIZE_PENALTY", True)):
            fleet_size = len(self.vehicles)
            baseline = int(getattr(ALNSConfig, "FLEET_SIZE_BASELINE", 25))
            penalty_rate = float(
                getattr(ALNSConfig, "FLEET_SIZE_PENALTY_PER_VEHICLE", 15.0)
            )
            if fleet_size > baseline:
                fleet_size_penalty = (fleet_size - baseline) * penalty_rate
            fleet_size_penalty *= fleet_weight

        # Two-phase shaping: when unmet demand exists, prioritize feasibility over fleet compression.
        if (
            bool(getattr(ALNSConfig, "SEARCH_FEASIBILITY_FIRST", True))
            and unmet_total > 0.0
        ):
            demand_boost = float(
                getattr(ALNSConfig, "SEARCH_FEASIBILITY_DEMAND_BOOST", 6.0)
            )
            fleet_scale = float(
                getattr(ALNSConfig, "SEARCH_FEASIBILITY_FLEET_SCALE_WHEN_UNMET", 0.15)
            )
            underload_scale = float(
                getattr(
                    ALNSConfig,
                    "SEARCH_FEASIBILITY_UNDERLOAD_SCALE_WHEN_UNMET",
                    0.25,
                )
            )
            demand_penalty *= max(1.0, demand_boost)
            fleet_size_penalty *= max(0.0, fleet_scale)
            underload_gap_penalty *= max(0.0, underload_scale)

        search_cost = (
            float(actual_base)
            + float(demand_penalty)
            + float(underload_gap_penalty)
            + float(fleet_size_penalty)
        )

        self.last_objective = search_cost * scale_factor
        self.objective_dirty = False
        return float(self.last_objective)

    def _vehicle_cost_components(self) -> Tuple[float, int, float]:
        """单次遍历车辆成本组件：固定成本、欠载车辆数、欠载差额惩罚（体积差额）。"""
        data = self.data
        fixed_cost = 0.0
        underload_vehicle_count = 0
        underload_gap_penalty = 0.0

        for veh in self.vehicles:
            fixed_cost += float(data.veh_type_cost.get(veh.type, 0.0))

            loaded = getattr(veh, "_loaded_volume", None)
            if loaded is None:
                lv = 0.0
                for (sku_id, _day), qty in veh.cargo.items():
                    lv += float(data.sku_sizes.get(sku_id, 0.0)) * float(qty)
                loaded = lv

            min_load = float(data.veh_type_min_load.get(veh.type, 0.0))
            if float(loaded) < min_load:
                underload_vehicle_count += 1
                underload_gap_penalty += float(data.param_pun_factor3) * (
                    min_load - float(loaded)
                )

        return fixed_cost, underload_vehicle_count, underload_gap_penalty

    def actual_objective(self) -> float:
        """
        实际目标函数（与 Gurobi 评估口径一致）：
        = 车辆固定成本 + 欠载车辆惩罚成本
        其中欠载车辆惩罚按“欠载车辆数量 * param_pun_objective”计算。
        """
        feasibility, _ = self.validate()
        if not feasibility:
            return float("inf")

        fixed_cost, underload_vehicle_count, _ = self._vehicle_cost_components()
        punishment_cost = float(underload_vehicle_count) * float(
            self.data.param_pun_objective
        )
        return float(fixed_cost + punishment_cost)

    def calculate_cost(self) -> float:
        """兼容旧接口：返回与 Gurobi 口径一致的实际目标值。"""
        return float(self.actual_objective())

    def punish_non_fulfill_demand(self, obj: float):
        """检查是否满足经销商的需求, 如果不满足, 需要再目标函数中添加惩罚项"""
        shipped = self.compute_shipped()
        for (dealer, sku_id), demand in self.data.demands.items():
            shipped_qty = shipped.get((dealer, sku_id), 0)
            if shipped_qty < demand:
                obj += self.data.param_pun_factor1 * (demand - shipped_qty)
        return obj

    def compute_veh_load(self, veh: Vehicle) -> float:
        """计算当前车辆装载量。优先使用 Vehicle._loaded_volume 缓存以避免遍历 cargo。"""
        # 优先使用 Vehicle 上的已装载体积缓存（若算子/改动已维护该缓存）
        loaded = getattr(veh, "_loaded_volume", None)
        if loaded is not None:
            return loaded
        # 回退：遍历 cargo 计算体积（兼容未维护缓存的 Vehicle 实例）
        total_volume = 0
        for (sku_id, day), qty in veh.cargo.items():
            total_volume += self.data.sku_sizes[sku_id] * qty
        return total_volume

    def punish_deficient_veh_min_load(self, obj: float):
        """判断当前车辆是否满足最小起运量, 如果不满足, 需要再目标函数中添加惩罚项"""
        for veh in self.vehicles:
            total_volume = self.compute_veh_load(veh)
            min_load = self.data.veh_type_min_load[veh.type]
            if total_volume < min_load:
                obj += self.data.param_pun_factor3 * (min_load - total_volume)
        return obj

    def compute_shipped(self) -> Dict[Tuple[str, str], int]:
        """
        计算在所有周期内, 满足经销商 j 对 SKU k 的需求量

        优化说明:
          - 使用 local defaultdict 聚合，避免在热路径中大量调用 dict.get。
          - 避免不必要的 dict.copy()，只在缓存时将结果转换为常规 dict 存储。
        """
        if (
            getattr(self, "shipped_cache", None) is not None
            and not self.objective_dirty
        ):
            shipped_cache = self.shipped_cache
            if shipped_cache is None:
                return {}
            return shipped_cache

        # 使用局部变量和 defaultdict 加速聚合
        shipped_agg = defaultdict(int)
        vehicles = self.vehicles  # local ref for speed
        for veh in vehicles:
            dealer = veh.dealer_id
            # micro-opt: avoid tuple allocations inside inner loop where possible
            for (sku_id, day), qty in veh.cargo.items():
                shipped_agg[(dealer, sku_id)] += qty

        # 将结果转换为普通 dict 并缓存
        self.shipped_cache = dict(shipped_agg)
        return self.shipped_cache if self.shipped_cache is not None else {}

    def construct_indices(self):
        """
        Construct and cache the (plant, sku, day) index set.
        Cache the intermediate plant-sku pairs on the state instance so repeated
        calls (e.g. during long-running profiling) avoid reconstructing supply
        chain mappings.
        """
        if getattr(self, "_plant_sku_pairs", None) is None:
            triple_plant_dealer_sku = {
                (plant, dealer, sku_id)
                for (plant, dealer), skus in self.data.construct_supply_chain().items()
                for sku_id in skus
            }
            # cache reduced (plant, sku) pairs for reuse
            self._plant_sku_pairs = {
                (plant, sku_id) for (plant, dealer, sku_id) in triple_plant_dealer_sku
            }
        s_indices = {
            (plant, sku_id, day)
            for (plant, sku_id) in self._plant_sku_pairs
            for day in range(self.data.horizons + 1)
        }
        return s_indices

    def compute_inventory(self):
        """
        全表库存重算: 优先尝试数组化/Numba 加速实现 (若可用)，在失败时回退到 Python 实现。
        目的：无缝替换以获得加速，同时保持原有行为与语义。
        """
        # Try fast path using multi_period_delivery.inventory_numba.compute_inventory_fast if available.
        try:
            try:
                from . import inventory_numba

                # inventory_numba.compute_inventory_fast returns True on success
                res = inventory_numba.compute_inventory_fast(self)
                if not res:
                    raise RuntimeError("inventory_numba fast path failed")
                # ensure s_initialized semantics preserved by fallback/core
                try:
                    self.s_initialized = True
                except Exception:
                    pass
                return res
            except Exception:
                # If import or fast path fails, fall through to pure-Python baseline below
                pass

            # --- Pure-Python baseline (original implementation) ---
            # 1) 预计算每个 (plant, sku, day) 的当日发运量，避免在每个索引上扫描车辆列表
            shipped_by_plant_sku_day = defaultdict(int)
            for veh in self.vehicles:
                fact = veh.fact_id
                d = veh.day
                for (sku_id, day_k), q in veh.cargo.items():
                    # 只计当日发运
                    if day_k == d:
                        shipped_by_plant_sku_day[(fact, sku_id, day_k)] += q

            # 为保证递推（s_t = s_{t-1} + prod_t - shipped_t）正确、稳定，按 (plant, sku)
            # 分组并对 day 升序迭代，而不是依赖 self.s_indices 的无序迭代。否则当 day
            # 的前驱 day-1 尚未计算时会得到错误的 prev_inventory=0。
            plant_sku_pairs = {(plant, sku) for (plant, sku, d) in self.s_indices}
            for plant, sku_id in plant_sku_pairs:
                for day in range(1, self.data.horizons + 1):  # 只处理 day>0 的情况
                    # 直接从预计算表查询当日发运量
                    shipped_from_plant = shipped_by_plant_sku_day.get(
                        (plant, sku_id, day), 0
                    )

                    # 获取前一天的库存, 确保期初库存被正确考虑
                    prev_inventory = self.s_ikt.get((plant, sku_id, day - 1), 0)

                    # 获取当天的生产量
                    production = self.data.sku_prod_each_day.get(
                        (plant, sku_id, day), 0
                    )

                    # 计算当前库存: 前一天库存 + 当天生产 - 当天发出
                    current_inventory = prev_inventory + production - shipped_from_plant

                    # 更新库存
                    self.s_ikt[(plant, sku_id, day)] = current_inventory

            # 完成全表重算后标记 s_ikt 已初始化 (future periods 已计算完)
            self.s_initialized = True
            return True
        finally:
            # ensure objective semantics: callers expect compute_inventory to set related caches as necessary
            try:
                # calling mark_objective_dirty false here would be incorrect; preserve existing behavior by not forcing it
                pass
            except Exception:
                pass

    def copy(self, clone_vehicles: bool = True):
        """
        Lightweight copy of the current state without invoking __post_init__.

        Parameters:
        - clone_vehicles: when True (default) create shallow clones of each Vehicle using
          Vehicle.shallow_clone() to ensure the returned state is fully independent.
          When False, perform a shallow list copy of the vehicles list (i.e. new list object
          but sharing Vehicle instances). The latter is much cheaper and safe for read-mostly
          simulation scenarios where callers guarantee not to mutate existing Vehicle objects.

        Use object.__new__ to avoid the overhead of constructing a fresh SolutionState
        (which would recompute indices and reinitialize structures). Fields are shallow-
        copied where safe.
        """
        new_state = object.__new__(SolutionState)
        # shallow copy simple containers / references
        new_state.data = self.data
        # Vehicles: either clone each vehicle (safe but more expensive) or shallow-copy list
        if clone_vehicles:
            new_state.vehicles = [v.shallow_clone() for v in self.vehicles]
        else:
            # create a new list object but reuse vehicle instances (caller must avoid mutating them)
            new_state.vehicles = list(self.vehicles)
        # copy inventory map (caller may mutate s_ikt during simulation)
        new_state.s_ikt = dict(self.s_ikt)
        # preserve the exact s_indices structure (triples) to avoid unpacking errors
        new_state.s_indices = set(self.s_indices)
        # preserve external references
        new_state.tracker = self.tracker
        new_state.param_tuner = self.param_tuner
        new_state.ml_selector = self.ml_selector
        new_state.inventory_manager = self.inventory_manager
        new_state.s_initialized = bool(self.s_initialized)
        # copy caches/cached values (ship cache copied shallowly)
        new_state.shipped_cache = (
            self.shipped_cache.copy() if self.shipped_cache is not None else None
        )
        new_state.last_objective = self.last_objective
        new_state.objective_dirty = self.objective_dirty
        # preserve suppress flag
        new_state._suppress_compute_inventory = getattr(
            self, "_suppress_compute_inventory", False
        )
        new_state.repair_audit = {
            k: dict(v) for k, v in getattr(self, "repair_audit", {}).items()
        }
        return new_state


def veh_loading(state: SolutionState, veh: Vehicle, orders: Dict[str, int]):
    """
    车辆装载 + 增量库存更新 (核心性能优化点)

    核心思想:
    - 不在每次“装一段货”后整体调用 compute_inventory()
    - 利用线性递推链条: 当某 (plant, sku) 在 day 发运量 +q 时
        对所有 d >= day 的 s_ikt(plant, sku, d) 都应 -q
      因此可以一次性对 day..H 区间做逐日扣减，等价于完整递推更新的累积效果。

    步骤:
    1. 若基线库存尚未初始化 (s_initialized=False) → compute_inventory()
    2. 针对每个订单 SKU:
       a. 计算可用库存 available = s_{day-1} + production_day - 已装载(同厂同日累计)
       b. 计算车辆容量上限转化为可装 SKU 数量
       c. 得到实际装载量 load_qty = min(订单剩余, available, 容量允许)
       d. veh.load() 记录货物, 更新 used_inv
       e. 对 day..H 的 s_ikt 逐日扣减 load_qty (增量反映链式影响)
       f. 若车辆装满继续以同样逻辑创建新车 (保持拆车粒度一致)
    3. 末尾将非空车辆加入解；标记 objective 缓存失效

    复杂度对比:
    - 传统: 每次装载后全表 O(P*S*H) 重算 → 多次装载代价巨大
    - 现方案: 单次装载 O(H - day + 1) 扣减；若装载发生在后期 day 接近 H 成本更低
    - 典型情形下大幅降低初始解生成与修复算子阶段的计算开销

    正确性依赖:
    - compute_inventory() 提供正确基线 (包含之前所有车辆发运量)
    - 增量扣减严格遵循递推链 (所有未来期同步调整)
    - destroy/repair 引起的车辆移除/新增用 batch_update_inventory 做对称加减

    注意:
    - 不处理跨日装载：cargo 中 day_k 必须等于车辆出车日 veh.day
    - available 计算基于“前一日库存 + 当日产出 - 已装载”，不直接看 s_ikt[day] 避免双重扣减
    - 若未来引入部分退货/回滚，需要提供与本逻辑对称的“增量加回”实现

    返回:
    - True 表示装载流程正常完成
    """
    data = state.data
    fact_id, dealer_id, veh_type, day = veh.fact_id, veh.dealer_id, veh.type, veh.day

    # 必须先建立库存基线，再做增量扣减；否则会出现双重扣减或从未初始化库存直接扣减。
    if not state.s_initialized:
        state.compute_inventory()

    # 合并遍历：一次遍历构建 used_inv 和 vehicle_ids（减少两次循环开销）
    used_inv = {}
    vehicles_list = state.vehicles
    vehicle_ids = set()
    for v in vehicles_list:
        if v.fact_id != fact_id or v.day != day:
            continue
        vehicle_ids.add(v.id)
        for (sku_id, d), qty in v.cargo.items():
            if d == day:
                used_inv[sku_id] = used_inv.get(sku_id, 0) + qty

    # prepare locals for hot loop
    s_ikt = state.s_ikt
    sku_sizes = data.sku_sizes
    prod_get = data.sku_prod_each_day.get
    horizons = data.horizons

    for sku_id, order_qty in orders.items():
        if sku_id not in data.skus_plant.get(fact_id, []):
            continue
        remain_qty = order_qty

        while remain_qty > 0:
            # 计算可用库存
            prev_inv = s_ikt.get((fact_id, sku_id, day - 1), 0)
            production = prod_get((fact_id, sku_id, day), 0)
            used_qty = used_inv.get(sku_id, 0)
            available = prev_inv + production - used_qty
            if available <= 0:
                break

            # 直接使用 veh.capacity 作为剩余可用体积，避免重复计算 current_load
            sku_size = sku_sizes[sku_id]
            max_qty_by_cap = veh.capacity // sku_size  # veh最多能装载的该SKU数量
            # 实际需要的装载量, 这里考虑剩余订单量、可用库存和车辆容量
            load_qty = int(min(remain_qty, available, max_qty_by_cap))

            if load_qty <= 0:
                # 当前车辆已满, 加入队列, 换新车
                if not veh.is_empty():
                    if veh.id not in vehicle_ids:
                        state.vehicles.append(veh)
                        vehicle_ids.add(veh.id)
                veh = Vehicle(fact_id, dealer_id, veh_type, day, data)
                continue

            veh.load(sku_id, int(load_qty))
            used_inv[sku_id] = used_inv.get(sku_id, 0) + load_qty
            remain_qty -= load_qty

            # 将该次装载量从装运当天及之后的库存中扣减
            for d in range(day, horizons + 1):
                key = (fact_id, sku_id, d)
                s_ikt[key] = int(s_ikt.get(key, 0) - load_qty)

        # 最后一次循环后, 若veh有货且未加入, 则加入（用 id 集合避免昂贵的 eq 比较）
        if not veh.is_empty() and veh.id not in vehicle_ids:
            state.vehicles.append(veh)

    # 增量更新已在装载时完成，不再进行全表 recompute 以减少开销

    # 装载操作已改变车辆与库存结构, 目标缓存失效
    state.mark_objective_dirty()
    # Clear inventory fast-path cache to ensure next compute_inventory rebuilds arrays
    try:
        if hasattr(state, "_inventory_cache"):
            state._inventory_cache = None
    except Exception:
        pass
    return True


def batch_update_inventory(state: SolutionState, vehicles: List[Vehicle], op_type: str):
    """
    批量更新库存并返回应用的更新摘要。
    返回值:
      inventory_updates: dict{(plant, sku, day): delta_applied}
    目的:
      - 让调用者能够校验实际应用的回补/扣减是否与预期一致
      - 在发生异常或不一致时便于记录/回退/诊断
    备注:
      - 对于 destroy: 将 removed 车辆的 cargo 按 day..H 加回库存 (delta 为正)
      - 对于 repair : 将新增车辆的 cargo 按 day..H 减去库存 (delta 为负)
    """
    inventory_updates = defaultdict(int)
    if not vehicles:
        return inventory_updates

    # 确保库存已经初始化
    if not state.s_initialized:
        state.compute_inventory()

    # 聚合所有待更新的 (plant, sku, day) -> delta
    for veh in vehicles:
        for (sku_id, d_shipped), qty in veh.cargo.items():
            # 收集所有需要更新的库存变化
            for d in range(d_shipped, state.data.horizons + 1):
                key = (veh.fact_id, sku_id, d)
                if op_type == "destroy":
                    inventory_updates[key] += qty
                elif op_type == "repair":
                    inventory_updates[key] -= qty

    # 应用所有更新并记录最终实际应用量（写入 state.s_ikt）
    s_ikt_get = state.s_ikt.get
    for key, delta in inventory_updates.items():
        state.s_ikt[key] = s_ikt_get(key, 0) + delta

    # 库存被批量改写, 使目标缓存失效
    state.mark_objective_dirty()
    # Clear inventory fast-path cache to ensure next compute_inventory rebuilds arrays
    try:
        if hasattr(state, "_inventory_cache"):
            state._inventory_cache = None
    except Exception:
        pass

    # 返回摘要，供调用方进行比对/记录
    return dict(inventory_updates)


def initial_solution(state: SolutionState, rng: rnd.Generator):
    """generate initial solution"""
    solution = improved_initial_solution(state, rng)
    return solution
