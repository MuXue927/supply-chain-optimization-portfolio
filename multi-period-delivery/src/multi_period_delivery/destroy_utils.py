"""
模块: destroy_utils.py

职责详细说明:
- 将 destroy_operators.post_destroy 中的辅助逻辑提取为独立函数, 遵循单一职责原则并提升复用性与可测试性。
- 提供一组与“销毁/destroy”相关的工具函数, 包括:
  * numpy -> Python 原生类型安全转换: to_native
  * 被移除车辆货物汇总: summarize_removed_cargo
  * 尝试调用批量库存回补: apply_batch_update
  * 计算改进并向参数调优器上报: compute_improvement_and_report
  * 生成标准化日志载荷: build_opjson_log

设计约定与使用说明, 与项目其它文件注释风格保持一致:
- 函数尽量保持无副作用, 只读/返回值, 除非函数文档明确说明会改变 state。
- 返回的数据结构应为原生 Python 类型或由 to_native 预处理后的结构, 便于日志序列化 (JSON 等)。
- 若调用方修改了 state (例如移除车辆后修改 state.vehicles 或 state.s_ikt), 调用方负责调用 alnsopt 中的批量更新或 compute_inventory 以保持一致性。
- 为避免循环依赖, 某些函数采用延迟导入 (delayed import) alnsopt.batch_update_inventory。

模块内的异常处理策略:
- 这些工具面向稳定的算法流程, 不应因单个异常中断主流程。因此大部分异常被捕获并以安全的“兜底值/标志”返回, 方便上层做进一步处理或回退。
"""

from collections import defaultdict
import time
import logging
import random
from typing import Any, Dict, Tuple, Optional, TYPE_CHECKING
import numpy as np

from .repair_utils import compute_mixed_operator_reward

logger = logging.getLogger(__name__)

# TYPE_CHECKING 用于避免循环导入类型依赖
if TYPE_CHECKING:
    from .alnsopt import SolutionState  # pragma: no cover
else:
    SolutionState = object


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, np.ndarray):
            arr = np.atleast_1d(value)
            return int(arr.flatten()[0])
        return int(value)
    except Exception:
        try:
            return int(np.array(value).flat[0])
        except Exception:
            return default


def _coerce_optional_float(
    value: Any, default: Optional[float] = None
) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        try:
            return float(np.array(value).flat[0])
        except Exception:
            return default


def to_native(v: Any):
    """Lightweight conversion of numpy scalars/arrays to Python native types for logging.

    中文注释, 设计意图与细节:
    - 目的: 将常见的 numpy 标量/数组转换为 Python 原生类型 (int/float/bool/list/None), 以便安全序列化与日志化。
    - 容错策略:
      * 若输入为 None, 直接返回 None。
      * 优先处理 numpy 的标量类型 (np.integer, np.floating, np.bool_), 分别转换为 int/float/bool。
      * 若是 numpy.ndarray, 将其转换为 list, 递归展开为嵌套 list。
      * 对于无法直接识别的类型, 采用逐级降级策略: int -> float -> str -> None, 保证不会抛出异常。
    - 设计考虑:
      * 日志系统与 JSON 序列化通常无法直接处理 numpy 类型, 本函数提供了一个安全边界, 便于在日志记录处调用。
      * 保证函数幂等性, 相同输入尽量返回相同可序列化表示。
    """
    try:
        if v is None:
            return None
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v
    except Exception:
        # 逐级兜底转换, 保证返回原生类型或可字符串化值
        try:
            return int(v)
        except Exception:
            try:
                return float(v)
            except Exception:
                try:
                    return str(v)
                except Exception:
                    return None


def summarize_removed_cargo(removed) -> Dict[Tuple[Any, Any, Any], int]:
    """
    Summarize cargo quantities from a list of removed vehicles.
    Returns dict keyed by (plant, sku_id, day) -> qty

    中文注释, 用途与实现细节:
    - 目的:
      * 把被移除车辆的 cargo 聚合为以 (plant, sku, day) 为键的数量映射, 常用于日志记录或批量库存回补。
    - 容错与实现细节:
      * 使用 getattr 安全访问 veh 的属性, 兼容不同实现的 Vehicle 对象。
      * 对键和值使用 to_native 进行安全转换, 避免 numpy 类型导致的序列化问题。
      * 对异常或非标 qty 采用 best-effort 策略: 先尝试直接 int(qty), 失败时尝试 np.array(qty).flat[0], 仍失败则跳过该条目。
      * 整体采用宽容策略: 单条解析失败不会影响其它条目的统计。
    - 返回:
      * 普通 dict, 键为 (plant, sku_id, day), 值为累积数量int。
    """
    summary = defaultdict(int)
    if not removed:
        return dict(summary)
    for veh in removed:
        try:
            plant = getattr(veh, "fact_id", None)
            for (sku_id, day), qty in getattr(veh, "cargo", {}).items():
                key = (to_native(plant), to_native(sku_id), to_native(day))
                summary[key] += _coerce_int(qty, default=0)
        except Exception:
            # 跳过解析失败的车辆记录, 避免单个错误影响全局统计
            continue
    return dict(summary)


def apply_batch_update(state, removed, skip_batch_update: bool):
    """
    Try to call batch_update_inventory(state, removed, op_type='destroy').
    Returns (inventory_updates dict or None, batch_conflict flag).

    中文注释, 行为与边界情况:
    - 目的:
      * 在 destroy 操作后尝试调用 alnsopt.batch_update_inventory 对 state 进行批量库存回补, destroy 的回补为正向改写。
      * 将库存回补的逻辑与 destroy 操作解耦, 便于在不同上下文中复用。
    - 行为:
      * 若 skip_batch_update 为 True 或 removed 为空, 则直接返回 (None, False) 表示未执行批量回补且无冲突。
      * 延迟导入 alnsopt.batch_update_inventory 以避免循环依赖；若导入并执行成功, 返回其结果, 可能为 {}。
      * 若导入或执行抛出异常, 则返回 (None, True) 表示发生冲突/异常, 上层可基于该标志决定是否采取回退或记录。
    - 错误处理:
      * 本函数不抛异常；通过返回的 batch_conflict 标志向上层传递错误信息, 保持上层流程稳定性。
    """
    inventory_updates = None
    batch_conflict = False
    if removed and not skip_batch_update:
        try:
            # delayed import to avoid circular deps
            from .alnsopt import batch_update_inventory as _batch_update_inventory

            inventory_updates = (
                _batch_update_inventory(state, removed, op_type="destroy") or {}
            )
        except Exception:
            inventory_updates = None
            batch_conflict = True
    return inventory_updates, batch_conflict


def apply_partial_inventory_updates(state, partial_updates: Optional[Dict]):
    """
    Apply additional inventory deltas from destroy operators.

    partial_updates format:
      {(plant, sku, day): delta, ...}
    where delta>0 means inventory add-back for destroy-style partial rollback.
    """
    applied = {}
    if not partial_updates:
        return applied

    if not getattr(state, "s_initialized", False):
        state.compute_inventory()

    s_ikt_get = state.s_ikt.get
    for key, raw_delta in partial_updates.items():
        try:
            plant, sku_id, day = key
        except Exception:
            continue
        delta = _coerce_int(raw_delta, default=0)
        if delta == 0:
            continue
        norm_key = (plant, sku_id, int(day))
        state.s_ikt[norm_key] = s_ikt_get(norm_key, 0) + delta
        applied[norm_key] = applied.get(norm_key, 0) + delta

    if applied:
        try:
            state.mark_objective_dirty()
        except Exception:
            pass
        try:
            if hasattr(state, "_inventory_cache"):
                state._inventory_cache = None
        except Exception:
            pass

    return applied


def compute_improvement_and_report(
    state: SolutionState, prev_obj: Optional[float], op_name: str
):
    """
    Compute objective improvement robustly and report to param_tuner if available.
    Returns (new_obj_or_None, improvement_float, success_flag_bool)

    中文注释, 目的与逻辑要点:
    - 目的:
      * 在算子执行后计算目标改进并, 若存在, 把性能数据回报给参数调优器 (param_tuner) 以便在线调参/学习。
    - 逻辑要点:
      * prev_obj 为 None 时表示无基线, 直接返回 (None, 0.0, False)。
      * 使用 state.objective() 计算 new_obj, 仅当 prev_obj 与 new_obj 均为有限值时计算改进 raw = prev_obj - new_obj。
      * 使用 np.clip 限制改进的数值范围, 避免异常大的数值影响后续统计, 例如数值溢出或异常样本。
      * 成功标志 success = improvement > 0。
      * 若存在 state.param_tuner 且其提供 update_operator_performance, 尝试上报改进与成功标志；上报过程对异常宽容以不影响主流程。
    - 返回:
      * (new_obj, improvement, success)
    """
    new_obj = None
    improvement = 0.0
    success = False
    if prev_obj is None:
        return new_obj, improvement, success

    try:
        new_obj, improvement, success = compute_mixed_operator_reward(
            state,
            prev_obj,
            actual_obj=None,
        )
    except Exception:
        new_obj = None
        improvement = 0.0
        success = False

    # report to param tuner if present (update + adjust with skip_update to avoid double counting)
    try:
        tuner = getattr(state, "param_tuner", None)
        if tuner and hasattr(tuner, "update_operator_performance"):
            try:
                tuner.update_operator_performance(op_name, improvement, success=success)
            except Exception:
                pass
            try:
                if hasattr(tuner, "adjust_operator_params"):
                    tuner.adjust_operator_params(
                        op_name, improvement, success=success, skip_update=True
                    )
            except Exception:
                pass
    except Exception:
        pass

    return new_obj, improvement, success


def build_opjson_log(
    op_name: str,
    early: bool,
    removed_cnt: int,
    veh_after: int,
    prev_obj,
    new_obj,
    improvement,
    elapsed,
    batch_conflict: bool,
    removed_cargo_summary: Dict,
    inventory_updates: Optional[Dict],
    extra: Optional[Dict],
):
    """
    Construct the standardized OPJSON log payload (as a plain dict).

    中文注释, 字段说明与序列化策略:
    - 目的:
      * 生成统一格式的 destroy 操作日志载荷, 便于集中记录、分析与后处理。
    - 字段要点:
      * removed_cargo_summary 仅保留前 200 条以控制日志体积, 避免单次日志过大。
      * inventory_updates 使用 to_native 转换以避免 numpy 数据导致序列化失败。
      * 所有易为 numpy 类型的字段 (objective_before/after/improvement/elapsed) 均通过 to_native 处理。
    - 返回:
      * 一个普通 dict, 可直接用于 logging 或 JSON 序列化。
    """
    log_entry = {
        "tag": "destroy",
        "op": op_name,
        "early": bool(early),
        "removed_cnt": int(removed_cnt),
        "veh_after": int(veh_after),
        "objective_before": to_native(prev_obj),
        "objective_after": to_native(new_obj),
        "improvement": to_native(improvement),
        "elapsed_sec": to_native(elapsed),
        "batch_update_conflict": bool(batch_conflict),
        # keep only first 200 entries for safety
        "removed_cargo_summary": {
            to_native(k): to_native(v)
            for k, v in list(removed_cargo_summary.items())[:200]
        },
        "inventory_updates": to_native(inventory_updates)
        if inventory_updates is not None
        else None,
        "extra": extra,
    }
    return log_entry


# ---------------------------------------------------------------------
# 以下为从 destroy_operators.py 提取过来的辅助函数
# ---------------------------------------------------------------------


def ensure_pyint(x):
    try:
        if isinstance(x, np.ndarray):
            arr = np.atleast_1d(x)
            return int(arr.flatten()[0])
        if isinstance(x, (np.generic,)):
            return int(x.item())
        if isinstance(x, (list, tuple)):
            return int(x[0])
        return int(x)
    except Exception:
        try:
            return int(np.array(x).flat[0])
        except Exception:
            return int(x) if isinstance(x, (int,)) else 0


def ensure_pylist(x):
    try:
        if isinstance(x, np.ndarray):
            return [int(i) for i in x.tolist()]
        if isinstance(x, (list, tuple)):
            return [int(i) for i in x]
        return [int(x)]
    except Exception:
        return []


def adjust_exploration(state: SolutionState):
    """
    记录当前算子调用发生在哪一次迭代, 以便 ParamAutoTuner 识别所处阶段:
      - exploration:  前期大范围探索, 参数调整更频繁
      - exploitation: 中期利用发现的结构/模板
      - refinement:   后期收敛细化, 参数扰动趋于收缩
    说明:
      tracker 维护全局统计(含 total_iterations / max_iterations)
      param_tuner.set_iteration 会内部调整 exploration_rate
    """
    try:
        if state.param_tuner and state.tracker:
            stats = state.tracker.get_statistics()
            current_iter = stats.get("total_iterations", 0)
            max_iter = stats.get("max_iterations", 1000)
            state.param_tuner.set_iteration(
                current_iter,
                max_iter,
                no_improve_iters=stats.get("no_improve_iters", 0),
            )
    except Exception:
        # 宽容处理，避免影响主流程
        pass


def get_adaptive_parameters(op_name: str, param_tuner):
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
        # 使用 logger.debug 替代 print，采样控制以减少热路径开销
        try:
            try:
                from .alns_config import default_config as ALNSConfig

                sampling = getattr(ALNSConfig, "OPJSON_SAMPLING_RATE", 1)
            except Exception:
                sampling = 1
            if sampling <= 1 or random.random() < 1.0 / max(1, int(sampling)):
                logger.debug(f"get_adaptive_parameters({op_name}) 异常: {e}")
        except Exception:
            pass
        return {}


def resolve_degree(op_name: str, state, degree):
    """
    统一解析与夹紧 destroy 算子的 degree 参数。
    优先级: 显式传入 degree → ParamAutoTuner.get_operator_params(op_name) → ALNSConfig 默认 → 兜底默认
    新增: 集成 adaptive_degree 功能，根据迭代进展动态调整破坏程度
    输出: float 且位于 [0,1]
    """
    # 获取集中配置默认
    try:
        from .alns_config import default_config as ALNSConfig

        op_defaults = ALNSConfig.get_operator_default(op_name).get("params", {}) or {}
    except Exception:
        op_defaults = {}

    # 兜底默认
    fallback = {
        "random_removal": 0.25,
        "worst_removal": 0.25,
        "surplus_inventory_removal": 0.25,
        "shaw_removal": 0.3,
        "periodic_shaw_removal": 0.3,
        "path_removal": 0.5,
    }
    base = op_defaults.get("degree", fallback.get(op_name, 0.25))

    if degree is None:
        params = (
            get_adaptive_parameters(op_name, getattr(state, "param_tuner", None)) or {}
        )
        raw = params.get("degree", base)
    else:
        raw = degree

    try:
        val = float(raw)
    except Exception:
        try:
            val = float(base)
        except Exception:
            val = 0.25

    # 集成 adaptive_degree 功能：如果存在 param_tuner，使用其 adaptive_degree 方法动态调整
    param_tuner = getattr(state, "param_tuner", None)
    if param_tuner and hasattr(param_tuner, "adaptive_degree"):
        try:
            adaptive_max_degree = 0.5
            if op_name in {
                "cross_period_dealer_removal",
                "period_range_removal",
                "periodic_shaw_removal",
                "path_removal",
            }:
                adaptive_max_degree = 0.75
            elif op_name in {
                "random_removal",
                "shaw_removal",
                "worst_removal",
                "surplus_inventory_removal",
                "inventory_level_destruction",
            }:
                adaptive_max_degree = 0.60

            # 使用 adaptive_degree 方法根据迭代进展调整破坏程度
            # base_degree=val 作为基础值，其他参数使用默认值
            val = param_tuner.adaptive_degree(
                base_degree=val,
                min_degree=0.05,
                max_degree=adaptive_max_degree,
                decay_rate=0.67,
            )
        except Exception as e:
            try:
                from .alns_config import default_config as ALNSConfig

                sampling = getattr(ALNSConfig, "OPJSON_SAMPLING_RATE", 1)
            except Exception:
                sampling = 1
            try:
                if sampling <= 1 or random.random() < 1.0 / max(1, int(sampling)):
                    logger.debug(f"adaptive_degree 调用失败 for {op_name}: {e}")
            except Exception:
                pass

    # clamp to [0,1] range
    if val < 0.0:
        val = 0.0
    if val > 1.0:
        val = 1.0
    return val


def post_destroy(
    state: SolutionState,
    removed,
    op_name: str,
    early: bool = False,
    prev_obj: Optional[float] = None,
    t0: Optional[float] = None,
    skip_batch_update: bool = False,
    extra: Optional[dict] = None,
):
    """
    精简版 post_destroy：协调 summarize_removed_cargo / apply_batch_update /
    compute_improvement_and_report / build_opjson_log / adjust_exploration / to_native 等工具函数。
    """
    # 1) 汇总 removed 的 cargo（用于日志摘要）
    try:
        removed_cargo_summary = summarize_removed_cargo(removed)
    except Exception:
        removed_cargo_summary = {}

    # 2) 执行批量回补（由 apply_batch_update 处理）
    inventory_updates = None
    batch_conflict = False
    try:
        inventory_updates, batch_conflict = apply_batch_update(
            state, removed, skip_batch_update
        )
    except Exception:
        inventory_updates = None
        batch_conflict = True

    # 2.1) 应用算子传入的额外库存增量(例如部分回退), 与批量回补合并
    partial_updates = None
    if isinstance(extra, dict):
        partial_updates = extra.get("partial_updates")
    if partial_updates:
        try:
            partial_applied = apply_partial_inventory_updates(state, partial_updates)
            if partial_applied:
                if inventory_updates is None:
                    inventory_updates = {}
                for k, v in partial_applied.items():
                    inventory_updates[k] = inventory_updates.get(k, 0) + v
        except Exception:
            batch_conflict = True

    # 3) 计算耗时
    elapsed = None
    if t0 is not None:
        try:
            elapsed = time.time() - t0
        except Exception:
            elapsed = None

    # 4) 早退逻辑：记录阶段 & 日志
    if early:
        try:
            adjust_exploration(state)
        except Exception:
            pass
        log_entry = {
            "tag": "destroy",
            "op": op_name,
            "early": True,
            "removed_cnt": int(len(removed) if removed else 0),
            "veh_after": int(len(state.vehicles)),
            "elapsed_sec": to_native(elapsed),
            "batch_update_conflict": bool(batch_conflict),
            "removed_cargo_summary": {
                k: to_native(v) for k, v in list(removed_cargo_summary.items())[:200]
            },
            "extra": extra,
        }
        try:
            try:
                from .alns_config import default_config as ALNSConfig

                sampling = getattr(ALNSConfig, "OPJSON_SAMPLING_RATE", 1)
            except Exception:
                sampling = 1
            if sampling <= 1 or random.random() < 1.0 / max(1, int(sampling)):
                logger.debug(f"[OPJSON] {log_entry}")
        except Exception:
            pass
        return

    # 5) 正常路径：计算 improvement 并上报 param_tuner（由 compute_improvement_and_report 处理）
    new_obj = None
    improvement = 0.0
    try:
        new_obj, improvement, _ = compute_improvement_and_report(
            state, prev_obj, op_name
        )
    except Exception:
        new_obj = None
        improvement = 0.0

    # 6) 阶段标记与最终日志
    try:
        adjust_exploration(state)
    except Exception:
        pass

    try:
        log_entry = build_opjson_log(
            op_name=op_name,
            early=False,
            removed_cnt=len(removed) if removed else 0,
            veh_after=len(state.vehicles),
            prev_obj=prev_obj,
            new_obj=new_obj,
            improvement=improvement,
            elapsed=elapsed,
            batch_conflict=batch_conflict,
            removed_cargo_summary=removed_cargo_summary,
            inventory_updates=inventory_updates,
            extra=extra,
        )
    except Exception:
        # fallback simple log
        try:
            log_entry = {
                "tag": "destroy",
                "op": op_name,
                "early": False,
                "removed_cnt": int(len(removed) if removed else 0),
                "veh_after": int(len(state.vehicles)),
                "objective_before": to_native(prev_obj),
                "objective_after": to_native(new_obj),
                "improvement": to_native(improvement),
                "elapsed_sec": to_native(elapsed),
                "batch_update_conflict": bool(batch_conflict),
                "removed_cargo_summary": {
                    k: to_native(v)
                    for k, v in list(removed_cargo_summary.items())[:200]
                },
                "inventory_updates": to_native(inventory_updates)
                if inventory_updates is not None
                else None,
                "extra": extra,
            }
        except Exception:
            log_entry = {"tag": "destroy", "op": op_name, "error": "failed_build_log"}

    try:
        try:
            from .alns_config import default_config as ALNSConfig

            sampling = getattr(ALNSConfig, "OPJSON_SAMPLING_RATE", 1)
        except Exception:
            sampling = 1
        if sampling <= 1 or random.random() < 1.0 / max(1, int(sampling)):
            logger.debug(f"[OPJSON] {log_entry}")
    except Exception:
        pass
