"""
ml_destroy_selection
====================

模块定位
    基于机器学习 (MLOperatorSelector) 的“自适应破坏算子选择”入口封装。
    通过读取当前解的特征 + 历史 (特征, 算子, 改进值) 样本，预测各破坏算子的期望改进，
    自动挑选合适的破坏算子执行；当训练数据不足时回退到随机选择策略。

核心流程
    1. 复制当前解 (保持函数纯度，外部原状态不被直接修改)。
    2. 若未绑定 ML 选择器 (state.ml_selector 为 None) → 回退 random_removal。
    3. 若 destroy 特征样本数 < min_sample_size → 随机算子 + 记录 (特征, 改进)。
    4. 否则：
         a. 调用 ml_selector.train_models() （内部按间隔、样本阈值决定是否训练）
         b. select_best_operator(..., op_type='destroy')
         c. 执行所选算子并评估改进
         d. 记录改进样本 + 调整算子参数 (ParamAutoTuner 存在时)
    5. 返回更新后的新解副本

算子集 (保持与 MLOperatorSelector.destroy_op_map 一致)
    - random_removal
    - shaw_removal
    - periodic_shaw_removal
    - path_removal
    - worst_removal
    - infeasible_removal
    - surplus_inventory_removal

命名说明
    函数名为：
        select_and_apply_destroy_operator
    以突出：函数既“选择”又“执行”破坏算子。

性能与安全
    - 仅对 state.copy() 副本进行操作
    - objective() 调用依赖缓存与增量库存保持高效
    - 日志轻量，可按需后续接入全局日志开关

后续可选增强
    - 引入算子失败/跳过的异常捕获与退化策略
    - 将 min_sample_size 外化为配置 (ALNSConfig) 便于不同数据规模调参
    - 记录算子执行耗时并纳入特征，建立“成本感知”模型
"""

# =========================
# 标准库
# =========================
import time
import math
from typing import Dict, List, Any, Optional

# =========================
# 第三方库
# =========================
import numpy as np
import numpy.random as rnd
from collections import (
    defaultdict,
)  # （当前文件未直接使用，但可能在扩展中需要，可保留或后续清理）

# =========================
# 项目内部
# =========================
from .alnsopt import SolutionState
from .ml_operator_selector import MLOperatorSelector
from .alns_config import default_config as ALNSConfig
from .repair_utils import compute_mixed_operator_reward

# 破坏算子集合
from .destroy_operators import (
    random_removal,
    worst_removal,
    infeasible_removal,
    surplus_inventory_removal,
    shaw_removal,
    periodic_shaw_removal,
    path_removal,
    cross_period_dealer_removal,
    period_range_removal,
    inventory_level_destruction,
)

# ---------------------------------------------------------------------
# 内部：算子映射构造（lambda 包装保持统一调用签名）
# ---------------------------------------------------------------------
_DESTROY_OPERATORS: Dict[str, callable] = {
    "random_removal": lambda s, r, **kw: random_removal(s, r, **kw),
    "shaw_removal": lambda s, r, **kw: shaw_removal(s, r, **kw),
    "periodic_shaw_removal": lambda s, r, **kw: periodic_shaw_removal(s, r, **kw),
    "path_removal": lambda s, r, **kw: path_removal(s, r, **kw),
    "cross_period_dealer_removal": lambda s, r, **kw: cross_period_dealer_removal(
        s, r, **kw
    ),
    "period_range_removal": lambda s, r, **kw: period_range_removal(s, r, **kw),
    "inventory_level_destruction": lambda s, r, **kw: inventory_level_destruction(
        s, r, **kw
    ),
    "worst_removal": lambda s, r, **kw: worst_removal(s, r, **kw),
    "infeasible_removal": lambda s, r, **kw: infeasible_removal(s, r, **kw),
    "surplus_inventory_removal": lambda s, r, **kw: surplus_inventory_removal(
        s, r, **kw
    ),
}


# Helper: sanitize and align ParamTuner kwargs to operator signatures
def _sanitize_destroy_params(op_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(params, dict) or not params:
        return {}
    allowed = {
        "random_removal": {"degree"},
        "shaw_removal": {"degree"},
        "periodic_shaw_removal": {"degree", "alpha", "beta", "gamma", "k_cluster"},
        "path_removal": {"degree"},
        "cross_period_dealer_removal": {"degree"},
        "period_range_removal": {"degree", "window_ratio"},
        "inventory_level_destruction": {"degree"},
        "worst_removal": {"degree", "value_bias"},
        "infeasible_removal": set(),
        "surplus_inventory_removal": {"degree"},
    }
    p = dict(params)
    if op_name == "periodic_shaw_removal":
        if "k_cluster" not in p and "k_clusters" in p:
            p["k_cluster"] = p.pop("k_clusters")
        else:
            p.pop("k_clusters", None)
    allow = allowed.get(op_name, set())
    return {k: v for k, v in p.items() if k in allow}


# ---------------------------------------------------------------------
# 公共函数：自适应破坏算子选择与执行
# ---------------------------------------------------------------------
def ml_based_destroy(
    current: SolutionState,
    rng: rnd.Generator,
    min_sample_size: Optional[int] = None,
) -> SolutionState:
    """
    自适应（ML + 回退策略）破坏算子选择与执行。

    参数:
        current          SolutionState  当前解（不在原对象上原地修改）
        rng              numpy.random.Generator 随机源
        min_sample_size  int  destroy 模型最小训练样本阈值，低于则随机探索

    返回:
        new_state        SolutionState  应用破坏算子后的新解副本

    流程:
        1. 若未配置 ML 选择器 → random_removal
        2. destroy 样本 < min_sample_size → 随机探索 + 记录样本
        3. 否则使用 ML 模型预测并 ε-贪心选择算子
        4. 执行算子 → 计算改进值 → 记录样本 → 调整参数

    改进值定义:
        improvement = prev_obj - new_obj
        >0 代表目标下降（改进），<=0 代表未改进或恶化

    数值稳定:
        - NaN → 0
        - +inf → 1e6
        - -inf → -1e6
    """
    t_start = time.time()
    state = current.copy()

    print("[OPLOG][Destroy-ML] 开始执行 adaptive destroy 选择")
    ml_selector: Optional[MLOperatorSelector] = state.ml_selector  # 可能为 None

    # 若未显式传入 min_sample_size，则从集中配置中读取默认值（向后兼容）
    if min_sample_size is None:
        min_samples_int = int(getattr(ALNSConfig, "ML_INITIAL_SAMPLE_SIZE", 20))
    else:
        min_samples_int = int(min_sample_size)

    takeover_min = int(
        getattr(ALNSConfig, "ML_TAKEOVER_MIN_SAMPLE_SIZE", min_samples_int)
    )
    if state.param_tuner is not None and bool(
        getattr(state.param_tuner, "small_instance", False)
    ):
        takeover_min = max(
            takeover_min,
            int(getattr(ALNSConfig, "ML_TAKEOVER_MIN_SAMPLE_SIZE_SMALL", takeover_min)),
        )

    effective_min_sample_size = max(min_samples_int, takeover_min)

    # 回退：无 ML 选择器
    if ml_selector is None:
        print("[OPLOG][Destroy-ML] 未绑定 ML 选择器 → 回退 random_removal")
        return random_removal(state, rng)

    # 回退：样本不足
    use_ml = False
    if hasattr(ml_selector, "should_use_ml"):
        use_ml = bool(
            ml_selector.should_use_ml(
                state,
                "destroy",
                min_sample_size=effective_min_sample_size,
            )
        )

    if not use_ml:
        op_name = rng.choice(list(_DESTROY_OPERATORS.keys()))
        params = (
            state.param_tuner.get_operator_params(op_name) if state.param_tuner else {}
        )
        params = _sanitize_destroy_params(op_name, params)
        prev_obj = state.objective()
        new_state = _DESTROY_OPERATORS[op_name](state, rng, **(params or {}))
        new_obj, improvement, _ = compute_mixed_operator_reward(
            new_state,
            prev_obj,
            new_obj=new_state.objective(),
        )
        if new_obj is None or math.isnan(improvement):
            improvement = 0.0
        elif math.isinf(improvement):
            improvement = 1e6 if improvement > 0 else -1e6

        ml_selector.record_operator_performance(op_name, "destroy", state, improvement)
        if hasattr(ml_selector, "update_op_feedback"):
            ml_selector.update_op_feedback("destroy", improvement)

        print(
            f"[OPLOG][Destroy-ML] 使用随机回退(samples={len(ml_selector.destroy_features)}/{effective_min_sample_size}) "
            f"随机选择 {op_name} 改进 {improvement:.4f} 用时 {time.time() - t_start:.4f}s"
        )
        return new_state

    # 模型端：尝试训练（内部有节流逻辑，满足间隔与样本才实际训练）
    ml_selector.train_models()

    # ML 选择
    op_name, params = ml_selector.select_best_operator(
        state,
        op_type="destroy",
        candidates=list(_DESTROY_OPERATORS.keys()),
    )
    params = _sanitize_destroy_params(op_name, params)

    prev_obj = state.objective()
    new_state = _DESTROY_OPERATORS[op_name](state, rng, **(params or {}))
    new_obj, improvement, _ = compute_mixed_operator_reward(
        new_state,
        prev_obj,
        new_obj=new_state.objective(),
    )
    if new_obj is None or math.isnan(improvement):
        improvement = 0.0
    elif math.isinf(improvement):
        improvement = 1e6 if improvement > 0 else -1e6

    ml_selector.record_operator_performance(op_name, "destroy", state, improvement)
    if hasattr(ml_selector, "update_op_feedback"):
        ml_selector.update_op_feedback("destroy", improvement)

    print(
        f"[OPLOG][Destroy-ML] 选择 {op_name} 改进 {improvement:.4f} 用时 {time.time() - t_start:.4f}s"
    )
    return new_state
