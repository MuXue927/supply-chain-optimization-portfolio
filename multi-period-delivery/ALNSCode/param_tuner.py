"""
模块: param_tuner
职责:
1. 定义可调算子参数数据结构 OperatorParams:
   - 记录算子当前参数(params)、可调范围(ranges)、历史改进值(performance_history)、使用/成功次数
   - 提供平均性能与成功率评估接口 (用于自适应选择/加权)
2. 定义参数自动调整器 ParamAutoTuner:
   - 按迭代进度划分阶段: exploration / exploitation / refinement
   - 基于探索率 exploration_rate 在「随机探索」与「利用当前参数」之间切换
   - 根据算子最近性能改进对参数进行方向性微调 (adjust_operator_params)
   - 提供获取特定类型算子最佳候选 get_best_operators
   - 支持动态调整 destroy 算子破坏强度 adaptive_degree

核心设计思想:
- 分离“参数抽样(get_operator_params)”与“学习反馈(update_operator_performance / adjust_operator_params)”
  使其可在外部 ALNS 流程中灵活调用。
- 将 performance_history 设定上限(100) 防止无限增长影响内存与平均值反应速度。
- 阶段控制:
    progress = current_iter / max_iter
    - 前 30%: exploration 高 → 逐步降低 (0.50 → 0.35)
    - 中 40%: exploitation 平衡探索与利用 (0.35 → 0.15)
    - 末 30%: refinement 精细调参 (0.15 → 0.0)
  末期仍可能出现浮点舍入导致 exploration_rate < 0，故需显式截断。

库存 / 车辆上下文无直接依赖:
- 本模块仅依赖 DataALNS 读取问题规模与数据特征 (为潜在特征驱动策略保留入口)。

潜在改进方向(未实现，仅注释):
- 引入「参数景观」记忆: 针对某算子不同参数组合的统计(离散采样)以加速后期收敛。
- 对 improvement 采用加权滑动平均 (指数衰减) 而非截断窗口，增强“惯性”表达。
- 针对多参数联动算子 (如 periodic_shaw_removal) 扩展为协方差自适应更新(CMA-like 简化版)。
"""

# =========================
# 标准库
# =========================
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

# =========================
# 第三方库
# =========================
import numpy as np
import numpy.random as rnd

# =========================
# 项目内部依赖
# =========================
from .InputDataALNS import DataALNS
from .alns_config import default_config as ALNSConfig

# ---------------------------------------------------------------------
# 数据结构: 单个算子的参数容器
# ---------------------------------------------------------------------
@dataclass
class OperatorParams:
    """
    算子参数描述:
    name:   算子唯一名称 (例如 'random_removal')
    params: 当前采用的基准参数 (在 exploitation 阶段直接使用)
    ranges: 各参数允许的连续区间 (用于探索阶段随机扰动与方向性微调边界)
    performance_history: 最近使用时的性能改进值列表 (improvement = obj_before - obj_after)
    usage_count / success_count: 用于成功率计算 (用于 get_best_operators 时的估值)
    """
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    performance_history: List[float] = field(default_factory=list)
    usage_count: int = 0
    success_count: int = 0

    def update_performance(self, improvement: float, success: bool = True):
        """
        记录一次使用后的性能表现:
        - improvement: 正值表示改进(下降为好/基于目标值缩放)
        - success: 是否计为一次“成功”提升 (允许外部在异常/不可行时标记为 False)
        - 限制 performance_history 长度到 100，加速近期表现响应
        """
        self.performance_history.append(improvement)
        self.usage_count += 1
        if success:
            self.success_count += 1
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]

    def get_avg_performance(self, window: int = 20) -> float:
        """
        计算最近 window 次的平均改进 (窗口 > 实际记录长度时取全部)。
        返回值为 0 表示没有历史或均值为零改进。
        """
        if not self.performance_history:
            return 0.0
        recent = self.performance_history[-min(window, len(self.performance_history)):]
        return sum(recent) / len(recent) if recent else 0.0

    def get_success_rate(self) -> float:
        """
        成功率 = success_count / usage_count
        使用 max(1, usage_count) 防止除零。
        """
        return self.success_count / max(1, self.usage_count)

# ---------------------------------------------------------------------
# 参数自动调节器
# ---------------------------------------------------------------------
class ParamAutoTuner:
    """
    功能概述:
    - 动态阶段控制 (exploration / exploitation / refinement)
    - 根据 exploration_rate 决定是否对算子参数进行随机扰动
    - 记录算子使用表现并可基于方向性变化做微调 (adjust_operator_params)
    - 支持查询特定类型(destroy/repair 等命名前缀)表现最佳的前 K 个算子

    使用流程建议:
    1. 初始化: tuner = ParamAutoTuner(data, rng)
    2. 每次迭代开始: tuner.set_iteration(cur_iter, max_iter)
    3. 算子选择阶段:
         params = tuner.get_operator_params(op_name)
         (外部据此调用对应算子)
    4. 评估后:
         tuner.update_operator_performance(op_name, improvement, success)
         tuner.adjust_operator_params(op_name, improvement, success)
    5. 需要时:
         best_ops = tuner.get_best_operators('removal', top_k=3)

    设计注意:
    - get_operator_params 中 exploration 随机采样不会回写到 op.params
      (op.params 代表“当前基准配置”，仅在 adjust_operator_params 中可能被方向性更新)
    - ranges 为空表示该算子不支持参数调节 (直接返回 {})
    """

    operators: Dict[str, OperatorParams]
    _prev_params: Dict[str, Dict[str, Any]]

    def __init__(self, data: DataALNS, rng: rnd.Generator):
        self.data = data
        self.rng = rng

        # 已注册的算子参数
        self.operators: Dict[str, OperatorParams] = {}

        # 问题实例特征，为更高级策略预留 (当前逻辑未直接使用)
        self.instance_features = self._extract_instance_features()
        self.instance_scale = self._estimate_instance_scale()
        self.small_instance = self.instance_scale <= int(
            getattr(ALNSConfig, "TUNER_SMALL_INSTANCE_SCALE_THRESHOLD", 250)
        )

        # 迭代与阶段
        self.iteration = 0
        self.max_iterations = 1000
        self.phase = "exploration"
        self.no_improve_iters = 0

        # 控制参数 (可由集中配置覆盖)
        self.exploration_rate = getattr(ALNSConfig, "TUNER_EXPLORATION_INITIAL", 0.3)
        self.learning_rate = getattr(ALNSConfig, "TUNER_LEARNING_RATE", 0.1)

        # 历史参数快照(用于计算方向 direction = value - prev_value)
        self._prev_params: Dict[str, Dict[str, Any]] = {}
        # 最近一次用于执行的参数快照（由 get_operator_params 记录），用于计算 used - baseline 的方向
        self._last_used: Dict[str, Dict[str, Any]] = {}

        # 初始化默认算子
        self._init_default_operators()

    # -----------------------------------------------------------------
    # 实例特征提取
    # -----------------------------------------------------------------
    def _canonical_name(self, name: str) -> str:
        """
        规范化算子名称，兼容 wrapper 名称（例如 'random_removal_deg_0.25' → 'random_removal'）
        仅在简单模式下去掉 '_deg_' 后缀，必要时可扩展更多规则/映射。
        """
        try:
            if isinstance(name, str) and "_deg_" in name:
                return name.split("_deg_")[0]
            return name
        except Exception:
            return name

    def _extract_instance_features(self) -> Dict[str, float]:
        """
        提取问题规模与结构性统计特征 (供未来 ML/自适应高阶策略使用)。
        当前仅计算基础聚合值，不参与核心参数调节过程。
        """
        data = self.data
        features: Dict[str, float] = {}

        # 规模类
        features["num_plants"] = len(data.plants)
        features["num_dealers"] = len(data.dealers)
        features["num_skus"] = len(data.all_skus)
        features["num_periods"] = data.horizons
        features["num_veh_types"] = len(data.all_veh_types)

        # 需求统计
        if data.demands:
            demands = list(data.demands.values())
            features["avg_demand"] = sum(demands) / len(demands)
            features["max_demand"] = max(demands)
            features["min_demand"] = min(d for d in demands if d > 0)
            features["demand_std"] = np.std(demands) if len(demands) > 1 else 0.0
        else:
            features.update({
                "avg_demand": 0.0, "max_demand": 0.0,
                "min_demand": 0.0, "demand_std": 0.0
            })

        # 期初库存
        if data.historical_s_ikt:
            init_inv = [inv for (_, _, day), inv in data.historical_s_ikt.items() if day == 0]
            features["avg_init_inv"] = sum(init_inv) / len(init_inv) if init_inv else 0.0
        else:
            features["avg_init_inv"] = 0.0

        # 生产统计
        if data.sku_prod_each_day:
            productions = list(data.sku_prod_each_day.values())
            features["avg_production"] = sum(productions) / len(productions) if productions else 0.0
        else:
            features["avg_production"] = 0.0

        # 车辆容量统计
        if data.veh_type_cap:
            capacities = list(data.veh_type_cap.values())
            features["avg_veh_cap"] = sum(capacities) / len(capacities)
            features["max_veh_cap"] = max(capacities)
            features["min_veh_cap"] = min(capacities)
        else:
            features["avg_veh_cap"] = 0.0
            features["max_veh_cap"] = 0.0
            features["min_veh_cap"] = 0.0

        # 需求与期初库存比
        total_demand = sum(data.demands.values()) if data.demands else 0
        total_init_inv = sum(inv for (_, _, day), inv in data.historical_s_ikt.items()
                             if day == 0) if data.historical_s_ikt else 0
        features["demand_inv_ratio"] = total_demand / max(1, total_init_inv)

        return features

    def _estimate_instance_scale(self) -> int:
        """Estimate a coarse instance scale for conservative tuning on small datasets."""
        try:
            num_dealers = int(self.instance_features.get("num_dealers", 0))
            num_skus = int(self.instance_features.get("num_skus", 0))
            num_periods = int(self.instance_features.get("num_periods", 0))
            return max(1, num_dealers * max(1, num_skus) * max(1, num_periods))
        except Exception:
            return 1

    # -----------------------------------------------------------------
    # 初始化默认算子注册
    # -----------------------------------------------------------------
    def _init_default_operators(self):
        """
        注册一组默认 destroy / repair 类算子的初始参数与合法范围。
        优先从集中配置模块 ALNSCode.alns_config.default_config 获取 OPERATOR_DEFAULTS，
        若获取失败则回退到内嵌的硬编码默认值（保证向后兼容）。
        ranges 为空表示该算子当前不做参数扰动 (保持接口一致性)。
        """
        try:
            # 局部导入以避免可能的循环导入问题
            from .alns_config import default_config
            op_defaults = default_config.get_tuner_operator_defaults()
        except Exception:
            op_defaults = None

        if op_defaults:
            # op_defaults expected shape: {op_name: {"params": {...}, "ranges": {...}}, ...}
            for name, struct in op_defaults.items():
                try:
                    params = struct.get("params", {}) or {}
                    ranges = struct.get("ranges", {}) or {}
                    self.register_operator(name, params, ranges)
                except Exception:
                    # 单个算子注册失败不影响其他算子
                    continue
            return

    # -----------------------------------------------------------------
    # 外部接口: 注册 / 更新
    # -----------------------------------------------------------------
    def register_operator(self, name: str,
                          params: Dict[str, Any],
                          ranges: Dict[str, Tuple[float, float]]):
        """
        注册算子:
        - params: 基准参数 (exploitation 阶段直接使用)
        - ranges: 连续/整数域上下界 (探索时随机采样; 调参时裁剪)
        """
        # name = self._canonical_name(name)
        self.operators[name] = OperatorParams(name=name, params=params, ranges=ranges)

    def update_operator_performance(self, name: str,
                                    improvement: float,
                                    success: bool = True):
        """
        外部在算子执行 + 目标值计算后调用:
        - improvement: 正值表示目标函数改进
        - success: 表示是否计入成功次数 (不可行 / 异常可置 False)
        """
        # name = self._canonical_name(name)
        if name in self.operators:
            self.operators[name].update_performance(improvement, success)

    # -----------------------------------------------------------------
    # 迭代阶段控制
    # -----------------------------------------------------------------
    def set_iteration(
        self,
        current: int,
        maximum: int,
        no_improve_iters: Optional[int] = None,
    ):
        """
        设置迭代进度并刷新阶段与探索率:
        - exploration (0%~30%): 0.50 → 0.35
        - exploitation (30%~70%): 0.35 → 0.15
        - refinement (70%~100%): 0.15 → 0.00
        使用线性分段函数，可后续替换为非线性衰减(例如指数)。
        """
        self.iteration = current
        self.max_iterations = maximum
        if no_improve_iters is not None:
            try:
                self.no_improve_iters = max(0, int(no_improve_iters))
            except Exception:
                self.no_improve_iters = 0

        progress = current / maximum if maximum > 0 else 1.0
        # 阶段阈值由集中配置提供，保持向后兼容默认 (0.3, 0.7)
        thresh_low, thresh_high = getattr(ALNSConfig, "TUNER_PHASE_THRESHOLDS", (0.3, 0.7))
        if progress < thresh_low:
            self.phase = "exploration"
            # 保持原有线性衰减形状，但允许阈值可配置
            self.exploration_rate = 0.5 - progress * 0.5
        elif progress < thresh_high:
            self.phase = "exploitation"
            self.exploration_rate = 0.35 - (progress - thresh_low) * 0.5
        else:
            self.phase = "refinement"
            self.exploration_rate = 0.15 - (progress - thresh_high) * 0.5

        # 数值稳定: 防止浮点误差略为负值
        if self.exploration_rate < 0:
            self.exploration_rate = 0.0
        if self.exploration_rate > 1:
            self.exploration_rate = 1.0

        if self.small_instance:
            self.exploration_rate = min(
                self.exploration_rate,
                float(getattr(ALNSConfig, "TUNER_SMALL_INSTANCE_EXPLORATION_CAP", 0.18)),
            )

    # -----------------------------------------------------------------
    # 参数获取 (探索 vs 利用)
    # -----------------------------------------------------------------
    def get_operator_params(self, name: str) -> Dict[str, Any]:
        """
        获取算子当前使用参数，并记录“本轮用于执行的参数快照”到 self._last_used[name]。
        说明：探索阶段的随机样本不回写到基线，仅用于本轮执行与方向性微调的参考。
        """
        # name = self._canonical_name(name)
        ret: Dict[str, Any] = {}
        if name not in self.operators:
            print(f"[WARNING] 未注册的算子: {name}, 使用默认参数")
            self._last_used[name] = {}
            return ret

        op = self.operators[name]

        if not op.params or not op.ranges:
            # 无可调参数或未配置 ranges 时，返回空并记录
            self._last_used[name] = {}
            return ret

        # 探索: 随机扰动（不回写 op.params）
        if self.rng.random() < self.exploration_rate:
            sampled = op.params.copy()
            sample_scale = 1.0
            if self.small_instance:
                sample_scale = float(
                    getattr(ALNSConfig, "TUNER_SMALL_INSTANCE_SAMPLE_SCALE", 0.35)
                )
            for param_name, (min_val, max_val) in op.ranges.items():
                base_val = sampled.get(param_name)
                if isinstance(base_val, int):
                    min_int = int(min_val)
                    max_int = int(max_val)
                    center = int(base_val) if base_val is not None else (min_int + max_int) // 2
                    half_width = max(1, int(round((max_int - min_int) * sample_scale * 0.5)))
                    low = max(min_int, center - half_width)
                    high = min(max_int, center + half_width)
                    if high < low:
                        low, high = min_int, max_int
                    sampled[param_name] = int(self.rng.integers(low, high + 1))
                else:
                    try:
                        center = float(base_val) if base_val is not None else (float(min_val) + float(max_val)) / 2.0
                    except Exception:
                        center = (float(min_val) + float(max_val)) / 2.0
                    half_width = max(1e-12, (float(max_val) - float(min_val)) * sample_scale * 0.5)
                    low = max(float(min_val), center - half_width)
                    high = min(float(max_val), center + half_width)
                    if high <= low:
                        low, high = float(min_val), float(max_val)
                    sampled[param_name] = float(self.rng.uniform(low, high))
            self._last_used[name] = sampled.copy()
            return sampled

        # 利用: 使用当前基线
        ret = op.params.copy()
        self._last_used[name] = ret.copy()
        return ret

    # -----------------------------------------------------------------
    # 参数微调 (基于改进方向)
    # -----------------------------------------------------------------
    def adjust_operator_params(self, name: str, improvement: float, success: bool = True, skip_update: bool = False):
        """
        根据一次成功改进对算子基准参数进行方向性微调 (可选跳过性能记录)。
        调整方向基于“本轮实际使用的参数”(self._last_used[name]) 与当前基线 op.params 的差异。
        其它说明同原实现。
        """
        # name = self._canonical_name(name)
        if name not in self.operators:
            return

        op = self.operators[name]
        if not skip_update:
            op.update_performance(improvement, success)

        if not success or improvement <= 0:
            return

        history_len = len(op.performance_history)
        min_history = int(getattr(ALNSConfig, "TUNER_MIN_HISTORY_FOR_ADJUST", 5))
        if history_len < min_history:
            return

        recent_window = min(20, history_len)
        recent = op.performance_history[-recent_window:]
        avg_perf = op.get_avg_performance(window=recent_window)
        if avg_perf <= 0 and history_len < min_history * 2:
            return

        confidence = min(1.0, history_len / max(1, min_history * 4))
        if self.small_instance:
            confidence *= 0.75

        required_gain = max(0.0, avg_perf)
        if required_gain > 0.0:
            if improvement <= required_gain * (1.0 + 0.15 * (1.0 - confidence)):
                return

        effective_lr = self.learning_rate * confidence
        max_rel_step = float(getattr(ALNSConfig, "TUNER_MAX_REL_STEP", 0.15))
        if self.small_instance:
            max_rel_step *= 0.75

        used_params = self._last_used.get(name, None)
        for param_name, value in op.params.items():
            if param_name not in op.ranges:
                continue
            min_val, max_val = op.ranges[param_name]

            used_val = (
                used_params.get(param_name, value)
                if isinstance(used_params, dict)
                else value
            )
            try:
                used_val_f = float(used_val)
                value_f = float(value)
            except Exception:
                continue

            direction = used_val_f - value_f
            if abs(direction) <= 1e-12:
                continue

            denom = avg_perf if avg_perf > 1e-12 else 1e-12
            adjustment = effective_lr * (improvement / denom) * direction
            new_value = value_f + adjustment

            step_limit = max(
                1e-12,
                (float(max_val) - float(min_val)) * max_rel_step * max(0.25, confidence),
            )
            lower_bound = value_f - step_limit
            upper_bound = value_f + step_limit
            new_value = min(upper_bound, max(lower_bound, new_value))
            new_value = max(float(min_val), min(float(max_val), new_value))

            if isinstance(value, int):
                new_value = int(max(min_val, min(max_val, round(new_value))))
            else:
                new_value = float(new_value)

            if new_value == value:
                continue

            op.params[param_name] = new_value

            try:
                if getattr(ALNSConfig, "TUNER_DEBUG", False):
                    print(
                        f"[TUNER] {name}.{param_name}: {value} -> {new_value} "
                        f"(used={used_val}, imp={improvement:.4f}, avg={avg_perf:.4f}, conf={confidence:.2f})"
                    )
            except Exception:
                pass

        # 保存当前参数用于后续参考
        if name not in self._prev_params:
            self._prev_params[name] = {}
        self._prev_params[name] = op.params.copy()

    # -----------------------------------------------------------------
    # 算子筛选 (Top-K)
    # -----------------------------------------------------------------
    def get_best_operators(self, operator_type: str, top_k: int = 3) -> List[str]:
        """
        获取名称中包含 operator_type (例如 'removal'/'repair') 的表现最佳前 top_k 个算子:
        评分 = max(0, avg_performance) * success_rate
        说明:
          - 若 avg_performance 为负(说明总体劣化), 视为 0 以避免“负负得正”或
            因绝对值较小(靠近0)被错误选入候选。
        仅统计 usage_count > 0 的算子
        """
        min_history = int(getattr(ALNSConfig, "TUNER_MIN_HISTORY_FOR_ADJUST", 5))
        candidates = []
        for name, op in self.operators.items():
            if operator_type in name and op.usage_count > 0:
                if op.usage_count < min_history:
                    continue
                recent_window = min(20, len(op.performance_history))
                recent = op.performance_history[-recent_window:]
                avg_perf = op.get_avg_performance(window=recent_window)
                success_rate = op.get_success_rate()
                variance_penalty = float(np.std(recent)) if len(recent) > 1 else 0.0
                confidence = min(1.0, op.usage_count / max(1, min_history * 4))
                history_bonus = np.log1p(op.usage_count)
                adj_avg = max(0.0, avg_perf)
                score = (
                    adj_avg
                    * (0.5 + 0.5 * success_rate)
                    * confidence
                    * history_bonus
                    / (1.0 + variance_penalty)
                )
                candidates.append((name, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in candidates[:top_k]]

    # -----------------------------------------------------------------
    # 动态破坏程度调整
    # -----------------------------------------------------------------
    def adaptive_degree(self,
                        base_degree: float,
                        min_degree: float = 0.05,
                        max_degree: float = 0.5,
                        decay_rate: float = 0.67) -> float:
        """
        根据迭代进展动态收缩破坏比例:
        - progress 定义在 [0, 0.7] 区间内归一化 (超过后固定)
        - adaptive_value = base_degree * (1 - decay_rate * progress)
        - 最终裁剪到 [min_degree, max_degree]
        用途:
          - 在 destroy 阶段控制扰动强度，避免后期解结构被大幅破坏
        """
        if self.max_iterations <= 0 or self.iteration <= 0:
            return base_degree

        progress = min(1.0, self.iteration / (0.7 * self.max_iterations))
        adaptive_value = base_degree * (1 - decay_rate * progress)

        stagnation_iters = max(0, int(getattr(self, "no_improve_iters", 0)))
        stagnation_threshold = int(
            getattr(ALNSConfig, "TUNER_STAGNATION_BOOST_THRESHOLD", 35)
        )
        if stagnation_iters >= stagnation_threshold:
            boost_factor = float(
                getattr(ALNSConfig, "TUNER_STAGNATION_BOOST_FACTOR", 0.35)
            )
            boost_scale = min(
                boost_factor,
                ((stagnation_iters - stagnation_threshold) / max(1, stagnation_threshold))
                * boost_factor,
            )
            adaptive_value *= 1.0 + boost_scale

        if self.small_instance:
            adaptive_value *= 1.0 + min(0.15, stagnation_iters / max(1, stagnation_threshold * 4) * 0.15)

        return max(min_degree, min(max_degree, adaptive_value))
