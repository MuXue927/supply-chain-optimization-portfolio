"""
ml_operator_selector
====================

模块定位
    为 ALNS 框架提供一个“机器学习增强的算子选择器”(MLOperatorSelector)，
    通过对历史 (状态特征, 算子, 性能提升) 样本进行监督学习，预测当前状态下
    不同 destroy / repair 算子的期望改进效果，从而以数据驱动方式替换纯粹的
    轮盘赌 / 评分衰减策略，实现更智能的算子调度。

设计核心
    1. 特征抽取 (extract_state_features):
         将当前解 (SolutionState) 转换为一个数值特征向量，覆盖：
           - 目标函数值 (已通过 validate 保证库存一致性)
           - 车辆规模/平均利用率/类型分布
           - 库存总体水平/负库存比例/库存容量利用率
           - 需求满足率 / 未满足需求条目比例
           - 搜索进度特征 (迭代进度/近期改进指标，如果 tracker 提供)
    2. 样本记录 (record_operator_performance):
         在一次 destroy+repair 迭代完成后（或任一算子执行后）调用，
         将 (状态特征 + 算子 one-hot) → 实际改进值 (improvement) 存入缓冲。
    3. 模型训练 (train_models):
         当样本数量达到 min_samples 且到达 retrain_interval 触发点，
         使用 StandardScaler 标准化，再依据样本规模选用:
           - 样本 < 50: Ridge (线性回归) 以防过拟合
           - 样本 ≥ 50: RandomForestRegressor (集成非线性)
         破坏和修复算子分别独立建模。
    4. 预测选择 (select_best_operator):
         若模型未就绪 → 随机选候选(保留探索随机性)；
         若模型已就绪 → 针对每个候选拼接 (状态特征+op one-hot)，
         经 scaler 变换并预测性能，按预测值降序排序，然后用 ε-贪心策略：
              ε 概率随机探索 (默认 0.1)
              1-ε 概率选择预测值最高的算子。
    5. 参数获取:
         通过外部 ParamAutoTuner.get_operator_params(op_name) 获取该算子的动态参数配置。

重要说明 / 约束
    - 不直接修改 state, 只读特征, 安全。
    - state.objective() 会触发 validate() → compute_inventory()，保证库存相关特征一致性。
    - 若 tracker 未提供指定统计键 (max_iterations / recent_improvement)，使用默认值 0。
    - 模型训练迭代 last_train_iter 仅在 train_models 调用尾部更新：
         可能出现：样本不足仍然“推迟”训练下一次触发；此为设计简化策略，可在未来改进。
    - 算子 one-hot 编码依赖 destroy_op_map / repair_op_map 初始定义。未注册算子会生成全 0 向量，不抛出异常。

潜在改进建议 (尚未实现，仅注释提示)
    - 针对 destroy / repair 目标差异化建模指标（例如惩罚负库存的权重不同）
    - 采用在线增量学习模型 (SGDRegressor) 避免重训练全量成本
    - 引入特征重要性输出以分析驱动信号
    - 动态调整 ε (从较高探索 → 逐步降低)

特征向量结构 (extract_state_features 返回顺序)
    [
      0: objective_value(若 inf → 1e6),
      1: num_vehicles,
      2: avg_vehicle_utilization,
      3..(2 + |veh_types|): each veh_type frequency ratio (sorted by数字可转换顺序),
      next: avg_inventory_level,
      next: negative_inventory_ratio,
      next: avg_inventory_usage_ratio (正库存对工厂上限之平均占用),
      next: demand_satisfaction_ratio,
      next: unmet_demand_ratio,
      next: search_progress_ratio,
      last: recent_improvement_indicator
    ]
    注：当某一类信息缺失(如无车辆或尚未初始化库存)则用 0 占位。

使用方式 (典型流程)
    1) 初始化 selector = MLOperatorSelector(param_tuner, rng)
    2) 每次算子执行后调用 record_operator_performance(...)
    3) 周期性调用 train_models()
    4) 在下一次需要选择算子时调用 select_best_operator(state, 'destroy', candidates)

"""

from __future__ import annotations

# =========================
# 标准库
# =========================
import math
from typing import Dict, List, Tuple, Any, Optional

# =========================
# 第三方库
# =========================
import numpy as np
import numpy.random as rnd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# =========================
# 项目内部
# =========================
from typing import TYPE_CHECKING
from .param_tuner import ParamAutoTuner
from .alns_config import default_config as ALNSConfig

if TYPE_CHECKING:
    from .alnsopt import SolutionState


class MLOperatorSelector:
    """
    机器学习增强的算子选择器 (Destroy / Repair 两者皆可使用)

    属性:
        param_tuner         ParamAutoTuner  算子参数调度器 (提供每个算子当前参数)
        rng                 numpy.random.Generator 随机源
        destroy_features    List[np.ndarray]  破坏算子训练样本特征
        destroy_performances List[float]      破坏算子对应性能改进
        repair_features     List[np.ndarray]  修复算子训练样本特征
        repair_performances List[float]       修复算子对应性能改进
        destroy_op_map      Dict[str,int]     破坏算子名称->one-hot索引
        repair_op_map       Dict[str,int]     修复算子名称->one-hot索引
        destroy_model       (Ridge | RandomForestRegressor | None)
        destroy_scaler      StandardScaler | None
        repair_model        (Ridge | RandomForestRegressor | None)
        repair_scaler       StandardScaler | None
        min_samples         int  触发首次训练的最小样本数 (小于此值不训练)
        retrain_interval    int  重训练间隔 (基于 param_tuner.iteration)
        last_train_iter     int  上一次训练时的迭代计数 (用于控制频次)

    训练与选择逻辑:
        - 样本量不足时不训练，直接随机选择候选 (保持探索)
        - 样本量达阈值后按间隔训练；数据量较小时选择线性模型，数据足够后用随机森林
        - 预测阶段采用 ε-贪心，避免过早完全利用
    """

    def __init__(self, param_tuner: ParamAutoTuner, rng: rnd.Generator):
        # 外部依赖
        self.param_tuner = param_tuner
        self.rng = rng

        # 样本缓存
        self.destroy_features: List[np.ndarray] = []
        self.destroy_performances: List[float] = []
        self.repair_features: List[np.ndarray] = []
        self.repair_performances: List[float] = []

        # 算子映射
        self.destroy_op_map: Dict[str, int] = {}
        self.repair_op_map: Dict[str, int] = {}

        # 模型与标准化器
        self.destroy_model: Optional[Any] = None
        self.destroy_scaler: Optional[StandardScaler] = None
        self.repair_model: Optional[Any] = None
        self.repair_scaler: Optional[StandardScaler] = None

        # 训练控制参数 (defaults from centralized config)
        self.min_samples: int = getattr(ALNSConfig, "ML_INITIAL_SAMPLE_SIZE", 20)
        self.retrain_interval: int = getattr(ALNSConfig, "ML_RETRAIN_INTERVAL", 50)
        self.last_train_iter: int = 0
        self._no_improve_streak: Dict[str, int] = {"destroy": 0, "repair": 0}
        self._fallback_cooldown: Dict[str, int] = {"destroy": 0, "repair": 0}

        # 初始化算子映射
        self._init_operator_maps()

    def set_iteration(
        self,
        current: int,
        maximum: int,
        no_improve_iters: int = 0,
    ):
        """Synchronize the selector with the current ALNS progress."""
        try:
            if self.param_tuner is not None:
                self.param_tuner.set_iteration(
                    int(current), int(maximum), no_improve_iters=int(no_improve_iters)
                )
        except Exception:
            pass

    def _resolve_progress(self, state: Optional["SolutionState"]) -> float:
        if state is not None:
            tracker = getattr(state, "tracker", None)
            if tracker is not None:
                try:
                    stats = tracker.get_statistics()
                    current_iter = int(stats.get("total_iterations", 0))
                    max_iter = int(stats.get("max_iterations", 0))
                    if max_iter > 0:
                        return max(0.0, min(1.0, current_iter / max_iter))
                except Exception:
                    pass
        if self.param_tuner is not None:
            try:
                max_iter = int(getattr(self.param_tuner, "max_iterations", 0))
                cur_iter = int(getattr(self.param_tuner, "iteration", 0))
                if max_iter > 0:
                    return max(0.0, min(1.0, cur_iter / max_iter))
            except Exception:
                pass
        return 0.0

    def _sample_count(self, op_type: str) -> int:
        if op_type == "destroy":
            return len(self.destroy_features)
        return len(self.repair_features)

    def _takeover_min_samples(self) -> int:
        base = int(getattr(ALNSConfig, "ML_TAKEOVER_MIN_SAMPLE_SIZE", self.min_samples))
        if self.param_tuner is not None and bool(getattr(self.param_tuner, "small_instance", False)):
            small_base = int(getattr(ALNSConfig, "ML_TAKEOVER_MIN_SAMPLE_SIZE_SMALL", base))
            return max(base, small_base)
        return base

    def should_force_fallback(self, op_type: str) -> bool:
        cooldown = int(self._fallback_cooldown.get(op_type, 0))
        if cooldown <= 0:
            return False
        self._fallback_cooldown[op_type] = max(0, cooldown - 1)
        return True

    def update_op_feedback(self, op_type: str, improvement: float):
        key = "destroy" if op_type == "destroy" else "repair"
        if improvement > 0:
            self._no_improve_streak[key] = 0
            return

        streak = int(self._no_improve_streak.get(key, 0)) + 1
        self._no_improve_streak[key] = streak
        limit = int(getattr(ALNSConfig, "ML_STAGNATION_NO_IMPROVE_LIMIT", 18))
        if streak >= limit:
            cooldown = int(getattr(ALNSConfig, "ML_FALLBACK_COOLDOWN_ITERS", 30))
            self._fallback_cooldown[key] = max(
                int(self._fallback_cooldown.get(key, 0)),
                max(1, cooldown),
            )
            self._no_improve_streak[key] = 0

    def should_use_ml(
        self,
        state: Optional["SolutionState"],
        op_type: str,
        min_sample_size: Optional[int] = None,
    ) -> bool:
        if self.should_force_fallback(op_type):
            return False

        required_samples = self._takeover_min_samples()
        if min_sample_size is not None:
            required_samples = max(required_samples, int(min_sample_size))

        if self._sample_count(op_type) < required_samples:
            return False

        progress = self._resolve_progress(state)
        takeover_progress = float(getattr(ALNSConfig, "ML_TAKEOVER_MIN_PROGRESS", 0.30))
        if progress < takeover_progress:
            return False

        model = self.destroy_model if op_type == "destroy" else self.repair_model
        scaler = self.destroy_scaler if op_type == "destroy" else self.repair_scaler
        return model is not None and scaler is not None

    def _dynamic_epsilon(self, state: Optional["SolutionState"]) -> float:
        progress = self._resolve_progress(state)
        base = float(getattr(ALNSConfig, "ML_EPSILON_BASE", 0.10))
        early_boost = float(getattr(ALNSConfig, "ML_EPSILON_EARLY_BOOST", 0.15))
        epsilon = base + (1.0 - progress) * early_boost
        if self.param_tuner is not None and bool(getattr(self.param_tuner, "small_instance", False)):
            epsilon += float(getattr(ALNSConfig, "ML_EPSILON_SMALL_INSTANCE_BOOST", 0.05))
        return max(0.02, min(0.60, epsilon))

    # ------------------------------------------------------------------
    # 初始化与特征提取
    # ------------------------------------------------------------------
    def _init_operator_maps(self):
        """初始化破坏 / 修复算子的 one-hot 编码索引"""
        destroy_ops = [
            "random_removal",
            "shaw_removal",
            "periodic_shaw_removal",
            "path_removal",
            "cross_period_dealer_removal",
            "period_range_removal",
            "worst_removal",
            "infeasible_removal",
            "surplus_inventory_removal",
            "inventory_level_destruction",
        ]

        repair_ops = [
            "greedy_repair",
            "local_search_repair",
            "inventory_balance_repair",
            "smart_batch_repair",
            "infeasible_repair",
            "regret_based_repair",
            "sequential_repair",
            "lookahead_repair",
            "fleet_consolidation_repair",
        ]

        for i, op in enumerate(destroy_ops):
            self.destroy_op_map[op] = i
        for i, op in enumerate(repair_ops):
            self.repair_op_map[op] = i

    def extract_state_features(self, state: SolutionState) -> np.ndarray:
        """
        从当前解状态提取特征 (见文件顶部特征结构说明)

        注意:
            - 调用 state.objective() 可能触发 validate()→compute_inventory()，应避免在热循环中过度调用。
            - 若无车辆 / 库存 / tracker，使用 0 占位保持维度稳定。
        """
        data = state.data
        features: List[float] = []

        # 1. 目标函数值 (不可行用大数代替以维持数值稳定)
        obj_value = state.objective()
        features.append(obj_value if not math.isinf(obj_value) else 1e6)

        # 2. 车辆指标
        num_vehicles = len(state.vehicles)
        features.append(num_vehicles)

        if num_vehicles > 0:
            # 平均利用率
            total_util = 0.0
            veh_type_counts: Dict[str, int] = {}
            for veh in state.vehicles:
                load = sum(
                    qty * data.sku_sizes[sku] for (sku, _), qty in veh.cargo.items()
                )
                cap = data.veh_type_cap[veh.type]
                util = load / cap if cap > 0 else 0.0
                total_util += util
                veh_type_counts[veh.type] = veh_type_counts.get(veh.type, 0) + 1
            avg_util = total_util / num_vehicles
            features.append(avg_util)

            # 车辆类型分布(归一化频率)。优先按数值排序，失败时按字符串排序，避免类型不可转 float 时崩溃。
            def _veh_type_sort_key(v):
                try:
                    return (0, float(v))
                except Exception:
                    return (1, str(v))

            for veh_type in sorted(data.all_veh_types, key=_veh_type_sort_key):
                features.append(veh_type_counts.get(veh_type, 0) / num_vehicles)
        else:
            features.append(0.0)  # 平均利用率
            features.extend([0.0] * len(data.all_veh_types))

        # 3. 库存指标 (objective() 已保证 compute_inventory 调用)
        if state.s_ikt:
            inv_values = list(state.s_ikt.values())
            avg_inv = sum(inv_values) / len(inv_values)
            features.append(avg_inv)

            neg_inv_ratio = sum(1 for v in inv_values if v < 0) / len(inv_values)
            features.append(neg_inv_ratio)

            # 计算 (plant, day) 聚合库存/上限占比的平均值
            plant_day_pos_inv: Dict[Tuple[str, int], float] = {}
            for (plant, _sku, day), inv in state.s_ikt.items():
                if inv > 0:
                    plant_day_pos_inv[(plant, day)] = (
                        plant_day_pos_inv.get((plant, day), 0) + inv
                    )
            usage_acc = 0.0
            usage_cnt = 0
            for (plant, day), inv_sum in plant_day_pos_inv.items():
                cap = data.plant_inv_limit[plant]
                if cap > 0:
                    usage_acc += inv_sum / cap
                    usage_cnt += 1
            avg_usage_ratio = usage_acc / usage_cnt if usage_cnt > 0 else 0.0
            features.append(avg_usage_ratio)
        else:
            features.extend([0.0, 0.0, 0.0])

        # 4. 需求满足程度
        shipped = state.compute_shipped()
        total_demand = sum(data.demands.values()) if data.demands else 0
        total_shipped = sum(shipped.values()) if shipped else 0
        satisfaction_ratio = total_shipped / total_demand if total_demand > 0 else 1.0
        features.append(satisfaction_ratio)

        unmet_count = sum(
            1
            for (dealer, sku), demand in data.demands.items()
            if shipped.get((dealer, sku), 0) < demand
        )
        unmet_ratio = unmet_count / len(data.demands) if data.demands else 0.0
        features.append(unmet_ratio)

        # 5. 搜索进度 (依赖 tracker，可选)
        if state.tracker:
            stats = state.tracker.get_statistics()
            current_iter = stats.get("total_iterations", 0)
            max_iter = stats.get("max_iterations", 1000)
            progress = current_iter / max_iter if max_iter > 0 else 0.0
            recent_improvement = stats.get("recent_improvement", 0.0)
            features.append(progress)
            features.append(recent_improvement)
        else:
            features.extend([0.0, 0.0])

        return np.array(features, dtype=float)

    # ------------------------------------------------------------------
    # 编码与记录
    # ------------------------------------------------------------------
    def encode_operator(self, op_name: str, op_type: str) -> np.ndarray:
        """
        将算子名称编码为 one-hot 向量。
        未注册名称 → 返回全零向量，避免抛出异常影响主流程。
        """
        op_map = self.destroy_op_map if op_type == "destroy" else self.repair_op_map
        encoding = np.zeros(len(op_map), dtype=float)
        idx = op_map.get(op_name)
        if idx is not None:
            encoding[idx] = 1.0
        return encoding

    def record_operator_performance(
        self,
        op_name: str,
        op_type: str,
        state: SolutionState,
        improvement: float,
    ):
        """
        记录算子在当前状态下带来的性能改进 (improvement):
            improvement > 0: 目标改进 (例如 cost 降低幅度)
            improvement = 0: 无变化
            improvement < 0: 退化
        说明: 具体 improvement 指标由外部调用者定义与计算。
        """
        state_features = self.extract_state_features(state)
        op_encoding = self.encode_operator(op_name, op_type)
        features = np.concatenate([state_features, op_encoding])

        if op_type == "destroy":
            self.destroy_features.append(features)
            self.destroy_performances.append(improvement)
        else:
            self.repair_features.append(features)
            self.repair_performances.append(improvement)

    # ------------------------------------------------------------------
    # 模型训练
    # ------------------------------------------------------------------
    def train_models(self, force: bool = False):
        """
        训练 / 重训练破坏与修复算子的性能预测模型。

        触发条件:
            force=True 强制训练 (忽略间隔)
            或 (当前迭代 - 上次训练迭代) >= retrain_interval

        注意:
            - 若样本不足 min_samples → 本轮跳过，且不会更新 last_train_iter（保持立即重试机制）
            - 仅当本轮实际完成至少一个模型训练 (destroy 或 repair) 时才更新 last_train_iter
            - 避免首次可训练点被 retrain_interval 推迟
        """
        current_iter = 0
        if self.param_tuner:
            current_iter = self.param_tuner.iteration

        need_train = force or (
            current_iter - self.last_train_iter >= self.retrain_interval
        )
        if not need_train:
            return

        trained_destroy = False
        trained_repair = False
        # ---------- 训练破坏算子模型 ----------
        if len(self.destroy_features) >= self.min_samples:
            X = np.array(self.destroy_features, dtype=float)
            y = np.array(self.destroy_performances, dtype=float)

            self.destroy_scaler = StandardScaler()
            X_scaled = self.destroy_scaler.fit_transform(X)

            if len(X) < getattr(ALNSConfig, "ML_USE_RF_THRESHOLD", 50):
                self.destroy_model = Ridge(
                    alpha=getattr(ALNSConfig, "ML_RIDGE_ALPHA", 1.0)
                )
            else:
                self.destroy_model = RandomForestRegressor(
                    n_estimators=getattr(ALNSConfig, "ML_RF_N_ESTIMATORS", 50),
                    max_depth=getattr(ALNSConfig, "ML_RF_MAX_DEPTH", 10),
                    random_state=int(self.rng.integers(0, 10_000)),
                )
            self.destroy_model.fit(X_scaled, y)
            print(f"[ML] 破坏算子模型训练完成，样本数: {len(X)}")
            trained_destroy = True

        # ---------- 训练修复算子模型 ----------
        if len(self.repair_features) >= self.min_samples:
            X = np.array(self.repair_features, dtype=float)
            y = np.array(self.repair_performances, dtype=float)

            self.repair_scaler = StandardScaler()
            X_scaled = self.repair_scaler.fit_transform(X)

            if len(X) < getattr(ALNSConfig, "ML_USE_RF_THRESHOLD", 50):
                self.repair_model = Ridge(
                    alpha=getattr(ALNSConfig, "ML_RIDGE_ALPHA", 1.0)
                )
            else:
                self.repair_model = RandomForestRegressor(
                    n_estimators=getattr(ALNSConfig, "ML_RF_N_ESTIMATORS", 50),
                    max_depth=getattr(ALNSConfig, "ML_RF_MAX_DEPTH", 10),
                    random_state=int(self.rng.integers(0, 10_000)),
                )
            self.repair_model.fit(X_scaled, y)
            print(f"[ML] 修复算子模型训练完成，样本数: {len(X)}")
            trained_repair = True

        # 若本轮至少训练了一个模型，则更新时间戳；否则保持原值以便下轮继续尝试
        if trained_destroy or trained_repair:
            self.last_train_iter = current_iter

    # ------------------------------------------------------------------
    # 算子选择
    # ------------------------------------------------------------------
    def select_best_operator(
        self,
        state: SolutionState,
        op_type: str,
        candidates: List[str],
    ) -> Tuple[str, Dict[str, Any]]:
        """
        基于当前状态与模型预测选择一个算子及其参数。

        参数:
            state      当前解
            op_type    'destroy' | 'repair'
            candidates 候选算子名称列表 (若为空给出默认回退算子)

        返回:
            (op_name, params_dict)

        策略:
            - 没有候选或模型未训练 → 随机 / 默认
            - 已训练 → 对每个候选作预测 → ε-贪心 (ε=0.1) 兼顾探索与利用
        """
        if not candidates:
            if op_type == "destroy":
                # fallback to centralized destroy defaults
                destroy_defaults = ALNSConfig.get_destroy_params()
                return "random_removal", {
                    "degree": destroy_defaults.get("random_removal_degree", 0.25)
                }
            # fallback to centralized repair defaults
            repair_defaults = ALNSConfig.get_repair_params()
            return "greedy_repair", repair_defaults.get("greedy_repair", {})

        state_features = self.extract_state_features(state)

        model = self.destroy_model if op_type == "destroy" else self.repair_model
        scaler = self.destroy_scaler if op_type == "destroy" else self.repair_scaler

        # 模型尚不可用：随机探索
        if model is None or scaler is None:
            selected = self.rng.choice(candidates)
            params = self.param_tuner.get_operator_params(selected)
            return selected, params

        predictions = []
        for op_name in candidates:
            op_encoding = self.encode_operator(op_name, op_type)
            features = np.concatenate([state_features, op_encoding]).reshape(1, -1)
            try:
                features_scaled = scaler.transform(features)
                pred = model.predict(features_scaled)[0]
            except Exception:
                # 任何不可预期的 transform 失败，降级为 0 预测，保证鲁棒性
                pred = 0.0
            params = self.param_tuner.get_operator_params(op_name)
            predictions.append((op_name, pred, params))

        # 预测性能降序
        predictions.sort(key=lambda x: x[1], reverse=True)

        # ε-贪心 (动态探索强度, 早期更大，后期收敛)
        epsilon = self._dynamic_epsilon(state)
        if self.rng.random() < epsilon:
            idx = int(self.rng.integers(0, len(predictions)))
            return predictions[idx][0], predictions[idx][2]
        return predictions[0][0], predictions[0][2]
