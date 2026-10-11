"""
模块: alnstrack
功能定位:
    提供 ALNS 求解过程中的统一跟踪(Tracking)与度量(Statistics)支持, 包含:
      1. 迭代级记录: 当前迭代号 / 当前目标值 / 历史最优值 / gap 序列 / 用时
      2. 最优解持久化: 自动保存迭代中发现的改进解 (深拷贝状态)
      3. 度量导出: 可选写入 CSV (便于后续分析/可视化/实验比较)
      4. ML 支持: 存储特征(feature)与标签(label)样本、训练后的模型与 scaler 缓存
      5. 外部交互: ParamAutoTuner 读取迭代统计 (total_iterations / max_iterations) 做阶段自适应

设计要点:
    - 轻量: 不介入搜索决策逻辑, 仅消费 state (SolutionState) 提供的接口
    - 解耦: ML/参数调优模块通过 get_statistics()/get_ml_cache_info() 获取所需信息
    - 容错: CSV 写入失败不应中断主流程 (可扩展 try/except)
    - 历史裁剪: (可选) 当前特征样本列表 features/labels 维持上限, 防止长期运行内存膨胀
    - 早停: 维护 no_improve_iters (自上次取得最优以来的迭代数) 与 early_stop_patience (容忍阈值)
            外部可在主循环中调用 tracker.should_stop() 判断是否提前终止

Gap 定义:
    gap = (current_obj - best_obj) / current_obj * 100%
    说明:
      - 目标为最小化场景; 当出现新最优时 current_obj == best_obj → gap = 0
      - 该实现假设 best_obj ≤ current_obj (若出现精度波动导致 best_obj > current_obj,
        gap 仍为正数; 如需严格单调可在更新前加 min/epsilon 保护)

使用流程参考:
    tracker = ALNSTracker(output_file='logs/alns_progress.csv')
    for iter in range(MAX):
        ... # 生成 / 接受新解
        tracker.on_iteration(state, rng)
    stats = tracker.get_statistics()

可改进点(未实现):
    - 引入滑动窗口/指数加权的 gap 平滑指标
    - 增加 JSON 行日志输出用于更丰富的后处理
"""

# =========================
# 标准库
# =========================
import os
import csv
import time
import copy
from typing import Optional, Any, Tuple, TYPE_CHECKING

# =========================
# 第三方库
# =========================
import numpy as np
import numpy.random as rnd

# =========================
# 项目内部依赖
# =========================
from .optutility import LogPrinter
from .alns_config import default_config as ALNSConfig

if TYPE_CHECKING:  # 仅类型检查时导入, 避免循环依赖与运行时开销
    from .alnsopt import SolutionState

# 默认日志打印器（仅作为兜底，不参与跨 run 共享计时）
DEFAULT_LOG_PRINTER = LogPrinter(time.time())


class ALNSTracker:
    """
    ALNS 过程跟踪器:
    负责记录优化搜索中的关键统计量, 支撑:
      - 结果分析 (目标值/Gap 曲线)
      - 阶段判断 (与 ParamAutoTuner 协调)
      - 机器学习算子特征/标签累积与模型缓存
    """

    def __init__(
        self,
        output_file: Optional[str] = None,
        log_printer: Optional[LogPrinter] = None,
    ):
        # 迭代/目标相关
        self.iteration: int = 0
        self.max_iterations: int = getattr(ALNSConfig, "MAX_ITERATIONS", 1000)
        self.current_obj: float = float("inf")
        self.best_obj: float = float("inf")
        self.best_solution: Optional["SolutionState"] = None
        self.best_actual_obj: float = float("inf")
        self.best_actual_solution: Optional["SolutionState"] = None
        self.initial_search_obj: float = float("inf")
        self.initial_actual_obj: float = float("inf")

        # 时间 & 历史序列
        self.start_time: float = time.time()
        self.log_printer: LogPrinter = log_printer or LogPrinter(self.start_time)
        self.objectives: list[float] = []
        self.gaps: list[float] = []

        # 可选结果持久化
        self.output_file: Optional[str] = output_file
        if self.output_file:
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            with open(self.output_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["Iteration", "Current_Obj", "Best_Obj", "GapVsInitial"]
                )

        # ML 相关 (learning_based_repair / 未来学习机制)
        self.features: list[
            list[float]
        ] = []  # 每条特征: 例如 [demand, size, day, inv, util, ...]
        self.labels: list[float] = []  # 对应改进值 improvement
        self._cached_model: Optional[Any] = None
        self._cached_scaler: Optional[Any] = None
        self._last_train_iteration: Optional[int] = None

        # 早停机制相关
        self.early_stop_patience: int = 500  # 可在外部根据问题规模/期望运行时间调节
        self.no_improve_iters: int = 0  # 自上次最优改进以来未改善的迭代计数
        self.best_update_iteration: int = (
            0  # 最近一次刷新最优解的迭代号 (用于分析改进分布)
        )
        self.recent_improvement: float = (
            0.0  # 最近一次改进幅度 (prev_best - new_best); 未改进迭代置 0
        )
        # 精英池: 记录若干个结构不同且质量较好的可行解, 供停滞重连/重启使用
        self.elite_pool: list[dict[str, Any]] = []
        self.elite_pool_max_size: int = int(
            getattr(ALNSConfig, "ELITE_POOL_MAX_SIZE", 8)
        )

    def get_iteration(self) -> int:
        return int(self.iteration)

    def get_max_iterations(self) -> int:
        return int(self.max_iterations)

    def set_max_iterations(self, max_iterations: int):
        self.max_iterations = int(max_iterations)

    def set_initial_reference(
        self,
        initial_search_obj: Optional[float] = None,
        initial_actual_obj: Optional[float] = None,
    ):
        if initial_search_obj is not None:
            self.initial_search_obj = float(initial_search_obj)
        if initial_actual_obj is not None:
            self.initial_actual_obj = float(initial_actual_obj)

    def _signed_gap_from_initial(self, value: float, baseline: float) -> float:
        if (
            not np.isfinite(value)
            or not np.isfinite(baseline)
            or abs(baseline) <= 1e-12
        ):
            return 0.0
        return float((value - baseline) / abs(baseline) * 100.0)

    def _profile_signature(self, state: "SolutionState") -> tuple:
        try:
            vehicles = (
                list(state.iter_used_vehicles())
                if hasattr(state, "iter_used_vehicles")
                else list(getattr(state, "vehicles", []) or [])
            )
        except Exception:
            vehicles = list(getattr(state, "vehicles", []) or [])

        veh_count = int(len(vehicles))
        load_ratio = 0.0
        large_ratio = 0.0
        fixed_cost = 0.0
        try:
            total_load = 0.0
            total_cap = 0.0
            threshold = 0.0
            caps = []
            for vt in list(getattr(state.data, "all_veh_types", []) or []):
                try:
                    caps.append(float(state.data.veh_type_cap[vt]))
                except Exception:
                    continue
            if caps:
                caps.sort()
                idx = int(round((len(caps) - 1) * 0.67))
                threshold = float(caps[idx])
            for veh in vehicles:
                try:
                    cap = float(
                        state.data.veh_type_cap.get(
                            veh.type, getattr(veh, "max_capacity", 0.0) or 0.0
                        )
                    )
                    total_cap += cap
                    total_load += float(state.compute_veh_load(veh))
                    fixed_cost += float(state.data.veh_type_cost.get(veh.type, 0.0))
                    if cap >= threshold:
                        large_ratio += 1.0
                except Exception:
                    continue
            if veh_count > 0 and total_cap > 1e-12:
                load_ratio = total_load / total_cap
                large_ratio = large_ratio / float(veh_count)
        except Exception:
            pass
        bin_size = max(1, int(getattr(ALNSConfig, "ELITE_SIGNATURE_BIN_SIZE", 10)))
        return (
            veh_count,
            int(round(load_ratio * bin_size)),
            int(round(large_ratio * bin_size)),
            int(round(fixed_cost / 10.0)),
        )

    def record_elite_candidate(self, state: "SolutionState") -> bool:
        """Record a feasible elite candidate into the bounded elite pool."""
        if state is None:
            return False
        try:
            feasible, _ = state.validate()
        except Exception:
            feasible = False
        if not feasible:
            return False

        try:
            obj = float(state.objective())
        except Exception:
            return False
        if not np.isfinite(obj):
            return False

        try:
            profile = {
                "vehicles": float(
                    len(list(state.iter_used_vehicles()))
                    if hasattr(state, "iter_used_vehicles")
                    else len(getattr(state, "vehicles", []) or [])
                ),
                "load_ratio": 0.0,
                "large_ratio": 0.0,
                "fixed_cost": float(state.calculate_cost()),
            }
            sig = self._profile_signature(state)
        except Exception:
            profile = {
                "vehicles": float(len(getattr(state, "vehicles", []) or [])),
                "load_ratio": 0.0,
                "large_ratio": 0.0,
                "fixed_cost": 0.0,
            }
            sig = (int(profile["vehicles"]), 0, 0, 0)

        pool = list(getattr(self, "elite_pool", []) or [])
        if pool:
            existing_best = min(float(item.get("obj", float("inf"))) for item in pool)
        else:
            existing_best = float("inf")

        # Fast de-duplication by signature; keep the better objective for the same signature.
        updated = False
        for item in pool:
            if tuple(item.get("signature", ())) == sig:
                if obj < float(item.get("obj", float("inf"))):
                    item["state"] = state.copy()
                    item["obj"] = obj
                    item["profile"] = profile
                    item["signature"] = sig
                updated = True
                break
        if not updated:
            pool.append(
                {
                    "state": state.copy(),
                    "obj": obj,
                    "profile": profile,
                    "signature": sig,
                }
            )

        # Keep only a small, quality-biased diverse pool.
        pool.sort(
            key=lambda x: (
                float(x.get("obj", float("inf"))),
                tuple(x.get("signature", ())),
            )
        )
        max_size = max(1, int(getattr(self, "elite_pool_max_size", 8)))
        trimmed = []
        seen_sig = set()
        for item in pool:
            sig_i = tuple(item.get("signature", ()))
            if sig_i in seen_sig:
                continue
            seen_sig.add(sig_i)
            trimmed.append(item)
            if len(trimmed) >= max_size:
                break
        self.elite_pool = trimmed
        return obj <= existing_best + 1e-9

    # -----------------------------------------------------------------
    # 主迭代回调
    # -----------------------------------------------------------------
    def on_iteration(self, state: "SolutionState", rng: rnd.Generator, **kwargs):
        """
        每次 ALNS 主循环结束后调用, 执行:
          1. 迭代号自增
          2. 获取当前解目标值 (内部会利用 objective 缓存)
          3. 若为可行且优于历史最优则更新 best_solution
          4. 计算并记录 gap
          5. 可选写入 CSV (频繁 IO 可根据需要做采样)
          6. 周期性打印进度 (当前: 每 100 次)

        参数:
            state : 当前解 (需提供 objective()/validate())
            rng   : 随机数生成器 (保留扩展接口; 当前未使用)
            kwargs: 预留扩展 (例如可传入附加日志/标记)
        """
        self.iteration += 1

        # 在每次迭代开始时同步更新 ParamAutoTuner 的迭代信息，
        # 以确保随后对算子参数的采样(get_operator_params)能够读取到最新的 exploration_rate。
        try:
            if (
                hasattr(state, "param_tuner")
                and state.param_tuner
                and hasattr(self, "max_iterations")
            ):
                current_iter = self.iteration
                max_iter = self.max_iterations
                state.param_tuner.set_iteration(
                    current_iter,
                    max_iter,
                    no_improve_iters=self.no_improve_iters,
                )
        except Exception as e:
            # 保护性容错：若更新失败仅打印警告，不影响主流程
            self.log_printer.print(
                f"[WARN] Failed to update param_tuner iteration: {e}", color="yellow"
            )

        # 当前目标值 (若 objective 缓存有效则为 O(1))
        current_obj = float(state.objective())
        self.current_obj = current_obj
        self.objectives.append(current_obj)
        current_actual_obj = float(state.calculate_cost())

        # 验证可行性 & 更新最优
        is_feasible, _ = state.validate()
        improved = False
        if current_obj < self.best_obj and is_feasible:
            prev_best = self.best_obj
            self.best_obj = current_obj
            # 深拷贝保存最优状态 (避免后续原地修改污染)
            try:
                # 优先使用 SolutionState.copy() 以获取轻量快照
                self.best_solution = state.copy()
            except Exception:
                # 兼容旧实现：若对象未实现 copy() 回退到 deepcopy
                self.best_solution = copy.deepcopy(state)
            improved = True
            # 记录本次实际改进幅度 (若之前 best 为 inf 则按 0 处理)
            if prev_best != float("inf"):
                self.recent_improvement = max(0.0, prev_best - current_obj)
            else:
                self.recent_improvement = 0.0

            search_improvement_vs_prev = float("nan")
            if np.isfinite(prev_best) and abs(prev_best) > 1e-12:
                search_improvement_vs_prev = float(
                    (prev_best - current_obj) / abs(prev_best) * 100.0
                )
            elapsed_time = time.time() - self.start_time
            actual_gap_from_initial = self._signed_gap_from_initial(
                current_actual_obj, self.initial_actual_obj
            )
            self.log_printer.print(
                f"Iteration {self.iteration}: New best feasible solution found!",
                color="bold green",
            )
            if np.isfinite(search_improvement_vs_prev):
                self.log_printer.print(
                    f"Search Objective: {self.best_obj:.6f}, Improvement vs Prev Best: +{search_improvement_vs_prev:.2f}%, Time: {elapsed_time:.2f}s",
                    color="bold green",
                )
            else:
                self.log_printer.print(
                    f"Search Objective: {self.best_obj:.6f}, Improvement vs Prev Best: N/A (first best), Time: {elapsed_time:.2f}s",
                    color="bold green",
                )
            try:
                self.log_printer.print(
                    f"Actual Objective (Gurobi-aligned): {current_actual_obj:.2f}, Feasible: Yes, Gap vs Initial: {actual_gap_from_initial:+.2f}%",
                    color="bold cyan",
                )
            except Exception:
                pass

        # 按实际目标函数独立维护最优可行解（用于最终业务口径输出）
        if is_feasible and current_actual_obj < self.best_actual_obj:
            self.best_actual_obj = current_actual_obj
            try:
                self.best_actual_solution = state.copy()
            except Exception:
                self.best_actual_solution = copy.deepcopy(state)

        # 维护一个小型精英池, 供后续 path relinking / stagnation escape 使用。
        try:
            elite_sample_interval = max(
                1, int(getattr(ALNSConfig, "ELITE_POOL_SAMPLE_INTERVAL", 50))
            )
            if is_feasible and (
                improved or self.iteration % elite_sample_interval == 0
            ):
                self.record_elite_candidate(state)
        except Exception:
            pass

        # 维护无改进计数 (早停)
        if improved:
            self.no_improve_iters = 0
            self.best_update_iteration = self.iteration
        else:
            self.no_improve_iters += 1
            # 未取得新最优时 recent_improvement 置 0，表示“最近一次迭代”无改进
            self.recent_improvement = 0.0

        # 运行过程中使用“相对初始解”的 signed gap
        gap_vs_initial = self._signed_gap_from_initial(
            current_actual_obj, self.initial_actual_obj
        )
        self.gaps.append(gap_vs_initial)

        # 周期性打印概览
        if self.iteration % 100 == 0:
            self.log_printer.print(
                f"Iteration: {self.iteration}\tSearch Obj: {self.current_obj:.6f}\t"
                f"Actual Obj: {current_actual_obj:.2f}\tFeasible: {'Yes' if is_feasible else 'No'}\t"
                f"Gap vs Initial: {gap_vs_initial:+.2f}%\tBest Search Obj: {self.best_obj:.6f}"
            )

        # 可选持久化
        if self.output_file:
            try:
                with open(self.output_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            self.iteration,
                            self.current_obj,
                            self.best_obj,
                            gap_vs_initial,
                        ]
                    )
            except Exception as e:
                self.log_printer.print(
                    f"[WARN] Failed to append tracker CSV: {e}", color="yellow"
                )

    # -----------------------------------------------------------------
    # ML 数据采集
    # -----------------------------------------------------------------
    def update_ml_data(self, feature: list, label: float) -> Tuple[int, int]:
        """
        存储一条 (feature, label) 样本。
        - feature: 自由结构 (约定维度在使用侧保持一致)
        - label:   通常为目标函数改进值 improvement (正数为好)

        返回:
            (当前特征样本数, 当前标签样本数)
        说明:
            这里设置一个硬上限 1000 (最近样本), 防止长时间运行内存膨胀。
            可扩展为基于时间/多级缓冲的采样策略。
        """
        self.features.append(feature)
        self.labels.append(label)
        if len(self.features) > 1000:  # 滑动窗口机制
            self.features = self.features[-1000:]
            self.labels = self.labels[-1000:]
        return len(self.features), len(self.labels)

    # -----------------------------------------------------------------
    # ML 模型缓存接口
    # -----------------------------------------------------------------
    def cache_ml_model(self, model: Any, scaler: Any, iteration: int):
        """
        缓存训练完成的 ML 模型及其标准化器, 并记录训练迭代号:
            model   : 已训练模型 (如 RandomForest / Ridge / 自定义)
            scaler  : 与 model 对应的特征预处理器
            iteration: 训练时对应的迭代编号
        用途:
            - 在后续迭代快速复用模型 (避免频繁重训)
            - 外部可据 _last_train_iteration 判断是否需要增量更新
        """
        self._cached_model = model
        self._cached_scaler = scaler
        self._last_train_iteration = iteration
        print(f"[Tracker] Cached ML model at iteration {iteration}")

    def get_cached_model(self) -> Tuple[Optional[Any], Optional[Any], Optional[int]]:
        """
        返回缓存 (model, scaler, last_train_iteration) 三元组。
        若任一为空, 表示尚未缓存或已被清除。
        """
        return self._cached_model, self._cached_scaler, self._last_train_iteration

    def has_cached_model(self) -> bool:
        """
        判断当前是否存在有效的模型缓存。
        条件: model, scaler, last_train_iteration 全部非 None。
        """
        return (
            self._cached_model is not None
            and self._cached_scaler is not None
            and self._last_train_iteration is not None
        )

    def clear_ml_cache(self):
        """
        清空模型缓存 (释放内存 / 触发重训)。
        """
        self._cached_model = None
        self._cached_scaler = None
        self._last_train_iteration = None
        print("[Tracker] Cleared ML model cache")

    def get_ml_cache_info(self) -> dict:
        """
        返回当前 ML 缓存状态摘要:
            has_model: 是否有缓存
            last_train_iteration: 上次训练迭代
            model_type / scaler_type: 类名 (便于调试)
            data_samples: 当前特征样本数量
        """
        return {
            "has_model": self.has_cached_model(),
            "last_train_iteration": self._last_train_iteration,
            "model_type": type(self._cached_model).__name__
            if self._cached_model
            else None,
            "scaler_type": type(self._cached_scaler).__name__
            if self._cached_scaler
            else None,
            "data_samples": len(self.features),
        }

    # -----------------------------------------------------------------
    # 早停判定
    # -----------------------------------------------------------------
    def should_stop(self) -> bool:
        """
        早停判定:
          若自上次取得最优解以来的未改进迭代数 >= early_stop_patience → 返回 True
        用法 (伪代码):
            while not tracker.should_stop():
                ...
        """
        return self.no_improve_iters >= self.early_stop_patience

    # -----------------------------------------------------------------
    # 汇总统计
    # -----------------------------------------------------------------
    def get_statistics(self) -> dict:
        """
        汇总当前追踪器的所有关键信息 (返回浅复制/值复制, 避免外部原地修改):
            total_iterations :     已完成迭代数
            max_iterations   :     规划的最大迭代数
            best_objective   :     历史最优目标值
            current_objective:     当前解目标值
            final_gap        :     当前 gap (若尚无历史则为 0)
            objectives_history:    目标值轨迹 (副本)
            gaps_history     :     gap 轨迹 (副本)
            elapsed_time     :     运行秒数
            best_solution    :     当前最优解 (对象引用; 深度使用需自行 deepcopy)
            features / labels:     ML 采样数据副本
            ml_cache_info    :     模型缓存摘要
            no_improve_iters :     自上次最优改进以来未改善的迭代计数
            early_stop_patience:   早停容忍阈值
            best_update_iteration: 最近一次刷新最优解的迭代号
            recent_improvement:    最近一次改进幅度 (prev_best - new_best); 未改进迭代置 0
        """
        return {
            "total_iterations": self.iteration,
            "max_iterations": self.max_iterations,
            "best_objective": self.best_obj,
            "best_actual_objective": self.best_actual_obj,
            "current_objective": self.current_obj,
            "final_gap": self._signed_gap_from_initial(
                self.current_obj, self.initial_search_obj
            )
            if self.objectives
            else 0.0,
            "final_actual_gap": self._signed_gap_from_initial(
                self.best_actual_solution.calculate_cost()
                if self.best_actual_solution is not None
                else (
                    self.best_solution.calculate_cost()
                    if self.best_solution is not None
                    else self.current_obj
                ),
                self.initial_actual_obj,
            )
            if self.objectives
            else 0.0,
            "objectives_history": self.objectives.copy(),
            "gaps_history": self.gaps.copy(),
            "elapsed_time": time.time() - self.start_time,
            "best_solution": self.best_solution,
            "best_actual_solution": self.best_actual_solution,
            "features": self.features.copy(),
            "labels": self.labels.copy(),
            "ml_cache_info": self.get_ml_cache_info(),
            "no_improve_iters": self.no_improve_iters,
            "early_stop_patience": self.early_stop_patience,
            "best_update_iteration": self.best_update_iteration,
            "recent_improvement": self.recent_improvement,
        }


# ---------------------------------------------------------------------
# Gap 计算函数 (保持与 ALNSTracker 独立, 便于复用/测试)
# ---------------------------------------------------------------------
def _is_all_zero(x) -> bool:
    ax = np.asarray(x)
    if ax.size == 1:
        # 单元素数组或标量: 用 isclose 判定 0
        return bool(np.isclose(float(ax), 0.0))
    # 多元素数组: 全部接近 0 则认为为 0 向量
    return bool(np.allclose(ax, 0.0))


def _to_scalar_or_mean(x) -> float:
    ax = np.asarray(x)
    if ax.size == 1:
        return float(ax)
    # 向量化结果退化为均值以保证数值可比性（保守处理）
    return float(np.mean(ax))


def calculate_gap(current_obj: float, best_obj: float) -> float:
    """
    计算 gap:
        gap = |current_obj - best_obj| / |current_obj| * 100
    说明:
      - 兼容 scalar 与 numpy array（单元素或多元素）输入
      - 边界处理与原实现一致，但避免在 numpy 对象上使用 Python 布尔上下文
    """
    # 两者均为零向量/标量
    if _is_all_zero(current_obj) and _is_all_zero(best_obj):
        return 0.0
    # current == 0 且 best > 0 -> inf
    if _is_all_zero(current_obj):
        best_val = _to_scalar_or_mean(best_obj)
        if best_val > 0:
            return float("inf")
    # 将可能的数组退化为标量以做相对差计算（向量以均值衡量）
    cur_val = _to_scalar_or_mean(current_obj)
    best_val = _to_scalar_or_mean(best_obj)
    return abs(cur_val - best_val) / max(1e-12, abs(cur_val)) * 100
