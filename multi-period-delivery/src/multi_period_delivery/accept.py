"""
模块: accept.py
职责:
- 提供一个用于在候选解目标值上施加未满足需求惩罚的包装器（DemandConstraintAccept）
- 通过 CandidateProxy 在不破坏原始 state 接口的前提下调整 objective() 返回值
- 统计调用/接受次数与累计惩罚，便于日志/调试/参数调整

注:
- 注释风格与项目中 multi_period_delivery/alnsopt.py 保持一致，采用详细中文说明与设计意图注释，
  目的是便于维护者理解实现细节和为何采用这种实现方式。
"""

from typing import Any, Dict
import numpy as np

from .alns_config import default_config as ALNSConfig


def _safe_float(value: Any, default: float = float("inf")) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


class _CandidateProxy:
    """
    候选解代理（Proxy）:
    - 将原始候选状态包装起来，覆盖 objective() 方法以在原始目标上加上计算得到的惩罚
    - 其它属性/方法委托给原始对象，以保持兼容性（例如 validate(), compute_shipped() 等）
    设计动机:
    - 不修改原始 candidate_state 对象的实现，仅在接受判定时临时改变 objective 行为
    - 便于将违约惩罚与任意基础接收准则（如 SimulatedAnnealing 等）组合使用
    """

    def __init__(self, original, penalty: float):
        self._orig = original
        self._penalty = float(penalty)

    def objective(self):
        """
        返回被惩罚后的目标值:
        - 尝试从原始对象获取目标值并转为 float
        - 若原始目标不可计算（异常或非有限），则保留该值（例如 inf）
        - 否则在原始目标上加上事先计算的惩罚值并返回
        """
        try:
            base = float(self._orig.objective())
        except Exception:
            base = float("inf")
        if not np.isfinite(base):
            return base
        return base + self._penalty

    def validate(self):
        """直接委托 validate 调用，保持行为一致性。"""
        return self._orig.validate()

    def compute_shipped(self):
        """
        保持与现有测试与调用的兼容性：
        - 某些状态对象可能实现 compute_shipped()，外部测试会直接查询该方法
        - 若包装的原始对象实现了该方法，则委托执行；否则返回空字典
        """
        if hasattr(self._orig, "compute_shipped"):
            return self._orig.compute_shipped()
        return {}

    def __getattr__(self, name: str) -> Any:
        """
        对未显式覆盖的属性访问进行委托：
        - 例如 .vehicles, .s_ikt 或其它方法/属性均由原始对象提供
        - 该实现确保代理对象在绝大多数上下文中可替代原始对象
        """
        return getattr(self._orig, name)


class DemandConstraintAccept:
    """
    包装式接收判定器（Acceptor）:
    - 在将候选态传递给基础接收器之前, 检测并基于未满足的需求计算惩罚
    - 将候选态替换为 _CandidateProxy，从而在不修改原始状态对象的前提下影响 objective 返回值
    - 兼容现有的基础接收器接口（例如 SimulatedAnnealing），并记录统计信息用于分析

    最小接口契约（被测试代码所期待）:
      - 对象为可调用: (rng, best_state, current_state, candidate_state) -> bool
      - get_statistics() -> dict

    惩罚计算规则:
      - 首先尝试调用 candidate_state.validate() 获取 (feasible, violations)
      - 若 infeasible 且 violations 中包含 "unmet_demand" 且为列表:
          * 尝试对 unmet_list 中每个元素以 (demand - shipped) 聚合求和 unmet_total
          * 惩罚 penalty = penalty_factor * unmet_total
      - 否则:
          * 退化为 penalty = penalty_factor * len(unmet_list) （或 1.0 作为兜底）
    """

    def __init__(self, base_acceptor, data, penalty_factor: float = 1000.0):
        """
        参数:
          - base_acceptor: 基础接收准则（callable 或具有相应调用签名的对象）
          - data: 问题数据引用（用于可能的额外统计/日志）
          - penalty_factor: 将 unmet demand 数量/量级映射到目标值尺度上的放大系数
        统计:
          - _calls: 被调用的次数
          - _accepted: 被接受的次数
          - _total_penalty: 累积应用的惩罚，用于计算平均惩罚等
        """
        self.base = base_acceptor
        self.data = data
        self.penalty_factor = float(penalty_factor)

        # statistics
        self._calls = 0
        self._accepted = 0
        self._total_penalty = 0.0

    def _resolve_progress(self, *states) -> float:
        for state in states:
            tracker = getattr(state, "tracker", None)
            if tracker is None:
                continue
            try:
                iteration = int(getattr(tracker, "iteration", 0))
            except Exception:
                iteration = 0
            try:
                max_iterations = int(getattr(tracker, "max_iterations", 0))
            except Exception:
                max_iterations = 0
            if max_iterations > 0:
                return max(0.0, min(1.0, iteration / max_iterations))
        return 0.0

    def _count_used_vehicles(self, state) -> int:
        if state is None:
            return 0
        try:
            if hasattr(state, "iter_used_vehicles") and callable(
                getattr(state, "iter_used_vehicles")
            ):
                return len(list(state.iter_used_vehicles()))
        except Exception:
            pass
        try:
            vehicles = list(getattr(state, "vehicles", []) or [])
        except Exception:
            vehicles = []
        count = 0
        for veh in vehicles:
            try:
                if veh is None or veh.is_empty():
                    continue
            except Exception:
                pass
            count += 1
        return count

    def _actual_priority_accept(self, current_state, candidate_state, progress: float) -> bool:
        current_actual = _safe_float(
            getattr(current_state, "calculate_cost", lambda: None)()
        )
        candidate_actual = _safe_float(
            getattr(candidate_state, "calculate_cost", lambda: None)()
        )
        current_search = _safe_float(
            getattr(current_state, "objective", lambda: None)()
        )
        candidate_search = _safe_float(
            getattr(candidate_state, "objective", lambda: None)()
        )

        actual_abs_tol = float(
            getattr(ALNSConfig, "ACTUAL_PRIORITY_ACTUAL_ABS_TOL", 0.25)
        )
        actual_rel_tol = float(
            getattr(ALNSConfig, "ACTUAL_PRIORITY_ACTUAL_REL_TOL", 0.002)
        )
        search_abs_tol = float(
            getattr(ALNSConfig, "ACTUAL_PRIORITY_SEARCH_ABS_TOL", 1.0)
        )
        search_rel_tol = float(
            getattr(ALNSConfig, "ACTUAL_PRIORITY_SEARCH_REL_TOL", 0.01)
        )

        actual_tol = max(actual_abs_tol, actual_rel_tol * abs(current_actual))
        search_tol = max(search_abs_tol, search_rel_tol * abs(current_search))

        fleet_hard_enabled = bool(
            getattr(ALNSConfig, "ENABLE_LATE_FLEET_HARD_PRIORITY", True)
        )
        fleet_hard_start = float(
            getattr(ALNSConfig, "LATE_FLEET_HARD_PRIORITY_START_PROGRESS", 0.60)
        )
        if fleet_hard_enabled and progress >= fleet_hard_start:
            cur_veh = self._count_used_vehicles(current_state)
            cand_veh = self._count_used_vehicles(candidate_state)
            max_actual_worse = float(
                getattr(ALNSConfig, "LATE_FLEET_HARD_PRIORITY_MAX_ACTUAL_WORSE", 0.75)
            )
            max_actual_rel_worse = float(
                getattr(ALNSConfig, "LATE_FLEET_HARD_PRIORITY_MAX_ACTUAL_REL_WORSE", 0.01)
            )
            allowed_actual_worse = max(
                max_actual_worse,
                max_actual_rel_worse * abs(current_actual),
            )

            if cand_veh < cur_veh:
                # 后期对更少车辆给予硬偏好：允许少量 actual 波动，但仍约束 search 不大幅变差。
                return (
                    candidate_actual <= current_actual + allowed_actual_worse
                    and candidate_search <= current_search + 2.0 * search_tol
                )
            if cand_veh > cur_veh:
                return candidate_actual + actual_tol < current_actual

        if candidate_actual + actual_tol < current_actual:
            return True
        if abs(candidate_actual - current_actual) <= actual_tol:
            return candidate_search <= current_search + search_tol
        return False

    def __call__(self, rng, best_state, current_state, candidate_state) -> bool:
        """
        被调用执行接受判定:
        1) 增加调用计数
        2) 通过 candidate_state.validate() 判断可行性及获取违约信息（兼容异常）
        3) 依据 violations 计算 penalty
        4) 用 _CandidateProxy 包装 candidate_state，使 objective() 返回值带上惩罚
        5) 委托给基础接收器进行接受判定（兼容多种调用签名）
        6) 记录接受统计并返回结果
        """
        self._calls += 1

        # 尝试获取可行性与违约详情; 若 validate 抛异常则认为可行且无违约
        try:
            feasible, violations = candidate_state.validate()
        except Exception:
            feasible, violations = True, {}

        penalty = 0.0
        if not feasible:
            # 关注 unmet_demand 条目（如果存在）
            unmet = violations.get("unmet_demand", [])
            if isinstance(unmet, list) and unmet:
                # 尝试解析每个违约项的 'demand' 与 'shipped' 字段以计算未满足总量
                unmet_total = 0.0
                for it in unmet:
                    try:
                        d = float(it.get("demand", 0))
                        s = float(it.get("shipped", 0))
                        unmet_total += max(0.0, d - s)
                    except Exception:
                        # 若无法解析具体数值, 退化为按条目计数
                        unmet_total += 1.0
                penalty = self.penalty_factor * unmet_total
            else:
                # 兜底策略: 若 unmet 不是非空列表, 则按条目数量或默认 1.0 计惩罚
                penalty = self.penalty_factor * (
                    len(unmet) if isinstance(unmet, list) else 1.0
                )

        # 记录用于统计的惩罚累计值
        self._total_penalty += float(penalty)

        progress = self._resolve_progress(current_state, candidate_state, best_state)
        actual_priority_start = float(
            getattr(ALNSConfig, "ACTUAL_PRIORITY_START_PROGRESS", 0.7)
        )

        if feasible and progress >= actual_priority_start:
            accepted = self._actual_priority_accept(
                current_state,
                candidate_state,
                progress,
            )
            if accepted:
                self._accepted += 1
            return accepted

        # 使用代理对象包装候选解，使其 objective() 返回 base + penalty
        proxy = _CandidateProxy(candidate_state, penalty)

        # Delegate to base acceptor (most acceptors follow signature used in tests)
        # 兼容多种可能的基础接收器签名: 首先尝试常见四参数签名, 若 TypeError 则尝试回退签名
        try:
            accepted = bool(self.base(rng, best_state, current_state, proxy))
        except TypeError:
            # 某些实现可能使用 (best_state, current_state, candidate_state) 之类的签名
            try:
                accepted = bool(self.base(best_state, current_state, proxy))
            except Exception:
                # 若仍然失败, 将接受视为 False（保守策略）
                accepted = False
        except Exception:
            # 其它异常统一捕获并视为不接受
            accepted = False

        if accepted:
            self._accepted += 1
        return accepted

    def get_statistics(self) -> Dict[str, float]:
        """
        返回统计信息摘要:
          - calls: 总调用次数
          - accepted: 被接受次数
          - accept_rate: 接受率
          - avg_penalty: 平均每次调用应用的惩罚
          - total_penalty: 累计惩罚
        该信息可用于调试、调参或记录日志。
        """
        calls = self._calls
        accepted = self._accepted
        accept_rate = (accepted / calls) if calls else 0.0
        avg_penalty = (self._total_penalty / calls) if calls else 0.0
        return {
            "calls": calls,
            "accepted": accepted,
            "accept_rate": float(accept_rate),
            "avg_penalty": float(avg_penalty),
            "total_penalty": float(self._total_penalty),
        }
