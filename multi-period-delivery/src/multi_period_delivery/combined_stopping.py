"""
combined_stopping
=================

模块定位
    封装可组合的停止准则 `CombinedStoppingCriterion`，用于在 ALNS/其他迭代式启发式中
    监控多个停止条件。支持两种使用方式：
    - 标准 OR 组合：传入若干单一准则，任一触发即停止。
    - 逻辑表达式组合：用 AND/OR 构造任意嵌套表达式，如 (A AND B) OR C。

依赖 (来自 alns.stop)
    - MaxRuntime
    - MaxIterations
    - NoImprovement
    以上对象均实现可调用协议：criterion(rng, best_state, current_state) -> bool

核心特性
    - 接受任意数量的单一准则 ( *criteria )；在未提供表达式时按 OR 逻辑评估。
    - expr 参数支持逻辑表达式树：('AND', a, b, ...) / ('OR', x, y, ...)，叶子为可调用准则。
    - 提供 AND(*nodes)/OR(*nodes) 辅助构造器以便直观地构建表达式。
    - 首次调用时开始计时，内部记录 _start_time；触发时输出触发者与耗时。
    - get_status 为“安全版”，仅返回 elapsed_time / triggered / criteria_types，避免读取第三方私有字段。

设计注意
    1) 表达式优先：若提供 expr，则优先评估表达式树；否则回退为对 *criteria 的 OR 评估。
    2) 触发记录：在表达式求值过程中，一旦某叶子返回 True，会记录其类型名到 triggered。
    3) 线程安全：当前未做并发保护，多线程下请在外部加锁或串行化访问。
    4) 日志输出：默认使用 print；如需统一日志体系可替换为 logging 或注入回调。
    5) 可扩展性：如需更多运算符 (NOT/XOR) 或回调 on_stop，可在现有结构上扩展。

使用示例
    1) 标准 OR 组合：
        stop = create_standard_combined_criterion(max_iterations=500, max_runtime=600, max_no_improvement=120)
        while not stop(rng, best_state, cur_state):
            ...
        print(stop.get_status())

    2) 逻辑表达式：(MaxIterations AND NoImprovement) OR MaxRuntime
        from alns.stop import MaxIterations, NoImprovement, MaxRuntime
        from multi_period_delivery.combined_stopping import (
            CombinedStoppingCriterion,
            AND,
            OR,
        )

        A = MaxIterations(500)
        B = NoImprovement(120)
        C = MaxRuntime(600)

        expr = OR(AND(A, B), C)
        stop = CombinedStoppingCriterion(expr=expr)

兼容性
    - 旧用法保持不变：不提供 expr 时，等价于对 *criteria 做 OR 运算。
    - 对外 API：CombinedStoppingCriterion(*criteria, expr=None), add_criterion, __call__, get_status
    - 工厂函数：create_standard_combined_criterion 提供常用“迭代+时间[+无改进]”组合
"""

# =========================
# 标准库
# =========================
import time
from typing import Any, Dict, List, Optional

# =========================
# 第三方类型
# =========================
from numpy.random import Generator

# =========================
# 外部库 (ALNS)
# =========================
from alns.State import State
from alns.stop import MaxRuntime, MaxIterations, NoImprovement


class CombinedStoppingCriterion:
    """
    组合停止准则：
        - 默认：对传入的单一停止准则执行 OR 逻辑
        - 扩展：支持通过 expr 传入逻辑表达式树 (AND/OR 嵌套)，如 ('OR', ('AND', A, B), C)
    """

    def __init__(self, *criteria: Any, expr: Any = None):
        """
        参数
        ----
        *criteria :
            任意数量的停止准则对象。典型包括：
              - MaxRuntime(max_runtime=...)
              - MaxIterations(max_iterations=...)
              - NoImprovement(max_iterations=...)
            只要对象实现 __call__(rng, best, current)->bool 即可。
        expr :
            逻辑表达式树，可为:
              - ('AND', a, b, ...) / ('OR', x, y, ...)
              - 列表形式 ['AND', a, b] 亦可
              - 叶子节点为可调用的单一准则对象
            若为 None，则对 *criteria 采用 OR 逻辑。
        """
        if not criteria and expr is None:
            raise ValueError("至少需要提供一个停止准则或表达式 (criteria/expr 皆为空)")

        self.criteria: List[Any] = list(criteria)
        self.expr: Any = expr
        self._start_time: Optional[float] = None  # 首次调用开始计时
        self._triggered: Optional[str] = None  # 记录触发的准则名称

    # ------------------------------------------------------------------
    # 扩展：动态添加准则
    # ------------------------------------------------------------------
    def add_criterion(self, criterion: Any) -> None:
        """追加一个新的停止准则"""
        self.criteria.append(criterion)

    def get_max_iterations(self) -> Optional[int]:
        for criterion in self.criteria:
            if isinstance(criterion, MaxIterations):
                return int(getattr(criterion, "max_iterations", 0))
        return None

    def set_max_iterations(self, max_iterations: int) -> bool:
        updated = False
        target = int(max_iterations)
        for criterion in self.criteria:
            if isinstance(criterion, MaxIterations):
                try:
                    # alns.stop.MaxIterations exposes a read-only property
                    # `max_iterations`; the effective mutable field is
                    # `_max_iterations`.
                    if hasattr(criterion, "_max_iterations"):
                        setattr(criterion, "_max_iterations", target)
                    else:
                        setattr(criterion, "max_iterations", target)
                    updated = True
                except Exception:
                    pass
        return updated

    def get_max_runtime(self) -> Optional[float]:
        for criterion in self.criteria:
            if isinstance(criterion, MaxRuntime):
                return float(getattr(criterion, "max_runtime", 0.0))
        return None

    def _eval_expr(
        self, node: Any, rng: Generator, best: State, current: State
    ) -> bool:
        """
        递归求值逻辑表达式:
            - 叶子为单一准则 (可调用) → 返回其结果
            - ('AND', a, b, ...) → 所有子节点为 True
            - ('OR', x, y, ...)  → 任一子节点为 True
        """
        # 叶子：单一准则
        if callable(getattr(node, "__call__", None)) and not isinstance(
            node, (tuple, list)
        ):
            res = bool(node(rng, best, current))
            if res:
                self._triggered = type(node).__name__
            return res

        # 复合：列表/元组 [op, *args]
        if isinstance(node, (tuple, list)) and node:
            op = str(node[0]).upper()
            args = node[1:]
            if op == "AND":
                for child in args:
                    if not self._eval_expr(child, rng, best, current):
                        return False
                return True
            if op == "OR":
                for child in args:
                    if self._eval_expr(child, rng, best, current):
                        return True
                return False

        return False

    # ------------------------------------------------------------------
    # 主调用：检查是否应停止
    # ------------------------------------------------------------------
    def __call__(self, rng: Generator, best: State, current: State) -> bool:
        """
        执行停止条件检查。
        返回
        ----
        bool :
            任一准则满足则 True；否则 False
        """
        # 初始化全局计时
        if self._start_time is None:
            self._start_time = time.perf_counter()

        # 使用表达式树 (若提供)
        if self.expr is not None:
            try:
                if self._eval_expr(self.expr, rng, best, current):
                    elapsed = time.perf_counter() - self._start_time
                    print(
                        f"[STOP] 逻辑表达式触发停止 (triggered={self._triggered}, elapsed={elapsed:.2f}s)"
                    )
                    return True
            except Exception as e:
                print(f"[WARNING] 评估逻辑表达式失败: {e}")
                return False

        # 否则：回退为对 criteria 的 OR 评估
        for criterion in self.criteria:
            try:
                if criterion(rng, best, current):
                    self._triggered = type(criterion).__name__
                    elapsed = time.perf_counter() - self._start_time
                    print(f"[STOP] {self._triggered} 触发停止 (elapsed={elapsed:.2f}s)")
                    return True
            except Exception as e:
                print(f"[WARNING] 停止准则 {type(criterion).__name__} 评估失败: {e}")
                continue

        return False

    # ------------------------------------------------------------------
    # 状态快照
    # ------------------------------------------------------------------
    def get_status(self) -> Dict[str, Any]:
        """
        返回当前停止准则集合状态快照 (安全版)。
        仅提供：
            - 总耗时
            - 已触发准则名称 (若有)
            - 准则类型列表
        不再读取第三方准则的私有/内部字段 (如 _current_iteration, _start_runtime)，
        避免对外部库内部实现产生脆弱依赖。
        若需要详细进度，请在自定义准则中添加公开属性或显式 status() 方法。
        """
        elapsed_time = (
            time.perf_counter() - self._start_time if self._start_time else 0.0
        )
        return {
            "elapsed_time": elapsed_time,
            "triggered": self._triggered,
            "criteria_types": [type(c).__name__ for c in self.criteria],
            "max_iterations": self.get_max_iterations(),
            "max_runtime": self.get_max_runtime(),
        }


# ----------------------------------------------------------------------
# 逻辑表达式辅助构造器
# ----------------------------------------------------------------------
def AND(*args):
    """构造 AND 表达式节点: AND(a,b,...) -> ('AND', a, b, ...)"""
    return ("AND",) + args


def OR(*args):
    """构造 OR 表达式节点: OR(x,y,...) -> ('OR', x, y, ...)"""
    return ("OR",) + args


# ----------------------------------------------------------------------
# 工厂方法
# ----------------------------------------------------------------------
def create_standard_combined_criterion(
    max_iterations: int = 300,
    max_runtime: float = 900,
    max_no_improvement: Optional[int] = None,
) -> CombinedStoppingCriterion:
    """
    创建一组常用组合停止准则 (迭代 + 运行时 [+ 无改进])。

    参数
    ----
    max_iterations : int
        最大迭代次数
    max_runtime : float
        最大运行时间 (秒)
    max_no_improvement : int | None
        若提供：添加 NoImprovement 停止准则 (参数 = 无改进迭代上限)

    返回
    ----
    CombinedStoppingCriterion
        配置好的组合对象
    """
    criteria: List[Any] = [
        MaxIterations(max_iterations=max_iterations),
        MaxRuntime(max_runtime=max_runtime),
    ]
    if max_no_improvement is not None:
        criteria.append(NoImprovement(max_iterations=max_no_improvement))
    return CombinedStoppingCriterion(*criteria)


# ----------------------------------------------------------------------
# 简单命令行测试
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== 组合停止准则测试 ===")
    combined_stop = create_standard_combined_criterion(
        max_iterations=300,
        max_runtime=5,  # 缩短演示用
        max_no_improvement=100,
    )
    print(
        f"创建组合停止准则: {', '.join(type(c).__name__ for c in combined_stop.criteria)}"
    )
    # 这里只演示状态输出；不运行完整 ALNS 循环
    print("初始状态:", combined_stop.get_status())
