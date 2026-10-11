"""
combined_stopping
=================

Module role
    Provides `CombinedStoppingCriterion`, a composable stopping criterion for
    ALNS and other iterative heuristics that monitors multiple stopping
    conditions. Two usage modes are supported:
    - Standard OR combination: pass several criteria; stop when any one fires.
    - Logical expression: compose arbitrary nested AND/OR expressions such as
      (A AND B) OR C.

Dependencies (from alns.stop)
    - MaxRuntime
    - MaxIterations
    - NoImprovement
    These objects follow the callable protocol:
    criterion(rng, best_state, current_state) -> bool.

Core features
    - Accept any number of individual criteria (*criteria) and evaluate them
      with OR semantics when no expression is provided.
    - The expr argument accepts a logical expression tree:
      ('AND', a, b, ...) / ('OR', x, y, ...), with callable criteria as leaves.
    - AND(*nodes) and OR(*nodes) helpers make expression construction explicit.
    - Start timing on the first call and record it in _start_time; report the
      triggering criterion and elapsed time when stopping.
    - get_status is a safe snapshot that returns only elapsed_time, triggered,
      and criteria_types, avoiding private fields from third-party objects.

Design notes
    1) Expression precedence: if expr is provided, evaluate the expression tree;
       otherwise fall back to OR evaluation over *criteria.
    2) Trigger tracking: whenever a leaf returns True during expression
       evaluation, record its type name in triggered.
    3) Thread safety: no concurrency protection is provided. Use external
       locking or serialize access when sharing across threads.
    4) Logging: print is used by default. Replace it with logging or a callback
       when a unified logging system is required.
    5) Extensibility: additional operators (NOT/XOR) or an on_stop callback can
       be added to the existing structure.

Usage examples
    1) Standard OR combination:
        stop = create_standard_combined_criterion(max_iterations=500, max_runtime=600, max_no_improvement=120)
        while not stop(rng, best_state, cur_state):
            ...
        print(stop.get_status())

    2) Logical expression: (MaxIterations AND NoImprovement) OR MaxRuntime
        from alns.stop import MaxIterations, NoImprovement, MaxRuntime
        from multi_period_delivery.alns.combined_stopping import (
            CombinedStoppingCriterion,
            AND,
            OR,
        )

        A = MaxIterations(500)
        B = NoImprovement(120)
        C = MaxRuntime(600)

        expr = OR(AND(A, B), C)
        stop = CombinedStoppingCriterion(expr=expr)

Compatibility
    - Existing behavior is unchanged: without expr, *criteria are combined
      with OR semantics.
    - Public API: CombinedStoppingCriterion(*criteria, expr=None),
      add_criterion, __call__, get_status.
    - Factory function: create_standard_combined_criterion builds the common
      iterations + runtime [+ no-improvement] combination.
"""

# =========================
# Standard library
# =========================
import time
from typing import Any, Dict, List, Optional

# =========================
# Third-party types
# =========================
from numpy.random import Generator

# =========================
# External library (ALNS)
# =========================
from alns.State import State
from alns.stop import MaxRuntime, MaxIterations, NoImprovement


class CombinedStoppingCriterion:
    """
    Combined stopping criterion:
        - Default: apply OR logic to the supplied individual criteria.
        - Extended: pass a nested AND/OR expression tree through expr, such as
          ('OR', ('AND', A, B), C).
    """

    def __init__(self, *criteria: Any, expr: Any = None):
        """
        Parameters
        ----------
        *criteria :
            Any number of stopping criterion objects. Typical examples:
              - MaxRuntime(max_runtime=...)
              - MaxIterations(max_iterations=...)
              - NoImprovement(max_iterations=...)
            Any object implementing __call__(rng, best, current) -> bool is
            accepted.
        expr :
            Logical expression tree, one of:
              - ('AND', a, b, ...) / ('OR', x, y, ...)
              - list form ['AND', a, b] is also accepted
              - leaves are callable criterion objects
            If None, *criteria are evaluated with OR semantics.
        """
        if not criteria and expr is None:
            raise ValueError(
                "At least one stopping criterion or expression is required "
                "(criteria and expr are both empty)"
            )

        self.criteria: List[Any] = list(criteria)
        self.expr: Any = expr
        self._start_time: Optional[float] = None  # Start timing on first call.
        self._triggered: Optional[str] = None  # Name of the triggering criterion.

    # ------------------------------------------------------------------
    # Extension: dynamically add criteria.
    # ------------------------------------------------------------------
    def add_criterion(self, criterion: Any) -> None:
        """Append a new stopping criterion."""
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
        Recursively evaluate a logical expression:
            - A callable leaf returns its own result.
            - ('AND', a, b, ...) returns True when all children are True.
            - ('OR', x, y, ...) returns True when any child is True.
        """
        # Leaf: individual criterion.
        if callable(getattr(node, "__call__", None)) and not isinstance(
            node, (tuple, list)
        ):
            res = bool(node(rng, best, current))
            if res:
                self._triggered = type(node).__name__
            return res

        # Compound node: list/tuple [op, *args].
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
    # Main call: determine whether to stop.
    # ------------------------------------------------------------------
    def __call__(self, rng: Generator, best: State, current: State) -> bool:
        """
        Evaluate the stopping conditions.

        Returns
        -------
        bool :
            True when any criterion is satisfied; False otherwise.
        """
        # Initialize global timing.
        if self._start_time is None:
            self._start_time = time.perf_counter()

        # Use the expression tree when provided.
        if self.expr is not None:
            try:
                if self._eval_expr(self.expr, rng, best, current):
                    elapsed = time.perf_counter() - self._start_time
                    print(
                        f"[STOP] logical expression triggered stop "
                        f"(triggered={self._triggered}, elapsed={elapsed:.2f}s)"
                    )
                    return True
            except Exception as e:
                print(f"[WARNING] Failed to evaluate logical expression: {e}")
                return False

        # Otherwise, fall back to OR evaluation over criteria.
        for criterion in self.criteria:
            try:
                if criterion(rng, best, current):
                    self._triggered = type(criterion).__name__
                    elapsed = time.perf_counter() - self._start_time
                    print(
                        f"[STOP] {self._triggered} triggered stop "
                        f"(elapsed={elapsed:.2f}s)"
                    )
                    return True
            except Exception as e:
                print(
                    f"[WARNING] Stopping criterion {type(criterion).__name__} "
                    f"failed during evaluation: {e}"
                )
                continue

        return False

    # ------------------------------------------------------------------
    # Status snapshot.
    # ------------------------------------------------------------------
    def get_status(self) -> Dict[str, Any]:
        """
        Return a safe status snapshot for the stopping criterion collection.
        The snapshot contains only:
            - total elapsed time
            - the triggering criterion name, if any
            - the list of criterion types
        It does not read private/internal fields from third-party criteria,
        such as _current_iteration or _start_runtime, which avoids brittle
        dependencies on external implementation details. For detailed progress,
        add public attributes or an explicit status() method to custom criteria.
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
# Logical expression helper constructors.
# ----------------------------------------------------------------------
def AND(*args):
    """Build an AND expression node: AND(a, b, ...) -> ('AND', a, b, ...)."""
    return ("AND",) + args


def OR(*args):
    """Build an OR expression node: OR(x, y, ...) -> ('OR', x, y, ...)."""
    return ("OR",) + args


# ----------------------------------------------------------------------
# Factory method.
# ----------------------------------------------------------------------
def create_standard_combined_criterion(
    max_iterations: int = 300,
    max_runtime: float = 900,
    max_no_improvement: Optional[int] = None,
) -> CombinedStoppingCriterion:
    """
    Create a common combined stopping criterion
    (iterations + runtime [+ no improvement]).

    Parameters
    ----------
    max_iterations : int
        Maximum number of iterations.
    max_runtime : float
        Maximum runtime in seconds.
    max_no_improvement : int | None
        If provided, add a NoImprovement criterion whose parameter is the
        no-improvement iteration limit.

    Returns
    -------
    CombinedStoppingCriterion
        Configured combined criterion.
    """
    criteria: List[Any] = [
        MaxIterations(max_iterations=max_iterations),
        MaxRuntime(max_runtime=max_runtime),
    ]
    if max_no_improvement is not None:
        criteria.append(NoImprovement(max_iterations=max_no_improvement))
    return CombinedStoppingCriterion(*criteria)


# ----------------------------------------------------------------------
# Simple command-line test.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Combined stopping criterion test ===")
    combined_stop = create_standard_combined_criterion(
        max_iterations=300,
        max_runtime=5,  # Short runtime for demonstration.
        max_no_improvement=100,
    )
    print(
        f"Created combined stopping criterion: "
        f"{', '.join(type(c).__name__ for c in combined_stop.criteria)}"
    )
    # Only demonstrate status output here; do not run a full ALNS loop.
    print("Initial status:", combined_stop.get_status())
