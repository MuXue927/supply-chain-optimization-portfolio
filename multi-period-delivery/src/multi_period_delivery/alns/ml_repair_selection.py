"""
Use the machine-learning operator selector to choose the best repair operator.
"""

import time
import math
from typing import Dict, Any, Optional
import numpy.random as rnd

from .alnsopt import SolutionState
from .alns_config import default_config as ALNSConfig
from .ml_operator_selector import (
    PARAMETERLESS_OPERATOR_NAMES,
    REPAIR_OPERATOR_NAMES,
)
from .repair_utils import compute_mixed_operator_reward

# Import the required repair operators
from .repair_operators import (
    greedy_repair,
    local_search_repair,
    inventory_balance_repair,
    smart_batch_repair,
    regret_based_repair,
    cross_period_consolidation_repair,
    sequential_repair,
    lookahead_repair,
    fleet_consolidation_repair,
)


# Helper: sanitize and align ParamTuner kwargs to operator signatures
def _sanitize_repair_params(op_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(params, dict) or not params:
        return {}
    allowed = {
        "greedy_repair": {"demand_weight", "stock_weight"},
        "local_search_repair": {"max_iter"},
        "inventory_balance_repair": set(),
        "smart_batch_repair": {"max_iter", "batch_size", "timeout"},
        "regret_based_repair": {"k", "topN", "time_limit"},
        "cross_period_consolidation_repair": set(),
        "sequential_repair": {"lookahead_horizon"},
        "lookahead_repair": {"lookahead_horizon"},
        "fleet_consolidation_repair": set(),
    }
    allow = allowed.get(op_name, set())
    return {k: v for k, v in params.items() if k in allow}


def _get_repair_params(state: SolutionState, op_name: str) -> Dict[str, Any]:
    """Read tuner parameters, skipping operators that have no parameters."""
    if state.param_tuner is None or op_name in PARAMETERLESS_OPERATOR_NAMES:
        return {}
    try:
        params = state.param_tuner.get_operator_params(op_name)
        return dict(params) if isinstance(params, dict) else {}
    except Exception:
        return {}


_REPAIR_OPERATOR_FUNCTIONS = {
    "greedy_repair": lambda s, r, **kwargs: greedy_repair(s, r, **kwargs),
    "cross_period_consolidation_repair": lambda s, r, **kwargs: (
        cross_period_consolidation_repair(s, r, **kwargs)
    ),
    "local_search_repair": lambda s, r, **kwargs: local_search_repair(
        s, r, **kwargs
    ),
    "inventory_balance_repair": lambda s, r, **kwargs: inventory_balance_repair(
        s, r, **kwargs
    ),
    "smart_batch_repair": lambda s, r, **kwargs: smart_batch_repair(s, r, **kwargs),
    "regret_based_repair": lambda s, r, **kwargs: regret_based_repair(
        s, r, **kwargs
    ),
    "fleet_consolidation_repair": lambda s, r, **kwargs: (
        fleet_consolidation_repair(s, r, **kwargs)
    ),
    "sequential_repair": lambda s, r, **kwargs: sequential_repair(s, r, **kwargs),
    "lookahead_repair": lambda s, r, **kwargs: lookahead_repair(s, r, **kwargs),
}

# Keep selection keys and one-hot indexes in registration order.
_REPAIR_OPERATORS = {
    name: _REPAIR_OPERATOR_FUNCTIONS[name] for name in REPAIR_OPERATOR_NAMES
}


def ml_based_repair(
    current: SolutionState, rng: rnd.Generator, min_sample_size: Optional[int] = None
) -> SolutionState:
    """
    Apply an ML-guided repair operator.

    The ML selector chooses the repair operator and parameters best suited to
    the current solution state. When training data is insufficient, choose a
    repair operator randomly.

    Args:
        current: Current solution state.
        rng: Random-number generator.
        min_sample_size: Minimum sample count. Below this threshold, select an
            operator randomly.

    Returns:
        The modified solution state.
    """
    t0 = time.time()
    print("[OPLOG] Starting ml_based_repair operator")
    state = current.copy()

    # If min_sample_size is not supplied, read the default from centralized
    # configuration for backward compatibility.
    if min_sample_size is None:
        min_sample_size = int(getattr(ALNSConfig, "ML_INITIAL_SAMPLE_SIZE", 20))
    else:
        min_sample_size = int(min_sample_size)

    takeover_min = int(
        getattr(ALNSConfig, "ML_TAKEOVER_MIN_SAMPLE_SIZE", min_sample_size)
    )
    if state.param_tuner is not None and bool(
        getattr(state.param_tuner, "small_instance", False)
    ):
        takeover_min = max(
            takeover_min,
            int(getattr(ALNSConfig, "ML_TAKEOVER_MIN_SAMPLE_SIZE_SMALL", takeover_min)),
        )
    effective_min_sample_size = max(min_sample_size, takeover_min)

    # Check whether the ML selector is available
    if not state.ml_selector:
        print("[OPLOG] ml_based_repair: ML selector unavailable; falling back to greedy repair")
        greedy_params = ALNSConfig.get_repair_params().get("greedy_repair", {})
        return greedy_repair(state, rng, **(greedy_params or {}))

    # Build the available repair-operator map from the shared registration
    # order so the selector and one-hot encoding cannot drift apart.
    repair_operators = _REPAIR_OPERATORS

    # Check whether training data is sufficient
    use_ml = False
    if hasattr(state.ml_selector, "should_use_ml"):
        use_ml = bool(
            state.ml_selector.should_use_ml(
                state,
                "repair",
                min_sample_size=effective_min_sample_size,
            )
        )

    if not use_ml:
        # Insufficient data: choose a repair operator randomly
        print(
            f"[OPLOG] ml_based_repair: using random fallback "
            f"({len(state.ml_selector.repair_features)} < {effective_min_sample_size})"
        )
        op_name = str(rng.choice(list(repair_operators.keys())))
        operator = repair_operators[op_name]

        # Retrieve parameters
        params = _get_repair_params(state, op_name)
        params = _sanitize_repair_params(op_name, params)

        # Execute the operator
        prev_obj = state.objective()
        new_state = operator(state, rng, **params)
        new_obj, improvement, _ = compute_mixed_operator_reward(
            new_state,
            prev_obj,
            new_obj=new_state.objective(),
        )
        if new_obj is None or math.isnan(improvement):
            improvement = 0.0
        elif math.isinf(improvement):
            improvement = 1e6 if improvement > 0 else -1e6

        # Record performance using the pre-operation state features to avoid
        # leaking post-operation information.
        state.ml_selector.record_operator_performance(
            op_name, "repair", state, improvement
        )
        if hasattr(state.ml_selector, "update_op_feedback"):
            state.ml_selector.update_op_feedback("repair", improvement)

        print(
            f"[OPLOG] ml_based_repair: randomly selected {op_name}; "
            f"improvement {improvement:.4f}"
        )
        return new_state

    # Sufficient training data: use the ML selector to choose the best operator
    state.ml_selector.train_models()

    # Select the best operator
    op_name, params = state.ml_selector.select_best_operator(
        state, "repair", list(repair_operators.keys())
    )
    params = _sanitize_repair_params(op_name, params)

    operator = repair_operators[op_name]

    # Execute the operator
    prev_obj = state.objective()
    new_state = operator(state, rng, **params)
    new_obj, improvement, _ = compute_mixed_operator_reward(
        new_state,
        prev_obj,
        new_obj=new_state.objective(),
    )
    if new_obj is None or math.isnan(improvement):
        improvement = 0.0
    elif math.isinf(improvement):
        improvement = 1e6 if improvement > 0 else -1e6

    # Record performance using the pre-operation state features to avoid
    # leaking post-operation information.
    state.ml_selector.record_operator_performance(op_name, "repair", state, improvement)
    if hasattr(state.ml_selector, "update_op_feedback"):
        state.ml_selector.update_op_feedback("repair", improvement)

    print(
        f"[OPLOG] ml_based_repair: selected {op_name}; improvement "
        f"{improvement:.4f} ({time.time() - t0:.4f}s)"
    )
    return new_state
