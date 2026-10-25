"""
ml_destroy_selection
====================

Purpose
    Provide the adaptive destroy-operator selection entry point backed by
    MLOperatorSelector. It reads current-state features and historical
    (feature, operator, improvement) samples, predicts the expected improvement
    of each destroy operator, and selects one automatically. If training data
    is insufficient, it falls back to random selection.

Core flow
    1. Copy the current solution so the caller's state is not modified.
    2. If no ML selector is attached (state.ml_selector is None), fall back to
       random_removal.
    3. If destroy feature count < min_sample_size, choose a random operator and
       record (feature, improvement).
    4. Otherwise:
         a. Call ml_selector.train_models(); internal interval and sample
            thresholds decide whether training occurs.
         b. Call select_best_operator(..., op_type='destroy').
         c. Execute the selected operator and evaluate improvement.
         d. Record the improvement sample and tune operator parameters when a
            ParamAutoTuner is present.
    5. Return the updated solution copy.

Operator set, kept consistent with MLOperatorSelector.destroy_op_map
    - random_removal
    - shaw_removal
    - periodic_shaw_removal
    - worst_removal
    - surplus_inventory_removal
    - path_removal
    - cross_period_dealer_removal
    - period_range_removal
    - infeasible_removal
    - inventory_level_removal

Performance and safety
    - Operate only on the state.copy() result.
    - objective() relies on caching and incremental inventory updates for
      efficiency.
    - Logging is lightweight and can later be connected to a global log switch.

Optional future enhancements
    - Add exception handling and fallback behavior for failed or skipped
      operators.
    - Expose min_sample_size through ALNSConfig for easier tuning across data
      scales.
    - Record operator runtime as a feature for a cost-aware model.
"""

# =========================
# Standard library
# =========================
import math
import time
from typing import Any

# =========================
# Third-party libraries
# =========================
import numpy.random as rnd

from .alns_config import default_config as ALNSConfig

# =========================
# Internal project dependencies
# =========================
from .alnsopt import SolutionState

# Destroy operator collection
from .destroy_operators import (
    cross_period_dealer_removal,
    infeasible_removal,
    inventory_level_removal,
    path_removal,
    period_range_removal,
    periodic_shaw_removal,
    random_removal,
    shaw_removal,
    surplus_inventory_removal,
    worst_removal,
)
from .errors import STATE_OPERATION_ERRORS
from .ml_operator_selector import (
    DESTROY_OPERATOR_NAMES,
    PARAMETERLESS_OPERATOR_NAMES,
    MLOperatorSelector,
)
from .repair_utils import compute_mixed_operator_reward

# ---------------------------------------------------------------------
# Internal: operator map construction. Lambda wrappers preserve the call
# signature.
# ---------------------------------------------------------------------
_DESTROY_OPERATOR_FUNCTIONS: dict[str, callable] = {
    "random_removal": lambda s, r, **kw: random_removal(s, r, **kw),
    "shaw_removal": lambda s, r, **kw: shaw_removal(s, r, **kw),
    "periodic_shaw_removal": lambda s, r, **kw: periodic_shaw_removal(s, r, **kw),
    "worst_removal": lambda s, r, **kw: worst_removal(s, r, **kw),
    "surplus_inventory_removal": lambda s, r, **kw: surplus_inventory_removal(
        s, r, **kw
    ),
    "path_removal": lambda s, r, **kw: path_removal(s, r, **kw),
    "cross_period_dealer_removal": lambda s, r, **kw: cross_period_dealer_removal(
        s, r, **kw
    ),
    "period_range_removal": lambda s, r, **kw: period_range_removal(s, r, **kw),
    "infeasible_removal": lambda s, r, **kw: infeasible_removal(s, r, **kw),
    "inventory_level_removal": lambda s, r, **kw: inventory_level_removal(
        s, r, **kw
    ),
}

# Derive the public mapping from the shared registration-order tuple. This
# makes a missing or extra operator visible at import time rather than silently
# changing the one-hot feature layout.
_DESTROY_OPERATORS: dict[str, callable] = {
    name: _DESTROY_OPERATOR_FUNCTIONS[name] for name in DESTROY_OPERATOR_NAMES
}


# Helper: sanitize and align ParamTuner kwargs to operator signatures
def _sanitize_destroy_params(op_name: str, params: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(params, dict) or not params:
        return {}
    allowed = {
        "random_removal": {"degree"},
        "shaw_removal": {"degree"},
        "periodic_shaw_removal": {"degree", "alpha", "beta", "gamma", "k_cluster"},
        "path_removal": {"degree"},
        "cross_period_dealer_removal": {"degree"},
        "period_range_removal": {"degree", "window_ratio"},
        "inventory_level_removal": {"degree"},
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


def _get_destroy_params(state: SolutionState, op_name: str) -> dict[str, Any]:
    """Read tuner parameters, skipping operators that have no parameters."""
    if state.param_tuner is None or op_name in PARAMETERLESS_OPERATOR_NAMES:
        return {}
    try:
        params = state.param_tuner.get_operator_params(op_name)
        return dict(params) if isinstance(params, dict) else {}
    except STATE_OPERATION_ERRORS:
        return {}


# ---------------------------------------------------------------------
# Public function: adaptive destroy-operator selection and execution
# ---------------------------------------------------------------------
def ml_based_destroy(
    current: SolutionState,
    rng: rnd.Generator,
    min_sample_size: int | None = None,
) -> SolutionState:
    """
    Select and execute a destroy operator with ML prediction and fallback
    behavior.

    Args:
        current: Current solution. The original object is not modified in place.
        rng: Random source.
        min_sample_size: Minimum training-sample threshold for the destroy
            model. Below this threshold, the function explores randomly.

    Returns:
        A new solution copy after applying the destroy operator.

    Flow:
        1. If no ML selector is configured, use random_removal.
        2. If destroy samples < min_sample_size, explore randomly and record
           the sample.
        3. Otherwise, use the ML model and epsilon-greedy operator selection.
        4. Execute the operator, compute improvement, record the sample, and
           tune parameters.

    Improvement definition:
        improvement = prev_obj - new_obj
        Values above 0 indicate improvement in the minimized objective; values
        at or below 0 indicate no improvement or degradation.

    Numerical stability:
        - NaN -> 0
        - +inf -> 1e6
        - -inf -> -1e6
    """
    t_start = time.time()
    state = current.copy()

    print("[OPLOG][Destroy-ML] Starting adaptive destroy selection")
    ml_selector: MLOperatorSelector | None = state.ml_selector  # May be None.

    # If min_sample_size is not supplied, read the default from centralized
    # configuration for backward compatibility.
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

    # Fallback: no ML selector
    if ml_selector is None:
        print("[OPLOG][Destroy-ML] No ML selector attached; falling back to random_removal")
        return random_removal(state, rng)

    # Train before asking whether to take over: ``should_use_ml`` requires a
    # trained model, so training after the check made takeover unreachable.
    ml_selector.train_models()

    # Fallback: insufficient samples
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
        op_name = str(rng.choice(list(_DESTROY_OPERATORS.keys())))
        params = _get_destroy_params(state, op_name)
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
            f"[OPLOG][Destroy-ML] Using random fallback "
            f"(samples={len(ml_selector.destroy_features)}/{effective_min_sample_size}); "
            f"selected {op_name}; improvement {improvement:.4f}; "
            f"elapsed {time.time() - t_start:.4f}s"
        )
        return new_state

    # ML selection
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
        f"[OPLOG][Destroy-ML] Selected {op_name}; improvement {improvement:.4f}; "
        f"elapsed {time.time() - t_start:.4f}s"
    )
    return new_state
