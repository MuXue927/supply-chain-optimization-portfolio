"""
Module: destroy_utils.py

Responsibilities:
- Extract helper logic from destroy_operators.post_destroy into standalone
  functions, following the single-responsibility principle and improving
  reuse and testability.
- Provide utility functions related to destroy operations:
  * Safe conversion from numpy values to native Python types: to_native
  * Cargo summary for removed vehicles: summarize_removed_cargo
  * Batch inventory restoration: apply_batch_update
  * Improvement calculation and reporting: compute_improvement_and_report
  * Standardized log payload construction: build_opjson_log

Design conventions:
- Functions should remain side-effect free and only read or return values,
  unless their documentation explicitly states that they modify state.
- Returned data structures should use native Python types, or data preprocessed
  by to_native, so they can be serialized for logging and JSON output.
- If a caller modifies state, such as changing state.vehicles or state.s_ikt
  after removing vehicles, the caller is responsible for invoking the
  corresponding batch update or compute_inventory routine to preserve
  consistency.
- To avoid circular dependencies, some functions import
  alnsopt.batch_update_inventory lazily.

Exception handling strategy:
- These utilities support the main algorithm flow and should not interrupt it
  because of a single exception. Most exceptions are caught and converted into
  safe fallback values or flags so the caller can recover or record the issue.
"""

import logging
import random
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np

# Numeric coercion lives in one place so the three copies cannot drift apart; the
# shared implementation also refuses to truncate a multi-element array.
from .coercion import coerce_int as _coerce_int
from .repair_utils import compute_mixed_operator_reward
from .snapshot import restore_state_snapshot
from .swallowed import record_swallowed

logger = logging.getLogger(__name__)

# Use TYPE_CHECKING to avoid importing type dependencies at runtime and
# creating a circular import.
if TYPE_CHECKING:
    from .alnsopt import SolutionState  # pragma: no cover
else:
    SolutionState = object


# Numeric coercion lives in one place so the three copies cannot drift apart; the
# shared implementation also refuses to truncate a multi-element array.
def to_native(v: Any):
    """Convert numpy scalars and arrays to native Python types for logging.

    Design details:
    - Convert common numpy scalars and arrays to native Python types
      (int/float/bool/list/None) for safe serialization and logging.
    - Error tolerance:
      * Return None unchanged.
      * Convert numpy scalar types (np.integer, np.floating, np.bool_) to
        int/float/bool respectively.
      * Convert numpy.ndarray values to lists, recursively expanding nested
        arrays.
      * For unrecognized types, fall back progressively through
        int -> float -> str -> None without raising an exception.
    - Design considerations:
      * Logging and JSON serialization usually cannot handle numpy types
        directly. This function provides a safe boundary for logging calls.
      * Preserve idempotence where possible so equal inputs produce the same
        serializable representation.
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
        # Fall back progressively to guarantee a native or stringifiable value.
        try:
            return int(v)
        except (TypeError, ValueError):
            try:
                return float(v)
            except (TypeError, ValueError):
                try:
                    return str(v)
                except Exception:
                    return None


def summarize_removed_cargo(removed) -> dict[tuple[Any, Any, Any], int]:
    """
    Summarize cargo quantities from a list of removed vehicles.
    Returns dict keyed by (plant, sku_id, day) -> qty

    Purpose and implementation details:
    - Aggregate removed vehicle cargo into a quantity mapping keyed by
      (plant, sku, day), typically for logging or batch inventory restoration.
    - Error tolerance:
      * Use getattr for safe access to vehicle attributes, supporting
        alternative Vehicle implementations.
      * Convert keys and values with to_native to avoid serialization issues
        caused by numpy types.
      * Use a best-effort strategy for unusual qty values: first try int(qty),
        then np.array(qty).flat[0], and skip the entry if both fail.
      * A failure to parse one entry should not affect the other entries.
    - Return:
      * A plain dict keyed by (plant, sku_id, day) with accumulated integer
        quantities.
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
            # Skip records that fail to parse so one bad entry does not affect
            # the overall summary.
            continue
    return dict(summary)


def apply_batch_update(state, removed, skip_batch_update: bool):
    """
    Try to call batch_update_inventory(state, removed, op_type='destroy').
    Returns (inventory_updates dict or None, batch_conflict flag).

    Behavior and edge cases:
    - Attempt to call alnsopt.batch_update_inventory after a destroy operation
      to restore inventory to state. Destroy operations add inventory back.
    - Keep inventory restoration decoupled from the destroy operation so it can
      be reused in different contexts.
    - Behavior:
      * If skip_batch_update is True or removed is empty, return
        (None, False), indicating that no batch update was performed and no
        conflict occurred.
      * Import alnsopt.batch_update_inventory lazily to avoid circular
        dependencies. If the import and execution succeed, return the result,
        which may be {}.
      * If the import or execution raises an exception, return (None, True) to
        indicate a conflict or exception. The caller can use this flag to roll
        back or record the issue.
    - Error handling:
      * This function does not raise exceptions. It reports errors to the
        caller through the batch_conflict flag so the main flow remains stable.
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


def partial_updates_overlap_removed_cargo(
    state, partial_updates: dict | None, removed
) -> bool:
    """
    Return whether partial inventory updates touch cargo restored by the batch
    update for fully removed vehicles.

    Destroy operators are not uniform: some include emptied vehicles in both
    ``partial_updates`` and ``removed``, while others use partial updates only
    for vehicles that remain. When the keys overlap, the incremental delta
    cannot distinguish duplicate restoration from a separate partial rollback.
    The caller can rebuild inventory from the already-mutated vehicles instead.
    """
    if not partial_updates or not removed:
        return False

    partial_keys = set()
    for key in partial_updates:
        try:
            plant, sku_id, day = key
        except (TypeError, ValueError):
            continue
        partial_keys.add((plant, sku_id, int(day)))

    if not partial_keys:
        return False

    try:
        horizons = int(state.data.horizons)
    except Exception:
        return False

    for (plant, sku_id, shipped_day), raw_qty in summarize_removed_cargo(
        removed
    ).items():
        qty = _coerce_int(raw_qty, default=0)
        if qty <= 0:
            continue
        try:
            start_day = int(shipped_day)
        except (TypeError, ValueError):
            continue
        for day in range(start_day, horizons + 1):
            if (plant, sku_id, day) in partial_keys:
                return True
    return False


def apply_partial_inventory_updates(state, partial_updates: dict | None):
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
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid partial inventory key: {key!r}") from exc
        try:
            delta = int(raw_delta)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid partial inventory delta for {key!r}: {raw_delta!r}"
            ) from exc
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


_DESTROY_HARD_VIOLATION_KEYS = (
    "veh_over_load",
    "negative_inventory",
    "plant_inv_exceed",
    "cargo_day_mismatch",
    "check_error",
    "validation_error",
)


def _has_hard_destroy_violation(violations) -> bool:
    """Return whether a destroy validation result contains a hard failure."""
    if not isinstance(violations, dict):
        return True
    return any(bool(violations.get(key)) for key in _DESTROY_HARD_VIOLATION_KEYS)


def compute_improvement_and_report(
    state: SolutionState, prev_obj: float | None, op_name: str
):
    """
    Compute objective improvement robustly and report to param_tuner if available.
    Returns (new_obj_or_None, improvement_float, success_flag_bool)

    Purpose and key logic:
    - Compute the objective improvement after an operator runs and, when
      available, report performance data to the parameter tuner for online
      tuning and learning.
    - If prev_obj is None, there is no baseline, so return
      (None, 0.0, False).
    - Use state.objective() to compute new_obj and calculate raw improvement
      as prev_obj - new_obj only when both values are finite.
    - Use np.clip to limit the improvement range and prevent unusually large
      values, such as overflows or outliers, from affecting statistics.
    - The success flag is improvement > 0.
    - If state.param_tuner exists and provides update_operator_performance,
      report the improvement and success flag. Reporting is exception-tolerant
      so it does not interrupt the main flow.
    - Return:
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
    post_destroy_certified: bool,
    post_destroy_error,
    removed_cargo_summary: dict,
    inventory_updates: dict | None,
    extra: dict | None,
):
    """
    Construct the standardized OPJSON log payload (as a plain dict).

    Field and serialization details:
    - Build a standardized destroy-operation log payload for centralized
      recording, analysis, and post-processing.
    - Field details:
      * Keep only the first 200 entries in removed_cargo_summary to control
        log size and avoid oversized log records.
      * Convert inventory_updates with to_native to avoid serialization
        failures from numpy data.
      * Convert all potentially numpy-typed fields
        (objective_before/after/improvement/elapsed) with to_native.
    - Return:
      * A plain dict suitable for logging or JSON serialization.
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
        "post_destroy_certified": bool(post_destroy_certified),
        "post_destroy_error": post_destroy_error,
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
# Helper functions extracted from destroy_operators.py.
# ---------------------------------------------------------------------


def ensure_pyint(x):
    """Return ``x`` as a single Python int, or 0 when it has no single value."""
    try:
        if isinstance(x, np.ndarray):
            if x.size == 0:
                return 0
            return int(np.atleast_1d(x).flatten()[0])
        if isinstance(x, np.generic):
            return int(x.item())
        if isinstance(x, (list, tuple)):
            return int(x[0]) if x else 0
        return int(x)
    except (TypeError, ValueError, IndexError):
        return 0


def ensure_pylist(x):
    """
    Return ``x`` as a list of ints.

    A 0-d array (``np.array(9)``) yields ``[9]``, matching
    :func:`ensure_pyint`, which reports ``9`` for the same input. It previously
    returned ``[]`` - the two helpers disagreed about what a 0-d array means.
    """
    try:
        if isinstance(x, np.ndarray):
            if x.ndim == 0:
                return [int(x.item())]
            return [int(i) for i in x.tolist()]
        if isinstance(x, (list, tuple)):
            return [int(i) for i in x]
        return [int(x)]
    except (TypeError, ValueError, IndexError):
        return []


def adjust_exploration(state: SolutionState):
    """
    Record which iteration the current operator invocation belongs to so
    ParamAutoTuner can identify the current phase:
      - exploration: broad exploration in the early phase, with more frequent
        parameter adjustment
      - exploitation: use discovered structures or templates in the middle
        phase
      - refinement: converge and refine in the late phase, with shrinking
        parameter perturbations
    Notes:
      tracker maintains global statistics, including total_iterations and
      max_iterations.
      param_tuner.set_iteration adjusts exploration_rate internally.
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
        # Fail gracefully so the main flow is not affected.
        pass


def get_adaptive_parameters(op_name: str, param_tuner):
    """
    Return the current dynamic tuning parameters for an operator, or an empty
    dictionary when ParamAutoTuner is unavailable or raises an exception.
    :param op_name: operator name, such as 'random_removal'
    :return: dict
    """
    if not param_tuner:
        return {}
    try:
        params = param_tuner.get_operator_params(op_name)
        return params if params else {}
    except Exception as e:
        # Use logger.debug instead of print and apply sampling to reduce
        # hot-path overhead.
        try:
            try:
                from .alns_config import default_config as ALNSConfig

                sampling = getattr(ALNSConfig, "OPJSON_SAMPLING_RATE", 1)
            except Exception:
                sampling = 1
            if sampling <= 1 or random.random() < 1.0 / max(1, int(sampling)):
                logger.debug(f"get_adaptive_parameters({op_name}) exception: {e}")
        except Exception:
            pass
        return {}


def resolve_degree(op_name: str, state, degree):
    """
    Resolve and clamp the degree parameter for a destroy operator.
    Priority: explicit degree -> ParamAutoTuner.get_operator_params(op_name)
    -> ALNSConfig default -> fallback default.
    Integrate adaptive_degree when available to adjust removal strength
    dynamically based on iteration progress.
    Output: a float in [0, 1].
    """
    # Load defaults from the centralized configuration.
    try:
        from .alns_config import default_config as ALNSConfig

        op_defaults = ALNSConfig.get_operator_default(op_name).get("params", {}) or {}
    except Exception:
        op_defaults = {}

    # Fallback defaults.
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
    except (TypeError, ValueError):
        try:
            val = float(base)
        except (TypeError, ValueError):
            val = 0.25

    # Integrate adaptive_degree when param_tuner is available.
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
                "inventory_level_removal",
            }:
                adaptive_max_degree = 0.60

            # Use adaptive_degree to adjust removal strength based on
            # iteration progress. base_degree=val provides the starting value.
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
                    logger.debug(f"adaptive_degree call failed for {op_name}: {e}")
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
    prev_obj: float | None = None,
    t0: float | None = None,
    skip_batch_update: bool = False,
    extra: dict | None = None,
):
    """
    Lightweight post_destroy coordinator for summarize_removed_cargo,
    apply_batch_update, compute_improvement_and_report, build_opjson_log,
    adjust_exploration, and to_native.
    """
    pre_destroy_snapshot = getattr(state, "_pre_destroy_snapshot", None)
    pre_destroy_snapshot_error = getattr(state, "_pre_destroy_snapshot_error", None)
    state._post_destroy_certified = None
    state._post_destroy_certification_error = None

    # 1) Summarize cargo from removed vehicles for the log summary.
    try:
        removed_cargo_summary = summarize_removed_cargo(removed)
    except Exception:
        removed_cargo_summary = {}

    # 2) Apply batch inventory restoration through apply_batch_update.
    inventory_updates = None
    batch_conflict = False
    try:
        inventory_updates, batch_conflict = apply_batch_update(
            state, removed, skip_batch_update
        )
    except Exception:
        inventory_updates = None
        batch_conflict = True

    # 2.1) Apply extra inventory deltas supplied by the operator, such as
    # partial rollbacks, and merge them with the batch update.
    partial_updates = None
    if isinstance(extra, dict):
        partial_updates = extra.get("partial_updates")
    if partial_updates:
        try:
            if (
                removed
                and inventory_updates is not None
                and not skip_batch_update
                and partial_updates_overlap_removed_cargo(
                    state, partial_updates, removed
                )
            ):
                # Rebuild from the final vehicle set. This is authoritative when
                # removed cargo and partial rollback cargo overlap, because the
                # incremental deltas cannot distinguish duplicate restoration
                # from a separate partial rollback.
                state.compute_inventory()
                partial_applied = {}
            else:
                partial_applied = apply_partial_inventory_updates(
                    state, partial_updates
                )
            if partial_applied:
                if inventory_updates is None:
                    inventory_updates = {}
                for k, v in partial_applied.items():
                    inventory_updates[k] = inventory_updates.get(k, 0) + v
        except Exception as exc:
            batch_conflict = True
            record_swallowed("destroy_utils.post_destroy.partial_updates", exc)

    # A failed batch or partial update must be repaired from the current fleet,
    # not followed by objective/reporting work on an unknown inventory table.
    post_destroy_certified = not batch_conflict
    post_destroy_error = None
    if batch_conflict:
        post_destroy_certified = False
        try:
            state.mark_objective_dirty()
        except (AttributeError, TypeError) as exc:
            post_destroy_error = (
                f"inventory cache invalidation: {type(exc).__name__}: {exc}"
            )
            record_swallowed("destroy_utils.post_destroy.invalidate_cache", exc)
        try:
            state.compute_inventory()
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            post_destroy_error = (
                f"inventory rebuild: {type(exc).__name__}: {exc}"
            )
            record_swallowed("destroy_utils.post_destroy.compute_inventory", exc)

        try:
            _feasible, violations = state.validate()
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            post_destroy_error = (
                f"{post_destroy_error}; validation: {type(exc).__name__}: {exc}"
                if post_destroy_error
                else f"validation: {type(exc).__name__}: {exc}"
            )
            record_swallowed("destroy_utils.post_destroy.validate", exc)
        else:
            post_destroy_certified = not _has_hard_destroy_violation(violations)

        if not post_destroy_certified and pre_destroy_snapshot is not None:
            try:
                restore_state_snapshot(state, pre_destroy_snapshot)
                state._destroy_noop = True
                state.compute_inventory()
                _feasible, violations = state.validate()
                post_destroy_certified = not _has_hard_destroy_violation(
                    violations
                )
                post_destroy_error = None
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                post_destroy_error = (
                    f"pre-destroy restore: {type(exc).__name__}: {exc}"
                )
                record_swallowed(
                    "destroy_utils.post_destroy.restore_snapshot", exc
                )

        if not post_destroy_certified and post_destroy_error is None:
            post_destroy_error = (
                "post-destroy state retained a hard violation"
                if pre_destroy_snapshot is not None
                else pre_destroy_snapshot_error
                or "inventory recovery had no pre-destroy snapshot"
            )

    state._post_destroy_certified = post_destroy_certified
    state._post_destroy_certification_error = post_destroy_error

    # 3) Calculate elapsed time.
    elapsed = None
    if t0 is not None:
        try:
            elapsed = time.time() - t0
        except Exception:
            elapsed = None

    # 4) Early-exit path: record the phase and log the result.
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
            "post_destroy_certified": bool(post_destroy_certified),
            "post_destroy_error": post_destroy_error,
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

    # 5) Normal path: compute improvement and report it to param_tuner through
    # compute_improvement_and_report.
    new_obj = None
    improvement = 0.0
    if post_destroy_certified:
        try:
            new_obj, improvement, _ = compute_improvement_and_report(
                state, prev_obj, op_name
            )
        except Exception:
            new_obj = None
            improvement = 0.0

    # 6) Record the phase and emit the final log.
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
            post_destroy_certified=post_destroy_certified,
            post_destroy_error=post_destroy_error,
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
                "post_destroy_certified": bool(post_destroy_certified),
                "post_destroy_error": post_destroy_error,
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
