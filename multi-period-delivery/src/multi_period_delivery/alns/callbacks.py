"""
callbacks.py

Callback factories and registration helpers for ``ALNSOptimizer``.

Design notes:
- Heavy dependencies are imported locally when needed.
- The optimizer instance provides contextual objects such as ``log_printer``,
  ``param_tuner``, ``ml_selector``, and ``tracker``.
- Optional methods and attributes are guarded with defensive checks and
  exception handling.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from math import isfinite
from typing import TYPE_CHECKING, Any

import numpy.random as rnd

from .alns_config import default_config as ALNSConfig
from .alnsopt import SolutionState
from .ml_destroy_selection import ml_based_destroy
from .ml_repair_selection import ml_based_repair
from .progress import resolve_progress
from .snapshot import safe_state_snapshot
from .swallowed import record_swallowed

logger = logging.getLogger(__name__)


class RuntimeGuardError(RuntimeError):
    """Raised when no verified feasible state can be restored."""


def _callback_console_enabled() -> bool:
    return bool(getattr(ALNSConfig, "ENABLE_CALLBACK_DIAGNOSTICS_CONSOLE", False))


def _runtime_rollback_sources(optimizer: Any) -> tuple[tuple[str, Any], ...]:
    result = getattr(optimizer, "result", None)
    return (
        ("best_feasible_state", getattr(optimizer, "best_feasible_state", None)),
        (
            "_incumbent_state_snapshot",
            getattr(optimizer, "_incumbent_state_snapshot", None),
        ),
        ("result.best_state", getattr(result, "best_state", None)),
    )


def _restore_verified_runtime_state(state: Any, optimizer: Any) -> bool:
    """Restore the first snapshot that still passes full validation."""
    sources = _runtime_rollback_sources(optimizer)
    seen: set[int] = set()
    for label, source in sources:
        if source is None or id(source) in seen:
            continue
        seen.add(id(source))
        try:
            snapshot = safe_state_snapshot(source)
            feasible, _ = snapshot.validate()
            if not feasible:
                continue
            state.replace_vehicles(list(snapshot.vehicles))
            state.compute_inventory()
            feasible_after, _ = state.validate()
            if feasible_after:
                logger.info("[RUNTIME_GUARD] Restored verified snapshot from %s", label)
                return True
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            record_swallowed(f"callbacks.runtime_guard.restore.{label}", exc)
    return False


def dump_state_snapshot(optimizer: Any, state: SolutionState | None, label: str):
    if state is None:
        return
    try:
        logger.debug("[SNAPSHOT] %s objective=%s", label, state.objective())
    except Exception:
        logger.debug("[SNAPSHOT] %s captured", label)


def _safe_float(value: Any, default: float = float("inf")) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _get_capacity_threshold(data) -> float:
    caps = []
    for vt in list(getattr(data, "all_veh_types", []) or []):
        try:
            caps.append(float(data.veh_type_cap[vt]))
        except Exception:
            continue

    if not caps:
        return 0.0

    caps.sort()
    q = float(getattr(ALNSConfig, "FLEET_LARGE_CAPACITY_QUANTILE", 0.67))
    q = min(1.0, max(0.0, q))
    idx = int(round((len(caps) - 1) * q))
    return float(caps[idx])


def build_state_profile(state: SolutionState) -> dict[str, float]:
    if hasattr(state, "iter_used_vehicles") and callable(
        getattr(state, "iter_used_vehicles")
    ):
        vehicles = list(state.iter_used_vehicles())
    else:
        vehicles = []
        for veh in list(getattr(state, "vehicles", []) or []):
            try:
                if veh is None or veh.is_empty():
                    continue
            except Exception:
                continue
            vehicles.append(veh)

    veh_count = float(len(vehicles))
    if veh_count <= 0:
        return {
            "vehicles": 0.0,
            "load_ratio": 0.0,
            "large_ratio": 0.0,
            "fixed_cost": 0.0,
        }

    total_load = 0.0
    total_cap = 0.0
    total_fixed_cost = 0.0
    threshold = _get_capacity_threshold(state.data)
    large_count = 0.0

    for veh in vehicles:
        cap = float(
            state.data.veh_type_cap.get(
                veh.type, getattr(veh, "max_capacity", 0.0) or 0.0
            )
        )
        total_cap += cap
        try:
            total_load += float(state.compute_veh_load(veh))
        except Exception:
            pass
        total_fixed_cost += float(state.data.veh_type_cost.get(veh.type, 0.0))
        if cap >= threshold:
            large_count += 1.0

    load_ratio = total_load / total_cap if total_cap > 1e-12 else 0.0
    large_ratio = large_count / veh_count
    return {
        "vehicles": veh_count,
        "load_ratio": load_ratio,
        "large_ratio": large_ratio,
        "fixed_cost": total_fixed_cost,
    }


def is_candidate_better(
    new_obj: float,
    new_profile: dict[str, float],
    old_obj: float | None,
    old_profile: dict[str, float] | None,
    progress: float | None = None,
) -> bool:
    if getattr(ALNSConfig, "ENABLE_FAST_ALNS_MODE", False):
        if old_obj is None:
            return True
        return new_obj < old_obj

    if old_obj is None or old_profile is None:
        return True

    if progress is None:
        progress = 0.0
    progress = max(0.0, min(1.0, float(progress)))

    atol = 1e-6
    rtol = 1e-9
    improve_margin = atol + rtol * abs(old_obj)
    if new_obj < old_obj - improve_margin:
        return True

    near_rel = float(getattr(ALNSConfig, "FLEET_NEAR_EQUAL_OBJ_REL_TOL", 0.001))
    near_abs = float(getattr(ALNSConfig, "FLEET_NEAR_EQUAL_OBJ_ABS_TOL", 0.10))
    near_tol = max(0.0, max(near_abs, near_rel * abs(old_obj)))

    load_margin = float(getattr(ALNSConfig, "FLEET_LOAD_RATIO_MARGIN", 1e-4))
    fixed_cost_margin = float(getattr(ALNSConfig, "FLEET_FIXED_COST_MARGIN", 1e-6))
    large_ratio_margin = float(getattr(ALNSConfig, "FLEET_LARGE_RATIO_MARGIN", 1e-6))

    new_load = float(new_profile.get("load_ratio", 0.0))
    old_load = float(old_profile.get("load_ratio", 0.0))
    new_cost = float(new_profile.get("fixed_cost", 0.0))
    old_cost = float(old_profile.get("fixed_cost", 0.0))
    new_veh = float(new_profile.get("vehicles", 0.0))
    old_veh = float(old_profile.get("vehicles", 0.0))
    new_large_ratio = float(new_profile.get("large_ratio", 0.0))
    old_large_ratio = float(old_profile.get("large_ratio", 0.0))
    obj_delta = new_obj - old_obj

    if bool(getattr(ALNSConfig, "ENABLE_PHASED_LEXICOGRAPHIC_BEST", False)):
        lex_start = float(
            getattr(ALNSConfig, "PHASED_LEXICOGRAPHIC_START_PROGRESS", 0.35)
        )
        if progress >= lex_start:
            lex_rel = float(
                getattr(ALNSConfig, "PHASED_LEXICOGRAPHIC_OBJ_REL_TOL", 0.01)
            )
            lex_abs = float(
                getattr(ALNSConfig, "PHASED_LEXICOGRAPHIC_OBJ_ABS_TOL", 2.0)
            )
            lex_tol = max(0.0, max(lex_abs, lex_rel * abs(old_obj)))
            if new_veh < old_veh:
                return obj_delta <= lex_tol
            if new_veh > old_veh:
                return False

    if abs(obj_delta) <= near_tol:
        if new_veh < old_veh:
            return True
        if new_veh > old_veh:
            return False
        if new_load > old_load + load_margin:
            return True
        if new_load < old_load - load_margin:
            return False
        if new_cost < old_cost - fixed_cost_margin:
            return True
        if new_cost > old_cost + fixed_cost_margin:
            return False
        if new_large_ratio < old_large_ratio - large_ratio_margin:
            return True
        return False

    if getattr(ALNSConfig, "ENABLE_VEHICLE_COUNT_PRIORITY", True):
        # Adjust tolerances by search progress:
        # early (<0.5): more aggressive fleet reduction;
        # middle (0.5-0.8): standard tolerance;
        # late (>=0.8): more conservative tolerance to preserve quality.
        phase_factor = 1.0
        if progress < 0.5:
            # Early exploration: allow 1.5x the standard tolerance.
            phase_factor = 1.5
        elif progress >= 0.8:
            # Late refinement: reduce tolerance by 50%.
            phase_factor = 0.5

        rel_tol = float(getattr(ALNSConfig, "VEHICLE_PRIORITY_OBJ_REL_TOL", 0.005))
        abs_tol = float(getattr(ALNSConfig, "VEHICLE_PRIORITY_OBJ_ABS_TOL", 0.25))

        # Apply the progress-dependent factor.
        rel_tol = rel_tol * phase_factor
        abs_tol = abs_tol * phase_factor

        allowed_worse = max(0.0, max(abs_tol, rel_tol * abs(old_obj)))
        if obj_delta > 0.0 and obj_delta <= allowed_worse and new_veh < old_veh:
            return True

    return False


if TYPE_CHECKING:
    from alns.ALNS import ALNS

    from .alns_optimizer import ALNSOptimizer


def _used_vehicle_count(state: SolutionState | None) -> int:
    if state is None:
        return 0
    if hasattr(state, "count_used_vehicles") and callable(
        getattr(state, "count_used_vehicles")
    ):
        try:
            return int(state.count_used_vehicles())
        except Exception:
            pass
    cnt = 0
    for veh in list(state.vehicles):
        try:
            if veh is None or veh.is_empty():
                continue
        except Exception:
            continue
        cnt += 1
    return cnt


def _profile_signature(profile: dict) -> tuple:
    """Build a coarse structural signature from a fleet profile."""
    bin_size = max(1, int(getattr(ALNSConfig, "ELITE_SIGNATURE_BIN_SIZE", 10)))
    veh = int(round(float(profile.get("vehicles", 0.0))))
    load_bin = int(round(float(profile.get("load_ratio", 0.0)) * bin_size))
    large_bin = int(round(float(profile.get("large_ratio", 0.0)) * bin_size))
    fixed_bin = int(round(float(profile.get("fixed_cost", 0.0)) / 10.0))
    return (veh, load_bin, large_bin, fixed_bin)


def _vehicle_route_signature(veh: Any) -> tuple:
    """Build a deterministic vehicle-level structural signature."""
    cargo_items = tuple(sorted(list(getattr(veh, "cargo", {}).items())))
    return (
        getattr(veh, "fact_id", None),
        getattr(veh, "dealer_id", None),
        int(getattr(veh, "day", 0) or 0),
        getattr(veh, "type", None),
        cargo_items,
    )


def _build_path_relinked_vehicle_list(
    current_state: SolutionState,
    target_state: SolutionState,
    blend_ratio: float = 0.35,
    max_target_veh: int = 20,
) -> list:
    """Create a relinked vehicle list by partially importing elite target structure."""
    current_used = list(current_state.iter_used_vehicles())
    target_used = list(target_state.iter_used_vehicles())

    if not target_used:
        return list(current_used)

    blend_ratio = min(1.0, max(0.0, float(blend_ratio)))
    max_target_veh = max(1, int(max_target_veh))

    target_count = max(1, int(round(len(target_used) * blend_ratio)))
    target_count = min(target_count, max_target_veh, len(target_used))

    # Prefer high-load target vehicles so injected structure carries stronger signal.
    target_sorted = sorted(
        target_used,
        key=lambda v: float(target_state.compute_veh_load(v)),
        reverse=True,
    )
    target_pick = target_sorted[:target_count]
    target_sigs = {_vehicle_route_signature(v) for v in target_pick}

    kept_current = [
        v for v in current_used if _vehicle_route_signature(v) not in target_sigs
    ]
    relinked = kept_current + target_pick
    return relinked


def _build_relink_progression(
    base_blend_ratio: float,
    base_max_target_veh: int,
    steps: int = 3,
    blend_step_delta: float = 0.10,
    target_step_delta: int = 4,
) -> list[tuple[float, int]]:
    """Build progressive relink schedule from conservative to stronger blend."""
    steps = max(1, int(steps))
    base_blend_ratio = min(1.0, max(0.0, float(base_blend_ratio)))
    base_max_target_veh = max(1, int(base_max_target_veh))
    blend_step_delta = max(0.0, float(blend_step_delta))
    target_step_delta = max(0, int(target_step_delta))

    schedule: list[tuple[float, int]] = []
    for i in range(steps):
        ratio_i = round(min(1.0, base_blend_ratio + i * blend_step_delta), 6)
        max_target_i = max(1, base_max_target_veh + i * target_step_delta)
        schedule.append((ratio_i, max_target_i))
    return schedule


def _vehicle_elimination_repair(state: SolutionState, rng: rnd.Generator, **kwargs):
    try:
        from . import repair_operators as repair_ops

        candidate = getattr(repair_ops, "cross_period_consolidation_repair", None)
        if candidate is not None:
            return candidate(state, rng)
        # Fall back to the ordinary repair only when the dedicated
        # consolidation operator is unavailable.
        fallback = getattr(repair_ops, "inventory_balance_repair", None)
        if fallback is None:
            return state
        return fallback(state, rng, **kwargs)
    except Exception:
        return state


def _is_relink_candidate_acceptable(
    current_state: SolutionState,
    candidate_state: SolutionState,
    max_rel_worse: float = 0.02,
) -> bool:
    """Check whether a relink candidate is acceptable against current state."""
    try:
        curr_obj = _safe_float(current_state.objective())
        cand_obj = _safe_float(candidate_state.objective())
        if not (isfinite(curr_obj) and isfinite(cand_obj)):
            return False

        curr_profile = build_state_profile(current_state)
        cand_profile = build_state_profile(candidate_state)
        if is_candidate_better(cand_obj, cand_profile, curr_obj, curr_profile):
            return True

        max_allow_obj = curr_obj * (1.0 + max(0.0, float(max_rel_worse)))
        return cand_obj <= max_allow_obj
    except Exception:
        return False


def _choose_elite_relink_target(
    current_state: SolutionState,
    elite_pool: list[dict],
    require_signature_diff: bool = True,
    max_rel_worse: float = 0.02,
) -> dict | None:
    """Pick an elite candidate for basin-jump path relinking under safety gates."""
    if not elite_pool:
        return None

    curr_obj = _safe_float(current_state.objective())
    curr_profile = build_state_profile(current_state)
    curr_sig = _profile_signature(curr_profile)
    max_allow_obj = curr_obj * (1.0 + max(0.0, float(max_rel_worse)))

    candidates: list[dict] = []
    for item in elite_pool:
        target_state = item.get("state")
        if target_state is None:
            continue

        try:
            target_profile = item.get("profile") or build_state_profile(target_state)
            target_obj = _safe_float(item.get("obj", target_state.objective()))
            target_sig = tuple(
                item.get("signature", _profile_signature(target_profile))
            )
        except Exception:
            continue

        if require_signature_diff and target_sig == curr_sig:
            continue

        if target_obj > max_allow_obj:
            continue

        if not (
            is_candidate_better(
                target_obj,
                target_profile,
                curr_obj,
                curr_profile,
            )
            or target_obj < curr_obj
        ):
            continue

        candidates.append(
            {
                "state": target_state,
                "obj": target_obj,
                "profile": target_profile,
                "signature": target_sig,
            }
        )

    if not candidates:
        return None

    candidates.sort(key=lambda x: float(x.get("obj", float("inf"))))
    return candidates[0]


def _is_best_feasible_solution(state: SolutionState) -> bool:
    """
    Return whether the state qualifies as a best feasible candidate.

    Requirements: validation succeeds, the objective is finite, and there are no
    unmet-demand violations.
    """
    if state is None:
        return False

    # Call compute_inventory defensively for compatibility with tests.
    try:
        if hasattr(state, "compute_inventory") and callable(
            getattr(state, "compute_inventory")
        ):
            # Force an inventory refresh so the state is internally consistent.
            state.compute_inventory()
    except Exception:
        pass

    # Validation includes demand-fulfillment constraints.
    is_feasible, violations = state.validate()

    if not is_feasible:
        return False

    # Reject non-finite objective values.
    obj = _safe_float(state.objective())

    if not isfinite(obj):
        return False

    return True


def _record_best_feasible(candidates: list[SolutionState]) -> SolutionState | None:
    """
    Return the best fully feasible state from the candidate list.

    A fully feasible state passes validation. Return ``None`` when no candidate
    qualifies.
    """
    best_state: SolutionState | None = None
    best_obj: float | None = None
    best_profile: dict | None = None

    for state in candidates:
        is_best_feasible = _is_best_feasible_solution(state)
        if not is_best_feasible:
            continue

        obj = _safe_float(state.objective())
        if not isfinite(obj):
            continue

        profile = build_state_profile(state)
        if is_candidate_better(obj, profile, best_obj, best_profile):
            best_state = state
            best_obj = obj
            best_profile = profile

    return best_state


def record_best_feasible(optimizer: ALNSOptimizer, state: SolutionState):
    """
    A candidate state is recorded as the current
    `best_feasible_state` if it satisfies hard feasibility (`validate()`).

    This candidate state should also be compared with all other candidates collected
    during the ALNS run to ensure it is indeed the best feasible solution found so far,
    including `ALNTracker.best_solution` found during iteration,
    and `ALNSOptimizer.best_feasible_state` during callback, if available.

    This version forces a deepcopy when recording the best feasible state and
    emits a JSON snapshot for offline debugging.
    """

    # Collect the callback state, tracker best solution, and recorded best state.
    candidates = [state]

    tracker = getattr(optimizer, "tracker", None)
    if tracker is not None:
        # 1. Consider tracker.best_solution.
        best_sol_any = getattr(tracker, "best_solution", None)
        if best_sol_any is not None:
            candidates.append(best_sol_any)

    # 2. Include the optimizer's recorded best feasible state.
    if optimizer.best_feasible_state is not None:
        candidates.append(optimizer.best_feasible_state)

    # Find the best fully feasible candidate.
    best = _record_best_feasible(candidates)

    if best is None:
        # No better fully feasible state was found.
        return

    # Keep the existing record when the objective does not improve.
    prev = optimizer.best_feasible_state

    # Objective of the previously recorded best feasible state.
    prev_obj = _safe_float(prev.objective()) if prev is not None else None
    best_obj = None

    # Objective of the newly selected best feasible state.
    best_obj = _safe_float(best.objective())

    if best_obj is None:
        return  # Invalid objective; keep the previous record.
    best_veh = _used_vehicle_count(best)
    best_profile = build_state_profile(best)
    prev_profile = build_state_profile(prev) if prev is not None else None

    # Decide whether to update the recorded best feasible state.
    # Compute progress for dynamic tolerance adjustments.
    progress = 0.0
    try:
        if hasattr(optimizer, "tracker") and optimizer.tracker is not None:
            tracker = optimizer.tracker
            # Try to read progress information from the tracker.
            if hasattr(tracker, "_iteration_count") and hasattr(
                tracker, "_max_iterations"
            ):
                max_iter = getattr(tracker, "_max_iterations", 1000)
                curr_iter = getattr(tracker, "_iteration_count", 0)
                if max_iter > 0:
                    progress = min(1.0, float(curr_iter) / float(max_iter))
    except Exception:
        progress = 0.0
    progress = max(0.0, min(1.0, progress))

    should_update = is_candidate_better(
        best_obj, best_profile, prev_obj, prev_profile, progress=progress
    )

    if should_update:
        # Record a snapshot of the new best feasible solution.
        optimizer.best_feasible_state = safe_state_snapshot(best)
        optimizer.best_feasible_obj = best_obj if best_obj is not None else float("inf")

        try:
            tracker = getattr(optimizer, "tracker", None)
            if tracker is not None and hasattr(tracker, "record_elite_candidate"):
                tracker.record_elite_candidate(best)
        except Exception:
            pass

        # write debug snapshot
        try:
            dump_state_snapshot(
                optimizer, optimizer.best_feasible_state, "best_feasible"
            )
        except Exception:
            pass

        if best_obj is not None:
            logger.info(
                "[BEST_FEASIBLE] Recorded fully feasible solution: obj=%.2f, vehicles=%d",
                best_obj,
                best_veh,
            )
            optimizer.log_printer.print(
                f"Found new best feasible solution with cost: {best_obj:.2f}, vehicles: {best_veh}"
            )

    else:
        if best_obj is not None and prev_obj is not None:
            logger.info(
                "[BEST_FEASIBLE] Candidate is not better; keeping record: best_obj=%.2f prev_obj=%.2f",
                best_obj,
                prev_obj,
            )


def make_callback_on_best(optimizer: ALNSOptimizer) -> Callable[..., None]:
    """Return an ``on_best`` callback that delegates to record_best_feasible()."""

    def callback_on_best(state, rng, **kwargs):
        try:
            record_best_feasible(optimizer, state)
        except Exception:
            optimizer.log_printer.print(
                "Callback on best: helper failed.", color="yellow"
            )

    return callback_on_best


def ml_warmup_sample(optimizer: ALNSOptimizer, state: SolutionState, budget: int = 1):
    """
    Run lightweight exploratory sampling on state copies.

    Purpose: accumulate ML training samples early so the model can reach its
    minimum sample threshold quickly. ``budget`` controls how many destroy/repair
    trials are attempted per call. Work is performed only on copies, and all
    exceptions are caught to avoid disrupting the main ALNS loop.
    """
    if not optimizer.ml_selector or not optimizer.param_tuner:
        return
    try:
        for _ in range(max(0, int(budget))):
            tmp_state = safe_state_snapshot(state)
            if tmp_state is None:
                continue
            # Try one destroy operation with explicit selector and tuner inputs.
            try:
                _ = ml_based_destroy(
                    tmp_state,
                    optimizer.rng,
                    1,
                )
            except Exception:
                # A failed destroy does not prevent the repair attempt.
                logger.debug("ML warmup: destroy failed on temp state, continuing")
            # Try one repair operation with explicit selector and tuner inputs.
            try:
                _ = ml_based_repair(
                    tmp_state,
                    optimizer.rng,
                    1,
                )
            except Exception:
                logger.debug("ML warmup: repair failed on temp state, continuing")
    except Exception:
        logger.exception("ML warmup sampling failed")


def make_callback_on_iteration(
    optimizer: ALNSOptimizer, stop: Any | None = None
) -> Callable[..., None]:
    """
    Return an ``on_iteration`` callback.

    The callback:
      - checks and repairs negative inventory;
      - runs a runtime feasibility guard and rolls back to the latest feasible
        snapshot when necessary;
      - prints periodic progress using ``stop`` when available;
      - updates the parameter tuner iteration count;
      - submits ML retraining and warmup sampling when enabled;
      - delegates to ``tracker.on_iteration`` when present.

    All optimizer access is guarded to avoid disrupting the main ALNS loop.
    """

    def callback_on_iteration(state: SolutionState, rng: rnd.Generator, **kwargs):
        fast_mode = bool(getattr(ALNSConfig, "ENABLE_FAST_ALNS_MODE", False))

        # Step 1: throttle full validation according to configuration to reduce
        # validate() overhead. The callback increments the counter at the end,
        # so use next_iteration for scheduling decisions.
        next_iteration = (
            optimizer.get_iteration() + 1
            if hasattr(optimizer, "get_iteration")
            and callable(getattr(optimizer, "get_iteration"))
            else 1
        )
        validate_every = max(
            1,
            int(
                getattr(
                    ALNSConfig,
                    "FAST_VALIDATE_EVERY_N_ITERS"
                    if fast_mode
                    else "VALIDATE_EVERY_N_ITERS",
                    1,
                )
            ),
        )
        warmup_iters = max(0, int(getattr(ALNSConfig, "VALIDATE_WARMUP_ITERS", 0)))
        run_full_guard = (next_iteration <= warmup_iters) or (
            next_iteration % validate_every == 0
        )
        if getattr(ALNSConfig, "ENABLE_STRICT_FEASIBILITY_ACCEPT", False):
            run_full_guard = True

        feasible = True
        _guard_viol = {}

        if run_full_guard:
            # Recompute inventory so loads and shipments stay consistent.
            state.compute_inventory()

            # Check for negative inventory.
            if (
                hasattr(state, "inventory_manager")
                and state.inventory_manager is not None
            ):
                negative_items = state.inventory_manager.scan_negative_inventory(state)
            else:
                negative_items = [(k, v) for k, v in state.s_ikt.items() if v < -0.001]

            if negative_items:
                logger.debug(
                    "[CALLBACK_INV_CHECK] Negative inventory detected: count=%d sample=%s",
                    len(negative_items),
                    negative_items[:5],
                )
                # Trigger automatic repair through the inventory manager.
                if (
                    hasattr(state, "inventory_manager")
                    and state.inventory_manager is not None
                ):
                    repaired = (
                        state.inventory_manager.auto_repair_inventory_inconsistency(
                            state, context="callback_on_iteration"
                        )
                    )
                    if repaired:
                        logger.debug("[CALLBACK_INV_CHECK] Negative inventory repaired")
                        # Mark feasible temporarily; validation follows below.
                        feasible = True
                    else:
                        logger.debug("[CALLBACK_INV_CHECK] Negative inventory repair failed")
                        # Mark infeasible to trigger the runtime guard.
                        feasible = False

        try:
            if run_full_guard:
                # Validate the current state.
                feasible, _guard_viol = state.validate()
                neg_viol = []
                if isinstance(_guard_viol, dict):
                    neg_viol = list(_guard_viol.get("negative_inventory") or [])

                # Run infeasible_repair only when negative inventory exists.
                if not feasible and len(neg_viol) > 0:
                    logger.debug(
                        "[VALIDATION_ERROR] Negative inventory detected: count=%d sample=%s",
                        len(neg_viol),
                        neg_viol[:5],
                    )

                    # Trigger infeasible_repair.
                    from .repair_operators import infeasible_repair

                    repaired_state = infeasible_repair(state, rng)
                    if repaired_state is not None:
                        # Replace the current vehicle set.
                        state.replace_vehicles(repaired_state.vehicles)
                        logger.info("[INFEASIBLE_REPAIR] Negative inventory repaired")

                if feasible:
                    # Perform an additional inventory consistency check.
                    if (
                        hasattr(state, "inventory_manager")
                        and state.inventory_manager is not None
                    ):
                        negative_items = (
                            state.inventory_manager.scan_negative_inventory(state)
                        )
                    else:
                        negative_items = [
                            (k, v) for k, v in state.s_ikt.items() if v < -0.001
                        ]

                    neg_count = len(negative_items)

                    if neg_count > 0:
                        logger.debug(
                            f"[GUARD_INV] Detected {neg_count} negative inventory entries after acceptance"
                        )
                        # Mark infeasible and trigger rollback.
                        feasible = False

                        # Record the violation.
                        if _guard_viol is None:
                            _guard_viol = {}
                        # Keep inventory_inconsistency as a list for compatibility
                        # with validate().
                        _guard_viol.setdefault("inventory_inconsistency", []).append(
                            {"negative_count": neg_count}
                        )

                # Runtime guard: if current state is infeasible, roll back to latest known feasible snapshot.
                if not feasible:
                    cand_iter = int(next_iteration)
                    viol_counts = {}
                    if isinstance(_guard_viol, dict):
                        for k, v in _guard_viol.items():
                            cnt = 0
                            if isinstance(v, list):
                                cnt = len(v)
                            elif isinstance(v, dict):
                                cnt = len(v)
                            elif v:
                                cnt = 1
                            if cnt > 0:
                                viol_counts[k] = cnt
                    reason_text = ", ".join(
                        [f"{k}:{c}" for k, c in sorted(viol_counts.items())]
                    )
                    if not reason_text:
                        reason_text = "unknown"

                    only_unmet = (
                        len(viol_counts) == 1 and "unmet_demand" in viol_counts
                    )
                    best_feasible = getattr(optimizer, "best_feasible_state", None)
                    if best_feasible is not None or not only_unmet:
                        if _restore_verified_runtime_state(state, optimizer):
                            if _callback_console_enabled():
                                optimizer.log_printer.print(
                                    f"[RUNTIME_GUARD][candidate iteration #{cand_iter}] Candidate infeasible ({reason_text}); restored a verified feasible snapshot",
                                    color="yellow",
                                )
                        elif any(
                            source is not None
                            for _label, source in _runtime_rollback_sources(optimizer)
                        ):
                            raise RuntimeGuardError(
                                "No verified feasible state is available for "
                                f"runtime rollback at iteration {cand_iter} "
                                f"({reason_text})"
                            )
                        else:
                            logger.error(
                                "[RUNTIME_GUARD] No feasible rollback source exists; "
                                "leaving the unverified candidate for rejection: %s",
                                reason_text,
                            )
                            if _callback_console_enabled():
                                optimizer.log_printer.print(
                                    f"[RUNTIME_GUARD][candidate iteration #{cand_iter}] "
                                    f"Candidate infeasible ({reason_text}) and no "
                                    "verified rollback source exists",
                                    color="bold red",
                                )
                    else:
                        # Early in the run, candidates may be temporarily infeasible
                        # because unmet demand exists before the first feasible incumbent.
                        should_log = next_iteration <= 5 or (next_iteration % 50 == 0)
                        if should_log:
                            if _callback_console_enabled():
                                optimizer.log_printer.print(
                                    f"[RUNTIME_GUARD][candidate iteration #{cand_iter}] Candidate temporarily infeasible (unmet_demand={int(viol_counts.get('unmet_demand', 0))}); no verified incumbent yet, continuing search",
                                    color="yellow",
                                )

            # Advance optimizer's internal iteration counter if available (best-effort).
            # Synchronize the optimizer counter and optional ML components.
            optimizer.increment_iteration()

            cur_it = optimizer.get_iteration()
            max_it = optimizer.get_max_iterations()

            # Synchronize the ML selector.
            if optimizer.ml_selector is not None:
                no_improve_iters = 0
                if optimizer.tracker is not None and hasattr(
                    optimizer.tracker, "get_statistics"
                ):
                    try:
                        no_improve_iters = int(
                            optimizer.tracker.get_statistics().get(
                                "no_improve_iters", 0
                            )
                        )
                    except Exception:
                        no_improve_iters = 0
                optimizer.ml_selector.set_iteration(
                    cur_it, max_it, no_improve_iters=no_improve_iters
                )

            # Synchronize the parameter tuner.
            if (
                optimizer.param_tuner is not None
                and hasattr(optimizer.param_tuner, "set_iteration")
                and max_it is not None
            ):
                no_improve_iters = 0
                if optimizer.tracker is not None and hasattr(
                    optimizer.tracker, "get_statistics"
                ):
                    try:
                        no_improve_iters = int(
                            optimizer.tracker.get_statistics().get(
                                "no_improve_iters", 0
                            )
                        )
                    except Exception:
                        no_improve_iters = 0
                optimizer.param_tuner.set_iteration(
                    cur_it, max_it, no_improve_iters=no_improve_iters
                )

            iteration = None
            # Primary iteration source: optimizer.get_iteration()
            # Fallbacks: optimizer.tracker.get_iteration() or tracker.get_statistics().get("total_iterations")

            if hasattr(optimizer, "get_iteration") and callable(
                getattr(optimizer, "get_iteration")
            ):
                iteration = optimizer.get_iteration()
            else:
                tracker = getattr(optimizer, "tracker", None)
                if tracker is not None:
                    if hasattr(tracker, "get_iteration") and callable(
                        getattr(tracker, "get_iteration")
                    ):
                        iteration = tracker.get_iteration()
                    elif hasattr(tracker, "get_statistics"):
                        iteration = tracker.get_statistics().get("total_iterations")

            # Compute the progress fraction (0..1) for the phase-gated policies
            # below. This uses the shared measure so every policy reads the same
            # number from the same time origin (tracker.start_time) instead of
            # each deriving its own.
            progress = resolve_progress((optimizer,))

            # Stagnation-gated light fleet consolidation from the best feasible snapshot.
            try:
                if getattr(ALNSConfig, "ENABLE_LIGHT_VEHICLE_INTENSIFY", False):
                    interval = max(
                        1,
                        int(getattr(ALNSConfig, "LIGHT_INTENSIFY_INTERVAL", 250)),
                    )
                    stagnation_need = max(
                        0,
                        int(getattr(ALNSConfig, "LIGHT_INTENSIFY_STAGNATION", 120)),
                    )
                    min_prog = float(
                        getattr(ALNSConfig, "LIGHT_INTENSIFY_MIN_PROGRESS", 0.20)
                    )
                    max_prog = float(
                        getattr(ALNSConfig, "LIGHT_INTENSIFY_MAX_PROGRESS", 0.95)
                    )
                    no_improve = int(
                        getattr(
                            optimizer.tracker,
                            "no_improve_iters",
                            getattr(optimizer.tracker, "no_improve_streak", 0),
                        )
                    )
                    iter_val = int(iteration) if iteration is not None else 0
                    cooldown_until = int(
                        getattr(optimizer, "_light_intensify_cooldown_until", 0)
                    )

                    should_intensify = (
                        iteration is not None
                        and iteration > 0
                        and iteration % interval == 0
                        and min_prog <= progress <= max_prog
                        and no_improve >= stagnation_need
                        and iter_val >= cooldown_until
                        and getattr(optimizer, "best_feasible_state", None) is not None
                    )

                    if should_intensify:
                        seed_source = getattr(optimizer, "best_feasible_state", None)
                        if seed_source is None:
                            return None

                        seed_state = safe_state_snapshot(seed_source)
                        if seed_state is None:
                            return None

                        cand = _vehicle_elimination_repair(
                            seed_state,
                            rng,
                            max_iter=int(
                                getattr(ALNSConfig, "LIGHT_INTENSIFY_MAX_ITER", 12)
                            ),
                            max_targets=int(
                                getattr(ALNSConfig, "LIGHT_INTENSIFY_MAX_TARGETS", 24)
                            ),
                            pre_greedy=False,
                        )

                        cand.compute_inventory()
                        cand_feasible, _ = cand.validate()
                        if cand_feasible:
                            cand_obj = _safe_float(cand.objective())
                            curr_obj = _safe_float(state.objective())
                            best_obj = _safe_float(
                                getattr(optimizer, "best_feasible_obj", None),
                                default=float("inf"),
                            )
                            cand_profile = build_state_profile(cand)
                            curr_profile = build_state_profile(state)
                            cand_sig = _profile_signature(cand_profile)
                            last_sig = getattr(
                                optimizer, "_light_intensify_last_signature", None
                            )
                            require_sig_change = bool(
                                getattr(
                                    ALNSConfig,
                                    "LIGHT_INTENSIFY_REQUIRE_SIG_CHANGE",
                                    True,
                                )
                            )

                            rel_allow = float(
                                getattr(
                                    ALNSConfig,
                                    "LIGHT_INTENSIFY_ACCEPT_REL_WORSE",
                                    0.01,
                                )
                            )
                            allow_obj = curr_obj * (1.0 + max(0.0, rel_allow))

                            better_or_near = (
                                is_candidate_better(
                                    cand_obj,
                                    cand_profile,
                                    curr_obj,
                                    curr_profile,
                                )
                                or cand_obj <= allow_obj
                            )
                            if require_sig_change and last_sig is not None:
                                if cand_sig == last_sig:
                                    better_or_near = False

                            no_gain_runs = int(
                                getattr(optimizer, "_light_intensify_no_gain_runs", 0)
                            )
                            if isfinite(best_obj) and cand_obj >= (best_obj - 1e-9):
                                no_gain_runs += 1
                            else:
                                no_gain_runs = 0
                            setattr(
                                optimizer, "_light_intensify_no_gain_runs", no_gain_runs
                            )

                            max_no_gain = max(
                                1,
                                int(
                                    getattr(
                                        ALNSConfig,
                                        "LIGHT_INTENSIFY_MAX_NO_GAIN",
                                        4,
                                    )
                                ),
                            )
                            if no_gain_runs >= max_no_gain:
                                cooldown_iters = max(
                                    interval,
                                    int(
                                        getattr(
                                            ALNSConfig,
                                            "LIGHT_INTENSIFY_COOLDOWN_ITERS",
                                            1200,
                                        )
                                    ),
                                )
                                setattr(
                                    optimizer,
                                    "_light_intensify_cooldown_until",
                                    iter_val + cooldown_iters,
                                )
                                setattr(optimizer, "_light_intensify_no_gain_runs", 0)
                                better_or_near = False

                            if better_or_near:
                                state.replace_vehicles(list(cand.vehicles))
                                state.compute_inventory()
                                setattr(
                                    optimizer,
                                    "_light_intensify_last_signature",
                                    cand_sig,
                                )
                                optimizer.log_printer.print(
                                    f"[LIGHT_INTENSIFY] it={iteration} apply vehicle elimination from best feasible snapshot"
                                )
                                record_best_feasible(optimizer, state)
            except Exception:
                logger.exception("Light vehicle intensification failed in callback")

            # Elite path relinking: select a structurally diverse elite target
            # and attempt a controlled basin jump.
            try:
                if getattr(ALNSConfig, "ENABLE_ELITE_PATH_RELINK", False):
                    interval = max(
                        1,
                        int(getattr(ALNSConfig, "ELITE_RELINK_INTERVAL", 300)),
                    )
                    stagnation_need = max(
                        0,
                        int(getattr(ALNSConfig, "ELITE_RELINK_STAGNATION", 150)),
                    )
                    min_prog = float(
                        getattr(ALNSConfig, "ELITE_RELINK_MIN_PROGRESS", 0.25)
                    )
                    max_prog = float(
                        getattr(ALNSConfig, "ELITE_RELINK_MAX_PROGRESS", 0.95)
                    )
                    no_improve = int(
                        getattr(
                            optimizer.tracker,
                            "no_improve_iters",
                            getattr(optimizer.tracker, "no_improve_streak", 0),
                        )
                    )

                    should_relink = (
                        iteration is not None
                        and iteration > 0
                        and iteration % interval == 0
                        and min_prog <= progress <= max_prog
                        and no_improve >= stagnation_need
                    )

                    if should_relink:
                        tracker = getattr(optimizer, "tracker", None)
                        elite_pool = (
                            getattr(tracker, "elite_pool", [])
                            if tracker is not None
                            else []
                        )
                        picked = _choose_elite_relink_target(
                            state,
                            elite_pool,
                            require_signature_diff=bool(
                                getattr(
                                    ALNSConfig,
                                    "ELITE_RELINK_REQUIRE_SIGNATURE_DIFF",
                                    True,
                                )
                            ),
                            max_rel_worse=float(
                                getattr(ALNSConfig, "ELITE_RELINK_MAX_REL_WORSE", 0.02)
                            ),
                        )

                        if picked is not None and picked.get("state") is not None:
                            backup_state = safe_state_snapshot(state)
                            target_snapshot = safe_state_snapshot(picked["state"])
                            if backup_state is None or target_snapshot is None:
                                return None
                            applied = False
                            try:
                                working_state = safe_state_snapshot(state)
                                last_feasible = safe_state_snapshot(state)
                                if working_state is None or last_feasible is None:
                                    return None
                                applied_steps = 0

                                progression = _build_relink_progression(
                                    base_blend_ratio=float(
                                        getattr(
                                            ALNSConfig,
                                            "ELITE_RELINK_BLEND_RATIO",
                                            0.35,
                                        )
                                    ),
                                    base_max_target_veh=int(
                                        getattr(
                                            ALNSConfig,
                                            "ELITE_RELINK_MAX_TARGET_VEH",
                                            20,
                                        )
                                    ),
                                    steps=int(
                                        getattr(ALNSConfig, "ELITE_RELINK_STEPS", 3)
                                    ),
                                    blend_step_delta=float(
                                        getattr(
                                            ALNSConfig,
                                            "ELITE_RELINK_BLEND_STEP_DELTA",
                                            0.10,
                                        )
                                    ),
                                    target_step_delta=int(
                                        getattr(
                                            ALNSConfig,
                                            "ELITE_RELINK_TARGET_STEP_DELTA",
                                            4,
                                        )
                                    ),
                                )

                                for ratio_i, max_target_i in progression:
                                    relinked_vehicles = (
                                        _build_path_relinked_vehicle_list(
                                            working_state,
                                            target_snapshot,
                                            blend_ratio=ratio_i,
                                            max_target_veh=max_target_i,
                                        )
                                    )

                                    working_state.replace_vehicles(
                                        list(relinked_vehicles)
                                    )
                                    working_state.compute_inventory()
                                    feasible, _ = working_state.validate()
                                    if not feasible:
                                        break

                                    applied_steps += 1
                                    last_feasible = safe_state_snapshot(working_state)

                                if applied_steps > 0 and last_feasible is not None:
                                    max_rel_worse = float(
                                        getattr(
                                            ALNSConfig,
                                            "ELITE_RELINK_MAX_REL_WORSE",
                                            0.02,
                                        )
                                    )
                                    if _is_relink_candidate_acceptable(
                                        backup_state,
                                        last_feasible,
                                        max_rel_worse=max_rel_worse,
                                    ):
                                        state.replace_vehicles(
                                            list(last_feasible.vehicles)
                                        )
                                        state.compute_inventory()
                                        applied = True
                                        optimizer.log_printer.print(
                                            f"[ELITE_RELINK] it={iteration} progressive relink steps={applied_steps} (target_obj={_safe_float(picked['obj']):.4f})"
                                        )
                                        record_best_feasible(optimizer, state)
                            except Exception:
                                applied = False

                            if not applied:
                                try:
                                    state.replace_vehicles(list(backup_state.vehicles))
                                    state.compute_inventory()
                                except Exception:
                                    logger.exception(
                                        "Elite relink rollback failed after unsuccessful jump"
                                    )
            except Exception:
                logger.exception("Elite path relink failed in callback")

            # Stagnation escape: actual-objective-aware basin jump from the best feasible state.
            # This is more aggressive than light intensify and uses a strong period-range destroy
            # followed by a forward-looking repair, then accepts the candidate if it improves
            # either search or actual objective without worsening fleet size.
            try:
                if getattr(ALNSConfig, "ENABLE_STAGNATION_ESCAPE_RESTART", True):
                    interval = max(
                        1,
                        int(getattr(ALNSConfig, "STAGNATION_ESCAPE_INTERVAL", 200)),
                    )
                    stagnation_need = max(
                        0,
                        int(getattr(ALNSConfig, "STAGNATION_ESCAPE_STAGNATION", 120)),
                    )
                    no_improve = int(
                        getattr(
                            optimizer.tracker,
                            "no_improve_iters",
                            getattr(optimizer.tracker, "no_improve_streak", 0),
                        )
                    )
                    cooldown_until = int(
                        getattr(optimizer, "_stagnation_escape_cooldown_until", 0)
                    )
                    iter_val = int(iteration) if iteration is not None else 0

                    should_escape = (
                        iteration is not None
                        and iteration > 0
                        and no_improve >= stagnation_need
                        and iter_val >= cooldown_until
                        and getattr(optimizer, "best_feasible_state", None) is not None
                        and (
                            no_improve == stagnation_need
                            or no_improve % interval == 0
                            or iter_val % interval == 0
                        )
                    )

                    if should_escape:
                        seed_source = getattr(optimizer, "best_feasible_state", None)
                        seed_state = safe_state_snapshot(seed_source)
                        if seed_state is not None:
                            from .destroy_operators import period_range_removal
                            from .repair_operators import lookahead_repair

                            cand = period_range_removal(
                                seed_state,
                                rng,
                                degree=float(
                                    getattr(
                                        ALNSConfig, "STAGNATION_ESCAPE_DEGREE", 0.55
                                    )
                                ),
                                window_ratio=float(
                                    getattr(
                                        ALNSConfig,
                                        "STAGNATION_ESCAPE_WINDOW_RATIO",
                                        0.45,
                                    )
                                ),
                            )
                            cand = lookahead_repair(
                                cand,
                                rng,
                                lookahead_horizon=int(
                                    getattr(
                                        ALNSConfig,
                                        "STAGNATION_ESCAPE_LOOKAHEAD_H",
                                        3,
                                    )
                                ),
                            )
                            cand.compute_inventory()
                            cand_feasible, _ = cand.validate()
                            if cand_feasible:
                                cand_obj = _safe_float(cand.objective())
                                cand_actual = _safe_float(cand.calculate_cost())
                                curr_obj = _safe_float(state.objective())
                                curr_actual = _safe_float(state.calculate_cost())
                                cand_profile = build_state_profile(cand)
                                curr_profile = build_state_profile(state)
                                cand_veh = _used_vehicle_count(cand)
                                curr_veh = _used_vehicle_count(state)

                                rel_worse_search = float(
                                    getattr(
                                        ALNSConfig,
                                        "STAGNATION_ESCAPE_MAX_REL_WORSE_SEARCH",
                                        0.05,
                                    )
                                )
                                rel_worse_actual = float(
                                    getattr(
                                        ALNSConfig,
                                        "STAGNATION_ESCAPE_MAX_REL_WORSE_ACTUAL",
                                        0.0,
                                    )
                                )

                                search_ok = cand_obj <= curr_obj * (
                                    1.0 + rel_worse_search
                                )
                                actual_ok = cand_actual <= curr_actual * (
                                    1.0 + rel_worse_actual
                                )
                                fleet_ok = cand_veh <= curr_veh
                                better_search = cand_obj < curr_obj - 1e-9
                                better_actual = cand_actual < curr_actual - 1e-9

                                if (
                                    fleet_ok
                                    and (better_search or better_actual)
                                    and (search_ok or actual_ok)
                                ):
                                    state.replace_vehicles(list(cand.vehicles))
                                    state.compute_inventory()
                                    setattr(
                                        optimizer,
                                        "_stagnation_escape_cooldown_until",
                                        iter_val
                                        + max(
                                            interval,
                                            int(
                                                getattr(
                                                    ALNSConfig,
                                                    "STAGNATION_ESCAPE_COOLDOWN",
                                                    300,
                                                )
                                            ),
                                        ),
                                    )
                                    optimizer.log_printer.print(
                                        f"[STAGNATION_ESCAPE] it={iteration} accepted strong period-range jump (search={cand_obj:.4f}, actual={cand_actual:.2f}, veh={cand_veh})"
                                    )
                                    record_best_feasible(optimizer, state)
            except Exception:
                logger.exception("Stagnation escape restart failed in callback")

            # Report status every 50 iterations when available.
            if iteration is not None and iteration % 50 == 0:
                status = (
                    stop.get_status() if (stop and hasattr(stop, "get_status")) else {}
                )
                elapsed = status.get("elapsed_time", 0.0)
                optimizer.log_printer.print(
                    f"Iteration {iteration}: elapsed time {elapsed:.1f}s"
                )

            if (
                iteration is not None
                and getattr(ALNSConfig, "ENABLE_ACCEPT_STATS_LOG", True)
                and iteration
                % max(1, int(getattr(ALNSConfig, "ACCEPT_STATS_LOG_INTERVAL", 200)))
                == 0
            ):
                acc = getattr(optimizer, "acceptance", None)
                if acc is not None and hasattr(acc, "get_statistics"):
                    try:
                        st = acc.get_statistics()
                        calls = int(st.get("calls", 0))
                        rej_bound = int(st.get("rejected_by_bound", 0))
                        rej_hard = int(st.get("rejected_by_hard_stage", 0))
                        acc_rate = float(st.get("accept_rate", 0.0)) * 100.0
                        optimizer.log_printer.print(
                            f"[ACCEPT_STATS] it={iteration} calls={calls} accept_rate={acc_rate:.2f}% bound_reject={rej_bound} hard_reject={rej_hard}"
                        )
                    except Exception:
                        logger.exception("Accept stats logging failed")

            # Update the parameter tuner when enabled.

            if getattr(optimizer, "param_tuner", None):
                # Prefer optimizer.get_max_iterations() when available, then tracker, then config fallback
                max_iter = (
                    optimizer.get_max_iterations()
                    if hasattr(optimizer, "get_max_iterations")
                    else None
                )

                if max_iter is None:
                    if hasattr(optimizer, "tracker") and optimizer.tracker is not None:
                        max_iter = optimizer.tracker.get_max_iterations()

                if max_iter is None:
                    max_iter = ALNSConfig.MAX_ITERATIONS

                if iteration is not None and max_iter is not None:
                    if (
                        hasattr(optimizer, "param_tuner")
                        and optimizer.param_tuner is not None
                    ):
                        no_improve_iters = 0
                        if optimizer.tracker is not None and hasattr(
                            optimizer.tracker, "get_statistics"
                        ):
                            try:
                                no_improve_iters = int(
                                    optimizer.tracker.get_statistics().get(
                                        "no_improve_iters", 0
                                    )
                                )
                            except Exception:
                                no_improve_iters = 0
                        optimizer.param_tuner.set_iteration(
                            int(iteration),
                            int(max_iter),
                            no_improve_iters=no_improve_iters,
                        )

            # ML selector handling: retrain submit & warmup sampling
            try:
                if getattr(optimizer, "ml_selector", None) and not fast_mode:
                    retrain_interval = ALNSConfig.ML_RETRAIN_INTERVAL
                    if (
                        iteration is not None
                        and retrain_interval
                        and iteration % retrain_interval == 0
                        and iteration > 0
                    ):
                        try:
                            if not hasattr(optimizer, "_background_executor"):
                                from concurrent.futures import ThreadPoolExecutor

                                optimizer._background_executor = ThreadPoolExecutor(
                                    max_workers=1
                                )
                                optimizer._ml_retrain_future = None

                            future = getattr(optimizer, "_ml_retrain_future", None)

                            if future is None or future.done():
                                if (
                                    optimizer._background_executor is not None
                                    and optimizer.ml_selector is not None
                                ):
                                    optimizer._ml_retrain_future = (
                                        optimizer._background_executor.submit(
                                            optimizer.ml_selector.train_models, True
                                        )
                                    )
                                    optimizer.log_printer.print(
                                        f"Iteration {iteration}: ML model retraining submitted to background worker"
                                    )
                            else:
                                optimizer.log_printer.print(
                                    f"Iteration {iteration}: previous ML training task is still running; skipping submission"
                                )

                        except Exception:
                            logger.exception("Failed to submit ML model training")

                    # Run warmup sampling when sample counts are insufficient.
                    try:
                        min_samples = max(
                            int(getattr(ALNSConfig, "ML_INITIAL_SAMPLE_SIZE", 20)),
                            int(
                                getattr(
                                    ALNSConfig,
                                    "ML_TAKEOVER_MIN_SAMPLE_SIZE",
                                    getattr(ALNSConfig, "ML_INITIAL_SAMPLE_SIZE", 20),
                                )
                            ),
                        )
                        warmup_budget = getattr(ALNSConfig, "ML_WARMUP_PER_ITER", 1)
                        sel = optimizer.ml_selector
                        if (
                            sel is not None
                            and hasattr(sel, "destroy_features")
                            and hasattr(sel, "repair_features")
                            and (
                                len(sel.destroy_features) < min_samples
                                or len(sel.repair_features) < min_samples
                            )
                        ):
                            ml_warmup_sample(optimizer, state, budget=warmup_budget)
                    except Exception:
                        logger.exception("Warmup sampling invocation failed")
            except Exception:
                logger.exception("ML selector handling failed in callback")

            # Finally delegate to tracker.on_iteration when present.
            try:
                if optimizer.tracker is not None and hasattr(
                    optimizer.tracker, "on_iteration"
                ):
                    optimizer.tracker.on_iteration(state, rng, **kwargs)
            except Exception:
                logger.exception("Tracker on_iteration call failed")

        except RuntimeGuardError:
            raise
        except Exception as e:
            logger.exception("callback_on_iteration unexpected error: %s", e)

    return callback_on_iteration


def register_callbacks(
    alns_instance: ALNS, optimizer: ALNSOptimizer, stop: Any | None = None
):
    """
    Register callbacks on the ALNS instance.

    ``on_best`` uses the combined callback, while ``on_better``, ``on_accept``,
    and ``on_reject`` route through ``on_iteration``.
    """
    # Build callbacks with the expected signatures.
    on_best = make_callback_on_best(optimizer)
    on_iter = make_callback_on_iteration(optimizer, stop=stop)

    def _set_incumbent_snapshot(state: SolutionState) -> None:
        try:
            optimizer._incumbent_state_snapshot = safe_state_snapshot(state)
        except Exception:
            pass

    # On a new global best, run on_best before on_iter to preserve behavior.
    def callback_on_best_combined(state, rng, **kwargs):
        _set_incumbent_snapshot(state)
        try:
            on_best(state, rng, **kwargs)
        except RuntimeGuardError:
            raise
        except Exception:
            optimizer.log_printer.print("Callback on best failed.", color="yellow")

        try:
            on_iter(state, rng, **kwargs)
        except RuntimeGuardError:
            raise
        except Exception:
            optimizer.log_printer.print(
                "Callback on iteration (from best) failed.", color="yellow"
            )

    def callback_on_accept_or_better(state, rng, **kwargs):
        _set_incumbent_snapshot(state)
        try:
            on_iter(state, rng, **kwargs)
        except RuntimeGuardError:
            raise
        except Exception:
            optimizer.log_printer.print("Callback on iteration failed.", color="yellow")

    def callback_on_reject(state, rng, **kwargs):
        # Some ALNS implementations pass rejected candidate to on_reject.
        # For tracking/intensification we must operate on incumbent state.
        incumbent = getattr(optimizer, "_incumbent_state_snapshot", None)
        if incumbent is None:
            incumbent = state
        else:
            try:
                incumbent = safe_state_snapshot(incumbent)
            except Exception:
                pass

        try:
            on_iter(incumbent, rng, **kwargs)
        except RuntimeGuardError:
            raise
        except Exception:
            optimizer.log_printer.print(
                "Callback on iteration (from reject) failed.", color="yellow"
            )

    try:
        alns_instance.on_best(callback_on_best_combined)
        alns_instance.on_better(callback_on_accept_or_better)
        alns_instance.on_accept(callback_on_accept_or_better)
        alns_instance.on_reject(callback_on_reject)
    except Exception:
        logger.exception("Failed to register callbacks on ALNS instance")
