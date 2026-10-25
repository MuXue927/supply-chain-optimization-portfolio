"""Fail-closed helpers for taking independent solution snapshots."""

from __future__ import annotations

import copy
from typing import Any

from .errors import STATE_OPERATION_ERRORS
from .swallowed import record_swallowed


def safe_state_snapshot(state: Any) -> Any:
    """Return an independent snapshot or raise instead of sharing live state."""
    if state is None:
        return None

    snapshot = None
    copy_error: Exception | None = None
    try:
        copy_method = getattr(state, "copy", None)
        if callable(copy_method):
            snapshot = copy_method()
            if snapshot is not None and snapshot is not state:
                return snapshot
    except STATE_OPERATION_ERRORS as exc:
        copy_error = exc

    try:
        snapshot = copy.deepcopy(state)
        if snapshot is not None and snapshot is not state:
            return snapshot
    except Exception as deepcopy_error:
        raise RuntimeError(
            "Could not create an independent state snapshot for "
            f"{type(state).__name__}"
        ) from deepcopy_error

    message = (
        "Could not create an independent state snapshot for "
        f"{type(state).__name__}: copy/deepcopy returned the original or None"
    )
    if copy_error is not None:
        message += f" (copy error: {type(copy_error).__name__}: {copy_error})"
    raise RuntimeError(message)


def replace_state_snapshot(
    owner: Any,
    attribute_name: str,
    state: Any,
    *,
    site: str,
) -> bool:
    """Replace a snapshot attribute, clearing stale state on copy failure."""
    try:
        setattr(owner, attribute_name, safe_state_snapshot(state))
        return True
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        try:
            setattr(owner, attribute_name, None)
        except (AttributeError, TypeError):
            pass
        record_swallowed(site, exc)
        return False


def restore_state_snapshot(target: Any, snapshot: Any) -> None:
    """Restore solution content in-place while preserving the live object."""
    restored = snapshot.copy()
    current_audit = {
        key: dict(value) for key, value in getattr(target, "repair_audit", {}).items()
    }

    target.data = restored.data
    target.vehicles = restored.vehicles
    target.s_ikt = restored.s_ikt
    target.s_indices = restored.s_indices
    target.s_initialized = restored.s_initialized
    target.shipped_cache = restored.shipped_cache
    target.last_objective = restored.last_objective
    target.objective_dirty = True
    target._suppress_compute_inventory = False
    target.repair_audit = current_audit
    target._inventory_cache = None
    target._validation_memo = None
    target._validation_memo_enabled = False

    for cache_name in (
        "_vehicles_by_plant_day_map",
        "_resource_pool_cache",
        "_unmet_cache",
    ):
        try:
            delattr(target, cache_name)
        except (AttributeError, TypeError):
            pass
