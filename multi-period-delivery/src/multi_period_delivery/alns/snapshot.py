"""Fail-closed helpers for taking independent solution snapshots."""

from __future__ import annotations

import copy
from typing import Any


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
    except Exception as exc:
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
