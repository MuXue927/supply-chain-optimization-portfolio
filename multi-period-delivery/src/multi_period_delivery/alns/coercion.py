"""
Numeric coercion helpers shared by the destroy and repair operators.

These were implemented three times (``destroy_operators``, ``destroy_utils`` and
``repair_operators``) with the same silent defect: converting a value that is not
directly numeric fell back to ``np.array(value).flat[0]``, so a **multi-element
array became its first element**. ``np.array([1, 2])`` therefore produced ``1`` -
a plausible-looking number produced from a caller bug, which is far harder to
notice than an exception.

The behaviour here is deliberate and documented:

===========================  ==================================================
input                        result
===========================  ==================================================
``None``                     the supplied default
``np.int64(7)``              7 (numpy scalars are fine)
``np.array(9)``              9 (0-d array)
``np.array([11])``           11 (1-element array)
``np.array([1, 2])``         ``ValueError`` - refused, not truncated
``float("nan")``             the supplied default (NaN breaks comparisons)
``"abc"`` / ``object()``     the supplied default
===========================  ==================================================

Only ``TypeError``/``ValueError`` are treated as "unusable value"; anything else
(``MemoryError``, ``KeyboardInterrupt``, a bug in a ``__float__`` implementation)
propagates, because swallowing it would hide a real fault.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_scalar(value: Any) -> float | None:
    """
    Return ``value`` as a float when it is a single number, else None.

    Arrays are reduced explicitly rather than handed to ``float()``: NumPy 1.25
    deprecates converting an ndim > 0 array to a scalar (it warns today, errors in
    a future release), so a 1-element array must be unwrapped here.

    Raises ValueError for a genuine multi-element array.
    """
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        if value.size != 1:
            raise ValueError(
                f"cannot coerce a {value.size}-element array to a scalar; "
                "pass a single value"
            )
        try:
            return float(value.flat[0])
        except (TypeError, ValueError):
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def scalar_from_array(value: Any) -> float | None:
    """
    Return ``value`` as a single float when it can be read as one number.

    Kept for callers that want a best-effort conversion without the multi-element
    refusal; prefer :func:`coerce_float`/:func:`coerce_int`.
    """
    return _as_scalar(value)


def coerce_float(value: Any, default: float = 0.0) -> float:
    """
    Return ``value`` as a float, or ``default`` when it is unusable.

    ``None`` and NaN yield ``default``: a NaN would otherwise propagate into
    comparisons (``score > best`` is always false for NaN) and change behaviour
    without any error being raised.
    """
    if value is None:
        return float(default)
    result = _as_scalar(value)
    if result is None or result != result:  # unusable, or NaN
        return float(default)
    return result


def coerce_int(value: Any, default: int = 0) -> int:
    """
    Return ``value`` as an int, or ``default`` when it is unusable.

    NaN yields ``default`` for the same reason as in :func:`coerce_float`:
    ``int(float("nan"))`` raises, and letting that escape would turn a
    "not supplied" value into a crash.
    """
    if value is None:
        return int(default)
    result = _as_scalar(value)
    if result is None or result != result:  # unusable, or NaN
        return int(default)
    return int(result)
