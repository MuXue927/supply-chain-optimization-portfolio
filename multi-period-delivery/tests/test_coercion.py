"""Tests for the shared numeric coercion helpers.

Three copies of these functions existed (destroy_operators, destroy_utils,
repair_operators) and all three silently truncated a multi-element array to its
first element: ``np.array([1, 2])`` became ``1``. A plausible-looking number
produced from a caller bug is far harder to notice than an exception, so the
shared implementation refuses it.

These tests pin the whole contract, including the NaN behaviour: NaN used to pass
through ``_coerce_float`` and then make every ``score > best`` comparison false,
changing behaviour with no error at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from multi_period_delivery.alns.coercion import coerce_float, coerce_int
from multi_period_delivery.alns.destroy_utils import ensure_pyint, ensure_pylist

# ---------------------------------------------------------------------------
# The truncation defect
# ---------------------------------------------------------------------------


def test_multi_element_array_is_refused_not_truncated() -> None:
    with pytest.raises(ValueError, match="2-element array"):
        coerce_int(np.array([1, 2]))
    with pytest.raises(ValueError, match="2-element array"):
        coerce_float(np.array([1, 2]))


def test_larger_arrays_are_refused_too() -> None:
    with pytest.raises(ValueError):
        coerce_int(np.array([[1, 2], [3, 4]]))


# ---------------------------------------------------------------------------
# Scalar-like inputs are accepted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected_int", "expected_float"),
    [
        (7, 7, 7.0),
        (7.5, 7, 7.5),
        ("8", 8, 8.0),
        (np.int64(7), 7, 7.0),
        (np.float64(2.5), 2, 2.5),
        (np.array(9), 9, 9.0),
        (np.array([11]), 11, 11.0),
    ],
)
def test_scalar_like_values_convert(
    value: object, expected_int: int, expected_float: float
) -> None:
    assert coerce_int(value) == expected_int
    assert coerce_float(value) == pytest.approx(expected_float)


# ---------------------------------------------------------------------------
# Unusable inputs fall back to the default
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, "abc", object(), np.array([])])
def test_unusable_values_yield_the_default(value: object) -> None:
    assert coerce_int(value) == 0
    assert coerce_float(value) == 0.0
    assert coerce_int(value, default=5) == 5
    assert coerce_float(value, default=2.5) == pytest.approx(2.5)


@pytest.mark.parametrize("value", [float("nan"), np.nan, np.float64("nan")])
def test_nan_yields_the_default(value: object) -> None:
    """NaN must not propagate into comparisons."""
    assert coerce_float(value) == 0.0
    assert coerce_int(value) == 0
    assert coerce_float(value, default=3.0) == pytest.approx(3.0)


def test_infinity_is_preserved_as_a_number() -> None:
    """Inf is a legitimate value here; only NaN is meaningless."""
    assert coerce_float(float("inf")) == float("inf")
    with pytest.raises((ValueError, OverflowError)):
        coerce_int(float("inf"))


# ---------------------------------------------------------------------------
# The sibling helpers must agree with each other
# ---------------------------------------------------------------------------


def test_zero_d_array_means_the_same_to_both_helpers() -> None:
    """These disagreed: ensure_pyint said 9, ensure_pylist said []."""
    assert ensure_pyint(np.array(9)) == 9
    assert ensure_pylist(np.array(9)) == [9]


def test_ensure_pylist_flattens_a_1d_array() -> None:
    assert ensure_pylist(np.array([1, 2])) == [1, 2]
    assert ensure_pylist([3, 4]) == [3, 4]
    assert ensure_pylist(5) == [5]


def test_ensure_pyint_falls_back_to_zero() -> None:
    assert ensure_pyint(None) == 0
    assert ensure_pyint("abc") == 0
    assert ensure_pyint(np.array([])) == 0
    assert ensure_pyint([]) == 0
