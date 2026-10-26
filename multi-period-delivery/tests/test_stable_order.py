"""Deterministic ordering for set-valued instance data.

Python randomises string hashing per process. Two runs of the same seed used to
disagree (basic/medium/s7 at 120 iterations: 791.74 vs 795.43 with ML off), and
the whole difference traced back to iterating ``set[str]`` containers whose
order feeds candidate generation.
"""

from __future__ import annotations

from types import SimpleNamespace

from multi_period_delivery.alns.stable_order import (
    stable_ids,
    stable_skus_by_size,
    stable_vehicle_types,
)


def _data(types: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        all_veh_types=set(types),
        veh_type_cap={"small": 10, "medium": 20, "large": 20},
        sku_sizes={"a": 3.0, "b": 5.0, "c": 5.0},
    )


def test_stable_ids_drops_none_and_sorts() -> None:
    assert stable_ids({"b", "a", None, "c"}) == ["a", "b", "c"]
    assert stable_ids(None) == []


def test_vehicle_type_order_does_not_depend_on_set_order() -> None:
    """Same members, different insertion order, identical output."""
    first = _data(["medium", "large", "small"])
    second = _data(["small", "large", "medium"])

    assert stable_vehicle_types(first) == stable_vehicle_types(second)
    assert stable_vehicle_types(first, largest_first=True) == stable_vehicle_types(
        second, largest_first=True
    )


def test_equal_capacity_types_break_ties_by_identifier() -> None:
    """``sorted(reverse=True)`` kept set order on ties, which is hash order."""
    data = _data(["medium", "large", "small"])

    order = stable_vehicle_types(data, largest_first=True)

    # large/medium share capacity 20: the name decides, deterministically.
    assert order == ["large", "medium", "small"]


def test_sku_size_ties_break_by_identifier() -> None:
    data = _data(["small"])

    assert stable_skus_by_size(data, {"b", "c", "a"}) == ["b", "c", "a"]
    assert stable_skus_by_size(data, {"a", "c", "b"}) == ["b", "c", "a"]
