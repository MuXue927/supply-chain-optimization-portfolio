"""Deterministic iteration order for set-valued instance data.

Python randomises string hashing per process, so iterating a ``set[str]``
visits its members in a different order in every run. The search consumes those
orders directly: ``regret_based_repair`` walks vehicle types to build candidate
loads, and the type-replacement repair takes the *first* acceptable type. Two
runs with the same seed therefore followed different trajectories (measured:
basic/medium/s7 at 120 iterations gave 791.74 and 795.43 with no ML involved at
all), which made every cross-process comparison partly a measurement of hash
order.

Ints are unaffected (``hash(int) == int``), so only the string-keyed containers
need this. Ordering keys are chosen to be intrinsic to the instance - capacity,
size, identifier - never to insertion or hash order.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def stable_ids(values: Iterable[Any] | None) -> list[str]:
    """Return identifiers in ascending string order, dropping ``None``."""
    if not values:
        return []
    return sorted(str(value) for value in values if value is not None)


def stable_vehicle_types(data: Any, *, largest_first: bool = False) -> list[str]:
    """Vehicle types in a reproducible order.

    ``largest_first`` matches the "biggest capacity first" greedy that the
    repair operators want; the identifier is the tie-break so two types with the
    same capacity cannot swap places between runs.
    """
    types = stable_ids(getattr(data, "all_veh_types", None))
    if not largest_first:
        return types
    caps = getattr(data, "veh_type_cap", {}) or {}

    def key(veh_type: str) -> tuple[float, str]:
        try:
            capacity = float(caps.get(veh_type, 0.0))
        except (TypeError, ValueError):
            capacity = 0.0
        return (-capacity, veh_type)

    return sorted(types, key=key)


def stable_skus_by_size(data: Any, skus: Iterable[Any] | None) -> list[str]:
    """SKUs largest-first, with the identifier breaking size ties."""
    sizes = getattr(data, "sku_sizes", {}) or {}

    def key(sku: str) -> tuple[float, str]:
        try:
            size = float(sizes.get(sku, 0.0))
        except (TypeError, ValueError):
            size = 0.0
        return (-size, sku)

    return sorted(stable_ids(skus), key=key)
