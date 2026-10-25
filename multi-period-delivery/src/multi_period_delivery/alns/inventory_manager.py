"""
inventory_manager.py

A minimal unified inventory manager used by callbacks/repair paths.
It centralizes inventory rebuild, negative-inventory scanning, and
basic resource-pool snapshot generation.
"""

from __future__ import annotations

import logging
from typing import Any

from .errors import STATE_OPERATION_ERRORS

logger = logging.getLogger(__name__)


class UnifiedInventoryManager:
    """Lightweight inventory facade for SolutionState."""

    def __init__(self, eps: float = 1e-6):
        self.eps = float(eps)

    def rebuild_inventory(self, state: Any) -> bool:
        """Rebuild inventory baseline from vehicles and production."""
        try:
            return bool(state.compute_inventory())
        except STATE_OPERATION_ERRORS:
            return False

    def scan_negative_inventory(
        self, state: Any
    ) -> list[tuple[tuple[str, str, int], float]]:
        """Return all (index, value) entries with negative inventory."""
        negatives: list[tuple[tuple[str, str, int], float]] = []
        try:
            for k, v in state.s_ikt.items():
                if float(v) < -self.eps:
                    negatives.append((k, float(v)))
        except STATE_OPERATION_ERRORS:
            return []
        return negatives

    def auto_repair_inventory_inconsistency(
        self, state: Any, context: str = ""
    ) -> bool:
        """
        Attempt a conservative repair by rebuilding inventory once.
        Returns True only if no negative inventory remains after rebuild.
        """
        ok = self.rebuild_inventory(state)
        if not ok:
            return False
        return len(self.scan_negative_inventory(state)) == 0

    def build_resource_pool(self, state: Any) -> dict[tuple[str, str, int], int]:
        """
        Return the available inventory pool for periods 1..H.

        This mirrors the repair-path ``build_resource_pool`` semantics. A
        shipment on day ``d`` lowers every closing balance from ``d`` onwards,
        so availability on ``d`` is the smallest balance over that suffix.
        Using only ``d``'s balance lets a repair consume stock already reserved
        for a later period.
        """
        pool: dict[tuple[str, str, int], int] = {}
        try:
            data = state.data
            if not getattr(state, "s_initialized", False):
                state.compute_inventory()

            for plant in data.plants:
                for sku in data.skus_plant.get(plant, ()):
                    suffix_min: int | None = None
                    for day in range(data.horizons, 0, -1):
                        balance = int(state.s_ikt.get((plant, sku, day), 0))
                        suffix_min = (
                            balance
                            if suffix_min is None
                            else min(suffix_min, balance)
                        )
                        pool[(plant, sku, day)] = max(0, suffix_min)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            # Returning an empty pool here meant "no inventory anywhere", which
            # the allocation operators read as a legitimate state and then
            # silently skipped every insertion - degrading the search to pure
            # deletion with no signal that anything went wrong. An empty pool is
            # also indistinguishable from a genuinely empty one, so record the
            # failure instead of hiding it. (No in-repo caller uses this method;
            # the repair operators use repair_utils.build_resource_pool, which
            # does not swallow errors at all.)
            logger.warning(
                "build_resource_pool failed; returning an empty pool: %s", exc
            )
            return {}
        return pool
