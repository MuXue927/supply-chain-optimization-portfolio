"""
inventory_manager.py

A minimal unified inventory manager used by callbacks/repair paths.
It centralizes inventory rebuild, negative-inventory scanning, and
basic resource-pool snapshot generation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


class UnifiedInventoryManager:
    """Lightweight inventory facade for SolutionState."""

    def __init__(self, eps: float = 1e-6):
        self.eps = float(eps)

    def rebuild_inventory(self, state: Any) -> bool:
        """Rebuild inventory baseline from vehicles and production."""
        try:
            return bool(state.compute_inventory())
        except Exception:
            return False

    def scan_negative_inventory(
        self, state: Any
    ) -> List[Tuple[Tuple[str, str, int], float]]:
        """Return all (index, value) entries with negative inventory."""
        negatives: List[Tuple[Tuple[str, str, int], float]] = []
        try:
            for k, v in state.s_ikt.items():
                if float(v) < -self.eps:
                    negatives.append((k, float(v)))
        except Exception:
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

    def build_resource_pool(self, state: Any) -> Dict[Tuple[str, str, int], float]:
        """Return a simple resource pool snapshot based on state inventory map."""
        pool: Dict[Tuple[str, str, int], float] = {}
        try:
            for (plant, sku, day), inv in state.s_ikt.items():
                if day <= 0:
                    continue
                pool[(plant, sku, day)] = max(0.0, float(inv))
        except Exception:
            return {}
        return pool
