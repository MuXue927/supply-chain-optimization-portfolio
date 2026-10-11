"""monolithic package initializer.

Make the monolithic directory importable as a Python package. Keep this file minimal
to avoid import-time side effects; try to provide convenient exports if available.

This file is safe to extend later if you want to expose more symbols at package level.
"""

__all__ = ["InputDataMultiple", "MonolithicModelMultiple", "optLogicMultiple"]

# Lazy, best-effort imports to allow `from multi_period_delivery.monolithic import InputDataMultiple` without
# forcing heavy import-time side effects. Fail silently to keep import robust.
try:
    from .InputDataMultiple import InputDataMultiple  # type: ignore
except Exception:
    pass

try:
    from .MonolithicModelMultiple import MonolithicModelMultiple  # type: ignore
except Exception:
    pass

# optLogicMultiple contains runner helpers (run_model). Import best-effort.
try:
    from . import optLogicMultiple  # type: ignore
except Exception:
    pass
