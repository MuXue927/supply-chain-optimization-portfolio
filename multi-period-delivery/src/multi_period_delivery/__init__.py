"""
multi_period_delivery package initialization.

Module role
    - Make multi_period_delivery an importable Python package.
    - Re-export common objects for convenient imports such as
      `from multi_period_delivery import X`.
    - Stay lightweight and robust by using lazy/fault-tolerant imports so
      optional dependency failures do not break the package import.

Usage notes
    - Export only common, stable interfaces. Submodules with heavy or optional
      dependencies are wrapped in try/except.
    - Export failures do not affect basic package availability, but the
      corresponding symbol will not be available.
"""

# Build the export list based on availability so it never advertises a symbol
# that failed to import.
__all__ = []

# Lightweight, commonly used data I/O classes.
try:
    from .InputDataALNS import DataALNS
    __all__.append("DataALNS")
except Exception:
    # Skip the export if an optional dependency is missing or the module fails.
    pass

try:
    from .OutputDataALNS import OutPutData
    __all__.append("OutPutData")
except Exception:
    pass

# Combined stopping criteria and AND/OR expression helpers.
try:
    from .combined_stopping import CombinedStoppingCriterion, AND, OR, create_standard_combined_criterion
    __all__ += ["CombinedStoppingCriterion", "AND", "OR", "create_standard_combined_criterion"]
except Exception:
    pass

# Solution consistency-check entry point.
try:
    from .check_solution import check_solution
    __all__.append("check_solution")
except Exception:
    pass

# Configuration object.
try:
    # Export ALNSConfig for backward compatibility while referencing the
    # module-level default_config instance.
    from .alns_config import default_config as ALNSConfig
    __all__.append("ALNSConfig")
except Exception:
    pass


def run_model(*args, **kwargs):
    """Run the multi-period optimizer via a lazy package-level import."""
    from .main import run_model as _run_model

    return _run_model(*args, **kwargs)


__all__.append("run_model")
