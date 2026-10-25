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
    from .alns.InputDataALNS import DataALNS as DataALNS
    __all__.append("DataALNS")
except ImportError:
    # Skip the export if an optional dependency is missing or the module fails.
    pass

try:
    from .alns.OutputDataALNS import OutPutData as OutPutData
    __all__.append("OutPutData")
except ImportError:
    pass

# Combined stopping criteria and AND/OR expression helpers.
try:
    from .alns.combined_stopping import (
        AND,
        OR,
        CombinedStoppingCriterion,
        create_standard_combined_criterion,
    )
    __all__ += ["CombinedStoppingCriterion", "AND", "OR", "create_standard_combined_criterion"]
except ImportError:
    pass

# Solution consistency-check entry point.
try:
    from .alns.check_solution import check_solution as check_solution
    __all__.append("check_solution")
except ImportError:
    pass

# Configuration object.
try:
    # Export ALNSConfig for backward compatibility while referencing the
    # module-level default_config instance.
    from .alns.alns_config import default_config

    ALNSConfig = default_config
    __all__.append("ALNSConfig")
except ImportError:
    pass


def run_model(*args, **kwargs):
    """Run the multi-period optimizer via a lazy package-level import."""
    from .alns.main import run_model as _run_model

    return _run_model(*args, **kwargs)


__all__.append("run_model")


def run_cplex_api_monolithic(*args, **kwargs):
    """Run the multi-period CPLEX Python API monolithic model."""
    from .cplex_api_monolithic.runner import (
        run_cplex_api_monolithic as _run,
    )

    return _run(*args, **kwargs)


__all__.append("run_cplex_api_monolithic")
