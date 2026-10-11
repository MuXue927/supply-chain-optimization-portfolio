"""Installable package for single-period delivery optimization."""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"
__all__ = [
    "run_column_generation",
    "run_heuristic",
    "run_monolithic",
]


def run_monolithic(*args: Any, **kwargs: Any):
    """Run the monolithic integer-programming model."""
    from .monolithic.optLogicSingle import run_model

    return run_model(*args, **kwargs)


def run_heuristic(*args: Any, **kwargs: Any):
    """Run the two-stage constructive heuristic."""
    from .heuristic.main import run_model

    return run_model(*args, **kwargs)


def run_column_generation(*args: Any, **kwargs: Any):
    """Run the column-generation algorithm."""
    from .column_generation.OPTLogicCG import run_model

    return run_model(*args, **kwargs)
