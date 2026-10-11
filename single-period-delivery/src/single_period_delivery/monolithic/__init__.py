"""Monolithic integer-programming solver for the single-period problem."""

from typing import Any


def run_model(*args: Any, **kwargs: Any):
    """Load and run the Gurobi monolithic implementation lazily."""
    from .optLogicSingle import run_model as _run_model

    return _run_model(*args, **kwargs)

__all__ = ["run_model"]
