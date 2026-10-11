"""Compatibility wrapper for the historical ``multi_period_delivery.main`` path."""

from .alns.main import run_model

__all__ = ["run_model"]


if __name__ == "__main__":
    run_model()
