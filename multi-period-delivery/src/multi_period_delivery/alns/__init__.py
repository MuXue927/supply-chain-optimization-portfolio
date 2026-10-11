"""ALNS implementation for the multi-period delivery problem.

The package root exposes the supported public interface. This subpackage keeps
the search implementation at the same structural level as ``monolithic`` and
``cplex_api_monolithic``.
"""

from importlib import import_module
from typing import Any

__all__ = ["ALNSConfig", "DataALNS", "OutPutData"]

_EXPORTS = {
    "ALNSConfig": (".alns_config", "default_config"),
    "DataALNS": (".InputDataALNS", "DataALNS"),
    "OutPutData": (".OutputDataALNS", "OutPutData"),
}


def __getattr__(name: str) -> Any:
    """Load public ALNS objects lazily to keep package import lightweight."""
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
