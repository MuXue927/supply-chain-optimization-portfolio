"""Guard the ALNS configuration surface.

The ALNS modules read many settings through
``getattr(ALNSConfig, "NAME", <fallback>)``, where ``ALNSConfig`` is the shared
``default_config`` instance. A name that is not declared on the dataclass still
works at runtime because the fallback supplies a value, but it stays invisible
in ``alns_config.py``: reading the configuration file never reveals that the
knob exists or what its default is.

This test walks the ALNS modules with ``ast`` (so comments and docstrings cannot
produce false positives) and asserts that every name read from ``ALNSConfig`` is
a declared field of the dataclass. Add the field to ``alns_config.py`` instead of
relying on an inline fallback.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path
from typing import Dict, Set

import multi_period_delivery.alns as alns_package

from multi_period_delivery.alns.alns_config import ALNSConfig


def _names_read_from_config(path: Path) -> Set[str]:
    """Return every string literal passed to ``getattr(ALNSConfig, ...)``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: Set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Name) and node.func.id == "getattr"):
            continue
        if len(node.args) < 2:
            continue
        target, name = node.args[0], node.args[1]
        if not (isinstance(target, ast.Name) and target.id == "ALNSConfig"):
            continue
        # The name is usually a constant, but callbacks use a
        # "A if condition else B" expression; collect every branch.
        for literal in ast.walk(name):
            if isinstance(literal, ast.Constant) and isinstance(literal.value, str):
                found.add(literal.value)
    return found


def test_every_config_name_read_via_getattr_is_declared() -> None:
    alns_dir = Path(alns_package.__file__).resolve().parent
    declared = {field.name for field in dataclasses.fields(ALNSConfig)}

    undeclared: Dict[str, list[str]] = {}
    for path in sorted(alns_dir.glob("*.py")):
        missing = sorted(_names_read_from_config(path) - declared)
        if missing:
            undeclared[path.name] = missing

    assert undeclared == {}, (
        "These names are read from ALNSConfig but never declared in "
        f"alns_config.py: {undeclared}"
    )
