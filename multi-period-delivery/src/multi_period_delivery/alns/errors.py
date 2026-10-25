"""Exception surfaces shared by the ALNS modules.

This module is deliberately a leaf: it imports nothing from the package, so any
module may use these tuples without creating an import cycle (``repair_utils``
imports ``alnsopt``, ``param_tuner``, ``snapshot`` and ``vehicle``, so the
constant cannot live there).
"""

from __future__ import annotations

#: The complete exception surface of a state/vehicle helper call.
#:
#: ``repair_utils`` (``ValueError`` at :65, ``RuntimeError`` at :318,
#: ``KeyError`` at :533/:540), ``alnsopt`` (``RuntimeError`` at :942) and
#: ``vehicle`` (``KeyError`` at :196/:290, ``ValueError`` at :443/:454,
#: ``RuntimeError`` at :471) raise exactly these three types explicitly.
#: Attribute access on a malformed state adds ``AttributeError``, and
#: arithmetic or coercion on a malformed value adds ``TypeError``.
#:
#: A handler whose try body only calls those helpers can therefore catch this
#: tuple instead of ``Exception``: real defects with any other type - including
#: ``MemoryError``, ``RecursionError`` and ``KeyboardInterrupt`` - stay visible
#: instead of being absorbed.
STATE_OPERATION_ERRORS = (
    AttributeError,
    KeyError,
    RuntimeError,
    TypeError,
    ValueError,
)
