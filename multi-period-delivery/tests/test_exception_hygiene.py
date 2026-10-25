"""Ratchet on the number of blind ``except Exception`` handlers.

Background: the audit of 2026-10-16 found 419 bare ``except Exception`` handlers
in the ALNS package alone. Most are deliberate resilience (one bad candidate must
not abort the search), so deleting them would be wrong - but a blind handler also
hides defects, and the count had been growing unchecked.

Enabling ruff's ``BLE001`` cannot gate this today: the existing handlers would all
need ``# noqa`` comments, which documents the debt without reducing it. A ratchet
does the job instead - the count may fall, never rise.

**When this test fails:** you added a blind ``except Exception``. Prefer catching
the exceptions the body can actually raise. If the breadth is genuinely intended
(optional dependency, best-effort cache, one-of-many-candidates), it is still
allowed - lower :data:`BLIND_EXCEPT_BASELINE` is not required, but the baseline
must be updated deliberately so the growth is visible in review.

The measured floors: **410** after the 2026-10-16 pass (from 467 before it),
across both packages (multi-period 401, single-period 9). That pass narrowed the
53 handlers whose try bodies contain only ``float``/``int``/``bool``/``len``
calls, and deleted 12 handlers that wrapped a plain attribute assignment in a
``try`` - ``SolutionState`` has no ``__slots__`` and no custom ``__setattr__``, so
those could never fire.

The 2026-10-17 pass lowered the count further (410 -> 381 -> 368) by narrowing
handlers whose try bodies are builtin coercions (``float``/``int``/``len``) plus
3-arg ``getattr`` (which cannot raise ``AttributeError``), and a handful of
manually reviewed handlers with an explicit, small exception set (dict subscripts,
``setattr``, ``json.load`` + ``float``).

The 2026-10-19/23 maintenance passes lowered 368 -> 350 -> 348 -> 343 -> 342 by narrowing cache cleanup
(``delattr``/attribute assignment), internal ``set_max_iterations`` coercions,
tracker capacity/profile conversion, CSV append, used-vehicle iteration, and the
inventory-freshness fail-closed check, then consolidating two duplicate
snapshot helpers into one fail-closed implementation and making runtime-guard
rollback failures and incumbent snapshot failures explicit. The acceptance
boundary audit then removed the nested fallback handler in
``DemandConstraintAccept`` (the acceptor's calling convention is now resolved
from its signature), taking 342 -> 341. The optimizer post-processing pass then
narrowed the dynamic-cap and final-gap handlers and made the final
light-consolidation failure restore its snapshot, taking 341 -> 336. The
callbacks/repair-utils pass narrowed 15 handlers whose try bodies only coerce
values or touch attributes, and made the post-repair fill/consolidation
failures observable, taking 336 -> 321. The operator pass narrowed 31 more
(``rng.choice``/``np.atleast_1d``/``int``/``float`` conversions,
``is_empty``/``is_full`` probes, dict copies, and cache invalidation), taking
321 -> 290. The destroy-utils/tracker/ML pass extracted the repeated
``OPJSON`` sampling block into one helper and narrowed a further 22 handlers,
taking 290 -> 268. The operator-site pass made the state-modifying calls
observable (greedy fast/slow load, fill-minimum-load inventory and supply
chain, local-search trial, sequential/lookahead shipped cache) and fixed the
regret fast-estimator fallback, taking 268 -> 251. The callback pass made the
light-intensify, elite-relink, stagnation-escape, tracker and registration
handlers countable, and fixed fleet consolidation returning an uncertified
merge, taking 251 -> 249.

The 2026-10-25 pass took 249 -> 113. It first established that the state and
vehicle helpers have a *closed* explicit raise surface - ``ValueError``,
``RuntimeError`` and ``KeyError`` in ``repair_utils``/``alnsopt``/``vehicle``,
plus ``AttributeError`` from attribute access and ``TypeError`` from arithmetic -
and recorded it once as ``alns.errors.STATE_OPERATION_ERRORS``. Every handler
whose try body only calls those helpers now catches that tuple, including the
two large operator bodies in ``regret_based_repair`` and the sequential/lookahead
loops. The pass also deleted handlers that guarded work which cannot raise:
``logger.debug(...)`` calls, ``print(...)`` diagnostics, and plain ``setattr``
on ``SolutionState`` (no ``__slots__``, no custom ``__setattr__``).

The 113 that remain are deliberate resilience boundaries with an
*open* exception surface:

- callback and operator isolation (``callbacks`` 21, ``alns_optimizer`` 18) -
  one failing candidate, callback, feature block or operator must not abort the
  whole search;
- lazy imports that avoid an import cycle, or optional dependencies
  (``numba``, ``sklearn``, ``matplotlib``) - those raise ``ImportError`` and
  worse;
- file/pandas I/O and CLI/report generation - ``OSError`` and pandas' own
  hierarchy are expected there;
- snapshot ``deepcopy`` and config-expression evaluation, where ``MemoryError``
  or ``RecursionError`` must still be caught to keep the fallback reachable;
- ``monolithic/``, ``cplex_api_*`` and ``single-period-delivery/`` (44 in
  total), which are outside this maintenance scope.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

#: Highest accepted number of ``except Exception`` handlers across both packages.
#: Lowered from 410 to 394 when ``post_repair``'s ~70-line try was extracted into
#: ``_check_inserted_feasibility`` (the extraction also removed redundant inner
#: handlers), to 381 in the follow-up narrowing pass, and to 368 on 2026-10-17.
#: The 2026-10-22/23 passes lowered it to 348, 343, 342, 341, 336, 321, 290,
#: 268, 251, then 249. The 2026-10-25 pass lowered it to 113 by catching
#: ``STATE_OPERATION_ERRORS`` wherever the try body only calls state/vehicle
#: helpers, and deleting guards around logging, ``print`` and plain ``setattr``.
#: Keep this ratcheted down as handlers are narrowed.
BLIND_EXCEPT_BASELINE = 113

_PACKAGES = ("multi-period-delivery", "single-period-delivery")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def count_blind_excepts() -> dict[str, int]:
    """Count ``except Exception`` per package (``except (A, B)`` is not blind)."""
    counts: dict[str, int] = {}
    for package in _PACKAGES:
        source_root = _repo_root() / package / "src"
        found = 0
        for path in sorted(source_root.rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:  # pragma: no cover - defensive
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Try):
                    continue
                for handler in node.handlers:
                    if isinstance(handler.type, ast.Name) and handler.type.id == "Exception":
                        found += 1
        counts[package] = found
    return counts


def test_blind_except_count_does_not_grow() -> None:
    counts = count_blind_excepts()
    total = sum(counts.values())

    detail = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
    assert total <= BLIND_EXCEPT_BASELINE, (
        f"blind `except Exception` handlers grew to {total} "
        f"(baseline {BLIND_EXCEPT_BASELINE}; {detail}).\n"
        "Catch the exceptions the body can actually raise, or update "
        "BLIND_EXCEPT_BASELINE deliberately so the growth is reviewed."
    )


def test_baseline_is_not_stale() -> None:
    """A baseline far above reality hides the next regression in the slack."""
    total = sum(count_blind_excepts().values())

    assert total >= BLIND_EXCEPT_BASELINE - 40, (
        f"blind handler count fell to {total} but the baseline is still "
        f"{BLIND_EXCEPT_BASELINE}; lower it so the ratchet keeps its bite."
    )


def test_counter_detects_a_blind_handler(tmp_path: Path) -> None:
    """The counter must ignore narrowed handlers and count blind ones."""
    sample = tmp_path / "sample.py"
    sample.write_text(
        "try:\n"
        "    a = 1\n"
        "except Exception:\n"
        "    pass\n"
        "try:\n"
        "    b = 2\n"
        "except (TypeError, ValueError):\n"
        "    pass\n",
        encoding="utf-8",
    )
    tree = ast.parse(sample.read_text(encoding="utf-8"))

    blind = sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        for handler in node.handlers
        if isinstance(handler.type, ast.Name) and handler.type.id == "Exception"
    )

    assert blind == 1


@pytest.mark.parametrize("package", _PACKAGES)
def test_each_package_is_measured(package: str) -> None:
    assert package in count_blind_excepts()
