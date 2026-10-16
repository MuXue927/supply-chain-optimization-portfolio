"""``validation_memo_ctx`` must reuse a validation only when it is provably safe.

``compute_improvement`` validates one state twice per simulated candidate - once
for the search objective (``state.objective()``) and once for the mixed reward
(``compute_mixed_operator_reward`` -> ``state.calculate_cost()``). On the medium
instance each validation traverses the whole solution, so that double traversal
dominated the repair operator.

``validation_memo_ctx`` lets those two calls share one traversal. It must never
change what ``validate()`` reports, so these tests pin:

* reuse happens only inside the scope, and only while the freshness stamp holds;
* a cargo or inventory mutation invalidates the memo;
* a memo hit equals a fresh full check and returns independent containers;
* outside the scope ``validate()`` always recomputes (the contract the freshness
  and violation-key tests rely on).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from multi_period_delivery.alns.alns_optimizer import ALNSOptimizer
from multi_period_delivery.alns.alnsopt import SolutionState

BASE = Path(__file__).resolve().parents[1]


def _state():
    optimizer = ALNSOptimizer(
        input_file_loc=str(BASE / "datasets" / "multiple-periods" / "small"),
        output_file_loc=str(Path(tempfile.gettempdir()) / "_validation_memo_probe"),
    )
    optimizer.load_data("dataset_1")
    return optimizer.create_initial_solution()


def _count_uncached(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Count real (un-memoized) traversals while keeping the real behaviour."""
    original = SolutionState._validate_uncached
    calls = {"n": 0}

    def counting(self):
        calls["n"] += 1
        return original(self)

    monkeypatch.setattr(SolutionState, "_validate_uncached", counting)
    return calls


# ---------------------------------------------------------------------------
# Reuse inside the scope, recompute outside it
# ---------------------------------------------------------------------------


def test_scope_serves_the_second_validation_from_the_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state()
    calls = _count_uncached(monkeypatch)

    with state.validation_memo_ctx():
        first = state.validate()
        second = state.validate()

    assert calls["n"] == 1, "the second validation must come from the memo"
    assert first == second


def test_outside_the_scope_validate_always_recomputes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public contract: without the scope, no memo is consulted."""
    state = _state()
    calls = _count_uncached(monkeypatch)

    state.validate()
    state.validate()

    assert calls["n"] == 2


def test_scope_is_reset_on_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _state()
    calls = _count_uncached(monkeypatch)

    with state.validation_memo_ctx():
        state.validate()

    assert state._validation_memo_enabled is False
    assert state._validation_memo is None

    state.validate()
    assert calls["n"] == 2


def test_scope_is_reset_even_when_the_body_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state()

    with pytest.raises(RuntimeError):
        with state.validation_memo_ctx():
            state.validate()
            raise RuntimeError("boom")

    assert state._validation_memo_enabled is False
    assert state._validation_memo is None


# ---------------------------------------------------------------------------
# Invalidation on a real mutation
# ---------------------------------------------------------------------------


def test_cargo_change_invalidates_the_memo(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _state()
    state.compute_inventory()
    calls = _count_uncached(monkeypatch)

    with state.validation_memo_ctx():
        state.validate()
        vehicle = next(v for v in state.vehicles if v.cargo)
        key = next(iter(vehicle.cargo))
        vehicle.cargo[key] = vehicle.cargo[key] + 1000
        state.validate()

    assert calls["n"] == 2, "a changed load must force a fresh traversal"


def test_inventory_write_invalidates_the_memo(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _state()
    state.compute_inventory()
    calls = _count_uncached(monkeypatch)

    with state.validation_memo_ctx():
        state.validate()
        key = next(iter(state.s_ikt))
        state.s_ikt[key] = state.s_ikt[key] - 10_000
        feasible, violations = state.validate()

    assert calls["n"] == 2
    assert feasible is False
    assert violations["negative_inventory"]


# ---------------------------------------------------------------------------
# A hit must equal a fresh check, and must not alias its containers
# ---------------------------------------------------------------------------


def test_memoized_result_equals_a_fresh_check() -> None:
    state = _state()

    with state.validation_memo_ctx():
        first = state.validate()
        second = state.validate()

    fresh = state._validate_uncached()

    assert first == second == fresh


def test_hit_returns_containers_that_do_not_alias_the_memo() -> None:
    state = _state()

    with state.validation_memo_ctx():
        _, violations_first = state.validate()
        violations_first["unmet_demand"].append("sentinel")
        _, violations_second = state.validate()

    assert "sentinel" not in violations_second["unmet_demand"]
