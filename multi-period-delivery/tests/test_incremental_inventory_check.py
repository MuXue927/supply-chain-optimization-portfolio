"""The incremental inventory pre-check must agree with ``validate()`` exactly.

``cross_period_consolidation_repair`` rejects ~89% of the candidates it evaluates,
and ~98% of those rejections are ``negative_inventory``. Each rejection used to
cost a full state clone plus a ``validate()`` (which recomputes the whole inventory
table).

``_cross_period_merge_keeps_inventory_non_negative`` decides the same question from
the current state alone. It is not a heuristic: moving a load from the source day to
the target day changes ``shipped`` for the affected (plant, sku) pairs on exactly
two days, production is untouched, and the recursion
``s[d] = s[d-1] + prod[d] - shipped[d]`` turns that into a running offset applied
from ``min(source.day, target.day)`` to the horizon. So ``s_old - run < 0`` is
*exactly* the set of new negative entries.

The probe ``_incremental_feasibility_check.py`` checked this against
``candidate.validate()`` for all 5,570 reachable merges on a real medium state with
0 disagreements. These tests pin the same property on small, hand-built cases so a
future change cannot silently break it.
"""

from __future__ import annotations

from types import SimpleNamespace

from multi_period_delivery.alns.repair_operators import (
    _cross_period_merge_keeps_inventory_non_negative,
)


def _vehicle(veh_id, fact_id, day, cargo):
    return SimpleNamespace(id=veh_id, fact_id=fact_id, day=day, cargo=dict(cargo))


def _state(s_ikt):
    return SimpleNamespace(s_ikt=dict(s_ikt))


H = 4


# ---------------------------------------------------------------------------
# Negative cases: the merge must be rejected
# ---------------------------------------------------------------------------


def test_reports_when_the_target_day_goes_negative() -> None:
    """Moving the load to an earlier day removes stock that existed there."""
    source = _vehicle(1, "P", 2, {("S", 2): 2})
    target = _vehicle(2, "P", 1, {})
    state = _state({("P", "S", 1): 0, ("P", "S", 2): 5})

    assert _cross_period_merge_keeps_inventory_non_negative(state, source, target, H) is False


def test_reports_when_the_offset_pushes_a_later_day_negative() -> None:
    """The offset is carried forward, so a later day can be the one that fails."""
    source = _vehicle(1, "P", 2, {("S", 2): 5})
    target = _vehicle(2, "P", 1, {})
    # Day 1 has enough, but the offset carries into day 3.
    state = _state({("P", "S", 1): 3, ("P", "S", 2): 0, ("P", "S", 3): 2})

    assert _cross_period_merge_keeps_inventory_non_negative(state, source, target, H) is False


def test_reports_negative_for_a_multi_sku_load() -> None:
    source = _vehicle(1, "P", 2, {("S1", 2): 2, ("S2", 2): 2})
    target = _vehicle(2, "P", 1, {})
    state = _state({("P", "S1", 1): 0, ("P", "S2", 1): 10})

    assert _cross_period_merge_keeps_inventory_non_negative(state, source, target, H) is False


# ---------------------------------------------------------------------------
# Positive cases: the merge must be allowed
# ---------------------------------------------------------------------------


def test_allows_when_every_affected_day_stays_non_negative() -> None:
    source = _vehicle(1, "P", 2, {("S", 2): 2})
    target = _vehicle(2, "P", 1, {})
    state = _state({("P", "S", 1): 5, ("P", "S", 2): 5})

    assert _cross_period_merge_keeps_inventory_non_negative(state, source, target, H) is True


def test_allows_an_empty_source() -> None:
    source = _vehicle(1, "P", 2, {})
    target = _vehicle(2, "P", 1, {})
    state = _state({("P", "S", 1): 0})

    assert _cross_period_merge_keeps_inventory_non_negative(state, source, target, H) is True


def test_allows_when_the_two_deltas_cancel_before_any_shortage() -> None:
    """Moving a load later only lowers days from the target onwards."""
    source = _vehicle(1, "P", 4, {("S", 4): 2})
    target = _vehicle(2, "P", 2, {})
    # Days 2 and 3 carry the offset; day 4 gets the load back.
    state = _state({("P", "S", 2): 2, ("P", "S", 3): 2, ("P", "S", 4): 2})

    assert _cross_period_merge_keeps_inventory_non_negative(state, source, target, H) is True


def test_missing_entries_are_treated_as_zero() -> None:
    """``s_ikt`` does not hold every (plant, sku, day) key."""
    source = _vehicle(1, "P", 2, {("S", 2): 1})
    target = _vehicle(2, "P", 1, {})
    state = _state({})

    # Day 1 defaults to 0, and the offset is -1 there, so this must be rejected.
    assert _cross_period_merge_keeps_inventory_non_negative(state, source, target, H) is False


# ---------------------------------------------------------------------------
# It must only look at the affected pairs
# ---------------------------------------------------------------------------


def test_an_unrelated_negative_entry_does_not_cause_a_rejection() -> None:
    """Other pairs are untouched by the merge and are judged elsewhere."""
    source = _vehicle(1, "P", 2, {("S1", 2): 2})
    target = _vehicle(2, "P", 1, {})
    state = _state({("P", "S1", 1): 5, ("P", "S2", 3): -7})

    assert _cross_period_merge_keeps_inventory_non_negative(state, source, target, H) is True


def test_another_plants_inventory_is_ignored() -> None:
    source = _vehicle(1, "P1", 2, {("S", 2): 2})
    target = _vehicle(2, "P1", 1, {})
    state = _state({("P1", "S", 1): 5, ("P2", "S", 1): -9})

    assert _cross_period_merge_keeps_inventory_non_negative(state, source, target, H) is True
