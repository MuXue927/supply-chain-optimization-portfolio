"""Tests for no-op-aware operator reward handling."""

from __future__ import annotations

from types import SimpleNamespace

from alns.Outcome import Outcome

from multi_period_delivery.alns.alns_optimizer import (
    NoOpAwareSegmentedRouletteWheel,
    _NoOpFlagPreservingRepair,
)


def _wheel() -> NoOpAwareSegmentedRouletteWheel:
    return NoOpAwareSegmentedRouletteWheel(
        scores=[8.0, 3.0, 1.0, 0.2],
        decay=0.9,
        seg_length=100,
        num_destroy=2,
        num_repair=2,
    )


def test_noop_destroy_receives_reject_score_but_real_repair_keeps_accept() -> None:
    wheel = _wheel()
    cand = SimpleNamespace(_destroy_noop=True, _repair_noop=False)

    wheel.update(cand, 0, 1, Outcome.ACCEPT)

    assert wheel._d_seg_weights[0] == 0.2
    assert wheel._r_seg_weights[1] == 1.0


def test_noop_repair_receives_reject_score() -> None:
    wheel = _wheel()
    cand = SimpleNamespace(_destroy_noop=False, _repair_noop=True)

    wheel.update(cand, 1, 0, Outcome.ACCEPT)

    assert wheel._d_seg_weights[1] == 1.0
    assert wheel._r_seg_weights[0] == 0.2


def test_repair_wrapper_preserves_destroy_noop_and_detects_noop_repair() -> None:
    state = SimpleNamespace(
        vehicles=[],
        s_ikt={},
        _destroy_noop=True,
    )
    wrapper = _NoOpFlagPreservingRepair(
        lambda current, rng: SimpleNamespace(
            vehicles=list(current.vehicles),
            s_ikt=dict(current.s_ikt),
        )
    )

    result = wrapper(state, None)

    assert result._destroy_noop is True
    assert result._repair_noop is True


def test_non_noop_destroy_keeps_normal_reward_path() -> None:
    wheel = _wheel()
    cand = SimpleNamespace(_destroy_noop=False, _repair_noop=False)

    wheel.update(cand, 0, 0, Outcome.ACCEPT)

    assert wheel._d_seg_weights[0] == 1.0
    assert wheel._r_seg_weights[0] == 1.0
