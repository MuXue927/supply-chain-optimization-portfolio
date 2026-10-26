"""The roulette's credit bookkeeping must survive the bandit tilt.

When the tilt hook first shipped it called ``RouletteWheel.__call__`` directly
instead of ``SegmentedRouletteWheel.__call__``. That skipped the per-segment
credit update entirely, so the wheel kept its initial uniform weights for the
whole run: "ML" became "a frozen uniform wheel plus a nudge" rather than "the
normal wheel plus a nudge". Every ML run then converged to the same ~805
attractor while basic reached 787 and 778 on two seeds.
"""

from __future__ import annotations

import numpy as np

from multi_period_delivery.alns.alns_optimizer import (
    NoOpAwareSegmentedRouletteWheel,
)


def _wheel(**kwargs) -> NoOpAwareSegmentedRouletteWheel:
    return NoOpAwareSegmentedRouletteWheel(
        scores=[5, 3, 2, 1],
        decay=0.5,
        seg_length=2,
        num_destroy=3,
        num_repair=3,
        **kwargs,
    )


def test_no_hook_path_still_updates_weights_per_segment() -> None:
    """Baseline: the parent's segment update is what makes the wheel adapt."""
    wheel = _wheel()
    rng = np.random.default_rng(0)
    before = wheel._d_weights.copy()

    for _ in range(4):
        wheel(rng, None, None)
        # Reward operator 0 heavily so the segment credit is unmistakable.
        wheel.update(None, 0, 0, 3)

    assert wheel._iter == 4
    assert not np.allclose(wheel._d_weights, before)


def test_tilt_hook_does_not_skip_the_segment_update() -> None:
    """The regression: a tilt must not freeze the wheel's accumulated credit."""
    neutral = lambda _best, _curr: (None, None)  # noqa: E731 - neutral tilt
    wheel = _wheel(weight_hook=neutral)
    rng = np.random.default_rng(0)
    before = wheel._d_weights.copy()

    for _ in range(4):
        wheel(rng, None, None)
        wheel.update(None, 0, 0, 3)

    assert wheel._iter == 4, "the iteration counter drives the segment boundary"
    assert not np.allclose(wheel._d_weights, before), "credit was never applied"


def test_tilt_is_applied_for_one_draw_only() -> None:
    """The bandit's opinion must not compound into the accumulated weights."""
    wheel = _wheel(weight_hook=lambda _b, _c: (np.full(3, 3.0), np.full(3, 3.0)))
    rng = np.random.default_rng(1)
    stored = wheel._d_weights.copy()

    wheel(rng, None, None)

    assert np.allclose(wheel._d_weights, stored), "tilt leaked into the credit"


def test_wide_tilt_really_changes_the_draw() -> None:
    """A strong tilt must be observable, or "it had no effect" proves nothing."""
    def favour_zero(_best, _curr):
        return np.array([1e6, 1.0, 1.0]), np.ones(3)

    tilted = _wheel(weight_hook=favour_zero)
    rng = np.random.default_rng(2)
    picks = [tilted(rng, None, None)[0] for _ in range(50)]

    assert picks.count(0) > 40
