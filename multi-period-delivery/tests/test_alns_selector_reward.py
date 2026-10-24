"""Neutrality of the operator-selection reward.

The segmented roulette wheel credits an operator with the *sum* of its scores
over a segment. Since a segment is a fixed number of iterations, that sum grows
with how often the operator was drawn, not only with how well it did - and the
draw probability is the weight being updated. The loop is self-reinforcing.

The test below is the property such a rule has to satisfy: if every operator's
scores come from the *same* distribution, no operator should end up with a
systematically larger share. Neither rule may be judged by the solution quality
of a real run, because a real run has different operators with different
quality; only identical distributions isolate the rule itself.

Both halves are asserted on purpose. The ``"sum"`` half documents the defect and
proves the test can fail; without it a bug that made every operator equally
likely regardless of score would also pass the ``"shrunk"`` half.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from alns.Outcome import Outcome

from multi_period_delivery.alns.alns_optimizer import NoOpAwareSegmentedRouletteWheel

SCORES = [8.0, 3.0, 1.0, 0.2]
OUTCOME_P = [0.04, 0.10, 0.26, 0.60]
OUTCOMES = [Outcome.BEST, Outcome.BETTER, Outcome.ACCEPT, Outcome.REJECT]


def _max_share(normalisation: str, iters: int, n: int, seed: int, seg: int) -> float:
    """Run the selector with identical score distributions for every operator."""
    wheel = NoOpAwareSegmentedRouletteWheel(
        scores=SCORES,
        decay=0.65,
        seg_length=seg,
        num_destroy=n,
        num_repair=n,
        normalisation=normalisation,
    )
    rng = np.random.default_rng(seed)
    cand = SimpleNamespace()
    counts = np.zeros(n)

    for _ in range(iters):
        d_idx, r_idx = wheel(rng, cand, cand)
        counts[d_idx] += 1
        outcome = OUTCOMES[int(rng.choice(4, p=OUTCOME_P))]
        wheel.update(cand, d_idx, r_idx, outcome)

    return float(counts.max() / counts.sum())


@pytest.mark.parametrize("seed", [1000, 1001, 1002, 1003, 1004])
def test_shrunk_credit_keeps_selection_neutral(seed: int) -> None:
    """Equal-quality operators must not drift apart under ``"shrunk"``."""
    n = 11
    share = _max_share("shrunk", iters=2000, n=n, seed=seed, seg=80)
    # 1/11 = 9.09%; measured 10.7% mean over 24 seeds, max 12.2%.
    assert share < 1.5 / n, f"max share {share:.3f} exceeds 1.5x uniform"


@pytest.mark.parametrize("seed", [1000, 1001, 1002])
def test_summed_credit_is_not_neutral(seed: int) -> None:
    """The library rule concentrates even when every operator is equal.

    This is the reason ``ROULETTE_REWARD_NORMALISATION`` exists. Measured: 26.3%
    mean, worst 36.9% over 24 seeds, never below 2x uniform.
    """
    n = 11
    share = _max_share("sum", iters=2000, n=n, seed=seed, seg=80)
    assert share > 2.0 / n, (
        f"summed credit gave {share:.3f}, expected the documented >2x drift; "
        "if this now passes, the reward rule changed and M28 needs revisiting"
    )


def test_shrunk_survives_short_segments_where_the_mean_drifts() -> None:
    """Short segments give each operator ~2 samples, where a plain mean drifts."""
    n = 11
    seg, iters, seed = 20, 2000, 1000
    mean_share = _max_share("mean", iters=iters, n=n, seed=seed, seg=seg)
    shrunk_share = _max_share("shrunk", iters=iters, n=n, seed=seed, seg=seg)

    assert mean_share > 1.5 / n, (
        f"plain mean gave {mean_share:.3f}; expected it to drift at seg_length=20"
    )
    assert shrunk_share < 1.5 / n, (
        f"shrunk gave {shrunk_share:.3f}, expected it to stay near uniform"
    )


def test_unknown_normalisation_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown roulette reward normalisation"):
        NoOpAwareSegmentedRouletteWheel(
            scores=SCORES,
            decay=0.65,
            seg_length=80,
            num_destroy=3,
            num_repair=2,
            normalisation="median",
        )
