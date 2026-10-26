"""Contextual Thompson bandit used for ML operator selection."""

from __future__ import annotations

import numpy as np
import pytest

from multi_period_delivery.alns.bandit_selector import (
    ContextualThompsonBandit,
    reward_from_outcome,
)


def test_reward_ladder_orders_the_alns_outcomes() -> None:
    """The bandit must optimise what the search values, in the same order."""
    best = reward_from_outcome("best")
    improved = reward_from_outcome("improved")
    accepted = reward_from_outcome("accepted")
    rejected = reward_from_outcome("rejected")

    assert best > improved > accepted > rejected
    assert 0.0 <= rejected and best <= 1.0
    # A no-op cannot be worth as much as a real accepted move.
    assert reward_from_outcome("accepted", noop=True) < accepted


def test_cells_split_progress_and_pressure() -> None:
    assert ContextualThompsonBandit.cell(0.0, 0.0) == 0
    assert ContextualThompsonBandit.cell(0.99, 0.99) == 8
    # Out-of-range inputs are clamped, never indexed past the grid.
    assert ContextualThompsonBandit.cell(-5.0, 5.0) == 2  # (0 * 3) + 2


def test_bandit_is_exactly_neutral_until_it_has_trials() -> None:
    """An information-free bandit must not tilt the wheel at all."""
    bandit = ContextualThompsonBandit(
        5, np.random.default_rng(1), min_samples=10
    )

    weights = bandit.weights(0)

    assert np.allclose(weights, 1.0)
    assert bandit.is_ready() is False


def test_bandit_favours_the_arm_that_keeps_winning() -> None:
    rng = np.random.default_rng(7)
    bandit = ContextualThompsonBandit(4, rng, prior_strength=2.0, min_samples=20)
    for _ in range(120):
        bandit.update(0, 0, 1.0)
        bandit.update(0, 1, 0.0)
        bandit.update(0, 2, 0.0)
        bandit.update(0, 3, 0.0)

    stats = bandit.statistics()
    assert stats["ready"] is True
    assert stats["global_rate"][0] > stats["global_rate"][1]

    # Over many draws the winning arm must be sampled highest on average.
    totals = np.zeros(4)
    for _ in range(200):
        totals += bandit.weights(0)
    assert totals.argmax() == 0


def test_sparse_cell_shrinks_to_the_arm_global_rate() -> None:
    """A cell with two trials must not report a wild estimate.

    This is the failure mode the supervised model kept hitting: a retrain on a
    handful of rows produced AUC 0.76 then 0.32 then 0.21 in the same run.
    """
    rng = np.random.default_rng(11)
    bandit = ContextualThompsonBandit(3, rng, prior_strength=8.0, min_samples=10)
    # Arm 1 wins consistently everywhere.
    for _ in range(200):
        bandit.update(0, 1, 1.0)
        bandit.update(0, 0, 0.0)
        bandit.update(0, 2, 0.0)
    # A brand-new cell with a single unlucky observation for arm 1.
    bandit.update(5, 1, 0.0)

    alpha, beta = bandit._params(5, 1)
    sparse_rate = alpha / (alpha + beta)

    assert sparse_rate > 0.7, "one miss must not overturn 200 wins"


def test_updates_ignore_bad_inputs_without_raising() -> None:
    bandit = ContextualThompsonBandit(2, np.random.default_rng(3))

    bandit.update(0, 99, 1.0)  # arm out of range
    bandit.update(0, 0, float("nan"))
    bandit.update(0, 0, 5.0)  # clamped to 1.0

    assert bandit.statistics()["total_trials"] == 1
    assert bandit.statistics()["global_wins"][0] == 1.0


def test_sampling_is_reproducible_for_a_seed() -> None:
    """Determinism is what makes an ablation repeatable."""
    def draws(seed: int) -> list[float]:
        bandit = ContextualThompsonBandit(4, np.random.default_rng(seed))
        for _ in range(40):
            bandit.update(0, 0, 1.0)
            bandit.update(0, 1, 0.0)
        return list(bandit.weights(0))

    assert draws(21) == draws(21)


def test_zero_arms_is_rejected() -> None:
    with pytest.raises(ValueError, match="num_arms"):
        ContextualThompsonBandit(0, np.random.default_rng(0))
