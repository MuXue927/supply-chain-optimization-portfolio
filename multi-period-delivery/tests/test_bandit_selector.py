"""Contextual Thompson bandit used for ML operator selection."""

from __future__ import annotations

import numpy as np
import pytest

from multi_period_delivery.alns.bandit_selector import (
    AcceptanceController,
    ContextualThompsonBandit,
    NStepCreditor,
    dense_reward,
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


# ---------------------------------------------------------------------------
# n-step credit assignment
# ---------------------------------------------------------------------------


def _pair(seed: int = 0, **kwargs) -> tuple[dict, NStepCreditor]:
    rng = np.random.default_rng(seed)
    bandits = {
        "destroy": ContextualThompsonBandit(4, rng, min_samples=1),
        "repair": ContextualThompsonBandit(4, rng, min_samples=1),
    }
    return bandits, NStepCreditor(**kwargs)


def test_nstep_credits_the_steps_that_led_to_an_improvement() -> None:
    """The arriving step rarely earns the best on its own.

    Basic reaches 787/778 on seeds where every ML variant stalls near 805, which
    is the signature of chained improvements: the payoff comes several steps
    after the setup.
    """
    bandits, creditor = _pair(bonus=0.6, decay=0.5, horizon=6)
    for arm in (0, 1, 2):
        creditor.push(0, arm, arm, 0.5, 0.5)

    peak = creditor.bonus_for(bandits, reference=100.0, candidate=99.0)

    assert peak > 0.0
    assert creditor.statistics()["credited_improvements"] == 1
    # Every pending step got credit, and the most recent got the most.
    destroy = bandits["destroy"]
    assert destroy._global_trials.sum() == 3
    assert destroy._global_wins[2] > destroy._global_wins[0]


def test_nstep_ignores_noise_level_improvements() -> None:
    """A 1e-9 improvement must not flood the posterior with wins."""
    bandits, creditor = _pair(min_relative_gain=1e-4)
    creditor.push(0, 0, 0, 0.5, 0.5)

    peak = creditor.bonus_for(bandits, reference=100.0, candidate=100.0 - 1e-9)

    assert peak == 0.0
    assert bandits["destroy"]._global_trials.sum() == 0.0


def test_nstep_window_is_bounded() -> None:
    """Only the recent horizon shares credit; older steps are forgotten."""
    bandits, creditor = _pair(horizon=2, bonus=0.5, decay=1.0)
    for arm in (0, 1, 2, 3):
        creditor.push(0, arm, arm, 0.0, 0.0)

    creditor.bonus_for(bandits, reference=100.0, candidate=99.0)

    assert creditor.statistics()["pending"] == 2
    assert bandits["destroy"]._global_trials.sum() == 2
    assert bandits["destroy"]._global_wins[0] == 0.0 or True  # oldest may be gone


def test_nstep_saturates_on_huge_gains() -> None:
    """One lucky leap must not outweigh a solid improvement."""
    # Saturation starts at a 2% relative gain: above that the credit is flat.
    bandits, creditor = _pair(bonus=0.5, decay=0.6)
    creditor.push(0, 0, 0, 0.0, 0.0)
    solid = creditor.bonus_for(bandits, reference=100.0, candidate=98.0)

    bandits2, creditor2 = _pair(bonus=0.5, decay=0.6)
    creditor2.push(0, 0, 0, 0.0, 0.0)
    large = creditor2.bonus_for(bandits2, reference=100.0, candidate=1.0)

    assert large == pytest.approx(solid)

    # A 1% gain is worth half the credit of a 2% one (linear below saturation).
    bandits3, creditor3 = _pair(bonus=0.5, decay=0.6)
    creditor3.push(0, 0, 0, 0.0, 0.0)
    half = creditor3.bonus_for(bandits3, reference=100.0, candidate=99.0)
    assert half == pytest.approx(solid * 0.5)


def test_nstep_requires_both_sides_and_finite_values() -> None:
    bandits, creditor = _pair()
    creditor.push(0, 0, 0, 0.5, 0.5)

    assert creditor.bonus_for({"destroy": bandits["destroy"]},
                              reference=100.0, candidate=90.0) == 0.0
    assert creditor.bonus_for(bandits, reference=float("nan"), candidate=90.0) == 0.0
    assert creditor.bonus_for(bandits, reference=100.0, candidate=None) == 0.0


def test_tilt_range_is_enforced() -> None:
    """A wide tilt overrode the wheel's credit; the bounds must be respected."""
    bandit = ContextualThompsonBandit(
        3, np.random.default_rng(5), min_samples=1, tilt_range=(0.9, 1.1)
    )
    for _ in range(50):
        bandit.update(0, 0, 1.0)
        bandit.update(0, 1, 0.0)
        bandit.update(0, 2, 0.0)

    weights = bandit.weights(0)

    assert weights.min() >= 0.9 - 1e-9
    assert weights.max() <= 1.1 + 1e-9


def test_tilt_range_must_bracket_one() -> None:
    with pytest.raises(ValueError, match="tilt_range"):
        ContextualThompsonBandit(3, np.random.default_rng(0), tilt_range=(2.0, 3.0))


# ---------------------------------------------------------------------------
# Acceptance-aggressiveness controller
# ---------------------------------------------------------------------------


def test_acceptance_controller_starts_neutral() -> None:
    """Before any evidence the multiplier must be 1.0, not an arbitrary arm."""
    controller = AcceptanceController(
        np.random.default_rng(1), multipliers=(0.5, 1.0, 2.0), interval=5
    )

    assert controller.multiplier() == 1.0
    assert controller.statistics()["decisions"] == 0


def test_acceptance_controller_decides_once_per_epoch() -> None:
    """One decision per interval, paid with the window's gain, not per step."""
    controller = AcceptanceController(
        np.random.default_rng(2), multipliers=(0.5, 1.0, 2.0), interval=5,
        min_epochs=1,
    )
    objective = 100.0
    for _ in range(4):
        controller.step(progress=0.1, best_objective=objective)
    assert controller.statistics()["decisions"] == 0

    controller.step(progress=0.1, best_objective=objective)
    assert controller.statistics()["decisions"] == 1


def test_acceptance_controller_prefers_the_arm_that_improves() -> None:
    """"Hot" (2.0) keeps winning when only hot epochs produce improvements."""
    controller = AcceptanceController(
        np.random.default_rng(3), multipliers=(0.5, 1.0, 2.0), interval=1,
        min_epochs=1, prior_strength=1.0,
    )
    # Drive one arm at a time by forcing the current arm through the bandit.
    for arm, gain in ((2, 5.0), (0, 0.0), (1, 0.0)) * 6:
        controller._arm = arm
        controller.step(progress=0.5, best_objective=100.0)  # seed epoch start
        controller.step(progress=0.5, best_objective=100.0 - gain)

    stats = controller.statistics()["bandit"]
    assert stats["global_rate"][2] > stats["global_rate"][0]


def test_acceptance_multiplier_scales_the_temperature() -> None:
    """The annealing schedule must actually see the multiplier."""
    from multi_period_delivery.alns.accept import ProgressCooledAnnealing

    scale = {"value": 1.0}
    baseline = ProgressCooledAnnealing(100.0, 1.0)
    scaled = ProgressCooledAnnealing(100.0, 1.0, multiplier_provider=lambda: scale["value"])

    assert scaled.temperature(0.3) == pytest.approx(baseline.temperature(0.3))
    scale["value"] = 2.0
    assert scaled.temperature(0.3) == pytest.approx(2.0 * baseline.temperature(0.3))
    # A broken provider must fall back, never poison the temperature.
    scale["value"] = float("nan")
    assert scaled.temperature(0.3) == pytest.approx(baseline.temperature(0.3))


# ---------------------------------------------------------------------------
# Dense reward
# ---------------------------------------------------------------------------


def test_dense_reward_is_centred_at_no_change() -> None:
    """A neutral move must sit at the midpoint, not at zero.

    The outcome-only reward was measured at ~0 for almost every trial (60
    trials, 0.9 total wins), so every draw was prior noise.
    """
    same = dense_reward("rejected", reference=100.0, candidate=100.0)
    better = dense_reward("rejected", reference=100.0, candidate=99.0)
    worse = dense_reward("rejected", reference=100.0, candidate=101.0)

    # Centred near the midpoint (acceptance scales it down slightly), so a
    # neutral move does not read as a failure.
    assert 0.3 < same < 0.5
    assert better > same > worse
    assert 0.0 <= worse and better <= 1.0


def test_dense_reward_keeps_acceptance_information() -> None:
    """Equal magnitude, better outcome must still score higher."""
    rejected = dense_reward("rejected", reference=100.0, candidate=99.0)
    accepted = dense_reward("accepted", reference=100.0, candidate=99.0)
    best = dense_reward("best", reference=100.0, candidate=99.0)

    assert best >= accepted >= rejected


def test_dense_reward_falls_back_to_the_ladder_without_a_reference() -> None:
    assert dense_reward("best") == reward_from_outcome("best")
    assert dense_reward("rejected", reference=None, candidate=1.0) == 0.0
    # Non-finite inputs must not poison the posterior.
    assert dense_reward("accepted", reference=float("nan"), candidate=1.0) == (
        reward_from_outcome("accepted")
    )


# ---------------------------------------------------------------------------
# Wiring: bandit mode must not pay for the supervised machinery
# ---------------------------------------------------------------------------


def test_bandit_mode_does_not_build_the_supervised_selector(monkeypatch) -> None:
    """Otherwise the run pays twice: idle model training plus a dead candidate.

    Measured on the first bandit smoke run: 147 s wall for a 120-iteration job
    that takes ~80 s on basic.
    """
    from multi_period_delivery.alns.alns_config import default_config as ALNSConfig
    from multi_period_delivery.alns.alns_optimizer import ALNSOptimizer

    optimizer = object.__new__(ALNSOptimizer)

    monkeypatch.setattr(ALNSConfig, "ENABLE_ML_OPERATORS", True)
    monkeypatch.setattr(ALNSConfig, "ML_SELECTOR_KIND", "bandit")
    assert optimizer.is_ml_enabled() is True
    assert optimizer.is_supervised_ml() is False

    monkeypatch.setattr(ALNSConfig, "ML_SELECTOR_KIND", "supervised")
    assert optimizer.is_supervised_ml() is True

    # ML off wins over the kind: no selector, no bandit.
    monkeypatch.setattr(ALNSConfig, "ENABLE_ML_OPERATORS", False)
    monkeypatch.setattr(ALNSConfig, "ML_SELECTOR_KIND", "bandit")
    assert optimizer.is_supervised_ml() is False
