"""Contextual Thompson-sampling bandit for operator selection.

Why this replaces the supervised predictor
------------------------------------------
The shipped design trains Ridge / logistic models on ~15-28 features to predict
how much an operator will improve the objective, then ranks operators by that
prediction. Measured on five seeds at a fixed 120 iterations, that is worse than
no model at all: ``ml - basic = +9.18`` (t = 1.92, better on 1 of 5 seeds). The
three causes are structural, not a matter of estimator choice:

1. **Wrong target.** Selection needs a *ranking*; regression on the improvement
   magnitude is dominated by a few huge repairs. The same buffers score R2 of
   -0.3 to -1.8e7 on regression but AUC ~0.66 on "did it improve at all".
2. **Too little data per dimension.** 40-110 rows against 15-28 features means
   the coefficients are re-estimated from noise at every retrain; the gate's
   AUC was observed to move 0.76 -> 0.32 -> 0.21 inside one run.
3. **Myopic reward.** A destroy operator's value depends on the repair that
   follows it, so "cost delta right after the destroy" is not its credit.

Thompson sampling addresses all three at once: it is a ranking method by
construction, it needs only counts per (context, arm) cell, and its reward can
be the ALNS outcome score rather than a raw cost delta.

Design
------
* **Arms** are operators (one bandit per side).
* **Context** is a coarse cell: progress tercile x unmet-demand tercile. Nine
  cells, which keeps a handful of observations per cell meaningful.
* **Posterior** per (cell, arm) is Beta, kept as counts.
* **Shrinkage** pulls each cell towards that arm's global posterior:
  ``alpha = prior * global_rate + wins``. A cell that has seen two trials
  therefore reports something close to the arm's overall behaviour instead of a
  wild estimate - the failure mode the supervised model kept hitting.
* **Sampling** draws one value per arm and returns them as *relative weights*.
  The caller multiplies them into the roulette weights rather than replacing
  the selection, so a bandit with no information is exactly neutral and the
  search degrades to the baseline instead of to something worse.

The reward is a binary "was this outcome worth repeating", taken from the ALNS
outcome class (new global best / improved / accepted / rejected) plus a small
penalty for a no-op. It is deliberately not the raw improvement.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np

#: Reward for an outcome class. Mirrors the ALNS score ladder (new global best
#: is worth the most, rejected is worth nothing) so the bandit optimises the
#: same thing the search does, instead of a proxy that disagrees with it.
REWARD_BY_OUTCOME: dict[str, float] = {
    "best": 1.0,
    "improved": 0.75,
    "accepted": 0.4,
    "rejected": 0.0,
}

#: A no-op (the operator changed nothing) is treated as a rejected trial with a
#: slight extra penalty: it consumed a slot without even a chance to help.
NOOP_PENALTY = 0.1


def reward_from_outcome(outcome: Any, *, noop: bool = False) -> float:
    """Map an ALNS outcome onto ``[0, 1]``."""
    if isinstance(outcome, str):
        name = outcome.strip().lower()
    else:
        name = str(getattr(outcome, "name", outcome)).strip().lower()
    value = REWARD_BY_OUTCOME.get(name, 0.0)
    if noop:
        value = max(0.0, value - NOOP_PENALTY)
    return value


def dense_reward(
    outcome: Any,
    *,
    reference: float | None = None,
    candidate: float | None = None,
    noop: bool = False,
    magnitude_weight: float = 0.3,
) -> float:
    """Blend the outcome ladder with how much the objective actually moved.

    The outcome alone is too sparse to learn from here. Measured on a 60-iteration
    bandit run: 60 trials, and the winning arms accumulated 0.9 wins in total -
    i.e. almost every outcome was ``rejected`` (0.0), so every draw was pure
    prior noise. Worse, a rejected move that still reduced the objective teaches
    nothing, even though it is exactly the kind of step a repair operator should
    be credited for.

    The magnitude term is a soft, scale-free function of the relative change:

        magnitude = 0.5 + 0.5 * tanh(rel_gain / scale)

    centred on 0.5 at "no change", so a neutral move is not rewarded or punished
    and the bandit sees a spread of values every iteration instead of a column of
    zeros. Acceptance is applied as a *scaling* of that term rather than as an
    additive offset:

        reward = magnitude * (1 - w + w * ladder)

    An additive blend was tried first and rejected: with ``rejected = 0`` the
    no-change point landed at 0.25 instead of the midpoint, so the bandit read
    "nothing happened" as a poor result and learned to avoid whatever it was
    doing when the search stalled. Scaling keeps the midpoint meaningful while
    still ranking "best" above "accepted" above "rejected" at equal magnitude.
    """
    base = reward_from_outcome(outcome, noop=noop)
    if reference is None or candidate is None:
        return base
    try:
        ref = float(reference)
        cand = float(candidate)
    except (TypeError, ValueError):
        return base
    if not (math.isfinite(ref) and math.isfinite(cand)):
        return base
    scale = max(abs(ref), 1.0)
    rel_gain = (ref - cand) / scale
    if not math.isfinite(rel_gain):
        return base
    # 1% relative change is already a strong signal on this objective.
    magnitude = 0.5 + 0.5 * math.tanh(rel_gain / 0.01)
    weight = min(0.9, max(0.0, float(magnitude_weight)))
    return min(1.0, max(0.0, magnitude * (1.0 - weight + weight * base)))


class ContextualThompsonBandit:
    """Beta-Bernoulli Thompson sampling over (context cell, operator) cells.

    Parameters
    ----------
    num_arms:
        Number of operators on this side.
    rng:
        Seeded generator. Drawing must be deterministic given the seed, which is
        what makes an ablation repeatable.
    prior_strength:
        Pseudo-observations behind the shrinkage target. Higher values keep
        cells closer to the arm's global rate for longer.
    min_samples:
        Below this many total trials the bandit reports neutral weights (all
        ones), so an information-free bandit cannot distort the search.
    """

    __slots__ = (
        "_num_arms",
        "_rng",
        "_prior",
        "_min_samples",
        "_global_wins",
        "_global_trials",
        "_cell_wins",
        "_cell_trials",
        "_total",
        "_tilt_low",
        "_tilt_high",
        "sampled",
    )

    def __init__(
        self,
        num_arms: int,
        rng: np.random.Generator,
        *,
        prior_strength: float = 4.0,
        min_samples: int = 0,
        tilt_range: tuple[float, float] = (0.25, 4.0),
    ) -> None:
        if int(num_arms) <= 0:
            raise ValueError(f"num_arms must be positive; got {num_arms!r}")
        self._num_arms = int(num_arms)
        self._rng = rng
        self._prior = max(0.0, float(prior_strength))
        self._min_samples = max(0, int(min_samples))
        low, high = float(tilt_range[0]), float(tilt_range[1])
        if not (0.0 < low <= 1.0 <= high):
            raise ValueError(
                "tilt_range must bracket 1.0 with positive bounds; got "
                f"{tilt_range!r}"
            )
        self._tilt_low = low
        self._tilt_high = high
        self._global_wins = np.zeros(self._num_arms, dtype=float)
        self._global_trials = np.zeros(self._num_arms, dtype=float)
        self._cell_wins: dict[tuple[int, int], np.ndarray] = {}
        self._cell_trials: dict[tuple[int, int], np.ndarray] = {}
        self._total = 0
        #: Last sampled weights, for diagnostics.
        self.sampled = np.ones(self._num_arms, dtype=float)

    # ------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------
    @staticmethod
    def cell(progress: float, pressure: float, *, buckets: int = 3) -> int:
        """Flatten (progress tercile, unmet-demand tercile) into one id.

        Both inputs are fractions in ``[0, 1]``; values outside are clamped.
        Nine cells keep each cell populated often enough to say anything.
        """
        bins = max(2, int(buckets))
        p = int(min(bins - 1, max(0, math.floor(min(1.0, max(0.0, progress)) * bins))))
        q = int(min(bins - 1, max(0, math.floor(min(1.0, max(0.0, pressure)) * bins))))
        return p * bins + q

    # ------------------------------------------------------------------
    # Posterior
    # ------------------------------------------------------------------
    def _rate(self, arm: int) -> float:
        """Shrunk global success rate of one arm, in ``(0, 1)``."""
        wins = float(self._global_wins[arm])
        trials = float(self._global_trials[arm])
        # Beta(1, 1) prior: an unseen arm is a coin flip, so it gets explored.
        return (wins + 1.0) / (trials + 2.0)

    def _params(self, cell: int, arm: int) -> tuple[float, float]:
        """Beta parameters for one cell, shrunk towards the arm's global rate."""
        rate = self._rate(arm)
        wins = float(self._cell_wins.get(cell, np.zeros(self._num_arms))[arm])
        trials = float(self._cell_trials.get(cell, np.zeros(self._num_arms))[arm])
        # ``prior`` pseudo-observations at the arm's own rate: shrinkage towards
        # what this arm usually does, not towards a uniform 0.5.
        alpha = self._prior * rate + wins
        beta = self._prior * (1.0 - rate) + (trials - wins)
        # Keep both strictly positive: Beta(0, x) is not sampled for.
        return max(alpha, 1e-6), max(beta, 1e-6)

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------
    def weights(self, cell: int) -> np.ndarray:
        """Sample one value per arm and return them as relative weights.

        Returned weights are normalised to mean 1, so the roulette's own credit
        stays the dominant signal and the bandit only tilts it.
        """
        if not self.is_ready():
            self.sampled = np.ones(self._num_arms, dtype=float)
            return self.sampled
        draws = np.empty(self._num_arms, dtype=float)
        for arm in range(self._num_arms):
            alpha, beta = self._params(cell, arm)
            draws[arm] = float(self._rng.beta(alpha, beta))
        mean = float(draws.mean())
        if not math.isfinite(mean) or mean <= 0.0:
            self.sampled = np.ones(self._num_arms, dtype=float)
            return self.sampled
        weights = draws / mean
        # A hard ceiling keeps one lucky draw from monopolising the wheel.
        self.sampled = np.clip(weights, self._tilt_low, self._tilt_high)
        return self.sampled

    def is_ready(self) -> bool:
        """Whether enough trials exist for the weights to mean anything."""
        return self._total >= self._min_samples

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def update(self, cell: int, arm: int, reward: float) -> None:
        """Record one Bernoulli trial for ``(cell, arm)``.

        The reward is treated as the probability of success rather than being
        thresholded, so the outcome ladder keeps its ordering.
        """
        if not 0 <= int(arm) < self._num_arms:
            return
        value = float(reward)
        if not math.isfinite(value):
            return
        value = min(1.0, max(0.0, value))
        arm = int(arm)
        self._global_wins[arm] += value
        self._global_trials[arm] += 1.0
        wins = self._cell_wins.setdefault(
            int(cell), np.zeros(self._num_arms, dtype=float)
        )
        trials = self._cell_trials.setdefault(
            int(cell), np.zeros(self._num_arms, dtype=float)
        )
        wins[arm] += value
        trials[arm] += 1.0
        self._total += 1

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def statistics(self) -> dict[str, Any]:
        """Counts and shrunk rates, so "it learned nothing" is visible."""
        return {
            "total_trials": int(self._total),
            "ready": bool(self.is_ready()),
            "min_samples": int(self._min_samples),
            "arms": int(self._num_arms),
            "cells": len(self._cell_wins),
            "global_trials": [float(v) for v in self._global_trials],
            "global_wins": [float(v) for v in self._global_wins],
            "global_rate": [round(self._rate(a), 4) for a in range(self._num_arms)],
            "sampled": [round(float(v), 4) for v in self.sampled],
        }


class NStepCreditor:
    """Credit the steps that *led* somewhere, not just the step that arrived.

    Every single-step reward tried so far is myopic: it asks "did this operator
    pay off immediately". That is the wrong question when the best sequences in
    this search are chains - a destroy that removes an awkward cluster only pays
    off two or three placements later. Measured evidence for the mismatch: at a
    fixed 120 iterations, basic reached 787.23 and 777.82 on two seeds while both
    supervised ML and the single-step bandit stalled around 805. The ML variants
    were not failing to *learn*; they were learning the wrong thing, and their
    preference then blocked the road that led deeper.

    This class keeps a short window of recent decisions. When the search reaches
    a new global best, every decision still in the window receives a decayed
    share of the credit, with the most recent step weighted highest.

    Parameters
    ----------
    horizon:
        How many recent decisions can share credit for one improvement.
    decay:
        Geometric factor per step back (``decay ** age``).
    bonus:
        Peak credit granted to the step that directly produced the best. Kept
        below 1.0 because these are extra trials layered on the dense reward.
    min_relative_gain:
        Improvement (relative to the incumbent) below which nothing is credited.
        Prevents "best improved by 1e-9" from flooding the posterior with wins.
    """

    __slots__ = ("_pending", "_horizon", "_decay", "_bonus", "_min_gain", "credited")

    def __init__(
        self,
        *,
        horizon: int = 6,
        decay: float = 0.6,
        bonus: float = 0.6,
        min_relative_gain: float = 1e-4,
    ) -> None:
        self._horizon = max(1, int(horizon))
        self._decay = min(1.0, max(0.0, float(decay)))
        self._bonus = min(1.0, max(0.0, float(bonus)))
        self._min_gain = max(0.0, float(min_relative_gain))
        self._pending: deque[tuple[int, int, int, float, float]] = deque(
            maxlen=self._horizon
        )
        #: How many improvements have been distributed (diagnostics).
        self.credited = 0

    def push(
        self,
        cell: int,
        destroy_idx: int,
        repair_idx: int,
        destroy_reward: float,
        repair_reward: float,
    ) -> None:
        """Remember one decision in case a later step capitalises on it."""
        self._pending.append(
            (
                int(cell),
                int(destroy_idx),
                int(repair_idx),
                float(destroy_reward),
                float(repair_reward),
            )
        )

    def bonus_for(
        self,
        bandits: dict[str, ContextualThompsonBandit],
        *,
        reference: float | None,
        candidate: float | None,
    ) -> float:
        """Distribute credit if ``candidate`` is a large enough improvement.

        Returns the peak bonus applied (0.0 when nothing was credited), which
        the caller reports so the mechanism is visible in run statistics.
        """
        if reference is None or candidate is None:
            return 0.0
        try:
            ref = float(reference)
            cand = float(candidate)
        except (TypeError, ValueError):
            return 0.0
        if not (math.isfinite(ref) and math.isfinite(cand)):
            return 0.0
        scale = max(abs(ref), 1.0)
        gain = (ref - cand) / scale
        if not math.isfinite(gain) or gain < self._min_gain:
            return 0.0

        destroy_bandit = bandits.get("destroy")
        repair_bandit = bandits.get("repair")
        if destroy_bandit is None or repair_bandit is None:
            return 0.0
        # Saturation: a huge improvement is not worth proportionally more credit
        # than a solid one, or one lucky leap would dominate the whole posterior.
        strength = min(1.0, gain / 0.02)
        peak = self._bonus * strength
        entries = list(self._pending)
        for age, (cell, d_idx, r_idx, d_reward, r_reward) in enumerate(
            reversed(entries)
        ):
            share = peak * (self._decay**age)
            if share <= 0.0:
                continue
            destroy_bandit.update(
                cell, d_idx, min(1.0, max(0.0, d_reward + share))
            )
            repair_bandit.update(
                cell, r_idx, min(1.0, max(0.0, r_reward + share))
            )
        if peak > 0.0:
            self.credited += 1
        return peak

    def statistics(self) -> dict[str, Any]:
        return {
            "horizon": self._horizon,
            "decay": round(self._decay, 3),
            "bonus": round(self._bonus, 3),
            "credited_improvements": int(self.credited),
            "pending": len(self._pending),
        }
