"""The exact-simulation guard in ``regret_based_repair`` relies on one invariant.

Measurement (``local-only/probes/_delta_vs_exact_verdict.py``) showed that every
candidate reaching the exact simulation in ``regret_based_repair`` does so for a
single reason: ``prev_obj`` - the while-iteration's ``state.objective()`` - is
``inf`` because the partial state has a hard physical violation. The operator now
skips that simulation, recording the same memo entry it would have produced.

That is only sound because of the contract pinned here: with a non-finite
baseline, ``compute_mixed_operator_reward`` returns *zero improvement and
failure* regardless of the state, because it rejects a non-finite baseline before
it ever looks at the candidate. If that contract ever changes, the guard would
silently start discarding candidates the simulation would have kept.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from multi_period_delivery.alns import repair_operators as ro
from multi_period_delivery.alns.alns_optimizer import ALNSOptimizer
from multi_period_delivery.alns.repair_utils import compute_mixed_operator_reward

BASE = Path(__file__).resolve().parents[1]


def _state():
    optimizer = ALNSOptimizer(
        input_file_loc=str(BASE / "datasets" / "multiple-periods" / "small"),
        output_file_loc=str(Path(tempfile.gettempdir()) / "_exact_sim_guard_probe"),
    )
    optimizer.load_data("dataset_1")
    state = optimizer.create_initial_solution()
    # Keep the reward independent of any live tracker reference.
    state.tracker = None
    return state


@pytest.mark.parametrize(
    "baseline",
    [float("inf"), float("-inf"), float("nan")],
    ids=["inf", "-inf", "nan"],
)
def test_non_finite_baseline_is_zero_improvement_and_failure(baseline: float) -> None:
    """The simulation cannot change this verdict, so skipping it is equivalent."""
    state = _state()

    new_obj, improvement, success = compute_mixed_operator_reward(
        state, baseline, new_obj=123.0
    )

    assert new_obj == 123.0
    assert improvement == 0.0
    assert success is False


def test_finite_baseline_is_evaluated_normally() -> None:
    """The guard's condition must not short-circuit ordinary candidates."""
    state = _state()

    new_obj, improvement, success = compute_mixed_operator_reward(
        state, 100.0, new_obj=50.0
    )

    assert new_obj == 50.0
    # search_norm = (100 - 50) / 100 = 0.5, weighted 0.6, scaled by 100.
    assert improvement == pytest.approx(30.0)
    assert success is True


def _nonfinite_state():
    """A partial state with a hard violation (``objective() == inf``) and unmet demand."""
    state = _state()
    state.vehicles = []
    key = next(iter(state.s_ikt))
    state.s_ikt[key] = -1000
    state.mark_objective_dirty()
    assert state.objective() == float("inf")
    assert ro.get_unmet_demand(state)
    return state


def test_regret_skips_the_simulation_when_the_baseline_is_non_finite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Candidates reach the delta stage, yet none is simulated."""
    state = _nonfinite_state()
    deltas = {"n": 0}
    sims = {"n": 0}
    real_delta = ro.compute_improvement_delta
    real_improvement = ro.compute_improvement

    def counted_delta(*args, **kwargs):
        deltas["n"] += 1
        return real_delta(*args, **kwargs)

    def counted_improvement(s, prev_obj=None):
        sims["n"] += 1
        return real_improvement(s, prev_obj)

    monkeypatch.setattr(ro, "compute_improvement_delta", counted_delta)
    monkeypatch.setattr(ro, "compute_improvement", counted_improvement)

    ro.regret_based_repair(state, np.random.default_rng(0), time_limit=1.0)

    # Candidates did reach the delta stage - the guard, not inactivity, is what
    # kept the simulation count at zero.
    assert deltas["n"] > 0
    assert sims["n"] == 0
