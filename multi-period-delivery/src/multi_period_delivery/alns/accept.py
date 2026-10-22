"""
Acceptance policy helpers.

Responsibilities:
- Provide ``DemandConstraintAccept`` to add unmet-demand penalties to candidate
  objective values.
- Adjust ``objective()`` through ``_CandidateProxy`` without changing the
  original state interface.
- Track calls, acceptances, and accumulated penalties for diagnostics and tuning.
"""

from typing import Any

import numpy as np

from .alns_config import default_config as ALNSConfig
from .progress import resolve_progress
from .swallowed import record_swallowed


def _safe_float(value: Any, default: float = float("inf")) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


class _CandidateProxy:
    """
    Proxy for a candidate state.

    The proxy wraps the original object, overrides ``objective()`` to add the
    computed penalty, and delegates all other attribute access for compatibility.

    Design rationale:
    - The original candidate state remains unchanged.
    - Objective behavior is adjusted only during the acceptance decision.
    - Constraint penalties can be combined with any base acceptor, such as
      simulated annealing.
    """

    def __init__(self, original, penalty: float):
        self._orig = original
        self._penalty = float(penalty)

    def objective(self):
        """
        Return the penalized objective value.

        The original objective is converted to float when possible. Exceptions
        and non-finite values are preserved, and finite values are increased by
        the precomputed penalty.
        """
        try:
            base = float(self._orig.objective())
        except Exception:
            base = float("inf")
        if not np.isfinite(base):
            return base
        return base + self._penalty

    def validate(self):
        """Delegate validation to preserve the original behavior."""
        return self._orig.validate()

    def compute_shipped(self):
        """
        Preserve compatibility with existing tests and callers.

        If the wrapped object implements ``compute_shipped()``, delegate to it;
        otherwise return an empty dictionary.
        """
        if hasattr(self._orig, "compute_shipped"):
            return self._orig.compute_shipped()
        return {}

    def __getattr__(self, name: str) -> Any:
        """
        Delegate attributes that are not explicitly overridden.

        This includes ``vehicles``, ``s_ikt``, and other methods or attributes
        provided by the wrapped object, allowing the proxy to substitute for it
        in most contexts.
        """
        return getattr(self._orig, name)


class DemandConstraintAccept:
    """
    Wrapper acceptor that adds unmet-demand penalties.

    Before delegating to the base acceptor, this class inspects the candidate's
    validation result and computes a penalty. It then wraps the candidate in
    ``_CandidateProxy`` so ``objective()`` includes the penalty without changing
    the original state object. Statistics are collected for later analysis.

    Minimum interface contract expected by tests:
      - Callable:
        ``(rng, best_state, current_state, candidate_state) -> bool``
      - ``get_statistics() -> dict``

    Penalty calculation:
      - Call ``candidate_state.validate()`` to obtain feasibility and violations.
      - If infeasible and ``violations["unmet_demand"]`` is a list:
          * Sum ``max(0, demand - shipped)`` across list entries.
          * Apply ``penalty = penalty_factor * unmet_total``.
      - Otherwise, use the number of entries or a fallback value of 1.0.
    """

    def __init__(self, base_acceptor, data, penalty_factor: float = 1000.0):
        """
        Parameters
        ----------
        base_acceptor
            Base acceptance policy or callable object.
        data
            Reference to the problem data for optional diagnostics.
        penalty_factor : float
            Scaling factor that maps unmet demand to the objective scale.

        Statistics:
          - ``_calls``: number of calls.
          - ``_accepted``: number of accepted candidates.
          - ``_total_penalty``: accumulated penalty for averages and diagnostics.
        """
        self.base = base_acceptor
        self.data = data
        self.penalty_factor = float(penalty_factor)

        # statistics
        self._calls = 0
        self._accepted = 0
        self._total_penalty = 0.0

    def _resolve_progress(self, *states) -> float:
        """
        Fraction of the search budget already consumed (0.0 - 1.0).

        Delegates to ``progress.resolve_progress``, which measures elapsed
        runtime against ``MAX_RUNTIME`` so the phase thresholds mean the same
        thing on every machine; see that module for why the iteration-based
        measure cannot be machine-independent.
        """
        return resolve_progress(states)

    def _count_used_vehicles(self, state) -> int:
        if state is None:
            return 0
        try:
            if hasattr(state, "iter_used_vehicles") and callable(
                getattr(state, "iter_used_vehicles")
            ):
                return len(list(state.iter_used_vehicles()))
        except Exception:
            pass
        try:
            vehicles = list(getattr(state, "vehicles", []) or [])
        except Exception:
            vehicles = []
        count = 0
        for veh in vehicles:
            try:
                if veh is None or veh.is_empty():
                    continue
            except (AttributeError, TypeError):
                pass
            count += 1
        return count

    def _actual_priority_accept(self, current_state, candidate_state, progress: float) -> bool:
        current_actual = _safe_float(
            getattr(current_state, "calculate_cost", lambda: None)()
        )
        candidate_actual = _safe_float(
            getattr(candidate_state, "calculate_cost", lambda: None)()
        )
        current_search = _safe_float(
            getattr(current_state, "objective", lambda: None)()
        )
        candidate_search = _safe_float(
            getattr(candidate_state, "objective", lambda: None)()
        )

        actual_abs_tol = float(
            getattr(ALNSConfig, "ACTUAL_PRIORITY_ACTUAL_ABS_TOL", 0.25)
        )
        actual_rel_tol = float(
            getattr(ALNSConfig, "ACTUAL_PRIORITY_ACTUAL_REL_TOL", 0.002)
        )
        search_abs_tol = float(
            getattr(ALNSConfig, "ACTUAL_PRIORITY_SEARCH_ABS_TOL", 1.0)
        )
        search_rel_tol = float(
            getattr(ALNSConfig, "ACTUAL_PRIORITY_SEARCH_REL_TOL", 0.01)
        )

        actual_tol = max(actual_abs_tol, actual_rel_tol * abs(current_actual))
        search_tol = max(search_abs_tol, search_rel_tol * abs(current_search))

        fleet_hard_enabled = bool(
            getattr(ALNSConfig, "ENABLE_LATE_FLEET_HARD_PRIORITY", True)
        )
        fleet_hard_start = float(
            getattr(ALNSConfig, "LATE_FLEET_HARD_PRIORITY_START_PROGRESS", 0.60)
        )
        if fleet_hard_enabled and progress >= fleet_hard_start:
            cur_veh = self._count_used_vehicles(current_state)
            cand_veh = self._count_used_vehicles(candidate_state)
            max_actual_worse = float(
                getattr(ALNSConfig, "LATE_FLEET_HARD_PRIORITY_MAX_ACTUAL_WORSE", 0.75)
            )
            max_actual_rel_worse = float(
                getattr(ALNSConfig, "LATE_FLEET_HARD_PRIORITY_MAX_ACTUAL_REL_WORSE", 0.01)
            )
            allowed_actual_worse = max(
                max_actual_worse,
                max_actual_rel_worse * abs(current_actual),
            )

            if cand_veh < cur_veh:
                # Late in the search, favor fewer vehicles while limiting actual
                # objective movement and preventing a large search degradation.
                return (
                    candidate_actual <= current_actual + allowed_actual_worse
                    and candidate_search <= current_search + 2.0 * search_tol
                )
            if cand_veh > cur_veh:
                return candidate_actual + actual_tol < current_actual

        if candidate_actual + actual_tol < current_actual:
            return True
        if abs(candidate_actual - current_actual) <= actual_tol:
            return candidate_search <= current_search + search_tol
        return False

    def __call__(self, rng, best_state, current_state, candidate_state) -> bool:
        """
        Execute the acceptance decision.

        1) Increment the call counter.
        2) Validate the candidate and collect violations.
        3) Compute the penalty from the violations.
        4) Wrap the candidate so ``objective()`` includes the penalty.
        5) Delegate to the base acceptor with signature fallbacks.
        6) Update acceptance statistics and return the result.
        """
        self._calls += 1

        # post_repair marks a candidate false when rollback or the post-rollback
        # certification could not be completed. Such a state must not be
        # accepted merely because a later validator call happens to succeed on
        # stale derived data.
        if getattr(candidate_state, "_post_repair_certified", None) is False:
            return False

        # An unverified candidate is not safe to accept. Record the failure so a
        # systematic validator problem cannot silently disable the constraint.
        try:
            feasible, violations = candidate_state.validate()
        except Exception as exc:
            record_swallowed("accept.DemandConstraintAccept.validate", exc)
            return False

        penalty = 0.0
        if not feasible:
            if not isinstance(violations, dict):
                return False
            hard_violation_keys = (
                "negative_inventory",
                "veh_over_load",
                "plant_inv_exceed",
                "cargo_day_mismatch",
                "check_error",
                "validation_error",
                "inventory_inconsistency",
            )
            if any(violations.get(key) for key in hard_violation_keys):
                return False

            # Inspect unmet-demand entries when present.
            unmet = violations.get("unmet_demand", [])
            if isinstance(unmet, list) and unmet:
                # Parse demand and shipped fields to compute the unmet total.
                unmet_total = 0.0
                for it in unmet:
                    try:
                        d = float(it.get("demand", 0))
                        s = float(it.get("shipped", 0))
                        unmet_total += max(0.0, d - s)
                    except (AttributeError, TypeError, ValueError):
                        # Fall back to counting the entry when values are invalid.
                        unmet_total += 1.0
                penalty = self.penalty_factor * unmet_total
            else:
                # Fall back to list length or a default penalty of 1.0.
                penalty = self.penalty_factor * (
                    len(unmet) if isinstance(unmet, list) else 1.0
                )
                if not unmet:
                    return False

        # Accumulate the penalty for reporting.
        self._total_penalty += float(penalty)

        progress = self._resolve_progress(current_state, candidate_state, best_state)
        actual_priority_start = float(
            getattr(ALNSConfig, "ACTUAL_PRIORITY_START_PROGRESS", 0.7)
        )

        if feasible and progress >= actual_priority_start:
            accepted = self._actual_priority_accept(
                current_state,
                candidate_state,
                progress,
            )
            if accepted:
                self._accepted += 1
            return accepted

        # Wrap the candidate so objective() returns base + penalty.
        proxy = _CandidateProxy(candidate_state, penalty)

        # Delegate to base acceptor (most acceptors follow signature used in tests)
        # Try the common four-argument signature, then a fallback signature.
        try:
            accepted = bool(self.base(rng, best_state, current_state, proxy))
        except TypeError:
            # Some implementations may use a three-argument signature.
            try:
                accepted = bool(self.base(best_state, current_state, proxy))
            except Exception:
                # Treat repeated failures as rejection.
                accepted = False
        except Exception:
            # Treat other exceptions as rejection.
            accepted = False

        if accepted:
            self._accepted += 1
        return accepted

    def get_statistics(self) -> dict[str, float]:
        """
        Return acceptance statistics.

        Keys:
          - ``calls``: total calls.
          - ``accepted``: accepted candidates.
          - ``accept_rate``: accepted candidates divided by calls.
          - ``avg_penalty``: average penalty per call.
          - ``total_penalty``: accumulated penalty.
        """
        calls = self._calls
        accepted = self._accepted
        accept_rate = (accepted / calls) if calls else 0.0
        avg_penalty = (self._total_penalty / calls) if calls else 0.0
        return {
            "calls": calls,
            "accepted": accepted,
            "accept_rate": float(accept_rate),
            "avg_penalty": float(avg_penalty),
            "total_penalty": float(self._total_penalty),
        }
