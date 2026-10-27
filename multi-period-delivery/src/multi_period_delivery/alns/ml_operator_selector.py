"""
ml_operator_selector
====================

Purpose
    Provide an ML-enhanced operator selector for the ALNS framework.
    MLOperatorSelector learns from historical
    (state feature, operator, improvement) samples to predict the expected
    improvement of destroy and repair operators in the current state. It can
    replace purely random roulette-wheel or score-decay selection with a
    data-driven scheduling policy.

Core design
    1. Feature extraction (extract_state_features):
         Convert the current SolutionState into a numeric feature vector:
           - objective value, with validate enforcing inventory consistency
           - vehicle count, average utilization, and type distribution
           - overall inventory level, negative-inventory ratio, and inventory
             capacity utilization
           - demand satisfaction ratio and unmet-demand ratio
           - search-progress features such as iteration progress and recent
             improvement when the tracker provides them
    2. Sample recording (record_operator_performance):
         After a destroy+repair iteration, or after any operator execution,
         append (state features + operator one-hot encoding) -> improvement to
         the buffer.
    3. Model training (train_models):
         When sample count reaches min_samples and the retrain interval triggers,
         standardize features with StandardScaler and choose a model by sample
         size:
           - fewer than 50 samples: Ridge linear regression to reduce overfitting
           - 50 or more samples: RandomForestRegressor for nonlinear effects
         Destroy and repair operators are modeled independently.
    4. Prediction and selection (select_best_operator):
         If no model is ready, choose a candidate randomly to preserve
         exploration. Otherwise concatenate state features and operator one-hot
         encodings, transform them with the scaler, predict performance, sort by
         prediction, and apply epsilon-greedy selection:
              epsilon probability of random exploration, default 0.1
              1 - epsilon probability of selecting the best predicted operator
    5. Parameter retrieval:
         Call ParamAutoTuner.get_operator_params(op_name) for dynamic operator
         parameters.

Important constraints
    - The selector does not modify state directly; it only reads features.
    - state.objective() may trigger validate() -> compute_inventory() and keeps
      inventory-related features consistent.
    - Missing tracker statistics such as max_iterations or recent_improvement
      default to 0.
    - last_train_iter is updated only at the end of train_models. Insufficient
      samples can therefore delay the next training trigger. This is a
      deliberate simplification.
    - Operator one-hot encoding depends on the registration-order tuples
      DESTROY_OPERATOR_NAMES and REPAIR_OPERATOR_NAMES. Unknown operators
      produce an all-zero vector instead of raising an exception.

Potential improvements (not implemented)
    - Use different modeling targets for destroy and repair, such as distinct
      penalties for negative inventory.
    - Use an online incremental model such as SGDRegressor to avoid full
      retraining cost.
    - Expose feature-importance diagnostics.
    - Dynamically reduce epsilon as the search progresses.

Feature-vector layout from extract_state_features
    [
      0: objective_value, or 1e6 when non-finite,
      1: num_vehicles,
      2: avg_vehicle_utilization,
      3..(2 + |veh_types|): frequency ratio for each vehicle type, sorted by
         numeric value when possible,
      next: avg_inventory_level,
      next: negative_inventory_ratio,
      next: avg_inventory_usage_ratio, the average positive inventory usage
         relative to plant capacity,
      next: demand_satisfaction_ratio,
      next: unmet_demand_ratio,
      next: search_progress_ratio,
      last: recent_improvement_indicator
    ]
    Missing information, such as no vehicles or uninitialized inventory, is
    represented by 0 to keep the vector shape stable.

Typical workflow
    1) selector = MLOperatorSelector(param_tuner, rng)
    2) Call record_operator_performance(...) after each operator execution.
    3) Call train_models() periodically.
    4) Call select_best_operator(state, 'destroy', candidates) when the next
       operator must be selected.
"""

from __future__ import annotations

# =========================
# Standard library
# =========================
import math
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

# =========================
# Third-party libraries
# =========================
import numpy as np
import numpy.random as rnd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from .alns_config import default_config as ALNSConfig

# =========================
# Internal project dependencies
# =========================
from .errors import STATE_OPERATION_ERRORS
from .param_tuner import ParamAutoTuner
from .progress import basis_name, resolve_progress
from .swallowed import record_swallowed

#: Exceptions a ``scaler.transform`` + ``model.predict`` pair can raise for this
#: selector. Ridge and RandomForest validate their input and raise ``ValueError``
#: (shape, feature count, non-finite); numpy's own solve can raise its
#: ``LinAlgError``; attribute and type errors cover a half-initialised model.
#: Nothing outside this set is expected, so anything else surfaces instead of
#: being absorbed as "prediction failed".
PREDICTION_ERRORS = STATE_OPERATION_ERRORS + (np.linalg.LinAlgError,)

if TYPE_CHECKING:
    from .alnsopt import SolutionState


# These tuples mirror the registration order in
# ``ALNSOptimizer._register_destroy_operators`` and
# ``ALNSOptimizer._register_repair_operators``. Keeping the order in one place
# prevents one-hot indexes from silently drifting between training and reports.
DESTROY_OPERATOR_NAMES: tuple[str, ...] = (
    "random_removal",
    "shaw_removal",
    "periodic_shaw_removal",
    "worst_removal",
    "surplus_inventory_removal",
    "path_removal",
    "cross_period_dealer_removal",
    "period_range_removal",
    "infeasible_removal",
    "inventory_level_removal",
)

REPAIR_OPERATOR_NAMES: tuple[str, ...] = (
    "greedy_repair",
    "cross_period_consolidation_repair",
    "local_search_repair",
    "inventory_balance_repair",
    "smart_batch_repair",
    "regret_based_repair",
    "fleet_consolidation_repair",
    "sequential_repair",
    "lookahead_repair",
)

# Operators in this set intentionally have no tunable parameters. Avoiding a
# tuner lookup for them prevents false "unregistered operator" warnings when a
# newly added ALNS operator has not yet been added to OPERATOR_DEFAULTS.
PARAMETERLESS_OPERATOR_NAMES = frozenset(
    {
        "infeasible_removal",
        "inventory_balance_repair",
        "cross_period_consolidation_repair",
        "fleet_consolidation_repair",
    }
)


def _rank_auc(scores: Any, labels: Any) -> float | None:
    """Rank-based AUC with ties counted as 1/2, or ``None`` for one class.

    ``sklearn.metrics.roc_auc_score`` would do the same, but it validates and
    copies its inputs on every call and the gate runs at each retrain; the
    rank formula is enough for tens of rows and keeps the dependency surface
    unchanged.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    finite = np.isfinite(scores)
    if not finite.all():
        scores, labels = scores[finite], labels[finite]
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=float)
    ranks[order] = np.arange(1, scores.size + 1, dtype=float)
    ordered = scores[order]
    start = 0
    for i in range(1, scores.size + 1):
        if i == scores.size or ordered[i] != ordered[start]:
            if i - start > 1:
                ranks[order[start:i]] = float(np.mean(ranks[order[start:i]]))
            start = i
    rank_sum = float(ranks[labels == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def _kind_label(kind: str) -> str:
    """Noun for the log line, so a classifier is not reported as a model."""
    return "classifier" if kind == "classification" else "model"


def _quality_label(quality: dict[str, Any]) -> str:
    """Format a fit report for the log line, tolerating an unreported metric."""
    if quality.get("kind") == "classification":
        auc = quality.get("auc")
        auc_text = "n/a" if auc is None else f"{float(auc):.3f}"
        return (
            f"pooled AUC={auc_text}, ok={quality.get('ok')}, "
            f"reason={quality.get('reason')}"
        )
    r2 = quality.get("r2")
    r2_text = "n/a" if r2 is None else f"{float(r2):.3f}"
    return (
        f"holdout R2={r2_text}, ok={quality.get('ok')}, "
        f"reason={quality.get('reason')}"
    )


#: Magnitude ``_finite_float`` substitutes for an infinite improvement, and the
#: marker used to recognise those rows when target scaling is estimated.
_INF_CLAMP = 1e6

#: Improvement strictly above this counts as "the operator helped". Zero and
#: negative improvements share the "no gain" class: for selection what matters
#: is whether trying this operator was worth it.
_CLASSIFICATION_THRESHOLD = 0.0


def _finite_float(value: Any, default: float = 0.0) -> float:
    """Return a finite float, replacing NaN and infinities with safe values."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)

    if math.isnan(result):
        return float(default)
    if math.isinf(result):
        return _INF_CLAMP if result > 0 else -_INF_CLAMP
    return result


class MLOperatorSelector:
    """
    ML-enhanced operator selector for both destroy and repair operators.

    Attributes:
        param_tuner: ParamAutoTuner that supplies current parameters for each
            operator, or ``None`` when only operator *selection* is enabled.
            With ``None`` the selector uses the configured parameter defaults;
            that is what makes "ML without tuner" a separate configuration.
        rng: numpy.random.Generator used as the random source.
        destroy_features: Training features for destroy operators.
        destroy_performances: Improvement values for destroy operators.
        repair_features: Training features for repair operators.
        repair_performances: Improvement values for repair operators.
        destroy_op_map: Destroy operator name to one-hot index.
        repair_op_map: Repair operator name to one-hot index.
        destroy_model: Ridge, RandomForestRegressor, or None.
        destroy_scaler: StandardScaler or None.
        repair_model: Ridge, RandomForestRegressor, or None.
        repair_scaler: StandardScaler or None.
        min_samples: Minimum samples required before the first training.
        retrain_interval: Retraining interval based on param_tuner.iteration.
        last_train_iter: Iteration number of the most recent training.

    Training and selection:
        - If there are too few samples, skip training and choose randomly to
          preserve exploration.
        - Once the sample threshold is reached, train at the configured
          interval. Use a linear model for small datasets and a random forest
          when enough data is available.
        - Use epsilon-greedy prediction to avoid exploiting too early.
    """

    def __init__(
        self, param_tuner: ParamAutoTuner | None, rng: rnd.Generator
    ):
        # External dependencies
        self.param_tuner = param_tuner
        self.rng = rng

        # Sample buffers
        self.destroy_features: list[np.ndarray] = []
        self.destroy_performances: list[float] = []
        self.repair_features: list[np.ndarray] = []
        self.repair_performances: list[float] = []

        # Operator maps
        self.destroy_op_map: dict[str, int] = {}
        self.repair_op_map: dict[str, int] = {}

        # Models and scalers
        self.destroy_model: Any | None = None
        self.destroy_scaler: StandardScaler | None = None
        self.repair_model: Any | None = None
        self.repair_scaler: StandardScaler | None = None

        # Training controls, with defaults from centralized config
        self.min_samples: int = getattr(ALNSConfig, "ML_INITIAL_SAMPLE_SIZE", 20)
        self.retrain_interval: int = getattr(ALNSConfig, "ML_RETRAIN_INTERVAL", 50)
        self.last_train_iter: int = 0
        # The selector keeps its own iteration clock. Deriving it from the
        # tuner meant that with ``ENABLE_PARAM_TUNER=False`` the iteration was
        # permanently 0 and the interval-triggered retraining never fired.
        self.iteration: int = 0
        self.max_iterations: int = 0
        # Counters that make "ML was actually used" verifiable instead of
        # assumed. They feed ``get_statistics`` and the ablation reports.
        self.selection_calls: dict[str, int] = {"destroy": 0, "repair": 0}
        self.model_selections: dict[str, int] = {"destroy": 0, "repair": 0}
        self.explore_picks: dict[str, int] = {"destroy": 0, "repair": 0}
        self.random_fallbacks: dict[str, int] = {"destroy": 0, "repair": 0}
        self.training_rounds: int = 0
        self.destroy_trained_rounds: int = 0
        self.repair_trained_rounds: int = 0
        #: Per-side out-of-sample quality from the last fit (see
        #: ``_fit_operator_model``). Takeover is gated on ``ok``.
        self.model_quality: dict[str, dict[str, Any]] = {
            "destroy": {"ok": False, "r2": None},
            "repair": {"ok": False, "r2": None},
        }
        # Why ``should_use_ml`` refused, so "ML was enabled but never used" is
        # diagnosable instead of a mystery.
        self.gate_rejections: dict[str, int] = {}
        self.last_gate: dict[str, float | int | bool | None] | None = None
        #: Recent fit quality per side, newest last. One retrain is a single
        #: measurement: a gate that passes once and fails every other time is
        #: not a working gate, and the toggle was invisible in the statistics.
        self.quality_history: dict[str, list[dict[str, Any]]] = {
            "destroy": [],
            "repair": [],
        }
        #: Whether each side is currently *established*: it passed the gate and
        #: has not yet accumulated enough consecutive misses to be dropped.
        #: Without this latch a single noisy retrain flipped takeover off, which
        #: the fit history showed happening four times in one run.
        self._quality_latched: dict[str, bool] = {
            "destroy": False,
            "repair": False,
        }
        self._quality_misses: dict[str, int] = {"destroy": 0, "repair": 0}
        # Feature-vector reuse for one decision (see ``extract_state_features``).
        self._feature_cache_key_value: tuple | None = None
        self._feature_cache_vector: np.ndarray | None = None
        self.feature_cache_hits: int = 0
        self.feature_cache_misses: int = 0
        # True while warm-up sampling runs: those trials still produce training
        # rows but must not touch the selection feedback (streak / cooldown).
        self._suppress_feedback: bool = False
        self.warmup_trials: int = 0
        #: Wall time warm-up sampling has consumed, and why it stopped. The
        #: sample-count threshold alone is not a budget: a slow instance can
        #: spend the search on warm-up and never reach a takeover decision.
        self.warmup_seconds: float = 0.0
        self.warmup_stop_reason: str | None = None
        self._no_improve_streak: dict[str, int] = {"destroy": 0, "repair": 0}
        self._fallback_cooldown: dict[str, int] = {"destroy": 0, "repair": 0}

        # Initialize operator maps
        self._init_operator_maps()
        self._init_param_feature_specs()
        self._ensure_parameterless_tuner_operators()

    @contextmanager
    def warmup_scope(self):
        """Collect training rows without feeding selection feedback.

        Used by ``callbacks.ml_warmup_sample``: the sampling runs operators on
        state copies, so its outcome says nothing about the *choice* the
        selector would have made.
        """
        previous = self._suppress_feedback
        self._suppress_feedback = True
        try:
            yield self
        finally:
            self._suppress_feedback = previous

    def warmup_time_limit_seconds(self) -> float:
        """Wall-clock budget warm-up sampling may consume in total.

        Only the runtime basis uses this. Under ``PROGRESS_BASIS="iteration"``
        the limit is a fraction of the planned iterations instead, because
        seconds spent warming up would otherwise depend on the host.
        """
        fraction = float(getattr(ALNSConfig, "ML_WARMUP_BUDGET_FRACTION", 0.15))
        budget = float(getattr(ALNSConfig, "MAX_RUNTIME", 0) or 0)
        if fraction <= 0.0 or budget <= 0.0:
            return 0.0
        return fraction * budget

    def warmup_budget_exhausted(self) -> bool:
        """Whether warm-up must stop starting new sampling trials.

        Follows ``PROGRESS_BASIS``: the runtime basis keeps the historical
        accumulated-seconds comparison, the iteration basis stops once warm-up
        has consumed its share of the planned iterations. Reading only the clock
        here would let a slow host exhaust a fixed-iteration run's warm-up budget
        early - the same class of defect as the tuner's destroy-degree schedule.
        """
        fraction = float(
            getattr(ALNSConfig, "ML_WARMUP_BUDGET_FRACTION", 0.15)
        )
        if fraction <= 0.0:
            return True
        if basis_name() == "iteration":
            planned = int(getattr(self, "max_iterations", 0) or 0)
            if planned <= 0:
                # The run has not told us its iteration budget yet; stopping
                # here would disable warm-up for the whole run.
                return False
            return float(getattr(self, "iteration", 0)) / planned >= fraction
        limit = self.warmup_time_limit_seconds()
        if limit <= 0.0:
            return True
        return self.warmup_seconds >= limit

    def note_warmup_seconds(self, seconds: float) -> None:
        """Accumulate observed warm-up cost; called once per completed trial."""
        self.warmup_trials += 1
        try:
            value = float(seconds)
        except (TypeError, ValueError):
            return
        if value > 0.0 and value == value and value != float("inf"):
            self.warmup_seconds += value
        if self.warmup_stop_reason is None and self.warmup_budget_exhausted():
            self.warmup_stop_reason = "budget"

    def set_iteration(
        self,
        current: int,
        maximum: int,
        no_improve_iters: int = 0,
    ):
        """Synchronize the selector with the current ALNS progress."""
        self.iteration = max(0, int(current))
        self.max_iterations = max(0, int(maximum))
        try:
            if self.param_tuner is not None:
                self.param_tuner.set_iteration(
                    int(current), int(maximum), no_improve_iters=int(no_improve_iters)
                )
        except STATE_OPERATION_ERRORS as exc:
            # The selector can run without the tuner staying in sync, but a
            # permanently failing sync would silently freeze the tuner's phase.
            record_swallowed("ml_operator_selector.set_iteration", exc)

    def _resolve_progress(self, state: SolutionState | None) -> float:
        """
        Fraction of the search budget consumed, via the shared measure.

        Gated by this value: the ML takeover threshold
        (``ML_TAKEOVER_MIN_PROGRESS``) and the epsilon-greedy exploration decay
        ``base + (1 - progress) * early_boost``. Both are fractions of the
        budget, so they must not depend on how fast the host is; see
        ``progress.py``.
        """
        return resolve_progress((state,) if state is not None else ())

    def _sample_count(self, op_type: str) -> int:
        if op_type == "destroy":
            return len(self.destroy_features)
        return len(self.repair_features)

    def _takeover_min_samples(self) -> int:
        base = int(getattr(ALNSConfig, "ML_TAKEOVER_MIN_SAMPLE_SIZE", self.min_samples))
        if self.param_tuner is not None and bool(getattr(self.param_tuner, "small_instance", False)):
            small_base = int(getattr(ALNSConfig, "ML_TAKEOVER_MIN_SAMPLE_SIZE_SMALL", base))
            return max(base, small_base)
        return base

    def should_force_fallback(self, op_type: str) -> bool:
        if self._suppress_feedback:
            # Warm-up must not drain a real cooldown either: otherwise the
            # sampling calls silently shorten the cooldown the selector needs.
            return False
        cooldown = int(self._fallback_cooldown.get(op_type, 0))
        if cooldown <= 0:
            return False
        self._fallback_cooldown[op_type] = max(0, cooldown - 1)
        return True

    def update_op_feedback(self, op_type: str, improvement: float):
        improvement = _finite_float(improvement, 0.0)
        key = "destroy" if op_type == "destroy" else "repair"

        # Warm-up sampling runs the operators on state *copies* to collect
        # training rows. Those trials are not selection decisions, so feeding
        # them into the stagnation streak armed the cooldown before the selector
        # ever made a choice (measured: 51 cooldown rejections, 2 model-based
        # selections in a 106-iteration run).
        if self._suppress_feedback:
            return

        if improvement > 0:
            self._no_improve_streak[key] = 0
            return

        # Only trials the selector actually chose may feed the streak. During a
        # cooldown the outcomes come from the random fallback (and from warmup
        # sampling on state copies), and those calls arrive faster than the
        # cooldown decrements: the streak reached 18 again inside the cooldown,
        # re-armed it, and ML stayed off for the whole run (measured: 90
        # cooldown rejections, 0 model-based selections).
        if int(self._fallback_cooldown.get(key, 0)) > 0:
            return

        streak = int(self._no_improve_streak.get(key, 0)) + 1
        self._no_improve_streak[key] = streak
        limit = int(getattr(ALNSConfig, "ML_STAGNATION_NO_IMPROVE_LIMIT", 18))
        if streak >= limit:
            cooldown = int(getattr(ALNSConfig, "ML_FALLBACK_COOLDOWN_ITERS", 30))
            self._fallback_cooldown[key] = max(
                int(self._fallback_cooldown.get(key, 0)),
                max(1, cooldown),
            )
            self._no_improve_streak[key] = 0

    def should_use_ml(
        self,
        state: SolutionState | None,
        op_type: str,
        min_sample_size: int | None = None,
    ) -> bool:
        if self.should_force_fallback(op_type):
            return self._record_rejection("cooldown")
        return self.is_ready(state, op_type, min_sample_size, record=True)

    def is_ready(
        self,
        state: SolutionState | None,
        op_type: str,
        min_sample_size: int | None = None,
        record: bool = False,
    ) -> bool:
        """Whether ML may act, *without* consuming a cooldown tick.

        ``should_use_ml`` is called once per executed operator and is allowed to
        advance the cooldown. Readiness is asked separately by the takeover hook
        that decides whether the ML operator should replace the roulette, so the
        two questions must not share side effects - otherwise every iteration
        would drain two cooldown ticks.
        """
        samples = self._sample_count(op_type)
        required_samples = self._takeover_min_samples()
        if min_sample_size is not None:
            required_samples = max(required_samples, int(min_sample_size))
        progress = self._resolve_progress(state)
        model = self.destroy_model if op_type == "destroy" else self.repair_model
        scaler = self.destroy_scaler if op_type == "destroy" else self.repair_scaler
        takeover_progress = float(
            getattr(ALNSConfig, "ML_TAKEOVER_MIN_PROGRESS", 0.30)
        )
        self.last_gate = {
            "samples": samples,
            "required_samples": required_samples,
            "progress": progress,
            "takeover_progress": takeover_progress,
            "model": model is not None,
        }

        if int(self._fallback_cooldown.get(op_type, 0)) > 0:
            return self._record_rejection("cooldown") if record else False
        if samples < required_samples:
            return self._record_rejection("samples") if record else False
        if progress < takeover_progress:
            return self._record_rejection("progress") if record else False
        if model is None or scaler is None:
            return self._record_rejection("model") if record else False
        if not self._quality_ok(op_type):
            # A fitted model is not the same as a *useful* one: the holdout must
            # beat the sample mean before the model is allowed to take over.
            return self._record_rejection("quality") if record else False
        return True

    def _record_rejection(self, reason: str) -> bool:
        self.gate_rejections[reason] = self.gate_rejections.get(reason, 0) + 1
        return False

    def record_operator_level_fallback(self, op_type: str) -> None:
        """Count a fallback decided *inside* the ML operator wrapper.

        ``select_best_operator`` already counts its own random fallbacks, but
        ``ml_based_destroy`` / ``ml_based_repair`` fall back *before* calling it
        whenever the gate refuses. Without this counter a run can report
        "0 random fallbacks" while most iterations never consulted the model.
        """
        key = "destroy" if str(op_type).lower() == "destroy" else "repair"
        self.random_fallbacks[key] = self.random_fallbacks.get(key, 0) + 1

    def _dynamic_epsilon(self, state: SolutionState | None) -> float:
        progress = self._resolve_progress(state)
        base = float(getattr(ALNSConfig, "ML_EPSILON_BASE", 0.10))
        early_boost = float(getattr(ALNSConfig, "ML_EPSILON_EARLY_BOOST", 0.15))
        epsilon = base + (1.0 - progress) * early_boost
        if self.param_tuner is not None and bool(getattr(self.param_tuner, "small_instance", False)):
            epsilon += float(getattr(ALNSConfig, "ML_EPSILON_SMALL_INSTANCE_BOOST", 0.05))
        return max(0.02, min(0.60, epsilon))

    # ------------------------------------------------------------------
    # Initialization and feature extraction
    # ------------------------------------------------------------------
    def _init_operator_maps(self):
        """Initialize one-hot encoding indexes for destroy and repair operators."""
        for i, op in enumerate(DESTROY_OPERATOR_NAMES):
            self.destroy_op_map[op] = i
        for i, op in enumerate(REPAIR_OPERATOR_NAMES):
            self.repair_op_map[op] = i

    def _init_param_feature_specs(self) -> None:
        """Build one-hot × normalised-parameter interaction features.

        ``encode_operator`` used to be pure one-hot, so a sample taken while the
        tuner had ``k=2`` and one taken at ``k=5`` produced the *same* row while
        their improvements differed. That multi-valued target is the
        non-stationarity behind ``full`` scoring worse than ``ml`` (measured
        803.23 vs 794.88 in M51). Each tunable parameter becomes its own feature
        slot, and only the row of the operator that owns the slot is non-zero,
        so a linear model can learn a per-operator slope instead of sharing one
        coefficient across operators.

        Parameters without a tuning range are excluded: they never change, so
        the one-hot already identifies them and an extra constant column would
        only add noise.
        """
        specs_by_op: dict[str, dict[str, list[tuple[int, str, float, float]]]] = {
            "destroy": {},
            "repair": {},
        }
        dims: dict[str, int] = {"destroy": 0, "repair": 0}
        for op_type, names in (
            ("destroy", DESTROY_OPERATOR_NAMES),
            ("repair", REPAIR_OPERATOR_NAMES),
        ):
            offset = 0
            for name in names:
                entry = ALNSConfig.get_operator_default(name) or {}
                ranges = entry.get("ranges") or {}
                slots: list[tuple[int, str, float, float]] = []
                for key in sorted(ranges):
                    bounds = ranges[key]
                    try:
                        lo, hi = float(bounds[0]), float(bounds[1])
                    except (TypeError, ValueError, IndexError):
                        continue
                    if not (hi > lo):
                        continue
                    slots.append((offset, str(key), lo, hi))
                    offset += 1
                specs_by_op[op_type][name] = slots
            dims[op_type] = offset
        self._param_specs_by_op = specs_by_op
        self._param_feature_dim = dims
        # Resolve the switches once, at construction. Training and prediction
        # must agree on the feature width; reading the config on every call
        # would let a mid-run edit silently change the width under a fitted
        # scaler.
        requested = bool(getattr(ALNSConfig, "ML_OPERATOR_PARAM_FEATURES", True))
        has_tuner = self.param_tuner is not None
        repair_requested = bool(
            getattr(ALNSConfig, "ML_REPAIR_PARAM_FEATURES", False)
        )
        self._param_features_enabled = {
            "destroy": requested and has_tuner,
            "repair": requested and has_tuner and repair_requested,
        }
        for op_type, enabled in self._param_features_enabled.items():
            if not enabled:
                self._param_feature_dim[op_type] = 0
        #: Which target each fitted model predicts. Prediction branches on this
        #: rather than on the model class, so a stub or wrapper model behaves
        #: exactly like the estimator it replaced.
        self.model_kind: dict[str, str] = {
            "destroy": "regression",
            "repair": "regression",
        }

    def _ensure_parameterless_tuner_operators(self):
        """Register missing no-parameter operators in the optional tuner.

        The ALNS registration layer wraps every repair operator and may ask the
        tuner for its parameters before execution. A newly added parameterless
        operator can therefore produce a harmless but noisy "unregistered
        operator" warning until the centralized tuner defaults are updated.
        The ML selector owns the candidate set and can close this gap safely.
        """
        registry = getattr(self.param_tuner, "operators", None)
        register = getattr(self.param_tuner, "register_operator", None)
        if not isinstance(registry, dict) or not callable(register):
            return

        known_names = set(DESTROY_OPERATOR_NAMES).union(REPAIR_OPERATOR_NAMES)
        for op_name in PARAMETERLESS_OPERATOR_NAMES:
            if op_name in registry or op_name not in known_names:
                continue
            try:
                register(op_name, {}, {})
            except (KeyError, TypeError, ValueError):
                # The selector remains usable even if a custom tuner rejects
                # registration; parameter lookups are already guarded.
                pass

    def extract_state_features(self, state: SolutionState) -> np.ndarray:
        """
        Extract features from the current solution state.

        See the feature-vector description at the top of the file.

        Notes:
            - state.objective() may trigger validate() -> compute_inventory()
              and should not be overused in hot loops.
            - When vehicles, inventory, or tracker data are unavailable, use 0
              placeholders to keep the vector shape stable.

        The vector is reused for the rest of the current decision: one ML
        operator invocation extracts it in ``select_best_operator`` and again in
        ``record_operator_performance`` with the same state object. Measured on
        medium, one extraction costs 4.3 ms, so a naive double extraction spends
        ~8.6 ms per operator call on identical work.

        The cache key is deliberately cheap (iteration, object identity, vehicle
        count, inventory revision) instead of ``_validation_key()``: the latter
        hashes every vehicle's cargo and would cost as much as the extraction it
        is meant to avoid. Because the key is narrow, a hit only happens for the
        same state object within the same iteration, which is exactly the
        select-then-record pair.
        """
        cache_key = self._feature_cache_key(state)
        if cache_key is not None and self._feature_cache_key_value == cache_key:
            self.feature_cache_hits += 1
            return self._feature_cache_vector

        self.feature_cache_misses += 1
        vector = self._extract_state_features_uncached(state)
        # Store the key *after* extraction: the extraction itself calls
        # ``objective()`` -> ``compute_inventory()``, which bumps
        # ``s_ikt.revision``. Caching the pre-extraction key made every lookup
        # miss (measured: revision 46 -> 276 on the first call).
        key_after = self._feature_cache_key(state)
        if key_after is not None:
            self._feature_cache_key_value = key_after
            self._feature_cache_vector = vector
        return vector

    def _feature_cache_key(self, state: SolutionState) -> tuple | None:
        """Cheap identity key for the select-then-record reuse, or ``None``."""
        s_ikt = getattr(state, "s_ikt", None)
        try:
            return (
                int(self.iteration),
                id(state),
                len(getattr(state, "vehicles", ()) or ()),
                getattr(s_ikt, "revision", None),
            )
        except (TypeError, ValueError):
            return None

    def _extract_state_features_uncached(self, state: SolutionState) -> np.ndarray:
        """Build the feature vector. Callers should use ``extract_state_features``."""
        data = state.data
        features: list[float] = []

        # 1. Objective value. Use a large finite value for infinities to
        # maintain numerical stability.
        try:
            obj_value = _finite_float(state.objective(), 1e6)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            obj_value = 1e6
        features.append(obj_value)

        # 2. Vehicle metrics
        num_vehicles = len(state.vehicles)
        features.append(num_vehicles)

        if num_vehicles > 0:
            # Average utilization
            total_util = 0.0
            veh_type_counts: dict[str, int] = {}
            for veh in state.vehicles:
                load = sum(
                    _finite_float(qty, 0.0)
                    * _finite_float(data.sku_sizes.get(sku, 0.0), 0.0)
                    for (sku, _), qty in veh.cargo.items()
                )
                cap = _finite_float(data.veh_type_cap.get(veh.type, 0.0), 0.0)
                util = load / cap if cap > 0 else 0.0
                total_util += _finite_float(util, 0.0)
                veh_type_counts[veh.type] = veh_type_counts.get(veh.type, 0) + 1
            avg_util = _finite_float(total_util / num_vehicles, 0.0)
            features.append(avg_util)

            # Vehicle-type distribution as normalized frequencies. Sort by
            # numeric value when possible and fall back to string ordering.
            def _veh_type_sort_key(v):
                try:
                    return (0, float(v))
                except (TypeError, ValueError):
                    return (1, str(v))

            for veh_type in sorted(
                getattr(data, "all_veh_types", set()) or set(),
                key=_veh_type_sort_key,
            ):
                features.append(veh_type_counts.get(veh_type, 0) / num_vehicles)
        else:
            features.append(0.0)  # Average utilization
            features.extend(
                [0.0] * len(getattr(data, "all_veh_types", set()) or set())
            )

        # 3. Inventory metrics. objective() has already ensured that
        # compute_inventory is called.
        if state.s_ikt:
            inv_values = [
                _finite_float(value, 0.0) for value in state.s_ikt.values()
            ]
            avg_inv = sum(inv_values) / len(inv_values)
            features.append(avg_inv)

            neg_inv_ratio = sum(1 for v in inv_values if v < 0) / len(inv_values)
            features.append(neg_inv_ratio)

            # Compute the mean inventory-to-capacity ratio over plant-day pairs
            plant_day_pos_inv: dict[tuple[str, int], float] = {}
            for (plant, _sku, day), inv in state.s_ikt.items():
                inv = _finite_float(inv, 0.0)
                if inv > 0:
                    plant_day_pos_inv[(plant, day)] = (
                        plant_day_pos_inv.get((plant, day), 0) + inv
                    )
            usage_acc = 0.0
            usage_cnt = 0
            for (plant, day), inv_sum in plant_day_pos_inv.items():
                cap = _finite_float(data.plant_inv_limit.get(plant, 0.0), 0.0)
                if cap > 0:
                    usage_acc += inv_sum / cap
                    usage_cnt += 1
            avg_usage_ratio = usage_acc / usage_cnt if usage_cnt > 0 else 0.0
            features.append(avg_usage_ratio)
        else:
            features.extend([0.0, 0.0, 0.0])

        # 4. Demand fulfillment
        try:
            shipped = state.compute_shipped() or {}
        except STATE_OPERATION_ERRORS as exc:
            record_swallowed("ml_operator_selector.compute_shipped", exc)
            shipped = {}
        total_demand = (
            sum(_finite_float(value, 0.0) for value in data.demands.values())
            if data.demands
            else 0.0
        )
        total_shipped = (
            sum(_finite_float(value, 0.0) for value in shipped.values())
            if shipped
            else 0.0
        )
        satisfaction_ratio = (
            total_shipped / total_demand if total_demand > 0 else 1.0
        )
        features.append(satisfaction_ratio)

        unmet_count = sum(
            1
            for (dealer, sku), demand in data.demands.items()
            if shipped.get((dealer, sku), 0) < demand
        )
        unmet_ratio = unmet_count / len(data.demands) if data.demands else 0.0
        features.append(unmet_ratio)

        # 5. Search progress, optional and tracker-dependent
        tracker = getattr(state, "tracker", None)
        if tracker is not None:
            try:
                stats = tracker.get_statistics() or {}
            except (AttributeError, TypeError, ValueError) as exc:
                record_swallowed("ml_operator_selector.tracker_stats", exc)
                stats = {}
            current_iter = _finite_float(
                stats.get("total_iterations", 0), 0.0
            )
            max_iter = _finite_float(
                stats.get("max_iterations", 1000), 1000.0
            )
            progress = current_iter / max_iter if max_iter > 0 else 0.0
            recent_improvement = _finite_float(
                stats.get("recent_improvement", 0.0), 0.0
            )
            features.append(max(0.0, min(1.0, progress)))
            features.append(recent_improvement)
        else:
            features.extend([0.0, 0.0])

        feature_vector = np.asarray(features, dtype=float)
        return np.nan_to_num(
            feature_vector,
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        )

    # ------------------------------------------------------------------
    # Encoding and recording
    # ------------------------------------------------------------------
    def encode_operator(self, op_name: str, op_type: str) -> np.ndarray:
        """
        Encode an operator name and its current parameters.

        The vector is ``one-hot(operator)`` followed by one slot per tunable
        parameter of that operator, normalised into ``[0, 1]``. Slots of other
        operators stay zero, so the interaction between "which operator" and
        "which parameter value" is explicit instead of being averaged away.
        ``ML_OPERATOR_PARAM_FEATURES=False`` restores the old pure one-hot
        encoding for A/B runs.

        Unknown names return an all-zero vector instead of raising an exception
        that could interrupt the main workflow.
        """
        op_map = self.destroy_op_map if op_type == "destroy" else self.repair_op_map
        # The switch is resolved at construction: without a tuner the parameters
        # never move (constant columns, pure collinearity with the one-hot), and
        # the repair side measured no gain from the slots at all. Resolving it
        # per call would let a config edit change the width under a fitted
        # scaler.
        use_params = bool(self._param_features_enabled.get(op_type, False))
        dim = int(self._param_feature_dim.get(op_type, 0)) if use_params else 0
        encoding = np.zeros(len(op_map) + dim, dtype=float)
        idx = op_map.get(op_name)
        if idx is None:
            return encoding
        encoding[idx] = 1.0
        if dim <= 0:
            return encoding

        slots = self._param_specs_by_op.get(op_type, {}).get(op_name) or ()
        if not slots:
            return encoding
        params = self._operator_params_for_encoding(op_name)
        base = len(op_map)
        for offset, key, lo, hi in slots:
            raw = params.get(key)
            if raw is None:
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            normalised = (value - lo) / (hi - lo)
            encoding[base + offset] = max(0.0, min(1.0, normalised))
        return encoding

    def _operator_params_for_encoding(self, op_name: str) -> dict[str, Any]:
        """Parameters to encode: the tuner's current values, else the defaults."""
        params = self._safe_operator_params(op_name)
        if params:
            return params
        entry = ALNSConfig.get_operator_default(op_name) or {}
        base = entry.get("params")
        return dict(base) if isinstance(base, dict) else {}

    def record_operator_performance(
        self,
        op_name: str,
        op_type: str,
        state: SolutionState,
        improvement: float,
    ):
        """
        Record the improvement produced by an operator in the current state.

        improvement > 0: objective improvement, such as a cost reduction.
        improvement = 0: no change.
        improvement < 0: degradation.

        The external caller defines and computes the concrete improvement
        metric. NaN, +inf, and -inf are converted to finite fallback values
        before they enter the training buffers.
        """
        op_type = str(op_type).lower()
        if op_type not in {"destroy", "repair"}:
            return

        improvement = _finite_float(improvement, 0.0)
        state_features = self.extract_state_features(state)
        op_encoding = self.encode_operator(op_name, op_type)
        features = np.concatenate([state_features, op_encoding])

        if op_type == "destroy":
            self.destroy_features.append(features)
            self.destroy_performances.append(improvement)
            self._trim_buffers("destroy")
        else:
            self.repair_features.append(features)
            self.repair_performances.append(improvement)
            self._trim_buffers("repair")

    @staticmethod
    def _max_samples() -> int:
        """Return the configured buffer cap, falling back to the tracker's cap."""
        try:
            cap = int(getattr(ALNSConfig, "ML_MAX_SAMPLES", 1000))
        except (TypeError, ValueError):
            cap = 1000
        return cap if cap > 0 else 1000

    def _trim_buffers(self, op_type: str) -> None:
        """
        Keep only the most recent samples, mirroring
        ``ALNSTracker.update_ml_data``.

        Without this the feature/performance buffers grow with every recorded
        operator execution and are never released, so memory scales with the
        total number of iterations and each retraining pass rebuilds a model
        from an ever-larger sample set.
        """
        cap = self._max_samples()
        if op_type == "destroy":
            if len(self.destroy_features) > cap:
                self.destroy_features = self.destroy_features[-cap:]
                self.destroy_performances = self.destroy_performances[-cap:]
        else:
            if len(self.repair_features) > cap:
                self.repair_features = self.repair_features[-cap:]
                self.repair_performances = self.repair_performances[-cap:]

    # ------------------------------------------------------------------
    # Model training
    # ------------------------------------------------------------------
    def train_models(self, force: bool = False):
        """
        Train or retrain performance-prediction models for destroy and repair
        operators.

        Training is triggered when force=True, ignoring the interval, or when
        current iteration - last training iteration >= retrain_interval.

        Notes:
            - If there are fewer than min_samples, skip this round and leave
              last_train_iter unchanged so training can be retried immediately.
            - Update last_train_iter only when at least one model, destroy or
              repair, is actually trained.
            - This avoids delaying the first trainable point by the retrain
              interval.
        """
        current_iter = self.iteration
        if self.param_tuner is not None:
            # Keep whichever clock is further along, so the interval is honoured
            # in both "tuner only", "selector only" and combined setups.
            current_iter = max(
                current_iter, int(getattr(self.param_tuner, "iteration", 0) or 0)
            )

        # First fit ignores the interval. ``should_use_ml`` requires a trained
        # model, and the operators only called ``train_models`` *after* that
        # check passed, so waiting for one full retrain interval before the
        # first fit made takeover impossible (measured: 0 training rounds and
        # 0 model-based selections across 20 runs). Steady-state retraining
        # still respects the interval.
        first_fit = self.last_train_iter == 0
        need_train = force or first_fit or (
            current_iter - self.last_train_iter >= self.retrain_interval
        )
        if not need_train:
            return

        trained_destroy = False
        trained_repair = False
        # ---------- Train destroy-operator model ----------
        if len(self.destroy_features) >= self.min_samples:
            X, y = self._valid_training_arrays(
                self.destroy_features,
                self.destroy_performances,
            )
            if X is not None and len(y) >= self.min_samples:
                try:
                    kind, model, scaler, quality = self._fit_for_target(
                        "destroy", X, y
                    )
                    self.destroy_scaler = scaler
                    self.destroy_model = model
                    self.model_kind["destroy"] = kind
                    self.model_quality["destroy"] = quality
                    self._record_quality("destroy", quality)
                    print(
                        f"[ML] Destroy-operator {_kind_label(kind)} trained with "
                        f"{len(X)} samples ({_quality_label(quality)})"
                    )
                    trained_destroy = True
                except PREDICTION_ERRORS as exc:
                    self.destroy_model = None
                    self.destroy_scaler = None
                    self.model_kind["destroy"] = "regression"
                    self.model_quality["destroy"] = {"ok": False, "r2": None}
                    self._record_quality(
                        "destroy", {"ok": False, "reason": "fit_failed"}
                    )
                    print(
                        f"[ML] Destroy-operator model training failed: {exc}; "
                        "falling back to exploration"
                    )

        # ---------- Train repair-operator model ----------
        if len(self.repair_features) >= self.min_samples:
            X, y = self._valid_training_arrays(
                self.repair_features,
                self.repair_performances,
            )
            if X is not None and len(y) >= self.min_samples:
                try:
                    kind, model, scaler, quality = self._fit_for_target(
                        "repair", X, y
                    )
                    self.repair_scaler = scaler
                    self.repair_model = model
                    self.model_kind["repair"] = kind
                    self.model_quality["repair"] = quality
                    self._record_quality("repair", quality)
                    print(
                        f"[ML] Repair-operator {_kind_label(kind)} trained with "
                        f"{len(X)} samples ({_quality_label(quality)})"
                    )
                    trained_repair = True
                except PREDICTION_ERRORS as exc:
                    self.repair_model = None
                    self.repair_scaler = None
                    self.model_kind["repair"] = "regression"
                    self.model_quality["repair"] = {"ok": False, "r2": None}
                    self._record_quality(
                        "repair", {"ok": False, "reason": "fit_failed"}
                    )
                    print(
                        f"[ML] Repair-operator model training failed: {exc}; "
                        "falling back to exploration"
                    )

        # Update the timestamp if at least one model was trained this round.
        # Otherwise keep the old value so the next call can try again.
        if trained_destroy or trained_repair:
            self.last_train_iter = current_iter
            self.training_rounds += 1
            if trained_destroy:
                self.destroy_trained_rounds += 1
            if trained_repair:
                self.repair_trained_rounds += 1

    def _make_regressor(self, n_samples: int):
        """Return an unfitted regressor sized for ``n_samples``."""
        if n_samples < getattr(ALNSConfig, "ML_USE_RF_THRESHOLD", 50):
            return Ridge(alpha=getattr(ALNSConfig, "ML_RIDGE_ALPHA", 1.0))
        return RandomForestRegressor(
            n_estimators=getattr(ALNSConfig, "ML_RF_N_ESTIMATORS", 50),
            max_depth=getattr(ALNSConfig, "ML_RF_MAX_DEPTH", 10),
            random_state=int(self.rng.integers(0, 10_000)),
        )

    # ------------------------------------------------------------------
    # Target selection
    # ------------------------------------------------------------------
    def _target_kind(self, op_type: str) -> str:
        """Which target this side predicts: ``"classification"`` or ``"regression"``.

        The repair side defaults to classification because the magnitude of a
        repair's improvement is not learnable at these sample sizes while
        "did it improve at all" is (see ``ML_REPAIR_TARGET``). The destroy side
        keeps regression: its magnitude did pass the R2 gate in several runs,
        and no measurement yet shows a classifier would do better there.
        """
        if op_type == "repair":
            mode = str(getattr(ALNSConfig, "ML_REPAIR_TARGET", "classification"))
            if mode.strip().lower() == "classification":
                return "classification"
        return "regression"

    def _fit_for_target(
        self,
        op_type: str,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[str, Any, Any, dict[str, Any]]:
        """Fit the estimator the configured target asks for."""
        kind = self._target_kind(op_type)
        if kind == "classification":
            model, scaler, quality = self._fit_classifier(X, y, op_type)
        else:
            model, scaler, quality = self._fit_operator_model(X, y, op_type)
        return kind, model, scaler, quality

    def _make_classifier(self):
        """Return an unfitted logistic classifier for the improvement target."""
        return LogisticRegression(
            C=float(getattr(ALNSConfig, "ML_CLASSIFIER_C", 1.0)),
            max_iter=1000,
            solver="lbfgs",
        )

    def _fit_classifier(
        self,
        X: np.ndarray,
        y: np.ndarray,
        op_type: str,
    ) -> tuple[Any, Any, dict[str, Any]]:
        """Fit a "will this operator improve" classifier and report its AUC.

        A single chronological holdout scores only ``n * 0.25`` rows here (11 at
        n=45) and swung between AUC 0.54 and 0.86 on identical configurations.
        Contiguous time blocks are therefore used and their out-of-fold
        probabilities pooled, so every row is scored once and the gate reads a
        number with usable variance. The deployed model is then refitted on all
        rows: blocking is a measurement device, not a training restriction.
        """
        labels = (np.asarray(y, dtype=float) > _CLASSIFICATION_THRESHOLD).astype(int)
        n = int(labels.size)
        min_auc = float(getattr(ALNSConfig, "ML_CLASSIFIER_MIN_AUC", 0.55))
        floor = int(getattr(ALNSConfig, "ML_CLASSIFIER_MIN_SAMPLES", 24))

        if n < max(floor, int(getattr(ALNSConfig, "ML_MODEL_MIN_TRAIN_SAMPLES", 8))):
            # Too few rows for a blocked estimate to mean anything: 12 rows over
            # 3 folds produced AUC 0.056 on an untrained model. Refuse instead
            # of publishing a number that looks like a result.
            return None, None, {
                "ok": False,
                "reason": "too_few_samples",
                "kind": "classification",
                "auc": None,
                "accuracy": None,
                "n": n,
                "min_auc": min_auc,
                "positive_rate": float(labels.mean()) if n else None,
            }

        if int(labels.min()) == int(labels.max()):
            return None, None, {
                "ok": False,
                "reason": "single_class",
                "kind": "classification",
                "auc": None,
                "accuracy": None,
                "n": n,
                "min_auc": min_auc,
                "positive_rate": float(labels.mean()),
            }

        folds = int(getattr(ALNSConfig, "ML_CLASSIFIER_CV_FOLDS", 5))
        folds = max(2, min(folds, n // 4 or 2))
        probabilities, scored = self._out_of_fold_probabilities(X, labels, folds)
        if probabilities is None or scored < max(8, n // 2):
            return None, None, {
                "ok": False,
                "reason": "too_few_samples",
                "kind": "classification",
                "auc": None,
                "accuracy": None,
                "n": n,
                "min_auc": min_auc,
                "positive_rate": float(labels.mean()),
            }

        auc = _rank_auc(probabilities, labels)
        finite = np.isfinite(probabilities)
        accuracy = (
            float(((probabilities[finite] >= 0.5) == labels[finite]).mean())
            if finite.any()
            else 0.0
        )
        majority = float(max(labels.mean(), 1.0 - labels.mean()))
        if auc is None:
            ok, reason = False, "single_class"
        else:
            ok = bool(auc >= min_auc)
            reason = "ok" if ok else "low_auc"

        # Refit on every row whether or not the gate passed: the regression
        # path keeps a fitted model and lets ``_quality_ok`` refuse takeover, so
        # doing otherwise here would make "model present" mean different things
        # per side in the run statistics.
        scaler = StandardScaler().fit(X)
        model = self._make_classifier().fit(scaler.transform(X), labels)

        return model, scaler, {
            "ok": ok,
            "reason": reason,
            "kind": "classification",
            "auc": auc,
            "accuracy": accuracy,
            "majority_accuracy": majority,
            "folds": folds,
            "n": n,
            "min_auc": min_auc,
            "positive_rate": float(labels.mean()),
        }

    def _out_of_fold_probabilities(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        folds: int,
    ) -> tuple[np.ndarray | None, int]:
        """Pooled out-of-fold positive probabilities over contiguous blocks.

        Blocks are contiguous in time, so each fold is scored by a model that
        only saw older (and, for later folds, newer) rows — the same "no future
        leakage" intent as the chronological holdout, but using every row.
        """
        n = int(labels.size)
        boundaries = np.linspace(0, n, folds + 1, dtype=int)
        probabilities = np.full(n, np.nan, dtype=float)
        for k in range(folds):
            start, stop = int(boundaries[k]), int(boundaries[k + 1])
            if stop <= start:
                continue
            test = np.arange(start, stop)
            train = np.setdiff1d(np.arange(n), test)
            if train.size < 4:
                continue
            train_labels = labels[train]
            if int(train_labels.min()) == int(train_labels.max()):
                # One class only: the honest out-of-fold score is the training
                # prevalence, which is what a model with no signal predicts.
                probabilities[test] = float(train_labels.mean())
                continue
            try:
                scaler = StandardScaler().fit(X[train])
                model = self._make_classifier().fit(
                    scaler.transform(X[train]), train_labels
                )
                probabilities[test] = model.predict_proba(
                    scaler.transform(X[test])
                )[:, 1]
            except PREDICTION_ERRORS as exc:
                record_swallowed("ml_operator_selector.cv_fold", exc)
        scored = int(np.isfinite(probabilities).sum())
        if scored == 0:
            return None, 0
        return probabilities, scored

    def _fit_operator_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        op_type: str,
    ) -> tuple[Any, Any, dict[str, Any]]:
        """Fit scaler+regressor and report out-of-sample quality.

        Takeover used to start as soon as *any* model could be fitted, with no
        evidence that the model predicted better than the sample mean. This
        validates on the most recent samples (a chronological holdout, matching
        how the buffers grow) and only then refits on everything.

        The quality gate is what makes "ML is on" mean "ML knows something":
        ``is_ready`` refuses takeover while ``ok`` is False.
        """
        n = int(len(y))
        clip_bounds = self._winsorise_target(y)
        if clip_bounds is not None:
            y = np.clip(y, clip_bounds[0], clip_bounds[1])
        holdout_fraction = float(
            getattr(ALNSConfig, "ML_MODEL_HOLDOUT_FRACTION", 0.25)
        )
        min_holdout = int(getattr(ALNSConfig, "ML_MODEL_MIN_HOLDOUT_SAMPLES", 8))
        min_train = max(4, int(getattr(ALNSConfig, "ML_MODEL_MIN_TRAIN_SAMPLES", 8)))
        min_r2 = float(getattr(ALNSConfig, "ML_MIN_HOLDOUT_R2", 0.0))

        if n - min_train < min_holdout:
            # Too few samples to validate: fit on everything but do not claim
            # quality, so takeover stays blocked until more samples arrive.
            scaler = StandardScaler().fit(X)
            model = self._make_regressor(n).fit(scaler.transform(X), y)
            return model, scaler, {
                "ok": False,
                "reason": "too_few_samples",
                "r2": None,
                "holdout": 0,
                "n": n,
                "clip": clip_bounds,
            }

        holdout = max(min_holdout, int(round(n * holdout_fraction)))
        holdout = min(holdout, n - min_train)

        split = n - holdout
        scaler = StandardScaler().fit(X[:split])
        model = self._make_regressor(split).fit(
            scaler.transform(X[:split]), y[:split]
        )
        predictions = np.asarray(
            model.predict(scaler.transform(X[split:])), dtype=float
        ).reshape(-1)
        actual = y[split:]
        residual = float(np.sum((actual - predictions) ** 2))
        variance = float(np.sum((actual - float(np.mean(actual))) ** 2))
        if variance <= 1e-12:
            # A constant holdout carries no signal; beating its mean is not
            # evidence of anything, so do not grant takeover.
            r2 = None
            ok = False
            reason = "constant_holdout"
        else:
            r2 = 1.0 - residual / variance
            ok = bool(r2 >= min_r2)
            reason = "ok" if ok else "low_r2"

        if ok:
            # Validation passed: spend the held-out rows on the deployed model.
            scaler = StandardScaler().fit(X)
            model = self._make_regressor(n).fit(scaler.transform(X), y)

        return model, scaler, {
            "ok": ok,
            "reason": reason,
            "r2": r2,
            "holdout": holdout,
            "n": n,
            "min_r2": min_r2,
            "clip": clip_bounds,
        }

    @staticmethod
    def _winsorise_target(y: np.ndarray) -> tuple[float, float] | None:
        """Clip the long tails of the improvement target, in place-of-use.

        Improvement labels carry ``inf -> 1e6`` clamps, so one run-away sample
        can dominate a squared loss and make a learnable signal look like noise
        (measured: repair holdout R2 down to -253.7). Clipping at a two-sided
        MAD band keeps the ordering that operator selection cares about while
        stopping a single row from setting the scale.

        A quantile rule was tried first and rejected by measurement: with 40
        samples the 99th percentile is interpolated between the largest normal
        value and the 1e6 clamp, i.e. it stays at ~6e5 and clips nothing.

        The band has to be estimated *without* the clamps. On a real full run
        the destroy labels were mostly ``+/-1e6`` (a destroy that makes the
        state infeasible), so the plain median/MAD band came out at
        ``[-2.36e6, +1.83e6]`` and clipped nothing. Clamped rows are therefore
        excluded when estimating the centre and spread, then clipped into it.
        """
        n = int(len(y))
        factor = float(getattr(ALNSConfig, "ML_TARGET_CLIP_MAD_FACTOR", 8.0))
        if factor <= 0.0 or n < 8:
            return None
        # ``_finite_float`` maps +/-inf onto +/-1e6. Those rows are a signal
        # ("this operator made the state invalid"), not a magnitude to regress.
        clamped = np.abs(y) >= _INF_CLAMP * 0.999
        clean = y[~clamped]
        base = clean if clean.size >= max(8, n // 2) else y
        try:
            median = float(np.median(base))
            mad = float(np.median(np.abs(base - median)))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(median):
            return None
        if not (math.isfinite(mad) and mad > 0.0):
            # More than half the labels are identical; use the standard
            # deviation scaled to a MAD-equivalent instead of giving up.
            std = float(np.std(base))
            if not (math.isfinite(std) and std > 0.0):
                return None
            mad = std * 0.6744897501960817
        lo, hi = median - factor * mad, median + factor * mad
        if not (math.isfinite(lo) and math.isfinite(hi) and hi > lo):
            return None
        if float(np.min(y)) >= lo and float(np.max(y)) <= hi:
            # Nothing to clip: reporting a band would imply a change that did
            # not happen.
            return None
        return lo, hi

    def _quality_ok(self, op_type: str) -> bool:
        """Whether this side may take over, honouring the hysteresis latch.

        The latch is what turns "passed the gate once" into "is in service".
        A side that never passed still needs the current fit to pass, so a bad
        model cannot be admitted by simply waiting.
        """
        if bool(self._quality_latched.get(op_type, False)):
            return True
        return bool(self.model_quality.get(op_type, {}).get("ok", False))

    def quality_latch_state(self) -> dict[str, dict[str, Any]]:
        """Latch and miss counters, for run statistics and tests."""
        return {
            side: {
                "latched": bool(self._quality_latched.get(side, False)),
                "consecutive_misses": int(self._quality_misses.get(side, 0)),
            }
            for side in ("destroy", "repair")
        }

    def _record_quality(self, op_type: str, quality: dict[str, Any]) -> None:
        """Append one fit's verdict to the bounded history."""
        history = self.quality_history.setdefault(op_type, [])
        history.append(
            {
                "iteration": int(self.iteration),
                "kind": str(quality.get("kind", "regression")),
                "metric": (
                    quality.get("auc")
                    if quality.get("kind") == "classification"
                    else quality.get("r2")
                ),
                "ok": bool(quality.get("ok", False)),
                "reason": str(quality.get("reason", "")),
            }
        )
        limit = int(getattr(ALNSConfig, "ML_QUALITY_HISTORY_SIZE", 20))
        if limit > 0 and len(history) > limit:
            del history[: len(history) - limit]
        self._update_quality_latch(op_type, bool(quality.get("ok", False)))

    def _update_quality_latch(self, op_type: str, passed: bool) -> None:
        """Apply the hysteresis rule to one side.

        A pass establishes (or keeps) service; a miss only counts while the side
        is established, and it takes ``ML_QUALITY_EXIT_MISSES`` consecutive
        misses to withdraw. A non-positive limit restores the old one-miss exit.
        """
        if passed:
            self._quality_latched[op_type] = True
            self._quality_misses[op_type] = 0
            return
        if not self._quality_latched.get(op_type, False):
            return
        misses = int(self._quality_misses.get(op_type, 0)) + 1
        self._quality_misses[op_type] = misses
        limit = int(getattr(ALNSConfig, "ML_QUALITY_EXIT_MISSES", 2))
        if limit <= 0 or misses >= limit:
            self._quality_latched[op_type] = False
            self._quality_misses[op_type] = 0

    def get_statistics(self) -> dict[str, Any]:
        """Return counters proving how the selector actually decided.

        A run can report ``ENABLE_ML_OPERATORS=True`` while every choice was a
        random fallback (model never trained, or training failed silently).
        These counters separate the three cases so an ablation cannot mistake
        "ML was on" for "ML was used".
        """
        return {
            "iteration": int(self.iteration),
            "max_iterations": int(self.max_iterations),
            "trained_with_param_tuner": self.param_tuner is not None,
            "training_rounds": int(self.training_rounds),
            "destroy_trained_rounds": int(self.destroy_trained_rounds),
            "repair_trained_rounds": int(self.repair_trained_rounds),
            "destroy_samples": len(self.destroy_features),
            "repair_samples": len(self.repair_features),
            "destroy_model": self.destroy_model is not None,
            "repair_model": self.repair_model is not None,
            "destroy_selections": int(self.selection_calls["destroy"]),
            "repair_selections": int(self.selection_calls["repair"]),
            "destroy_model_selections": int(self.model_selections["destroy"]),
            "repair_model_selections": int(self.model_selections["repair"]),
            "destroy_explore_picks": int(self.explore_picks["destroy"]),
            "repair_explore_picks": int(self.explore_picks["repair"]),
            "destroy_random_fallbacks": int(self.random_fallbacks["destroy"]),
            "repair_random_fallbacks": int(self.random_fallbacks["repair"]),
            "warmup_trials": int(self.warmup_trials),
            "warmup_seconds": round(float(self.warmup_seconds), 4),
            "warmup_stop_reason": self.warmup_stop_reason,
            "gate_rejections": dict(self.gate_rejections),
            "last_gate": self.last_gate,
            "feature_cache_hits": int(self.feature_cache_hits),
            "feature_cache_misses": int(self.feature_cache_misses),
            "destroy_quality": dict(self.model_quality.get("destroy", {})),
            "repair_quality": dict(self.model_quality.get("repair", {})),
            "quality_history": {
                side: list(entries)
                for side, entries in self.quality_history.items()
            },
            "quality_latch": self.quality_latch_state(),
            # What each side is *configured* to predict, and what the model
            # currently fitted actually predicts. They differ before the first
            # fit, and a failed fit resets the second to the regression default.
            "target_kind": {
                side: self._target_kind(side) for side in ("destroy", "repair")
            },
            "fitted_kind": dict(self.model_kind),
            "param_features": dict(self._param_features_enabled),
            "repair_feature_width": (
                len(self.repair_op_map) + int(self._param_feature_dim.get("repair", 0))
            ),
        }

    # ------------------------------------------------------------------
    # Operator selection
    # ------------------------------------------------------------------
    @staticmethod
    def _valid_training_arrays(
        feature_buffer: list[np.ndarray],
        performance_buffer: list[float],
    ) -> tuple[np.ndarray | None, np.ndarray]:
        """Return finite, aligned training arrays for one operator family."""
        try:
            X = np.asarray(feature_buffer, dtype=float)
            y = np.asarray(performance_buffer, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return None, np.asarray([], dtype=float)

        if X.ndim != 2 or X.shape[0] == 0 or y.size == 0:
            return None, np.asarray([], dtype=float)

        sample_count = min(X.shape[0], y.size)
        X = X[:sample_count]
        y = y[:sample_count]
        valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[valid]
        y = y[valid]
        if X.shape[0] == 0:
            return None, np.asarray([], dtype=float)
        return X, y

    def _safe_operator_params(self, op_name: str) -> dict[str, Any]:
        """Return operator parameters without allowing tuner failures to spread."""
        if (
            self.param_tuner is None
            or op_name in PARAMETERLESS_OPERATOR_NAMES
        ):
            return {}
        try:
            params = self.param_tuner.get_operator_params(op_name)
            return dict(params) if isinstance(params, dict) else {}
        except (AttributeError, TypeError, ValueError):
            return {}

    def _default_operator(
        self,
        op_type: str,
    ) -> tuple[str, dict[str, Any]]:
        """Return a valid default operator for an empty or invalid candidate set."""
        if op_type == "destroy":
            destroy_defaults = ALNSConfig.get_destroy_params()
            return "random_removal", {
                "degree": destroy_defaults.get("random_removal_degree", 0.25)
            }
        repair_defaults = ALNSConfig.get_repair_params()
        return "greedy_repair", dict(repair_defaults.get("greedy_repair", {}))

    def _random_valid_operator(
        self,
        op_type: str,
        candidates: list[str],
    ) -> tuple[str, dict[str, Any]]:
        """Choose a valid candidate uniformly at random."""
        op_map = self.destroy_op_map if op_type == "destroy" else self.repair_op_map
        valid_candidates = [
            str(name) for name in candidates if str(name) in op_map
        ]
        if not valid_candidates:
            valid_candidates = list(op_map)
        if not valid_candidates:
            return self._default_operator(op_type)

        selected = str(self.rng.choice(valid_candidates))
        return selected, self._safe_operator_params(selected)

    def _batched_predictions(
        self,
        model: Any,
        scaler: Any,
        state_features: np.ndarray,
        op_type: str,
        candidates: list[str],
    ) -> list[float | None]:
        """Predict every candidate in one transform+predict call.

        Falls back to the per-candidate loop if the batch fails, so a single
        unusable row still only removes its own candidate. Non-finite
        predictions become ``None`` (never a silent NaN in the sort key).
        """
        kind = self.model_kind.get(op_type, "regression")
        try:
            rows = np.vstack(
                [
                    np.concatenate(
                        [state_features, self.encode_operator(name, op_type)]
                    )
                    for name in candidates
                ]
            )
            raw = self._predict_scores(model, scaler.transform(rows), kind)
            if raw.size != len(candidates):
                raise ValueError(
                    f"model returned {raw.size} predictions for {len(candidates)} candidates"
                )
            return [float(value) if math.isfinite(value) else None for value in raw]
        except PREDICTION_ERRORS as exc:
            record_swallowed("ml_operator_selector.predict_batch", exc)

        predictions: list[float | None] = []
        for name in candidates:
            try:
                row = np.concatenate(
                    [state_features, self.encode_operator(name, op_type)]
                ).reshape(1, -1)
                raw_prediction = float(
                    self._predict_scores(model, scaler.transform(row), kind)[0]
                )
                predictions.append(
                    raw_prediction if math.isfinite(raw_prediction) else None
                )
            except PREDICTION_ERRORS as exc:
                # A failed prediction must not become the selected operator
                # merely because NaN sorts unexpectedly.
                record_swallowed("ml_operator_selector.predict_candidate", exc)
                predictions.append(None)
        return predictions

    @staticmethod
    def _predict_scores(model: Any, X: Any, kind: str) -> np.ndarray:
        """Uniform 1-D score vector for a regression or classification model.

        Both are used only to *rank* candidates, so the classifier's positive
        probability is the natural score: it is monotone in "more likely to
        help", which is exactly the ordering selection needs.
        """
        if kind == "classification":
            probabilities = np.asarray(model.predict_proba(X), dtype=float)
            if probabilities.ndim != 2 or probabilities.shape[1] < 2:
                raise ValueError(
                    "classifier returned no positive-class probability column"
                )
            return probabilities[:, 1].reshape(-1)
        return np.asarray(model.predict(X), dtype=float).reshape(-1)

    def select_best_operator(
        self,
        state: SolutionState,
        op_type: str,
        candidates: list[str],
    ) -> tuple[str, dict[str, Any]]:
        """
        Select an operator and its parameters from the current state and model
        predictions.

        Args:
            state: Current solution.
            op_type: 'destroy' or 'repair'.
            candidates: Candidate operator names. If empty, return a default
                fallback operator.

        Returns:
            A tuple of (op_name, params_dict).

        Strategy:
            - With no candidates or no trained model, choose randomly or use
                the default.
            - With a trained model, predict every candidate and apply
                epsilon-greedy selection to balance exploration and exploitation.
        """
        op_type = str(op_type).lower()
        if op_type not in {"destroy", "repair"}:
            op_type = "repair"

        self.selection_calls[op_type] += 1
        op_map = self.destroy_op_map if op_type == "destroy" else self.repair_op_map
        if not candidates:
            return self._default_operator(op_type)

        valid_candidates = [
            str(name) for name in candidates if str(name) in op_map
        ]
        if not valid_candidates:
            return self._default_operator(op_type)

        try:
            state_features = self.extract_state_features(state)
        except STATE_OPERATION_ERRORS as exc:
            # Falling back to a random operator is the right behaviour, but doing
            # it silently means the run reports that ML operators were enabled
            # while this call site actually chose at random. Record it so a
            # systematically failing feature extractor is visible.
            record_swallowed(
                "ml_operator_selector.extract_state_features", exc
            )
            self.random_fallbacks[op_type] += 1
            return self._random_valid_operator(op_type, valid_candidates)

        model = self.destroy_model if op_type == "destroy" else self.repair_model
        scaler = self.destroy_scaler if op_type == "destroy" else self.repair_scaler

        # The model is not available yet: explore randomly.
        if model is None or scaler is None:
            self.random_fallbacks[op_type] += 1
            return self._random_valid_operator(op_type, valid_candidates)

        # One batched transform+predict instead of one call per candidate.
        # Measured on medium with 19 candidates: 3.42 ms -> 0.29 ms (11.7x).
        # The per-candidate loop is kept as the failure path so that a single
        # bad row still only drops its own candidate, exactly as before.
        raw_predictions = self._batched_predictions(
            model, scaler, state_features, op_type, valid_candidates
        )
        predictions = []
        for op_name, pred in zip(valid_candidates, raw_predictions):
            params = self._safe_operator_params(op_name)
            predictions.append((op_name, pred, params))

        finite_predictions = [
            prediction for prediction in predictions if prediction[1] is not None
        ]
        if not finite_predictions:
            self.random_fallbacks[op_type] += 1
            return self._random_valid_operator(op_type, valid_candidates)

        self.model_selections[op_type] += 1

        # Sort predicted performance in descending order
        finite_predictions.sort(
            key=lambda item: float(item[1]),
            reverse=True,
        )

        # Epsilon-greedy selection with stronger exploration early on and
        # convergence later.
        epsilon = self._dynamic_epsilon(state)
        if self.rng.random() < epsilon:
            self.explore_picks[op_type] += 1
            idx = int(self.rng.integers(0, len(finite_predictions)))
            return finite_predictions[idx][0], finite_predictions[idx][2]
        return finite_predictions[0][0], finite_predictions[0][2]
