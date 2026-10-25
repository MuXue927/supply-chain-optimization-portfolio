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
from typing import TYPE_CHECKING, Any

# =========================
# Third-party libraries
# =========================
import numpy as np
import numpy.random as rnd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .alns_config import default_config as ALNSConfig

# =========================
# Internal project dependencies
# =========================
from .errors import STATE_OPERATION_ERRORS
from .param_tuner import ParamAutoTuner
from .progress import resolve_progress
from .swallowed import record_swallowed

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


def _finite_float(value: Any, default: float = 0.0) -> float:
    """Return a finite float, replacing NaN and infinities with safe values."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)

    if math.isnan(result):
        return float(default)
    if math.isinf(result):
        return 1e6 if result > 0 else -1e6
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
        self._no_improve_streak: dict[str, int] = {"destroy": 0, "repair": 0}
        self._fallback_cooldown: dict[str, int] = {"destroy": 0, "repair": 0}

        # Initialize operator maps
        self._init_operator_maps()
        self._ensure_parameterless_tuner_operators()

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
        cooldown = int(self._fallback_cooldown.get(op_type, 0))
        if cooldown <= 0:
            return False
        self._fallback_cooldown[op_type] = max(0, cooldown - 1)
        return True

    def update_op_feedback(self, op_type: str, improvement: float):
        improvement = _finite_float(improvement, 0.0)
        key = "destroy" if op_type == "destroy" else "repair"
        if improvement > 0:
            self._no_improve_streak[key] = 0
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
            return False

        required_samples = self._takeover_min_samples()
        if min_sample_size is not None:
            required_samples = max(required_samples, int(min_sample_size))

        if self._sample_count(op_type) < required_samples:
            return False

        progress = self._resolve_progress(state)
        takeover_progress = float(getattr(ALNSConfig, "ML_TAKEOVER_MIN_PROGRESS", 0.30))
        if progress < takeover_progress:
            return False

        model = self.destroy_model if op_type == "destroy" else self.repair_model
        scaler = self.destroy_scaler if op_type == "destroy" else self.repair_scaler
        return model is not None and scaler is not None

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
        """
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
        Encode an operator name as a one-hot vector.

        Unknown names return an all-zero vector instead of raising an exception
        that could interrupt the main workflow.
        """
        op_map = self.destroy_op_map if op_type == "destroy" else self.repair_op_map
        encoding = np.zeros(len(op_map), dtype=float)
        idx = op_map.get(op_name)
        if idx is not None:
            encoding[idx] = 1.0
        return encoding

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

        need_train = force or (
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
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    if len(X) < getattr(ALNSConfig, "ML_USE_RF_THRESHOLD", 50):
                        model = Ridge(
                            alpha=getattr(ALNSConfig, "ML_RIDGE_ALPHA", 1.0)
                        )
                    else:
                        model = RandomForestRegressor(
                            n_estimators=getattr(
                                ALNSConfig, "ML_RF_N_ESTIMATORS", 50
                            ),
                            max_depth=getattr(ALNSConfig, "ML_RF_MAX_DEPTH", 10),
                            random_state=int(self.rng.integers(0, 10_000)),
                        )
                    model.fit(X_scaled, y)
                    self.destroy_scaler = scaler
                    self.destroy_model = model
                    print(
                        f"[ML] Destroy-operator model trained with {len(X)} samples"
                    )
                    trained_destroy = True
                except Exception as exc:
                    self.destroy_model = None
                    self.destroy_scaler = None
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
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    if len(X) < getattr(ALNSConfig, "ML_USE_RF_THRESHOLD", 50):
                        model = Ridge(
                            alpha=getattr(ALNSConfig, "ML_RIDGE_ALPHA", 1.0)
                        )
                    else:
                        model = RandomForestRegressor(
                            n_estimators=getattr(
                                ALNSConfig, "ML_RF_N_ESTIMATORS", 50
                            ),
                            max_depth=getattr(ALNSConfig, "ML_RF_MAX_DEPTH", 10),
                            random_state=int(self.rng.integers(0, 10_000)),
                        )
                    model.fit(X_scaled, y)
                    self.repair_scaler = scaler
                    self.repair_model = model
                    print(
                        f"[ML] Repair-operator model trained with {len(X)} samples"
                    )
                    trained_repair = True
                except Exception as exc:
                    self.repair_model = None
                    self.repair_scaler = None
                    print(
                        f"[ML] Repair-operator model training failed: {exc}; "
                        "falling back to exploration"
                    )

        # Update the timestamp if at least one model was trained this round.
        # Otherwise keep the old value so the next call can try again.
        if trained_destroy or trained_repair:
            self.last_train_iter = current_iter

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
            return self._random_valid_operator(op_type, valid_candidates)

        model = self.destroy_model if op_type == "destroy" else self.repair_model
        scaler = self.destroy_scaler if op_type == "destroy" else self.repair_scaler

        # The model is not available yet: explore randomly.
        if model is None or scaler is None:
            return self._random_valid_operator(op_type, valid_candidates)

        predictions = []
        for op_name in valid_candidates:
            op_encoding = self.encode_operator(op_name, op_type)
            features = np.concatenate([state_features, op_encoding]).reshape(1, -1)
            try:
                features_scaled = scaler.transform(features)
                raw_prediction = float(model.predict(features_scaled)[0])
                pred = raw_prediction if math.isfinite(raw_prediction) else None
            except Exception as exc:
                # A failed prediction is marked explicitly; it must not be
                # allowed to become the selected operator merely because NaN
                # happens to sort in an unexpected position. Recorded because a
                # model that fails for every candidate would silently turn the
                # selector back into a random choice.
                record_swallowed(
                    "ml_operator_selector.predict_candidate", exc
                )
                pred = None
            params = self._safe_operator_params(op_name)
            predictions.append((op_name, pred, params))

        finite_predictions = [
            prediction for prediction in predictions if prediction[1] is not None
        ]
        if not finite_predictions:
            return self._random_valid_operator(op_type, valid_candidates)

        # Sort predicted performance in descending order
        finite_predictions.sort(
            key=lambda item: float(item[1]),
            reverse=True,
        )

        # Epsilon-greedy selection with stronger exploration early on and
        # convergence later.
        epsilon = self._dynamic_epsilon(state)
        if self.rng.random() < epsilon:
            idx = int(self.rng.integers(0, len(finite_predictions)))
            return finite_predictions[idx][0], finite_predictions[idx][2]
        return finite_predictions[0][0], finite_predictions[0][2]
