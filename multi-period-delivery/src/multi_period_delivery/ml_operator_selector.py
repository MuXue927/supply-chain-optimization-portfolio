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
    - Operator one-hot encoding depends on the initial destroy_op_map and
      repair_op_map. Unknown operators produce an all-zero vector instead of
      raising an exception.

Potential improvements (not implemented)
    - Use different modeling targets for destroy and repair, such as distinct
      penalties for negative inventory.
    - Use an online incremental model such as SGDRegressor to avoid full
      retraining cost.
    - Expose feature-importance diagnostics.
    - Dynamically reduce epsilon as the search progresses.

Feature-vector layout from extract_state_features
    [
      0: objective_value, or 1e6 when infinite,
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
from typing import Dict, List, Tuple, Any, Optional

# =========================
# Third-party libraries
# =========================
import numpy as np
import numpy.random as rnd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# =========================
# Internal project dependencies
# =========================
from typing import TYPE_CHECKING
from .param_tuner import ParamAutoTuner
from .alns_config import default_config as ALNSConfig

if TYPE_CHECKING:
    from .alnsopt import SolutionState


class MLOperatorSelector:
    """
    ML-enhanced operator selector for both destroy and repair operators.

    Attributes:
        param_tuner: ParamAutoTuner that supplies current parameters for each
            operator.
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

    def __init__(self, param_tuner: ParamAutoTuner, rng: rnd.Generator):
        # External dependencies
        self.param_tuner = param_tuner
        self.rng = rng

        # Sample buffers
        self.destroy_features: List[np.ndarray] = []
        self.destroy_performances: List[float] = []
        self.repair_features: List[np.ndarray] = []
        self.repair_performances: List[float] = []

        # Operator maps
        self.destroy_op_map: Dict[str, int] = {}
        self.repair_op_map: Dict[str, int] = {}

        # Models and scalers
        self.destroy_model: Optional[Any] = None
        self.destroy_scaler: Optional[StandardScaler] = None
        self.repair_model: Optional[Any] = None
        self.repair_scaler: Optional[StandardScaler] = None

        # Training controls, with defaults from centralized config
        self.min_samples: int = getattr(ALNSConfig, "ML_INITIAL_SAMPLE_SIZE", 20)
        self.retrain_interval: int = getattr(ALNSConfig, "ML_RETRAIN_INTERVAL", 50)
        self.last_train_iter: int = 0
        self._no_improve_streak: Dict[str, int] = {"destroy": 0, "repair": 0}
        self._fallback_cooldown: Dict[str, int] = {"destroy": 0, "repair": 0}

        # Initialize operator maps
        self._init_operator_maps()

    def set_iteration(
        self,
        current: int,
        maximum: int,
        no_improve_iters: int = 0,
    ):
        """Synchronize the selector with the current ALNS progress."""
        try:
            if self.param_tuner is not None:
                self.param_tuner.set_iteration(
                    int(current), int(maximum), no_improve_iters=int(no_improve_iters)
                )
        except Exception:
            pass

    def _resolve_progress(self, state: Optional["SolutionState"]) -> float:
        if state is not None:
            tracker = getattr(state, "tracker", None)
            if tracker is not None:
                try:
                    stats = tracker.get_statistics()
                    current_iter = int(stats.get("total_iterations", 0))
                    max_iter = int(stats.get("max_iterations", 0))
                    if max_iter > 0:
                        return max(0.0, min(1.0, current_iter / max_iter))
                except Exception:
                    pass
        if self.param_tuner is not None:
            try:
                max_iter = int(getattr(self.param_tuner, "max_iterations", 0))
                cur_iter = int(getattr(self.param_tuner, "iteration", 0))
                if max_iter > 0:
                    return max(0.0, min(1.0, cur_iter / max_iter))
            except Exception:
                pass
        return 0.0

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
        state: Optional["SolutionState"],
        op_type: str,
        min_sample_size: Optional[int] = None,
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

    def _dynamic_epsilon(self, state: Optional["SolutionState"]) -> float:
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
        destroy_ops = [
            "random_removal",
            "shaw_removal",
            "periodic_shaw_removal",
            "path_removal",
            "cross_period_dealer_removal",
            "period_range_removal",
            "worst_removal",
            "infeasible_removal",
            "surplus_inventory_removal",
            "inventory_level_destruction",
        ]

        repair_ops = [
            "greedy_repair",
            "local_search_repair",
            "inventory_balance_repair",
            "smart_batch_repair",
            "infeasible_repair",
            "regret_based_repair",
            "sequential_repair",
            "lookahead_repair",
            "fleet_consolidation_repair",
        ]

        for i, op in enumerate(destroy_ops):
            self.destroy_op_map[op] = i
        for i, op in enumerate(repair_ops):
            self.repair_op_map[op] = i

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
        features: List[float] = []

        # 1. Objective value. Use a large finite value for infinities to
        # maintain numerical stability.
        obj_value = state.objective()
        features.append(obj_value if not math.isinf(obj_value) else 1e6)

        # 2. Vehicle metrics
        num_vehicles = len(state.vehicles)
        features.append(num_vehicles)

        if num_vehicles > 0:
            # Average utilization
            total_util = 0.0
            veh_type_counts: Dict[str, int] = {}
            for veh in state.vehicles:
                load = sum(
                    qty * data.sku_sizes[sku] for (sku, _), qty in veh.cargo.items()
                )
                cap = data.veh_type_cap[veh.type]
                util = load / cap if cap > 0 else 0.0
                total_util += util
                veh_type_counts[veh.type] = veh_type_counts.get(veh.type, 0) + 1
            avg_util = total_util / num_vehicles
            features.append(avg_util)

            # Vehicle-type distribution as normalized frequencies. Sort by
            # numeric value when possible and fall back to string ordering.
            def _veh_type_sort_key(v):
                try:
                    return (0, float(v))
                except Exception:
                    return (1, str(v))

            for veh_type in sorted(data.all_veh_types, key=_veh_type_sort_key):
                features.append(veh_type_counts.get(veh_type, 0) / num_vehicles)
        else:
            features.append(0.0)  # Average utilization
            features.extend([0.0] * len(data.all_veh_types))

        # 3. Inventory metrics. objective() has already ensured that
        # compute_inventory is called.
        if state.s_ikt:
            inv_values = list(state.s_ikt.values())
            avg_inv = sum(inv_values) / len(inv_values)
            features.append(avg_inv)

            neg_inv_ratio = sum(1 for v in inv_values if v < 0) / len(inv_values)
            features.append(neg_inv_ratio)

            # Compute the mean inventory-to-capacity ratio over plant-day pairs
            plant_day_pos_inv: Dict[Tuple[str, int], float] = {}
            for (plant, _sku, day), inv in state.s_ikt.items():
                if inv > 0:
                    plant_day_pos_inv[(plant, day)] = (
                        plant_day_pos_inv.get((plant, day), 0) + inv
                    )
            usage_acc = 0.0
            usage_cnt = 0
            for (plant, day), inv_sum in plant_day_pos_inv.items():
                cap = data.plant_inv_limit[plant]
                if cap > 0:
                    usage_acc += inv_sum / cap
                    usage_cnt += 1
            avg_usage_ratio = usage_acc / usage_cnt if usage_cnt > 0 else 0.0
            features.append(avg_usage_ratio)
        else:
            features.extend([0.0, 0.0, 0.0])

        # 4. Demand fulfillment
        shipped = state.compute_shipped()
        total_demand = sum(data.demands.values()) if data.demands else 0
        total_shipped = sum(shipped.values()) if shipped else 0
        satisfaction_ratio = total_shipped / total_demand if total_demand > 0 else 1.0
        features.append(satisfaction_ratio)

        unmet_count = sum(
            1
            for (dealer, sku), demand in data.demands.items()
            if shipped.get((dealer, sku), 0) < demand
        )
        unmet_ratio = unmet_count / len(data.demands) if data.demands else 0.0
        features.append(unmet_ratio)

        # 5. Search progress, optional and tracker-dependent
        if state.tracker:
            stats = state.tracker.get_statistics()
            current_iter = stats.get("total_iterations", 0)
            max_iter = stats.get("max_iterations", 1000)
            progress = current_iter / max_iter if max_iter > 0 else 0.0
            recent_improvement = stats.get("recent_improvement", 0.0)
            features.append(progress)
            features.append(recent_improvement)
        else:
            features.extend([0.0, 0.0])

        return np.array(features, dtype=float)

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
        metric.
        """
        state_features = self.extract_state_features(state)
        op_encoding = self.encode_operator(op_name, op_type)
        features = np.concatenate([state_features, op_encoding])

        if op_type == "destroy":
            self.destroy_features.append(features)
            self.destroy_performances.append(improvement)
        else:
            self.repair_features.append(features)
            self.repair_performances.append(improvement)

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
        current_iter = 0
        if self.param_tuner:
            current_iter = self.param_tuner.iteration

        need_train = force or (
            current_iter - self.last_train_iter >= self.retrain_interval
        )
        if not need_train:
            return

        trained_destroy = False
        trained_repair = False
        # ---------- Train destroy-operator model ----------
        if len(self.destroy_features) >= self.min_samples:
            X = np.array(self.destroy_features, dtype=float)
            y = np.array(self.destroy_performances, dtype=float)

            self.destroy_scaler = StandardScaler()
            X_scaled = self.destroy_scaler.fit_transform(X)

            if len(X) < getattr(ALNSConfig, "ML_USE_RF_THRESHOLD", 50):
                self.destroy_model = Ridge(
                    alpha=getattr(ALNSConfig, "ML_RIDGE_ALPHA", 1.0)
                )
            else:
                self.destroy_model = RandomForestRegressor(
                    n_estimators=getattr(ALNSConfig, "ML_RF_N_ESTIMATORS", 50),
                    max_depth=getattr(ALNSConfig, "ML_RF_MAX_DEPTH", 10),
                    random_state=int(self.rng.integers(0, 10_000)),
                )
            self.destroy_model.fit(X_scaled, y)
            print(f"[ML] Destroy-operator model trained with {len(X)} samples")
            trained_destroy = True

        # ---------- Train repair-operator model ----------
        if len(self.repair_features) >= self.min_samples:
            X = np.array(self.repair_features, dtype=float)
            y = np.array(self.repair_performances, dtype=float)

            self.repair_scaler = StandardScaler()
            X_scaled = self.repair_scaler.fit_transform(X)

            if len(X) < getattr(ALNSConfig, "ML_USE_RF_THRESHOLD", 50):
                self.repair_model = Ridge(
                    alpha=getattr(ALNSConfig, "ML_RIDGE_ALPHA", 1.0)
                )
            else:
                self.repair_model = RandomForestRegressor(
                    n_estimators=getattr(ALNSConfig, "ML_RF_N_ESTIMATORS", 50),
                    max_depth=getattr(ALNSConfig, "ML_RF_MAX_DEPTH", 10),
                    random_state=int(self.rng.integers(0, 10_000)),
                )
            self.repair_model.fit(X_scaled, y)
            print(f"[ML] Repair-operator model trained with {len(X)} samples")
            trained_repair = True

        # Update the timestamp if at least one model was trained this round.
        # Otherwise keep the old value so the next call can try again.
        if trained_destroy or trained_repair:
            self.last_train_iter = current_iter

    # ------------------------------------------------------------------
    # Operator selection
    # ------------------------------------------------------------------
    def select_best_operator(
        self,
        state: SolutionState,
        op_type: str,
        candidates: List[str],
    ) -> Tuple[str, Dict[str, Any]]:
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
        if not candidates:
            if op_type == "destroy":
                # fallback to centralized destroy defaults
                destroy_defaults = ALNSConfig.get_destroy_params()
                return "random_removal", {
                    "degree": destroy_defaults.get("random_removal_degree", 0.25)
                }
            # fallback to centralized repair defaults
            repair_defaults = ALNSConfig.get_repair_params()
            return "greedy_repair", repair_defaults.get("greedy_repair", {})

        state_features = self.extract_state_features(state)

        model = self.destroy_model if op_type == "destroy" else self.repair_model
        scaler = self.destroy_scaler if op_type == "destroy" else self.repair_scaler

        # The model is not available yet: explore randomly.
        if model is None or scaler is None:
            selected = self.rng.choice(candidates)
            params = self.param_tuner.get_operator_params(selected)
            return selected, params

        predictions = []
        for op_name in candidates:
            op_encoding = self.encode_operator(op_name, op_type)
            features = np.concatenate([state_features, op_encoding]).reshape(1, -1)
            try:
                features_scaled = scaler.transform(features)
                pred = model.predict(features_scaled)[0]
            except Exception:
                # Any unexpected transform failure degrades to a zero
                # prediction to keep the selector robust.
                pred = 0.0
            params = self.param_tuner.get_operator_params(op_name)
            predictions.append((op_name, pred, params))

        # Sort predicted performance in descending order
        predictions.sort(key=lambda x: x[1], reverse=True)

        # Epsilon-greedy selection with stronger exploration early on and
        # convergence later.
        epsilon = self._dynamic_epsilon(state)
        if self.rng.random() < epsilon:
            idx = int(self.rng.integers(0, len(predictions)))
            return predictions[idx][0], predictions[idx][2]
        return predictions[0][0], predictions[0][2]
