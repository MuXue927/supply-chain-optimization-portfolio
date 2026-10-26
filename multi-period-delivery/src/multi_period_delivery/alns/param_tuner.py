"""
Module: param_tuner

Responsibilities:
1. Define OperatorParams, the data structure for tunable operator parameters:
   - Store the current parameters, tuning ranges, performance history, and
     usage/success counts for each operator.
   - Expose average-performance and success-rate metrics for adaptive
     selection and weighting.
2. Define ParamAutoTuner, the automatic parameter tuner:
   - Split the search into exploration, exploitation, and refinement phases.
   - Use exploration_rate to switch between random exploration and exploitation
     of the current parameters.
   - Make directional parameter adjustments from recent operator improvements
     through adjust_operator_params.
   - Return the strongest candidates for a requested operator type through
     get_best_operators.
   - Dynamically adjust destroy strength through adaptive_degree.

Core design:
- Separate parameter sampling (get_operator_params) from learning feedback
  (update_operator_performance and adjust_operator_params), so external ALNS
  workflows can call them independently.
- Cap performance_history at 100 entries to bound memory and keep averages
  responsive to recent behavior.
- Phase control:
    progress = current_iter / max_iter
    - First 30%: high exploration, decreasing from 0.50 to 0.35.
    - Middle 40%: exploitation balancing exploration and use, from 0.35 to 0.15.
    - Final 30%: refinement from 0.15 to 0.0.
  Floating-point rounding can still produce a negative exploration rate near
  the end, so the value is explicitly clipped.

Inventory and vehicle context:
- This module depends on DataALNS only for problem-scale and data features,
  leaving an entry point for future feature-driven policies.

Potential improvements (not implemented):
- Add parameter-landscape memory by tracking statistics for different parameter
  combinations and using discrete sampling to improve late-stage convergence.
- Replace the truncated improvement window with an exponentially weighted
  moving average to emphasize inertia.
- Extend jointly tuned operators such as periodic_shaw_removal to a simplified
  covariance-adaptive update.
"""

# =========================
# Standard library
# =========================
import copy
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# =========================
# Third-party libraries
# =========================
import numpy as np
import numpy.random as rnd

from .alns_config import default_config as ALNSConfig
from .errors import STATE_OPERATION_ERRORS

# =========================
# Internal project dependencies
# =========================
from .InputDataALNS import DataALNS
from .progress import progress_from_elapsed


# ---------------------------------------------------------------------
# Data structure: parameter container for a single operator
# ---------------------------------------------------------------------
@dataclass
class OperatorParams:
    """
    Operator parameter description:
    name: unique operator name, such as 'random_removal'.
    params: current baseline parameters used directly during exploitation.
    ranges: valid continuous ranges used for random exploration and directional
        adjustment bounds.
    performance_history: recent improvement values, where
        improvement = obj_before - obj_after.
    usage_count / success_count: counters used to estimate the success rate
        and score operators in get_best_operators.
    """
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    performance_history: list[float] = field(default_factory=list)
    usage_count: int = 0
    success_count: int = 0

    def update_performance(self, improvement: float, success: bool = True):
        """
        Record the performance of one operator use.

        Args:
            improvement: Positive values indicate improvement in the minimized
                objective, typically scaled by the objective value.
            success: Whether this use counts as a success. External callers may
                set this to False for exceptions or infeasible outcomes.

        The history is capped at 100 values to keep recent behavior responsive.
        """
        self.performance_history.append(improvement)
        self.usage_count += 1
        if success:
            self.success_count += 1
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]

    def get_avg_performance(self, window: int = 20) -> float:
        """
        Return the mean improvement over the most recent window.

        If window exceeds the history length, all available values are used.
        A return value of 0 means there is no history or the mean is zero.
        """
        if not self.performance_history:
            return 0.0
        recent = self.performance_history[-min(window, len(self.performance_history)):]
        return sum(recent) / len(recent) if recent else 0.0

    def get_success_rate(self) -> float:
        """
        Return success_count / usage_count.

        max(1, usage_count) prevents division by zero.
        """
        return self.success_count / max(1, self.usage_count)

# ---------------------------------------------------------------------
# Automatic parameter tuner
# ---------------------------------------------------------------------
class ParamAutoTuner:
    """
    Overview:
    - Dynamic phase control across exploration, exploitation, and refinement.
    - exploration_rate determines whether operator parameters are randomly
      perturbed.
    - Operator performance is recorded and can be used for directional tuning
      through adjust_operator_params.
    - The tuner can query the top K operators whose names contain a requested
      type such as 'destroy' or 'repair'.

    Suggested workflow:
    1. Initialize with tuner = ParamAutoTuner(data, rng).
    2. At the start of each iteration, call
       tuner.set_iteration(cur_iter, max_iter).
    3. During operator selection, call
       params = tuner.get_operator_params(op_name), then execute the operator.
    4. After evaluation, call
       tuner.update_operator_performance(op_name, improvement, success) and
       tuner.adjust_operator_params(op_name, improvement, success).
    5. When needed, call
       best_ops = tuner.get_best_operators('removal', top_k=3).

    Design notes:
    - Random samples from exploration are not written back to op.params.
      op.params represents the current baseline and is changed only through
      directional updates in adjust_operator_params.
    - An empty ranges mapping means the operator does not support tuning and
      get_operator_params returns an empty dictionary.
    """

    operators: dict[str, OperatorParams]
    _prev_params: dict[str, dict[str, Any]]

    def __init__(self, data: DataALNS, rng: rnd.Generator):
        self.data = data
        self.rng = rng

        # Registered operator parameters
        self.operators: dict[str, OperatorParams] = {}

        # Instance features reserved for higher-level policies. The current
        # logic does not consume them directly.
        self.instance_features = self._extract_instance_features()
        self.instance_scale = self._estimate_instance_scale()
        self.small_instance = self.instance_scale <= int(
            getattr(ALNSConfig, "TUNER_SMALL_INSTANCE_SCALE_THRESHOLD", 250)
        )

        # Iteration and phase state
        self.iteration = 0
        self.max_iterations = 1000
        self.phase = "exploration"
        self.no_improve_iters = 0
        # Origin for the progress measure. The tuner is fed iteration counts
        # rather than a tracker, so it keeps a monotonic origin of its own; the
        # optimizer replaces the clock via ``set_elapsed_clock`` so every policy
        # measures from the run's single origin.
        self.start_time = time.perf_counter()
        self._elapsed_clock: Callable[[], float] | None = None

        # Control parameters, overridable through central configuration
        self.exploration_rate = getattr(ALNSConfig, "TUNER_EXPLORATION_INITIAL", 0.3)
        self.learning_rate = getattr(ALNSConfig, "TUNER_LEARNING_RATE", 0.1)

        # Historical parameter snapshots used to compute
        # direction = value - prev_value.
        self._prev_params: dict[str, dict[str, Any]] = {}
        #: How many times each operator's baseline has actually moved. Used by
        #: ``TUNER_MAX_ADJUSTMENTS`` to let a parameter settle instead of
        #: wandering for the whole run.
        self._adjust_counts: dict[str, int] = {}
        # Most recent execution-time parameter snapshot recorded by
        # get_operator_params, used to compute the used minus baseline direction.
        self._last_used: dict[str, dict[str, Any]] = {}

        # Initialize default operators
        self._init_default_operators()

    # -----------------------------------------------------------------
    # Instance feature extraction
    # -----------------------------------------------------------------
    def _canonical_name(self, name: str) -> str:
        """
        Normalize an operator name for wrapper-style names such as
        'random_removal_deg_0.25' -> 'random_removal'.

        This implementation only removes the '_deg_' suffix. Additional rules
        or mappings can be added when needed.
        """
        try:
            if isinstance(name, str) and "_deg_" in name:
                return name.split("_deg_")[0]
            return name
        except STATE_OPERATION_ERRORS:
            return name

    def _extract_instance_features(self) -> dict[str, float]:
        """
        Extract problem-scale and structural statistics for future ML or
        advanced adaptive policies.

        The current implementation computes only basic aggregates and does not
        participate in the core parameter-tuning process.
        """
        data = self.data
        features: dict[str, float] = {}

        # Scale features
        features["num_plants"] = len(data.plants)
        features["num_dealers"] = len(data.dealers)
        features["num_skus"] = len(data.all_skus)
        features["num_periods"] = data.horizons
        features["num_veh_types"] = len(data.all_veh_types)

        # Demand statistics
        if data.demands:
            demands = list(data.demands.values())
            features["avg_demand"] = sum(demands) / len(demands)
            features["max_demand"] = max(demands)
            features["min_demand"] = min(d for d in demands if d > 0)
            features["demand_std"] = np.std(demands) if len(demands) > 1 else 0.0
        else:
            features.update({
                "avg_demand": 0.0, "max_demand": 0.0,
                "min_demand": 0.0, "demand_std": 0.0
            })

        # Initial inventory
        if data.historical_s_ikt:
            init_inv = [inv for (_, _, day), inv in data.historical_s_ikt.items() if day == 0]
            features["avg_init_inv"] = sum(init_inv) / len(init_inv) if init_inv else 0.0
        else:
            features["avg_init_inv"] = 0.0

        # Production statistics
        if data.sku_prod_each_day:
            productions = list(data.sku_prod_each_day.values())
            features["avg_production"] = sum(productions) / len(productions) if productions else 0.0
        else:
            features["avg_production"] = 0.0

        # Vehicle capacity statistics
        if data.veh_type_cap:
            capacities = list(data.veh_type_cap.values())
            features["avg_veh_cap"] = sum(capacities) / len(capacities)
            features["max_veh_cap"] = max(capacities)
            features["min_veh_cap"] = min(capacities)
        else:
            features["avg_veh_cap"] = 0.0
            features["max_veh_cap"] = 0.0
            features["min_veh_cap"] = 0.0

        # Demand-to-initial-inventory ratio
        total_demand = sum(data.demands.values()) if data.demands else 0
        total_init_inv = sum(inv for (_, _, day), inv in data.historical_s_ikt.items()
                             if day == 0) if data.historical_s_ikt else 0
        features["demand_inv_ratio"] = total_demand / max(1, total_init_inv)

        return features

    def _estimate_instance_scale(self) -> int:
        """Estimate a coarse instance scale for conservative tuning on small datasets."""
        try:
            num_dealers = int(self.instance_features.get("num_dealers", 0))
            num_skus = int(self.instance_features.get("num_skus", 0))
            num_periods = int(self.instance_features.get("num_periods", 0))
            return max(1, num_dealers * max(1, num_skus) * max(1, num_periods))
        except (TypeError, ValueError):
            return 1

    # -----------------------------------------------------------------
    # Default operator registration
    # -----------------------------------------------------------------
    def _init_default_operators(self):
        """
        Register initial parameters and valid ranges for default destroy and
        repair operators.

        OPERATOR_DEFAULTS is loaded first from
        multi_period_delivery.alns.alns_config.default_config. If loading fails, the
        implementation falls back to built-in defaults for backward
        compatibility. An empty ranges mapping means the operator is not
        currently perturbed, while preserving a consistent interface.
        """
        try:
            # Import locally to avoid potential circular imports
            from .alns_config import default_config
            op_defaults = default_config.get_tuner_operator_defaults()
        except STATE_OPERATION_ERRORS:
            op_defaults = None

        if op_defaults:
            # op_defaults expected shape: {op_name: {"params": {...}, "ranges": {...}}, ...}
            for name, struct in op_defaults.items():
                try:
                    params = struct.get("params", {}) or {}
                    ranges = struct.get("ranges", {}) or {}
                    self.register_operator(name, params, ranges)
                except STATE_OPERATION_ERRORS:
                    # A failure for one operator does not affect the others
                    continue
            return

    # -----------------------------------------------------------------
    # Public interface: registration and updates
    # -----------------------------------------------------------------
    def register_operator(self, name: str,
                          params: dict[str, Any],
                          ranges: dict[str, tuple[float, float]]):
        """
        Register an operator.

        Args:
            params: Baseline parameters used directly during exploitation.
            ranges: Continuous or integer bounds used for random exploration
                and clipped during tuning.

        The params and ranges are copied before they are stored. Directional
        tuning mutates ``op.params`` in place, and the default operator table
        lives on the shared configuration object: without the copy, one run's
        tuning would leak into the next run in the same process.
        """
        # name = self._canonical_name(name)
        self.operators[name] = OperatorParams(
            name=name,
            params=copy.deepcopy(params),
            ranges=copy.deepcopy(ranges),
        )

    def update_operator_performance(self, name: str,
                                    improvement: float,
                                    success: bool = True):
        """
        Record performance after an operator executes and the objective is
        evaluated.

        Args:
            improvement: Positive values indicate objective improvement.
            success: Whether the use counts as successful. Use False for
                infeasible outcomes or exceptions.
        """
        # name = self._canonical_name(name)
        if name in self.operators:
            self.operators[name].update_performance(improvement, success)

    # -----------------------------------------------------------------
    # Iteration-phase control
    # -----------------------------------------------------------------
    def set_iteration(
        self,
        current: int,
        maximum: int,
        no_improve_iters: int | None = None,
    ):
        """
        Set iteration progress and refresh the phase and exploration rate.

        - exploration (0%-30%): 0.50 to 0.35
        - exploitation (30%-70%): 0.35 to 0.15
        - refinement (70%-100%): 0.15 to 0.00

        A piecewise linear schedule is used and can later be replaced by a
        nonlinear decay such as an exponential schedule.
        """
        self.iteration = current
        self.max_iterations = maximum
        if no_improve_iters is not None:
            try:
                self.no_improve_iters = max(0, int(no_improve_iters))
            except (TypeError, ValueError):
                self.no_improve_iters = 0

        progress = current / maximum if maximum > 0 else 1.0
        # Phase thresholds come from central configuration with backward
        # compatible defaults of (0.3, 0.7).
        thresh_low, thresh_high = getattr(ALNSConfig, "TUNER_PHASE_THRESHOLDS", (0.3, 0.7))
        if progress < thresh_low:
            self.phase = "exploration"
            # Preserve the original linear decay shape while allowing
            # configurable thresholds.
            self.exploration_rate = 0.5 - progress * 0.5
        elif progress < thresh_high:
            self.phase = "exploitation"
            self.exploration_rate = 0.35 - (progress - thresh_low) * 0.5
        else:
            self.phase = "refinement"
            self.exploration_rate = 0.15 - (progress - thresh_high) * 0.5

        # Numerical stability: prevent small negative values from floating-point
        # rounding.
        if self.exploration_rate < 0:
            self.exploration_rate = 0.0
        if self.exploration_rate > 1:
            self.exploration_rate = 1.0

        if self.small_instance:
            self.exploration_rate = min(
                self.exploration_rate,
                float(getattr(ALNSConfig, "TUNER_SMALL_INSTANCE_EXPLORATION_CAP", 0.18)),
            )

    # -----------------------------------------------------------------
    # Parameter retrieval: exploration versus exploitation
    # -----------------------------------------------------------------
    def get_operator_params(self, name: str) -> dict[str, Any]:
        """
        Return the parameters to use for an operator and record the execution
        snapshot in self._last_used[name].

        Random samples from exploration are not written back to the baseline.
        They are used only for the current execution and as a reference for
        directional tuning.
        """
        # name = self._canonical_name(name)
        ret: dict[str, Any] = {}
        if name not in self.operators:
            print(f"[WARNING] Unregistered operator: {name}; using default parameters")
            self._last_used[name] = {}
            return ret

        op = self.operators[name]

        if not op.params or not op.ranges:
            # Return an empty mapping when there are no tunable parameters or
            # ranges.
            self._last_used[name] = {}
            return ret

        # Exploration: random perturbation without writing back to op.params
        gate = int(getattr(ALNSConfig, "TUNER_EXPLORATION_GATE_USES", 12))
        exploring = self.rng.random() < self.exploration_rate
        if gate > 0 and op.usage_count >= gate:
            # Past the seeding phase a random sample is pure noise: it is never
            # written back, so it cannot teach anything, and because the
            # directional update reads ``used - baseline`` it also drags the
            # baseline in a random direction.
            exploring = False
        if exploring:
            sampled = op.params.copy()
            sample_scale = 1.0
            if self.small_instance:
                sample_scale = float(
                    getattr(ALNSConfig, "TUNER_SMALL_INSTANCE_SAMPLE_SCALE", 0.35)
                )
            for param_name, (min_val, max_val) in op.ranges.items():
                base_val = sampled.get(param_name)
                if isinstance(base_val, int):
                    min_int = int(min_val)
                    max_int = int(max_val)
                    center = int(base_val) if base_val is not None else (min_int + max_int) // 2
                    half_width = max(1, int(round((max_int - min_int) * sample_scale * 0.5)))
                    low = max(min_int, center - half_width)
                    high = min(max_int, center + half_width)
                    if high < low:
                        low, high = min_int, max_int
                    sampled[param_name] = int(self.rng.integers(low, high + 1))
                else:
                    try:
                        center = float(base_val) if base_val is not None else (float(min_val) + float(max_val)) / 2.0
                    except (TypeError, ValueError):
                        center = (float(min_val) + float(max_val)) / 2.0
                    half_width = max(1e-12, (float(max_val) - float(min_val)) * sample_scale * 0.5)
                    low = max(float(min_val), center - half_width)
                    high = min(float(max_val), center + half_width)
                    if high <= low:
                        low, high = float(min_val), float(max_val)
                    sampled[param_name] = float(self.rng.uniform(low, high))
            self._last_used[name] = sampled.copy()
            return sampled

        # Exploitation: use the current baseline
        ret = op.params.copy()
        self._last_used[name] = ret.copy()
        return ret

    # -----------------------------------------------------------------
    # Parameter tuning based on improvement direction
    # -----------------------------------------------------------------
    def adjust_operator_params(self, name: str, improvement: float, success: bool = True, skip_update: bool = False):
        """
        Directionally tune an operator's baseline parameters after a successful
        improvement. Performance recording can be skipped when requested.

        The adjustment direction is based on the difference between the
        parameters actually used in the current iteration
        (self._last_used[name]) and the current baseline in op.params.
        """
        # name = self._canonical_name(name)
        if name not in self.operators:
            return

        op = self.operators[name]
        if not skip_update:
            op.update_performance(improvement, success)

        if not success or improvement <= 0:
            return

        history_len = len(op.performance_history)
        min_history = int(getattr(ALNSConfig, "TUNER_MIN_HISTORY_FOR_ADJUST", 5))
        if history_len < min_history:
            return

        max_adjust = int(getattr(ALNSConfig, "TUNER_MAX_ADJUSTMENTS", 12))
        if max_adjust > 0:
            adjustments = int(getattr(self, "_adjust_counts", {}).get(name, 0))
            if adjustments >= max_adjust:
                # The baseline has had its chances; further moves would only
                # chase noise. Freezing is what lets the parameter settle.
                return

        recent_window = min(20, history_len)
        avg_perf = op.get_avg_performance(window=recent_window)
        if avg_perf <= 0 and history_len < min_history * 2:
            return

        confidence = min(1.0, history_len / max(1, min_history * 4))
        if self.small_instance:
            confidence *= 0.75

        required_gain = max(0.0, avg_perf)
        if required_gain > 0.0:
            if improvement <= required_gain * (1.0 + 0.15 * (1.0 - confidence)):
                return

        effective_lr = self.learning_rate * confidence
        max_rel_step = float(getattr(ALNSConfig, "TUNER_MAX_REL_STEP", 0.15))
        if self.small_instance:
            max_rel_step *= 0.75

        used_params = self._last_used.get(name, None)
        changed = False
        for param_name, value in op.params.items():
            if param_name not in op.ranges:
                continue
            min_val, max_val = op.ranges[param_name]

            used_val = (
                used_params.get(param_name, value)
                if isinstance(used_params, dict)
                else value
            )
            try:
                used_val_f = float(used_val)
                value_f = float(value)
            except (TypeError, ValueError):
                continue

            direction = used_val_f - value_f
            if abs(direction) <= 1e-12:
                continue

            denom = avg_perf if avg_perf > 1e-12 else 1e-12
            adjustment = effective_lr * (improvement / denom) * direction
            new_value = value_f + adjustment

            step_limit = max(
                1e-12,
                (float(max_val) - float(min_val)) * max_rel_step * max(0.25, confidence),
            )
            lower_bound = value_f - step_limit
            upper_bound = value_f + step_limit
            new_value = min(upper_bound, max(lower_bound, new_value))
            new_value = max(float(min_val), min(float(max_val), new_value))

            if isinstance(value, int):
                new_value = int(max(min_val, min(max_val, round(new_value))))
            else:
                new_value = float(new_value)

            if new_value == value:
                continue

            op.params[param_name] = new_value
            changed = True

            if getattr(ALNSConfig, "TUNER_DEBUG", False):
                print(
                    f"[TUNER] {name}.{param_name}: {value} -> {new_value} "
                    f"(used={used_val}, imp={improvement:.4f}, avg={avg_perf:.4f}, conf={confidence:.2f})"
                )

        if changed:
            self._adjust_counts[name] = int(self._adjust_counts.get(name, 0)) + 1

        # Save the current parameters for later reference
        if name not in self._prev_params:
            self._prev_params[name] = {}
        self._prev_params[name] = op.params.copy()

    # -----------------------------------------------------------------
    # Operator ranking: top K
    # -----------------------------------------------------------------
    def get_best_operators(self, operator_type: str, top_k: int = 3) -> list[str]:
        """
        Return the top-performing operators whose names contain operator_type,
        such as 'removal' or 'repair'.

        Score = max(0, avg_performance) * success_rate.

        A negative average performance indicates overall degradation and is
        treated as zero. This prevents sign interactions and avoids selecting
        weak operators merely because their negative values are close to zero.
        Only operators with usage_count > 0 are considered.
        """
        min_history = int(getattr(ALNSConfig, "TUNER_MIN_HISTORY_FOR_ADJUST", 5))
        candidates = []
        for name, op in self.operators.items():
            if operator_type in name and op.usage_count > 0:
                if op.usage_count < min_history:
                    continue
                recent_window = min(20, len(op.performance_history))
                recent = op.performance_history[-recent_window:]
                avg_perf = op.get_avg_performance(window=recent_window)
                success_rate = op.get_success_rate()
                variance_penalty = float(np.std(recent)) if len(recent) > 1 else 0.0
                confidence = min(1.0, op.usage_count / max(1, min_history * 4))
                history_bonus = np.log1p(op.usage_count)
                adj_avg = max(0.0, avg_perf)
                score = (
                    adj_avg
                    * (0.5 + 0.5 * success_rate)
                    * confidence
                    * history_bonus
                    / (1.0 + variance_penalty)
                )
                candidates.append((name, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in candidates[:top_k]]

    def set_elapsed_clock(self, clock: Callable[[], float] | None) -> None:
        """
        Adopt the run's stopwatch for progress measurement.

        The optimizer injects the same clock the stopping rule uses, so the
        decay schedule reads one origin rather than keeping a private timer.
        """
        self._elapsed_clock = clock

    def _elapsed_progress(self) -> float:
        """Fraction of the runtime budget consumed, clamped to [0, 1]."""
        if self._elapsed_clock is not None:
            return progress_from_elapsed(float(self._elapsed_clock()))
        return progress_from_elapsed(time.perf_counter() - self.start_time)

    # -----------------------------------------------------------------
    # Dynamic destroy-strength adjustment
    # -----------------------------------------------------------------
    def adaptive_degree(self,
                        base_degree: float,
                        min_degree: float = 0.05,
                        max_degree: float = 0.5,
                        decay_rate: float = 0.67) -> float:
        """
        Dynamically shrink the destroy ratio as the run progresses.

        - progress is normalized over the first 70% of the runtime budget and
          then held constant.
        - adaptive_value = base_degree * (1 - decay_rate * progress).
        - The result is clipped to [min_degree, max_degree].

        This controls perturbation strength during the destroy phase and avoids
        large structural disruptions late in the search.

        The schedule reads the shared progress measure (elapsed runtime against
        ``MAX_RUNTIME``) rather than iterations against the iteration ceiling, so
        it does not depend on the host's speed or on how the ceiling was sized.
        The tuner keeps its own ``start_time`` because it is fed iteration counts
        instead of a tracker; the measure itself is the same one.
        """
        if self.max_iterations <= 0 or self.iteration <= 0:
            return base_degree

        # Normalize over the first 70% of the budget, as before, then hold.
        progress = min(1.0, self._elapsed_progress() / 0.7)
        adaptive_value = base_degree * (1 - decay_rate * progress)

        stagnation_iters = max(0, int(getattr(self, "no_improve_iters", 0)))
        stagnation_threshold = int(
            getattr(ALNSConfig, "TUNER_STAGNATION_BOOST_THRESHOLD", 35)
        )
        if stagnation_iters >= stagnation_threshold:
            boost_factor = float(
                getattr(ALNSConfig, "TUNER_STAGNATION_BOOST_FACTOR", 0.35)
            )
            boost_scale = min(
                boost_factor,
                ((stagnation_iters - stagnation_threshold) / max(1, stagnation_threshold))
                * boost_factor,
            )
            adaptive_value *= 1.0 + boost_scale

        if self.small_instance:
            adaptive_value *= 1.0 + min(0.15, stagnation_iters / max(1, stagnation_threshold * 4) * 0.15)

        return max(min_degree, min(max_degree, adaptive_value))
