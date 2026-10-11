"""
Module: alnstrack

Purpose:
    Provide unified tracking and statistics support for the ALNS search:
      1. Iteration-level records: current iteration, current objective,
         historical best objective, gap series, and elapsed time.
      2. Best-solution persistence: automatically save improved solutions found
         during the search as copied states.
      3. Metric export: optionally write CSV records for later analysis,
         visualization, or experiment comparison.
      4. ML support: store feature and label samples, trained models, and the
         scaler cache.
      5. External integration: ParamAutoTuner reads iteration statistics such as
         total_iterations and max_iterations for phase adaptation.

Design notes:
    - Lightweight: the tracker does not participate in search decisions and
      consumes only the interface exposed by SolutionState.
    - Decoupled: ML and parameter-tuning modules obtain information through
      get_statistics() and get_ml_cache_info().
    - Fault tolerant: CSV write failures do not interrupt the main search and
      are handled by try/except blocks.
    - History trimming: feature and label samples are capped to prevent
      unbounded memory growth in long runs.
    - Early stopping: no_improve_iters tracks iterations since the last best
      solution, and early_stop_patience defines the tolerance threshold. The
      main loop can call tracker.should_stop() to terminate early.

Gap definition:
    gap = (current_obj - best_obj) / current_obj * 100%.

    Notes:
      - The objective is minimized. When a new best solution appears,
        current_obj == best_obj and gap becomes 0.
      - The implementation assumes best_obj <= current_obj. If numerical noise
        makes best_obj > current_obj, the gap remains positive. Strict
        monotonicity would require an additional min/epsilon guard before the
        update.

Reference workflow:
    tracker = ALNSTracker(output_file='logs/alns_progress.csv')
    for iter in range(MAX):
        ... # Generate or accept a new solution.
        tracker.on_iteration(state, rng)
    stats = tracker.get_statistics()

Possible future improvements:
    - Add sliding-window or exponentially weighted gap smoothing.
    - Add JSON-lines logs for richer post-processing.
"""

# =========================
# Standard library
# =========================
import os
import csv
import time
import copy
from typing import Optional, Any, Tuple, TYPE_CHECKING

# =========================
# Third-party libraries
# =========================
import numpy as np
import numpy.random as rnd

# =========================
# Internal project dependencies
# =========================
from .optutility import LogPrinter
from .alns_config import default_config as ALNSConfig

if TYPE_CHECKING:  # Import only for type checking to avoid runtime cycles.
    from .alnsopt import SolutionState

# Default log printer used as a fallback and not shared across runs.
DEFAULT_LOG_PRINTER = LogPrinter(time.time())


class ALNSTracker:
    """
    ALNS process tracker.

    Records the key statistics of an optimization run to support:
      - result analysis through objective and gap curves,
      - phase decisions coordinated with ParamAutoTuner,
      - ML operator feature/label accumulation and model caching.
    """

    def __init__(
        self,
        output_file: Optional[str] = None,
        log_printer: Optional[LogPrinter] = None,
    ):
        # Iteration and objective state
        self.iteration: int = 0
        self.max_iterations: int = getattr(ALNSConfig, "MAX_ITERATIONS", 1000)
        self.current_obj: float = float("inf")
        self.best_obj: float = float("inf")
        self.best_solution: Optional["SolutionState"] = None
        self.best_actual_obj: float = float("inf")
        self.best_actual_solution: Optional["SolutionState"] = None
        self.initial_search_obj: float = float("inf")
        self.initial_actual_obj: float = float("inf")

        # Time and historical series
        self.start_time: float = time.time()
        self.log_printer: LogPrinter = log_printer or LogPrinter(self.start_time)
        self.objectives: list[float] = []
        self.gaps: list[float] = []

        # Optional result persistence
        self.output_file: Optional[str] = output_file
        if self.output_file:
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            with open(self.output_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["Iteration", "Current_Obj", "Best_Obj", "GapVsInitial"]
                )

        # ML state for learning-based repair and future learning mechanisms
        self.features: list[
            list[float]
        ] = []  # Each feature can be [demand, size, day, inventory, utilization, ...].
        self.labels: list[float] = []  # Corresponding improvement values.
        self._cached_model: Optional[Any] = None
        self._cached_scaler: Optional[Any] = None
        self._last_train_iteration: Optional[int] = None

        # Early-stopping state
        self.early_stop_patience: int = 500  # Tunable by problem scale/runtime.
        self.no_improve_iters: int = 0  # Iterations since the last best update.
        self.best_update_iteration: int = (
            0  # Iteration of the latest best update, used for improvement analysis.
        )
        self.recent_improvement: float = (
            0.0  # Latest improvement, prev_best - new_best; 0 when unchanged.
        )
        # Elite pool: store diverse, high-quality feasible solutions for
        # stagnation relinking or restart strategies.
        self.elite_pool: list[dict[str, Any]] = []
        self.elite_pool_max_size: int = int(
            getattr(ALNSConfig, "ELITE_POOL_MAX_SIZE", 8)
        )

    def get_iteration(self) -> int:
        return int(self.iteration)

    def get_max_iterations(self) -> int:
        return int(self.max_iterations)

    def set_max_iterations(self, max_iterations: int):
        self.max_iterations = int(max_iterations)

    def set_initial_reference(
        self,
        initial_search_obj: Optional[float] = None,
        initial_actual_obj: Optional[float] = None,
    ):
        if initial_search_obj is not None:
            self.initial_search_obj = float(initial_search_obj)
        if initial_actual_obj is not None:
            self.initial_actual_obj = float(initial_actual_obj)

    def _signed_gap_from_initial(self, value: float, baseline: float) -> float:
        if (
            not np.isfinite(value)
            or not np.isfinite(baseline)
            or abs(baseline) <= 1e-12
        ):
            return 0.0
        return float((value - baseline) / abs(baseline) * 100.0)

    def _profile_signature(self, state: "SolutionState") -> tuple:
        try:
            vehicles = (
                list(state.iter_used_vehicles())
                if hasattr(state, "iter_used_vehicles")
                else list(getattr(state, "vehicles", []) or [])
            )
        except Exception:
            vehicles = list(getattr(state, "vehicles", []) or [])

        veh_count = int(len(vehicles))
        load_ratio = 0.0
        large_ratio = 0.0
        fixed_cost = 0.0
        try:
            total_load = 0.0
            total_cap = 0.0
            threshold = 0.0
            caps = []
            for vt in list(getattr(state.data, "all_veh_types", []) or []):
                try:
                    caps.append(float(state.data.veh_type_cap[vt]))
                except Exception:
                    continue
            if caps:
                caps.sort()
                idx = int(round((len(caps) - 1) * 0.67))
                threshold = float(caps[idx])
            for veh in vehicles:
                try:
                    cap = float(
                        state.data.veh_type_cap.get(
                            veh.type, getattr(veh, "max_capacity", 0.0) or 0.0
                        )
                    )
                    total_cap += cap
                    total_load += float(state.compute_veh_load(veh))
                    fixed_cost += float(state.data.veh_type_cost.get(veh.type, 0.0))
                    if cap >= threshold:
                        large_ratio += 1.0
                except Exception:
                    continue
            if veh_count > 0 and total_cap > 1e-12:
                load_ratio = total_load / total_cap
                large_ratio = large_ratio / float(veh_count)
        except Exception:
            pass
        bin_size = max(1, int(getattr(ALNSConfig, "ELITE_SIGNATURE_BIN_SIZE", 10)))
        return (
            veh_count,
            int(round(load_ratio * bin_size)),
            int(round(large_ratio * bin_size)),
            int(round(fixed_cost / 10.0)),
        )

    def record_elite_candidate(self, state: "SolutionState") -> bool:
        """Record a feasible elite candidate into the bounded elite pool."""
        if state is None:
            return False
        try:
            feasible, _ = state.validate()
        except Exception:
            feasible = False
        if not feasible:
            return False

        try:
            obj = float(state.objective())
        except Exception:
            return False
        if not np.isfinite(obj):
            return False

        try:
            profile = {
                "vehicles": float(
                    len(list(state.iter_used_vehicles()))
                    if hasattr(state, "iter_used_vehicles")
                    else len(getattr(state, "vehicles", []) or [])
                ),
                "load_ratio": 0.0,
                "large_ratio": 0.0,
                "fixed_cost": float(state.calculate_cost()),
            }
            sig = self._profile_signature(state)
        except Exception:
            profile = {
                "vehicles": float(len(getattr(state, "vehicles", []) or [])),
                "load_ratio": 0.0,
                "large_ratio": 0.0,
                "fixed_cost": 0.0,
            }
            sig = (int(profile["vehicles"]), 0, 0, 0)

        pool = list(getattr(self, "elite_pool", []) or [])
        if pool:
            existing_best = min(float(item.get("obj", float("inf"))) for item in pool)
        else:
            existing_best = float("inf")

        # Fast de-duplication by signature; keep the better objective for the same signature.
        updated = False
        for item in pool:
            if tuple(item.get("signature", ())) == sig:
                if obj < float(item.get("obj", float("inf"))):
                    item["state"] = state.copy()
                    item["obj"] = obj
                    item["profile"] = profile
                    item["signature"] = sig
                updated = True
                break
        if not updated:
            pool.append(
                {
                    "state": state.copy(),
                    "obj": obj,
                    "profile": profile,
                    "signature": sig,
                }
            )

        # Keep only a small, quality-biased diverse pool.
        pool.sort(
            key=lambda x: (
                float(x.get("obj", float("inf"))),
                tuple(x.get("signature", ())),
            )
        )
        max_size = max(1, int(getattr(self, "elite_pool_max_size", 8)))
        trimmed = []
        seen_sig = set()
        for item in pool:
            sig_i = tuple(item.get("signature", ()))
            if sig_i in seen_sig:
                continue
            seen_sig.add(sig_i)
            trimmed.append(item)
            if len(trimmed) >= max_size:
                break
        self.elite_pool = trimmed
        return obj <= existing_best + 1e-9

    # -----------------------------------------------------------------
    # Main iteration callback
    # -----------------------------------------------------------------
    def on_iteration(self, state: "SolutionState", rng: rnd.Generator, **kwargs):
        """
        Call after each ALNS main-loop iteration.

        Steps:
          1. Increment the iteration counter.
          2. Read the current objective, using the objective cache when valid.
          3. Update best_solution when the state is feasible and better.
          4. Compute and record the gap.
          5. Optionally write a CSV row. Frequent I/O can be sampled if needed.
          6. Print a periodic overview, currently every 100 iterations.

        Args:
            state: Current solution, which must provide objective() and validate().
            rng: Random-number generator retained for interface compatibility;
                it is not used currently.
            kwargs: Reserved for future extensions such as extra logs or tags.
        """
        self.iteration += 1

        # Synchronize ParamAutoTuner at the start of each iteration so
        # get_operator_params can read the latest exploration_rate.
        try:
            if (
                hasattr(state, "param_tuner")
                and state.param_tuner
                and hasattr(self, "max_iterations")
            ):
                current_iter = self.iteration
                max_iter = self.max_iterations
                state.param_tuner.set_iteration(
                    current_iter,
                    max_iter,
                    no_improve_iters=self.no_improve_iters,
                )
        except Exception as e:
            # Protective fallback: log a warning without interrupting the main
            # workflow if the update fails.
            self.log_printer.print(
                f"[WARN] Failed to update param_tuner iteration: {e}", color="yellow"
            )

        # Current objective, O(1) when the objective cache is valid
        current_obj = float(state.objective())
        self.current_obj = current_obj
        self.objectives.append(current_obj)
        current_actual_obj = float(state.calculate_cost())

        # Validate feasibility and update the best solution
        is_feasible, _ = state.validate()
        improved = False
        if current_obj < self.best_obj and is_feasible:
            prev_best = self.best_obj
            self.best_obj = current_obj
            # Save a deep copy of the best state so later in-place mutations do
            # not affect it.
            try:
                # Prefer SolutionState.copy() for a lightweight snapshot
                self.best_solution = state.copy()
            except Exception:
                # Backward compatibility: fall back to deepcopy.
                self.best_solution = copy.deepcopy(state)
            improved = True
            # Record the actual improvement, treating a previous inf as 0.
            if prev_best != float("inf"):
                self.recent_improvement = max(0.0, prev_best - current_obj)
            else:
                self.recent_improvement = 0.0

            search_improvement_vs_prev = float("nan")
            if np.isfinite(prev_best) and abs(prev_best) > 1e-12:
                search_improvement_vs_prev = float(
                    (prev_best - current_obj) / abs(prev_best) * 100.0
                )
            elapsed_time = time.time() - self.start_time
            actual_gap_from_initial = self._signed_gap_from_initial(
                current_actual_obj, self.initial_actual_obj
            )
            self.log_printer.print(
                f"Iteration {self.iteration}: New best feasible solution found!",
                color="bold green",
            )
            if np.isfinite(search_improvement_vs_prev):
                self.log_printer.print(
                    f"Search Objective: {self.best_obj:.6f}, Improvement vs Prev Best: +{search_improvement_vs_prev:.2f}%, Time: {elapsed_time:.2f}s",
                    color="bold green",
                )
            else:
                self.log_printer.print(
                    f"Search Objective: {self.best_obj:.6f}, Improvement vs Prev Best: N/A (first best), Time: {elapsed_time:.2f}s",
                    color="bold green",
                )
            try:
                self.log_printer.print(
                    f"Actual Objective (Gurobi-aligned): {current_actual_obj:.2f}, Feasible: Yes, Gap vs Initial: {actual_gap_from_initial:+.2f}%",
                    color="bold cyan",
                )
            except Exception:
                pass

        # Maintain a separate best feasible solution under the actual
        # objective for final reporting.
        if is_feasible and current_actual_obj < self.best_actual_obj:
            self.best_actual_obj = current_actual_obj
            try:
                self.best_actual_solution = state.copy()
            except Exception:
                self.best_actual_solution = copy.deepcopy(state)

        # Maintain a small elite pool for path relinking and stagnation escape.
        try:
            elite_sample_interval = max(
                1, int(getattr(ALNSConfig, "ELITE_POOL_SAMPLE_INTERVAL", 50))
            )
            if is_feasible and (
                improved or self.iteration % elite_sample_interval == 0
            ):
                self.record_elite_candidate(state)
        except Exception:
            pass

        # Maintain the no-improvement count for early stopping
        if improved:
            self.no_improve_iters = 0
            self.best_update_iteration = self.iteration
        else:
            self.no_improve_iters += 1
            # Set recent_improvement to 0 when the most recent iteration did not
            # improve the best solution.
            self.recent_improvement = 0.0

        # Use the signed gap relative to the initial solution during the run
        gap_vs_initial = self._signed_gap_from_initial(
            current_actual_obj, self.initial_actual_obj
        )
        self.gaps.append(gap_vs_initial)

        # Periodic overview logging
        if self.iteration % 100 == 0:
            self.log_printer.print(
                f"Iteration: {self.iteration}\tSearch Obj: {self.current_obj:.6f}\t"
                f"Actual Obj: {current_actual_obj:.2f}\tFeasible: {'Yes' if is_feasible else 'No'}\t"
                f"Gap vs Initial: {gap_vs_initial:+.2f}%\tBest Search Obj: {self.best_obj:.6f}"
            )

        # Optional persistence
        if self.output_file:
            try:
                with open(self.output_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            self.iteration,
                            self.current_obj,
                            self.best_obj,
                            gap_vs_initial,
                        ]
                    )
            except Exception as e:
                self.log_printer.print(
                    f"[WARN] Failed to append tracker CSV: {e}", color="yellow"
                )

    # -----------------------------------------------------------------
    # ML data collection
    # -----------------------------------------------------------------
    def update_ml_data(self, feature: list, label: float) -> Tuple[int, int]:
        """
        Store one (feature, label) sample.

        feature: Free-form structure. Its dimensionality is maintained by the
            consumer.
        label: Usually the objective improvement; positive values are better.

        Returns:
            Current feature-sample count and current label-sample count.

        A hard cap of 1000 recent samples prevents unbounded memory growth in
        long runs. This can later be extended to time-based or multi-level
        buffering.
        """
        self.features.append(feature)
        self.labels.append(label)
        if len(self.features) > 1000:  # Sliding-window mechanism
            self.features = self.features[-1000:]
            self.labels = self.labels[-1000:]
        return len(self.features), len(self.labels)

    # -----------------------------------------------------------------
    # ML model cache interface
    # -----------------------------------------------------------------
    def cache_ml_model(self, model: Any, scaler: Any, iteration: int):
        """
        Cache a trained ML model and its scaler, recording the training
        iteration.

        model: Trained model such as RandomForest, Ridge, or a custom model.
        scaler: Feature preprocessor corresponding to model.
        iteration: Iteration number at training time.

        This allows later iterations to reuse the model without frequent
        retraining. External callers can inspect _last_train_iteration to decide
        whether incremental updates are needed.
        """
        self._cached_model = model
        self._cached_scaler = scaler
        self._last_train_iteration = iteration
        print(f"[Tracker] Cached ML model at iteration {iteration}")

    def get_cached_model(self) -> Tuple[Optional[Any], Optional[Any], Optional[int]]:
        """
        Return the (model, scaler, last_train_iteration) cache tuple.

        If any element is None, the cache has not been populated or has been
        cleared.
        """
        return self._cached_model, self._cached_scaler, self._last_train_iteration

    def has_cached_model(self) -> bool:
        """
        Return whether a valid model cache exists.

        All three values, model, scaler, and last_train_iteration, must be
        non-None.
        """
        return (
            self._cached_model is not None
            and self._cached_scaler is not None
            and self._last_train_iteration is not None
        )

    def clear_ml_cache(self):
        """
        Clear the model cache to release memory or trigger retraining.
        """
        self._cached_model = None
        self._cached_scaler = None
        self._last_train_iteration = None
        print("[Tracker] Cleared ML model cache")

    def get_ml_cache_info(self) -> dict:
        """
        Return a summary of the ML cache state.

        has_model: Whether a cache exists.
        last_train_iteration: Iteration of the last training.
        model_type / scaler_type: Class names for debugging.
        data_samples: Current number of feature samples.
        """
        return {
            "has_model": self.has_cached_model(),
            "last_train_iteration": self._last_train_iteration,
            "model_type": type(self._cached_model).__name__
            if self._cached_model
            else None,
            "scaler_type": type(self._cached_scaler).__name__
            if self._cached_scaler
            else None,
            "data_samples": len(self.features),
        }

    # -----------------------------------------------------------------
    # Early-stopping decision
    # -----------------------------------------------------------------
    def should_stop(self) -> bool:
        """
        Return True when iterations since the last best solution reach
        early_stop_patience.

        Usage:
            while not tracker.should_stop():
                ...
        """
        return self.no_improve_iters >= self.early_stop_patience

    # -----------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------
    def get_statistics(self) -> dict:
        """
        Summarize all key tracker information. Values are copied or lightly
        copied to avoid accidental in-place external mutation.

        total_iterations: Completed iteration count.
        max_iterations: Planned maximum iteration count.
        best_objective: Historical best objective.
        current_objective: Current solution objective.
        final_gap: Current gap, or 0 when no history exists.
        objectives_history: Objective trajectory copy.
        gaps_history: Gap trajectory copy.
        elapsed_time: Elapsed seconds.
        best_solution: Current best solution object reference. Deep usage
            should call deepcopy.
        features / labels: Copies of ML samples.
        ml_cache_info: Model-cache summary.
        no_improve_iters: Iterations since the last best improvement.
        early_stop_patience: Early-stop tolerance threshold.
        best_update_iteration: Iteration of the latest best update.
        recent_improvement: Latest improvement, prev_best - new_best; 0 when
            unchanged.
        """
        return {
            "total_iterations": self.iteration,
            "max_iterations": self.max_iterations,
            "best_objective": self.best_obj,
            "best_actual_objective": self.best_actual_obj,
            "current_objective": self.current_obj,
            "final_gap": self._signed_gap_from_initial(
                self.current_obj, self.initial_search_obj
            )
            if self.objectives
            else 0.0,
            "final_actual_gap": self._signed_gap_from_initial(
                self.best_actual_solution.calculate_cost()
                if self.best_actual_solution is not None
                else (
                    self.best_solution.calculate_cost()
                    if self.best_solution is not None
                    else self.current_obj
                ),
                self.initial_actual_obj,
            )
            if self.objectives
            else 0.0,
            "objectives_history": self.objectives.copy(),
            "gaps_history": self.gaps.copy(),
            "elapsed_time": time.time() - self.start_time,
            "best_solution": self.best_solution,
            "best_actual_solution": self.best_actual_solution,
            "features": self.features.copy(),
            "labels": self.labels.copy(),
            "ml_cache_info": self.get_ml_cache_info(),
            "no_improve_iters": self.no_improve_iters,
            "early_stop_patience": self.early_stop_patience,
            "best_update_iteration": self.best_update_iteration,
            "recent_improvement": self.recent_improvement,
        }


# ---------------------------------------------------------------------
# Gap calculation functions kept independent of ALNSTracker for reuse/testing
# ---------------------------------------------------------------------
def _is_all_zero(x) -> bool:
    ax = np.asarray(x)
    if ax.size == 1:
        # Single-element array or scalar: use isclose to test zero
        return bool(np.isclose(float(ax), 0.0))
    # Multi-element array: treat as zero when all values are close to zero
    return bool(np.allclose(ax, 0.0))


def _to_scalar_or_mean(x) -> float:
    ax = np.asarray(x)
    if ax.size == 1:
        return float(ax)
    # Reduce vector results to a mean for comparable, conservative values
    return float(np.mean(ax))


def calculate_gap(current_obj: float, best_obj: float) -> float:
    """
    Calculate gap:
        gap = |current_obj - best_obj| / |current_obj| * 100

    Notes:
      - Supports scalar and numpy array inputs with one or more elements.
      - Boundary handling matches the original implementation while avoiding
        Python boolean contexts on numpy objects.
    """
    # Both values are zero vectors or scalars
    if _is_all_zero(current_obj) and _is_all_zero(best_obj):
        return 0.0
    # current == 0 and best > 0 -> inf
    if _is_all_zero(current_obj):
        best_val = _to_scalar_or_mean(best_obj)
        if best_val > 0:
            return float("inf")
    # Reduce possible arrays to scalars for relative-difference calculation;
    # vectors are measured by their mean.
    cur_val = _to_scalar_or_mean(current_obj)
    best_val = _to_scalar_or_mean(best_obj)
    return abs(cur_val - best_val) / max(1e-12, abs(cur_val)) * 100
