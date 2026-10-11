"""
Centralized ALNS configuration module.

Purpose:
- Collect all top-level algorithm configuration, per-operator default parameters and ranges,
  tuner defaults, ML settings, and helper methods to read/update configs.
- Keep this module free of runtime imports (SolutionState, Vehicle, etc.) to avoid circular imports.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Optional


@dataclass
class ALNSConfig:
    # --------------------------
    # Meta / run control
    # --------------------------
    SEED: int = 15926535
    LOG_LEVEL: str = "INFO"  # DEBUG / INFO / WARN / ERROR
    VERBOSE: bool = True

    # Feature toggles
    ENABLE_PARAM_TUNER: bool = False
    ENABLE_ML_OPERATORS: bool = False
    TRACKER_ENABLED: bool = False
    # Enable paper-oriented visual reports by default (objective/gap/operator charts).
    ENABLE_REPORTS: bool = True
    # Keep startup plotting off by default to avoid extra overhead.
    ENABLE_STARTUP_PLOTTING: bool = False
    # Explicitly disable supply-chain 3D interactive plotting (time-consuming, low thesis value).
    ENABLE_SUPPLY_CHAIN_3D: bool = False
    QUIET_OPERATOR_LOGS: bool = True
    SHOW_BEST_FEASIBLE_PROGRESS: bool = True
    QUIET_INITIAL_SOLUTION_LOGS: bool = True
    VALIDATE_PRINT_STACK_ON_OVERLOAD: bool = False

    # Dataset defaults (can be overridden by user config)
    DATASET_TYPE: str = "large"
    DATASET_IDX: int = 1

    # --------------------------
    # Stopping / scheduling
    # --------------------------
    MAX_RUNTIME: int = 600  # seconds (10 minutes) — increased for long-run profiling
    MAX_ITERATIONS: int = 1000
    MAX_ITERATIONS_NO_IMPROVEMENT: int = 100  # currently unused
    DYNAMIC_MAX_ITERATIONS_PER_SECOND: float = 1000.0
    DYNAMIC_MAX_ITERATION_BUFFER: int = 200
    DYNAMIC_MAX_ITERATION_SAFETY_FACTOR: float = 1.5
    DYNAMIC_MAX_ITERATION_MIN_RATE: float = 1.0
    PREFER_FINAL_BEST_ACTUAL_SOLUTION: bool = True
    TERMINATE_ON_INFEASIBLE_INITIAL: bool = True
    # Hard-feasibility policy: only states satisfying all hard constraints
    # (except min-load) are considered feasible.
    STRICT_FEASIBILITY_MODE: bool = True
    # Validate demand fulfillment as a hard constraint in validate().
    VALIDATE_REQUIRE_FULL_DEMAND: bool = True
    # Always recompute inventory in validate(), even when suppression flag is set.
    VALIDATE_FORCE_RECOMPUTE_INVENTORY: bool = True
    # In callback runtime guard, force full validate each iteration under strict mode.
    ENABLE_STRICT_FEASIBILITY_ACCEPT: bool = True
    # Initial-solution strict finalization passes.
    INITIAL_STRICT_FINALIZE_MAX_PASSES: int = 3
    # Callback diagnostics (runtime-guard / inventory-check) console visibility.
    ENABLE_CALLBACK_DIAGNOSTICS_CONSOLE: bool = False
    # Stagnation diversification switches.
    ENABLE_LIGHT_VEHICLE_INTENSIFY: bool = True
    LIGHT_INTENSIFY_INTERVAL: int = 250
    LIGHT_INTENSIFY_STAGNATION: int = 120
    LIGHT_INTENSIFY_MIN_PROGRESS: float = 0.20
    LIGHT_INTENSIFY_MAX_PROGRESS: float = 0.98
    LIGHT_INTENSIFY_MAX_ITER: int = 12
    LIGHT_INTENSIFY_MAX_TARGETS: int = 24
    LIGHT_INTENSIFY_ACCEPT_REL_WORSE: float = 0.01
    LIGHT_INTENSIFY_MAX_NO_GAIN: int = 4
    LIGHT_INTENSIFY_COOLDOWN_ITERS: int = 1200
    LIGHT_INTENSIFY_REQUIRE_SIG_CHANGE: bool = True
    ENABLE_ELITE_PATH_RELINK: bool = False
    ELITE_POOL_MAX_SIZE: int = 8
    ELITE_SIGNATURE_BIN_SIZE: int = 10
    ELITE_POOL_SAMPLE_INTERVAL: int = 50
    ELITE_RELINK_INTERVAL: int = 200
    ELITE_RELINK_STAGNATION: int = 90
    ELITE_RELINK_MIN_PROGRESS: float = 0.15
    ELITE_RELINK_MAX_PROGRESS: float = 0.95
    ELITE_RELINK_REQUIRE_SIGNATURE_DIFF: bool = True
    ELITE_RELINK_MAX_REL_WORSE: float = 0.05
    ELITE_RELINK_BLEND_RATIO: float = 0.25
    ELITE_RELINK_MAX_TARGET_VEH: int = 18
    ELITE_RELINK_STEPS: int = 4
    ELITE_RELINK_BLEND_STEP_DELTA: float = 0.12
    ELITE_RELINK_TARGET_STEP_DELTA: int = 4
    ENABLE_STAGNATION_ESCAPE_RESTART: bool = True
    STAGNATION_ESCAPE_INTERVAL: int = 80
    STAGNATION_ESCAPE_STAGNATION: int = 120
    STAGNATION_ESCAPE_COOLDOWN: int = 300
    STAGNATION_ESCAPE_MAX_REL_WORSE_SEARCH: float = 0.05
    STAGNATION_ESCAPE_MAX_REL_WORSE_ACTUAL: float = 0.00
    STAGNATION_ESCAPE_DEGREE: float = 0.55
    STAGNATION_ESCAPE_WINDOW_RATIO: float = 0.45
    STAGNATION_ESCAPE_LOOKAHEAD_H: int = 3

    # --------------------------
    # Mixed operator reward / late acceptance policy
    # --------------------------
    MIXED_REWARD_SEARCH_WEIGHT: float = 0.60
    MIXED_REWARD_ACTUAL_WEIGHT: float = 0.40
    MIXED_REWARD_SCORE_SCALE: float = 100.0
    ACTUAL_PRIORITY_START_PROGRESS: float = 0.70
    ACTUAL_PRIORITY_ACTUAL_REL_TOL: float = 0.002
    ACTUAL_PRIORITY_ACTUAL_ABS_TOL: float = 0.25
    ACTUAL_PRIORITY_SEARCH_REL_TOL: float = 0.01
    ACTUAL_PRIORITY_SEARCH_ABS_TOL: float = 1.0
    ENABLE_LATE_FLEET_HARD_PRIORITY: bool = True
    LATE_FLEET_HARD_PRIORITY_START_PROGRESS: float = 0.60
    LATE_FLEET_HARD_PRIORITY_MAX_ACTUAL_WORSE: float = 0.75
    LATE_FLEET_HARD_PRIORITY_MAX_ACTUAL_REL_WORSE: float = 0.01
    # Deprecated: late fleet hard priority is now applied to all instance scales.
    LATE_FLEET_HARD_PRIORITY_MAX_SCALE: int = 250

    # --------------------------
    # Simulated annealing / acceptance
    # --------------------------
    SA_START_TEMP: float = 1000.0
    SA_END_TEMP: float = 15.0
    SA_STEP: float = 1.0 - 5e-4

    # --------------------------
    # Selection / roulette wheel
    # --------------------------
    ROULETTE_SCORES: List[float] = field(default_factory=lambda: [8.0, 3.0, 1.0, 0.2])
    ROULETTE_DECAY: float = 0.65
    ROULETTE_SEG_LENGTH: int = 80

    # --------------------------
    # Initial solution / parallelism
    # --------------------------
    PARALLEL_DEMAND_THRESHOLD: int = 1000
    MAX_INIT_THREADS: int = 8
    INITIAL_SOLUTION_STRATEGY: str = "improved_greedy"  # placeholder key
    INITIAL_USE_DEALER_GROUPING: bool = True
    INITIAL_ENABLE_CUMULATIVE_CONSTRAINT: bool = True
    INITIAL_ENABLE_RELAXED_REALLOCATION: bool = True

    # --------------------------
    # Random removal variants (convenience)
    # --------------------------
    RANDOM_REMOVAL_VARIANTS: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {"degree": 0.15, "name": "random_gentle"},
            {"degree": 0.25, "name": "random_normal"},
            {"degree": 0.35, "name": "random_aggressive"},
        ]
    )

    # --------------------------
    # ML / learning-based repair defaults
    # --------------------------
    ML_LEARNING_ENABLED: bool = False
    ML_MODEL_TYPE: str = "adaptive"  # 'linear', 'random_forest', 'adaptive'
    ML_MIN_SCORE: float = 0.4
    ML_INITIAL_SAMPLE_SIZE: int = 25
    ML_ADAPTIVE_SAMPLE_SIZE: int = 200
    ML_RETRAIN_INTERVAL: int = 80
    ML_TAKEOVER_MIN_SAMPLE_SIZE: int = 80
    ML_TAKEOVER_MIN_SAMPLE_SIZE_SMALL: int = 120
    ML_TAKEOVER_MIN_PROGRESS: float = 0.30
    ML_EPSILON_BASE: float = 0.10
    ML_EPSILON_EARLY_BOOST: float = 0.15
    ML_EPSILON_SMALL_INSTANCE_BOOST: float = 0.05
    ML_STAGNATION_NO_IMPROVE_LIMIT: int = 18
    ML_FALLBACK_COOLDOWN_ITERS: int = 30
    ML_WARMUP_PER_ITER: int = 2

    # --------------------------
    # Param tuner defaults (sync with ParamAutoTuner)
    # --------------------------
    TUNER_ENABLED: bool = False
    TUNER_LEARNING_RATE: float = 0.1
    TUNER_EXPLORATION_INITIAL: float = 0.3
    TUNER_PHASE_THRESHOLDS: Tuple[float, float] = (
        0.3,
        0.7,
    )  # exploration/exploitation/refinement splits
    TUNER_SMALL_INSTANCE_SCALE_THRESHOLD: int = 250
    TUNER_SMALL_INSTANCE_EXPLORATION_CAP: float = 0.18
    TUNER_SMALL_INSTANCE_SAMPLE_SCALE: float = 0.35
    TUNER_MIN_HISTORY_FOR_ADJUST: int = 5
    TUNER_MAX_REL_STEP: float = 0.15
    TUNER_STAGNATION_BOOST_THRESHOLD: int = 35
    TUNER_STAGNATION_BOOST_FACTOR: float = 0.35

    # --------------------------
    # Operator defaults (params + ranges) - used by ParamAutoTuner._init_default_operators
    # Keep keys in sync with ParamAutoTuner usage.
    # --------------------------
    OPERATOR_DEFAULTS: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            # Destroy operators
            "random_removal": {
                "params": {"degree": 0.25},
                "ranges": {"degree": (0.1, 0.5)},
            },
            "shaw_removal": {
                "params": {"degree": 0.3},
                "ranges": {"degree": (0.1, 0.5)},
            },
            "periodic_shaw_removal": {
                "params": {
                    "degree": 0.3,
                    "alpha": 0.4,
                    "beta": 0.3,
                    "gamma": 0.3,
                    "k_clusters": 3,
                },
                "ranges": {
                    "degree": (0.1, 0.5),
                    "alpha": (0.2, 0.6),
                    "beta": (0.1, 0.5),
                    "gamma": (0.1, 0.5),
                    "k_clusters": (2, 10),
                },
            },
            "path_removal": {
                "params": {"degree": 0.5},
                "ranges": {"degree": (0.2, 0.7)},
            },
            "cross_period_dealer_removal": {
                "params": {"degree": 0.45},
                "ranges": {"degree": (0.2, 0.8)},
            },
            "period_range_removal": {
                "params": {"degree": 0.35, "window_ratio": 0.30},
                "ranges": {"degree": (0.15, 0.8), "window_ratio": (0.15, 0.7)},
            },
            "worst_removal": {
                "params": {"degree": 0.25, "value_bias": 2.0},
                "ranges": {"degree": (0.1, 0.5), "value_bias": (1.0, 3.0)},
            },
            "infeasible_removal": {"params": {}, "ranges": {}},
            "surplus_inventory_removal": {
                "params": {"degree": 0.25},
                "ranges": {"degree": (0.1, 0.5)},
            },
            "inventory_level_destruction": {
                "params": {"degree": 0.25},
                "ranges": {"degree": (0.1, 0.5)},
            },
            # Repair operators
            "greedy_repair": {
                "params": {"demand_weight": 0.8, "stock_weight": 0.2},
                "ranges": {"demand_weight": (0.0, 1.0), "stock_weight": (0.0, 1.0)},
            },
            "local_search_repair": {"params": {"max_iter": 10}, "ranges": {}},
            "inventory_balance_repair": {"params": {}, "ranges": {}},
            "infeasible_repair": {"params": {}, "ranges": {}},
            "regret_based_repair": {
                "params": {"k": 2, "topN": 6},
                "ranges": {"k": (1, 5), "topN": (1, 20)},
            },
            "smart_batch_repair": {
                "params": {"batch_size": 10, "max_iter": 10, "timeout": None},
                "ranges": {"batch_size": (5, 50)},
            },
            "sequential_repair": {"params": {}, "ranges": {}},
            "lookahead_repair": {
                "params": {"lookahead_horizon": 2},
                "ranges": {"lookahead_horizon": (1, 4)},
            },
            "fleet_consolidation_repair": {"params": {}, "ranges": {}},
        }
    )

    # --------------------------
    # Convenience - high-level mapping for destroy/repair param quick access
    # --------------------------
    DESTROY_DEFAULTS: Dict[str, Any] = field(
        default_factory=lambda: {
            "random_removal_degree": 0.25,
            "shaw_removal_degree": 0.3,
            "periodic_shaw_params": {
                "degree": 0.3,
                "alpha": 0.4,
                "beta": 0.3,
                "gamma": 0.3,
                "k_clusters": 3,
            },
            "cross_period_dealer_removal_degree": 0.45,
            "period_range_removal_degree": 0.35,
            "period_range_removal_window_ratio": 0.30,
            "inventory_level_destruction_degree": 0.25,  # P1：库存水平破坏
        }
    )

    REPAIR_DEFAULTS: Dict[str, Any] = field(
        default_factory=lambda: {
            "greedy_repair": {"demand_weight": 0.8, "stock_weight": 0.2},
            "local_search_repair": {"max_iter": 10},
            "smart_batch_repair": {"max_iter": 10, "batch_size": 10, "timeout": None},
            "regret_based_repair": {"k": 2, "topN": 6, "time_limit": 10.0},
            "sequential_repair": {},  # P0：顺序修复
            "lookahead_repair": {"lookahead_horizon": 2},
            "fleet_consolidation_repair": {},
        }
    )

    # lookahead 修复中对“延后交付天数”的惩罚系数（越大越偏好当期修复）
    LOOKAHEAD_REPAIR_DELAY_PENALTY: float = 0.5

    # --------------------------
    # Repair post-processing knobs
    # --------------------------
    REPAIR_ENABLE_POST_CONSOLIDATION: bool = True
    REPAIR_POST_CONSOLIDATION_MAX_PAIRS: int = 300

    # Optional final pass on best solution before output
    FINAL_BEST_ENABLE_CONSOLIDATION: bool = True
    FINAL_BEST_CONSOLIDATION_MAX_PAIRS: int = 800

    # --------------------------
    # Best-solution fleet preference (used in callbacks/solution ranking)
    # --------------------------
    ENABLE_VEHICLE_COUNT_PRIORITY: bool = True
    VEHICLE_PRIORITY_OBJ_REL_TOL: float = 0.08
    VEHICLE_PRIORITY_OBJ_ABS_TOL: float = 12.0
    FLEET_NEAR_EQUAL_OBJ_REL_TOL: float = 0.01
    FLEET_NEAR_EQUAL_OBJ_ABS_TOL: float = 1.0
    FLEET_LOAD_RATIO_MARGIN: float = 1e-4
    FLEET_FIXED_COST_MARGIN: float = 1e-6
    FLEET_LARGE_RATIO_MARGIN: float = 1e-6

    # --------------------------
    # Explicit fleet-size penalty (in objective function)
    # --------------------------
    ENABLE_FLEET_SIZE_PENALTY: bool = True
    FLEET_SIZE_PENALTY_PER_VEHICLE: float = 15.0  # Cost penalty per additional vehicle
    FLEET_SIZE_BASELINE: int = (
        25  # Reference fleet size; excess vehicles over this incur penalty
    )
    # Search objective decomposition knobs
    SEARCH_USE_DEMAND_PENALTY: bool = True
    SEARCH_USE_UNDERLOAD_GAP_PENALTY: bool = True
    SEARCH_PHASED_VEHICLE_PRIORITY: bool = True
    SEARCH_PHASE1_END_PROGRESS: float = 0.45
    SEARCH_PHASE2_END_PROGRESS: float = 0.80
    SEARCH_PHASE1_DEMAND_WEIGHT: float = 0.35
    SEARCH_PHASE1_UNDERLOAD_WEIGHT: float = 0.80
    SEARCH_PHASE1_FLEET_WEIGHT: float = 4.00
    SEARCH_PHASE2_DEMAND_WEIGHT: float = 0.80
    SEARCH_PHASE2_UNDERLOAD_WEIGHT: float = 1.00
    SEARCH_PHASE2_FLEET_WEIGHT: float = 2.00
    SEARCH_PHASE3_DEMAND_WEIGHT: float = 1.00
    SEARCH_PHASE3_UNDERLOAD_WEIGHT: float = 1.00
    SEARCH_PHASE3_FLEET_WEIGHT: float = 1.00
    # Feasibility-first shaping when unmet demand exists (mainly for non-strict runs).
    SEARCH_FEASIBILITY_FIRST: bool = True
    SEARCH_FEASIBILITY_DEMAND_BOOST: float = 6.0
    SEARCH_FEASIBILITY_FLEET_SCALE_WHEN_UNMET: float = 0.15
    SEARCH_FEASIBILITY_UNDERLOAD_SCALE_WHEN_UNMET: float = 0.25
    SCALE_FACTOR: float = 1e-3

    # Large-jump destroy operator tuning
    CROSS_PERIOD_REMOVAL_PHASED: bool = True
    CROSS_PERIOD_REMOVAL_EARLY_BOOST: float = 1.35
    CROSS_PERIOD_REMOVAL_LATE_SHRINK: float = 0.75
    CROSS_PERIOD_REMOVAL_STAGNATION_WINDOW: int = 80
    CROSS_PERIOD_REMOVAL_STAGNATION_BOOST: float = 1.20

    # --------------------------
    # periodic_shaw KMeans tuning knobs (micro-tuning / perf)
    # Exposed here so users/CI can reduce KMeans cost without code edits.
    # --------------------------
    PERIODIC_SHAW_N_INIT: int = 3
    PERIODIC_SHAW_MAX_ITER: int = 50
    PERIODIC_SHAW_TOL: float = 1e-3
    PERIODIC_SHAW_USE_FLOAT32: bool = True

    # --------------------------
    # Regret operator practical caps (safeguards for production)
    # These limit candidate branching / caching granularity.
    # --------------------------
    REGRET_MAX_TOPN: int = 12
    REGRET_MAX_K: int = 5

    # --------------------------
    # Regret simulation tuning (controls how many candidates are fully simulated
    # and the fast-estimator threshold to gate full simulations). Adjust to
    # trade off CPU vs solution-quality in production.
    # --------------------------
    # Maximum number of heuristic-selected candidates to perform full exact
    # simulation on (overrides the per-call fallback of max(6, topN*2)).
    # Keep conservative default but allow tightening via config.
    REGRET_SIM_MAX_SIMULATE: int = 4

    # Require the fast estimator improvement to exceed this threshold before
    # performing an expensive full simulation. Value is in the "fast-impr"
    # estimator scale (typically small); increase to be more conservative.
    # Lowering this value makes the gate stricter (fewer full sims). Set to 1e-3.
    REGRET_FULL_SIM_THRESHOLD: float = 0.001

    # --------------------------
    # Logging / persistence
    # --------------------------
    LOG_DIR: Optional[str] = "logs-alns"
    SAVE_BEST_EVERY_N_ITERS: int = 50

    # --------------------------
    # Helper instance methods
    # --------------------------
    def get_destroy_params(self) -> Dict[str, Any]:
        """Return a compact dict of commonly used destroy parameters."""
        return {
            "random_removal_degree": self.DESTROY_DEFAULTS.get(
                "random_removal_degree", 0.25
            ),
            "shaw_removal_degree": self.DESTROY_DEFAULTS.get(
                "shaw_removal_degree", 0.3
            ),
            "periodic_shaw_params": self.DESTROY_DEFAULTS.get(
                "periodic_shaw_params", {}
            ),
            "cross_period_dealer_removal_degree": self.DESTROY_DEFAULTS.get(
                "cross_period_dealer_removal_degree", 0.45
            ),
            "period_range_removal_degree": self.DESTROY_DEFAULTS.get(
                "period_range_removal_degree", 0.35
            ),
            "period_range_removal_window_ratio": self.DESTROY_DEFAULTS.get(
                "period_range_removal_window_ratio", 0.30
            ),
            "inventory_level_destruction_degree": self.DESTROY_DEFAULTS.get(
                "inventory_level_destruction_degree", 0.25
            ),
        }

    def get_repair_params(self) -> Dict[str, Any]:
        """Return a compact dict of commonly used repair defaults."""
        return self.REPAIR_DEFAULTS.copy()

    def get_operator_default(self, op_name: str) -> Dict[str, Any]:
        """Return the operator default structure {params, ranges} for a given operator."""
        return self.OPERATOR_DEFAULTS.get(op_name, {"params": {}, "ranges": {}})

    def get_tuner_operator_defaults(self) -> Dict[str, Dict[str, Any]]:
        """
        Return a mapping suitable for ParamAutoTuner._init_default_operators:
        {op_name: {"params":..., "ranges":...}, ...}
        """
        return self.OPERATOR_DEFAULTS.copy()

    def update_from_dict(self, cfg: Dict[str, Any]):
        """
        Update configuration fields from a dict (in-place).
        Accepts nested structures for OPERATOR_DEFAULTS / DESTROY_DEFAULTS etc.
        """
        for k, v in cfg.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                # Allow updating nested dict entries
                if k in ("OPERATOR_DEFAULTS", "DESTROY_DEFAULTS", "REPAIR_DEFAULTS"):
                    existing = getattr(self, k, None)
                    if isinstance(existing, dict) and isinstance(v, dict):
                        existing.update(v)
                        setattr(self, k, existing)
                    else:
                        setattr(self, k, v)
                else:
                    # Unknown keys stored in a generic attribute map
                    setattr(self, k, v)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the main configuration to a plain dict (shallow for convenience)."""
        out = {}
        for name, val in self.__dict__.items():
            out[name] = val
        return out


# Module-level default config instance for easy import/use
default_config = ALNSConfig()
