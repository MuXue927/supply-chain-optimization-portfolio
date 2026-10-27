"""
Centralized ALNS configuration module.

Purpose:
- Collect all top-level algorithm configuration, per-operator default parameters and ranges,
  tuner defaults, ML settings, and helper methods to read/update configs.
- Keep this module free of runtime imports (SolutionState, Vehicle, etc.) to avoid circular imports.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ALNSConfig:
    # --------------------------
    # Meta / run control
    # --------------------------
    # Seed for every stochastic component of the search (operator choice,
    # removals, acceptance sampling). The CLI and the experiment runners set it
    # from --seed, which is what makes a run reproducible.
    SEED: int = 15926535
    # Python logging level ("DEBUG" / "INFO" / "WARNING" / "ERROR") used when
    # the script entry point configures its file log handler.
    LOG_LEVEL: str = "INFO"  # DEBUG / INFO / WARN / ERROR

    # --------------------------
    # Feature toggles
    # --------------------------
    # Master switch for the online parameter tuner (ParamAutoTuner). The CLI
    # flag --enable-param-tuner sets it; the optimizer reads it through
    # is_param_tuner_enabled() and registers the ALNS-T / ALNS-F variants.
    # Tuner is **off** by default, and the stop-loss criterion for this line has
    # been triggered: on the deterministic protocol (medium, 120 iterations,
    # 15 paired seeds, pinned hash seed) tuner-only scored **+3.95** against
    # basic (95% CI [-1.77, +9.66], 5 wins / 9 losses / 1 tie) - no improvement,
    # point estimate worse. Per that criterion the tuner is default-disabled and
    # ``full`` is, in measured effect, equal to ``ml``.
    #
    # The mechanism is kept (implementation intact, switches available) so the
    # measurement can be reproduced; enabling it should be treated as an
    # experiment, not a performance option.
    ENABLE_PARAM_TUNER: bool = False
    # Master switch for machine-learning operator selection
    # (MLOperatorSelector). The CLI flag --enable-ml-operators sets it; the
    # optimizer reads it through is_ml_enabled() for the ALNS-ML / ALNS-F
    # variants.
    ENABLE_ML_OPERATORS: bool = False
    # Generate the post-run plots (objective, gap and operator-performance
    # charts) in ALNSOptimizer._generate_reports.
    ENABLE_REPORTS: bool = True
    # Show the interactive starting plots before the search begins. The heavy
    # 3D view additionally requires ENABLE_SUPPLY_CHAIN_3D.
    ENABLE_STARTUP_PLOTTING: bool = False
    # Build the interactive Plotly 3D supply-chain network. It is expensive and
    # needs ENABLE_STARTUP_PLOTTING as well, so it stays off by default.
    ENABLE_SUPPLY_CHAIN_3D: bool = False
    # Silence the per-operator "[OPLOG] ..." stdout lines while ALNS iterates.
    # Turn it off when debugging a single operator.
    QUIET_OPERATOR_LOGS: bool = True
    # Silence the verbose initial-solution construction output.
    QUIET_INITIAL_SOLUTION_LOGS: bool = True
    # Destroy operators omitted from the roulette entirely, by registration name.
    # This is stricter than a zero weight: a disabled operator is not registered,
    # so it neither executes nor consumes a roulette draw. The default is empty,
    # preserving the historical operator set and trajectory bit-for-bit.
    # Intended for attribution experiments; supervised ML keeps a static
    # operator map and is outside this switch's measured scope.
    DISABLED_DESTROY_OPERATORS: tuple[str, ...] = ()

    # --------------------------
    # Dataset defaults (overwritten by the CLI --scale / --dataset-id)
    # --------------------------
    # Scale sub-directory under datasets/multiple-periods: small / medium / large.
    DATASET_TYPE: str = "large"
    # Which dataset_N directory inside that scale to load.
    DATASET_IDX: int = 1

    # --------------------------
    # Stopping / scheduling
    # --------------------------
    # Wall-clock budget in seconds. This is the MaxRuntime stopping criterion
    # and the value the CLI sets from --max-runtime.
    MAX_RUNTIME: int = 600  # seconds (10 minutes) — increased for long-run profiling
    # Hard iteration cap. The dynamic cap below may raise it further; whichever
    # of runtime and iteration limits fires first stops the search.
    MAX_ITERATIONS: int = 1000
    # What "how far through the run are we" measures.
    #
    # ``"runtime"`` is the production default: elapsed seconds / MAX_RUNTIME, so
    # a threshold fires at the same point of the wall clock on any machine.
    #
    # ``"iteration"`` is for **fixed-iteration experiments only**. A paired test
    # that only caps MAX_ITERATIONS still leaks wall clock: the slower mode (more
    # per-iteration work) reaches a higher progress value at the same iteration,
    # which moves the SA cooling, the phase thresholds, the roulette segments and
    # the ML takeover gate. Measuring progress by iteration removes that leak.
    # It is not a better production measure - throughput is machine-specific, so
    # no iteration-based threshold means the same thing everywhere.
    PROGRESS_BASIS: str = "runtime"
    # Make ``MAX_ITERATIONS`` the only stopping condition.
    #
    # Off by default. When on, ``compute_dynamic_iteration_cap`` stops widening
    # the ceiling from the runtime budget, so a run stops at exactly
    # ``MAX_ITERATIONS`` iterations; ``MAX_RUNTIME`` must then be set high enough
    # not to bind. This exists because a fixed-iteration paired experiment is
    # otherwise impossible: today the runtime budget both raises the iteration
    # ceiling and acts as a stop, so "run exactly N iterations" cannot be
    # expressed. Production runs leave it off and keep the runtime budget.
    HARD_ITERATION_BUDGET: bool = False
    # Upper bound on iterations per second, used only to keep the dynamic
    # iteration ceiling from being absurdly small.
    #
    # This is **a safety net, not a schedule input.** The phase policies
    # (LIGHT_INTENSIFY_MIN_PROGRESS 0.20, LATE_FLEET_HARD_PRIORITY_START 0.60,
    # ACTUAL_PRIORITY_START_PROGRESS 0.70, the elite-relink window, ML takeover
    # at 0.30, the tuner decay and the epsilon schedule) measure progress as
    # elapsed-runtime / MAX_RUNTIME - see ``progress.py`` - so they no longer
    # depend on this number at all.
    #
    # That change makes the ceiling's only remaining job to stop a pathological
    # run. It must therefore NOT bind in normal operation: if the iteration
    # ceiling fires before MAX_RUNTIME, the run is truncated and the later phase
    # gates never execute. Sizing this to *measured* throughput (an earlier
    # revision used 5.0) caused exactly that - on ``small/dataset_1`` the
    # ceiling bound after ~152 s of a 600 s budget, so progress only reached
    # ~0.25 and the 0.60/0.70 gates still could not fire.
    #
    # Measured throughput on one host spans 0.1-40 iterations/second, so any
    # value that a real machine can reach is safe here; the default is
    # deliberately generous.
    DYNAMIC_MAX_ITERATIONS_PER_SECOND: float = 1000.0
    # Flat number of extra iterations added on top of the estimated dynamic cap.
    DYNAMIC_MAX_ITERATION_BUFFER: int = 200
    # Multiplier applied to the estimated dynamic cap (1.5 = 50% headroom).
    DYNAMIC_MAX_ITERATION_SAFETY_FACTOR: float = 1.5
    # Report the final solution chosen by the actual (Gurobi-aligned) objective
    # instead of the best state seen by the search objective.
    PREFER_FINAL_BEST_ACTUAL_SOLUTION: bool = True
    # Abort the run when the initial solution is infeasible instead of letting
    # the repair operators try to recover it.
    TERMINATE_ON_INFEASIBLE_INITIAL: bool = True
    # Hard-feasibility policy: only states satisfying all hard constraints
    # (except min-load) are considered feasible.
    # When on, validate() also recomputes inventory so the verdict is
    # authoritative rather than cache-based.
    STRICT_FEASIBILITY_MODE: bool = True
    # Treat unmet demand as a hard violation inside validate(). Turning it off
    # allows partial solutions to pass validation, which the search uses as a
    # temporary state rather than an accepted answer.
    VALIDATE_REQUIRE_FULL_DEMAND: bool = True
    # Recompute inventory during validate() even while inventory updates are
    # suppressed (batch updates, simulations). Slower but authoritative.
    VALIDATE_FORCE_RECOMPUTE_INVENTORY: bool = True
    # In the callback runtime guard, force a full validate() every iteration
    # while strict feasibility mode is active.
    ENABLE_STRICT_FEASIBILITY_ACCEPT: bool = True
    # Number of deterministic repair passes used to push the constructed
    # initial solution into strict feasibility.
    INITIAL_STRICT_FINALIZE_MAX_PASSES: int = 3
    # Print the runtime-guard / inventory-check diagnostics from the callbacks
    # to the console. Intended for debugging, not for normal runs.
    ENABLE_CALLBACK_DIAGNOSTICS_CONSOLE: bool = False

    # --- Stagnation diversification: light vehicle intensification ---
    # Periodically re-run a small vehicle-elimination repair from the best
    # feasible state to squeeze out fleet size without a full restart.
    # Master switch for that callback.
    ENABLE_LIGHT_VEHICLE_INTENSIFY: bool = True
    # Run it when the iteration counter is a multiple of this value.
    LIGHT_INTENSIFY_INTERVAL: int = 250
    # Also require at least this many consecutive non-improving iterations.
    LIGHT_INTENSIFY_STAGNATION: int = 120
    # Only run between these fractions of the total iteration budget (0-1).
    LIGHT_INTENSIFY_MIN_PROGRESS: float = 0.20
    # Upper bound of that progress window.
    LIGHT_INTENSIFY_MAX_PROGRESS: float = 0.98
    # Maximum local-search iterations inside one intensification call.
    LIGHT_INTENSIFY_MAX_ITER: int = 12
    # Maximum number of vehicles the elimination repair may touch per call.
    LIGHT_INTENSIFY_MAX_TARGETS: int = 24
    # Accept a candidate that is up to this fraction worse than the incumbent.
    LIGHT_INTENSIFY_ACCEPT_REL_WORSE: float = 0.01
    # Give up on the call after this many consecutive non-improving moves.
    LIGHT_INTENSIFY_MAX_NO_GAIN: int = 4
    # Number of iterations to wait before the next intensification call.
    LIGHT_INTENSIFY_COOLDOWN_ITERS: int = 1200
    # Only start when the solution signature differs from the previous call, so
    # the same basin is not revisited.
    LIGHT_INTENSIFY_REQUIRE_SIG_CHANGE: bool = True

    # --- Stagnation diversification: elite path relinking ---
    # Relink the current solution toward solutions stored in the elite pool.
    # Master switch for that callback.
    ENABLE_ELITE_PATH_RELINK: bool = False
    # How many elite solutions the tracker keeps (best-first, de-duplicated).
    ELITE_POOL_MAX_SIZE: int = 8
    # Bin width used to discretize the solution signature that de-duplicates
    # the elite pool.
    ELITE_SIGNATURE_BIN_SIZE: int = 10
    # Sample a new elite candidate every this many iterations.
    ELITE_POOL_SAMPLE_INTERVAL: int = 50
    # Run a relinking attempt every this many iterations.
    ELITE_RELINK_INTERVAL: int = 200
    # Also run a relinking attempt after this many non-improving iterations.
    ELITE_RELINK_STAGNATION: int = 90
    # Allowed progress window (fraction of the iteration budget) for relinking.
    ELITE_RELINK_MIN_PROGRESS: float = 0.15
    # Upper bound of that progress window.
    ELITE_RELINK_MAX_PROGRESS: float = 0.95
    # Require the elite to have a different signature from the current state.
    ELITE_RELINK_REQUIRE_SIGNATURE_DIFF: bool = True
    # Accept the relinked solution only if it is at most this fraction worse
    # than the incumbent.
    ELITE_RELINK_MAX_REL_WORSE: float = 0.05
    # Starting blend ratio between the current solution and the elite.
    ELITE_RELINK_BLEND_RATIO: float = 0.25
    # Maximum number of vehicles modified per relinking step.
    ELITE_RELINK_MAX_TARGET_VEH: int = 18
    # Number of blending steps performed per relinking attempt.
    ELITE_RELINK_STEPS: int = 4
    # How much the blend ratio changes from one step to the next.
    ELITE_RELINK_BLEND_STEP_DELTA: float = 0.12
    # How many additional target vehicles each successive step adds.
    ELITE_RELINK_TARGET_STEP_DELTA: int = 4

    # --- Stagnation diversification: basin-jump restart ---
    # The most aggressive escape: a strong period-range removal followed by a
    # lookahead repair, seeded from the best feasible state.
    ENABLE_STAGNATION_ESCAPE_RESTART: bool = True
    # Trigger when the stagnation counter or the iteration counter is a
    # multiple of this value.
    STAGNATION_ESCAPE_INTERVAL: int = 80
    # Minimum number of consecutive non-improving iterations before escaping.
    STAGNATION_ESCAPE_STAGNATION: int = 120
    # Iterations to wait after an escape before the next one is allowed.
    STAGNATION_ESCAPE_COOLDOWN: int = 300
    # Accept the escaped candidate only if its search objective is at most this
    # fraction worse than the current one.
    STAGNATION_ESCAPE_MAX_REL_WORSE_SEARCH: float = 0.05
    # Same check for the actual objective; 0.0 means never accept a worse
    # actual objective.
    STAGNATION_ESCAPE_MAX_REL_WORSE_ACTUAL: float = 0.00
    # Removal degree handed to period_range_removal during the escape.
    STAGNATION_ESCAPE_DEGREE: float = 0.55
    # Day-window ratio handed to period_range_removal during the escape.
    STAGNATION_ESCAPE_WINDOW_RATIO: float = 0.45
    # Lookahead horizon handed to lookahead_repair during the escape.
    STAGNATION_ESCAPE_LOOKAHEAD_H: int = 3

    # --------------------------
    # Mixed operator reward / late acceptance policy
    # --------------------------
    # Weight of the normalized search-objective improvement in the blended
    # operator reward computed by compute_mixed_operator_reward().
    MIXED_REWARD_SEARCH_WEIGHT: float = 0.60
    # Weight of the normalized actual-objective improvement in the same blended
    # reward. The two weights are applied to independently normalized terms.
    MIXED_REWARD_ACTUAL_WEIGHT: float = 0.40
    # Multiplier that scales the blended reward into the range the tuner and the
    # ML selector expect.
    MIXED_REWARD_SCORE_SCALE: float = 100.0
    # Progress fraction after which accept() switches to the actual-priority
    # rule, which compares candidates on the actual objective, not the search
    # objective.
    ACTUAL_PRIORITY_START_PROGRESS: float = 0.70
    # Relative tolerance for comparing actual objectives inside that rule.
    ACTUAL_PRIORITY_ACTUAL_REL_TOL: float = 0.002
    # Absolute tolerance for the same comparison; the effective tolerance is
    # max(absolute, relative * |current|).
    ACTUAL_PRIORITY_ACTUAL_ABS_TOL: float = 0.25
    # Relative tolerance for comparing search objectives under the same rule.
    ACTUAL_PRIORITY_SEARCH_REL_TOL: float = 0.01
    # Absolute tolerance for the same comparison.
    ACTUAL_PRIORITY_SEARCH_ABS_TOL: float = 1.0
    # Late in the search, prefer a candidate that uses fewer vehicles even when
    # its actual objective is slightly worse.
    ENABLE_LATE_FLEET_HARD_PRIORITY: bool = True
    # Progress fraction from which that fleet-count preference is active.
    LATE_FLEET_HARD_PRIORITY_START_PROGRESS: float = 0.60
    # How much worse the actual objective may be while still accepting a
    # smaller fleet, in absolute terms.
    LATE_FLEET_HARD_PRIORITY_MAX_ACTUAL_WORSE: float = 0.75
    # Relative counterpart; the effective allowance is
    # max(absolute, relative * |current|).
    LATE_FLEET_HARD_PRIORITY_MAX_ACTUAL_REL_WORSE: float = 0.01

    # --------------------------
    # Simulated annealing / acceptance
    # --------------------------
    # Start temperature of the simulated-annealing acceptance criterion. A high
    # value accepts many worsening moves early in the search.
    SA_START_TEMP: float = 1000.0
    # End temperature, at which the criterion is effectively greedy. Cooling is
    # driven by SA_STEP and the run's progress.
    SA_END_TEMP: float = 15.0
    # Multiplicative decay applied to the temperature after every iteration.
    # 1.0 - 5e-4 is a very slow cooling schedule suited to long runs.
    SA_STEP: float = 1.0 - 5e-4
    # Start temperature as a fraction of the initial search objective. When
    # positive, cooling follows elapsed/MAX_RUNTIME instead of a per-iteration
    # step, so the schedule does not depend on throughput. Set to 0 to keep the
    # legacy SA_START_TEMP / SA_STEP schedule.
    #
    # Measured on large at a 150s budget: the legacy schedule kept T near 1000
    # for the whole run, the incumbent drifted from 2075.80 to 3134-3785 and
    # never returned; repair then worked on a much worse state. With 0.02 the
    # incumbent stayed within 2075.80-2078 and delivered 2043.43 on average
    # (seeds 42/7/123) versus 2049.25 for the legacy schedule.
    SA_START_REL_TEMP: float = 0.02

    # --------------------------
    # Selection / roulette wheel
    # --------------------------
    # Roulette-wheel score levels for [new best, better, accepted, rejected]
    # moves. Higher values make the operators that produced them more likely to
    # be selected again by the ALNS selector.
    ROULETTE_SCORES: list[float] = field(default_factory=lambda: [8.0, 3.0, 1.0, 0.2])
    # Decay factor applied to accumulated operator scores each segment.
    ROULETTE_DECAY: float = 0.65
    # Number of iterations per scoring segment before scores are decayed.
    ROULETTE_SEG_LENGTH: int = 80
    # Divide the run into this many scoring segments by *progress* instead of
    # by iteration count. Zero keeps the iteration-based behaviour above.
    #
    # Rationale: ``ROULETTE_SEG_LENGTH = 80`` never completes on large, where a
    # 150 s run finishes 49-80 iterations, so the operator weights never update
    # and selection stays effectively uniform. Progress is elapsed/MAX_RUNTIME,
    # the same measure the phase policies and the cooling schedule use.
    ROULETTE_PROGRESS_SEGMENTS: int = 0

    # How a segment's scores become one weight update.
    #
    # ``"sum"`` (library default) credits the score total, so an operator that
    # is selected more often also accumulates more credit - a positive feedback
    # loop unrelated to quality. Measured with eleven operators drawing from one
    # identical score distribution: ``"sum"`` gives the luckiest operator 2.9x
    # the uniform share on 24/24 seeds.
    #
    # ``"mean"`` divides by the per-segment call count. That fixes the volume
    # bias but not the variance one: with few calls per operator per segment a
    # single lucky score sets the mean, and it drifts to 1.8x at seg_length=20.
    #
    # ``"shrunk"`` divides *and* pulls the ratio towards the segment's overall
    # mean with a prior of ``ROULETTE_REWARD_PRIOR_K`` pseudo-calls. It stays at
    # 1.2x at both segment lengths (10.7% vs 9.09% uniform at seg_length=80,
    # 10.2% at seg_length=20).
    #
    # Left at ``"sum"`` by default: this changes which operators run, so it is
    # opt-in until an A/B on cost confirms it.
    ROULETTE_REWARD_NORMALISATION: str = "sum"
    # Pseudo-calls used by the ``"shrunk"`` normalisation. Larger values trust
    # the segment mean more than any single operator's ratio.
    ROULETTE_REWARD_PRIOR_K: float = 2.0

    # --------------------------
    # Initial solution
    # --------------------------
    # Allocate demand dealer by dealer (True) instead of one demand at a time.
    # Disabling it selects the legacy demand-by-demand allocation path.
    INITIAL_USE_DEALER_GROUPING: bool = True
    # Choose the vehicle type by total cost rather than "smallest that fits".
    #
    # The fitting rule only guarantees that one vehicle of the chosen type can
    # hold the chunk; it never compares what a split into several smaller
    # vehicles would cost. On the large catalogue the two rules disagree on 24
    # of 60 sampled load sizes (up to 3.88 per load), and the initial solution
    # was 69.2% full with 11.9% of its cost coming from minimum-load penalties.
    INITIAL_COST_BASED_VEHICLE_CHOICE: bool = True
    # Regroup each (plant, dealer, day) route's cargo into the cheapest
    # vehicles for that route's volume, after allocation.
    #
    # The allocator builds one vehicle per (dealer, sku, plant, day) chunk and
    # ``_merge_small_vehicles`` can only merge into capacity that already
    # exists - it never upgrades a type. On large the result was 2,958 vehicles
    # over 920 routes at 69.2% fill; the same cargo fits in 1,847 vehicles for
    # 2,735.97 instead of 3,428.91 (-20.2%).
    #
    # Shipment-preserving: only cargo *within* one route is regrouped, so every
    # (plant, sku, day) quantity is unchanged and the inventory table is
    # identical. The pass verifies that and reverts otherwise.
    INITIAL_REPACK_ROUTES: bool = True
    # Run the existing cross-period consolidation operator as a final initial
    # solution pass. It can move a whole vehicle's cargo to another vehicle of
    # the same (plant, dealer) route on a different day when the resulting
    # inventory remains feasible, removing vehicles the same-day repack cannot.
    #
    # Measured on large dataset_3: the pass runs out of improving merges at
    # 295 moves, taking 2791.68 -> 2075.80 (1759 -> 1464 vehicles, ~2s).
    # medium dataset_13 stops at 103 moves and dataset_26 at 138, so the cap
    # only binds on large. The default is a fixed move cap with no wall-clock
    # deadline: a time cutoff made the initial solution, and hence the whole
    # search, depend on machine load.
    #
    # End-to-end at a fixed 150s search budget: 2148.46 -> 2062.26 (seed 42)
    # and 2143.44 -> 2039.14 (seed 7) versus the old 150-move cap.
    INITIAL_CROSS_PERIOD_CONSOLIDATION: bool = True
    # Zero means "no wall-clock deadline"; the move cap below is the budget.
    INITIAL_CROSS_PERIOD_CONSOLIDATION_TIME_LIMIT: float = 0.0
    INITIAL_CROSS_PERIOD_CONSOLIDATION_MAX_MOVES: int = 300
    # Enforce the cumulative (plant, SKU, day) availability constraint while
    # allocating, which prevents later allocations from over-committing stock.
    INITIAL_ENABLE_CUMULATIVE_CONSTRAINT: bool = True
    # Allow a relaxed reallocation pass when the strict pass cannot fit a
    # demand, instead of leaving it unmet.
    INITIAL_ENABLE_RELAXED_REALLOCATION: bool = True

    # --------------------------
    # ML / learning-based repair defaults
    # --------------------------
    # Number of recorded samples required before the ML selector starts being
    # used at all (the warm-up phase).
    #
    # Calibrated against measured throughput (medium/dataset_13, 90 s budget =
    # 106 iterations). Samples grow by 2 per iteration only while warming up,
    # so a threshold of S costs S/2 iterations: 12 costs ~6 iterations (6% of
    # the run) instead of 40 iterations (38%) at the previous value of 25 with
    # a takeover threshold of 80.
    ML_INITIAL_SAMPLE_SIZE: int = 12
    # Retrain the ridge / random-forest models every this many iterations.
    #
    # The previous value (80) exceeded the number of iterations a 90 s run
    # performs (106), so at most one retrain happened; 25 gives 3-4 retrains
    # inside the same budget.
    ML_RETRAIN_INTERVAL: int = 25
    # Hard cap on the number of samples kept in the ML operator selector's
    # training buffers. Keeping only the most recent window bounds memory (the
    # buffers would otherwise grow with every iteration and never shrink) and
    # keeps each retraining pass O(window) instead of O(samples so far).
    # Matches the sliding window already used by ALNSTracker.update_ml_data.
    ML_MAX_SAMPLES: int = 1000
    # Samples required before ML selection may replace the default selector.
    #
    # Measured: reaching 80 samples consumed 39 iterations (37% of a 106-
    # iteration run) and left the selector with almost nothing to act on.
    # 40 costs ~20 iterations (19%), which leaves the majority of the budget
    # to actual model-based selection.
    ML_TAKEOVER_MIN_SAMPLE_SIZE: int = 40
    # Higher takeover requirement for small instances, which converge quickly
    # and would otherwise train on too little data.
    # Scaled with the value above (was 120 for a base of 80).
    ML_TAKEOVER_MIN_SAMPLE_SIZE_SMALL: int = 60
    # --- Model quality gate ---------------------------------------------
    # Takeover requires evidence that the model beats the sample mean on data
    # it was not fitted on. Without this gate a model that had merely *seen*
    # enough rows was allowed to replace the selector, which made "ML is on"
    # and "ML knows something" the same claim (measured: three enhanced modes
    # all slightly worse than the plain baseline).
    #
    # Holdout is chronological (the most recent samples), matching how the
    # buffers grow, and is validated before the deployed model is refitted on
    # everything.
    ML_MODEL_HOLDOUT_FRACTION: float = 0.25
    # Minimum holdout rows; below ``MIN_TRAIN + MIN_HOLDOUT`` samples the model
    # is fitted but takeover stays blocked (no evidence yet).
    ML_MODEL_MIN_HOLDOUT_SAMPLES: int = 8
    ML_MODEL_MIN_TRAIN_SAMPLES: int = 8
    # Required out-of-sample R2. 0.0 means "at least as good as predicting the
    # mean"; raise it to demand a real signal before ML may take over.
    ML_MIN_HOLDOUT_R2: float = 0.0
    # Earliest progress fraction (0-1) at which ML takeover is allowed.
    ML_TAKEOVER_MIN_PROGRESS: float = 0.30
    # Base epsilon-greedy exploration probability for ML selection.
    ML_EPSILON_BASE: float = 0.10
    # Extra exploration added early in the run as (1 - progress) * this value,
    # so epsilon decays toward the base over time.
    ML_EPSILON_EARLY_BOOST: float = 0.15
    # Extra exploration on small instances. The final epsilon is clamped to
    # [0.02, 0.60].
    ML_EPSILON_SMALL_INSTANCE_BOOST: float = 0.05
    # Consecutive non-improving *selection* outcomes for one operator type after
    # which the ML selector temporarily falls back to the default selection.
    # Warm-up sampling is excluded from this counter (it runs on state copies).
    ML_STAGNATION_NO_IMPROVE_LIMIT: int = 18
    # Selector calls that the forced fallback stays active once triggered.
    #
    # Note the unit: this is *selector calls*, not ALNS iterations. The ML
    # operator is one of eleven destroy candidates, so a 106-iteration run
    # offers it only ~10-17 calls in total; a cooldown of 30 therefore lasted
    # longer than the entire search. 5 restores the intent ("pause briefly
    # after a stagnant streak") without locking the selector out.
    ML_FALLBACK_COOLDOWN_ITERS: int = 5
    # Extra warm-up samples to collect per iteration while in the warm-up phase.
    ML_WARMUP_PER_ITER: int = 2
    # Hard cap on the wall-clock share warm-up sampling may consume, as a
    # fraction of ``MAX_RUNTIME``.
    #
    # Warm-up stops when both buffers reach the takeover sample threshold, and
    # that threshold is reached in ~20 iterations on medium (measured 8-9% of a
    # 45 s budget, 4.0-4.7 s). Nothing bounded the *time* it might spend, so a
    # slow instance or a side whose sampling keeps failing could pay warm-up cost
    # for the rest of the run and never get a model-based decision back. The cap
    # only bites in that pathological case; measured medium runs stay below it.
    ML_WARMUP_BUDGET_FRACTION: float = 0.15
    # Include the operator's current (normalised) tuner parameters in the ML
    # feature vector. Without them the same one-hot row maps to different
    # improvements as the tuner moves the parameters, which is a multi-valued
    # target; ``False`` restores the pure one-hot encoding for A/B runs.
    ML_OPERATOR_PARAM_FEATURES: bool = True
    # Robust winsorising of the improvement target: values outside
    # ``median +/- k * MAD`` are clipped before fitting. Improvement labels
    # contain ``inf -> 1e6`` clamps, and a single such row dominates Ridge's
    # squared loss (measured repair holdout R2 as low as -253.7). A quantile
    # rule does not work at these sample sizes (at n=40 the 99th percentile is
    # interpolated to ~6e5 by the clamp itself); MAD adapts to the sample. 0
    # disables the clipping.
    ML_TARGET_CLIP_MAD_FACTOR: float = 8.0
    # What the repair side predicts. ``"classification"`` answers "will this
    # operator improve the objective at all"; ``"regression"`` predicts the
    # improvement magnitude (the old behaviour, kept for A/B runs).
    #
    # Measured with 5 contiguous time blocks on five real runs (medium/45 s):
    # regression R2 ranged from -0.3 down to -1.8e7 (never positive), while the
    # classifier's pooled out-of-fold AUC stayed in 0.65-0.77. The magnitude is
    # dominated by a handful of large repairs, so ranking operators by
    # "improves / does not improve" carries the signal that regression loses.
    ML_REPAIR_TARGET: str = "classification"
    # Pooled out-of-fold AUC the repair classifier must reach before takeover.
    # 0.5 is chance; the measured means were 0.65-0.77, so this is a floor
    # against a useless model rather than a bar the model has to stretch for.
    ML_CLASSIFIER_MIN_AUC: float = 0.55
    # Contiguous time blocks used for that estimate. A single chronological
    # holdout at n~45 scored only 11 rows and swung between AUC 0.54 and 0.86
    # on identical configurations, which is too noisy to gate on.
    ML_CLASSIFIER_CV_FOLDS: int = 5
    # Rows needed before the pooled AUC means anything. The generic training
    # floor is 12, and a 12-row / 3-fold estimate was measured at AUC 0.056 on
    # a run whose model was simply untrained: reporting that as a metric invites
    # reading noise as a result.
    ML_CLASSIFIER_MIN_SAMPLES: int = 24
    # Fits kept per side in the diagnostic history. One retrain is a single
    # measurement; a gate that passes once and fails every other time would look
    # identical in "last fit" statistics.
    ML_QUALITY_HISTORY_SIZE: int = 20
    # Consecutive failed fits that end an *established* takeover.
    #
    # Without hysteresis every retrain re-votes: measured over 30 fixed-iteration
    # runs, 79 repair fits passed 58 times (73%), and inside a single run the
    # AUC went 0.639 -> 0.528 -> 0.434 -> 0.462, so takeover switched on and off
    # four times. Two consecutive misses (not one) keeps a side in service
    # through a single noisy estimate while still dropping a model that has
    # genuinely degraded. 0 disables hysteresis (old behaviour: one miss exits).
    ML_QUALITY_EXIT_MISSES: int = 2
    # Which decision rule ML uses.
    #
    # ``"supervised"`` keeps the Ridge / logistic ranking that shipped through
    # M57. ``"bandit"`` switches to the contextual Thompson sampler in
    # ``bandit_selector.py``, which tilts the roulette instead of replacing it.
    # The supervised path measured worse than no model at all
    # (ml - basic = +9.18, t = 1.92), so the bandit is the intended replacement;
    # it stays opt-in until it has been measured on the same clean protocol.
    ML_SELECTOR_KIND: str = "bandit"
    # Pseudo-observations behind the bandit's shrinkage towards each arm's
    # global success rate. Higher keeps sparse cells closer to the arm mean.
    ML_BANDIT_PRIOR: float = 4.0
    # Trials before the bandit may tilt the wheel at all. Below this it returns
    # neutral weights, so an information-free bandit cannot distort the search.
    ML_BANDIT_MIN_TRIALS: int = 30
    # Wall-clock budget below which the bandit is not used at all: on short runs
    # it cannot collect enough trials to pay for itself, and the honest action is
    # to behave exactly like basic.
    ML_BANDIT_MIN_RUNTIME_SECONDS: float = 90.0
    # How far the bandit may pull each operator's weight away from the wheel's
    # own value. 1.0 on either side means "no opinion for this operator".
    #
    # The first bandit used (0.25, 4.0) - a 16x spread - and lost to basic by
    # +8.28 on five seeds. Diagnosis: the tilt was strong enough to override the
    # wheel's accumulated credit, which is what allows the search to walk the
    # deep chains (basic reached 787/778 on two seeds; every ML variant stalled
    # near 805). A nudge keeps the wheel authoritative and lets the bandit only
    # break near-ties.
    ML_BANDIT_TILT_LOW: float = 0.75
    ML_BANDIT_TILT_HIGH: float = 1.33
    # n-step credit assignment: how many recent decisions may share credit for
    # one new global best, the geometric decay per step back, and the peak bonus.
    ML_BANDIT_NSTEP_HORIZON: int = 20
    ML_BANDIT_NSTEP_DECAY: float = 0.9
    ML_BANDIT_NSTEP_BONUS: float = 0.6
    # Progress fraction before the bandit may tilt at all.
    #
    # Every losing run so far has the same shape: basic stumbles into a deep
    # basin (787.23 and 777.82 on two seeds) while every ML variant stalls near
    # 805. The deep basins are found *early*, when the SA temperature is still
    # high and the wheel is still nearly uniform; tilting there removes exactly
    # the diversity that finds them. Restricting the tilt to the late, refining
    # half of the run keeps the exploration phase identical to basic.
    ML_BANDIT_MIN_PROGRESS: float = 0.5
    # Tilt only after this many iterations without a new global best.
    #
    # The late-progress gate fixed most of the damage (ml-basic went +8.28 ->
    # +0.96, winning 3/5 seeds) but one seed still lost 18.4: basic found its
    # deep basin *late*, so a time-based gate still cut into it. Stagnation is
    # the actual condition ML should react to: while the search is still making
    # progress, any tilt is pure interference; once it stalls, a nudge is the
    # only thing that can help. 0 disables the trigger (tilt whenever ready).
    ML_BANDIT_STAGNATION_TRIGGER: int = 0
    # Acceptance-aggressiveness bandit (a second action space, not a retune of
    # operator selection).
    #
    # Operator choice affects one iteration and then decays. The annealing
    # temperature governs how far the incumbent may wander, so its effect lasts
    # for many iterations - evidence about it stays useful, which is what an
    # online learner needs. Decisions are taken once per ``interval`` iterations
    # and paid with the incumbent's gain over that window.
    #
    # ``ML_ACCEPT_ENABLED=False`` keeps the production schedule bit-identical.
    #
    # Measured OFF vs ON on the same four seeds at a fixed 120 iterations:
    # ON cost +3.23 / +4.28 / +10.60 / +19.50 (worse on 4 of 4). The mechanism
    # works (decisions are taken and paid per epoch) but the search it steers is
    # already well served by the production schedule. Kept implemented, default
    # off, so the measurement can be reproduced with one switch.
    ML_ACCEPT_ENABLED: bool = False
    ML_ACCEPT_INTERVAL: int = 15
    ML_ACCEPT_MULTIPLIERS: tuple[float, float, float] = (0.5, 1.0, 2.0)
    ML_ACCEPT_MIN_EPOCHS: int = 3
    # Inverse L2 strength for the repair classifier.
    ML_CLASSIFIER_C: float = 1.0
    # Repair side uses the operator-parameter interaction slots.
    #
    # Offline, dropping them on regression-era buffers looked neutral (AUC
    # 0.685 -> 0.704). Under the classification target that was wrong: on the
    # same seed the gate's AUC went 0.546 -> 0.722 (s42) and 0.341 -> 0.784
    # (s11) when the slots were present, with the s11 final cost 804.60 ->
    # 798.41. A logistic model needs the parameter to tell "regret repair at
    # k=2" from "regret repair at k=5"; without it those rows share a column.
    ML_REPAIR_PARAM_FEATURES: bool = True

    # --------------------------
    # Param tuner defaults (sync with ParamAutoTuner)
    # --------------------------
    # Step size for the tuner's parameter adjustment after each feedback event.
    TUNER_LEARNING_RATE: float = 0.1
    # Initial probability of sampling random parameters instead of exploiting
    # the learned baseline.
    TUNER_EXPLORATION_INITIAL: float = 0.3
    # (low, high) progress fractions that split the search into exploration,
    # exploitation and refinement phases.
    TUNER_PHASE_THRESHOLDS: tuple[float, float] = (
        0.3,
        0.7,
    )  # exploration/exploitation/refinement splits
    # Estimated instance scale below which the small-instance tuning profile is
    # applied.
    TUNER_SMALL_INSTANCE_SCALE_THRESHOLD: int = 250
    # Upper bound on the exploration rate for small instances.
    TUNER_SMALL_INSTANCE_EXPLORATION_CAP: float = 0.18
    # Multiplier applied to the tuner's sample requirement on small instances.
    TUNER_SMALL_INSTANCE_SAMPLE_SCALE: float = 0.35
    # Minimum number of performance observations before a parameter is adjusted,
    # so early noise cannot move the baseline.
    TUNER_MIN_HISTORY_FOR_ADJUST: int = 5
    # Maximum relative change applied to a parameter in a single update.
    TUNER_MAX_REL_STEP: float = 0.15
    # Non-improving iterations after which the tuner increases exploration.
    TUNER_STAGNATION_BOOST_THRESHOLD: int = 35
    # Size of that stagnation boost.
    TUNER_STAGNATION_BOOST_FACTOR: float = 0.35
    # Evidence gate for random parameter exploration.
    #
    # Exploration sampled parameters for 35-50% of operator calls for the whole
    # run, and those samples were **never written back**, so they taught the
    # tuner nothing - they only injected parameter noise into the search. Worse,
    # the directional update reads ``used - baseline``, so a random sample also
    # drags the baseline in a random direction. Exploration is now limited to the
    # first N uses of an operator, which is when it can actually seed evidence.
    # 0 disables the gate (old behaviour).
    TUNER_EXPLORATION_GATE_USES: int = 12
    # Cap on how many times one operator's baseline may be re-adjusted. Without
    # it the baseline keeps moving on every lucky iteration, so the parameter
    # "converges" only when the run ends. 0 disables the cap.
    TUNER_MAX_ADJUSTMENTS: int = 12
    # Regret: trust that ``veh_loading(commit=True)`` has already brought
    # ``s_ikt`` up to date, so recomputing the search objective does not need to
    # re-run ``validate()``/``compute_inventory()`` first.
    #
    # M64 measured one full search objective per pass, 7.56 s of a 40.18 s call.
    # The formula is unchanged - only the redundant inventory re-derivation is
    # skipped - and every use is verified against a full ``objective()`` by
    # ``local-only/probes/_m66_equivalence.py`` (zero disagreements required).
    # ``False`` restores the pre-M66 behaviour.
    REGRET_TRUST_CACHED_OBJECTIVE: bool = True

    # --------------------------
    # Operator defaults (params + ranges)
    #
    # "params" are the baseline values the operator is registered with; the
    # online tuner starts from them and moves a parameter only if "ranges" gives
    # it bounds. An empty "ranges" entry means the parameter is fixed.
    # ParamAutoTuner._init_default_operators reads this mapping, and
    # get_operator_default() serves the per-operator lookups in the optimizer.
    #
    # The keys must match the operator names ALNSOptimizer registers (see
    # ml_operator_selector.DESTROY_OPERATOR_NAMES / REPAIR_OPERATOR_NAMES).
    # --------------------------
    OPERATOR_DEFAULTS: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            # --- Destroy operators ---
            #
            # ``degree`` (the fraction of the fleet a destroy operator removes)
            # is deliberately **not** tunable here. Measured on large with the
            # progress-cooled acceptance (M43): halving every degree raised the
            # iteration count by ~70% and made the final cost worse
            # (2040.26 -> 2043.16 over three seeds). Tuning it therefore spends
            # budget on a dimension with no measured payoff. The operators keep
            # their registered degree from DESTROY_DEFAULTS; only the *other*
            # parameters below are exposed to the tuner.
            "random_removal": {
                "params": {"degree": 0.25},
                "ranges": {},
            },
            "shaw_removal": {
                "params": {"degree": 0.3},
                "ranges": {},
            },
            # alpha/beta/gamma: weights of the time / demand / inventory
            # components of the clustering distance.
            # k_clusters: number of KMeans clusters used to group vehicles.
            "periodic_shaw_removal": {
                "params": {
                    "degree": 0.3,
                    "alpha": 0.4,
                    "beta": 0.3,
                    "gamma": 0.3,
                    "k_clusters": 3,
                },
                "ranges": {
                    "alpha": (0.2, 0.6),
                    "beta": (0.1, 0.5),
                    "gamma": (0.1, 0.5),
                    "k_clusters": (2, 10),
                },
            },
            "path_removal": {
                "params": {"degree": 0.5},
                "ranges": {},
            },
            "cross_period_dealer_removal": {
                "params": {"degree": 0.45},
                "ranges": {},
            },
            # window_ratio: length of that day window as a fraction of the horizon.
            "period_range_removal": {
                "params": {"degree": 0.35, "window_ratio": 0.30},
                "ranges": {"window_ratio": (0.15, 0.7)},
            },
            # value_bias: exponent that skews the sampling toward the
            # worst-ranked vehicles.
            "worst_removal": {
                "params": {"degree": 0.25, "value_bias": 2.0},
                "ranges": {"value_bias": (1.0, 3.0)},
            },
            # No tunable parameters: it rolls back the vehicles that cause a
            # negative inventory violation.
            "infeasible_removal": {"params": {}, "ranges": {}},
            "surplus_inventory_removal": {
                "params": {"degree": 0.25},
                "ranges": {},
            },
            "inventory_level_removal": {
                "params": {"degree": 0.25},
                "ranges": {},
            },
            # --- Repair operators ---
            # demand_weight / stock_weight: relative preference for closing a
            # demand gap versus using the nearest available stock.
            "greedy_repair": {
                "params": {"demand_weight": 0.8, "stock_weight": 0.2},
                "ranges": {"demand_weight": (0.0, 1.0), "stock_weight": (0.0, 1.0)},
            },
            # Cross-period same-route consolidation. Its own work budget is
            # configured by CROSS_PERIOD_CONSOLIDATION_* below, not here.
            "cross_period_consolidation_repair": {"params": {}, "ranges": {}},
            # max_iter: cap on the local type/loading improvement passes.
            "local_search_repair": {"params": {"max_iter": 10}, "ranges": {}},
            # No tunable parameters: it orders plants by available inventory.
            "inventory_balance_repair": {"params": {}, "ranges": {}},
            # k: regret order. topN: candidate shortlist size before the regret
            # score is computed.
            "regret_based_repair": {
                "params": {"k": 2, "topN": 6},
                "ranges": {"k": (1, 5), "topN": (1, 20)},
            },
            # batch_size: demands handled per batch. max_iter: batch rounds.
            # timeout: optional per-call time cap in seconds (None = no cap).
            "smart_batch_repair": {
                "params": {"batch_size": 10, "max_iter": 10, "timeout": None},
                "ranges": {"batch_size": (5, 50)},
            },
            # No tunable parameters: it repairs strictly in period order.
            "sequential_repair": {"params": {}, "ranges": {}},
            # lookahead_horizon: how many future periods the insertion scoring
            # evaluates (t .. t+h).
            "lookahead_repair": {
                "params": {"lookahead_horizon": 2},
                "ranges": {"lookahead_horizon": (1, 4)},
            },
            # No tunable parameters: it prefers merging small vehicles into
            # larger ones.
            "fleet_consolidation_repair": {"params": {}, "ranges": {}},
        }
    )

    # --------------------------
    # Flat mapping used by ALNSConfig.get_destroy_params() and by the optimizer
    # when it registers the destroy operators. Values must stay consistent with
    # the matching OPERATOR_DEFAULTS entries above.
    # --------------------------
    DESTROY_DEFAULTS: dict[str, Any] = field(
        default_factory=lambda: {
            # Removal degree per operator (fraction of vehicles removed).
            "random_removal_degree": 0.25,
            "shaw_removal_degree": 0.3,
            # Bundle of periodic-Shaw clustering weights and cluster count.
            "periodic_shaw_params": {
                "degree": 0.3,
                "alpha": 0.4,
                "beta": 0.3,
                "gamma": 0.3,
                "k_clusters": 3,
            },
            "cross_period_dealer_removal_degree": 0.45,
            # Removal degree and day-window ratio of period_range_removal.
            "period_range_removal_degree": 0.35,
            "period_range_removal_window_ratio": 0.30,
            "inventory_level_removal_degree": 0.25,
        }
    )

    # Flat mapping returned by ALNSConfig.get_repair_params() and read by the
    # optimizer when it registers the repair operators. An empty dict marks an
    # operator with no tunable parameters.
    REPAIR_DEFAULTS: dict[str, Any] = field(
        default_factory=lambda: {
            "greedy_repair": {"demand_weight": 0.8, "stock_weight": 0.2},
            "local_search_repair": {"max_iter": 10},
            "inventory_balance_repair": {},
            "smart_batch_repair": {"max_iter": 10, "batch_size": 10, "timeout": None},
            "regret_based_repair": {"k": 2, "topN": 6, "time_limit": 10.0},
            "cross_period_consolidation_repair": {},
            "sequential_repair": {},
            "lookahead_repair": {"lookahead_horizon": 2},
            "fleet_consolidation_repair": {},
        }
    )

    # Penalty for deferred delivery days in lookahead repair.
    # Larger values favor fulfillment in the current period.
    # Multiplied by the number of periods a candidate insertion is deferred and
    # subtracted from its score, so a higher value prefers early delivery.
    LOOKAHEAD_REPAIR_DELAY_PENALTY: float = 0.5

    # --------------------------
    # Repair post-processing knobs
    # --------------------------
    # Run same-route consolidation after every completed repair.
    REPAIR_ENABLE_POST_CONSOLIDATION: bool = True
    # Maximum number of vehicle pairs the post-repair consolidation examines,
    # which bounds the cost of that pass.
    REPAIR_POST_CONSOLIDATION_MAX_PAIRS: int = 300

    # Work budget for one cross-period consolidation call. Candidates are
    # scored incrementally; only the selected move is applied to the operator's
    # private state. These bounds apply to every in-search invocation; the final
    # pre-output pass may spend slightly longer (see below).
    # Seconds one in-search consolidation call may spend.
    CROSS_PERIOD_CONSOLIDATION_TIME_LIMIT: float = 5.0
    # Maximum accepted consolidation moves per in-search call.
    CROSS_PERIOD_CONSOLIDATION_MAX_MOVES: int = 25
    # Looser budget for the single consolidation pass run before output.
    FINAL_CROSS_PERIOD_CONSOLIDATION_TIME_LIMIT: float = 20.0
    # Maximum accepted consolidation moves for that final pass.
    FINAL_CROSS_PERIOD_CONSOLIDATION_MAX_MOVES: int = 60

    # Optional same-route consolidation of the best solution before output.
    FINAL_BEST_ENABLE_CONSOLIDATION: bool = True
    # Vehicle-pair budget for that final consolidation pass.
    FINAL_BEST_CONSOLIDATION_MAX_PAIRS: int = 800

    # --------------------------
    # Best-solution fleet preference (used in callbacks/solution ranking)
    # --------------------------
    # When two solutions are close in objective, prefer the one with fewer
    # vehicles. This drives which incumbent the callbacks keep.
    ENABLE_VEHICLE_COUNT_PRIORITY: bool = True
    # How much worse the objective may be while still accepting a solution with
    # fewer vehicles; the effective allowance is max(abs, rel * |current|),
    # scaled by the phase factor described in callbacks._is_fleet_better.
    VEHICLE_PRIORITY_OBJ_REL_TOL: float = 0.08
    # Absolute counterpart of that allowance.
    VEHICLE_PRIORITY_OBJ_ABS_TOL: float = 12.0
    # Two objectives within this relative tolerance are treated as equal, which
    # then activates the tie-breakers below (vehicle count, load ratio, fixed
    # cost and share of large vehicles, in that order).
    FLEET_NEAR_EQUAL_OBJ_REL_TOL: float = 0.01
    # Absolute counterpart of that equality tolerance.
    FLEET_NEAR_EQUAL_OBJ_ABS_TOL: float = 1.0
    # Minimum load-ratio difference required by the load-ratio tie-breaker.
    FLEET_LOAD_RATIO_MARGIN: float = 1e-4
    # Minimum fixed-cost difference required by the fixed-cost tie-breaker.
    FLEET_FIXED_COST_MARGIN: float = 1e-6
    # Minimum difference in the share of large vehicles required by the final
    # tie-breaker. All three margins keep numerical noise from flipping the
    # comparison.
    FLEET_LARGE_RATIO_MARGIN: float = 1e-6

    # --------------------------
    # Explicit fleet-size penalty (in objective function)
    # --------------------------
    # Add a fleet-size penalty to the search objective. Off by default because
    # the mathematical objective does not contain such a term; enable it only
    # as extra search guidance, never for reported results.
    ENABLE_FLEET_SIZE_PENALTY: bool = False
    # Penalty added per vehicle above FLEET_SIZE_BASELINE.
    FLEET_SIZE_PENALTY_PER_VEHICLE: float = 15.0  # Cost penalty per additional vehicle
    # Reference fleet size; vehicles above it incur the penalty above.
    FLEET_SIZE_BASELINE: int = (
        25  # Reference fleet size; excess vehicles over this incur penalty
    )

    # --------------------------
    # Search objective decomposition
    #
    # The search objective is a weighted blend of the unmet-demand penalty, an
    # optional underload-gap term and a fleet-size term. The weights below make
    # the blend shift as the search progresses, so early iterations chase
    # feasibility and later ones consolidate the fleet.
    # --------------------------
    # Include the unmet-demand penalty term. This is the only extra guidance
    # term kept by default; it keeps destroyed states informative.
    SEARCH_USE_DEMAND_PENALTY: bool = True
    # Include the volume-based underload gap term. Disabled by default because
    # it is not part of the mathematical objective.
    SEARCH_USE_UNDERLOAD_GAP_PENALTY: bool = False
    # Use different demand / underload / fleet weights in each of the three
    # progress phases; when off, only the phase-3 weights are used.
    SEARCH_PHASED_VEHICLE_PRIORITY: bool = True
    # Progress fractions that end phase 1 and phase 2 (0-1).
    SEARCH_PHASE1_END_PROGRESS: float = 0.45
    # Progress fraction that ends phase 2 and starts phase 3.
    SEARCH_PHASE2_END_PROGRESS: float = 0.80
    # Phase 1 (early): fleet weight dominates, demand weight is low.
    SEARCH_PHASE1_DEMAND_WEIGHT: float = 0.35
    # Phase 1 underload weight (applies only when the underload term is enabled).
    SEARCH_PHASE1_UNDERLOAD_WEIGHT: float = 0.80
    # Phase 1 fleet-size weight; the dominant term early on.
    SEARCH_PHASE1_FLEET_WEIGHT: float = 4.00
    # Phase 2 (middle): demand and fleet weights are balanced.
    SEARCH_PHASE2_DEMAND_WEIGHT: float = 0.80
    # Phase 2 underload weight.
    SEARCH_PHASE2_UNDERLOAD_WEIGHT: float = 1.00
    # Phase 2 fleet-size weight.
    SEARCH_PHASE2_FLEET_WEIGHT: float = 2.00
    # Phase 3 (late): the model terms dominate so the reported objective is
    # what the search actually optimizes.
    SEARCH_PHASE3_DEMAND_WEIGHT: float = 1.00
    # Phase 3 underload weight.
    SEARCH_PHASE3_UNDERLOAD_WEIGHT: float = 1.00
    # Phase 3 fleet-size weight.
    SEARCH_PHASE3_FLEET_WEIGHT: float = 1.00
    # Feasibility-first shaping: while demand is unmet, boost the demand term
    # and damp the fleet and underload terms so the search repairs feasibility
    # before optimizing cost. Mainly relevant for non-strict runs.
    SEARCH_FEASIBILITY_FIRST: bool = True
    # Multiplier applied to the demand weight while demand is unmet.
    SEARCH_FEASIBILITY_DEMAND_BOOST: float = 6.0
    # Damping factor applied to the fleet-size weight in that state.
    SEARCH_FEASIBILITY_FLEET_SCALE_WHEN_UNMET: float = 0.15
    # Damping factor applied to the underload weight in that state.
    SEARCH_FEASIBILITY_UNDERLOAD_SCALE_WHEN_UNMET: float = 0.25
    # Keep search and actual objective values in the same unit. The only
    # additional guidance term is the finite penalty for unmet demand.
    # 1.0 means "no rescaling"; keep it at 1.0 so temperatures and tolerances
    # stay comparable to the reported objective.
    SCALE_FACTOR: float = 1.0

    # --------------------------
    # Inventory holding cost
    # --------------------------
    # Optional model term for the tight-capacity variant family.  It is charged
    # once per standard-size unit held at the end of a positive period:
    #
    #     rate * sum_{plant, sku, day>0} max(0, s_ikt) * sku_size
    #
    # The default is exactly zero, so archived instances keep their historical
    # objective and the extra traversal is skipped entirely.  Enable a positive
    # value only on the tight-capacity variants; it changes the meaning of the
    # objective and therefore invalidates comparisons with the archived family.
    INVENTORY_HOLDING_COST: float = 0.0

    # --------------------------
    # Large-jump destroy operator tuning
    # (cross_period_dealer_removal)
    # --------------------------
    # Scale the removal degree with search progress instead of using it as is.
    CROSS_PERIOD_REMOVAL_PHASED: bool = True
    # Degree multiplier applied early in the search (bigger jumps).
    CROSS_PERIOD_REMOVAL_EARLY_BOOST: float = 1.35
    # Degree multiplier applied late in the search (smaller jumps).
    CROSS_PERIOD_REMOVAL_LATE_SHRINK: float = 0.75
    # Non-improving iterations after which the stagnation boost applies.
    CROSS_PERIOD_REMOVAL_STAGNATION_WINDOW: int = 80
    # Degree multiplier applied while the search is stagnating.
    CROSS_PERIOD_REMOVAL_STAGNATION_BOOST: float = 1.20

    # --------------------------
    # Regret operator practical caps (safeguards for production)
    # These limit candidate branching / caching granularity.
    # --------------------------
    # Hard cap on the regret candidate shortlist size, whatever the request was.
    REGRET_MAX_TOPN: int = 12
    # Hard cap on the regret order k.
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
    # taking a candidate further (delta estimate, else full simulation).
    #
    # Direction: a *higher* threshold is stricter (fewer candidates pass);
    # lowering it admits more. At the 1e-3 default the threshold is effectively
    # a sign test - "estimated improvement is positive" - because the estimator
    # returns ``(benefit - cost - penalty) * SCALE_FACTOR`` with magnitudes far
    # above 1e-3. Measured on medium/dataset_13 (2026-10-23): 98.6% of
    # candidates pass it, so this gate filters by sign, not by magnitude.
    REGRET_FULL_SIM_THRESHOLD: float = 0.001

    # Reuse each demand's candidate shortlist across the while passes of one
    # ``regret_based_repair`` call.
    #
    # A pass commits a single vehicle, so only demands on that vehicle's SKU
    # can see availability move; everything else re-enumerates an identical
    # candidate set. Measured on medium/dataset_13: 98.4% of demand x pass
    # pairs have unchanged inputs, and their shortlists match element by
    # element (set, order and truncation) - reuse is bit-for-bit equivalent
    # (verified: 429 passes, identical commit at every one, identical final
    # state, 13.29 s -> 9.23 s for that call).
    #
    # Only the shortlist is cached. The improvement scores are recomputed
    # every pass: reusing them was measured to drift by 1.5e-8..3.0e-8
    # (``prev_obj - (prev_obj + delta)`` rounding), which is not equivalent.
    #
    # **Off by default.** The operator's own 10 s budget was implicitly
    # calibrated against the old per-pass cost, so a ~3x cheaper pass turns one
    # repair into an exhaustive greedy fill: at a fixed 150 s run budget the
    # iteration count moved 181->227, 324->122, 252->192 on seeds 42/7/123,
    # and final cost moved -4.7..+52.7 (one seed clearly worse, none clearly
    # better). Rebalancing belongs to the next round - cap commits per call or
    # normalise the operator reward by time - before this is enabled.
    REGRET_SHORTLIST_CACHE: bool = False

    # Hard cap on how many vehicles one ``regret_based_repair`` call may commit
    # before returning control to the ALNS loop. ``0`` means "no bound: run
    # until ``time_limit``", which is the historical behaviour.
    #
    # Why this exists: the shortlist cache makes a pass ~1.6x cheaper, and at a
    # fixed wall-clock budget the operator simply did more work per call -
    # measured median commits per call 124.5 -> 341 (2.7x), p90 274 -> 437, at
    # 27.7 -> 43.7 commits/s. That is not a quality improvement: one "repair"
    # turns into an exhaustive greedy fill and the destroy/repair alternation
    # that the search relies on disappears.
    #
    # Bound the work instead of the clock so a call does the same amount of
    # repair whether or not the cache is warm.
    REGRET_MAX_COMMITS_PER_CALL: int = 0

    # --------------------------
    # Logging / persistence
    # --------------------------
    # Directory where the script entry point writes alns_optimization.log. The
    # CLI overrides it with --log-dir and the experiment runners point it at the
    # per-run directory.
    LOG_DIR: str | None = "logs-alns"

    # --------------------------
    # Lexicographic best-solution rule
    #
    # A stricter alternative to ENABLE_VEHICLE_COUNT_PRIORITY: once the search
    # is far enough along, a candidate with fewer vehicles is accepted whenever
    # its objective stays inside the tolerance below. These four settings are
    # read by callbacks._is_fleet_better and were previously only reachable
    # through getattr defaults; they are declared here so they are visible.
    # --------------------------
    # Enable the lexicographic rule (off by default).
    ENABLE_PHASED_LEXICOGRAPHIC_BEST: bool = False
    # Progress fraction from which the rule applies.
    PHASED_LEXICOGRAPHIC_START_PROGRESS: float = 0.35
    # Allowed objective worsening: max(abs, rel * |current|).
    PHASED_LEXICOGRAPHIC_OBJ_REL_TOL: float = 0.01
    # Absolute counterpart of that allowance.
    PHASED_LEXICOGRAPHIC_OBJ_ABS_TOL: float = 2.0

    # --------------------------
    # Promoted settings that used to be read only through getattr defaults
    #
    # Every field below was previously read as
    # ``getattr(ALNSConfig, "NAME", <default>)`` without being declared here.
    # Those reads still worked, because the inline default supplied the value,
    # but the knobs were invisible: nothing in this file hinted that they
    # existed and the fallback value was the only documentation. They are
    # declared now with exactly the default each call site used, so behaviour is
    # unchanged while every value becomes discoverable, typed and settable from
    # config JSON.
    # --------------------------

    # --- Fast mode / validation throttling ---
    # Coarse "fast" screening profile. When on, the callbacks skip the expensive
    # validation guard and the ML retraining/warm-up work. It is off in every
    # published run; enable it only for quick exploratory sweeps, because it
    # also disables ML operator training while it is active.
    ENABLE_FAST_ALNS_MODE: bool = False
    # Iterations to complete before the runtime validation guard starts its
    # periodic validation. 0 validates from the first iteration.
    VALIDATE_WARMUP_ITERS: int = 0
    # Validation period used while ENABLE_FAST_ALNS_MODE is on. 1 means "validate
    # every iteration", i.e. no throttling.
    FAST_VALIDATE_EVERY_N_ITERS: int = 1
    # Validation period used in normal mode. 1 means "validate every iteration".
    VALIDATE_EVERY_N_ITERS: int = 1

    # --- Acceptance diagnostics ---
    # Emit the periodic acceptance-statistics block from the iteration callback.
    ENABLE_ACCEPT_STATS_LOG: bool = True
    # Iterations between two acceptance-statistics log lines.
    ACCEPT_STATS_LOG_INTERVAL: int = 200
    # Absolute tolerance below which an inventory inconsistency is treated as
    # numerical noise and left out of the recorded inconsistency report.
    RECORD_INCONSISTENCY_EPS: float = 1e-6

    # --- Initial-solution time budget ---
    # Lower bound of the wall-clock budget (seconds) granted to the initial
    # solution construction.
    MIN_INIT_TIMEOUT: float = 30.0
    # Upper bound of that budget. The effective budget scales with the number of
    # outstanding demands and is clamped into [MIN_INIT_TIMEOUT, MAX_INIT_TIMEOUT].
    MAX_INIT_TIMEOUT: float = 120.0
    # Seconds added to the initial-solution budget per outstanding demand.
    INIT_TIMEOUT_PER_DEMAND: float = 0.05
    # Deterministic repair passes the construction pipeline runs while pushing
    # the initial solution into strict feasibility.
    INITIAL_REPAIR_MAX_ITER: int = 5

    # --- Lookahead repair weights ---
    # Weight of the next-period demand term in the lookahead insertion score.
    LOOKAHEAD_REPAIR_FUTURE_WEIGHT: float = 0.35
    # Fraction of the vehicle capacity held back as a safety floor before a
    # candidate insertion is accepted.
    LOOKAHEAD_REPAIR_SAFETY_FLOOR_RATIO: float = 0.10
    # Score bonus granted when a candidate reuses a vehicle that is already open.
    LOOKAHEAD_REPAIR_REUSE_BONUS: float = 0.10

    # --- Fleet classification ---
    # Capacity quantile above which a vehicle type counts as a "large vehicle" in
    # the fleet-composition tie-breakers.
    FLEET_LARGE_CAPACITY_QUANTILE: float = 0.67

    # --- Destroy-operator diagnostics ---
    # Sampling rate of the per-operator JSON dumps written by destroy_utils.
    # 1 dumps every call; larger values keep only a sample.
    OPJSON_SAMPLING_RATE: int = 1

    # --- Machine-learning operator selector hyper-parameters ---
    # L2 penalty of the ridge fallback model trained on operator outcomes.
    ML_RIDGE_ALPHA: float = 1.0
    # Sample count from which the random-forest model is trained instead of the
    # ridge fallback.
    ML_USE_RF_THRESHOLD: int = 50
    # Number of trees in that random-forest model.
    ML_RF_N_ESTIMATORS: int = 50
    # Maximum tree depth of that random-forest model.
    ML_RF_MAX_DEPTH: int = 10

    # --- Reporting and debugging ---
    # Dots-per-inch of the matplotlib figures written by the reporting step.
    REPORT_DPI: int = 150
    # Print the parameter tuner's internal adjustment trace. Debugging only.
    TUNER_DEBUG: bool = False

    # --------------------------
    # Helper instance methods
    # --------------------------
    def get_destroy_params(self) -> dict[str, Any]:
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
            "inventory_level_removal_degree": self.DESTROY_DEFAULTS.get(
                "inventory_level_removal_degree", 0.25
            ),
        }

    def get_repair_params(self) -> dict[str, Any]:
        """Return a compact dict of commonly used repair defaults."""
        return self.REPAIR_DEFAULTS.copy()

    def get_operator_default(self, op_name: str) -> dict[str, Any]:
        """Return the operator default structure {params, ranges} for a given operator."""
        return self.OPERATOR_DEFAULTS.get(op_name, {"params": {}, "ranges": {}})

    def get_tuner_operator_defaults(self) -> dict[str, dict[str, Any]]:
        """
        Return a mapping suitable for ParamAutoTuner._init_default_operators:
        {op_name: {"params":..., "ranges":...}, ...}
        """
        return self.OPERATOR_DEFAULTS.copy()

    def update_from_dict(self, cfg: dict[str, Any]):
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

    def to_dict(self) -> dict[str, Any]:
        """Serialize the main configuration to a plain dict (shallow for convenience)."""
        out = {}
        for name, val in self.__dict__.items():
            out[name] = val
        return out


# Module-level default config instance for easy import/use
default_config = ALNSConfig()
