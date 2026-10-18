"""
alns_optimizer.py
High-level controller and result post-processing for the ALNS workflow.

Responsibilities:
1. Manage input/output paths and data loading through ``load_data``.
2. Generate and validate the initial solution in ``create_initial_solution``.
3. Register destroy and repair operators.
4. Configure ALNS selection, acceptance, stopping rules, and iteration execution.
5. Write CSV results and generate statistics and visual reports.
6. Support optional ``ParamAutoTuner`` and ``MLOperatorSelector`` components.

Main public interface:
  - ``ALNSOptimizer``: load_data, create_initial_solution, setup_alns,
    run_optimization.
  - ``is_file_in_use(file_path)``: best-effort file lock check that may be
    platform-dependent.

Maintenance notes:
  * Keep imports grouped by standard library, third-party, and local packages.
  * Import plotting dependencies lazily in functions that need them.
  * Prefer atomic writes for append-sensitive outputs when concurrency matters.
  * Derive roulette-wheel operator counts from actual registrations.
  * Prefer public methods over private attributes where practical.
  * Guard optional DataFrame columns and empty tracker structures.
  * Keep logger usage for system errors and LogPrinter for user-facing output.

Apply behavioral changes incrementally and run tests after each step.
"""

from __future__ import annotations

import builtins
import csv
import datetime
import io
import logging
import os
import time
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.random as rnd
import pandas as pd

from .accept import DemandConstraintAccept
from .alns_config import default_config as ALNSConfig

# Package-internal imports used by the optimizer
from .alnsopt import SolutionState, initial_solution
from .alnstrack import ALNSTracker
from .callbacks import register_callbacks
from .combined_stopping import create_standard_combined_criterion
from .destroy_operators import (
    create_cross_period_dealer_removal,
    create_inventory_level_removal,
    create_path_removal,
    create_period_range_removal,
    create_periodic_shaw_removal,
    # wrappers / factories
    create_random_removal,
    create_shaw_removal,
    create_surplus_inventory_removal,
    create_worst_removal,
    infeasible_removal,
    state_structure_signature,
    wrap_operator_no_args,
)
from .InputDataALNS import DataALNS
from .ml_destroy_selection import ml_based_destroy
from .ml_operator_selector import MLOperatorSelector
from .ml_repair_selection import ml_based_repair
from .optutility import LogPrinter

# visualization functions are imported lazily in _generate_reports / startup plotting
# to reduce module import overhead (matplotlib/pandas are heavy). They will be imported
# inside those functions when actually needed.
from .param_tuner import ParamAutoTuner
from .repair_operators import (
    FleetConsolidationRepairOperator,
    create_cross_period_consolidation_repair,
    create_local_search_repair,
    create_lookahead_repair,
    create_regret_repair,
    create_sequential_repair,
    create_smart_batch_repair,
    cross_period_consolidation_repair,
    greedy_repair,
    inventory_balance_repair,
    # wrapper factories for repair operators
    wrap_repair_no_args,
)
from .repair_utils import consolidate_route_vehicles
from .swallowed import record_swallowed, report_swallowed

# For type checking only (avoid runtime import when modules are optional)
if TYPE_CHECKING:
    from .ml_operator_selector import MLOperatorSelector
    from .param_tuner import ParamAutoTuner

# Third-party ALNS imports
from alns import ALNS
from alns.accept import SimulatedAnnealing
from alns.Outcome import Outcome
from alns.Result import Result
from alns.select import SegmentedRouletteWheel


class NoOpAwareSegmentedRouletteWheel(SegmentedRouletteWheel):
    """Roulette wheel that does not reward unchanged destroy/repair actions."""

    def update(self, cand, d_idx, r_idx, outcome):
        destroy_outcome = outcome
        repair_outcome = outcome
        if outcome == Outcome.ACCEPT:
            if bool(getattr(cand, "_destroy_noop", False)):
                destroy_outcome = Outcome.REJECT
            if bool(getattr(cand, "_repair_noop", False)):
                repair_outcome = Outcome.REJECT

        self._d_seg_weights[d_idx] += self._scores[int(destroy_outcome)]
        self._r_seg_weights[r_idx] += self._scores[int(repair_outcome)]


class _NoOpFlagPreservingRepair:
    """Carry destroy no-op state through a repair operator."""

    def __init__(self, operator: Any) -> None:
        self._operator = operator
        self.__name__ = getattr(operator, "__name__", "repair_operator")

    def __call__(self, state: Any, rng: Any, **kwargs: Any) -> Any:
        destroy_noop = bool(getattr(state, "_destroy_noop", False))
        before = state_structure_signature(state) if destroy_noop else None
        result = self._operator(state, rng, **kwargs)
        if destroy_noop and result is not None:
            try:
                result._destroy_noop = True
                if before is not None:
                    result._repair_noop = (
                        before == state_structure_signature(result)
                    )
            except (AttributeError, TypeError):
                pass
        return result


@contextmanager
def suppress_noisy_operator_stdout(enabled: bool = True):
    """Suppress high-volume operator prints while preserving key progress logs."""
    if not enabled:
        yield
        return

    noisy_tags = ("[OPLOG]", "[DIAG]", "[OPJSON]")
    original_print = builtins.print

    def filtered_print(*args, **kwargs):
        try:
            msg = " ".join(str(a) for a in args)
        except Exception:
            msg = ""
        if any(tag in msg for tag in noisy_tags):
            return
        return original_print(*args, **kwargs)

    builtins.print = filtered_print
    try:
        yield
    finally:
        builtins.print = original_print


# Config constants referenced by optimizer (use getattr for defensive access)
SEED = ALNSConfig.SEED
DATASET_TYPE = ALNSConfig.DATASET_TYPE
DATASET_IDX = ALNSConfig.DATASET_IDX

logger = logging.getLogger(__name__)


def compute_dynamic_iteration_cap(config: Any = None) -> int:
    """
    Size the runtime-derived iteration ceiling for a budgeted search.

    The stop rule already honours ``MAX_RUNTIME``; this ceiling exists so that
    ``MaxIterations`` does not truncate that budget too early. The result is also
    copied onto the optimizer and the tracker, where it becomes the denominator
    of ``progress`` in ``accept._resolve_progress`` - and ``progress`` gates the
    phase-dependent policies (actual-priority acceptance, late fleet priority,
    light intensification, elite relinking, ML takeover).

    That coupling is why ``DYNAMIC_MAX_ITERATIONS_PER_SECOND`` must be calibrated
    to real throughput: an unrealistically large value inflates the ceiling, so
    ``progress`` stays near zero and those policies never activate. If the
    ceiling is far above what the runtime budget can execute, the guarantees the
    thresholds encode (for example "the last 30% of the budget") no longer hold.

    ``max(..., 1.0)`` keeps zeroed configuration pins behaving as "no dynamic
    widening", which is how the tests pin the budget.
    """
    config = ALNSConfig if config is None else config
    runtime_budget = float(getattr(config, "MAX_RUNTIME", 900))
    base_iter_cap = int(getattr(config, "MAX_ITERATIONS", 300))
    iter_per_second = float(
        getattr(config, "DYNAMIC_MAX_ITERATIONS_PER_SECOND", 5.0)
    )
    safety_factor = float(
        getattr(config, "DYNAMIC_MAX_ITERATION_SAFETY_FACTOR", 1.5)
    )
    buffer_iters = int(getattr(config, "DYNAMIC_MAX_ITERATION_BUFFER", 200))
    return int(
        max(
            base_iter_cap,
            runtime_budget * max(iter_per_second, 1.0) * max(safety_factor, 1.0),
        )
    ) + max(0, buffer_iters)


# Utility: check if file is in use (moved here to avoid circular imports)
def is_file_in_use(file_path):
    """
    Return whether a file is currently locked by another process.
    """
    if not os.path.exists(file_path):
        return False
    try:
        with open(file_path, "a"):
            pass
        return False
    except PermissionError:
        return True


def safe_state_snapshot(state: Any) -> Any:
    if state is None:
        return None
    try:
        if hasattr(state, "copy") and callable(getattr(state, "copy")):
            return state.copy()
    except Exception:
        pass
    try:
        import copy as _copy

        return _copy.deepcopy(state)
    except Exception:
        return state


class ALNSOptimizer:
    """High-level controller for the complete ALNS optimization workflow."""

    def __init__(self, input_file_loc: Any = None, output_file_loc: Any = None):
        """
        Initialize the ALNS optimizer.

        Parameters:
        -----------
        input_file_loc : str, optional
            Input file location. Uses the default directory when omitted.
        output_file_loc : str, optional
            Output file location. Uses the default directory when omitted.
        """
        # Set default paths.
        #
        # Defaults are resolved against the working directory, matching the CLI
        # (`--data-dir` / `--output-dir`), ``alns.main.run_model`` and the
        # documented layout: `datasets/multiple-periods/<scale>` sits in the
        # project checkout, not inside the installed package. Resolving them
        # relative to this module pointed inside `src/` and therefore at a
        # directory that does not exist.
        if input_file_loc is None or output_file_loc is None:
            project_root = Path.cwd()

            if input_file_loc is None:
                input_file_loc = (
                    project_root / "datasets" / "multiple-periods" / DATASET_TYPE
                )
            if output_file_loc is None:
                output_file_loc = (
                    project_root / "OutPut-ALNS" / "multiple-periods" / DATASET_TYPE
                )

        self.input_file_loc = Path(input_file_loc)
        self.output_file_loc = Path(output_file_loc)

        # Initialize the log printer.
        self.log_printer = LogPrinter(time.time())

        # Initialize data and optimization state.
        self.data: DataALNS | None = None
        self.best_solution: SolutionState | None = None
        self.best_feasible_state: SolutionState | None = None
        self.best_feasible_obj: float = float("inf")
        self.result: Result | None = None
        self.tracker: ALNSTracker | None = None

        # Initialize optional parameter tuner and ML selector.
        self.param_tuner: ParamAutoTuner | None = None
        self.ml_selector: MLOperatorSelector | None = None

        # Random number generator.
        self.rng = rnd.default_rng(seed=SEED)
        # Track registered operator counts for selector configuration.
        self._num_registered_destroy_ops: int = 0
        self._num_registered_repair_ops: int = 0
        self._iteration: int = 0
        self._max_iterations: int = int(getattr(ALNSConfig, "MAX_ITERATIONS", 0))
        self.progress: float = 0.0
        self.acceptance = None
        self._background_executor: Any = None
        self._ml_retrain_future: Any = None
        self._incumbent_state_snapshot: SolutionState | None = None

    def get_iteration(self) -> int:
        return int(getattr(self, "_iteration", 0))

    def is_param_tuner_enabled(self) -> bool:
        return bool(getattr(ALNSConfig, "ENABLE_PARAM_TUNER", False))

    def is_ml_enabled(self) -> bool:
        return bool(getattr(ALNSConfig, "ENABLE_ML_OPERATORS", False))

    def increment_iteration(self) -> int:
        self._iteration = int(getattr(self, "_iteration", 0)) + 1
        return self._iteration

    def get_max_iterations(self) -> int:
        return int(
            getattr(self, "_max_iterations", getattr(ALNSConfig, "MAX_ITERATIONS", 0))
        )

    def set_max_iterations(self, max_iterations: int):
        self._max_iterations = int(max_iterations)

    def load_data(self, dataset_name: str) -> bool:
        """
        Load input data.

        Parameters:
        -----------
        dataset_name : str
            Dataset name.

        Returns:
        --------
        bool
            Whether loading succeeded.
        """
        try:
            self.log_printer.print("Loading Data...")
            start_time = time.time()

            # Build the full output path.
            full_output_path = self.output_file_loc / dataset_name

            self.data = DataALNS(
                str(self.input_file_loc), str(full_output_path), dataset_name
            )
            self.data.load()

            end_time = time.time()
            self.log_printer.print(
                f"Data loading elapsed time: {end_time - start_time:.2f}s"
            )

            # Clear output files.
            self._clear_output_files(full_output_path)

            return True

        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            self.log_printer.print(f"Error loading data: {e}", color="bold red")
            return False

    def _clear_output_files(self, output_path: Path):
        """Clear and reinitialize output files."""
        try:
            output_path.mkdir(parents=True, exist_ok=True)

            # Define files that require clearing.
            files_to_clear = {
                "opt_result.csv": [
                    "day",
                    "plant_code",
                    "client_code",
                    "product_code",
                    "vehicle_id",
                    "vehicle_type",
                    "qty",
                ],
                "non_fulfill.csv": ["client_code", "product_code", "volume"],
            }

            # Clear and initialize each file.
            for file_name, header in files_to_clear.items():
                file_path = output_path / file_name
                with open(file_path, "w", newline="", encoding="utf-8") as file:
                    writer = csv.writer(file)
                    writer.writerow(header)

        except Exception as e:
            logger.error(f"Failed to clear output files: {e}")

    def create_initial_solution(self) -> SolutionState | None:
        """
        Create the initial solution.

        Returns:
        --------
        SolutionState or None
            Initial solution, or ``None`` when creation fails.
        """
        if self.data is None:
            self.log_printer.print("Error: Data not loaded", color="bold red")
            return None

        try:
            self.log_printer.print("Initializing the solution...")

            # Create the SolutionState instance.
            state = SolutionState(self.data)

            # Initialize optional tuner and ML selector components.
            use_tuner = self.is_param_tuner_enabled()
            use_ml = self.is_ml_enabled()
            if use_tuner or use_ml:
                self.log_printer.print("Initializing parameter auto-tuner...")
                self.param_tuner = ParamAutoTuner(self.data, self.rng)

                # Attach the parameter tuner.
                state.set_param_tuner(self.param_tuner)

            if use_ml and self.param_tuner is not None:
                self.log_printer.print("Initializing ML selector...")
                self.ml_selector = MLOperatorSelector(self.param_tuner, self.rng)
                state.set_ml_selector(self.ml_selector)

            # Build the initial solution, optionally suppressing noisy prints.
            quiet_init_logs = bool(
                getattr(ALNSConfig, "QUIET_INITIAL_SOLUTION_LOGS", True)
            )
            if quiet_init_logs:
                with redirect_stdout(io.StringIO()):
                    init_sol = initial_solution(state, rng=self.rng)
            else:
                init_sol = initial_solution(state, rng=self.rng)

            # Evaluate the initial solution.
            initial_feasibility, violations = init_sol.validate()

            init_actual_obj = float(init_sol.calculate_cost())
            self.log_printer.print(
                f"Initial Solution Objective (Gurobi-aligned): {init_actual_obj:.2f}"
            )
            self.log_printer.print(
                f"Vehicles Used in Initial Solution: {len(init_sol.vehicles)}"
            )
            self.log_printer.print(f"Initial Solution Cost: {init_actual_obj:.2f}")

            if not initial_feasibility:
                self.log_printer.print(
                    "Warning: The initial solution is infeasible!", color="bold red"
                )
                self._print_violation_details(violations)

                # Decide whether to terminate according to configuration.
                if getattr(ALNSConfig, "TERMINATE_ON_INFEASIBLE_INITIAL", True):
                    self.log_printer.print(
                        "Terminating due to infeasible initial solution.",
                        color="bold red",
                    )
                    return None
                else:
                    self.log_printer.print(
                        "Continuing with infeasible initial solution.", color="yellow"
                    )

            return init_sol

        except Exception as e:
            logger.error(f"Failed to create initial solution: {e}")
            self.log_printer.print(
                f"Error creating initial solution: {e}", color="bold red"
            )
            return None

    def _print_violation_details(self, violations: dict):
        """Print violation details."""
        if violations.get("negative_inventory"):
            self.log_printer.print("Negative inventory violations:")
            for key, inv in violations["negative_inventory"]:
                self.log_printer.print(
                    f"  Plant: {key[0]}, SKU: {key[1]}, Day: {key[2]}, Inventory: {inv}"
                )

        if violations.get("veh_overload"):
            self.log_printer.print("Vehicle overload violations:")
            for v in violations["veh_overload"]:
                veh = v["veh"]
                self.log_printer.print(
                    f"  Vehicle: Plant {veh.fact_id}, Dealer {veh.dealer_id}, Type {veh.type}, Day {veh.day}, Loaded: {v['loaded']}, Capacity: {v['cap']}"
                )

        if violations.get("unmet_demand"):
            self.log_printer.print("Unmet demand violations:")
            for d in violations["unmet_demand"]:
                self.log_printer.print(
                    f"  Dealer: {d['dealer']}, SKU: {d['sku_id']}, Demand: {d['demand']}, Shipped: {d['shipped']}"
                )

    def setup_alns(self) -> ALNS:
        """
        Configure the ALNS algorithm instance.

        Returns:
        --------
        ALNS
            Configured ALNS instance.
        """
        try:
            # Counts are derived during registration. Reset them so repeated
            # setup calls cannot accumulate stale operator counts.
            self._num_registered_destroy_ops = 0
            self._num_registered_repair_ops = 0
            # Create the ALNS instance.
            alns = ALNS(rng=rnd.default_rng(seed=SEED))

            # Register destroy operators.
            self._register_destroy_operators(alns)

            # Register repair operators.
            self._register_repair_operators(alns)
            # Preserve no-op markers from destroy to repair so the selector can
            # avoid rewarding an unchanged operator pair.
            for name, operator in list(alns._r_ops.items()):
                alns._r_ops[name] = _NoOpFlagPreservingRepair(operator)

            return alns

        except Exception as e:
            logger.error(f"Failed to configure ALNS: {e}")
            raise

    def _register_destroy_operators(self, alns: ALNS):
        """Register destroy operators."""
        self.log_printer.print("Registering destroy operators...")

        try:
            # Load base parameters.
            destroy_params = ALNSConfig.get_destroy_params()

            # Add the ML destroy operator when tuner and selector are enabled.
            if self.is_ml_enabled() and self.param_tuner and self.ml_selector:
                self.log_printer.print("Registering ML-enhanced destroy operator...")
                ml_destroy_op = self._create_named_partial(
                    ml_based_destroy,
                    name="ml_based_destroy",
                    min_sample_size=getattr(ALNSConfig, "ML_INITIAL_SAMPLE_SIZE", 20),
                )
                alns.add_destroy_operator(ml_destroy_op)
                self._num_registered_destroy_ops = (
                    getattr(self, "_num_registered_destroy_ops", 0) + 1
                )

            # Register random removal through a wrapper factory.
            random_removal_op = create_random_removal(
                destroy_params["random_removal_degree"]
            )
            alns.add_destroy_operator(random_removal_op)
            self._num_registered_destroy_ops = (
                getattr(self, "_num_registered_destroy_ops", 0) + 1
            )

            # Register Shaw removal through a wrapper factory.
            shaw_removal_op = create_shaw_removal(destroy_params["shaw_removal_degree"])
            alns.add_destroy_operator(shaw_removal_op)
            self._num_registered_destroy_ops = (
                getattr(self, "_num_registered_destroy_ops", 0) + 1
            )

            # Register periodic Shaw removal through its parameterized factory.
            periodic_shaw_removal_op = create_periodic_shaw_removal(
                **destroy_params.get("periodic_shaw_params", {})
            )
            alns.add_destroy_operator(periodic_shaw_removal_op)
            self._num_registered_destroy_ops = (
                getattr(self, "_num_registered_destroy_ops", 0) + 1
            )

            # Register remaining parameterized operators through their factories.
            # worst_removal: prefer operator-level default if not provided in DESTROY_DEFAULTS
            worst_defaults = ALNSConfig.get_operator_default("worst_removal").get(
                "params", {}
            )
            worst_removal_op = create_worst_removal(
                destroy_params.get(
                    "worst_removal_degree", worst_defaults.get("degree", 0.25)
                )
            )
            alns.add_destroy_operator(worst_removal_op)
            self._num_registered_destroy_ops = (
                getattr(self, "_num_registered_destroy_ops", 0) + 1
            )

            # surplus_inventory_removal: use operator default fallback
            surplus_defaults = ALNSConfig.get_operator_default(
                "surplus_inventory_removal"
            ).get("params", {})
            surplus_inventory_removal_op = create_surplus_inventory_removal(
                destroy_params.get(
                    "surplus_inventory_removal_degree",
                    surplus_defaults.get("degree", 0.25),
                )
            )
            alns.add_destroy_operator(surplus_inventory_removal_op)
            self._num_registered_destroy_ops = (
                getattr(self, "_num_registered_destroy_ops", 0) + 1
            )

            # path_removal: use operator default fallback
            path_defaults = ALNSConfig.get_operator_default("path_removal").get(
                "params", {}
            )
            path_removal_op = create_path_removal(
                destroy_params.get(
                    "path_removal_degree", path_defaults.get("degree", 0.5)
                )
            )
            alns.add_destroy_operator(path_removal_op)
            self._num_registered_destroy_ops = (
                getattr(self, "_num_registered_destroy_ops", 0) + 1
            )

            # cross_period_dealer_removal: large-jump neighborhood for cross-period reconstruction
            block_defaults = ALNSConfig.get_operator_default(
                "cross_period_dealer_removal"
            ).get("params", {})
            cross_period_dealer_removal_op = create_cross_period_dealer_removal(
                destroy_params.get(
                    "cross_period_dealer_removal_degree",
                    block_defaults.get("degree", 0.45),
                )
            )
            alns.add_destroy_operator(cross_period_dealer_removal_op)
            self._num_registered_destroy_ops = (
                getattr(self, "_num_registered_destroy_ops", 0) + 1
            )

            # period_range_removal: contiguous time-window removal
            period_range_defaults = ALNSConfig.get_operator_default(
                "period_range_removal"
            ).get("params", {})
            period_range_removal_op = create_period_range_removal(
                destroy_params.get(
                    "period_range_removal_degree",
                    period_range_defaults.get("degree", 0.35),
                ),
                destroy_params.get(
                    "period_range_removal_window_ratio",
                    period_range_defaults.get("window_ratio", 0.30),
                ),
            )
            alns.add_destroy_operator(period_range_removal_op)
            self._num_registered_destroy_ops = (
                getattr(self, "_num_registered_destroy_ops", 0) + 1
            )

            # infeasible_removal takes no parameters; wrap it for compatibility.
            alns.add_destroy_operator(wrap_operator_no_args(infeasible_removal))
            self._num_registered_destroy_ops = (
                getattr(self, "_num_registered_destroy_ops", 0) + 1
            )

            # inventory_level_removal: remove deliveries around abnormal dealer inventory.
            inv_defaults = ALNSConfig.get_operator_default(
                "inventory_level_removal"
            ).get("params", {})
            inventory_level_removal_op = create_inventory_level_removal(
                destroy_params.get(
                    "inventory_level_removal_degree",
                    inv_defaults.get("degree", 0.25),
                )
            )
            alns.add_destroy_operator(inventory_level_removal_op)
            self._num_registered_destroy_ops = (
                getattr(self, "_num_registered_destroy_ops", 0) + 1
            )

            self.log_printer.print("All destroy operators registered.")

        except Exception as e:
            logger.error(f"Failed to register destroy operators: {e}")
            raise

    def _register_repair_operators(self, alns: ALNS):
        """Register repair operators."""
        self.log_printer.print("Registering repair operators...")

        try:
            # Add the ML repair operator when tuner and selector are enabled.
            if self.is_ml_enabled() and self.param_tuner and self.ml_selector:
                self.log_printer.print("Registering ML-enhanced repair operator...")
                ml_repair_op = self._create_named_partial(
                    ml_based_repair,
                    name="ml_based_repair",
                    min_sample_size=getattr(ALNSConfig, "ML_INITIAL_SAMPLE_SIZE", 20),
                )
                alns.add_repair_operator(ml_repair_op)
                self._num_registered_repair_ops = (
                    getattr(self, "_num_registered_repair_ops", 0) + 1
                )

            # Load repair parameters through centralized accessors.
            repair_params = ALNSConfig.get_repair_params()
            # per-operator fallbacks (read from OPERATOR_DEFAULTS when available)
            local_defaults = ALNSConfig.get_operator_default("local_search_repair").get(
                "params", {}
            )
            smart_defaults = ALNSConfig.get_operator_default("smart_batch_repair").get(
                "params", {}
            )
            regret_defaults = ALNSConfig.get_operator_default(
                "regret_based_repair"
            ).get("params", {})
            lookahead_defaults = ALNSConfig.get_operator_default(
                "lookahead_repair"
            ).get("params", {})

            # Register repair operators through wrappers and factories.
            alns.add_repair_operator(wrap_repair_no_args(greedy_repair))
            self._num_registered_repair_ops = (
                getattr(self, "_num_registered_repair_ops", 0) + 1
            )

            # Register cross-period consolidation as a first-class repair
            # neighborhood. It can reduce the number of vehicle-period
            # activations that same-route/day consolidation cannot remove.
            alns.add_repair_operator(
                create_cross_period_consolidation_repair()
            )
            self._num_registered_repair_ops = (
                getattr(self, "_num_registered_repair_ops", 0) + 1
            )

            local_iter = repair_params.get("local_search_repair", {}).get(
                "max_iter", local_defaults.get("max_iter", 10)
            )
            alns.add_repair_operator(create_local_search_repair(local_iter))
            self._num_registered_repair_ops = (
                getattr(self, "_num_registered_repair_ops", 0) + 1
            )

            alns.add_repair_operator(wrap_repair_no_args(inventory_balance_repair))
            self._num_registered_repair_ops = (
                getattr(self, "_num_registered_repair_ops", 0) + 1
            )

            smart_iter = repair_params.get("smart_batch_repair", {}).get(
                "max_iter", smart_defaults.get("max_iter", 10)
            )
            smart_batch_size = repair_params.get("smart_batch_repair", {}).get(
                "batch_size", smart_defaults.get("batch_size", 10)
            )
            smart_timeout = repair_params.get("smart_batch_repair", {}).get(
                "timeout", smart_defaults.get("timeout", None)
            )
            alns.add_repair_operator(
                create_smart_batch_repair(
                    smart_iter, batch_size=smart_batch_size, timeout=smart_timeout
                )
            )
            self._num_registered_repair_ops = (
                getattr(self, "_num_registered_repair_ops", 0) + 1
            )

            regret_k = repair_params.get("regret_based_repair", {}).get(
                "k", regret_defaults.get("k", 2)
            )
            regret_topN = repair_params.get("regret_based_repair", {}).get(
                "topN", regret_defaults.get("topN", 6)
            )
            regret_time = repair_params.get("regret_based_repair", {}).get(
                "time_limit", regret_defaults.get("time_limit", 10.0)
            )
            alns.add_repair_operator(
                wrap_repair_no_args(
                    create_regret_repair(regret_k, regret_topN, regret_time)
                )
            )
            self._num_registered_repair_ops = (
                getattr(self, "_num_registered_repair_ops", 0) + 1
            )

            # Register fleet consolidation to reduce fragmented assignments.
            alns.add_repair_operator(FleetConsolidationRepairOperator())
            self._num_registered_repair_ops = (
                getattr(self, "_num_registered_repair_ops", 0) + 1
            )

            # Register the sequential repair operator.
            alns.add_repair_operator(create_sequential_repair())
            self._num_registered_repair_ops = (
                getattr(self, "_num_registered_repair_ops", 0) + 1
            )

            # Register lookahead repair for stagnation escape; default horizon is 2.
            lookahead_h = repair_params.get("lookahead_repair", {}).get(
                "lookahead_horizon",
                lookahead_defaults.get("lookahead_horizon", 2),
            )
            alns.add_repair_operator(create_lookahead_repair(lookahead_h))
            self._num_registered_repair_ops = (
                getattr(self, "_num_registered_repair_ops", 0) + 1
            )

            self.log_printer.print("All repair operators registered.")

        except Exception as e:
            logger.error(f"Failed to register repair operators: {e}")
            raise

    def _create_named_partial(self, func, name=None, **kwargs):
        """
        Create a callable with a readable ``__name__`` for ALNS compatibility.

        Parameters:
        -----------
        func : callable
            Function to wrap.
        name : str, optional
            Wrapper name. Generated automatically when omitted.
        **kwargs : dict
            Keyword arguments to bind.

        Returns:
        --------
        partial object with __name__ attribute
        """

        class _NamedCallable:
            def __init__(self, wrapped, callable_name: str, fixed_kwargs: dict):
                self._wrapped = wrapped
                self._fixed_kwargs = dict(fixed_kwargs)
                self.__name__ = callable_name

            def __call__(self, state, rng, **call_kwargs):
                merged = dict(self._fixed_kwargs)
                merged.update(call_kwargs or {})
                return self._wrapped(state, rng, **merged)

        if name is None:
            param_str = "_".join([f"{k}_{v}" for k, v in kwargs.items()])
            name = f"{func.__name__}_{param_str}"

        return _NamedCallable(func, name, kwargs)

    @staticmethod
    def _run_elapsed_seconds(stop: Any):
        """
        Return a callable reporting seconds consumed by the runtime budget.

        Reads the stopping rule's own stopwatch, which is the single origin for
        runtime-budgeted policies. Returns None when the rule cannot report
        elapsed time, leaving the tracker on its own monotonic origin.
        """
        if stop is None or not hasattr(stop, "get_status"):
            return None

        def _elapsed() -> float:
            status = stop.get_status() or {}
            return float(status.get("elapsed_time", 0.0) or 0.0)

        return _elapsed

    def run_optimization(self, dataset_name: str) -> bool:
        """
        Run the complete optimization workflow.

        Parameters:
        -----------
        dataset_name : str
            Dataset name.

        Returns:
        --------
        bool
            Whether optimization completed successfully.
        """
        try:
            self.log_printer.print(f"Starting optimization / {datetime.datetime.now()}")
            optimization_start_time = time.time()

            # 1. Load data.
            t0 = time.time()
            self.log_printer.print("Step 1: Loading data...")
            if not self.load_data(dataset_name):
                return False

            # Optional startup plotting (guarded by config to avoid heavy startup cost).
            # The supply-chain 3D plot is disabled by default; enable both
            # ENABLE_STARTUP_PLOTTING and ENABLE_SUPPLY_CHAIN_3D to use it.
            if getattr(ALNSConfig, "ENABLE_STARTUP_PLOTTING", False) and getattr(
                ALNSConfig, "ENABLE_SUPPLY_CHAIN_3D", False
            ):
                try:
                    # delayed import to avoid importing matplotlib unless requested
                    from .visualization import (
                        plot_supply_chain_network_3d_enhanced as _plot_startup,
                    )

                    if self.data is not None:
                        _plot_startup(
                            self.data, str(self.output_file_loc / dataset_name)
                        )
                except Exception as e:
                    logger.debug(f"Startup plotting skipped: {e}")

            self.log_printer.print(f"Step 1 elapsed: {time.time() - t0:.2f}s")

            # 2. Create the initial solution.
            t0 = time.time()
            self.log_printer.print("Step 2: Creating initial solution...")
            init_sol = self.create_initial_solution()
            if init_sol is None:
                return False
            self.log_printer.print(f"Step 2 elapsed: {time.time() - t0:.2f}s")

            # 3. Configure ALNS.
            t0 = time.time()
            self.log_printer.print("Step 3: Configuring ALNS...")
            alns = self.setup_alns()
            self.log_printer.print(f"Step 3 elapsed: {time.time() - t0:.2f}s")

            # 4. Configure operator selection.
            t0 = time.time()
            self.log_printer.print("Step 4: Configuring operator selection...")
            # Use actual registration counts instead of hard-coded values.
            num_destroy = getattr(self, "_num_registered_destroy_ops", 7) or 7
            num_repair = getattr(self, "_num_registered_repair_ops", 6) or 6
            select = NoOpAwareSegmentedRouletteWheel(
                scores=getattr(ALNSConfig, "ROULETTE_SCORES", [1.0]),
                decay=getattr(ALNSConfig, "ROULETTE_DECAY", 0.9),
                seg_length=getattr(ALNSConfig, "ROULETTE_SEG_LENGTH", 100),
                num_destroy=num_destroy,
                num_repair=num_repair,
            )
            self.log_printer.print(f"Step 4 elapsed: {time.time() - t0:.2f}s")

            # 5. Configure acceptance rules.
            t0 = time.time()
            self.log_printer.print("Step 5: Configuring acceptance policy...")
            sa_accept = SimulatedAnnealing(
                start_temperature=getattr(ALNSConfig, "SA_START_TEMP", 1.0),
                end_temperature=getattr(ALNSConfig, "SA_END_TEMP", 1e-3),
                step=getattr(ALNSConfig, "SA_STEP", 0.99),
                method="exponential",
            )
            self.acceptance = DemandConstraintAccept(sa_accept, self.data)
            self.log_printer.print(f"Step 5 elapsed: {time.time() - t0:.2f}s")

            # 6. Configure stopping rules.
            t0 = time.time()
            self.log_printer.print("Step 6: Configuring stopping rules...")

            # Combined stopping rule by dataset scale:
            # small: 300 iterations OR 900 seconds;
            # medium: 600 iterations OR 1800 seconds;
            # large: 1000 iterations OR 3600 seconds.
            stop = create_standard_combined_criterion(
                max_iterations=getattr(ALNSConfig, "MAX_ITERATIONS", 300),
                max_runtime=getattr(ALNSConfig, "MAX_RUNTIME", 900),
                max_no_improvement=None,  # No-improvement stopping is disabled.
            )

            # Initialize a conservative iteration ceiling so MaxIterations does
            # not truncate a runtime-budgeted search too early.
            try:
                dynamic_floor = compute_dynamic_iteration_cap(ALNSConfig)
                if hasattr(stop, "set_max_iterations"):
                    stop.set_max_iterations(dynamic_floor)
                self._dynamic_iteration_cap = int(dynamic_floor)

                # Keep optimizer/tracker iteration horizon aligned with the effective
                # stopping configuration so progress-based policies remain consistent.
                try:
                    self.set_max_iterations(dynamic_floor)
                except Exception:
                    pass
                try:
                    if self.tracker is not None and hasattr(
                        self.tracker, "set_max_iterations"
                    ):
                        self.tracker.set_max_iterations(dynamic_floor)
                except Exception:
                    pass
            except Exception:
                pass

            self.log_printer.print(
                f"Combined stopping rule: dynamic iteration cap, runtime budget {getattr(ALNSConfig, 'MAX_RUNTIME', 900)} seconds"
            )
            self.log_printer.print(f"Step 6 elapsed: {time.time() - t0:.2f}s")

            # 7. Configure tracker and callbacks.
            t0 = time.time()
            self.log_printer.print("Step 7: Configuring tracker and callbacks...")
            # The tracker's runtime clock is the stop rule's own stopwatch, so
            # every progress-driven policy shares one origin with the stopping
            # decision instead of measuring elapsed time twice.
            self.tracker = ALNSTracker(
                log_printer=self.log_printer,
                elapsed_seconds=self._run_elapsed_seconds(stop),
            )
            try:
                dynamic_cap = getattr(self, "_dynamic_iteration_cap", None)
                if dynamic_cap is not None and hasattr(
                    self.tracker, "set_max_iterations"
                ):
                    self.tracker.set_max_iterations(int(dynamic_cap))
            except Exception:
                pass
            # Attach the tracker to the initial solution for repair operators.
            init_sol.set_tracker(self.tracker)
            try:
                self.tracker.set_initial_reference(
                    initial_search_obj=float(init_sol.objective()),
                    initial_actual_obj=float(init_sol.calculate_cost()),
                )
            except Exception:
                pass

            # Record a strictly feasible initial state as an early rollback baseline.
            try:
                init_feasible, _ = init_sol.validate()
                if init_feasible:
                    self.best_feasible_state = safe_state_snapshot(init_sol)
                    self.best_feasible_obj = float(init_sol.objective())
            except Exception:
                pass

            register_callbacks(alns, self, stop=stop)
            # Give the tuner the same stopwatch the stop rule and tracker use,
            # so its decay schedule shares the run's single time origin.
            if self.param_tuner is not None and hasattr(
                self.param_tuner, "set_elapsed_clock"
            ):
                self.param_tuner.set_elapsed_clock(self._run_elapsed_seconds(stop))
            self.log_printer.print(f"Step 7 elapsed: {time.time() - t0:.2f}s")

            # 8. Run ALNS.
            t0 = time.time()
            self.log_printer.print("Step 8: Running ALNS...")
            try:
                quiet_operator_logs = bool(
                    getattr(ALNSConfig, "QUIET_OPERATOR_LOGS", True)
                )
                with suppress_noisy_operator_stdout(enabled=quiet_operator_logs):
                    self.result = alns.iterate(
                        init_sol, select, cast(Any, self.acceptance), stop
                    )
                self.log_printer.print(f"Step 8 elapsed: {time.time() - t0:.2f}s")
            except Exception as e:
                # Capture the full traceback for comparison failures.
                import traceback

                tb = traceback.format_exc()
                # print to stdout as well as logger in case logger not configured
                try:
                    print("ALNS iterate exception traceback:\\n", tb)
                except Exception:
                    pass
                logger.error("ALNS iterate failed: %s", e)
                logger.debug(tb)
                self.log_printer.print(f"ALNS iterate failed: {e}", color="bold red")
                return False

            optimization_end_time = time.time()
            self.log_printer.print(
                f"Total optimization time: {optimization_end_time - optimization_start_time:.2f}s"
            )

            # 9. Process results.
            t0 = time.time()
            self.log_printer.print("Step 9: Processing results...")
            self._process_results()
            self.log_printer.print(f"Step 9 elapsed: {time.time() - t0:.2f}s")

            # 10. Generate reports and figures.
            t0 = time.time()
            self.log_printer.print("Step 10: Generating reports and figures...")
            self._generate_reports()
            self.log_printer.print(f"Step 10 elapsed: {time.time() - t0:.2f}s")

            # Report every exception that a handler deliberately swallowed during
            # the run. The search tolerates local failures, but "tolerated" must
            # not mean "invisible": a systematically failing site shows up here
            # instead of silently disabling an operator.
            report_swallowed(self.log_printer)

            self.log_printer.print_title(f"Optimization complete! {datetime.datetime.now()}")

            return True

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            self.log_printer.print(f"Optimization failed: {e}", color="bold red")
            return False

    def _process_results(self):
        """Process optimization results."""
        if self.result is None:
            return

        self.best_solution = safe_state_snapshot(self.result.best_state)

        # Optionally select the tracked best feasible state by actual objective.
        tracker_stats = self.tracker.get_statistics()
        if bool(getattr(ALNSConfig, "PREFER_FINAL_BEST_ACTUAL_SOLUTION", True)):
            candidate_actual_best = tracker_stats.get("best_actual_solution")
            if candidate_actual_best is not None:
                self.best_solution = safe_state_snapshot(candidate_actual_best)

        # The callback-maintained fully feasible state has highest priority.
        callback_best_feasible = getattr(self, "best_feasible_state", None)
        if callback_best_feasible is not None:
            callback_snapshot = safe_state_snapshot(callback_best_feasible)
            if self.best_solution is None:
                self.best_solution = callback_snapshot
            else:
                try:
                    current_cost = float(self.best_solution.calculate_cost())
                    callback_cost = float(callback_snapshot.calculate_cost())
                    if callback_cost < current_cost or (
                        abs(callback_cost - current_cost) <= 1e-9
                        and len(callback_snapshot.vehicles)
                        < len(self.best_solution.vehicles)
                    ):
                        self.best_solution = callback_snapshot
                except Exception:
                    # Keep the safer callback-maintained feasible state if
                    # objective comparison is unavailable.
                    self.best_solution = callback_snapshot

        # Run one light consolidation before output and roll back if infeasible.
        if self.best_solution is not None and getattr(
            ALNSConfig, "FINAL_BEST_ENABLE_CONSOLIDATION", True
        ):
            before_cnt = len(self.best_solution.vehicles)
            try:
                max_pairs = int(
                    getattr(ALNSConfig, "FINAL_BEST_CONSOLIDATION_MAX_PAIRS", 800)
                )
            except (TypeError, ValueError):
                max_pairs = 800

            snapshot_before = safe_state_snapshot(self.best_solution)
            merged_cnt = 0
            try:
                merged_cnt = consolidate_route_vehicles(
                    self.best_solution, max_pairs_per_group=max(1, max_pairs)
                )
            except Exception:
                merged_cnt = 0

            if merged_cnt > 0:
                try:
                    feasible_after, _ = self.best_solution.validate()
                except Exception:
                    feasible_after = False
                cost_after = None
                cost_before = None
                try:
                    cost_after = float(self.best_solution.calculate_cost())
                    if snapshot_before is not None:
                        cost_before = float(snapshot_before.calculate_cost())
                except Exception:
                    pass
                if (
                    not feasible_after
                    or (
                        cost_after is not None
                        and cost_before is not None
                        and cost_after > cost_before + 1e-9
                    )
                ) and snapshot_before is not None:
                    self.best_solution = snapshot_before
                    self.log_printer.print(
                        "Final consolidation did not improve the actual objective; rolled back.",
                        color="yellow",
                    )
                else:
                    after_cnt = len(self.best_solution.vehicles)
                    self.log_printer.print(
                        f"Final consolidation merged {merged_cnt} vehicles ({before_cnt} -> {after_cnt})."
                    )

        # Run a final cross-period consolidation pass. It only keeps moves
        # that remain feasible and reduce the actual objective.
        if self.best_solution is not None:
            before_cross_period_cnt = len(self.best_solution.vehicles)
            before_cross_period_cost = None
            try:
                before_cross_period_cost = float(
                    self.best_solution.calculate_cost()
                )
            except Exception:
                before_cross_period_cost = None
            consolidation_failed = False
            try:
                # The try covers only the consolidation itself, so a failure in
                # the reporting below cannot be mistaken for a consolidation
                # failure and silently swallowed.
                consolidated = cross_period_consolidation_repair(
                    self.best_solution,
                    self.rng,
                    time_limit=float(
                        getattr(
                            ALNSConfig,
                            "FINAL_CROSS_PERIOD_CONSOLIDATION_TIME_LIMIT",
                            20.0,
                        )
                    ),
                    max_moves=int(
                        getattr(
                            ALNSConfig,
                            "FINAL_CROSS_PERIOD_CONSOLIDATION_MAX_MOVES",
                            60,
                        )
                    ),
                )
                self.best_solution = consolidated
            except Exception as exc:
                # Never silent: the run still reports "Optimization complete!"
                # and the caller receives whatever best_solution now holds, so
                # report it and restore the pre-consolidation solution so the
                # delivered answer is the verified one.
                consolidation_failed = True
                self.log_printer.print(
                    "Warning: final cross-period consolidation failed "
                    f"({type(exc).__name__}: {exc}); "
                    + (
                        "restored the pre-consolidation solution."
                        if snapshot_before is not None
                        else "kept the previous solution."
                    ),
                    color="bold red",
                )
                if snapshot_before is not None:
                    self.best_solution = snapshot_before

            # Report whichever solution is actually adopted. Computed after
            # the handler so the numbers always describe the delivered
            # solution. The cost is best-effort: the pre-consolidation
            # solution may carry a defect that surfaced only inside the
            # consolidation, and a run must not abort merely because its
            # summary could not be computed.
            after_cross_period_cnt = len(self.best_solution.vehicles)
            try:
                after_cross_period_cost = float(
                    self.best_solution.calculate_cost()
                )
            except Exception as exc:
                record_swallowed(
                    "alns_optimizer.final_consolidation_summary", exc
                )
                after_cross_period_cost = None
            if (
                not consolidation_failed
                and before_cross_period_cost is not None
                and after_cross_period_cost is not None
            ):
                self.log_printer.print(
                    "Final cross-period consolidation: "
                    f"{before_cross_period_cnt} -> {after_cross_period_cnt} "
                    f"vehicles, cost {before_cross_period_cost:.2f} -> "
                    f"{after_cross_period_cost:.2f}."
                )

        # Validate the final solution.
        final_feasibility, violations = self.best_solution.validate()
        if not final_feasibility:
            self.log_printer.print(
                "Warning: The final best solution is infeasible!", color="bold red"
            )
            fallback = getattr(self, "best_feasible_state", None)
            if fallback is not None:
                fallback_snapshot = safe_state_snapshot(fallback)
                fallback_feasible, _ = fallback_snapshot.validate()
                if fallback_feasible:
                    self.best_solution = fallback_snapshot
                    self.log_printer.print(
                        "Recovered the last verified feasible incumbent.",
                        color="yellow",
                    )
                else:
                    raise RuntimeError(
                        "The final solution and the fallback incumbent are infeasible."
                    )
            else:
                raise RuntimeError(
                    "The final solution is infeasible and no fallback exists."
                )

        # The final cross-period consolidation pass and any feasibility
        # recovery run after the tracker snapshot was taken, so they can
        # change the incumbent. Recompute the summary values from the
        # solution that is actually emitted, rather than reporting a stale
        # pre-consolidation objective next to the final solution cost.
        best_search_obj = float(self.best_solution.objective())
        best_actual_obj = float(self.best_solution.calculate_cost())
        try:
            final_gap = float(
                self.tracker._signed_gap_from_initial(
                    best_search_obj, self.tracker.initial_search_obj
                )
            )
        except Exception:
            final_gap = float(tracker_stats["final_gap"])
        try:
            final_actual_gap = float(
                self.tracker._signed_gap_from_initial(
                    best_actual_obj, self.tracker.initial_actual_obj
                )
            )
        except Exception:
            final_actual_gap = float(tracker_stats["final_actual_gap"])

        # Print statistics.
        self.log_printer.print(
            f"Total iterations tracked: {tracker_stats['total_iterations']}"
        )
        self.log_printer.print(f"Best objective found: {best_search_obj:.4f}")
        self.log_printer.print(
            f"Best actual objective found: {best_actual_obj:.4f}"
        )
        self.log_printer.print(
            f"Final Search Gap vs Initial: {final_gap:+.2f}%"
        )
        self.log_printer.print(
            f"Final Actual Gap vs Initial: {final_actual_gap:+.2f}%"
        )
        self.log_printer.print(
            f"Best Search Objective (ALNS-guiding): {best_search_obj:.6f}"
        )
        self.log_printer.print(
            f"Best Actual Objective (Gurobi-aligned): {best_actual_obj:.2f}"
        )
        self.log_printer.print(
            f"Vehicles Used in Best Solution: {len(self.best_solution.vehicles)}"
        )
        self.log_printer.print(f"Best Solution Cost: {best_actual_obj:.2f}")

        # Update data and write outputs.
        self._update_demands()
        self._update_inventory()
        self._output_results()
        self._generate_statistics()
        self._write_repair_audit_report()

    def _write_repair_audit_report(self):
        """Write per-repair-operator feasibility audit summary to CSV."""
        try:
            if self.best_solution is None:
                return
            audit = getattr(self.best_solution, "repair_audit", None)
            if not audit:
                return

            rows = []
            for op_name, stats in audit.items():
                attempts = int(stats.get("attempts", 0))
                feasible = int(stats.get("feasible", 0))
                infeasible = int(stats.get("infeasible", 0))
                rolled_back = int(stats.get("rolled_back", 0))
                rows.append(
                    {
                        "operator": op_name,
                        "attempts": attempts,
                        "feasible": feasible,
                        "infeasible": infeasible,
                        "feasible_rate": (float(feasible) / attempts)
                        if attempts > 0
                        else 0.0,
                        "rolled_back": rolled_back,
                        "inserted_total": int(stats.get("inserted_total", 0)),
                        "improvement_total": float(stats.get("improvement_total", 0.0)),
                        "fail_negative_inventory": int(
                            stats.get("fail_negative_inventory", 0)
                        ),
                        "fail_veh_over_load": int(stats.get("fail_veh_over_load", 0)),
                        "fail_plant_inv_exceed": int(
                            stats.get("fail_plant_inv_exceed", 0)
                        ),
                        "fail_unmet_demand": int(stats.get("fail_unmet_demand", 0)),
                    }
                )

            if not rows:
                return

            df = pd.DataFrame(rows).sort_values(
                by=["attempts", "operator"], ascending=[False, True]
            )
            out = Path(self.data.output_file_loc) / "repair_feasibility_audit.csv"
            df.to_csv(out, index=False)
            self.log_printer.print(f"Repair feasibility audit saved: {out}")
        except Exception as e:
            logger.error(f"Failed to write repair feasibility audit: {e}")

    def _update_demands(self):
        """Update demand data."""
        try:
            # Work on a local copy so output generation never mutates the
            # shared model input stored in DataALNS.demands.
            remaining_demands = {
                key: int(value) for key, value in self.data.demands.items()
            }
            for veh in self.best_solution.vehicles:
                for (sku_id, day), qty in veh.cargo.items():
                    key = (veh.dealer_id, sku_id)
                    if key in remaining_demands:
                        remaining_demands[key] -= int(qty)

            # Collect unmet demand.
            non_fulfill_data = []
            for (dealer, sku_id), qty in remaining_demands.items():
                if qty > 0:
                    non_fulfill_data.append(
                        {"client_code": dealer, "product_code": sku_id, "volume": qty}
                    )

            # Save unmet demand.
            if non_fulfill_data:
                df_non_fulfill = pd.DataFrame(non_fulfill_data)
                output_file = Path(self.data.output_file_loc) / "non_fulfill.csv"
                # Check whether the file is locked.
                if is_file_in_use(output_file):
                    self.log_printer.print(
                        f"File {output_file} is in use; close it and retry.",
                        color="bold red",
                    )
                    return
                df_non_fulfill.to_csv(output_file, mode="a", header=False, index=False)

        except Exception as e:
            logger.error(f"Failed to update demand: {e}")
            raise

    def _update_inventory(self):
        """Update inventory data."""
        try:
            sku_inv_left = {}

            # Iterate over every plant and SKU.
            for plant_id in self.data.plants:
                for sku_id in self.data.all_skus:
                    # Read initial inventory.
                    current_inv = self.data.sku_initial_inv.get((plant_id, sku_id), 0)

                    # Compute end-of-day inventory in chronological order.
                    for day in range(1, self.data.horizons + 1):
                        # Add same-day production.
                        current_inv += self.data.sku_prod_each_day.get(
                            (plant_id, sku_id, day), 0
                        )

                        # Subtract same-day shipments.
                        shipped_qty = 0
                        for veh in self.best_solution.vehicles:
                            if veh.fact_id == plant_id and (sku_id, day) in veh.cargo:
                                shipped_qty += veh.cargo[(sku_id, day)]
                        current_inv -= shipped_qty

                        # Store end-of-day inventory.
                        sku_inv_left[(plant_id, sku_id, day)] = current_inv

            # Create and save the inventory DataFrame.
            df_sku_inv_left = pd.DataFrame(
                [
                    {
                        "plant_code": plant,
                        "product_code": sku_id,
                        "day": day,
                        "volume": qty,
                    }
                    for (plant, sku_id, day), qty in sku_inv_left.items()
                ]
            )
            output_file = Path(self.data.output_file_loc) / "sku_inv_left.csv"
            if is_file_in_use(output_file):
                self.log_printer.print(
                    f"File {output_file} is in use; close it and retry.",
                    color="bold red",
                )
                return
            df_sku_inv_left.to_csv(output_file, index=False)

        except Exception as e:
            logger.error(f"Failed to update inventory: {e}")
            raise

    def _output_results(self):
        """Write optimization results to file."""
        try:
            output_file = Path(self.data.output_file_loc) / "opt_result.csv"
            df_result = self._create_result_dataframe()
            if is_file_in_use(output_file):
                self.log_printer.print(
                    f"File {output_file} is in use; close it and retry.",
                    color="bold red",
                )
                return
            df_result.to_csv(output_file, mode="a", header=False, index=False)

        except Exception as e:
            logger.error(f"Failed to output results: {e}")
            raise

    def _create_result_dataframe(self) -> pd.DataFrame:
        """Create the result DataFrame."""
        rows = []
        for idx, veh in enumerate(self.best_solution.vehicles):
            for (sku_id, day), qty in veh.cargo.items():
                rows.append(
                    {
                        "day": day,
                        "plant_code": veh.fact_id,
                        "client_code": veh.dealer_id,
                        "product_code": sku_id,
                        "vehicle_id": idx + 1,
                        "vehicle_type": veh.type,
                        "qty": qty,
                    }
                )
        return pd.DataFrame(rows)

    def _generate_statistics(self):
        """Generate summary statistics."""
        dataset = self.data.dataset_name
        output_location = Path(self.data.output_file_loc)
        input_location = self.input_file_loc / dataset

        # Read result files.
        df_result = pd.read_csv(output_location / "opt_result.csv")
        df_sku_size = pd.read_csv(input_location / "product_size.csv")
        df_all_veh = pd.read_csv(input_location / "vehicle.csv")

        # Normalize join-key types before merging.
        if "product_code" in df_result.columns:
            df_result["product_code"] = df_result["product_code"].astype(str)
        if "product_code" in df_sku_size.columns:
            df_sku_size["product_code"] = df_sku_size["product_code"].astype(str)
        if "vehicle_type" in df_result.columns:
            df_result["vehicle_type"] = df_result["vehicle_type"].astype(str)
        if "vehicle_type" in df_all_veh.columns:
            df_all_veh["vehicle_type"] = df_all_veh["vehicle_type"].astype(str)
        df_result = pd.merge(df_result, df_sku_size, on=["product_code"])
        df_result = pd.merge(df_result, df_all_veh, on=["vehicle_type"])

        # Remove unneeded columns.
        if "cost_to_use" in df_result.columns:
            del df_result["cost_to_use"]

        # Compute occupied volume.
        df_result["occupied_size"] = df_result["qty"] * df_result["standard_size"]

        # Save detailed results.
        details_file = output_location / "opt_details.csv"
        if is_file_in_use(details_file):
            self.log_printer.print(
                f"File {details_file} is in use; close it and retry.",
                color="bold red",
            )
            return
        df_result.to_csv(details_file, index=False)

        # Generate summary statistics.
        df_stat1 = df_result.copy()
        if "product_code" in df_stat1.columns:
            df_stat1 = df_stat1.drop(["product_code", "standard_size"], axis=1)

        df_summary = df_stat1.groupby(
            ["day", "plant_code", "client_code", "vehicle_id"]
        ).agg(
            {
                "vehicle_type": "first",
                "qty": "sum",
                "carry_standard_size": "first",
                "min_standard_size": "first",
                "occupied_size": "sum",
            }
        )

        # Reset the index.
        df_summary = df_summary.reset_index()

        # Add the same cost decomposition used by the actual objective.
        df_summary["below_min_load"] = (
            df_summary["occupied_size"] + 1e-9
            < df_summary["min_standard_size"]
        )
        df_summary["fixed_cost"] = df_summary["vehicle_type"].map(
            self.data.veh_type_cost
        )
        df_summary["minimum_load_penalty"] = np.where(
            df_summary["below_min_load"],
            float(self.data.param_alpha) * df_summary["fixed_cost"],
            0.0,
        )
        df_summary["total_cost"] = (
            df_summary["fixed_cost"] + df_summary["minimum_load_penalty"]
        )

        summary_file = output_location / "opt_summary.csv"
        if is_file_in_use(summary_file):
            self.log_printer.print(
                f"File {summary_file} is in use; close it and retry.",
                color="bold red",
            )
            return
        df_summary.to_csv(summary_file, index=False)

        # Validate capacity constraints.
        self._validate_capacity_constraints(df_summary)

    def _validate_capacity_constraints(self, df_summary: pd.DataFrame):
        """Validate vehicle capacity constraints."""
        try:
            # Check whether every vehicle load satisfies its capacity.
            is_cap_exceed = pd.Series(
                df_summary["occupied_size"] <= df_summary["carry_standard_size"]
            )

            if is_cap_exceed.all():
                self.log_printer.print(
                    "[OK] All vehicle loading volumes satisfy capacity constraints"
                )
            else:
                self.log_printer.print(
                    "Some vehicle loads exceed capacity limits", color="yellow"
                )
                df_exceed = df_summary[
                    df_summary["occupied_size"] > df_summary["carry_standard_size"]
                ]

                df_exceed_info = df_exceed.loc[
                    :,
                    [
                        "vehicle_id",
                        "vehicle_type",
                        "carry_standard_size",
                        "occupied_size",
                    ],
                ]
                df_exceed_info["exceeded_volume"] = (
                    df_exceed_info["occupied_size"]
                    - df_exceed_info["carry_standard_size"]
                )

                output_path = Path(self.data.output_file_loc) / "extra_volume.csv"
                if is_file_in_use(output_path):
                    self.log_printer.print(
                        f"File {output_path} is in use; close it and retry.",
                        color="bold red",
                    )
                    return
                df_exceed_info.to_csv(output_path, index=False)
                self.log_printer.print(f"Capacity excess details saved to: {output_path}")

        except Exception as e:
            logger.error(f"Failed to validate capacity constraints: {e}")

    def _generate_reports(self):
        """Generate reports and figures."""
        try:
            # Allow reports and plotting to be disabled for profiling and CI.
            if not getattr(ALNSConfig, "ENABLE_REPORTS", True):
                try:
                    self.log_printer.print(
                        "Report generation disabled (ENABLE_REPORTS=False); skipping plots."
                    )
                except Exception:
                    pass
                return
            if self.result is None or self.best_solution is None:
                return
            # delayed import of visualization tools to avoid importing matplotlib at module import time
            try:
                from .visualization import (
                    plot_gap_changes,
                    plot_objective_changes,
                    plot_operator_performance,
                )
            except Exception as e:
                logger.debug(f"Visualization tools unavailable: {e}")
                return
            # Create the image directory.
            images_dir = Path(self.data.output_file_loc) / "images"
            images_dir.mkdir(exist_ok=True)
            # Plot search and actual objective changes.
            plot_objective_changes(
                self.result,
                str(self.data.output_file_loc),
                tracker=self.tracker,
            )
            # Plot operator performance.
            plot_operator_performance(self.best_solution, self.result)
            # Plot gap changes.
            plot_gap_changes(self.tracker, str(self.data.output_file_loc))

            # Generate tuner and ML selector reports.
            if self.param_tuner:
                self._generate_param_tuning_report()

        except Exception as e:
            logger.error(f"Failed to generate reports: {e}")
            self.log_printer.print(
                f"Warning: Failed to generate reports: {e}", color="yellow"
            )

    def _generate_param_tuning_report(self):
        """Generate parameter tuning and ML selector reports."""
        if not self.param_tuner:
            return

        try:
            self.log_printer.print("Generating parameter tuning and ML selector reports...")

            # Create the report directory.
            report_dir = self.output_file_loc / "param_tuning_report"
            report_dir.mkdir(parents=True, exist_ok=True)

            # Generate the parameter tuning report.
            param_report_path = report_dir / "param_tuning_report.csv"
            with open(param_report_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "Operator",
                        "Parameter",
                        "Final Value",
                        "Range",
                        "Usage Count",
                        "Success Rate",
                        "Avg Performance",
                    ]
                )

                for op_name, op_params in self.param_tuner.operators.items():
                    for param_name, param_value in op_params.params.items():
                        if param_name in op_params.ranges:
                            param_range = f"{op_params.ranges[param_name][0]} - {op_params.ranges[param_name][1]}"
                        else:
                            param_range = "N/A"

                        writer.writerow(
                            [
                                op_name,
                                param_name,
                                param_value,
                                param_range,
                                op_params.usage_count,
                                f"{op_params.get_success_rate():.4f}",
                                f"{op_params.get_avg_performance():.4f}",
                            ]
                        )

            self.log_printer.print(f"Parameter tuning report saved to: {param_report_path}")

            # Generate ML selector reports when available.
            if self.ml_selector:
                # Generate the destroy-operator performance report.
                destroy_report_path = report_dir / "destroy_operators_performance.csv"
                with open(destroy_report_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Operator", "Usage Count", "Average Performance"])

                    # Count usage by operator.
                    destroy_op_stats = {}
                    for op_name in self.param_tuner.operators:
                        if "removal" in op_name:
                            destroy_op_stats[op_name] = {"count": 0, "total_perf": 0.0}

                    # Read performance data from the ML selector.
                    for i, op_encoding in enumerate(self.ml_selector.destroy_features):
                        if len(op_encoding) > 0:
                            # The final elements encode the selected operator.
                            op_idx = np.argmax(
                                op_encoding[-len(self.ml_selector.destroy_op_map) :]
                            )
                            op_name = list(self.ml_selector.destroy_op_map.keys())[
                                op_idx
                            ]
                            perf = self.ml_selector.destroy_performances[i]

                            if op_name in destroy_op_stats:
                                destroy_op_stats[op_name]["count"] += 1
                                destroy_op_stats[op_name]["total_perf"] += perf

                    # Write the report.
                    for op_name, stats in destroy_op_stats.items():
                        avg_perf = (
                            stats["total_perf"] / stats["count"]
                            if stats["count"] > 0
                            else 0
                        )
                        writer.writerow([op_name, stats["count"], f"{avg_perf:.4f}"])

                self.log_printer.print(
                    f"Destroy operator performance report saved to: {destroy_report_path}"
                )

                # Generate the repair-operator performance report.
                repair_report_path = report_dir / "repair_operators_performance.csv"
                with open(repair_report_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Operator", "Usage Count", "Average Performance"])

                    # Count usage by operator.
                    repair_op_stats = {}
                    for op_name in self.param_tuner.operators:
                        if "repair" in op_name:
                            repair_op_stats[op_name] = {"count": 0, "total_perf": 0.0}

                    # Read performance data from the ML selector.
                    for i, op_encoding in enumerate(self.ml_selector.repair_features):
                        if len(op_encoding) > 0:
                            # The final elements encode the selected operator.
                            op_idx = np.argmax(
                                op_encoding[-len(self.ml_selector.repair_op_map) :]
                            )
                            op_name = list(self.ml_selector.repair_op_map.keys())[
                                op_idx
                            ]
                            perf = self.ml_selector.repair_performances[i]

                            if op_name in repair_op_stats:
                                repair_op_stats[op_name]["count"] += 1
                                repair_op_stats[op_name]["total_perf"] += perf

                    # Write the report.
                    for op_name, stats in repair_op_stats.items():
                        avg_perf = (
                            stats["total_perf"] / stats["count"]
                            if stats["count"] > 0
                            else 0
                        )
                        writer.writerow([op_name, stats["count"], f"{avg_perf:.4f}"])

                self.log_printer.print(
                    f"Repair operator performance report saved to: {repair_report_path}"
                )

        except Exception as e:
            logger.error(f"Failed to generate tuning and ML selector reports: {e}")
            self.log_printer.print(
                f"Failed to generate tuning and ML selector reports: {e}",
                color="bold red",
            )
