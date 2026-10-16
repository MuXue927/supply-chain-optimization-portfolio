"""
Core solution state and inventory update utilities for the ALNS model.

Main responsibilities:
1. Define ``SolutionState``:
   - Store vehicles, per-period inventory ``s_ikt``, index sets, and optional
     references to adaptive and machine-learning components.
   - Provide cached objective evaluation, feasibility validation, inventory
     recomputation, and vehicle-load queries.
2. Objective caching:
   - ``last_objective`` stores the previous result.
   - ``objective_dirty`` marks structural changes in vehicles, cargo, or inventory.
   - Call ``mark_objective_dirty()`` after structural changes so the next
     ``objective()`` call recomputes the value.
3. Inventory update paths:
   A. Full recomputation in ``compute_inventory()``:
      - Uses ``s_t = s_{t-1} + prod_t - shipped_t``.
      - Has ``O(V + P*S*H)`` complexity, where ``V`` is vehicles, ``P`` is
        plants, ``S`` is SKUs, and ``H`` is the horizon.
      - Establishes the baseline inventory view for the current vehicle set.
   B. Incremental deduction in ``veh_loading()``:
      - Subtracts ``load_qty`` from ``s_ikt`` for ``day..H`` when new shipments
        are loaded.
      - A shipment affects every future period through the inventory recursion.
      - Avoids a full recomputation after each load and reduces repeated local
        loading costs to ``O(sum(H - day + 1))``.
      - Requires a valid baseline established by ``compute_inventory()`` and
        tracked by ``s_initialized``.
4. Batch inventory updates in ``batch_update_inventory()``:
   - Restores or deducts inventory for vehicles removed or added by destroy and
     repair operators.
   - Uses the same ``day..H`` incremental logic as ``veh_loading()`` with the
     opposite sign.
5. Important invariants:
   - ``s_ikt[(plant, sku, 0)]`` is the initial inventory from
     ``historical_s_ikt``.
   - For ``day > 0``:
     ``s_ikt[plant, sku, day] = s_ikt[plant, sku, day-1]
     + production(plant, sku, day) - shipped(plant, sku, day)``.
   - Incremental loading preserves this invariant when combined with the
     baseline inventory.
6. Typical call order:
   - Initialize initial inventory in ``__post_init__``.
   - Recompute inventory on first use through ``compute_inventory()``.
   - Load vehicles through ``veh_loading()``, then update inventory incrementally.
   - Use ``batch_update_inventory()`` after destroy or repair changes vehicles.
   - Evaluate ``objective()`` with caching and validation-aware inventory checks.

Maintenance notes:
- Call ``mark_objective_dirty()`` after any operator that changes vehicles or
  inventory.
- Any finer-grained rollback, such as a partial SKU rollback, must preserve the
  same ``day..H`` update consistency.
"""

# =========================
# Standard library
# =========================
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

# =========================
# Third-party libraries
# =========================
import numpy as np
import numpy.random as rnd

from .alns_config import default_config as ALNSConfig

# =========================
# Internal project modules
# =========================
from .alnstrack import ALNSTracker
from .initial_solution_utils import improved_initial_solution
from .InputDataALNS import DataALNS
from .inventory_manager import UnifiedInventoryManager
from .ml_operator_selector import MLOperatorSelector
from .optutility import POSITIVEEPS
from .param_tuner import ParamAutoTuner
from .swallowed import record_swallowed
from .vehicle import SiktDict, Vehicle, inventory_stamp, note_solution_mutation


@dataclass
class SolutionState:
    data: DataALNS
    vehicles: list[Vehicle] = field(default_factory=list)
    s_ikt: dict[tuple[str, str, int], int] = field(default_factory=dict)
    s_indices: set[tuple[str, str, int]] = field(default_factory=set)
    tracker: Optional["ALNSTracker"] = (
        None  # Optional tracker for iteration history used by ML training.
    )
    s_initialized: bool = field(default=False)  # Whether s_ikt is initialized.
    param_tuner: Optional["ParamAutoTuner"] = None  # Automatic parameter tuner.
    ml_selector: Optional["MLOperatorSelector"] = None  # ML operator selector.
    inventory_manager: Any | None = field(default=None, repr=False, compare=False)
    _inventory_cache: Any | None = field(default=None, repr=False, compare=False)
    _suppress_compute_inventory: bool = field(default=False, repr=False, compare=False)
    repair_audit: dict[str, dict[str, Any]] = field(
        default_factory=dict, repr=False, compare=False
    )
    # ---- Objective cache ----
    shipped_cache: dict[tuple[str, str], int] | None = field(
        default=None
    )  # cache for compute_shipped
    last_objective: float | None = field(default=None)  # Last objective value.
    objective_dirty: bool = field(
        default=True
    )  # Whether vehicle or inventory changes require objective recomputation.

    def __post_init__(self):
        self.vehicles = []
        # Tracked mapping: its revision lets the inventory fast path prove that a
        # cached array derived from s_ikt is still current (see SiktDict).
        self.s_ikt = SiktDict()
        self.s_indices = self.construct_indices()
        self._iteration_count = 0  # Iteration counter used by ALNSTracker.
        if self.inventory_manager is None:
            self.inventory_manager = UnifiedInventoryManager()

        # Initialize inventory; day zero stores the initial inventory.
        for (plant, sku_id, day), inv in self.data.historical_s_ikt.items():
            if day == 0:
                self.s_ikt[plant, sku_id, day] = inv

    def set_tracker(self, tracker: "ALNSTracker"):
        """Set the tracker reference used by ML-based operators."""
        self.tracker = tracker

    def set_param_tuner(self, param_tuner: "ParamAutoTuner"):
        """Set the automatic parameter tuner reference."""
        self.param_tuner = param_tuner

    def set_ml_selector(self, ml_selector: "MLOperatorSelector"):
        """Set the ML operator selector reference."""
        self.ml_selector = ml_selector

    def iter_used_vehicles(self):
        """Iterate over non-empty vehicles in the current state."""
        for veh in self.vehicles:
            try:
                if veh is None or veh.is_empty():
                    continue
            except Exception:
                continue
            yield veh

    def count_used_vehicles(self) -> int:
        """Return the number of non-empty vehicles currently in the state."""
        return sum(1 for _ in self.iter_used_vehicles())

    def replace_vehicles(self, vehicles: list[Vehicle]):
        """Replace the vehicle list and invalidate all derived caches."""
        self.vehicles = list(vehicles or [])
        self.s_initialized = False
        self.mark_objective_dirty()

    # ---- compute_inventory suppression for batch and simulation scenarios ----
    def suppress_compute_inventory(self):
        """
        Suppress inventory recomputation in later validate() and objective() calls.

        Callers should call resume_compute_inventory() after batch changes or use
        the context manager so suppression is restored automatically.
        """
        setattr(self, "_suppress_compute_inventory", True)

    def resume_compute_inventory(self):
        """Restore normal compute_inventory() calls in validate() and objective()."""
        setattr(self, "_suppress_compute_inventory", False)

    def _inventory_is_fresh(self):
        """
        Whether ``s_ikt`` provably reflects the current vehicle set.

        The fast inventory path records the mutation stamp it built its cached
        shipments from (``_inventory_cache["stamp"]``) and rebuilds whenever that
        stamp moves. Comparing stamps is O(1); the previous version re-derived a
        per-vehicle digest here, which profiling on the medium instance showed
        costing ~1.6 ms - as much as the inventory rebuild it was guarding, so the
        guard nearly cancelled the benefit it existed for.

        The stamp is bumped by every declared solution mutation
        (``SolutionState.mark_objective_dirty``, which all accepted mutations call
        including the structural ones) and by every in-place cargo change
        (``CargoDict``). It can only move *more* often than the truth, which costs
        a rebuild - never a stale reuse.

        Deliberately conservative: it returns False (forcing the recompute) on
        anything unexpected - no cache, no recorded stamp, or a mismatch.
        """
        cache = getattr(self, "_inventory_cache", None)
        if not cache:
            return False
        try:
            recorded = cache.get("stamp")
            if recorded is None or cache.get("shipments_arr") is None:
                return False
            return inventory_stamp(getattr(self, "vehicles", ())) == recorded
        except Exception:
            # Any doubt at all means "not provably fresh", so recompute.
            return False

    def suppress_inventory_ctx(self):
        """
        Return a context manager that temporarily suppresses compute_inventory().

        Usage:
            with state.suppress_inventory_ctx():
                ... batch changes will not trigger compute_inventory ...
            # Suppression is restored when the context exits.
        """

        class _Ctx:
            def __init__(self, state):
                self._state = state

            def __enter__(self):
                setattr(self._state, "_suppress_compute_inventory", True)
                return self._state

            def __exit__(self, exc_type, exc, tb):
                try:
                    setattr(self._state, "_suppress_compute_inventory", False)
                except Exception:
                    pass

        return _Ctx(self)

    # ---- Objective cache invalidation ----
    def mark_objective_dirty(self, clear_shipped: bool = True):
        """
        Mark the current objective cache as stale.

        Parameters
        ----------
        clear_shipped : bool
            When True, also clear ``shipped_cache`` for backward compatibility.
            When False, invalidate the objective and related state caches while
            preserving ``shipped_cache`` for incremental maintenance.
        """
        self.objective_dirty = True
        # Bump the declared-mutation revision. Every accepted solution mutation
        # passes through here - repair and destroy operators call it, and the few
        # direct ``state.vehicles[i] = ...`` sites call it immediately afterwards.
        # Cargo changes are counted separately by ``CargoDict``. Together they let
        # ``_inventory_is_fresh`` answer in O(1) instead of re-deriving a
        # per-vehicle signature, which profiling showed cost as much as the
        # inventory rebuild it was guarding.
        note_solution_mutation()
        # Clear the last objective; shipped_cache is controlled by clear_shipped.
        self.last_objective = None
        if clear_shipped:
            self.shipped_cache = None
        # Clear state-dependent aggregation caches so they are rebuilt consistently.
        try:
            if hasattr(self, "_vehicles_by_plant_day_map"):
                delattr(self, "_vehicles_by_plant_day_map")
        except Exception:
            pass
        try:
            if hasattr(self, "_resource_pool_cache"):
                delattr(self, "_resource_pool_cache")
        except Exception:
            pass
        try:
            if hasattr(self, "_unmet_cache"):
                delattr(self, "_unmet_cache")
        except Exception:
            pass

    def record_repair_audit(
        self,
        op_name: str,
        feasible: bool,
        violations: dict[str, Any] | None = None,
        rolled_back: bool = False,
        inserted_cnt: int = 0,
        improvement: float = 0.0,
    ):
        """Aggregate repair feasibility diagnostics per operator."""
        name = str(op_name or "unknown_repair")
        entry = self.repair_audit.setdefault(
            name,
            {
                "attempts": 0,
                "feasible": 0,
                "infeasible": 0,
                "rolled_back": 0,
                "inserted_total": 0,
                "improvement_total": 0.0,
                "fail_negative_inventory": 0,
                "fail_veh_over_load": 0,
                "fail_plant_inv_exceed": 0,
                "fail_unmet_demand": 0,
            },
        )
        entry["attempts"] += 1
        if feasible:
            entry["feasible"] += 1
        else:
            entry["infeasible"] += 1
        if rolled_back:
            entry["rolled_back"] += 1
        entry["inserted_total"] += int(max(0, inserted_cnt))
        entry["improvement_total"] += float(improvement)

        viol = violations or {}
        if viol.get("negative_inventory"):
            entry["fail_negative_inventory"] += 1
        if viol.get("veh_over_load"):
            entry["fail_veh_over_load"] += 1
        if viol.get("plant_inv_exceed"):
            entry["fail_plant_inv_exceed"] += 1
        if viol.get("unmet_demand"):
            entry["fail_unmet_demand"] += 1

    def validate(self):
        """
        Validate solution feasibility without mutating solution content.

        Returns
        -------
        tuple
            ``(is_feasible, violations)`` where ``is_feasible`` is a boolean and
            ``violations`` contains detailed constraint violations.
        """
        # In strict feasibility mode the inventory must be recomputed, so that
        # suppressed updates cannot leave stale inventory that is then accepted.
        # The exception is when the fast path proves ``s_ikt`` already matches the
        # current vehicle set (see ``_inventory_is_fresh``): recomputing cannot
        # change the result, it only costs time - and on the medium instance that
        # cost dominated the entire run.
        #
        # So: recompute unless strict mode is on AND freshness is proven.
        force_recompute = bool(
            getattr(ALNSConfig, "VALIDATE_FORCE_RECOMPUTE_INVENTORY", True)
            or getattr(ALNSConfig, "STRICT_FEASIBILITY_MODE", True)
        )
        suppressed = bool(getattr(self, "_suppress_compute_inventory", False))
        if not suppressed and not (force_recompute and self._inventory_is_fresh()):
            self.compute_inventory()
        violations = {
            "negative_inventory": [],
            "veh_over_load": [],
            "plant_inv_exceed": [],
            "unmet_demand": [],
            "cargo_day_mismatch": [],
            # A check that could not be *completed* (as opposed to one that found
            # a problem). Kept separate so "we could not measure this" is never
            # reported as "this is overloaded"; it still makes the state
            # infeasible, because an unverified state is not a feasible one.
            "check_error": [],
        }

        # Use NumPy/Numba arrays when available; otherwise fall back to dict iteration.
        try:
            from . import inventory_numba

            maps = getattr(self, "_inventory_maps", None)
            if maps is None:
                maps = inventory_numba.build_index_maps(self.data)
            # Build s_arr with shape (P, S, H + 1).
            s_arr, _, _ = inventory_numba.state_to_inventory_arrays(self, maps)
            # Locate negative inventory entries.
            neg_idx = np.where(s_arr < 0)
            for pi, si, d in zip(*neg_idx):
                plant = maps.idx_to_plant[pi]
                sku = maps.idx_to_sku[si]
                inv = float(s_arr[pi, si, d])
                violations["negative_inventory"].append(((plant, sku, int(d)), inv))
        except Exception:
            # Fall back to direct dict traversal.
            for key, inv in self.s_ikt.items():
                if inv < 0:
                    violations["negative_inventory"].append((key, inv))

        # Check vehicle capacity limits.
        for veh in self.vehicles:
            try:
                for sku_id, cargo_day in veh.cargo.keys():
                    if int(cargo_day) != int(veh.day):
                        violations["cargo_day_mismatch"].append(
                            {
                                "veh_id": getattr(veh, "id", None),
                                "vehicle_day": int(veh.day),
                                "sku_id": sku_id,
                                "cargo_day": int(cargo_day),
                            }
                        )
                loaded = self.compute_veh_load(veh)
                cap = self.data.veh_type_cap[veh.type]
                if loaded - cap > POSITIVEEPS:
                    info = {
                        "veh_id": getattr(veh, "id", None),
                        "fact_id": getattr(veh, "fact_id", None),
                        "dealer_id": getattr(veh, "dealer_id", None),
                        "type": getattr(veh, "type", None),
                        "day": getattr(veh, "day", None),
                        "loaded": loaded,
                        "cap": cap,
                        "cargo": dict(getattr(veh, "cargo", {})),
                    }
                    violations["veh_over_load"].append(info)
                    # Overload is a solution-feasibility issue, not a code-error signal.
            except Exception as exc:
                # A failure while *measuring* the load is not an overload: the
                # vehicle's load is unknown. Recording it under ``veh_over_load``
                # (as before) meant a code defect triggered the rollback path with
                # the reason "vehicle overloaded", which sends anyone reading the
                # violations - or the audit report - after the wrong cause.
                #
                # It stays a hard violation, so the state remains infeasible: an
                # unmeasured load cannot be certified as feasible.
                record_swallowed("alnsopt.validate.vehicle_load", exc)
                violations["check_error"].append(
                    {
                        "veh_id": getattr(veh, "id", None),
                        "check": "vehicle_load",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        # Check whether plant inventory exceeds its limit in any period.
        # Aggregate with arrays when possible and fall back to dict aggregation.
        try:
            from . import inventory_numba

            maps = getattr(self, "_inventory_maps", None)
            if maps is None:
                maps = inventory_numba.build_index_maps(self.data)
            s_arr, _, _ = inventory_numba.state_to_inventory_arrays(self, maps)
            # sum over sku axis -> shape (P, H+1)
            plant_day_tot = s_arr.sum(axis=1)
            for pi, plant in enumerate(maps.idx_to_plant):
                for d in range(plant_day_tot.shape[1]):
                    total_inv = float(plant_day_tot[pi, d])
                    max_cap = self.data.plant_inv_limit.get(plant, float("inf"))
                    if total_inv - max_cap > POSITIVEEPS:
                        violations["plant_inv_exceed"].append(
                            {
                                "plant": plant,
                                "day": int(d),
                                "total_inv": total_inv,
                                "max_cap": max_cap,
                            }
                        )
        except Exception:
            plant_day_inventory = defaultdict(int)
            for (plant, sku, day), inv in self.s_ikt.items():
                plant_day_inventory[(plant, day)] += inv

            for (plant, day), total_inv in plant_day_inventory.items():
                max_cap = self.data.plant_inv_limit[plant]
                if total_inv - max_cap > POSITIVEEPS:
                    violations["plant_inv_exceed"].append(
                        {
                            "plant": plant,
                            "day": day,
                            "total_inv": total_inv,
                            "max_cap": max_cap,
                        }
                    )

        # In strict mode, fully satisfying demand is a hard feasibility condition.
        require_full_demand = bool(
            getattr(ALNSConfig, "VALIDATE_REQUIRE_FULL_DEMAND", True)
            or getattr(ALNSConfig, "STRICT_FEASIBILITY_MODE", True)
        )
        if require_full_demand:
            shipped = self.compute_shipped()
            for (dealer, sku_id), demand in self.data.demands.items():
                shipped_qty = shipped.get((dealer, sku_id), 0)
                if shipped_qty + POSITIVEEPS < demand:
                    violations["unmet_demand"].append(
                        {
                            "dealer": dealer,
                            "sku_id": sku_id,
                            "demand": demand,
                            "shipped": shipped_qty,
                        }
                    )

        # Combine all violation categories into the final feasibility decision.
        is_feasible = not (
            violations["negative_inventory"]
            or violations["veh_over_load"]
            or violations["plant_inv_exceed"]
            or violations["unmet_demand"]
            or violations["cargo_day_mismatch"]
            or violations["check_error"]
        )

        return is_feasible, violations

    def objective(self) -> float:
        """
        Return the cached search objective used by ALNS guidance and acceptance.

        The value starts from the actual objective and adds search-time guidance
        penalties for unmet demand, fleet size, and underloaded vehicles. This is
        not the external evaluation metric; use ``actual_objective()`` for that.
        """
        # Return the cached value when it is still valid.
        if (not self.objective_dirty) and (self.last_objective is not None):
            return self.last_objective

        scale_factor = float(getattr(ALNSConfig, "SCALE_FACTOR", 1e-3))

        # validate() triggers compute_inventory() when needed.
        feasibility, violations = self.validate()
        hard_violation = bool(
            violations.get("negative_inventory")
            or violations.get("veh_over_load")
            or violations.get("plant_inv_exceed")
            or violations.get("cargo_day_mismatch")
            # An incomplete check leaves the state unverified, so treat it as
            # physical uncertainty rather than as guidance.
            or violations.get("check_error")
        )
        # A destroyed state may be temporarily short of demand. It is still a
        # useful search candidate because the demand penalty below is finite.
        # Physical violations, by contrast, must never be accepted.
        if hard_violation:
            self.last_objective = float("inf")
            self.objective_dirty = False
            return self.last_objective

        data = self.data

        # ---- Base term: fixed cost plus minimum-load penalty ----
        # Compute this directly instead of calling actual_objective(), which
        # treats unmet demand as infeasible. The demand gap is priced by the
        # finite guidance term below so ALNS can navigate through partial
        # repairs during a destroy-repair step.
        fixed_cost, underload_vehicle_count, _ = self._vehicle_cost_components()
        if self.data.param_pun_objective > 0.0:
            minimum_load_penalty = float(underload_vehicle_count) * float(
                self.data.param_pun_objective
            )
        else:
            minimum_load_penalty = 0.0
            for veh in self.iter_used_vehicles():
                loaded = self.compute_veh_load(veh)
                min_load = float(self.data.veh_type_min_load.get(veh.type, 0.0))
                if loaded < min_load:
                    minimum_load_penalty += float(self.data.param_alpha) * float(
                        self.data.veh_type_cost.get(veh.type, 0.0)
                    )
        actual_base = fixed_cost + minimum_load_penalty

        # Use phase-dependent weights: compress fleet size early and return to
        # the composite cost later.
        progress = 0.0
        if bool(getattr(ALNSConfig, "SEARCH_PHASED_VEHICLE_PRIORITY", True)):
            try:
                if self.tracker is not None:
                    cur_it = int(self.tracker.get_iteration())
                    max_it = int(self.tracker.get_max_iterations())
                    if max_it > 0:
                        progress = max(0.0, min(1.0, float(cur_it) / float(max_it)))
            except Exception:
                progress = 0.0

        phase1_end = float(getattr(ALNSConfig, "SEARCH_PHASE1_END_PROGRESS", 0.45))
        phase2_end = float(getattr(ALNSConfig, "SEARCH_PHASE2_END_PROGRESS", 0.80))
        if progress < phase1_end:
            demand_weight = float(
                getattr(ALNSConfig, "SEARCH_PHASE1_DEMAND_WEIGHT", 0.35)
            )
            underload_weight = float(
                getattr(ALNSConfig, "SEARCH_PHASE1_UNDERLOAD_WEIGHT", 0.80)
            )
            fleet_weight = float(
                getattr(ALNSConfig, "SEARCH_PHASE1_FLEET_WEIGHT", 4.00)
            )
        elif progress < phase2_end:
            demand_weight = float(
                getattr(ALNSConfig, "SEARCH_PHASE2_DEMAND_WEIGHT", 0.80)
            )
            underload_weight = float(
                getattr(ALNSConfig, "SEARCH_PHASE2_UNDERLOAD_WEIGHT", 1.00)
            )
            fleet_weight = float(
                getattr(ALNSConfig, "SEARCH_PHASE2_FLEET_WEIGHT", 2.00)
            )
        else:
            demand_weight = float(
                getattr(ALNSConfig, "SEARCH_PHASE3_DEMAND_WEIGHT", 1.00)
            )
            underload_weight = float(
                getattr(ALNSConfig, "SEARCH_PHASE3_UNDERLOAD_WEIGHT", 1.00)
            )
            fleet_weight = float(
                getattr(ALNSConfig, "SEARCH_PHASE3_FLEET_WEIGHT", 1.00)
            )

        # ---- Guidance term 1: unmet-demand penalty for search guidance ----
        demand_penalty = 0.0
        unmet_total = 0.0
        demand_total = 0.0
        if bool(getattr(ALNSConfig, "SEARCH_USE_DEMAND_PENALTY", True)):
            shipped = self.compute_shipped()
            for (dealer, sku_id), demand in data.demands.items():
                demand_total += float(demand)
                shipped_qty = shipped.get((dealer, sku_id), 0)
                if shipped_qty < demand:
                    gap = float(demand - shipped_qty)
                    unmet_total += gap
                    demand_penalty += data.param_pun_factor1 * gap
            demand_penalty *= demand_weight

        # ---- Guidance term 2: minimum-load gap penalty for search guidance ----
        underload_gap_penalty = 0.0
        if bool(getattr(ALNSConfig, "SEARCH_USE_UNDERLOAD_GAP_PENALTY", True)):
            _, _, underload_gap_penalty = self._vehicle_cost_components()
            underload_gap_penalty *= underload_weight

        # ---- Guidance term 3: explicit fleet-size penalty ----
        fleet_size_penalty = 0.0
        if bool(getattr(ALNSConfig, "ENABLE_FLEET_SIZE_PENALTY", True)):
            fleet_size = self.count_used_vehicles()
            baseline = int(getattr(ALNSConfig, "FLEET_SIZE_BASELINE", 25))
            penalty_rate = float(
                getattr(ALNSConfig, "FLEET_SIZE_PENALTY_PER_VEHICLE", 15.0)
            )
            if fleet_size > baseline:
                fleet_size_penalty = (fleet_size - baseline) * penalty_rate
            fleet_size_penalty *= fleet_weight

        # Two-phase shaping: when unmet demand exists, prioritize feasibility over fleet compression.
        if (
            bool(getattr(ALNSConfig, "SEARCH_FEASIBILITY_FIRST", True))
            and unmet_total > 0.0
        ):
            demand_boost = float(
                getattr(ALNSConfig, "SEARCH_FEASIBILITY_DEMAND_BOOST", 6.0)
            )
            fleet_scale = float(
                getattr(ALNSConfig, "SEARCH_FEASIBILITY_FLEET_SCALE_WHEN_UNMET", 0.15)
            )
            underload_scale = float(
                getattr(
                    ALNSConfig,
                    "SEARCH_FEASIBILITY_UNDERLOAD_SCALE_WHEN_UNMET",
                    0.25,
                )
            )
            demand_penalty *= max(1.0, demand_boost)
            fleet_size_penalty *= max(0.0, fleet_scale)
            underload_gap_penalty *= max(0.0, underload_scale)

        search_cost = (
            float(actual_base)
            + float(demand_penalty)
            + float(underload_gap_penalty)
            + float(fleet_size_penalty)
        )

        self.last_objective = search_cost * scale_factor
        self.objective_dirty = False
        return float(self.last_objective)

    def _vehicle_cost_components(self) -> tuple[float, int, float]:
        """Return fixed cost, underloaded count, and underload gap penalty."""
        data = self.data
        fixed_cost = 0.0
        underload_vehicle_count = 0
        underload_gap_penalty = 0.0

        for veh in self.iter_used_vehicles():
            fixed_cost += float(data.veh_type_cost.get(veh.type, 0.0))

            loaded = getattr(veh, "_loaded_volume", None)
            if loaded is None:
                lv = 0.0
                for (sku_id, _day), qty in veh.cargo.items():
                    lv += float(data.sku_sizes.get(sku_id, 0.0)) * float(qty)
                loaded = lv

            min_load = float(data.veh_type_min_load.get(veh.type, 0.0))
            if float(loaded) < min_load:
                underload_vehicle_count += 1
                underload_gap_penalty += float(data.param_pun_factor3) * (
                    min_load - float(loaded)
                )

        return fixed_cost, underload_vehicle_count, underload_gap_penalty

    def actual_objective(self) -> float:
        """
        Return the objective aligned with the ILP evaluation metric.

        The value equals vehicle fixed cost plus the minimum-load penalty. By
        default, the penalty follows the thesis formula
        ``alpha * vehicle_cost`` for every used vehicle whose load is below its
        minimum shipment quantity. ``param_pun_objective`` remains available as
        a legacy override when explicitly configured.
        """
        feasibility, _ = self.validate()
        if not feasibility:
            return float("inf")

        fixed_cost, underload_vehicle_count, _ = self._vehicle_cost_components()
        if self.data.param_pun_objective > 0.0:
            punishment_cost = float(underload_vehicle_count) * float(
                self.data.param_pun_objective
            )
        else:
            punishment_cost = 0.0
            for veh in self.iter_used_vehicles():
                loaded = self.compute_veh_load(veh)
                min_load = float(self.data.veh_type_min_load.get(veh.type, 0.0))
                if loaded < min_load:
                    punishment_cost += float(self.data.param_alpha) * float(
                        self.data.veh_type_cost.get(veh.type, 0.0)
                    )
        return float(fixed_cost + punishment_cost)

    def calculate_cost(self) -> float:
        """Backward-compatible alias for the Gurobi-aligned actual objective."""
        return float(self.actual_objective())

    def punish_non_fulfill_demand(self, obj: float):
        """Add unmet-demand penalties to the supplied objective value."""
        shipped = self.compute_shipped()
        for (dealer, sku_id), demand in self.data.demands.items():
            shipped_qty = shipped.get((dealer, sku_id), 0)
            if shipped_qty < demand:
                obj += self.data.param_pun_factor1 * (demand - shipped_qty)
        return obj

    def compute_veh_load(self, veh: Vehicle) -> float:
        """Return the vehicle load, using the ``_loaded_volume`` cache when valid."""
        # Prefer the cached loaded volume maintained by operators when available.
        loaded = getattr(veh, "_loaded_volume", None)
        if loaded is not None:
            return loaded
        # Fall back to cargo traversal for vehicles without a maintained cache.
        total_volume = 0
        for (sku_id, day), qty in veh.cargo.items():
            total_volume += self.data.sku_sizes[sku_id] * qty
        return total_volume

    def punish_deficient_veh_min_load(self, obj: float):
        """Add penalties for vehicles below their minimum load requirement."""
        for veh in self.vehicles:
            total_volume = self.compute_veh_load(veh)
            min_load = self.data.veh_type_min_load[veh.type]
            if total_volume < min_load:
                obj += self.data.param_pun_factor3 * (min_load - total_volume)
        return obj

    def compute_shipped(self) -> dict[tuple[str, str], int]:
        """
        Compute shipped quantity for each dealer and SKU across all periods.

        Optimization notes:
          - Aggregate with a local defaultdict to reduce dict.get calls in hot paths.
          - Avoid unnecessary dict.copy() calls and convert to a regular dict only
            when storing the cache.

        Validity is decided by the O(1) mutation stamp, not by ``objective_dirty``.
        Shipped depends only on the vehicles' cargo, so it is exactly as current as
        the vehicles are - and ``objective_dirty`` is a *different* question. Using
        it here made this function recompute almost every call: candidate
        evaluation calls ``mark_objective_dirty`` (which sets the flag) far more
        often than it actually changes a load. Profiling showed 12,281 calls
        costing ~9.6 s in a 120 s run.

        ``mark_objective_dirty(clear_shipped=True)`` still clears ``shipped_cache``
        for callers that relied on that; the stamp then guarantees a rebuild.
        """
        cached = getattr(self, "shipped_cache", None)
        if cached is not None and getattr(self, "_shipped_stamp", None) == inventory_stamp(
            getattr(self, "vehicles", ())
        ):
            return cached

        # Use local variables and defaultdict to speed up aggregation.
        shipped_agg = defaultdict(int)
        vehicles = self.vehicles  # local ref for speed
        for veh in vehicles:
            dealer = veh.dealer_id
            # Micro-optimization: avoid tuple allocations in the inner loop.
            for (sku_id, day), qty in veh.cargo.items():
                shipped_agg[(dealer, sku_id)] += qty

        # Convert to a regular dict and cache the result.
        result = dict(shipped_agg)
        try:
            self.shipped_cache = result
            self._shipped_stamp = inventory_stamp(vehicles)
        except (AttributeError, TypeError):
            # Best effort: without the cache the next call recomputes.
            pass
        return result

    def construct_indices(self):
        """
        Construct and cache the (plant, sku, day) index set.
        Cache the intermediate plant-sku pairs on the state instance so repeated
        calls (e.g. during long-running profiling) avoid reconstructing supply
        chain mappings.
        """
        if getattr(self, "_plant_sku_pairs", None) is None:
            triple_plant_dealer_sku = {
                (plant, dealer, sku_id)
                for (plant, dealer), skus in self.data.construct_supply_chain().items()
                for sku_id in skus
            }
            # cache reduced (plant, sku) pairs for reuse
            self._plant_sku_pairs = {
                (plant, sku_id) for (plant, dealer, sku_id) in triple_plant_dealer_sku
            }
        s_indices = {
            (plant, sku_id, day)
            for (plant, sku_id) in self._plant_sku_pairs
            for day in range(self.data.horizons + 1)
        }
        return s_indices

    def compute_inventory(self):
        """
        Recompute the full inventory table.

        Use the array/Numba path when available and fall back to the pure-Python
        implementation without changing behavior or semantics.
        """
        # Try the fast path using
        # multi_period_delivery.alns.inventory_numba.compute_inventory_fast
        # when available.
        try:
            try:
                from . import inventory_numba

                # inventory_numba.compute_inventory_fast returns True on success
                res = inventory_numba.compute_inventory_fast(self)
                if not res:
                    raise RuntimeError("inventory_numba fast path failed")
                # ensure s_initialized semantics preserved by fallback/core
                try:
                    self.s_initialized = True
                except (TypeError, ValueError):
                    pass
                return res
            except Exception:
                # If import or fast path fails, fall through to pure-Python baseline below
                pass

            # --- Pure-Python baseline (original implementation) ---
            # 1) Precompute same-day shipments by (plant, sku, day) to avoid
            # scanning the vehicle list for every inventory index.
            shipped_by_plant_sku_day = defaultdict(int)
            for veh in self.vehicles:
                fact = veh.fact_id
                d = veh.day
                for (sku_id, day_k), q in veh.cargo.items():
                    # Count same-day shipments only.
                    if day_k == d:
                        shipped_by_plant_sku_day[(fact, sku_id, day_k)] += q

            # To keep the recursion correct, iterate by (plant, sku) in ascending
            # day order instead of relying on unordered self.s_indices. Otherwise
            # day-1 may be unavailable and produce an incorrect prev_inventory=0.
            plant_sku_pairs = {(plant, sku) for (plant, sku, d) in self.s_indices}
            for plant, sku_id in plant_sku_pairs:
                for day in range(1, self.data.horizons + 1):  # Only day > 0.
                    # Read same-day shipments from the precomputed table.
                    shipped_from_plant = shipped_by_plant_sku_day.get(
                        (plant, sku_id, day), 0
                    )

                    # Use the previous period inventory, including initial stock.
                    prev_inventory = self.s_ikt.get((plant, sku_id, day - 1), 0)

                    # Read same-day production.
                    production = self.data.sku_prod_each_day.get(
                        (plant, sku_id, day), 0
                    )

                    # current = previous inventory + production - shipments.
                    current_inventory = prev_inventory + production - shipped_from_plant

                    # Store the updated inventory.
                    self.s_ikt[(plant, sku_id, day)] = current_inventory

            # Mark s_ikt as initialized after all future periods are computed.
            self.s_initialized = True
            try:
                self._inventory_cache = None
            except (TypeError, ValueError):
                pass
            return True
        finally:
            # ensure objective semantics: callers expect compute_inventory to set related caches as necessary
            try:
                # calling mark_objective_dirty false here would be incorrect; preserve existing behavior by not forcing it
                pass
            except (TypeError, ValueError):
                pass

    def copy(self, clone_vehicles: bool = True):
        """
        Lightweight copy of the current state without invoking __post_init__.

        Parameters:
        - clone_vehicles: when True (default) create shallow clones of each Vehicle using
          Vehicle.shallow_clone() to ensure the returned state is fully independent.
          When False, perform a shallow list copy of the vehicles list (i.e. new list object
          but sharing Vehicle instances). The latter is much cheaper and safe for read-mostly
          simulation scenarios where callers guarantee not to mutate existing Vehicle objects.

        Use object.__new__ to avoid the overhead of constructing a fresh SolutionState
        (which would recompute indices and reinitialize structures). Fields are shallow-
        copied where safe.
        """
        new_state = object.__new__(SolutionState)
        # shallow copy simple containers / references
        new_state.data = self.data
        # Vehicles: either clone each vehicle (safe but more expensive) or shallow-copy list
        if clone_vehicles:
            new_state.vehicles = [v.shallow_clone() for v in self.vehicles]
        else:
            # create a new list object but reuse vehicle instances (caller must avoid mutating them)
            new_state.vehicles = list(self.vehicles)
        # copy inventory map (caller may mutate s_ikt during simulation). Built as
        # a tracked mapping so those later writes keep being counted; a plain
        # ``dict`` copy would silently stop the revision from moving and the
        # cached array could go stale.
        new_state.s_ikt = SiktDict(self.s_ikt)
        # preserve the exact s_indices structure (triples) to avoid unpacking errors
        new_state.s_indices = set(self.s_indices)
        # preserve external references
        new_state.tracker = self.tracker
        new_state.param_tuner = self.param_tuner
        new_state.ml_selector = self.ml_selector
        new_state.inventory_manager = self.inventory_manager
        new_state.s_initialized = bool(self.s_initialized)
        # copy caches/cached values (ship cache copied shallowly)
        new_state.shipped_cache = (
            self.shipped_cache.copy() if self.shipped_cache is not None else None
        )
        new_state.last_objective = self.last_objective
        new_state.objective_dirty = self.objective_dirty
        # preserve suppress flag
        new_state._suppress_compute_inventory = getattr(
            self, "_suppress_compute_inventory", False
        )
        new_state.repair_audit = {
            k: dict(v) for k, v in getattr(self, "repair_audit", {}).items()
        }
        return new_state


def veh_loading(state: SolutionState, veh: Vehicle, orders: dict[str, int]):
    """
    Load a vehicle and update inventory incrementally.

    Core idea:
    - Avoid calling compute_inventory() after every partial load.
    - A shipment of ``q`` units for a ``(plant, sku)`` on ``day`` reduces all
      ``s_ikt(plant, sku, d)`` entries for ``d >= day`` by ``q``.
    - Updating ``day..H`` once is equivalent to the cumulative effect of the
      full inventory recursion and avoids a full-table recomputation.

    Steps:
    1. If the inventory baseline is not initialized, call compute_inventory().
    2. For each ordered SKU:
       a. Compute available inventory as
          ``s[day-1] + production[day] - already_loaded_same_plant_day``.
       b. Convert vehicle capacity into the maximum quantity of this SKU.
       c. Set ``load_qty = min(remaining_order, available, capacity_limit)``.
       d. Call ``veh.load()`` and update ``used_inv``.
       e. Subtract ``load_qty`` from ``s_ikt`` for ``day..H``.
       f. Create another vehicle when the current one is full.
    3. Add non-empty vehicles to the solution and invalidate the objective cache.

    Complexity:
    - The previous approach recomputed ``O(P*S*H)`` inventory after each load.
    - This approach performs an ``O(H - day + 1)`` update per load.
    - This substantially reduces initial-solution and repair costs.

    Correctness depends on:
    - A baseline from compute_inventory() that includes all existing shipments.
    - Incremental deductions that update every future period consistently.
    - Symmetric batch updates when destroy/repair remove or add vehicles.

    Notes:
    - Cross-day loading is not supported; every cargo day must equal ``veh.day``.
    - Available inventory is computed from the previous period, current
      production, and already loaded quantity to avoid double deduction.
    - Any future partial rollback feature should provide a symmetric incremental
      add-back implementation.

    Returns
    -------
    bool
        True when the loading process completes successfully.
    """
    data = state.data
    fact_id, dealer_id, veh_type, day = veh.fact_id, veh.dealer_id, veh.type, veh.day

    # Establish the inventory baseline before incremental deductions.
    if not state.s_initialized:
        state.compute_inventory()

    # Build used_inv and vehicle_ids in one pass.
    used_inv = {}
    vehicles_list = state.vehicles
    vehicle_ids = set()
    for v in vehicles_list:
        if v.fact_id != fact_id or v.day != day:
            continue
        vehicle_ids.add(v.id)
        for (sku_id, d), qty in v.cargo.items():
            if d == day:
                used_inv[sku_id] = used_inv.get(sku_id, 0) + qty

    # prepare locals for hot loop
    s_ikt = state.s_ikt
    sku_sizes = data.sku_sizes
    prod_get = data.sku_prod_each_day.get
    horizons = data.horizons

    for sku_id, order_qty in orders.items():
        if sku_id not in data.skus_plant.get(fact_id, []):
            continue
        remain_qty = order_qty

        while remain_qty > 0:
            # Compute available inventory.
            prev_inv = s_ikt.get((fact_id, sku_id, day - 1), 0)
            production = prod_get((fact_id, sku_id, day), 0)
            used_qty = used_inv.get(sku_id, 0)
            available = prev_inv + production - used_qty
            if available <= 0:
                break

            # Use veh.capacity as remaining volume to avoid recomputing current_load.
            sku_size = sku_sizes[sku_id]
            max_qty_by_cap = veh.capacity // sku_size  # Max SKU quantity by capacity.
            # Residual demand, available inventory, and capacity bound the load.
            load_qty = int(min(remain_qty, available, max_qty_by_cap))

            if load_qty <= 0:
                # The current vehicle is full; store it and switch to a new one.
                if not veh.is_empty():
                    if veh.id not in vehicle_ids:
                        state.vehicles.append(veh)
                        vehicle_ids.add(veh.id)
                veh = Vehicle(fact_id, dealer_id, veh_type, day, data)
                continue

            veh.load(sku_id, int(load_qty))
            used_inv[sku_id] = used_inv.get(sku_id, 0) + load_qty
            remain_qty -= load_qty

            # Deduct this load from the shipment day and all future periods.
            for d in range(day, horizons + 1):
                key = (fact_id, sku_id, d)
                s_ikt[key] = int(s_ikt.get(key, 0) - load_qty)

        # Add the final non-empty vehicle once, using IDs to avoid costly equality checks.
        if not veh.is_empty() and veh.id not in vehicle_ids:
            state.vehicles.append(veh)

    # Incremental updates are complete, so no full recomputation is needed.

    # Loading changed vehicles and inventory, so invalidate the objective cache.
    state.mark_objective_dirty()
    # Clear inventory fast-path cache to ensure next compute_inventory rebuilds arrays
    try:
        if hasattr(state, "_inventory_cache"):
            state._inventory_cache = None
    except Exception:
        pass
    return True


def batch_update_inventory(state: SolutionState, vehicles: list[Vehicle], op_type: str):
    """
    Batch-update inventory and return a summary of applied changes.

    Returns
    -------
    dict
        Mapping ``(plant, sku, day)`` to the applied inventory delta.

    Purpose:
    - Allow callers to verify that applied additions or deductions match intent.
    - Support logging, rollback, or diagnostics when an inconsistency occurs.

    Notes:
    - For ``destroy``, add removed vehicle cargo back to inventory for
      ``day..H`` using positive deltas.
    - For ``repair``, deduct new vehicle cargo from inventory for ``day..H``
      using negative deltas.
    """
    inventory_updates = defaultdict(int)
    if not vehicles:
        return inventory_updates

    # Ensure inventory has been initialized.
    if not state.s_initialized:
        state.compute_inventory()

    # Aggregate all (plant, sku, day) inventory deltas.
    for veh in vehicles:
        for (sku_id, d_shipped), qty in veh.cargo.items():
            # Collect every inventory change that must be applied.
            for d in range(d_shipped, state.data.horizons + 1):
                key = (veh.fact_id, sku_id, d)
                if op_type == "destroy":
                    inventory_updates[key] += qty
                elif op_type == "repair":
                    inventory_updates[key] -= qty

    # Apply all updates to state.s_ikt.
    s_ikt_get = state.s_ikt.get
    for key, delta in inventory_updates.items():
        state.s_ikt[key] = s_ikt_get(key, 0) + delta

    # Batch inventory changes invalidate the objective cache.
    state.mark_objective_dirty()
    # Clear inventory fast-path cache to ensure next compute_inventory rebuilds arrays
    try:
        if hasattr(state, "_inventory_cache"):
            state._inventory_cache = None
    except Exception:
        pass

    # Return the summary for caller verification and logging.
    return dict(inventory_updates)


def initial_solution(state: SolutionState, rng: rnd.Generator):
    """generate initial solution"""
    solution = improved_initial_solution(state, rng)
    return solution
