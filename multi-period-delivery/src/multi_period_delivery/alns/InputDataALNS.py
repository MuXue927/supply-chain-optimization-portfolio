"""
Purpose and design:
- This module manages the static input data required by the ALNS solver. It
  loads and standardizes data files, validates them, computes derived mappings
  and summary metrics, and exposes the unified DataALNS container used by
  initial-solution generation, destroy and repair operators, inventory
  evolution, objective evaluation, and the ML selector.
- The module separates I/O and data-cleaning logic from algorithm logic and
  provides deterministic, type-aware query methods to reduce duplicated
  computation and inconsistent data handling downstream.

Core design:
- Structured pipeline: parameter loading -> file-path construction -> batch
  DataFrame loading -> normalization -> derived-data computation -> integrity
  validation. Each stage is isolated and easy to unit test.
- Type and key normalization: identifier fields such as plant, dealer, SKU, and
  vehicle type are converted to strings early to avoid subtle key mismatches.
- Derived caches: expensive or reusable mappings such as supply_chain are
  cached per instance. Clear the cache explicitly after programmatic changes to
  the underlying data.
- Light runtime dependencies: avoid importing heavy dependencies or business
  logic at module import time. Lazy imports inside functions avoid circular
  dependencies and startup overhead.

Public interface:
- DataALNS(input_file_loc, output_file_loc, dataset_name):
  - load(): runs the complete data-loading and processing pipeline. On success,
    the instance contains all mappings and sets required by the algorithm.
  - Main query methods: available_skus_in_plant(plant_id),
    plan_dealer_skus(plant_id), available_skus_to_dealer(plant_id, dealer_id),
    construct_supply_chain(), get_total_demand_for_sku(sku_id),
    get_total_production_for_sku(sku_id, day=None),
    get_total_initial_inventory_for_sku(sku_id),
    get_sku_supply_demand_balance(sku_id), and get_summary_statistics().
  - Lifecycle: call load() before accessing derived fields.

Implementation details and cautions:
- Validation: _load_csv_safely raises and logs an error when required columns
  are missing. This fail-fast behavior protects downstream consistency. Higher
  layers can catch the error to implement more complex fallback behavior.
- DataFrame to native structures: intermediate pandas results are converted to
  pure Python types such as dict and set to reduce pandas dependency on hot
  paths and improve algorithm performance.
- Cache invalidation: after modifying underlying tables or incrementally
  loading data, delete the cache or set self._supply_chain_cache = None.
- Extensibility: if the data schema becomes more complex, consider a schema
  validation library such as pydantic or jsonschema and make field mappings
  configurable.

Maintenance:
- When adding source files or changing CSV columns, update required_columns in
  _load_all_dataframes and title_map/type_mapping in _process_dataframes.
- Add unit tests for boundary cases such as empty tables and missing SKU sizes.
- In large experiments, stream or chunk large tables if memory becomes a
  bottleneck.

Import and performance conventions:
- Keep imports in three groups: standard library, third-party, and internal
  project dependencies.
- Defer heavy imports or imports with circular-dependency risk to function
  bodies.

The original implementation is preserved; module comments and import grouping
have been reorganized for maintainability.
"""

# =========================
# Standard library
# =========================
import os
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Union, Optional

# =========================
# Third-party libraries
# =========================
import pandas as pd

# =========================
# Public API
# =========================
__all__ = ["DataALNS"]


@dataclass
class DataALNS:
    """
    DataALNS
    --------
    Unified data container for the ALNS solver. It encapsulates raw-data
    loading, transformation, derivation, and validation.
    """
    input_file_loc: str
    output_file_loc: str
    dataset_name: str  # Dataset directory name, relative to input_file_loc.

    # Planning horizon. Defaults to 1 and is updated when production data
    # contains a larger day value.
    horizons: int = 1
    # Current starting day, retained for future rolling-horizon planning.
    day: int = 1

    # ==== Penalty parameters loaded from configuration ====
    param_pun_factor1: float = field(init=False)
    param_pun_factor2: float = field(init=False)
    param_pun_factor3: float = field(init=False)
    param_pun_factor4: float = field(init=False)
    param_pun_objective: float = field(init=False)
    param_alpha: float = field(init=False)

    # ==== Base sets and attributes ====
    plants: Set[str] = field(init=False)
    dealers: Set[str] = field(init=False)
    all_skus: Set[str] = field(init=False)
    all_veh_types: Set[str] = field(init=False)
    sku_sizes: Dict[str, float] = field(init=False)

    # Production: daily and aggregate views
    sku_prod_each_day: Dict[Tuple[str, str, int], int] = field(init=False)   # (plant, sku, day) -> qty
    sku_prod_total: Dict[Tuple[str, int], int] = field(init=False)           # (sku, day) -> aggregated qty

    # Inventory
    sku_initial_inv: Dict[Tuple[str, str], int] = field(init=False)          # (plant, sku) -> init qty
    skus_initial: Dict[str, Set[str]] = field(init=False)                    # plant -> init SKUs
    skus_prod: Dict[str, Set[str]] = field(init=False)                       # plant -> produced SKUs
    skus_plant: Dict[str, Set[str]] = field(init=False)                      # plant -> available SKUs (init ∪ produced)

    # Demand
    demands: Dict[Tuple[str, str], int] = field(init=False)                  # (dealer, sku) -> demand
    skus_dealer: Dict[str, Set[str]] = field(init=False)                     # dealer -> demanded SKUs

    # Historical inventory used as the initial recurrence value, day=0 only
    historical_s_ikt: Dict[Tuple[str, str, int], int] = field(init=False)    # (plant, sku, 0) -> init inv

    # Vehicle parameters
    veh_type_cost: Dict[str, float] = field(init=False)
    veh_type_cap: Dict[str, float] = field(init=False)
    veh_type_min_load: Dict[str, float] = field(init=False)

    # Plant inventory capacity limits
    plant_inv_limit: Dict[str, float] = field(init=False)

    # Normalized order DataFrame retained for statistics or debugging
    df_order: pd.DataFrame = field(default_factory=pd.DataFrame)

    # ------------------------------------------------------------------
    # Initialization: set the logger only to avoid duplicate handlers
    # ------------------------------------------------------------------
    def __post_init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Base file and DataFrame loading utilities
    # ------------------------------------------------------------------
    def _validate_file_exists(self, file_path: str, file_description: str) -> bool:
        """Return whether a file exists, logging an error when it does not."""
        if not os.path.exists(file_path):
            self.logger.error(f"{file_description} file does not exist: {file_path}")
            return False
        return True

    def _load_csv_safely(
        self,
        file_path: str,
        file_description: str,
        required_cols: Optional[List[str]] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Safely read a CSV file:
          - check existence
          - load it as a DataFrame
          - validate required columns when provided
          - normalize key columns to strings
          - handle exceptions and record logs

        Args:
            file_path: File path.
            file_description: Description used in log messages.
            required_cols: Required raw column names.

        Returns:
            A DataFrame, or None on failure.
        """
        if not self._validate_file_exists(file_path, file_description):
            return None
        try:
            df = pd.read_csv(file_path, header=0)
            # Early required-column validation
            if required_cols:
                missing = set(required_cols) - set(df.columns)
                if missing:
                    self.logger.error(
                        f"{file_description} is missing required columns: "
                        f"{missing} (file: {file_path})"
                    )
                    raise ValueError(
                        f"{file_description} is missing required columns: {missing}"
                    )
            key_cols = [
                c
                for c in ["product_code", "plant_code", "client_code", "vehicle_type"]
                if c in df.columns
            ]
            for col in key_cols:
                df[col] = df[col].astype(str)
            self.logger.info(
                f"Successfully loaded {file_description}: {len(df)} rows"
            )
            return df
        except Exception as e:
            self.logger.error(f"Failed to load {file_description}: {e}")
            return None

    def _reorganize_dataframe(self, df_input: pd.DataFrame, title_map: Dict[str, str]) -> pd.DataFrame:
        """
        Reorganize a DataFrame:
          - rename columns using title_map, mapping original names to new names
          - normalize column order to the mapped names
          - raise when required columns are missing
        """
        if df_input is None or df_input.empty:
            self.logger.warning("Input DataFrame is empty; skipping reorganization")
            return df_input

        missing_cols = set(title_map.keys()) - set(df_input.columns)
        if missing_cols:
            self.logger.error(f"DataFrame is missing required columns: {missing_cols}")
            raise ValueError(f"DataFrame is missing required columns: {missing_cols}")

        df_result = df_input.rename(columns=title_map)
        ordered_cols = list(title_map.values())
        return df_result.reindex(columns=ordered_cols)

    def _convert_dataframe_types(self, df: pd.DataFrame, type_mapping: Dict[str, type]) -> pd.DataFrame:
        """
        Convert selected columns to target types:
          - safely skip missing columns
          - use to_numeric(errors='coerce') plus fillna for numeric columns
          - normalize key columns to strings
        """
        if df is None or df.empty:
            return df
        df_result = df.copy()
        key_cols = [
            c
            for c in ["product_code", "plant_code", "client_code", "vehicle_type"]
            if c in df_result.columns
        ]
        for col in key_cols:
            df_result[col] = df_result[col].astype(str)
        for col, dtype in type_mapping.items():
            if col not in df_result.columns:
                continue
            try:
                if dtype == str:
                    df_result[col] = df_result[col].astype(str)
                elif dtype == int:
                    df_result[col] = (
                        pd.to_numeric(df_result[col], errors="coerce").fillna(0).astype(int)
                    )
                elif dtype == float:
                    df_result[col] = pd.to_numeric(df_result[col], errors="coerce").fillna(0.0)
            except Exception as e:
                self.logger.warning(
                    f"Failed to convert column {col} to type {dtype}: {e}"
                )
        return df_result

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def load(self):
        """
        Load and build all data interfaces. The operation is idempotent.

        If any stage fails, raise an exception so external callers observe a
        consistent failure state.
        """
        try:
            self._load_parameters()
            file_paths = self._construct_file_paths()
            dataframes = self._load_all_dataframes(file_paths)
            self._process_dataframes(dataframes)
            self._compute_derived_data()
            self._validate_data_integrity()
            self.logger.info("Data loading and processing completed")
        except Exception as e:
            self.logger.error(f"Data loading failed: {e}")
            raise

    # ------------------------------------------------------------------
    # Step 1: parameter loading
    # ------------------------------------------------------------------
    def _load_parameters(self):
        """Read penalty parameters from JSON configuration and fail if missing."""
        config_file = os.path.join(self.input_file_loc, "model_config_alns.json")
        if not self._validate_file_exists(config_file, "configuration file"):
            raise FileNotFoundError(f"Configuration file does not exist: {config_file}")

        try:
            with open(config_file, "r", encoding="utf-8") as fp:
                param = json.load(fp)

            self.param_pun_factor1 = float(param.get("pun_factor1", 0))
            self.param_pun_factor2 = float(param.get("pun_factor2", 0))
            self.param_pun_factor3 = float(param.get("pun_factor3", 0))
            self.param_pun_factor4 = float(param.get("pun_factor4", 0))
            self.param_pun_objective = float(param.get("pun_objective", 0))
            self.param_alpha = float(param.get("alpha", 2.0))

            self.logger.info("Configuration parameters loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load configuration parameters: {e}")
            raise

    # ------------------------------------------------------------------
    # Step 2: build file paths
    # ------------------------------------------------------------------
    def _construct_file_paths(self) -> Dict[str, str]:
        """Build the mapping of data-file paths inside the dataset directory."""
        dataset_path = os.path.join(self.input_file_loc, self.dataset_name)
        return {
            "order": os.path.join(dataset_path, "order.csv"),
            "plant_cap": os.path.join(dataset_path, "plant_size_upper_limit.csv"),
            "sku_size": os.path.join(dataset_path, "product_size.csv"),
            "vehicle": os.path.join(dataset_path, "vehicle.csv"),
            "production": os.path.join(dataset_path, "warehouse_production.csv"),
            "inventory": os.path.join(dataset_path, "warehouse_storage.csv"),
        }

    # ------------------------------------------------------------------
    # Step 3: load all DataFrames
    # ------------------------------------------------------------------
    def _load_all_dataframes(self, file_paths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """Load CSV files into a DataFrame mapping and stop on any failure."""
        dataframes = {}
        file_desc = {
            "order": "order demand data",
            "plant_cap": "plant inventory capacity limits",
            "sku_size": "SKU size data",
            "vehicle": "vehicle-type data",
            "production": "production plan",
            "inventory": "initial inventory",
        }
        required_columns = {
            "order": ["client_code", "product_code", "volume"],
            "plant_cap": ["plant_code", "max_volume"],
            "sku_size": ["product_code", "standard_size"],
            "vehicle": ["vehicle_type", "carry_standard_size", "min_standard_size", "cost_to_use"],
            "production": ["plant_code", "product_code", "produce_date", "volume"],
            "inventory": ["plant_code", "product_code", "volume"],
        }
        for key, path in file_paths.items():
            df = self._load_csv_safely(path, file_desc[key], required_cols=required_columns.get(key))
            if df is None:
                raise FileNotFoundError(
                    f"Required file failed to load: {file_desc[key]}"
                )
            dataframes[key] = df
        return dataframes

    # ------------------------------------------------------------------
    # Step 4: normalize DataFrames
    # ------------------------------------------------------------------
    def _process_dataframes(self, dataframes: Dict[str, pd.DataFrame]):
        """Apply column mappings and type conversion to _df_* intermediates."""

        # Orders
        title_map = {"client_code": "dealer_id", "product_code": "sku_id", "volume": "order_qty"}
        type_mapping = {"dealer_id": str, "sku_id": str, "order_qty": int}
        df_order = self._reorganize_dataframe(dataframes["order"], title_map)
        self.df_order = self._convert_dataframe_types(df_order, type_mapping)

        # Plant capacity
        title_map = {"plant_code": "fact_id", "max_volume": "max_cap"}
        type_mapping = {"fact_id": str, "max_cap": float}
        df_plant_cap = self._reorganize_dataframe(dataframes["plant_cap"], title_map)
        self._df_plant_cap = self._convert_dataframe_types(df_plant_cap, type_mapping)

        # SKU sizes
        title_map = {"product_code": "sku_id", "standard_size": "size"}
        type_mapping = {"sku_id": str, "size": float}
        df_sku_size = self._reorganize_dataframe(dataframes["sku_size"], title_map)
        self._df_sku_size = self._convert_dataframe_types(df_sku_size, type_mapping)

        # Vehicles
        title_map = {
            "vehicle_type": "vehicle_type",
            "carry_standard_size": "capacity",
            "min_standard_size": "min_load",
            "cost_to_use": "cost",
        }
        type_mapping = {"vehicle_type": str, "capacity": float, "min_load": float, "cost": float}
        df_all_veh = self._reorganize_dataframe(dataframes["vehicle"], title_map)
        self._df_all_veh = self._convert_dataframe_types(df_all_veh, type_mapping)

        # Production plan
        title_map = {"plant_code": "fact_id", "product_code": "sku_id", "produce_date": "day", "volume": "prod_qty"}
        type_mapping = {"fact_id": str, "sku_id": str, "day": int, "prod_qty": int}
        df_sku_prod = self._reorganize_dataframe(dataframes["production"], title_map)
        self._df_sku_prod = self._convert_dataframe_types(df_sku_prod, type_mapping)

        # Initial inventory
        title_map = {"plant_code": "fact_id", "product_code": "sku_id", "volume": "inv"}
        type_mapping = {"fact_id": str, "sku_id": str, "inv": int}
        df_sku_inv = self._reorganize_dataframe(dataframes["inventory"], title_map)
        self._df_sku_inv = self._convert_dataframe_types(df_sku_inv, type_mapping)

    # ------------------------------------------------------------------
    # Step 5: compute derived structures
    # ------------------------------------------------------------------
    def _compute_derived_data(self):
        """Aggregate sets, mappings, and the initial inventory recurrence."""
        # Planning horizon
        if len(self._df_sku_prod) > 0:
            self.horizons = int(self._df_sku_prod["day"].max())

        # Base sets
        self.plants = set(self._df_plant_cap["fact_id"].unique())
        self.dealers = set(self.df_order["dealer_id"].unique())
        self.all_skus = set(self._df_sku_size["sku_id"].unique())
        self.all_veh_types = set(self._df_all_veh["vehicle_type"].unique())

        # SKU sizes
        self.sku_sizes = dict(zip(self._df_sku_size["sku_id"], self._df_sku_size["size"]))

        # Production
        self.sku_prod_each_day = (
            self._df_sku_prod.groupby(["fact_id", "sku_id", "day"])["prod_qty"]
            .sum()
            .to_dict()
        )
        self.sku_prod_total = self._df_sku_prod.groupby(["sku_id", "day"])["prod_qty"].sum().to_dict()

        # Initial inventory
        self.sku_initial_inv = (
            self._df_sku_inv.groupby(["fact_id", "sku_id"])["inv"].sum().to_dict()
        )

        # Historical inventory, day=0 only
        self.historical_s_ikt = {}
        for (plant, sku), inv in self.sku_initial_inv.items():
            self.historical_s_ikt[plant, sku, 0] = inv

        # SKU availability mappings
        self._compute_sku_mappings()

        # Demand
        self.demands = (
            self.df_order.groupby(["dealer_id", "sku_id"])["order_qty"]
            .sum()
            .to_dict()
        )
        self.skus_dealer = {}
        for (dealer, sku) in self.demands:
            self.skus_dealer.setdefault(dealer, set()).add(sku)

        # Vehicles
        self.veh_type_cost = dict(zip(self._df_all_veh["vehicle_type"], self._df_all_veh["cost"]))
        self.veh_type_cap = dict(zip(self._df_all_veh["vehicle_type"], self._df_all_veh["capacity"]))
        self.veh_type_min_load = dict(zip(self._df_all_veh["vehicle_type"], self._df_all_veh["min_load"]))

        # Plant capacity limits
        self.plant_inv_limit = dict(zip(self._df_plant_cap["fact_id"], self._df_plant_cap["max_cap"]))

    def _compute_sku_mappings(self):
        """Build initial, production, and combined SKU-set mappings."""
        # Initial-inventory SKUs
        self.skus_initial = defaultdict(set)
        for (plant, sku) in self.sku_initial_inv:
            self.skus_initial[plant].add(sku)
        self.skus_initial = dict(self.skus_initial)

        # Produced SKUs
        self.skus_prod = defaultdict(set)
        for (plant, sku, day) in self.sku_prod_each_day:
            self.skus_prod[plant].add(sku)
        self.skus_prod = dict(self.skus_prod)

        # Available SKUs: initial inventory union production
        self.skus_plant = {}
        for plant in self.plants:
            initial = self.skus_initial.get(plant, set())
            prod = self.skus_prod.get(plant, set())
            self.skus_plant[plant] = initial.union(prod)

    # ------------------------------------------------------------------
    # Step 6: integrity validation
    # ------------------------------------------------------------------
    def _validate_data_integrity(self):
        """Check existence and cross-consistency of key sets and mappings."""
        errors = []

        # Required sets
        if not self.plants:
            errors.append("No plant data found")
        if not self.dealers:
            errors.append("No dealer data found")
        if not self.all_skus:
            errors.append("No SKU data found")
        if not self.all_veh_types:
            errors.append("No vehicle-type data found")

        # Missing SKU sizes
        order_skus = {sku for dealer, sku in self.demands.keys()}
        missing_sku_sizes = order_skus - set(self.sku_sizes.keys())
        if missing_sku_sizes:
            errors.append(
                f"Order SKUs are missing size information: {missing_sku_sizes}"
            )

        # Plants without available SKUs, warning only
        plants_with_skus = set(self.skus_plant.keys())
        plants_without = self.plants - plants_with_skus
        if plants_without:
            self.logger.warning(
                f"The following plants have no available SKUs: {plants_without}"
            )

        if errors:
            msg = "Data integrity validation failed:\\n" + "\\n".join(
                f"- {e}" for e in errors
            )
            self.logger.error(msg)
            raise ValueError(msg)

        self.logger.info("Data integrity validation passed")

    # ------------------------------------------------------------------
    # Supply-chain and accessor helpers
    # ------------------------------------------------------------------
    def available_skus_in_plant(self, plant_id: str) -> Set[str]:
        """Return all SKUs available from a plant, or an empty set with a warning."""
        if plant_id not in self.plants:
            self.logger.warning(f"Plant {plant_id} does not exist in the data")
            return set()
        return self.skus_plant.get(plant_id, set())

    def plan_dealer_skus(self, plant_id: str) -> Set[Tuple[str, str]]:
        """Return potential (dealer, SKU) pairs served by the plant."""
        if plant_id not in self.plants:
            self.logger.warning(f"Plant {plant_id} does not exist in the data")
            return set()
        supply_chain = self.construct_supply_chain()
        result = set()
        for (plant, dealer), skus in supply_chain.items():
            if plant == plant_id:
                for sku in skus:
                    result.add((dealer, sku))
        return result

    def available_skus_to_dealer(self, plant_id: str, dealer_id: str) -> Set[str]:
        """Return SKUs that a plant can supply to a dealer."""
        if plant_id not in self.plants:
            self.logger.warning(f"Plant {plant_id} does not exist in the data")
            return set()
        if dealer_id not in self.dealers:
            self.logger.warning(f"Dealer {dealer_id} does not exist in the data")
            return set()
        supply_chain = self.construct_supply_chain()
        return supply_chain.get((plant_id, dealer_id), set())

    def construct_supply_chain(self) -> Dict[Tuple[str, str], Set[str]]:
        """
        Build the supply-chain relation:
            {(plant, dealer): intersect_skus}, where intersect_skus contains
            SKUs both available from the plant and demanded by the dealer.

        The result is cached to avoid repeated computation.
        """
        if not hasattr(self, "_supply_chain_cache"):
            supply_chain = defaultdict(set)
            for plant, skus_supply in self.skus_plant.items():
                for dealer, skus_need in self.skus_dealer.items():
                    inter = skus_supply.intersection(skus_need)
                    if inter:
                        supply_chain[plant, dealer] = inter
            self._supply_chain_cache = dict(supply_chain)
        return self._supply_chain_cache

    # ------------------------------------------------------------------
    # Metric aggregation and supply-demand analysis
    # ------------------------------------------------------------------
    def get_total_demand_for_sku(self, sku_id: str) -> int:
        """Aggregate total demand for a SKU."""
        total = 0
        for (dealer, sku), demand in self.demands.items():
            if sku == sku_id:
                total += demand
        return total

    def get_total_production_for_sku(self, sku_id: str, day: Optional[int] = None) -> int:
        """Return total production for a SKU on one day or across all days."""
        if day is not None:
            return self.sku_prod_total.get((sku_id, day), 0)
        total = 0
        for (sku, d), qty in self.sku_prod_total.items():
            if sku == sku_id:
                total += qty
        return total

    def get_total_initial_inventory_for_sku(self, sku_id: str) -> int:
        """Return total initial inventory for a SKU."""
        total = 0
        for (plant, sku), inv in self.sku_initial_inv.items():
            if sku == sku_id:
                total += inv
        return total

    def get_sku_supply_demand_balance(self, sku_id: str) -> Dict[str, Union[int, float]]:
        """
        Supply-demand balance:
            total_supply = initial + production_all
            balance = supply - demand
            balance_ratio = supply / demand, or inf when demand is zero.
        """
        total_demand = self.get_total_demand_for_sku(sku_id)
        total_init = self.get_total_initial_inventory_for_sku(sku_id)
        total_prod = self.get_total_production_for_sku(sku_id)
        total_supply = total_init + total_prod
        balance = total_supply - total_demand
        ratio = total_supply / total_demand if total_demand > 0 else float("inf")
        return {
            "sku_id": sku_id,
            "total_demand": total_demand,
            "total_initial_inventory": total_init,
            "total_production": total_prod,
            "total_supply": total_supply,
            "balance": balance,
            "balance_ratio": ratio,
            "status": "surplus" if balance > 0 else "deficit" if balance < 0 else "balanced",
        }

    def get_summary_statistics(self) -> Dict[str, Union[int, float]]:
        """Return overall summary metrics for logging or a quick sanity check."""
        total_demand = sum(self.demands.values())
        total_initial = sum(self.sku_initial_inv.values())
        total_prod = sum(self.sku_prod_total.values())
        total_supply = total_initial + total_prod
        return {
            "num_plants": len(self.plants),
            "num_dealers": len(self.dealers),
            "num_skus": len(self.all_skus),
            "num_vehicle_types": len(self.all_veh_types),
            "planning_horizon": self.horizons,
            "total_demand": total_demand,
            "total_initial_inventory": total_initial,
            "total_production": total_prod,
            "total_supply": total_supply,
            "supply_demand_ratio": total_supply / total_demand if total_demand > 0 else float("inf"),
        }
