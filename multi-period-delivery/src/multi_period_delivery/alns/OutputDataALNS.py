"""
OutputDataALNS

Purpose
    Read ALNS output CSVs (opt_result, opt_details, opt_summary, non_fulfill,
    and sku_inv_left), normalize column names and types, build lookup indexes,
    and provide consistency validation and aggregate statistics.
"""

# =========================
# Standard library
# =========================
import os
import logging
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

# =========================
# Third-party libraries
# =========================
import pandas as pd

# Configure logging in the main program. This module only obtains a logger.
logger = logging.getLogger(__name__)


@dataclass
class OutPutData:
    """Output result access and validation wrapper:
    - load() reads and normalizes output CSVs.
    - _generate_indexes() builds dictionary indexes for fast lookup.
    - _validate_data() performs basic consistency checks.
    - get_summary_stats() returns aggregate statistics.
    """
    input_file_loc: str
    dataset_name: str
    
    # Data attributes initialized by __post_init__
    order_fulfill: Dict[Tuple[int, str, str, str], int] = field(init=False)
    vehicle_load: Dict[Tuple[int, str, str, int, str], float] = field(init=False)
    non_fulfill: Dict[Tuple[str, str], int] = field(init=False)
    sku_inv_left: Dict[Tuple[str, str, int], int] = field(init=False)
    
    # DataFrame cache
    df_result: Optional[pd.DataFrame] = field(init=False, default=None)
    df_details: Optional[pd.DataFrame] = field(init=False, default=None)
    df_summary: Optional[pd.DataFrame] = field(init=False, default=None)
    df_non_fulfill: Optional[pd.DataFrame] = field(init=False, default=None)
    df_sku_inv_left: Optional[pd.DataFrame] = field(init=False, default=None)
    
    def __post_init__(self):
        """Initialize the data structures."""
        self.order_fulfill = {}
        self.vehicle_load = {}
        self.non_fulfill = {}
        self.sku_inv_left = {}
    
    def _reorganize_dataframe(self, df: pd.DataFrame, title_map: Dict[str, str]) -> pd.DataFrame:
        """
        Reorganize DataFrame column names and order.

        Args:
            df: Input DataFrame.
            title_map: Column-name mapping.

        Returns:
            Reorganized DataFrame.
        """
        # Copy to avoid modifying the input data
        df_copy = df.copy()
        
        # Rename columns
        df_copy.rename(columns=title_map, inplace=True)
        
        # Reorder columns while keeping only columns that exist
        ordered_cols = [col for col in title_map.values() if col in df_copy.columns]
        df_copy = df_copy.reindex(columns=ordered_cols)
        
        return df_copy
    
    def _load_and_process_csv(self, file_path: str, title_map: Dict[str, str], 
                             dtype_map: Dict[str, type]) -> pd.DataFrame:
        """
        Load and process a CSV file.

        Args:
            file_path: File path.
            title_map: Column-name mapping.
            dtype_map: Target data-type mapping.

        Returns:
            Processed DataFrame.
        """
        if not os.path.exists(file_path):
            logger.warning(f"File does not exist: {file_path}")
            # Return an empty DataFrame with the expected column names
            empty_df = pd.DataFrame(columns=list(title_map.values()))
            return self._apply_dtypes(empty_df, dtype_map)
        
        try:
            # Read the CSV file
            df = pd.read_csv(file_path, header=0)
            logger.info(f"Successfully loaded file: {file_path}, shape: {df.shape}")
            
            # Reorganize column names
            df = self._reorganize_dataframe(df, title_map)
            
            # Apply data types
            df = self._apply_dtypes(df, dtype_map)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load file {file_path}: {str(e)}")
            # Return an empty DataFrame
            empty_df = pd.DataFrame(columns=list(title_map.values()))
            return self._apply_dtypes(empty_df, dtype_map)
    
    def _apply_dtypes(self, df: pd.DataFrame, dtype_map: Dict[str, type]) -> pd.DataFrame:
        """
        Apply data-type conversions.

        Args:
            df: Input DataFrame.
            dtype_map: Target data-type mapping.

        Returns:
            DataFrame with converted types.
        """
        for col, dtype in dtype_map.items():
            if col in df.columns:
                try:
                    if dtype == int:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                    elif dtype == float:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(float)
                    elif dtype == str:
                        df[col] = df[col].astype(str)
                except Exception as e:
                    logger.warning(f"Failed to convert column {col}: {str(e)}")
        
        return df
    
    def load(self) -> None:
        """
        Load all output data files.
        """
        logger.info(f"Starting dataset load: {self.dataset_name}")
        
        # Build the dataset directory path
        dataset_path = os.path.join(self.input_file_loc, self.dataset_name)
        # Check directory existence. Missing files will produce empty tables.
        if not os.path.isdir(dataset_path):
            logger.warning(f"Dataset directory does not exist: {dataset_path}")
        
        # File configuration:
        # - title_map: original CSV column name -> unified internal field name
        # - dtype_map: target data type (int/float/str)
        # Table semantics:
        #   result: shipment detail records by quantity
        #   details: shipment details with size and capacity metrics for
        #            unit checks
        #   summary: per-vehicle occupied volume and minimum-load metrics
        #   non_fulfill: unmet demand quantities
        #   sku_inv_left: daily remaining inventory, using units consistent with
        #                 validation logic
        file_configs = {
            'result': {
                'filename': 'opt_result.csv',
                'title_map': {
                    'day': 'day', 'plant_code': 'fact_id', 
                    'client_code': 'dealer_id', 'product_code': 'sku_id', 
                    'vehicle_id': 'vehicle_id', 'vehicle_type': 'vehicle_type',
                    'qty': 'qty'
                },
                'dtype_map': {
                    'day': int, 'fact_id': str, 'dealer_id': str, 
                    'sku_id': str, 'vehicle_id': int, 'vehicle_type': str, 'qty': int
                }
            },
            'details': {
                'filename': 'opt_details.csv',
                'title_map': {
                    'day': 'day', 'plant_code': 'fact_id', 
                    'client_code': 'dealer_id', 'product_code': 'sku_id', 
                    'vehicle_id': 'vehicle_id', 'vehicle_type': 'vehicle_type',
                    'qty': 'qty', 'standard_size': 'size', 'carry_standard_size': 'capacity',
                    'min_standard_size': 'min_load', 'occupied_size': 'occupied_size'
                },
                'dtype_map': {
                    'day': int, 'fact_id': str, 'dealer_id': str, 'sku_id': str,
                    'vehicle_id': int, 'vehicle_type': str, 'qty': int,
                    'size': float, 'capacity': int, 'min_load': int, 'occupied_size': float
                }
            },
            'summary': {
                'filename': 'opt_summary.csv',
                'title_map': {
                    'day': 'day', 'plant_code': 'fact_id', 
                    'client_code': 'dealer_id', 'vehicle_id': 'vehicle_id', 
                    'vehicle_type': 'vehicle_type', 'qty': 'qty', 
                    'carry_standard_size': 'capacity', 'min_standard_size': 'min_load', 
                    'occupied_size': 'occupied_size'
                },
                'dtype_map': {
                    'day': int, 'fact_id': str, 'dealer_id': str, 'vehicle_id': int,
                    'vehicle_type': str, 'qty': int, 'capacity': int, 
                    'min_load': int, 'occupied_size': float
                }
            },
            'non_fulfill': {
                'filename': 'non_fulfill.csv',
                'title_map': {
                    'client_code': 'dealer_id', 'product_code': 'sku_id', 'volume': 'non_fulfill_qty'
                },
                'dtype_map': {
                    'dealer_id': str, 'sku_id': str, 'non_fulfill_qty': int
                }
            },
            'sku_inv_left': {
                'filename': 'sku_inv_left.csv',
                'title_map': {
                    'plant_code': 'fact_id', 'product_code': 'sku_id', 'day': 'day', 'volume': 'inv_left'
                },
                'dtype_map': {
                    'fact_id': str, 'sku_id': str, 'day': int, 'inv_left': int
                }
            }
        }
        
        # Load all files
        loaded_data = {}
        for key, config in file_configs.items():
            file_path = os.path.join(dataset_path, config['filename'])
            df = self._load_and_process_csv(
                file_path, 
                config['title_map'], 
                config['dtype_map']
            )
            loaded_data[key] = df
        
        # Save DataFrame references
        self.df_result = loaded_data['result']
        self.df_details = loaded_data['details']
        self.df_summary = loaded_data['summary']
        self.df_non_fulfill = loaded_data['non_fulfill']
        self.df_sku_inv_left = loaded_data['sku_inv_left']
        
        # Generate dictionary indexes
        self._generate_indexes()
        
        # Validate data integrity
        self._validate_data()
        
        logger.info("Data loading completed")
    
    def _generate_indexes(self) -> None:
        """Generate dictionary indexes for fast lookup.

        Constructed keys and values:
        - order_fulfill[(day, fact_id, dealer_id, sku_id)] = qty
        - vehicle_load[(day, fact_id, dealer_id, vehicle_id, vehicle_type)] =
          occupied_size
        - non_fulfill[(dealer_id, sku_id)] = non_fulfill_qty
        - sku_inv_left[(fact_id, sku_id, day)] = inv_left
        """
        try:
            # Order fulfillment
            if not self.df_result.empty:
                self.order_fulfill = self.df_result.groupby(
                    ['day', 'fact_id', 'dealer_id', 'sku_id']
                )['qty'].sum().to_dict()
            
            # Vehicle loading
            if not self.df_summary.empty:
                self.vehicle_load = self.df_summary.set_index(
                    ['day', 'fact_id', 'dealer_id', 'vehicle_id', 'vehicle_type']
                )['occupied_size'].to_dict()
            
            # Unmet demand
            if not self.df_non_fulfill.empty:
                self.non_fulfill = self.df_non_fulfill.set_index(
                    ['dealer_id', 'sku_id']
                )['non_fulfill_qty'].to_dict()
            
            # Remaining inventory
            if not self.df_sku_inv_left.empty:
                self.sku_inv_left = self.df_sku_inv_left.set_index(
                    ['fact_id', 'sku_id', 'day']
                )['inv_left'].to_dict()
                
            logger.info("Dictionary indexes generated")
            
        except Exception as e:
            logger.error(f"Failed to generate dictionary indexes: {str(e)}")
    
    def _validate_data(self) -> None:
        """Validate data integrity.
        - Check whether key tables are empty.
        - Check vehicle-ID consistency between result and summary tables.
        - Check for invalid negative values.
        """
        issues = []
        
        # Check whether key data is empty
        if self.df_result.empty:
            issues.append("Result data (opt_result.csv) is empty")
        
        if self.df_summary.empty:
            issues.append("Summary data (opt_summary.csv) is empty")
        
        # Check data consistency
        if not self.df_result.empty and not self.df_summary.empty:
            result_vehicles = set(self.df_result['vehicle_id'].unique())
            summary_vehicles = set(self.df_summary['vehicle_id'].unique())
            if result_vehicles != summary_vehicles:
                issues.append(
                    "Vehicle IDs differ between result data and summary data"
                )
        
        # Check for negative values
        if not self.df_result.empty and (self.df_result['qty'] < 0).any():
            issues.append("Negative delivery quantities found")
        
        if not self.df_sku_inv_left.empty and (self.df_sku_inv_left['inv_left'] < 0).any():
            issues.append("Negative inventory quantities found")
        
        # Emit validation results
        if issues:
            logger.warning("Data validation found issues:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("Data validation passed")
    
    def get_summary_stats(self) -> Dict[str, int]:
        """
        Return aggregate output statistics.

        Keys:
        - total_deliveries: number of shipment entries after aggregating by
          day, fact, dealer, and SKU.
        - total_vehicles_used: number of vehicle-load entries after aggregating
          by day, fact, dealer, vehicle, and vehicle type.
        - unfulfilled_demands: number of unmet-demand entries by dealer and SKU.
        - inventory_records: number of inventory records by fact, SKU, and day.
        - total_quantity_shipped: total shipped quantity from opt_result.qty.
        - unique_vehicles: number of unique vehicles from opt_result.vehicle_id.

        Returns:
            Dictionary of aggregate statistics.
        """
        stats = {
            'total_deliveries': len(self.order_fulfill),
            'total_vehicles_used': len(self.vehicle_load),
            'unfulfilled_demands': len(self.non_fulfill),
            'inventory_records': len(self.sku_inv_left)
        }
        
        if not self.df_result.empty:
            stats['total_quantity_shipped'] = int(self.df_result['qty'].sum())
            stats['unique_vehicles'] = int(self.df_result['vehicle_id'].nunique())
        
        return stats
