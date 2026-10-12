"""
Legacy script-style entry point for the multi-period ALNS.

``multi_period_delivery.__init__`` re-exports ``run_model`` from this module
for backward compatibility with the pre-package layout. The supported entry
points are the console script (``multi-period-delivery``) and
``python -m multi_period_delivery``, both of which delegate to
``multi_period_delivery.cli``.

Note: importing this module reads the default ALNS configuration and opens the
log file, so the package root imports it lazily rather than at import time.
"""

import os
import sys
import logging
from pathlib import Path
from numpy import random as rnd

# Use package-internal modules via relative imports. This file must be run
# as a module (python -m multi_period_delivery.alns.main) or the package must
# be installed.

from .alns_config import default_config as ALNSConfig

PARAM_TUNING_ENABLED = ALNSConfig.ENABLE_PARAM_TUNER

SEED = ALNSConfig.SEED
RNG = rnd.default_rng(SEED)

DATASET_TYPE = ALNSConfig.DATASET_TYPE
DATASET_IDX = ALNSConfig.DATASET_IDX
log_dir = ALNSConfig.LOG_DIR

os.makedirs(log_dir, exist_ok=True)
log_file = str(Path(log_dir) / 'alns_optimization.log')
logging.basicConfig(
    level=ALNSConfig.LOG_LEVEL,
    format='%(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w')
    ]
)
logger = logging.getLogger(__name__)

# Imported after the logging setup above so the optimizer can reuse this logger.
from .alns_optimizer import ALNSOptimizer  # noqa: E402

def run_model(input_file_loc=None, output_file_loc=None):
    """
    Run the multi-period optimization model.
    
    Parameters:
    -----------
    input_file_loc : str, optional
        Input data location.
    output_file_loc : str, optional
        Output data location.
    """
    try:
        # Use default paths when no explicit paths are provided.
        if input_file_loc is None:
            project_root = Path.cwd()
            input_file_loc = (
                project_root / 'datasets' / 'multiple-periods' / DATASET_TYPE
            )
        if output_file_loc is None:
            project_root = Path.cwd()
            output_file_loc = (
                project_root / 'OutPut-ALNS' / 'multiple-periods' / DATASET_TYPE
            )

        dataset_name = f'dataset_{DATASET_IDX}'

        # Create the optimizer instance.
        optimizer = ALNSOptimizer(input_file_loc, output_file_loc)
        # Run optimization.
        success = optimizer.run_optimization(dataset_name)
        if success:
            print("=" * 100)
            optimizer.log_printer.print_title("The Vehicle Loading Plan is Done!")
        else:
            print("Optimization process failed.")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to run model: {e}")
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Running the file directly (for example `python alns/main.py`) leaves
    # __package__ empty and breaks the relative imports above. Only the module
    # form (`python -m multi_period_delivery.alns.main`) is supported.
    if not __package__:
        print(
            "Please run this module as a package or install the project first.\n"
            "Development options:\n"
            "  1) Run as a module: python -m multi_period_delivery.alns.main\n"
            "  2) Install editable: pip install -e . "
            "(then run: python -m multi_period_delivery)"
        )
        sys.exit(2)
    run_model()
