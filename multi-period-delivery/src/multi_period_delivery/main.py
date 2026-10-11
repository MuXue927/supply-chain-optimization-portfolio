import os
import sys
import logging
from pathlib import Path
from numpy import random as rnd

# Use package-internal modules via relative imports. This file must be run
# as a module (python -m multi_period_delivery.main) or the package must be installed
# (pip install -e .). Running the file directly (python multi_period_delivery/main.py)
# will raise a clear error below.

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

# Import ALNSOptimizer implementation moved to separate module
from .alns_optimizer import ALNSOptimizer, is_file_in_use

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
    # Prefer running this file as a module or install the package.
    # Running the file directly (python multi_period_delivery/main.py) can break relative imports.
    msg = (
        "Please run this module as a package or install the project first.\n"
        "Development options:\n"
        "  1) Run as a module: python -m multi_period_delivery.main\n"
        "  2) Install editable: pip install -e . (then python -m multi_period_delivery)\n"
        "If you intended to run this as a script, change imports to absolute ones or run via -m."
    )
    print(msg)
    # Optionally attempt to run when executed as package (works when invoked via -m)
    # If running via `python -m multi_period_delivery.main`, __package__ is set and run_model will work.
    if __package__:
        run_model()
    else:
        # Exit with non-zero status to indicate incorrect invocation
        sys.exit(2)
