"""
check_solution

Module role
    Perform systematic consistency and feasibility checks on ALNS output,
    including demand satisfaction, plant capacity, vehicle capacity, and
    inventory flow. It also produces a readable validation summary for
    debugging and result review.

Design principles
    - Do not mutate input or output data; validation is read-only.
    - Use a module logger consistently and avoid changing global logging
      configuration from inside the library.
    - Keep each validation method separate for easier debugging and extension.

Inputs/Outputs
    - Input: DataALNS (static data) and OutPutData (algorithm output).
    - Output: a list of ValidationResult objects and an overall feasibility
      boolean.
"""

# =========================
# Standard library
# =========================
import os
import logging
from dataclasses import dataclass
from typing import Tuple, Dict, List

# =========================
# Internal project modules
# =========================
from .InputDataALNS import DataALNS
from .OutputDataALNS import OutPutData
from .alns_config import default_config as ALNSConfig

# Configure logging in the main program; this module only obtains its logger.
logger = logging.getLogger(__name__)

# Default paths are resolved from the current working directory. This keeps the
# source-tree and editable-install workflows consistent.
project_root = os.getcwd()
input_loc = os.path.join(
    project_root, 'datasets', 'multiple-periods', ALNSConfig.DATASET_TYPE
)
output_loc = os.path.join(
    project_root, 'OutPut-ALNS', 'multiple-periods', ALNSConfig.DATASET_TYPE
)


@dataclass
class ValidationResult:
    """Data class for one validation result."""
    constraint_name: str
    is_satisfied: bool
    violations: List[str]
    total_violations: int
    details: Dict


class SolutionValidator:
    """Validator for solution feasibility and consistency."""
    
    def __init__(self, input_data: DataALNS, output_data: OutPutData):
        self.input_data = input_data
        self.output_data = output_data
        self.validation_results = []
    
    def check_demand_satisfaction(self) -> ValidationResult:
        """Check whether demand is satisfied."""
        logger.info("Checking demand satisfaction...")
        violations = []
        details = {
            'total_demands': len(self.input_data.demands),
            'satisfied_demands': 0,
            'oversupplied_demands': 0,
            'undersupplied_demands': 0,
            'total_shortage': 0,
            'total_oversupply': 0
        }
        
        # Iterate over all demand records.
        for (dealer_id, sku_id), demand in self.input_data.demands.items():
            # Compute the actual shipped quantity.
            actual_shipped = 0
            # Accumulate shipped quantity from the output fulfillment records.
            for day in range(1, self.input_data.horizons + 1):
                for fact_id in self.input_data.plants:
                    actual_shipped += self.output_data.order_fulfill.get((day, fact_id, dealer_id, sku_id), 0)
            
            # Check demand satisfaction.
            if actual_shipped < demand:
                shortage = demand - actual_shipped
                details['undersupplied_demands'] += 1
                details['total_shortage'] += shortage
                violations.append(
                    f"Demand for dealer {dealer_id}, SKU {sku_id} is not "
                    f"satisfied: demand {demand}, shipped {actual_shipped}, "
                    f"shortage {shortage}"
                )
            elif actual_shipped > demand:
                oversupply = actual_shipped - demand
                details['oversupplied_demands'] += 1
                details['total_oversupply'] += oversupply
                # Oversupply is not a constraint violation, but it is logged.
                logger.info(
                    f"Oversupply for dealer {dealer_id}, SKU {sku_id}: "
                    f"demand {demand}, shipped {actual_shipped}, "
                    f"excess {oversupply}"
                )
            else:
                details['satisfied_demands'] += 1
        
        is_satisfied = len(violations) == 0
        return ValidationResult(
            constraint_name="Demand satisfaction constraint",
            is_satisfied=is_satisfied,
            violations=violations,
            total_violations=len(violations),
            details=details
        )

    
    def check_plant_capacity(self) -> ValidationResult:
        """Check the plant inventory-capacity constraint."""
        logger.info("Checking plant inventory-capacity constraints...")
        violations = []
        details = {
            'total_checks': 0,
            'violations_count': 0,
            'max_violation': 0,
            'total_excess': 0
        }
        
        # Iterate over each plant and time period.
        for plant_id in self.input_data.plants:
            for day in range(1, self.input_data.horizons + 1):
                details['total_checks'] += 1
                total_inventory = 0
                # Compute total inventory for the day.
                for sku_id in self.input_data.all_skus:
                    if (plant_id, sku_id, day) in self.output_data.sku_inv_left:
                        total_inventory += self.output_data.sku_inv_left[(plant_id, sku_id, day)]
                
                # Check the inventory limit, defined as the maximum quantity
                # of stored SKUs.
                capacity_limit = self.input_data.plant_inv_limit[plant_id]
                if total_inventory > capacity_limit:
                    excess = total_inventory - capacity_limit
                    details['violations_count'] += 1
                    details['total_excess'] += excess
                    details['max_violation'] = max(details['max_violation'], excess)
                    violations.append(
                        f"Plant {plant_id} exceeded its inventory limit on "
                        f"day {day}: limit {capacity_limit}, actual "
                        f"{total_inventory}, excess {excess}"
                    )
        
        is_satisfied = len(violations) == 0
        return ValidationResult(
            constraint_name="Plant inventory capacity constraint",
            is_satisfied=is_satisfied,
            violations=violations,
            total_violations=len(violations),
            details=details
        )
    
    def check_vehicle_capacity(self) -> ValidationResult:
        """Check vehicle-capacity constraints."""
        logger.info("Checking vehicle-capacity constraints...")
        violations = []
        details = {
            'total_vehicles': len(self.output_data.vehicle_load),
            'capacity_violations': 0,
            'min_load_warnings': 0,
            'max_capacity_violation': 0,
            'total_excess_capacity': 0
        }
        
        # Iterate over each delivery task.
        for (day, fact_id, dealer_id, vehicle_id, vehicle_type), load in self.output_data.vehicle_load.items():
            # Check the vehicle capacity limit.
            capacity_limit = self.input_data.veh_type_cap[vehicle_type]
            if load > capacity_limit:
                excess = load - capacity_limit
                details['capacity_violations'] += 1
                details['total_excess_capacity'] += excess
                details['max_capacity_violation'] = max(details['max_capacity_violation'], excess)
                violations.append(
                    f"Vehicle {vehicle_id} (type: {vehicle_type}) exceeded "
                    f"capacity on day {day}: limit {capacity_limit}, loaded "
                    f"{load}, excess {excess}"
                )
            
            # Check the minimum-load requirement. This is a soft constraint and
            # does not affect feasibility.
            min_load_req = self.input_data.veh_type_min_load[vehicle_type]
            if load < min_load_req:
                details['min_load_warnings'] += 1
                logger.info(
                    f"Vehicle {vehicle_id} (type: {vehicle_type}) did not "
                    f"reach minimum load on day {day}: minimum "
                    f"{min_load_req}, loaded {load}"
                )
        
        is_satisfied = len(violations) == 0
        return ValidationResult(
            constraint_name="Vehicle capacity constraint",
            is_satisfied=is_satisfied,
            violations=violations,
            total_violations=len(violations),
            details=details
        )
    
    def check_inventory_flow(self) -> ValidationResult:
        """Check inventory-flow constraints."""
        logger.info("Checking inventory-flow constraints...")
        violations = []
        details = {
            'total_flow_checks': 0,
            'negative_inventory_violations': 0,
            'record_inconsistencies': 0,
            'min_negative_inventory': 0,
            'max_inconsistency': 0
        }
        
        # Tolerance for record comparisons, overridable through central config.
        record_eps = getattr(ALNSConfig, "RECORD_INCONSISTENCY_EPS", 1e-6)
        
        # Iterate over each plant and SKU.
        for plant_id in self.input_data.plants:
            for sku_id in self.input_data.all_skus:
                # Get opening inventory.
                current_inv = self.input_data.sku_initial_inv.get((plant_id, sku_id), 0)
                
                # Check inventory flow in chronological order.
                for day in range(1, self.input_data.horizons + 1):
                    details['total_flow_checks'] += 1
                    
                    # Add production for the day.
                    current_inv += self.input_data.sku_prod_each_day.get((plant_id, sku_id, day), 0)
                    
                    # Subtract shipments for the day.
                    for dealer_id in self.input_data.dealers:
                        current_inv -= self.output_data.order_fulfill.get((day, plant_id, dealer_id, sku_id), 0)
                    
                    # Check for negative inventory.
                    if current_inv < 0:
                        details['negative_inventory_violations'] += 1
                        details['min_negative_inventory'] = min(details['min_negative_inventory'], current_inv)
                        violations.append(
                            f"Plant {plant_id}, SKU {sku_id} has negative "
                            f"inventory on day {day}: {current_inv}"
                        )
                    
                    # Compare against recorded ending inventory.
                    recorded_inv = self.output_data.sku_inv_left.get((plant_id, sku_id, day), 0)
                    inconsistency = abs(current_inv - recorded_inv)
                    if inconsistency > record_eps:
                        details['record_inconsistencies'] += 1
                        details['max_inconsistency'] = max(details['max_inconsistency'], inconsistency)
                        violations.append(
                            f"Plant {plant_id}, SKU {sku_id} has inconsistent "
                            f"inventory on day {day}: calculated "
                            f"{current_inv}, recorded {recorded_inv}, "
                            f"difference {inconsistency}"
                        )
        
        is_satisfied = len(violations) == 0
        return ValidationResult(
            constraint_name="Inventory flow constraint",
            is_satisfied=is_satisfied,
            violations=violations,
            total_violations=len(violations),
            details=details
        )
    
    def validate_all(self) -> Tuple[bool, List[ValidationResult]]:
        """Run all constraint checks."""
        logger.info("Starting solution validation...")
        
        # Run all constraint checks.
        validation_methods = [
            self.check_demand_satisfaction,
            self.check_plant_capacity,
            self.check_vehicle_capacity,
            self.check_inventory_flow
        ]
        
        all_results = []
        overall_feasible = True
        
        for validation_method in validation_methods:
            try:
                result = validation_method()
                all_results.append(result)
                if not result.is_satisfied:
                    overall_feasible = False
            except Exception as e:
                logger.error(
                    f"Validation method {validation_method.__name__} failed: "
                    f"{str(e)}"
                )
                # Create an error result.
                error_result = ValidationResult(
                    constraint_name=validation_method.__name__,
                    is_satisfied=False,
                    violations=[f"Validation process error: {str(e)}"],
                    total_violations=1,
                    details={}
                )
                all_results.append(error_result)
                overall_feasible = False
        
        self.validation_results = all_results
        return overall_feasible, all_results
    
    def print_validation_summary(self):
        """Print a validation summary."""
        print("\n" + "="*60)
        print("Solution validation summary")
        print("="*60)
        
        total_violations = 0
        for result in self.validation_results:
            status = "[PASS]" if result.is_satisfied else "[FAIL]"
            print(f"{result.constraint_name}: {status}")
            if not result.is_satisfied:
                print(f"  Violations: {result.total_violations}")
                total_violations += result.total_violations
                # Show the first three violation details.
                for i, violation in enumerate(result.violations[:3]):
                    print(f"    {i+1}. {violation}")
                if len(result.violations) > 3:
                    print(
                        f"    ... {len(result.violations) - 3} additional "
                        f"violations"
                    )
            print()
        
        print(f"Total violations: {total_violations}")
        overall_status = "feasible" if total_violations == 0 else "infeasible"
        print(f"Solution status: {overall_status}")
        print("="*60)


def check_solution(input_loc: str, output_loc: str) -> bool:
    """
    Main validation entry point.

    Parameters:
        input_loc   Root directory for input data (datasets/...).
        output_loc  Root directory for output data (OutPut-ALNS/...).

    Returns:
        bool        Whether all validation checks passed.
    """
    try:
        medium_path = f'dataset_{ALNSConfig.DATASET_IDX}'
        dataset_output_loc = os.path.join(output_loc, medium_path)
        
        # Load input data.
        logger.info("Loading input data...")
        input_data = DataALNS(input_loc, dataset_output_loc, medium_path)
        input_data.load()
        
        # Load output data.
        logger.info("Loading output data...")
        output_parent_dir = os.path.dirname(dataset_output_loc)
        output_data = OutPutData(output_parent_dir, medium_path)
        output_data.load()
        
        # Print data statistics.
        print("\nData statistics:")
        print(
            f"Input data - plants: {len(input_data.plants)}, dealers: "
            f"{len(input_data.dealers)}, SKUs: {len(input_data.all_skus)}"
        )
        output_stats = output_data.get_summary_stats()
        print(f"Output data statistics: {output_stats}")
        
        # Create the validator and run validation.
        validator = SolutionValidator(input_data, output_data)
        overall_feasible, validation_results = validator.validate_all()
        
        # Print detailed results.
        validator.print_validation_summary()
        
        return overall_feasible
        
    except Exception as e:
        logger.error(f"Solution validation failed: {str(e)}")
        print(f"\nAn error occurred during validation: {str(e)}")
        return False

if __name__ == "__main__":
    check_solution(input_loc, output_loc)
