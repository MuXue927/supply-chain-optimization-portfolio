# Shared dataset generator for the supply-chain optimization portfolio.
#
# WARNING: This generator has drifted from the archived benchmark CSV files.
# Regenerating with the documented seed (20260408) does not reproduce the
# checked-in medium dataset (different vehicle types and row counts).  The
# archived CSVs under multi-period-delivery/datasets/ are the only
# authoritative benchmark source.  Do not use this script to regenerate
# benchmark instances without re-baselining every experiment.
#
# Examples:
#   python tools/generate_data.py --problem single --scale small --count 3
#   python tools/generate_data.py --problem multi --periods 5 --scale small --count 3

# Single-period datasets contain the following CSV files:
# 1) order.csv
#    Columns: client_code, product_code, and volume.
# 2) product_size.csv
#    Columns: product_code and standard_size.
# 3) vehicle.csv
#    Columns: vehicle_type, carry_standard_size, min_standard_size, and cost_to_use.
# 4) warehouse_storage.csv
#    Columns: plant_code, product_code, and volume.
# Each plant is allowed to supply the same set of SKUs.

# Multi-period datasets contain the following CSV files:
# 1) order.csv
#    Columns: client_code, product_code, and volume.
# 2) product_size.csv
#    Columns: product_code and standard_size.
# 3) vehicle.csv
#    Columns: vehicle_type, carry_standard_size, min_standard_size, and cost_to_use.
# 4) warehouse_storage.csv
#    Columns: plant_code, product_code, and volume.
# Each plant is allowed to supply the same set of SKUs.
# 5) plant_size_upper_limit.csv
#    Columns: plant_code and max_volume.
# 6) warehouse_production.csv
#    Columns: plant_code, product_code, produce_date, and volume.
# Each plant is also allowed to supply the same set of SKUs in this file.

# The generator also writes instance_statistics.csv, which records the size of
# each dataset: plant count, dealer count, SKU count, vehicle type count, and
# planning horizon.

import os
import json
import argparse
import random
import string
import time
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

# Resolve the script directory so default outputs work from any current directory.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

parser = argparse.ArgumentParser(description="Generate datasets for both optimization projects")
parser.add_argument(
    "--problem",
    choices=["single", "multi"],
    required=True,
    help="Problem type. Use 'single' for the single-period project and 'multi' for the multi-period project.",
)
parser.add_argument(
    "--scale",
    choices=["small", "medium", "large"],
    default="small",
    help="Dataset scale.",
)
parser.add_argument(
    "--periods",
    type=int,
    default=5,
    help="Planning horizon for multi-period datasets. Single-period generation always uses one period.",
)
parser.add_argument(
    "--count",
    type=int,
    default=1,
    help="Number of dataset_N directories to generate.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Optional random seed for reproducible generation.",
)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=None,
    help="Optional exact output directory. Defaults to the target project's datasets folder.",
)
args = parser.parse_args()
HORIZONS = 1 if args.problem == "single" else args.periods
DATASET_SIZE = {"small": 0, "medium": 1, "large": 2}[args.scale]
DATASET_NUMS = args.count
# Configure logging.
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def generate_dataset():
    horizons = HORIZONS
    dataset_size = DATASET_SIZE
    dataset_nums = DATASET_NUMS

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    logger.info(f"Start generating datasets: horizons={horizons}, dataset_size={dataset_size}, dataset_nums={dataset_nums}")

    if args.output_dir:
        base_file_loc = Path(args.output_dir).resolve()
    else:
        project_dir = (
            "single-period-delivery"
            if args.problem == "single"
            else "multi-period-delivery"
        )
        base_file_loc = PROJECT_ROOT / project_dir / "datasets"

    if args.problem == "single":
        print("Generating datasets for single period.")
        base_file_loc = base_file_loc / "single-period"
    else:
        print("Generating datasets for multiple periods.")
        base_file_loc = base_file_loc / "multiple-periods"

    if args.scale == "small":
        base_file_loc = base_file_loc / "small"
    elif args.scale == "medium":
        base_file_loc = base_file_loc / "medium"
    else:
        base_file_loc = base_file_loc / "large"

    # Store summary statistics for every generated dataset.
    dataset_statistics = []

    for i in range(1, dataset_nums + 1):
        output_file_loc = base_file_loc / f"dataset_{i}"
        # Create the dataset output directory if it does not already exist.
        output_file_loc.mkdir(parents=True, exist_ok=True)
        logger.info(f"Preparing dataset {i}/{dataset_nums} at {output_file_loc}")
        start_time = time.time()

        if dataset_size == 0:  # Small scale
            plant_nums = random.randint(2, 3)
            dealer_nums = random.randint(2, 30)
            sku_nums = random.randint(10, 40)
            veh_type_nums = random.randint(2, 3)
            # plant_nums = 2
            # dealer_nums = 2
            # sku_nums = 5
            # veh_type_nums = 1
        elif dataset_size == 1:  # Medium scale
            plant_nums = random.randint(3, 4)
            dealer_nums = random.randint(30, 50)
            sku_nums = random.randint(40, 60)
            veh_type_nums = random.randint(2, 4)
        else:  # Large scale
            plant_nums = random.randint(4, 6)
            dealer_nums = random.randint(50, 90)
            sku_nums = random.randint(60, 100)
            veh_type_nums = random.randint(2, 4)

        # Generate identifiers for plants, dealers, SKUs, and vehicle types.
        plant_ids = generate_ids(length=4, nums=plant_nums, id_type='factory')
        dealer_ids = generate_ids(length=8, nums=dealer_nums, id_type='dealer')
        sku_ids = generate_ids(length=11, nums=sku_nums, id_type='sku')
        veh_types = generate_ids(length=3, nums=veh_type_nums, id_type='vehicle')
        logger.info(f"Generated ids: plants={len(plant_ids)}, dealers={len(dealer_ids)}, skus={len(sku_ids)}, veh_types={len(veh_types)}")

        # ==============================================================================
        # The following files are shared by the single- and multi-period problems.
        # Generate order data as {(dealer, sku): qty}.
        orders = generate_order(dealer_ids, sku_ids)
        # Generate the volume of each SKU as {sku: size}.
        sku_sizes = generate_sku_size(sku_ids)
        # Generate vehicle data:
        # {v_type: capacity}, {v_type: min_load}, and {v_type: cost}.
        veh_cap, veh_min_load, veh_cost = generate_vehicle(veh_types)

        # Write the generated data to the corresponding CSV files.
        order_to_csv(orders, file_loc=output_file_loc)
        logger.info(f"Saved order.csv to {output_file_loc / 'order.csv'}")
        sku_size_to_csv(sku_sizes, file_loc=output_file_loc)
        logger.info(f"Saved product_size.csv to {output_file_loc / 'product_size.csv'}")
        vehicle_to_csv(veh_cap, veh_min_load, veh_cost, file_loc=output_file_loc)
        logger.info(f"Saved vehicle.csv to {output_file_loc / 'vehicle.csv'}")
        # ==============================================================================

        if horizons == 1:  # Single period
            # Generate SKU inventory data as {plant: {sku: qty}}.
            storage = generate_storage(plant_ids, sku_ids, orders)
            storage_to_csv(storage, file_loc=output_file_loc)
        else:  # Multiple periods
            # Total demand for each SKU across all dealers and periods.
            sku_demands = aggregate_order(orders)
            
            # Compute total demand across all SKUs.
            total_sku_qty = sum(sku_demands.values())  # Total demand from all dealers
            # Add a 10% allowance so plants have sufficient capacity.
            total_sku_qty = int(total_sku_qty * 1.1)
            # Inventory capacity limit for each plant as {plant: capacity}.
            # Allocate total demand across plants with limited imbalance and a
            # small 5%-25% buffer so total capacity is not overly loose.
            # The planning horizon also accounts for peak production pressure.
            plant_capacity = generate_plant_inv_limit(plant_ids, total_sku_qty, horizons)
            plant_capacity_to_csv(plant_capacity, file_loc=output_file_loc)
            
            # SKU inventory and scheduled production:
            # {plant: {sku: qty}} and {plant: {(sku, day): qty}}.
            storage, production = generate_storage_and_production(plant_ids, sku_ids, orders, plant_capacity, horizons)
            storage_to_csv(storage, file_loc=output_file_loc)
            production_to_csv(production, file_loc=output_file_loc)
        # Collect dataset statistics, including supply-demand aggregates for
        # later diagnostics. Total demand sums all dealer demand for all SKUs
        # across all periods.
        sku_demands = aggregate_order(orders)
        total_demand_qty = int(sum(sku_demands.values())) if sku_demands else 0

        # Compute total initial inventory and total production. The production
        # dictionary exists only for multi-period datasets.
        total_initial_inv = 0
        total_production_qty = 0
        # Single period: storage is {plant: {sku: qty}}.
        # Multi-period: storage has the same shape, while production is
        # {plant: {(sku, day): qty}}.
        try:
            # storage is created for both single- and multi-period datasets.
            for plant, sku_map in storage.items():
                for sku, qty in sku_map.items():
                    total_initial_inv += int(qty)
        except NameError:
            total_initial_inv = 0

        try:
            for plant, prod_map in production.items():
                for _, qty in prod_map.items():
                    total_production_qty += int(qty)
        except NameError:
            total_production_qty = 0

        total_supply_qty = int(total_initial_inv + total_production_qty)
        supply_demand_ratio = round(total_supply_qty / total_demand_qty, 3) if total_demand_qty > 0 else None

        dataset_statistics.append({
            "dataset_name": f"dataset_{i}",
            "plant_nums": plant_nums,
            "dealer_nums": dealer_nums,
            "sku_nums": sku_nums,
            "vehicle_type_nums": veh_type_nums,
            "horizons": horizons,
            "total_demand_qty": total_demand_qty,
            "total_initial_inv": total_initial_inv,
            "total_production_qty": total_production_qty,
            "total_supply_qty": total_supply_qty,
            "supply_demand_ratio": supply_demand_ratio
        })
        elapsed = time.time() - start_time
        logger.info(f"Finished dataset {i}/{dataset_nums} in {elapsed:.2f}s")
    # Save summary statistics after all datasets have been generated.
    save_dataset_statistics(dataset_statistics, file_loc=base_file_loc)
    print(f"All datasets have been saved in the location {base_file_loc}")
    print("Success!")

def save_dataset_statistics(dataset_statistics, file_loc):
    """
    Save size statistics for each generated dataset.

    :param dataset_statistics: list containing statistics for all datasets
    :param file_loc: output directory
    :return: None
    """
    df_statistics = pd.DataFrame(dataset_statistics)
    file_loc = Path(file_loc)
    file = file_loc / "instance_statistics.csv"
    df_statistics.to_csv(file, index=False)

def order_to_csv(orders, file_loc):
    """
    Write order data to a CSV file.

    :param orders: order data
    :param file_loc: output directory
    :return: None
    """
    rows = []
    if args.problem == "single":
        for (dealer, sku), qty in orders.items():
            rows.append({
                'client_code': dealer,
                'product_code': sku,
                'volume': qty
            })
    else:  # Multi-period: total demand for each SKU across all periods.
        for (dealer, sku), qty in orders.items():
            rows.append({
                'client_code': dealer,
                'product_code': sku,
                'volume': qty,
            })
    df_order = pd.DataFrame(rows)
    file_loc = Path(file_loc)
    file = file_loc / "order.csv"
    df_order.to_csv(file, index=False)  # Do not write the DataFrame index.

def sku_size_to_csv(sku_sizes, file_loc):
    """
    Write SKU volume data to a CSV file.

    :param sku_sizes: volume of each SKU
    :param file_loc: output directory
    :return: None
    """
    rows = []
    for sku, size in sku_sizes.items():
        rows.append({
            'product_code': sku,
            'standard_size': size
        })
    df_sku_size = pd.DataFrame(rows)
    file_loc = Path(file_loc)
    file = file_loc / "product_size.csv"
    df_sku_size.to_csv(file, index=False)

def vehicle_to_csv(veh_cap, veh_min_load, veh_cost, file_loc):
    """
    Write vehicle type data to a CSV file.

    :param veh_cap: vehicle capacity by type
    :param veh_min_load: minimum shipment quantity by vehicle type
    :param veh_cost: vehicle usage cost by type
    :param file_loc: output directory
    :return: None
    """
    if set(veh_cap.keys()) != set(veh_min_load.keys()) or set(veh_cap.keys()) != set(veh_cost.keys()):
        raise ValueError("Input dictionaries must have the same keys (vehicle types)")

    rows = [{
        'vehicle_type': v_type,
        'carry_standard_size': veh_cap[v_type],
        'min_standard_size': veh_min_load[v_type],
        'cost_to_use': veh_cost[v_type]
    } for v_type in veh_cap]
    df_vehicle = pd.DataFrame(rows)
    file_loc = Path(file_loc)
    file = file_loc / "vehicle.csv"
    df_vehicle.to_csv(file, index=False)

def storage_to_csv(storage, file_loc):
    """
    Write SKU inventory data to a CSV file.

    :param storage: SKU inventory data
    :param file_loc: output directory
    :return: None
    """
    rows = []
    for plant, skus_supply in storage.items():
        for sku, qty in skus_supply.items():
            rows.append({
                'plant_code': plant,
                'product_code': sku,
                'volume': qty
            })
    df_storage = pd.DataFrame(rows)
    # Boolean indexing efficiently removes rows from a large DataFrame.
    # Specify the column to filter.
    col_name = 'volume'
    # Drop rows whose volume is zero and reset the DataFrame index.
    df_storage = df_storage[df_storage[col_name] != 0].reset_index(drop=True)
    file_loc = Path(file_loc)
    file = file_loc / "warehouse_storage.csv"
    df_storage.to_csv(file, index=False)

def plant_capacity_to_csv(plant_capacity, file_loc):
    """
    Write plant inventory capacity limits to a CSV file.

    :param plant_capacity: inventory capacity limit for each plant
    :param file_loc: output directory
    :return: None
    """
    rows = []
    for plant, cap in plant_capacity.items():
        rows.append({
            'plant_code': plant,
            'max_volume': cap
        })
        # print(f"DEBUG: plant = {plant}, max_volume = {cap}")
    df_plant_cap = pd.DataFrame(rows)
    file_loc = Path(file_loc)
    file = file_loc / "plant_size_upper_limit.csv"
    df_plant_cap.to_csv(file, index=False)

def production_to_csv(production, file_loc):
    """
    Write scheduled SKU production quantities to a CSV file.

    :param production: scheduled production quantity for each SKU
    :param file_loc: output directory
    :return: None
    """
    rows = []
    for plant, prod_data in production.items():
        for (sku, day), qty in prod_data.items():
            rows.append({
                'plant_code': plant,
                'product_code': sku,
                'produce_date': day,
                'volume': qty
            })
    df_production = pd.DataFrame(rows)
    col_name = 'volume'
    df_production = df_production[df_production[col_name] != 0].reset_index(drop=True)
    file_loc = Path(file_loc)
    file = file_loc / "warehouse_production.csv"
    df_production.to_csv(file, index=False)

def generate_ids(length, nums, id_type='default'):
    """
    Generate the requested number of identifiers with a fixed length.

    :param length: fixed identifier length
    :param nums: number of identifiers to generate
    :param id_type: identifier type: 'factory', 'vehicle', 'dealer', or 'sku'
    :return: list of generated identifiers
    """
    ids = set()  # Deduplicate identifiers with a set.

    for _ in range(nums):
        if id_type == 'factory':
            first = str(random.randint(1, 9))
            rest = ''.join(random.choices(string.digits, k=length - 1))
            gen_id = first + rest
        elif id_type == 'vehicle':
            v_type = generate_float_num(lb=4.0, ub=20.0, precision=1)
            gen_id = str(v_type)
        elif id_type == 'dealer':
            first = random.choice(string.ascii_uppercase)
            rest = ''.join(random.sample(string.digits, k=length-1))
            gen_id = first + rest
        elif id_type == 'sku':
            first = random.choice(string.ascii_uppercase)
            rest = ''.join(random.sample(string.digits, k=length-1))
            gen_id = first + rest
        else:
            gen_id = ''.join(random.sample(string.ascii_uppercase + string.digits, k=length))
        ids.add(gen_id)

    return list(ids)

def generate_order(dealers, skus):
    """
    Generate order data.

    :param dealers: dealer identifiers
    :param skus: SKU identifiers
    :return: dictionary containing order data
    """
    # SKUs required by each dealer.
    dealer_sku_map = {dealer: set() for dealer in dealers}
    sku_dealer_map = {sku: set() for sku in skus}
    
    # Ensure that every dealer demands at least one SKU.
    for dealer in dealers:
        nums = random.randint(1, len(skus))
        selected_skus = random.sample(skus, nums)
        for sku in selected_skus:
            dealer_sku_map[dealer].add(sku)
            sku_dealer_map[sku].add(dealer)

    # Ensure that every SKU is demanded by at least one dealer.
    for sku in skus:
        if not sku_dealer_map[sku]:  # No dealer currently demands this SKU.
            dealer = random.choice(dealers)
            dealer_sku_map[dealer].add(sku)
            sku_dealer_map[sku].add(dealer)

    orders = {}
    if args.problem == "single":
        for dealer, skus_need in dealer_sku_map.items():
            for sku in skus_need:
                orders[dealer, sku] = int(round(random.uniform(5, 40)))
    
    else:  # Multi-period: total demand for each SKU across all periods.
        for dealer, skus_need in dealer_sku_map.items():
            for sku in skus_need:
                orders[dealer, sku] = int(round(random.uniform(5, 40)))
    return orders

def generate_float_num(lb, ub, precision=1):
    """
    Generate a floating-point number with the requested precision.

    :param lb: lower bound
    :param ub: upper bound
    :param precision: number of decimal places
    """
    return round(random.uniform(lb, ub), precision)

def generate_sku_size(skus):
    """
    Generate volume data for every SKU.

    :param skus: SKU identifiers
    :return: dictionary mapping each SKU to its volume
    """
    sku_sizes = dict.fromkeys(skus)
    for sku in skus:
        sku_sizes[sku] = generate_float_num(lb=1.0, ub=3.0, precision=1)
    return sku_sizes

def generate_vehicle(veh_types):
    """
    Generate capacity, minimum shipment quantity, and usage cost by vehicle type.

    :param veh_types: vehicle types
    :return: dictionaries for vehicle capacity, minimum shipment quantity, and cost
    """
    veh_cap = {}
    veh_min_load = {}
    veh_cost = {}
    # Sort vehicle types in ascending order.
    sorted_veh_types = sorted(veh_types, key=float)
    for i, v_type in enumerate(sorted_veh_types):
        type_value = float(v_type)
        # Compute base capacity from the vehicle type value.
        base_cap = int(10 * type_value)
        # Randomize actual capacity between 80% and 120% of the base value.
        lb = round(base_cap * 0.8)
        ub = round(base_cap * 1.2)
        if i == 0:
            veh_cap[v_type] = random.randint(lb, ub)
        else:
            # Previous vehicle type.
            pre_type = sorted_veh_types[i - 1]
            pre_type_value = float(pre_type)
            # Capacity of the previous vehicle type.
            pre_veh_cap = veh_cap[pre_type]
            if type_value - pre_type_value <= 1:  # The two vehicle types are close.
                # Ensure that the current capacity is not smaller than the previous one.
                lb = pre_veh_cap + random.randint(1, 5)
                ub = pre_veh_cap + random.randint(6, 10)
                veh_cap[v_type] = random.randint(lb, ub)
            else:
                lb = pre_veh_cap + random.randint(10, 30)
                ub = pre_veh_cap + random.randint(40, 60)
                veh_cap[v_type] = random.randint(lb, ub)
        # Capacity of the current vehicle type.
        capacity = veh_cap[v_type]
        # Set the minimum shipment quantity to 30%-50% of vehicle capacity.
        lb = round(0.3 * capacity)
        ub = round(0.5 * capacity)
        veh_min_load[v_type] = random.randint(lb, ub)
        # Cost of each vehicle type. Larger vehicles must cost more than
        # smaller vehicles, while economies of scale keep one large vehicle
        # cheaper than two small vehicles.
        min_cost = 1
        if i == 0:
            veh_cost[v_type] = min_cost
        else:
            # Previous vehicle type.
            pre_type = sorted_veh_types[i - 1]
            # Usage cost of the previous vehicle type.
            pre_veh_cost = veh_cost[pre_type]
            # Vehicle cost grows with capacity, but with a small coefficient.
            # Difference in capacity between the two vehicle types.
            cap_diff = veh_cap[v_type] - veh_cap[pre_type]
            cost_increase = round(cap_diff * 0.8) / veh_cap[pre_type]
            cost_increase = round(cost_increase, 2)
            veh_cost[v_type] = pre_veh_cost + cost_increase

    return veh_cap, veh_min_load, veh_cost

def aggregate_order(orders):
    """
    Aggregate dealer demand by SKU.

    :param orders: dealer demand by SKU
    :return: dictionary containing demand quantities by SKU
    """
    sku_demands = defaultdict(int)
    if args.problem == "single":
        for (dealer_id, sku_id), qty in orders.items():
            sku_demands[sku_id] += qty
    else:  # Multi-period
        for (dealer_id, sku_id), qty in orders.items():
            sku_demands[sku_id] += qty
    return dict(sku_demands)

def create_plant_sku_mapping(plants, skus):
    """
    Create a mapping from each plant to the SKUs it can supply.

    :param plants: plant identifiers
    :param skus: SKU identifiers
    :return: dictionary mapping each plant to its supplyable SKU set
    """
    # SKUs that each plant can supply.
    plant_sku_map = {plant: set() for plant in plants}
    sku_plant_map = {sku: set() for sku in skus}
    
    # Ensure that every plant supplies at least one SKU.
    for plant in plants:
        nums = random.randint(1, len(skus))
        selected_skus = random.sample(skus, nums)
        for sku in selected_skus:
            plant_sku_map[plant].add(sku)
            sku_plant_map[sku].add(plant)
    
    # Ensure that every SKU is supplied by at least one plant.
    for sku in skus:
        if not sku_plant_map[sku]:  # No plant currently supplies this SKU.
            plant = random.choice(plants)
            plant_sku_map[plant].add(sku)
            sku_plant_map[sku].add(plant)

    return plant_sku_map

def generate_storage(plants, skus, orders):
    """
    Generate the single-period inventory data and ensure that total inventory
    for each SKU exceeds total dealer demand for that SKU.

    :param plants: plant identifiers
    :param skus: SKU identifiers
    :param orders: dealer demand by SKU
    :return: dictionary containing SKU inventory data
    """
    plant_sku_map = create_plant_sku_mapping(plants, skus)
    storage = {}  # Nested dictionary: {plant: {sku: qty}}.
    for plant, skus_supply in plant_sku_map.items():
        storage[plant] = {}
        for sku in skus_supply:
            # Initialize inventory quantity to 1.
            storage[plant][sku] = 1

    # Total demand for each SKU.
    sku_demands = aggregate_order(orders)
    for sku, demand in sku_demands.items():
        # Existing inventory for the SKU.
        total_inv = sum(storage[plant].get(sku, 0) for plant in plant_sku_map)
        # Add a random positive buffer so inventory exceeds demand.
        demand += random.randint(10, 50)
        remaining_demand = int(demand - total_inv)
        if remaining_demand > 0:
            # Find all plants that can supply this SKU.
            available_plants = [plant for plant in plant_sku_map if sku in plant_sku_map[plant]]
            # Allocate the remaining demand across these plants.
            while remaining_demand >0 and available_plants:
                plant = random.choice(available_plants)
                allocation = random.randint(1, remaining_demand)
                storage[plant][sku] += allocation
                remaining_demand -= allocation
                if remaining_demand == 0:
                    break
                # Remove the plant from the available list after its allocation.
                available_plants.remove(plant)

            # Distribute any remaining demand across all available plants.
            if remaining_demand > 0:
                available_plants = [plant for plant in plant_sku_map if sku in plant_sku_map[plant]]
                per_plant_qty = remaining_demand // len(available_plants) + 1
                for plant in available_plants:
                    storage[plant][sku] += per_plant_qty
                    remaining_demand -= per_plant_qty
                    if remaining_demand <= 0:
                        break
    return storage

def total_production(production, sku):
    """
    Compute total production of an SKU across all plants and periods.

    :param production: scheduled SKU production by plant and period
    :param sku: target SKU
    :return: total production of the SKU across all plants and periods
    """
    sku_production = defaultdict(int)
    for plant in production:
        for (sku_id, day), qty in production[plant].items():
            if sku_id == sku:
                sku_production[sku_id, day] += qty
    return dict(sku_production)

def weighted_choice(available_plants, weights):
    """
    Select a plant with weighted random sampling.

    :param available_plants: candidate plants
    :param weights: sampling weights
    """
    # print(f"DEBUG: weights = {weights}, sum(weights) = {np.sum(weights)}")  # Debug output.
    weights = [abs(w) for w in weights]
    if sum(weights) != 0:
        normalized_weights = np.array(weights) / sum(weights)
        return np.random.choice(available_plants, p=normalized_weights)
    else:
        return random.choice(available_plants)

def generate_plant_inv_limit(plants, total_sku_qty, horizons=1):
    """
    Compute an inventory capacity limit for each plant.

    The method:
    - allocates total demand across plants as base shares, with a random
      imbalance factor to represent plant-size differences;
    - adds only a small 5%-25% buffer so capacities are not overly loose;
    - uses the planning horizon to estimate peak production pressure; and
    - scales capacities proportionally when total capacity is below total demand.

    :param plants: plant identifiers (list)
    :param total_sku_qty: total quantity across all SKUs (integer)
    :param horizons: planning horizon used to estimate peak production
    :return: dictionary mapping each plant to its inventory capacity
    """
    if not plants:
        return {}

    n_plants = len(plants)
    # Base share: distribute the total evenly across plants.
    avg_share = float(total_sku_qty) / n_plants if n_plants > 0 else float(total_sku_qty)

    plant_inv_limit = {}
    for plant in plants:
        # Apply an imbalance factor to represent plant-size differences (0.6-1.4).
        imbalance = random.uniform(0.6, 1.4)
        base = max(1, int(avg_share * imbalance))

        # Estimate the single-period peak production each plant may need.
        # Conservatively divide total demand by the horizon and plant count.
        per_period_expected = max(1, int(round(total_sku_qty / max(1, horizons) / n_plants)))

        # Use a small 5%-25% buffer to avoid excessively loose capacity.
        buffer = random.uniform(0.05, 0.25)

        # Keep capacity above the buffered base and reserve for a single-period peak.
        cap = int(max(base * (1.0 + buffer), per_period_expected * 2))
        plant_inv_limit[plant] = cap

    # Scale capacities proportionally when total capacity is below total demand.
    total_cap = sum(plant_inv_limit.values())
    if total_cap < total_sku_qty:
        if total_cap == 0:
            # Fallback protection: distribute evenly.
            for plant in plant_inv_limit:
                plant_inv_limit[plant] = max(1, int(total_sku_qty // n_plants))
        else:
            scale = float(total_sku_qty) / total_cap
            for plant in plant_inv_limit:
                plant_inv_limit[plant] = max(1, int(round(plant_inv_limit[plant] * scale)))

    return plant_inv_limit

def distribute_prod(prod, selected_periods):
    """
    Distribute a production quantity across the selected periods.

    :param prod: total production quantity to distribute
    :param selected_periods: periods that receive production
    :return: dictionary mapping each period to its allocated quantity
    """
    if not selected_periods:
        return {}

    n = len(selected_periods)
    # Compute the base quantity for each period.
    base_amount = prod // n
    # Compute the number of units left after the base allocation.
    remaining = prod % n
    distribution = {}
    for i, period in enumerate(selected_periods):
        if i < remaining:
            distribution[period] = base_amount + 1
        else:
            distribution[period] = base_amount
    return distribution

def generate_storage_and_production(plants, skus, orders, plant_capacity, horizons):
    """
    Generate inventory and scheduled production for the multi-period problem.
    For each SKU, total supply across all periods, consisting of production
    plus initial inventory, must exceed total dealer demand.

    :param plants: plant identifiers
    :param skus: SKU identifiers
    :param orders: dealer demand by SKU
    :param plant_capacity: capacity limit for each plant
    :param horizons: planning horizon
    :return: SKU inventory dictionary and scheduled production dictionary
    """
    # 1. Compute total demand per SKU and add a buffer of 10-50 units.
    sku_demands = aggregate_order(orders)
    sku_total_supply = {sku: int(np.ceil(sku_demands.get(sku, 0) + random.randint(10, 50))) for sku in skus}

    # Protection: keep global supply within total plant capacity while
    # ensuring that each SKU supply is at least its demand.
    plant_remain_cap = plant_capacity.copy()
    try:
        total_supply_desired = sum(int(v) for v in sku_total_supply.values())
        total_cap = sum(int(v) for v in plant_capacity.values()) if plant_capacity else 0
        total_demand = sum(int(v) for v in sku_demands.values()) if sku_demands else 0
    except Exception:
        total_supply_desired = sum(sku_total_supply.values())
        total_cap = sum(plant_capacity.values()) if plant_capacity else 0
        total_demand = sum(sku_demands.values()) if sku_demands else 0

    if total_cap <= 0:
        # No valid capacity constraint; log a warning and keep the allocation logic.
        logger.warning("Total plant capacity is zero or undefined; generated per-SKU supply may exceed capacity.")
    else:
        # Extra supply allowed above minimum total demand: capacity - demand.
        allowed_extra = max(0, total_cap - total_demand)
        current_extra = max(0, total_supply_desired - total_demand)

        if current_extra > 0 and current_extra > allowed_extra:
            # Scale down only the portion above demand so total capacity is respected.
            scale = allowed_extra / float(current_extra)
            for sku in list(sku_total_supply.keys()):
                base = int(sku_demands.get(sku, 0))
                extra = max(0, int(sku_total_supply[sku]) - base)
                new_extra = int(round(extra * scale))
                sku_total_supply[sku] = base + new_extra
            logger.info(
                "Scaled down SKU extras to fit plant capacity: desired=%d, cap=%d, scaled_total=%d",
                total_supply_desired, total_cap, sum(sku_total_supply.values())
            )
    # -----------------------------------------------------------------------
    plant_sku_supply = {plant: {sku: 0 for sku in skus} for plant in plants}
    for sku in skus:
        remaining = sku_total_supply[sku]
        # Allocate with vectorized operations to avoid slow unit loops and
        # potential infinite loops.
        n_plants = len(plants)
        plant_indices = list(range(n_plants))
        # Build a NumPy array of remaining capacity for each plant.
        plant_remain_arr = np.array([int(plant_remain_cap.get(p, 0)) for p in plants], dtype=np.int64)

        total_cap = int(plant_remain_arr.sum())
        if total_cap <= 0:
            # No available capacity: distribute evenly across all plants to
            # ensure that SKU supply remains at least demand.
            logger.warning(f"No remaining capacity detected when allocating SKU {sku}; forcing equal distribution to guarantee supply >= demand.")
            # Split the remainder across plants, assigning one extra unit to
            # the first plants when needed.
            n_plants = len(plants) if plants else 0
            if n_plants > 0:
                per = remaining // n_plants
                extra = remaining % n_plants
                for idx, plant in enumerate(plants):
                    add = per + (1 if idx < extra else 0)
                    if add > 0:
                        plant_sku_supply[plant][sku] += int(add)
                        # Update plant_remain_arr even if plant_remain_cap was zero.
                        plant_remain_arr[idx] = max(0, plant_remain_arr[idx] - int(add))
                # Write back remaining capacity with a lower bound of zero.
                for idx, plant in enumerate(plants):
                    plant_remain_cap[plant] = int(max(0, plant_remain_arr[idx]))
                # Forced allocation is complete; skip proportional allocation.
                continue
            else:
                logger.warning(f"No plants available to force-distribute SKU {sku!s}; leaving unassigned.")
                continue

        # Initial allocation proportional to remaining capacity, rounded down.
        alloc = (remaining * plant_remain_arr) // total_cap
        # Do not exceed each plant's remaining capacity.
        alloc = np.minimum(alloc, plant_remain_arr).astype(np.int64)

        # Apply the initial allocation.
        for idx, plant in enumerate(plants):
            qty = int(alloc[idx])
            plant_sku_supply[plant][sku] = qty
            plant_remain_arr[idx] -= qty
        remaining_after_alloc = int(remaining - int(alloc.sum()))

        # Greedily fill any remaining units, starting with the largest
        # remaining capacities.
        if remaining_after_alloc > 0:
            # Sort plant indices by remaining capacity in descending order.
            order = np.argsort(-plant_remain_arr)
            rem = remaining_after_alloc
            for idx in order:
                if rem <= 0:
                    break
                cap_here = int(plant_remain_arr[idx])
                if cap_here <= 0:
                    continue
                take = min(cap_here, rem)
                plant_name = plants[int(idx)]
                plant_sku_supply[plant_name][sku] += int(take)
                plant_remain_arr[idx] -= int(take)
                rem -= int(take)
            if rem > 0:
                # Force the remaining demand into available plants, ignoring
                # current plant_remain_cap to guarantee supply >= demand.
                available_plants = [plant for plant in plants if sku in plant_sku_supply[plant]]
                if available_plants:
                    per = rem // len(available_plants)
                    extra = rem % len(available_plants)
                    for p in available_plants:
                        add = per + (1 if extra > 0 else 0)
                        if add > 0:
                            plant_sku_supply[p][sku] += int(add)
                            rem -= int(add)
                        if extra > 0:
                            extra -= 1
                    # Assign any residual units to the first available plant.
                    if rem > 0:
                        plant_sku_supply[available_plants[0]][sku] += int(rem)
                        rem = 0
                    logger.warning(f"Forced fill: assigned remaining units of SKU {sku!s} to plants to guarantee supply >= demand.")
                else:
                    logger.warning(f"No available plants to force-fill SKU {sku!s}; {rem} units remain unassigned.")
        # Write updated remaining capacity back to plant_remain_cap.
        for idx, plant in enumerate(plants):
            plant_remain_cap[plant] = int(plant_remain_arr[idx])

    storage = {plant: {sku: 0 for sku in skus} for plant in plants}
    production = {plant: {(sku, day): 0 for sku in skus for day in range(1, horizons + 1)} for plant in plants}
    for plant in plants:
        for sku in skus:
            total = plant_sku_supply[plant][sku]
            if total == 0:
                continue
            # Assign 10%-30% to initial inventory and the rest to production.
            inv = max(1, int(total * random.uniform(0.1, 0.3)))
            prod_total = int(max(0, total - inv))
            storage[plant][sku] = inv
            # Distribute production across periods so the total equals prod_total.
            days = list(range(1, horizons + 1))
            prod_dist = distribute_prod(prod_total, days)
            for day in days:
                production[plant][(sku, day)] = int(prod_dist.get(day, 0))

    return storage, production

# Run the dataset generator when this file is executed directly.
if __name__ == '__main__':
    generate_dataset()
