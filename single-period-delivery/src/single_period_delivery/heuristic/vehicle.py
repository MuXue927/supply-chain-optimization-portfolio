# Define the Vehicle class used to construct feasible heuristic solutions.
# The class must track:
# 1. which production plant i serves which dealer j with which vehicle type f;
# 2. the volume of goods loaded onto the vehicle; and
# 3. whether the vehicle is full.
from dataclasses import dataclass, field

from .InputDataHeuristic import InputDataHeuristic

# dataclasses -- Data Classes
# This module provides a decorator and functions for automatically adding generated special methods such as
# __init__() and __repr__() to user-defined classes.


@dataclass
class Vehicle:  # fact_id, dealer_id, type, and data must be provided when instantiating a Vehicle.
    fact_id: str
    dealer_id: str
    type: str
    data: InputDataHeuristic
    # Class-level vehicle identifier used to distinguish vehicle instances.
    id = 0

    # dataclasses.field(*, default=MISSING, default_factory=MISSING, init=True, repr=True,
    # hash=None, compare=True, metadata=None, kw_only=MISSING)

    # default_factory: If provided, it must be a zero-argument callable that will be called when a default
    # value is needed for this field. Among other purposes, this can be used to specify fields with mutable
    # default values, as discussed below. It is an error to specify both default and default_factory.

    # init: If true (the default), this field is included as a parameter to the generated __init__() method.

    capacity: float = field(init=False)  # Remaining usable volume of the vehicle.
    cargo: dict[str, int] = field(default_factory=dict)  # SKUs loaded on the vehicle and their quantities.

    def __post_init__(self):
        Vehicle.id += 1
        self.cargo = {}
        self.capacity = self.data.veh_type_cap[self.type]

    def get_avail_space(self):
        return self.capacity

    # sku_id identifies the SKU to load, and num is the requested quantity.
    def load(self, sku_id: str, num: int):
        # Return the quantity that could not be loaded. If all 10 requested units remain unloaded, the vehicle is likely full.
        data = self.data
        sku_size = data.sku_sizes[sku_id]
        num_loaded = int(self.capacity // sku_size)  # Maximum quantity that fits in the remaining capacity.
        num_loaded = min(num, num_loaded)       # Actual quantity loaded.

        if num_loaded > 0:
            if sku_id in self.cargo:
                self.cargo[sku_id] += num_loaded
            else:
                self.cargo[sku_id] = num_loaded
            self.capacity -= num_loaded * sku_size
        return num - num_loaded  # Quantity that remains unloaded.

    def clear(self):
        self.capacity = self.data.veh_type_cap[self.type]  # Reset usable volume to the empty vehicle capacity.
        self.cargo.clear()  # dict.clear() method --> remove all items from the dictionary

    # Check whether the vehicle carries any cargo.
    def is_empty(self):
        if len(self.cargo) == 0:
            return True  # Return True when the vehicle is empty.
        return False
