# 定义Vehicle类，用于模型可行解的构造
# Vehicle类需要提供的功能包括
# 1. 明确是哪个生产基地 i 为哪个经销商 j 使用哪种车型 f
# 2. 需要能够计算车辆上装载的商品的体积
# 3. 需要判断当前车辆是否满载
from .InputDataHeuristic import InputDataHeuristic
from dataclasses import dataclass, field
from typing import Dict

# dataclasses -- Data Classes
# This module provides a decorator and functions for automatically adding generated special methods such as
# __init__() and __repr__() to user-defined classes.


@dataclass
class Vehicle:  # fact_id, dealer_id, type, data 都是需要在实例化Vehicle对象时，要提供的
    fact_id: str
    dealer_id: str
    type: str
    data: InputDataHeuristic
    # Vehicle类的id，便于区分不同的车辆
    id = 0

    # dataclasses.field(*, default=MISSING, default_factory=MISSING, init=True, repr=True,
    # hash=None, compare=True, metadata=None, kw_only=MISSING)

    # default_factory: If provided, it must be a zero-argument callable that will be called when a default
    # value is needed for this field. Among other purposes, this can be used to specify fields with mutable
    # default values, as discussed below. It is an error to specify both default and default_factory.

    # init: If true (the default), this field is included as a parameter to the generated __init__() method.

    capacity: float = field(init=False)  # 车辆的可用体积
    cargo: Dict[str, int] = field(default_factory=dict)  # 车辆上装载的SKU以及数量

    def __post_init__(self):
        Vehicle.id += 1
        self.cargo = {}
        self.capacity = self.data.veh_type_cap[self.type]

    def get_avail_space(self):
        return self.capacity

    # sku_id代表要装载的SKU编号，num代表要装多少sku
    def load(self, sku_id: str, num: int):
        # 返回装车以后剩余多少sku装不上车，如果装10，剩余10个装不上，很可能满载了
        data = self.data
        sku_size = data.sku_sizes[sku_id]
        num_loaded = int(self.capacity // sku_size)  # 计算当前车辆容量可以装载的sku数量
        num_loaded = min(num, num_loaded)       # 实际装载的sku数量

        if num_loaded > 0:
            if sku_id in self.cargo:
                self.cargo[sku_id] += num_loaded
            else:
                self.cargo[sku_id] = num_loaded
            self.capacity -= num_loaded * sku_size
        return num - num_loaded  # 返回剩余多少sku未装载

    def clear(self):
        self.capacity = self.data.veh_type_cap[self.type]  # 重置车辆的可用体积为车辆空载时候的容量
        self.cargo.clear()  # dict.clear() method --> remove all items from the dictionary

    # 用于判断车辆上是否装载有货物
    def is_empty(self):
        if len(self.cargo) == 0:
            return True  # 如果未装载，返回 Ture
        return False
