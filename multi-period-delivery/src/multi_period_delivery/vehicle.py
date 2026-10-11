"""
vehicle

模块定位
    表示单次运输动作中的一辆可装载车辆实体。该类作为 ALNS 解状态的一部分，
    记录“某生产基地 -> 某经销商”在特定日、特定车辆类型下的装载决策。

核心职责
    - 标识信息: (fact_id, dealer_id, type, day) + 唯一 id
    - 容量管理: max_capacity / 当前剩余 capacity
    - 装载记录: cargo[(sku_id, day)] = 装载数量
    - 操作方法: load / clear / is_full / is_empty / get_avail_space

说明:
    - 引入 CargoDict：一个轻量的 dict 子类，用于在直接修改 cargo 时自动维护
      Vehicle._loaded_volume 与 capacity（避免代码库大量替换点）。
    - cargo 属性暴露为一个包装器（property），当外部赋值（例如 veh.cargo = other.cargo.copy()）
      时会自动包装为 CargoDict 并同步已装载体积与剩余容量。
"""

# =========================
# 项目内部依赖
# =========================
from .InputDataALNS import DataALNS

# =========================
# 标准库 / 类型
# =========================
from dataclasses import dataclass, field
from typing import Dict, Tuple, MutableMapping

# -------------------------
# CargoDict: 自动维护 loaded_volume 与 capacity 的 dict wrapper
# -------------------------
class CargoDict(dict):
    """
    A dict-like container that updates its parent Vehicle's _loaded_volume and capacity
    whenever items are added/removed/updated in the mapping.

    注意:
      - copy() 返回一个内建 dict（调用方若将其赋值给 veh.cargo 会触发 Vehicle.cargo setter来重新包装）
      - 该类仅在 Vehicle 实例内部用于保证对 cargo 的原地修改能同步缓存
    """
    def __init__(self, vehicle, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._veh = vehicle

    def _sku_size(self, sku):
        try:
            return float(self._veh.data.sku_sizes.get(sku, 0))
        except Exception:
            return 0.0

    def __setitem__(self, key, value):
        prev = self.get(key, 0)
        super().__setitem__(key, value)
        try:
            sku, d = key
            delta = (value - prev) * self._sku_size(sku)
            self._veh._loaded_volume = getattr(self._veh, "_loaded_volume", 0.0) + delta
            # keep capacity consistent and non-negative
            self._veh.capacity = max(0, self._veh.max_capacity - getattr(self._veh, "_loaded_volume", 0.0))
        except Exception:
            # best-effort: leave vehicle state as-is on error
            pass

    def __delitem__(self, key):
        prev = self.get(key, 0)
        super().__delitem__(key)
        try:
            sku, d = key
            delta = - prev * self._sku_size(sku)
            self._veh._loaded_volume = max(0, getattr(self._veh, "_loaded_volume", 0.0) + delta)
            self._veh.capacity = max(0, self._veh.max_capacity - getattr(self._veh, "_loaded_volume", 0.0))
        except Exception:
            pass

    def clear(self):
        super().clear()
        try:
            self._veh._loaded_volume = 0
            self._veh.capacity = self._veh.max_capacity
        except Exception:
            pass

    def update(self, *args, **kwargs):
        other = dict(*args, **kwargs)
        for k, v in other.items():
            self.__setitem__(k, v)

    def pop(self, key, default=None):
        prev = self.get(key, default)
        res = super().pop(key, default)
        try:
            sku, d = key
            self._veh._loaded_volume = max(0, getattr(self._veh, "_loaded_volume", 0.0) - prev * self._sku_size(sku))
            self._veh.capacity = max(0, self._veh.max_capacity - getattr(self._veh, "_loaded_volume", 0.0))
        except Exception:
            pass
        return res

    def popitem(self):
        k, v = super().popitem()
        try:
            sku, d = k
            self._veh._loaded_volume = max(0, getattr(self._veh, "_loaded_volume", 0.0) - v * self._sku_size(sku))
            self._veh.capacity = max(0, self._veh.max_capacity - getattr(self._veh, "_loaded_volume", 0.0))
        except Exception:
            pass
        return k, v

    def copy(self):
        # return a plain dict to match expectations of other code paths
        return dict(self)

# =========================
# 数据类: Vehicle
# =========================
@dataclass
class Vehicle:
    """
    车辆实体:
        fact_id: str
        dealer_id: str
        type: str
        day: int
        data: DataALNS

    动态属性（init=False）:
        id, max_capacity, capacity, cargo (包装为 CargoDict)
    """
    fact_id: str
    dealer_id: str
    type: str
    day: int
    data: DataALNS

    id: int = field(init=False, default=-1)
    max_capacity: int = field(init=False, default=0)
    capacity: int = field(init=False, default=0)
    # cargo 使用 init=False 并通过 property 管理，以便在赋值时自动包装为 CargoDict
    cargo: Dict[Tuple[str, int], int] = field(init=False, repr=False, default_factory=dict)

    _id_counter = 0  # 类变量

    def __post_init__(self):
        # 分配唯一 id
        self.id = Vehicle._id_counter
        Vehicle._id_counter += 1
        # 初始化容量
        try:
            self.max_capacity = int(self.data.veh_type_cap[self.type])
        except Exception:
            self.max_capacity = 0
        self.capacity = self.max_capacity
        # 初始化装载记录为 CargoDict，以保证原地修改能更新缓存
        try:
            self._cargo = CargoDict(self, {})
        except Exception:
            # 兼容性保护
            self._cargo = {}
        # 缓存已装载总体积，避免频繁遍历 cargo 计算体积
        self._loaded_volume = 0

    # cargo property: 对外以 veh.cargo 访问, 赋值时会包装并同步 _loaded_volume/capacity
    @property
    def cargo(self):
        return self._cargo

    @cargo.setter
    def cargo(self, mapping):
        # mapping 可能是普通 dict 或其他映射；包装为 CargoDict 并同步已装载量
        try:
            if isinstance(mapping, CargoDict):
                new_map = CargoDict(self, dict(mapping))
            else:
                new_map = CargoDict(self, dict(mapping) if mapping is not None else {})
            # 计算已装载体积
            loaded = 0
            for (sku_k, d_k), q_k in new_map.items():
                try:
                    loaded += q_k * float(self.data.sku_sizes.get(sku_k, 0))
                except Exception:
                    continue
            self._cargo = new_map
            self._loaded_volume = loaded
            self.capacity = max(0, self.max_capacity - loaded)
        except Exception:
            # 兼容性回退: 将 cargo 赋为普通 dict 并尽量保留旧缓存
            try:
                self._cargo = dict(mapping) if mapping is not None else {}
            except Exception:
                self._cargo = {}
            self._loaded_volume = getattr(self, "_loaded_volume", 0)
            self.capacity = getattr(self, "capacity", self.max_capacity)

    def __eq__(self, other):
        # Use id-only equality for speed: id is unique per Vehicle instance
        if not isinstance(other, Vehicle):
            return False
        return self.id == other.id

    def __hash__(self):
        # Hash based on unique id for faster hashing/comparisons
        return hash(self.id)

    def __repr__(self):
        return (f"Vehicle(id={self.id}, fact_id={self.fact_id}, dealer_id={self.dealer_id}, "
                f"type={self.type}, day={self.day}, capacity={self.capacity}, cargo={self.cargo})")

    def get_avail_space(self) -> int:
        """
        返回当前剩余容量 (体积单位)
        """
        return self.capacity

    def is_full(self) -> bool:
        """
        判断车辆是否已无剩余容量
        """
        return self.capacity <= 0

    def load(self, sku_id: str, num: int) -> int:
        """
        试图装载指定 SKU 数量（原子/防御性实现）:
            - 先做本地计算，只有在确认不会导致不一致时再提交状态更新
            - 若 SKU 不存在抛出异常
            - 返回未成功装载的剩余件数 (num - 实际装载)
        """
        if num <= 0:
            return num
        if sku_id not in self.data.sku_sizes:
            raise ValueError(f"SKU {sku_id} not found in data.sku_sizes")
        sku_size = float(self.data.sku_sizes[sku_id])

        # 计算可装载数量（基于剩余容量），确保整数除法语义一致
        max_by_cap = 0
        if sku_size > 0:
            max_by_cap = int(self.capacity // sku_size)
        else:
            # 防御性处理: sku_size 不应为 0，但若发生则阻止装载以避免无限循环/错误
            raise ValueError(f"Invalid sku_size=0 for SKU {sku_id}")

        num_loaded = min(num, max_by_cap)

        if num_loaded <= 0:
            return num  # 无法装载任何件数

        # 计算新的状态（本地变量），确保原子提交
        prev_qty = self._cargo.get((sku_id, self.day), 0)
        new_qty = prev_qty + num_loaded
        new_capacity = self.capacity - num_loaded * sku_size
        new_loaded_volume = getattr(self, "_loaded_volume", 0.0) + num_loaded * sku_size

        # 防御性检查：不应出现负容量或不一致的已装载体积
        if new_capacity < 0 or new_loaded_volume < 0:
            # 保护：不提交任何修改，报告异常
            raise RuntimeError(f"Load would cause negative capacity or loaded volume (sku_id={sku_id}, load={num_loaded}, new_capacity={new_capacity})")

        # 提交状态更新（原子） — 使用 CargoDict 的 __setitem__ 以保持缓存一致
        self._cargo[(sku_id, self.day)] = new_qty
        # __setitem__ 会更新 _loaded_volume 与 capacity，但确保一致性
        self._loaded_volume = new_loaded_volume
        self.capacity = new_capacity

        return num - num_loaded  # 返回未装载数量

    def clear(self):
        """
        重置车辆: 清空装载并恢复剩余容量
        """
        try:
            # 使用 CargoDict.clear 保证缓存一致
            if isinstance(self._cargo, CargoDict):
                self._cargo.clear()
            else:
                self._cargo.clear()
        except Exception:
            pass
        self.capacity = self.max_capacity
        self._loaded_volume = 0

    def is_empty(self) -> bool:
        """
        判断是否尚未装载任何货物
        """
        try:
            return not bool(self._cargo)
        except Exception:
            return True

    def shallow_clone(self):
        """
        Create a shallow clone of this Vehicle without incrementing the global id counter.
        The clone shares immutable references (data) and copies mutable fields (cargo dict, capacity).
        Used by optimized state.copy() to avoid expensive deepcopy in hot paths.
        """
        # Create instance without calling __init__ to avoid changing _id_counter
        obj = object.__new__(Vehicle)
        # copy simple/immutable attributes
        obj.fact_id = self.fact_id
        obj.dealer_id = self.dealer_id
        obj.type = self.type
        obj.day = self.day
        obj.data = self.data
        # preserve identity fields and capacities
        obj.id = self.id
        obj.max_capacity = self.max_capacity
        obj.capacity = self.capacity
        # shallow copy mutable cargo mapping (keys/values are small primitives)
        # use the cargo property setter to wrap the dict into CargoDict and sync loaded_volume
        try:
            obj.cargo = self._cargo.copy() if isinstance(self._cargo, dict) else dict(self._cargo)
        except Exception:
            obj._cargo = {}
            obj._loaded_volume = getattr(self, "_loaded_volume", 0)
        # preserve cached loaded volume if not set by setter
        if not hasattr(obj, "_loaded_volume"):
            obj._loaded_volume = getattr(self, "_loaded_volume", 0)
        return obj
