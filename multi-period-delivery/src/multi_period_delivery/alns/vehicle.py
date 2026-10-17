"""
vehicle

Module role
    Represents one loadable vehicle in a single transportation action. This
    class is part of the ALNS solution state and records the loading decision
    from a plant to a dealer on a specific day and vehicle type.

Responsibilities
    - Identity: (fact_id, dealer_id, type, day) plus a unique id.
    - Capacity management: max_capacity and current remaining capacity.
    - Cargo record: cargo[(sku_id, day)] = loaded quantity.
    - Operations: load / clear / is_full / is_empty / get_avail_space.

Implementation notes:
    - CargoDict is a lightweight dict subclass that keeps
      Vehicle._loaded_volume and capacity synchronized when cargo is modified
      in place.
    - The cargo property wraps external assignments (for example,
      veh.cargo = other.cargo.copy()) in CargoDict and synchronizes loaded
      volume and remaining capacity.
"""

# =========================
# Standard library / typing
# =========================
from dataclasses import dataclass, field

# =========================
# Internal project dependencies
# =========================
from .InputDataALNS import DataALNS

# -------------------------
# CargoDict: a dict wrapper that maintains loaded_volume and capacity.
# -------------------------
_CARGO_MUTATIONS = 0
_SOLUTION_MUTATIONS = 0


def cargo_mutation_count() -> int:
    """
    Monotonic count of in-place cargo changes, process-wide.

    ``CargoDict`` is the only writer of cargo contents (every ``cargo[...] = ...``
    site in the codebase goes through ``__setitem__``), so this changes exactly
    when a load changes.

    Process-wide (not per-state) on purpose: vehicles are shared between a state
    and its snapshots, so a per-state counter would miss changes made through the
    copy. A global counter can only err by counting *more* changes than a given
    state saw, which costs a needless rebuild, never a stale reuse.
    """
    return _CARGO_MUTATIONS


def solution_mutation_count() -> int:
    """
    Monotonic count of declared solution mutations, process-wide.

    Bumped by ``SolutionState.mark_objective_dirty``, which every accepted
    mutation calls - including the structural ones a cargo counter cannot see,
    such as ``state.vehicles[i] = ...`` or replacing the vehicle list.

    Kept separate from :func:`cargo_mutation_count` so that marking a state dirty
    when nothing actually changed does not needlessly invalidate the inventory
    cache for every other state in the process.
    """
    return _SOLUTION_MUTATIONS


def note_solution_mutation() -> None:
    """Record a declared solution mutation (see :func:`solution_mutation_count`)."""
    global _SOLUTION_MUTATIONS
    _SOLUTION_MUTATIONS += 1


def _note_cargo_mutation() -> None:
    global _CARGO_MUTATIONS
    _CARGO_MUTATIONS += 1


def inventory_stamp(vehicles=()) -> tuple:
    """
    Value that changes whenever anything feeding the shipments array changes.

    Compared as a whole for cache validity, so it must be cheap and total. The
    previous approach re-derived a per-vehicle digest on every check, which
    profiling on the medium instance showed costing ~1.6 ms - as much as the
    inventory rebuild it was meant to avoid. This is O(1).

    Three sources, because no single one is complete:

    * cargo changes, counted by ``CargoDict``;
    * declared mutations, counted by ``SolutionState.mark_objective_dirty``;
    * the vehicle list's shape - its length and last id - which catches the
      ``state.vehicles = [...]`` and ``state.vehicles.append(...)`` forms that
      neither counter sees.

    A *middle* element being replaced without marking anything dirty would still
    be missed. That pattern does not occur in the codebase: the only
    ``state.vehicles[i] = ...`` sites (repair_operators) call
    ``mark_objective_dirty`` immediately afterwards.
    """
    try:
        count = len(vehicles)
    except TypeError:
        return (_CARGO_MUTATIONS, _SOLUTION_MUTATIONS, None, None)
    last_id = getattr(vehicles[-1], "id", None) if count else None
    return (_CARGO_MUTATIONS, _SOLUTION_MUTATIONS, count, last_id)


class CargoDict(dict):
    """
    A dict-like container that updates its parent Vehicle's _loaded_volume and capacity
    whenever items are added/removed/updated in the mapping.

    Notes:
      - copy() returns a built-in dict. Assigning it back to veh.cargo triggers
        the Vehicle.cargo setter so it is wrapped again.
      - This class is used only inside Vehicle instances to synchronize caches
        when cargo is modified in place.
    """
    def __init__(self, vehicle, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._veh = vehicle

    def _sku_size(self, sku):
        try:
            return float(self._veh.data.sku_sizes.get(sku, 0))
        except (AttributeError, KeyError, TypeError, ValueError):
            return 0.0

    def __setitem__(self, key, value):
        prev = self.get(key, 0)
        existed = key in self
        super().__setitem__(key, value)
        if not existed or value != prev:
            # Only a real change counts: assigning the value that is already there
            # must not invalidate the inventory cache.
            _note_cargo_mutation()
        try:
            sku, d = key
            delta = (value - prev) * self._sku_size(sku)
            self._veh._loaded_volume = getattr(self._veh, "_loaded_volume", 0.0) + delta
            # Keep capacity consistent and non-negative.
            self._veh.capacity = max(0, self._veh.max_capacity - getattr(self._veh, "_loaded_volume", 0.0))
        except (AttributeError, KeyError, TypeError, ValueError):
            # Best effort: leave vehicle state unchanged on error.
            pass

    def __delitem__(self, key):
        prev = self.get(key, 0)
        super().__delitem__(key)
        _note_cargo_mutation()
        try:
            sku, d = key
            delta = - prev * self._sku_size(sku)
            self._veh._loaded_volume = max(0, getattr(self._veh, "_loaded_volume", 0.0) + delta)
            self._veh.capacity = max(0, self._veh.max_capacity - getattr(self._veh, "_loaded_volume", 0.0))
        except (AttributeError, KeyError, TypeError, ValueError):
            pass

    def clear(self):
        if len(self):
            _note_cargo_mutation()
        super().clear()
        try:
            self._veh._loaded_volume = 0
            self._veh.capacity = self._veh.max_capacity
        except (AttributeError, KeyError, TypeError, ValueError):
            pass

    def update(self, *args, **kwargs):
        other = dict(*args, **kwargs)
        for k, v in other.items():
            self.__setitem__(k, v)

    def pop(self, key, default=None):
        prev = self.get(key, default)
        res = super().pop(key, default)
        _note_cargo_mutation()
        try:
            sku, d = key
            self._veh._loaded_volume = max(0, getattr(self._veh, "_loaded_volume", 0.0) - prev * self._sku_size(sku))
            self._veh.capacity = max(0, self._veh.max_capacity - getattr(self._veh, "_loaded_volume", 0.0))
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
        return res

    def popitem(self):
        k, v = super().popitem()
        _note_cargo_mutation()
        try:
            sku, d = k
            self._veh._loaded_volume = max(0, getattr(self._veh, "_loaded_volume", 0.0) - v * self._sku_size(sku))
            self._veh.capacity = max(0, self._veh.max_capacity - getattr(self._veh, "_loaded_volume", 0.0))
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
        return k, v

    def copy(self):
        # Return a plain dict to match the expectations of other code paths.
        return dict(self)


class SiktDict(dict):
    """
    ``(plant, sku, day) -> inventory`` mapping that counts changes made to itself.

    ``SolutionState.s_ikt`` is written from several places (inventory write-back,
    the destroy operators, the repair operators, and ``load``), and the inventory
    fast path has to know whether the array it derived from ``s_ikt`` is still
    valid. Rebuilding that array from the dict costs ~378 us on the medium
    instance (~48% of ``state_to_inventory_arrays``), so it is worth caching - but
    only if staleness can be detected cheaply.

    Counting mutations here makes that check O(1) and complete for in-place
    writes, which is how every observed call site modifies it
    (``s_ikt[key] = value``). A wholesale replacement is caught separately by the
    cached array's object identity check.

    Only a *real* change counts. Writing back the value that is already stored is
    extremely common - the inventory write-back assigns every entry of the
    recomputed array on every call - and counting those would invalidate the cache
    constantly for no reason: measured over a medium run, unconditional counting
    gave a 14% hit rate.

    Like ``CargoDict``, this tracks its own writes; a change made by bypassing the
    mapping (writing to an underlying view) would not be counted, and that pattern
    does not occur in the codebase.
    """

    __slots__ = ("revision",)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.revision = 0

    def _note(self, key, value) -> None:
        """Bump the revision only when ``key`` did not already hold ``value``."""
        if not dict.__contains__(self, key) or dict.__getitem__(self, key) != value:
            self.revision += 1

    def __setitem__(self, key, value):
        self._note(key, value)
        super().__setitem__(key, value)

    def __delitem__(self, key):
        super().__delitem__(key)
        self.revision += 1

    def clear(self):
        if len(self):
            self.revision += 1
        super().clear()

    def pop(self, key, default=None):
        existed = key in self
        result = super().pop(key, default)
        if existed:
            self.revision += 1
        return result

    def popitem(self):
        result = super().popitem()
        self.revision += 1
        return result

    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]

    def copy(self):
        # Preserve the type: callers that clone a state need a tracked mapping,
        # or later in-place writes would silently stop being counted.
        return SiktDict(self)


# =========================
# Data class: Vehicle
# =========================
@dataclass
class Vehicle:
    """
    Vehicle entity:
        fact_id: str
        dealer_id: str
        type: str
        day: int
        data: DataALNS

    Dynamic attributes (init=False):
        id, max_capacity, capacity, cargo (wrapped as CargoDict)
    """
    fact_id: str
    dealer_id: str
    type: str
    day: int
    data: DataALNS

    id: int = field(init=False, default=-1)
    max_capacity: int = field(init=False, default=0)
    capacity: int = field(init=False, default=0)
    # ``cargo`` is intentionally NOT a dataclass field: it is a property (see
    # below) so that every assignment is wrapped in ``CargoDict`` and keeps
    # ``_loaded_volume``/``capacity`` in sync.
    #
    # Declaring it as ``field(init=False, ...)`` did not work: the ``@property``
    # below shadows the class attribute, so ``@dataclass`` never saw
    # ``init=False`` and instead read the property object as the field default.
    # That produced ``__init__(..., cargo=<property object>)`` - a parameter a
    # caller could pass, whose value was then silently discarded (the generated
    # ``__init__`` skips fields whose default is a descriptor). Passing
    # ``cargo={...}`` therefore lost the payload without any error, and
    # constructing a ``Vehicle`` raised ``TypeError: 'property' object is not
    # iterable`` twice for every instance - the only exception path that actually
    # fired during a full search run.

    _id_counter = 0  # Class-level counter.

    def __post_init__(self):
        # Assign a unique id.
        self.id = Vehicle._id_counter
        Vehicle._id_counter += 1
        # Initialize capacity.
        try:
            self.max_capacity = int(self.data.veh_type_cap[self.type])
        except (KeyError, TypeError, ValueError):
            self.max_capacity = 0
        self.capacity = self.max_capacity
        # Initialize cargo as CargoDict so in-place changes update the cache.
        self._cargo = CargoDict(self, {})
        # Cache total loaded volume to avoid repeatedly scanning cargo.
        self._loaded_volume = 0

    # Cargo property: access as veh.cargo; assignments are wrapped and
    # synchronize _loaded_volume/capacity.
    @property
    def cargo(self):
        return self._cargo

    @cargo.setter
    def cargo(self, mapping):
        # mapping may be a plain dict or another mapping; wrap it in CargoDict
        # and synchronize the loaded volume. ``dict(None)`` would raise, so the
        # empty case is spelled out.
        new_map = CargoDict(self, {} if mapping is None else dict(mapping))
        # Compute loaded volume. A non-numeric quantity is a data error worth
        # skipping rather than aborting the assignment; anything else is a bug
        # and propagates.
        loaded = 0
        for (sku_k, d_k), q_k in new_map.items():
            try:
                loaded += q_k * float(self.data.sku_sizes.get(sku_k, 0))
            except (TypeError, ValueError):
                continue
        self._cargo = new_map
        self._loaded_volume = loaded
        self.capacity = max(0, self.max_capacity - loaded)

    def __eq__(self, other):
        # Use id-only equality for speed: id is unique per Vehicle instance.
        if not isinstance(other, Vehicle):
            return False
        return self.id == other.id

    def __hash__(self):
        # Hash based on unique id for faster hashing/comparisons.
        return hash(self.id)

    def __repr__(self):
        return (f"Vehicle(id={self.id}, fact_id={self.fact_id}, dealer_id={self.dealer_id}, "
                f"type={self.type}, day={self.day}, capacity={self.capacity}, cargo={self.cargo})")

    def get_avail_space(self) -> int:
        """
        Return the current remaining capacity in volume units.
        """
        return self.capacity

    def is_full(self) -> bool:
        """
        Return True when the vehicle has no remaining capacity.
        """
        return self.capacity <= 0

    def load(self, sku_id: str, num: int) -> int:
        """
        Attempt to load a quantity of the specified SKU using an atomic,
        defensive implementation:
            - Compute locally first and commit only when consistency is
              guaranteed.
            - Raise an exception when the SKU is unknown.
            - Return the quantity that could not be loaded (num - loaded).
        """
        if num <= 0:
            return num
        if sku_id not in self.data.sku_sizes:
            raise ValueError(f"SKU {sku_id} not found in data.sku_sizes")
        sku_size = float(self.data.sku_sizes[sku_id])

        # Compute the loadable quantity from remaining capacity and keep
        # integer-division semantics explicit.
        max_by_cap = 0
        if sku_size > 0:
            max_by_cap = int(self.capacity // sku_size)
        else:
            # Defensive handling: sku_size should never be zero. If it is,
            # block loading to avoid infinite loops or invalid state.
            raise ValueError(f"Invalid sku_size=0 for SKU {sku_id}")

        num_loaded = min(num, max_by_cap)

        if num_loaded <= 0:
            return num  # No quantity can be loaded.

        # Compute the new state in local variables for an atomic commit.
        prev_qty = self._cargo.get((sku_id, self.day), 0)
        new_qty = prev_qty + num_loaded
        new_capacity = self.capacity - num_loaded * sku_size
        new_loaded_volume = getattr(self, "_loaded_volume", 0.0) + num_loaded * sku_size

        # Defensive check: negative capacity or inconsistent loaded volume
        # must never occur.
        if new_capacity < 0 or new_loaded_volume < 0:
            # Guard: report the inconsistency without committing changes.
            raise RuntimeError(f"Load would cause negative capacity or loaded volume (sku_id={sku_id}, load={num_loaded}, new_capacity={new_capacity})")

        # Commit atomically through CargoDict.__setitem__ to keep caches
        # consistent.
        self._cargo[(sku_id, self.day)] = new_qty
        # __setitem__ updates _loaded_volume and capacity; assign again to
        # guarantee consistency.
        self._loaded_volume = new_loaded_volume
        self.capacity = new_capacity

        return num - num_loaded  # Return the unloaded quantity.

    def clear(self):
        """
        Reset the vehicle by clearing cargo and restoring remaining capacity.
        """
        try:
            # Use CargoDict.clear to keep caches consistent.
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
        Return True when no cargo has been loaded.
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
