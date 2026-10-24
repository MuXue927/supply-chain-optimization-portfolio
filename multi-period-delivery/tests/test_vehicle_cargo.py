"""Tests for the ``Vehicle`` cargo contract.

Regression for a dataclass/property collision: ``cargo`` was declared as
``field(init=False, ...)`` *and* as a ``@property``. The property shadows the
class attribute, so ``@dataclass`` never saw ``init=False``; it read the
property object as the field default and generated

    Vehicle.__init__(..., cargo=<property object>)

A caller could therefore pass ``cargo={...}`` and have it silently discarded
(the generated initialiser skips fields whose default is a descriptor). The same
collision also re-raised ``TypeError: 'property' object is not iterable`` on
every construction, which was the only exception path that actually fired during
a full search run.

The field is gone; ``cargo`` is only the property.
"""

from __future__ import annotations

import dataclasses
import inspect

import pytest

from multi_period_delivery.alns.vehicle import CargoDict, Vehicle


class _Data:
    """Minimal stand-in exposing the two attributes Vehicle reads."""

    def __init__(self) -> None:
        self.sku_sizes = {"S1": 2.0, "S2": 3.0}
        self.veh_type_cap = {"T1": 100}


def _vehicle() -> Vehicle:
    return Vehicle("P1", "D1", "T1", 1, _Data())


# ---------------------------------------------------------------------------
# The collision must not come back
# ---------------------------------------------------------------------------


def test_cargo_is_not_a_dataclass_field() -> None:
    names = {field.name for field in dataclasses.fields(Vehicle)}

    assert "cargo" not in names
    assert "fact_id" in names


def test_constructor_does_not_accept_cargo() -> None:
    """A parameter whose value is silently dropped must not exist."""
    assert "cargo" not in inspect.signature(Vehicle.__init__).parameters


def test_passing_cargo_is_rejected_not_ignored() -> None:
    with pytest.raises(TypeError):
        Vehicle("P1", "D1", "T1", 1, _Data(), cargo={("S1", 1): 5})


# ---------------------------------------------------------------------------
# The property still behaves
# ---------------------------------------------------------------------------


def test_construction_initialises_an_empty_cargo_dict() -> None:
    vehicle = _vehicle()

    assert isinstance(vehicle.cargo, CargoDict)
    assert dict(vehicle.cargo) == {}
    assert vehicle._loaded_volume == 0
    assert vehicle.capacity == vehicle.max_capacity == 100


def test_direct_cargo_dict_construction_synchronises_volume() -> None:
    vehicle = _vehicle()

    cargo = CargoDict(vehicle, {("S1", 1): 4})

    assert dict(cargo) == {("S1", 1): 4}
    assert vehicle._loaded_volume == pytest.approx(8.0)
    assert vehicle.capacity == pytest.approx(92.0)


def test_assignment_wraps_and_synchronises_volume_and_capacity() -> None:
    vehicle = _vehicle()

    vehicle.cargo = {("S1", 1): 5}

    assert isinstance(vehicle.cargo, CargoDict)
    assert vehicle._loaded_volume == pytest.approx(10.0)  # 5 x 2.0
    assert vehicle.capacity == pytest.approx(90.0)


def test_assignment_accepts_none() -> None:
    vehicle = _vehicle()
    vehicle.cargo = {("S1", 1): 5}

    vehicle.cargo = None

    assert dict(vehicle.cargo) == {}
    assert vehicle._loaded_volume == 0
    assert vehicle.capacity == vehicle.max_capacity


def test_assignment_accepts_another_cargo_dict() -> None:
    source = _vehicle()
    source.cargo = {("S2", 1): 4}

    target = _vehicle()
    target.cargo = source.cargo

    assert dict(target.cargo) == {("S2", 1): 4}
    assert target._loaded_volume == pytest.approx(12.0)  # 4 x 3.0


def test_in_place_mutation_keeps_the_cache_in_sync() -> None:
    vehicle = _vehicle()

    vehicle.cargo[("S1", 1)] = 3
    assert vehicle._loaded_volume == pytest.approx(6.0)

    del vehicle.cargo[("S1", 1)]
    assert vehicle._loaded_volume == pytest.approx(0.0)
    assert vehicle.capacity == vehicle.max_capacity


def test_non_numeric_quantity_is_skipped_not_fatal() -> None:
    """A malformed quantity is data, so it is skipped; the assignment stands."""
    vehicle = _vehicle()

    vehicle.cargo = {("S1", 1): "not-a-number", ("S2", 1): 2}

    assert vehicle._loaded_volume == pytest.approx(6.0)  # only the valid entry


def test_setdefault_keeps_volume_and_cache_synchronised() -> None:
    vehicle = _vehicle()
    vehicle.cargo[("S1", 1)] = 2

    assert vehicle.cargo.setdefault(("S2", 1), 4) == 4

    assert dict(vehicle.cargo) == {("S1", 1): 2, ("S2", 1): 4}
    assert vehicle._loaded_volume == pytest.approx(16.0)  # 2*2 + 4*3
    assert vehicle.capacity == pytest.approx(84.0)


def test_ior_keeps_volume_and_cache_synchronised() -> None:
    vehicle = _vehicle()
    vehicle.cargo[("S1", 1)] = 2

    vehicle.cargo |= {("S2", 1): 4}

    assert vehicle._loaded_volume == pytest.approx(16.0)
    assert vehicle.capacity == pytest.approx(84.0)


def test_pop_missing_key_with_a_valid_default_does_not_change_cargo_state() -> None:
    """A missing key must not be treated as a removed load.

    ``pop(missing, default)`` used to subtract ``default * sku_size`` from the
    vehicle even though nothing was removed, silently corrupting both the
    loaded-volume cache and the remaining capacity.
    """
    vehicle = _vehicle()
    vehicle.cargo[("S1", 1)] = 3
    before_loaded = vehicle._loaded_volume
    before_capacity = vehicle.capacity

    assert vehicle.cargo.pop(("S2", 1), 5) == 5

    assert vehicle._loaded_volume == before_loaded
    assert vehicle.capacity == before_capacity


def test_pop_without_a_default_still_raises_key_error() -> None:
    vehicle = _vehicle()

    with pytest.raises(KeyError):
        vehicle.cargo.pop(("S1", 1))
