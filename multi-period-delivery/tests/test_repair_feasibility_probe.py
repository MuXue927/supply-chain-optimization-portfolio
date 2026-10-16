"""Tests for the post-repair feasibility probe.

``_check_inserted_feasibility`` was extracted from a ~70-line ``try`` inside
``post_repair``. That try also swallowed every error from the inventory work that
followed it, so a real defect there was invisible; the extracted function keeps the
same conservative policy but its failure behaviour is now explicit and testable.

Policy under test:

* an empty insertion list is trivially feasible;
* a locally detected overload reports ``veh_over_load`` with the offending ids;
* a vehicle the probe cannot measure makes it inconclusive, so the real validator
  is consulted instead;
* if the validator also raises, the result is ``(False, {"validation_error": ...})``
  - reporting "feasible" let an unverified state through and disabled rollback.
"""

from __future__ import annotations

from types import SimpleNamespace

from multi_period_delivery.alns.repair_utils import _check_inserted_feasibility


class _State:
    """State stand-in for the feasibility probe."""

    def __init__(self, *, sku_sizes=None, veh_type_cap=None, validator=None) -> None:
        self.data = SimpleNamespace(
            sku_sizes=sku_sizes if sku_sizes is not None else {"S1": 2.0},
            veh_type_cap=veh_type_cap if veh_type_cap is not None else {"T1": 10.0},
        )
        self._validator = validator

    def suppress_inventory_ctx(self):
        class _Ctx:
            def __enter__(self_inner):
                return None

            def __exit__(self_inner, *exc):
                return False

        return _Ctx()

    def validate(self):
        if self._validator is not None:
            return self._validator()
        return True, {}


def _vehicle(*, veh_id=1, veh_type="T1", day=1, cargo=None):
    return SimpleNamespace(
        id=veh_id,
        type=veh_type,
        day=day,
        cargo=cargo if cargo is not None else {},
    )


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_no_inserted_vehicles_is_feasible() -> None:
    assert _check_inserted_feasibility(_State(), []) == (True, {})


def test_within_capacity_is_feasible() -> None:
    state = _State()
    vehicles = [_vehicle(cargo={("S1", 1): 3})]  # 3 x 2.0 = 6.0 <= 10.0

    assert _check_inserted_feasibility(state, vehicles) == (True, {})


def test_cargo_on_another_day_is_ignored() -> None:
    """Only the vehicle's own day counts toward its load."""
    state = _State()
    vehicles = [_vehicle(day=1, cargo={("S1", 2): 99})]

    assert _check_inserted_feasibility(state, vehicles) == (True, {})


# ---------------------------------------------------------------------------
# Overload
# ---------------------------------------------------------------------------


def test_overload_is_reported_with_the_vehicle_id() -> None:
    state = _State()
    vehicles = [_vehicle(veh_id=7, cargo={("S1", 1): 6})]  # 12.0 > 10.0

    feasible, violations = _check_inserted_feasibility(state, vehicles)

    assert feasible is False
    assert violations == {"veh_over_load": [7]}


def test_only_the_overloaded_vehicles_are_listed() -> None:
    state = _State()
    vehicles = [
        _vehicle(veh_id=1, cargo={("S1", 1): 2}),  # 4.0 fine
        _vehicle(veh_id=2, cargo={("S1", 1): 9}),  # 18.0 overloaded
    ]

    feasible, violations = _check_inserted_feasibility(state, vehicles)

    assert feasible is False
    assert violations == {"veh_over_load": [2]}


# ---------------------------------------------------------------------------
# Inconclusive -> validator
# ---------------------------------------------------------------------------


def test_unmeasurable_vehicle_falls_back_to_the_validator() -> None:
    calls: list[int] = []

    def validator():
        calls.append(1)
        return False, {"negative_inventory": ["P1"]}

    state = _State(validator=validator)
    # cargo is not iterable as key/value pairs -> probe raises.
    vehicles = [SimpleNamespace(id=1, type="T1", day=1, cargo=5)]

    feasible, violations = _check_inserted_feasibility(state, vehicles)

    assert calls == [1], "the validator must be consulted"
    assert feasible is False
    assert violations == {"negative_inventory": ["P1"]}


def test_validator_failure_is_reported_as_inconclusive_not_feasible() -> None:
    def validator():
        raise AttributeError("state.data is None")

    state = _State(validator=validator)
    vehicles = [SimpleNamespace(id=1, type="T1", day=1, cargo=5)]

    feasible, violations = _check_inserted_feasibility(state, vehicles)

    assert feasible is False, "an unverified state must not be called feasible"
    assert violations == {"validation_error": ["inconclusive"]}


# ---------------------------------------------------------------------------
# Malformed but iterable input
# ---------------------------------------------------------------------------


def test_unmeasurable_single_vehicle_marks_probe_inconclusive() -> None:
    """A vehicle whose type is unknown yields cap 0, so any load overloads it."""
    state = _State(veh_type_cap={})
    vehicles = [_vehicle(veh_id=3, cargo={("S1", 1): 1})]

    feasible, violations = _check_inserted_feasibility(state, vehicles)

    assert feasible is False
    assert violations["veh_over_load"] == [3]
