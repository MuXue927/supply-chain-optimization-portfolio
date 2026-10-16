"""Tests for ``SiktDict``, the change-counting inventory mapping.

``SolutionState.s_ikt`` is written from several places, and the inventory fast path
caches the ndarray derived from it. That cache is only sound if every in-place
write is *counted*, and it is only useful if unchanged writes are *not* counted -
both directions matter:

* under-counting -> a stale array looks fresh (silent wrong inventory);
* over-counting -> the cache never hits. Measured on the medium instance,
  unconditional counting gave a 14% hit rate because the inventory write-back
  re-assigns every entry on every call; counting only real changes raised it to
  18% and the end-to-end gain became measurable.

The array cache itself is covered in ``test_inventory_cache_helpers.py``.
"""

from __future__ import annotations

from multi_period_delivery.alns.vehicle import SiktDict


def _tracked(mapping=None):
    return SiktDict(mapping or {})


# ---------------------------------------------------------------------------
# It is still a dict
# ---------------------------------------------------------------------------


def test_it_behaves_like_a_dict() -> None:
    tracked = _tracked({("P", "S", 1): 5})

    assert tracked[("P", "S", 1)] == 5
    assert len(tracked) == 1
    assert dict(tracked) == {("P", "S", 1): 5}
    assert ("P", "S", 1) in tracked
    assert isinstance(tracked, dict)


def test_solution_state_actually_uses_it() -> None:
    """If the state ever went back to a plain dict, the cache would never hit."""
    import tempfile
    from pathlib import Path

    from multi_period_delivery.alns.alns_optimizer import ALNSOptimizer

    base = Path(__file__).resolve().parents[1]
    optimizer = ALNSOptimizer(
        input_file_loc=str(base / "datasets" / "multiple-periods" / "small"),
        output_file_loc=str(Path(tempfile.gettempdir()) / "_sikt_dict_probe"),
    )
    optimizer.load_data("dataset_1")
    state = optimizer.create_initial_solution()

    assert isinstance(state.s_ikt, SiktDict)
    # And a state clone must stay tracked, or later writes would go uncounted.
    assert isinstance(state.copy(clone_vehicles=False).s_ikt, SiktDict)


# ---------------------------------------------------------------------------
# Real changes are counted
# ---------------------------------------------------------------------------


def test_new_key_bumps_the_revision() -> None:
    tracked = _tracked()
    before = tracked.revision

    tracked[("P", "S", 1)] = 5

    assert tracked.revision == before + 1


def test_changed_value_bumps_the_revision() -> None:
    tracked = _tracked({("P", "S", 1): 5})
    before = tracked.revision

    tracked[("P", "S", 1)] = 9

    assert tracked.revision == before + 1


def test_deletion_bumps_the_revision() -> None:
    tracked = _tracked({("P", "S", 1): 5})
    before = tracked.revision

    del tracked[("P", "S", 1)]

    assert tracked.revision == before + 1


def test_clear_bumps_the_revision_when_non_empty() -> None:
    tracked = _tracked({("P", "S", 1): 5})
    before = tracked.revision

    tracked.clear()

    assert tracked.revision == before + 1


def test_pop_bumps_the_revision_for_a_present_key() -> None:
    tracked = _tracked({("P", "S", 1): 5})
    before = tracked.revision

    tracked.pop(("P", "S", 1))

    assert tracked.revision == before + 1


def test_update_counts_each_real_change() -> None:
    tracked = _tracked({("P", "S", 1): 5})
    before = tracked.revision

    tracked.update({("P", "S", 1): 5, ("P", "S", 2): 1})

    # One entry unchanged, one new.
    assert tracked.revision == before + 1


def test_setdefault_counts_only_a_new_key() -> None:
    tracked = _tracked({("P", "S", 1): 5})
    before = tracked.revision

    assert tracked.setdefault(("P", "S", 1), 9) == 5
    assert tracked.revision == before, "an existing key must not count"

    assert tracked.setdefault(("P", "S", 2), 9) == 9
    assert tracked.revision == before + 1


# ---------------------------------------------------------------------------
# Unchanged writes must NOT be counted
# ---------------------------------------------------------------------------


def test_rewriting_the_same_value_does_not_bump() -> None:
    """This is the case the inventory write-back hits on every single call."""
    tracked = _tracked({("P", "S", 1): 5})
    before = tracked.revision

    for _ in range(50):
        tracked[("P", "S", 1)] = 5

    assert tracked.revision == before


def test_clearing_an_empty_mapping_does_not_bump() -> None:
    tracked = _tracked()
    before = tracked.revision

    tracked.clear()

    assert tracked.revision == before


def test_popping_a_missing_key_does_not_bump() -> None:
    tracked = _tracked()
    before = tracked.revision

    assert tracked.pop(("P", "S", 1), None) is None

    assert tracked.revision == before


# ---------------------------------------------------------------------------
# copy() must preserve tracking
# ---------------------------------------------------------------------------


def test_copy_returns_a_tracked_mapping() -> None:
    """A state clone must keep counting, or its later writes would go unnoticed."""
    original = _tracked({("P", "S", 1): 5})

    clone = original.copy()

    assert isinstance(clone, SiktDict)
    assert dict(clone) == dict(original)
    before = clone.revision
    clone[("P", "S", 2)] = 1
    assert clone.revision == before + 1
    # And the original is untouched.
    assert original.revision == 0
