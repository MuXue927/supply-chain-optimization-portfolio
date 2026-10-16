"""
inventory_numba.py

Helper module providing array-based and optional Numba-accelerated
implementations of compute_inventory.

Goals:
 - Provide stable index mappings (plant/sku/day -> integer index).
 - Convert state/data into ndarray representations (s_ikt, shipments, prod).
 - Provide an optional Numba njit kernel when available and a NumPy fallback.
 - Provide a safe wrapper that returns False on failure so
   state.compute_inventory() can use the pure-Python fallback.

Notes:
 - The implementation is deliberately conservative: all changes are written to
   state.s_ikt (ending inventory), and failures fall back to the original path.
 - Deeper integration of this kernel (for example advisory validation or
   candidate evaluation in regret operators) requires callers to maintain
   consistency for all other state caches and variables, such as
   shipped_cache.
"""

import hashlib
import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Try to import numba; if not available we fall back to a numpy-only implementation.
try:
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False


@dataclass
class IndexMaps:
    plant_to_idx: dict[str, int]
    sku_to_idx: dict[str, int]
    idx_to_plant: list[str]
    idx_to_sku: list[str]
    horizons: int
    n_plants: int
    n_skus: int


def build_index_maps(data) -> IndexMaps:
    """
    Build stable integer indices for plants and skus based on DataALNS.
    - data.plants: iterable of plant ids
    - data.skus_plant: mapping plant -> iterable of sku ids (used to collect sku universe)
    - data.horizons: planning horizon (int)
    """
    # Collect plants in deterministic order
    plants = list(getattr(data, "plants", []))
    plant_to_idx = {p: i for i, p in enumerate(plants)}

    # Collect sku universe from data.skus_plant (fall back to data.skus if present)
    skus_set = set()
    sp = getattr(data, "skus_plant", None)
    if sp:
        for p, skus in sp.items():
            for s in skus:
                skus_set.add(s)
    else:
        skus_all = getattr(data, "skus", None)
        if skus_all:
            skus_set.update(list(skus_all))

    idx_to_sku = sorted(list(skus_set))
    sku_to_idx = {s: i for i, s in enumerate(idx_to_sku)}

    horizons = int(getattr(data, "horizons", 0))
    return IndexMaps(
        plant_to_idx=plant_to_idx,
        sku_to_idx=sku_to_idx,
        idx_to_plant=plants,
        idx_to_sku=idx_to_sku,
        horizons=horizons,
        n_plants=len(plants),
        n_skus=len(idx_to_sku),
    )


def state_to_inventory_arrays(state, maps: IndexMaps):
    """
    Convert state.s_ikt, state.vehicles (cargo) and data.sku_prod_each_day into numpy arrays:
      - s_ikt_arr: shape (P, S, H+1)  (H+1 to include day 0 baseline)
      - shipments_arr: same shape, shipments_arr[p,s,d] = total shipped (used) at day d by vehicles
      - prod_arr: same shape, production at day d
    Returns (s_ikt_arr, shipments_arr, prod_arr)
    """
    P = maps.n_plants
    S = maps.n_skus
    H = maps.horizons
    # use float64 for numeric safety
    s_ikt_arr = np.zeros((P, S, H + 1), dtype=np.float64)
    shipments_arr = np.zeros((P, S, H + 1), dtype=np.float64)
    prod_arr = np.zeros((P, S, H + 1), dtype=np.float64)

    # fill s_ikt from state.s_ikt: keys are (plant, sku, day)
    s_ikt = getattr(state, "s_ikt", {})
    for (plant, sku, day), val in s_ikt.items():
        pi = maps.plant_to_idx.get(plant)
        si = maps.sku_to_idx.get(sku)
        if pi is None or si is None:
            continue
        if 0 <= day <= H:
            s_ikt_arr[pi, si, int(day)] = float(val)

    # fill production array from state.data.sku_prod_each_day (or data)
    data = getattr(state, "data", None)
    if data is not None:
        sku_prod = getattr(data, "sku_prod_each_day", {})
        for (plant, sku, day), val in sku_prod.items():
            pi = maps.plant_to_idx.get(plant)
            si = maps.sku_to_idx.get(sku)
            if pi is None or si is None:
                continue
            if 0 <= day <= H:
                prod_arr[pi, si, int(day)] = float(val)

    # Accumulate same-day shipments exactly as state.compute_inventory() does.
    shipments_arr = _build_shipments_from_vehicles(state, maps, P, S, H)

    return s_ikt_arr, shipments_arr, prod_arr


if NUMBA_AVAILABLE:

    @njit
    def _compute_inventory_core_numba(s_ikt_arr, shipments_arr, prod_arr, H):
        P, S, _ = s_ikt_arr.shape
        for p in range(P):
            for s in range(S):
                for d in range(1, H + 1):
                    prev = s_ikt_arr[p, s, d - 1]
                    prod = prod_arr[p, s, d]
                    used = shipments_arr[p, s, d]
                    s_ikt_arr[p, s, d] = prev + prod - used
        return s_ikt_arr
# Ensure a numpy fallback implementation is always exposed, even when Numba is available.
# This makes the `_compute_inventory_core_numpy` symbol available for tests and as a
# robust fallback path that can be called explicitly.
def _compute_inventory_core_numpy(s_ikt_arr, shipments_arr, prod_arr, H):
    # vectorized per-day update (operates inplace on s_ikt_arr)
    for d in range(1, H + 1):
        prev = s_ikt_arr[:, :, d - 1]
        prod = prod_arr[:, :, d]
        used = shipments_arr[:, :, d]
        s_ikt_arr[:, :, d] = prev + prod - used
    return s_ikt_arr


def arrays_to_state_s_ikt(state, maps: IndexMaps, s_ikt_arr: np.ndarray):
    """
    Write back the s_ikt ndarray values into state.s_ikt (dict keyed by (plant, sku, day)).

    The reference recursion updates future periods for pairs in ``state.s_indices``
    and leaves all other inventory entries untouched.  The same rule is applied
    here so the array path does not create zero-valued keys for unrelated
    plant/SKU pairs.  Values are stored as plain integers because ``s_ikt`` is an
    integer inventory model; retaining float values would violate that invariant
    in equality checks and batch-delta updates.
    """
    H = int(maps.horizons)
    # ensure state.s_ikt exists
    if not hasattr(state, "s_ikt") or getattr(state, "s_ikt") is None:
        try:
            state.s_ikt = {}
        except (TypeError, ValueError):
            # if cannot set attribute, try to modify existing mapping if it exists
            pass
    s_dict = getattr(state, "s_ikt", {})

    state_indices = getattr(state, "s_indices", None)
    if state_indices is None:
        valid_pairs = {
            (plant, sku)
            for plant in maps.idx_to_plant
            for sku in maps.idx_to_sku
        }
    else:
        valid_pairs = {
            (plant, sku)
            for plant, sku, _day in state_indices
            if plant in maps.plant_to_idx and sku in maps.sku_to_idx
        }

    for plant, sku in valid_pairs:
        pi = maps.plant_to_idx[plant]
        si = maps.sku_to_idx[sku]

        # The reference recursion writes only periods 1..H.  Preserve an
        # existing day-0 opening balance without inventing a new zero key.
        day_zero_key = (plant, sku, 0)
        if day_zero_key in s_dict:
            s_dict[day_zero_key] = int(s_ikt_arr[pi, si, 0])
        for day in range(1, H + 1):
            s_dict[(plant, sku, day)] = int(s_ikt_arr[pi, si, day])

    # try storing back
    try:
        state.s_ikt = s_dict
    except (TypeError, ValueError):
        pass
    return


def _build_shipments_from_vehicles(state, maps: IndexMaps, P: int, S: int, H: int):
    """
    Helper to construct shipments array from vehicles. Factored out to reuse.
    """
    shipments_arr = np.zeros((P, S, H + 1), dtype=np.float64)
    for veh in getattr(state, "vehicles", []):
        plant = getattr(veh, "fact_id", None)
        if plant is None:
            continue
        pi = maps.plant_to_idx.get(plant)
        if pi is None:
            continue
        cargo = getattr(veh, "cargo", {})
        veh_day = getattr(veh, "day", None)
        for (sku_k, d), q in cargo.items():
            # The reference computes same-day shipments only.
            if veh_day is not None and d != veh_day:
                continue
            si = maps.sku_to_idx.get(sku_k)
            if si is None:
                continue
            if 0 <= d <= H:
                try:
                    shipments_arr[pi, si, int(d)] += float(q)
                except Exception:
                    continue
    return shipments_arr


def _vehicle_signature(veh):
    """
    Stable, deterministic signature of a vehicle's cargo, used to detect changes.

    MD5 over sorted ``"sku|day|qty"`` items, deliberately not ``hash()``, which
    varies between processes (PYTHONHASHSEED) and would make the cached shipments
    array look stale on every run.

    The signature covers the *number* of cargo entries and the total quantity as
    well as the per-entry text, so a changed quantity that is not representable
    as a float still invalidates the cache. The previous inline version dropped
    those from the fallback signature and silently skipped unparseable
    quantities, so the cache could be reused for a genuinely different load.
    """
    fact = getattr(veh, "fact_id", None)
    vehicle_type = getattr(veh, "type", None)
    day = getattr(veh, "day", None)
    try:
        cargo = getattr(veh, "cargo", {}) or {}
        items_source = list(cargo.items())
    except Exception:
        # A vehicle whose cargo cannot be read still gets a signature; it just
        # cannot describe the load. Treating it as "no entries" keeps the rest of
        # the list cacheable instead of disabling the cache for every vehicle.
        # Deliberately broad: reading a property is caller-supplied code, and the
        # only consequence of getting this wrong is a needless cache rebuild,
        # because the enclosing computation has its own pure-Python fallback.
        cargo = {}
        items_source = []

    total_qty = 0.0
    items = []
    for (sku_k, cargo_day), qty in items_source:
        try:
            total_qty += float(qty)
        except (TypeError, ValueError):
            # Keep the entry itself in the signature (below); only the numeric
            # total cannot account for it.
            pass
        items.append(f"{sku_k}|{cargo_day}|{qty!r}")
    items.sort()

    digest = hashlib.md5()
    digest.update("|".join(items).encode("utf-8"))
    return (fact, vehicle_type, day, len(cargo), total_qty, digest.hexdigest())


def _vehicles_signature(vehicles):
    """
    Signature of a whole vehicle list, or None when it cannot be computed.

    None means "cannot tell whether the cached shipments still apply", so the
    caller rebuilds them instead of reusing a possibly stale array. Keeping this
    fallback here rather than around the cache decision keeps the fallback
    explicit: the failure degrades the incremental cache, never correctness.
    """
    try:
        return tuple(_vehicle_signature(veh) for veh in vehicles)
    except (AttributeError, TypeError, ValueError, KeyError):
        return None


def _resolve_inventory_cache(state, maps, P, S, H):
    """
    Return ``(cache, prod_arr, baseline0, shipments_arr)`` for this state.

    Reuses the cached arrays when they belong to ``maps`` and have the expected
    shapes; otherwise rebuilds them from ``state``. The rebuilt cache is attached
    to the state best-effort - failing to attach only costs the next call the
    rebuild, so it must not fail the computation itself.

    This is preparation, not the fast path: a failure here propagates to the
    caller's handler, which falls back to the pure-Python implementation.
    """
    cache = getattr(state, "_inventory_cache", None)
    prod_arr = baseline0 = shipments_arr = None
    cache_valid = False

    if cache:
        try:
            cache_valid = cache.get("maps", None) is maps
        except (AttributeError, TypeError):
            cache_valid = False

    if cache_valid:
        prod_arr = cache.get("prod_arr")
        baseline0 = cache.get("baseline0")
        shipments_arr = cache.get("shipments_arr")
        # Shapes must match the current maps, or the arrays describe another model.
        if (
            prod_arr is None
            or baseline0 is None
            or prod_arr.shape != (P, S, H + 1)
            or baseline0.shape != (P, S)
            or (shipments_arr is not None and shipments_arr.shape != (P, S, H + 1))
        ):
            cache_valid = False

    if not cache_valid:
        # Build fresh arrays; only the day-0 baseline and production are retained.
        s_full, _, prod_full = state_to_inventory_arrays(state, maps)
        baseline0 = s_full[:, :, 0].copy()
        prod_arr = prod_full
        cache = {"maps": maps, "prod_arr": prod_arr, "baseline0": baseline0}
        try:
            setattr(state, "_inventory_cache", cache)
        except (AttributeError, TypeError):
            # Best effort: without the persistent cache the next call rebuilds.
            pass

    return cache, prod_arr, baseline0, shipments_arr


def compute_inventory_numba_safe(state):
    """
    Enhanced safe wrapper with conservative caching of baseline arrays and shipments incremental cache.

    Strategy (safe and conservative):
      - Cache index maps on state._inventory_maps (existing).
      - Cache production array and day-0 baseline s_ikt slice on state._inventory_cache to avoid
        reconstructing them on each call.
      - Cache shipments_arr and a lightweight vehicles signature to avoid rebuilding shipments
        if vehicles did not change since last compute.
      - Rebuild shipments_arr when signature differs; otherwise reuse cached shipments_arr.
      - Create working s_ikt_arr with cached baseline in [:,:,0] and run core to compute forward days.
      - Write back results and mark objective dirty as before.
      - On any unexpected error, return False so the caller can use the
        pure-Python compute_inventory() implementation.
      - Return True only after a successful write-back.
    """
    try:
        data = getattr(state, "data", None)
        if data is None:
            return state.compute_inventory()

        # build or reuse maps
        maps = getattr(state, "_inventory_maps", None)
        if maps is None:
            maps = build_index_maps(data)
            try:
                setattr(state, "_inventory_maps", maps)
            except Exception:
                pass

        P = maps.n_plants
        S = maps.n_skus
        H = maps.horizons

        # Resolve the reusable arrays (maps, production, day-0 baseline, and any
        # cached shipments). Extracted so this try body stays small and the cache
        # rules are testable on their own.
        cache, prod_arr, baseline0, shipments_arr = _resolve_inventory_cache(
            state, maps, P, S, H
        )

        # compute a lightweight vehicles signature to detect changes (see the
        # module-level helper; kept out of this try so the body stays small)
        vehicles = getattr(state, "vehicles", [])
        current_vehicles_sig = _vehicles_signature(vehicles)

        use_cached_shipments = False
        if cache and current_vehicles_sig is not None:
            cached_sig = cache.get("vehicles_sig")
            cached_shipments = cache.get("shipments_arr")
            if (
                cached_sig is not None
                and cached_shipments is not None
                and cached_sig == current_vehicles_sig
            ):
                # The signature matches, so the cached array describes this load
                # exactly; reuse it. The previous try/except around this
                # assignment could not fire (assigning a value raises nothing) and
                # only obscured that.
                shipments_arr = cached_shipments
                use_cached_shipments = True

        if not use_cached_shipments:
            # rebuild shipments array each call (vehicles change frequently)
            shipments_arr = _build_shipments_from_vehicles(state, maps, P, S, H)
            # store into cache for future reuse
            if cache is None:
                cache = {"maps": maps, "prod_arr": prod_arr, "baseline0": baseline0}
            cache["shipments_arr"] = shipments_arr
            cache["vehicles_sig"] = current_vehicles_sig
            try:
                setattr(state, "_inventory_cache", cache)
            except Exception:
                pass

        # prepare working s_ikt_arr using cached baseline0
        s_ikt_arr = np.zeros((P, S, H + 1), dtype=np.float64)
        s_ikt_arr[:, :, 0] = baseline0

        # run fast core
        if NUMBA_AVAILABLE:
            try:
                _compute_inventory_core_numba(s_ikt_arr, shipments_arr, prod_arr, H)
            except Exception:
                _compute_inventory_core_numpy(s_ikt_arr, shipments_arr, prod_arr, H)
        else:
            _compute_inventory_core_numpy(s_ikt_arr, shipments_arr, prod_arr, H)

        # write back s_ikt to state
        arrays_to_state_s_ikt(state, maps, s_ikt_arr)
        try:
            state.s_initialized = True
        except (TypeError, ValueError):
            pass

        # Best-effort: invalidate any derived caches that depend on s_ikt
        try:
            if hasattr(state, "mark_objective_dirty"):
                state.mark_objective_dirty()
        except Exception:
            pass

        return True
    except Exception as exc:
        # Let the caller (alnsopt.compute_inventory) decide the fallback path to
        # avoid recursively calling this wrapper.
        #
        # The fast path stores a reusable cache on the state *before* it writes
        # the authoritative ``state.s_ikt`` back (see the ``setattr(state,
        # "_inventory_cache", ...)`` calls above), so a failure in between leaves
        # a cache that no completed inventory run produced. Drop it rather than
        # let a later call treat it as a verified baseline, and record the
        # failure: the caller only sees ``False`` and would otherwise fall back
        # with no trace of why the fast path was abandoned.
        try:
            state._inventory_cache = None
        except (TypeError, ValueError):
            pass
        logger.warning(
            "numba inventory fast path failed; falling back to the Python "
            "baseline: %s",
            exc,
            exc_info=True,
        )
        return False


# Expose a convenience name for callers to use
compute_inventory_fast = compute_inventory_numba_safe

# End of file
