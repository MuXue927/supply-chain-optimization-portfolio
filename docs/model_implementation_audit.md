# Model vs. implementation audit

> Scope: the single-period and multi-period **monolithic** models
> (`monolithic/`, `cplex_api_monolithic/`) and their input-data loaders, checked
> against the English problem descriptions in
> `single-period-delivery/docs/problem_description.md` and
> `multi-period-delivery/docs/problem_description.md`.
>
> Date: 2026-10-13. Environment: the archived code as pushed to `main`, with the
> bundled `dataset_1` instances. No solver licence is required for anything
> reported here: every number below comes from constructing the *index sets*,
> not from solving.

---

## 1. Summary

| # | Finding | Severity | Affects |
| ---: | --- | --- | --- |
| 1 | The fleet's **vehicle-type mix is drawn at random and never seeded** in the Gurobi monolithic path, so the model differs on every run | **High** | both projects, Gurobi path |
| 2 | The description assumes **unlimited vehicle availability**, but the code caps the fleet at a pre-estimated size and *fixes* each vehicle's type | **High** | both projects |
| 3 | The "upper bound" on the fleet size uses `floor` instead of `ceil` | Medium | both projects |
| 4 | Because no constraint links a vehicle across periods, the cap means "*L* vehicles **per period**", not "*L* vehicles in total" | Medium | multi-period |
| 5 | The `x` variable is indexed by individual vehicle in a **global** fleet; a route-local reformulation was measured at **14.8×** smaller (single) | Medium | both projects |
| 6 | The thesis' demand constraint is an **equality**; the code uses `>=` | Low (documented) | both projects |

Everything else that was checked lines up; see §8.

> **Status table (2026-10-16, later the same day).** The body below is the original
> audit text; this table records what has since been done, so a reader does not have
> to reconstruct the state from the per-finding notes.
>
> | # | Finding | Status |
> | ---: | --- | --- |
> | 1 | Unseeded vehicle-type draw | **Fixed** - both loaders seed before the draw; regression tests exist (see the note below) |
> | 2 | Availability caps + fixed type mix, not "unlimited availability" | **Open, by design** - the type mix remains instance data; this is a modelling decision for the author, not a bug to patch silently |
> | 3 | `floor` where an upper bound needs `ceil` | **Open** - low priority; the route-local path is not affected |
> | 4 | Fleet cap is per period, not in total | **Open** - holds for the monolithic formulation |
> | 5 | Global per-vehicle indexing inflates the model | **Addressed on the route-local path** (measured 14.8x smaller); the legacy monolithic path is unchanged |
> | 6 | Demand constraint is documented as an equality, coded as `>=` | **Open, documented** - intentional per §9 |
>
> **Scope note.** Findings 2-6 concern the **monolithic / column-generation models
> that the commercial solvers run**. They do **not** block work on the multi-period
> **ALNS** heuristic, which is a separate subsystem in this repository.

> **Status update (2026-10-16).** Finding 1 is **fixed**. `InputDataSingle.load()`
> and `InputDataMultiple.load()` now call `random.seed(self.seed)` immediately
> before the vehicle-type draw, with `seed` defaulting to `20260408` - the same
> value the CPLEX runners publish. Model construction is therefore reproducible
> for every caller, including callers that never touch the global `random`
> module. Regression tests: `multi-period-delivery/tests/test_cplex_api.py` and
> `single-period-delivery/tests/test_cplex_api.py` (`test_fleet_composition_*`).
> Note that Finding 2 stands: the type mix is still part of the instance data
> rather than a model decision, so this change makes the *existing* behaviour
> deterministic without adopting the "better" option in §7.2.

> **Status update (2026-10-13).** The route-local reformulation in §7.1 has
> been implemented in both Gurobi and CPLEX monolithic models for the
> single-period and multi-period projects. The single-period column-generation
> implementation has the corrected objective accounting described in §7.1.
> The public model specifications are
> [`single-period-delivery/docs/mathematical_models.md`](../single-period-delivery/docs/mathematical_models.md)
> and
> [`multi-period-delivery/docs/mathematical_models.md`](../multi-period-delivery/docs/mathematical_models.md).
> The sections below preserve the historical audit language so the reason for
> each change remains visible.

---

## 2. Finding 1 (High): the model is not reproducible

> **RESOLVED (2026-10-16).** The loader now seeds the draw itself
> (`random.seed(self.seed)`, `seed` default `20260408`), so the fleet
> composition no longer depends on the caller. The text below is the historical
> audit; it is kept so the reason for the change stays visible.

### What the code does

`InputDataSingle.load()` and `InputDataMultiple.load()` build the fleet like this:

```python
self.all_vehicles = {v_id for v_id in range(1, self.max_veh_num + 1)}
self.veh_type = {}
for v_id in self.all_vehicles:
    v_type = random.sample(self.all_veh_types, k=1)   # <-- global RNG
    self.veh_type[v_id] = v_type[0]
```

Every vehicle's type — and therefore its capacity, its minimum load and its
cost — is drawn at random from the global `random` module.

### The Gurobi path never seeds it

`rg "random.seed" single-period-delivery/src multi-period-delivery/src` finds
exactly three call sites, and none of them is on the Gurobi monolithic path:

| Call site | Seeds? |
| --- | --- |
| `single-period-delivery/.../cplex_api_monolithic/runner.py` | ✅ `random.seed(20260408)` |
| `multi-period-delivery/.../cplex_api_monolithic/runner.py` | ✅ `random.seed(20260408)` |
| `single-period-delivery/.../heuristic/main.py` | ✅ `random.seed(1234)` |
| `single-period-delivery/.../monolithic/optLogicSingle.py` | ❌ |
| `multi-period-delivery/.../monolithic/optLogicMultiple.py` | ❌ |

`multi-period-delivery/experiments/run_experiments.py` accepts `--seeds` and
writes `"SEED": int(seed)` into the configuration — but only on the **ALNS**
branch. The monolithic branch never touches `random`, so the `seed` recorded in
`run_metadata.json` for a monolithic run does not describe that run.

### Evidence

Loading the same instance three times in three processes:

| Run | `max_veh_num` | Type mix |
| ---: | ---: | --- |
| 1 | 36 | `{10.3: 19, 11.4: 17}` |
| 2 | 36 | `{11.4: 22, 10.3: 14}` |
| 3 | 36 | `{10.3: 21, 11.4: 15}` |

(multi-period, `small/dataset_1`; capacities 118 and 149, minimum loads 48 and 68.)

The single-period instance varies the same way: `{6.4: 78, 10.5: 68, 12.2: 89}`
in one run and `{6.4: 91, 10.5: 69, 12.2: 75}` in the next.

### Why it matters

- Two runs of "the same benchmark" solve **different models**. Reported
  objective values, gaps and run times are not comparable across runs.
- The Gurobi and CPLEX replications of the same mathematical model are **not
  the same model**: CPLEX is deterministic, Gurobi is not. Any Gurobi-vs-CPLEX
  comparison currently has this confound baked in.
- The archived thesis numbers cannot be reproduced from the archived code.

---

## 3. Finding 2 (High): "unlimited vehicle availability" is not what is implemented

Both problem descriptions state the assumption plainly:

> **Unlimited vehicle availability** — The 3PL can provide an unlimited number
> of vehicles. Vehicle-type selection is driven only by cost.

The implementation instead:

1. computes one global fleet size `max_veh_num` and never allows more; and
2. **fixes** each vehicle's type at load time, so "which vehicle type is used"
   is not a decision the model can make — it is a random parameter.

So the implemented model is not "an unlimited fleet whose type mix is chosen by
cost"; it is "a pre-sized fleet with a randomly fixed type mix".

The practical risks:

- If the random draw under-provides a type the optimal plan would want, the
  model can be **infeasible or artificially expensive** relative to the stated
  problem. (With `dataset_1` the fleet is generous enough that feasibility is
  not in danger, but this is instance-dependent.)
- Conversely, a lucky draw makes the instance easier, so measured run times
  understate the intended difficulty.

**This is the same issue the user raised directly**: "the assumption says the
number of vehicles is unbounded, but the model needs a fixed vehicle count to
build indices, so it has to be estimated." The estimate itself is defensible;
what makes it fragile is that the *type* of each vehicle is random rather than
either (a) optimised or (b) made irrelevant by a type-based formulation.

---

## 4. Finding 3 (Medium): `floor` where an upper bound needs `ceil`

```python
veh_num_ub1 = math.floor(total_volume1 / min_veh_cap)   # initial inventory volume
veh_num_ub2 = math.floor(total_volume2 / min_veh_cap)   # total demand volume
veh_num_ub3 = plant_nums * dealer_nums
self.max_veh_num = max(veh_num_ub1, veh_num_ub2, veh_num_ub3)
```

`total_volume / min_cap` counts how many *smallest-type* vehicles the volume
would fill; a count of vehicles needed must be rounded **up**. With
`small/dataset_1` (multi-period):

| Term | Value |
| --- | ---: |
| initial inventory volume | 951.2 |
| total demand volume | 4254.3 |
| smallest capacity (`10.3`) | 118 |
| `floor(4254.3 / 118)` | **36** |
| `ceil(4254.3 / 118)` | **37** |
| `plants × dealers` | 16 |
| chosen `max_veh_num` | **36** |

So the "upper bound" is one vehicle below the value it is supposed to bound.
On this instance the shortfall is absorbed by the slack in the plan, but the
intent of the code is not met. The same expression is used in both projects.

---

## 5. Finding 4 (Medium, multi-period): the cap is per period, not in total

`z_lt`, `zz_lt` and `p_lt` are indexed by `(v_id, t)`. Searching every
constraint that mentions `z_lt` shows it appears only in:

- the objective (`Σ v_id,t`),
- Constraint 1, `Σ_(i,j) y_ijlt == z_lt` — **per (v_id, t)**,
- Constraint 4, `z_lt + zz_lt − 1 ≤ p_lt` — **per (v_id, t)**.

No constraint ties a vehicle across periods (`Σ_t z_lt ≤ 1`, or "a vehicle used
in *t* is available in *t+1*", or similar). So the same 36 vehicles are
independently available in **each** of the 5 periods: up to 180 vehicle-trips,
against roughly 29 trips (`4254.3 / 149`) actually needed.

Two consequences:

- The derived bound is a bound on *distinct* vehicles, but what the model
  enforces is a bound on *vehicle-period activations*. These are different
  quantities; the model is far more permissive than the bound suggests.
- The variable count is multiplied by the horizon for no modelling gain.

Constraint 1 also means a vehicle serves **at most one (plant, dealer) route per
period**, which is consistent with the decision description ("which plant-dealer
shipment is assigned to each vehicle").

---

## 6. Finding 5 (Medium): individual vehicle indices inflate the model

Both models index the loading variable by vehicle id:

```python
self.x_indices = [
    (plant, dealer, sku_id, v_id, t)
    for (plant, dealer, sku_id) in triple_ijk
    for (v_id, t) in self.auxi_indices
]
```

Counted from the bundled instances (index sets only, no solve):

| Model | Instance | `z` | `y` | `x` | Total |
| --- | --- | ---: | ---: | ---: | ---: |
| single-period, as implemented | small/`dataset_1` | 235 | 11,280 | 114,210 | **125,725** |
| single-period, route-local slots | small/`dataset_1` | 3,038 | — | 5,463 | **8,501** |
| single-period, as implemented | small/`dataset_3` | 20 | 960 | 920 | **1,900** |
| single-period, route-local slots | small/`dataset_3` | 180 | — | 180 | **360** |

The vehicle index multiplies the loading variables by `max_veh_num`; replacing
it with route-local slots was measured at **14.8×** fewer variables on
`dataset_1` and **3.3×** on `dataset_3`, *while keeping the per-vehicle
minimum-load penalty exactly as it is*. This is the mechanical reason the
monolithic benchmark is slow, and it is why the single-period project needs a
column-generation alternative at all.

An earlier draft of this audit claimed **78×** by comparing against a
*type-aggregated* model that drops the per-vehicle penalty. That comparison was
not like-for-like and has been replaced by the measured figures above.

### The global fleet cap is also *tighter* than the routes need

This workstation has the size-unlimited CPLEX Studio runtime installed (see
`docs/uv_environment.md`), so the measurements below are not Community-Edition
artefacts: a 125,725-variable model solves, and `medium/dataset_1` returns
*"time limit exceeded, no integer solution"* rather than a size error.

For the bundled single-period instances:

| Scale | Dataset | Global fleet `L` | Routes | SKUs | Route-local slots |
| --- | --- | ---: | ---: | ---: | ---: |
| small | `dataset_3` | 20 | 12 | 12 | 36 |
| small | `dataset_9` | 152 | 61 | 29 | 306 |
| small | `dataset_1` | 235 | 48 | 22 | 434 |
| medium | `dataset_3` | 473 | 130 | 50 | 1157 |
| medium | `dataset_1` | 610 | 144 | 49 | 870 |
| large | `dataset_1` | 1343 | 262 | 76 | 3269 |

The global fleet is shared by every route, so on `large/dataset_1` the average
allowance is `1343 / 262 ≈ 5` vehicles per route, while the largest route needs
about `3269 / 262 ≈ 12`. The old formulation can therefore *starve* a
volume-heavy route unless other routes leave vehicles unused — the cap couples
routes that have nothing to do with each other. That coupling is a third reason
the global index is the wrong abstraction, on top of its size and its random
type mixture.

> **A design that does not work — recorded so it is not retried.** The first
> route-local draft gave every `(vehicle type, copy number)` pair its own slot
> dimension. Measured on `small/dataset_1` that produced **993 slots against a
> 235-vehicle global fleet**, i.e. *more* variables than the model it was meant
> to replace: a route never needs a full complement of every type. The
> reformulation only pays off when the slot is type-agnostic and the type is a
> model decision (`z_ijfn` with `sum_f z_ijfn <= 1`), and when the number of
> slots per route is bounded by `ceil(D_ij / min_cap) + 1` rather than by a
> per-type count.

### Why the obvious fix is not the whole story

Dropping the vehicle index is not free, because the **minimum-load penalty is a
per-vehicle cost**: a used vehicle loaded below its threshold pays
`α · cost_f`. An aggregate "how many type-f vehicles on this route" variable does
not determine how many of them are underloaded.

---

## 7. Recommendations

### 7.1 Route-local vehicle slots (implemented; measured at 14.8× on `dataset_1`)

Because Constraint 1 already forbids a vehicle from serving more than one route
per period, vehicle identities are **not shared between routes**. The global
fleet index can therefore be replaced by a *route-local* slot index whose type
is decided by the model:

```text
slots:  n ∈ {1, …, K_ij}                per route (i, j)
        K_ij = ceil(D_ij / min_cap) + 1,  D_ij = demand volume the route serves

vars:   z_ijfn, p_ijfn ∈ {0,1}          slot n of route (i,j) uses type f
        zz_ijn, x_ijkn
constr: sum_f z_ijfn            <= 1
        sum_k size_k x_ijkn     <= sum_f cap_f * z_ijfn
        sum_k size_k x_ijkn + BIG*zz_ijn >= sum_f min_load_f * z_ijfn
        z_ijfn + zz_ijn - 1     <= p_ijfn
```

This keeps the per-vehicle minimum-load penalty **exactly** as it is — the
penalty is still charged per activated vehicle, and vehicles are now per route,
which is where they are actually used — while shrinking the index set by more
than an order of magnitude. It also removes the symmetry that a global fleet
index creates between interchangeable vehicles.

#### Verification of the reformulation (single-period CPLEX API)

The reformulation is implemented in
`single-period-delivery/src/single_period_delivery/cplex_api_monolithic/model.py`
and checked two ways.

The same route-local structure is now implemented by the Gurobi single-period
model, the Gurobi and CPLEX multi-period models, and the CPLEX
column-generation implementation. The current model specifications and
implementation maps are in the linked `mathematical_models.md` files.

**1. Equivalence on a synthetic instance.** The bundled instances cannot settle
an equivalence question: `BIG_NUM = 99999` makes the LP relaxation weak enough
that even `small/dataset_3` stops on the time limit. A hand-built instance
(1 plant, 2 dealers, 2 SKUs, 2 vehicle types) is solved to *proven optimality*
by both formulations:

| Formulation | Status | Objective | Best bound |
| --- | --- | ---: | ---: |
| global fleet (before) | optimal | **2.6** | 2.6 |
| route-local slots (after) | optimal | **2.6** | 2.6 |

(2.6 = one type-`10.0` vehicle for dealer D1 at 1.6 plus one type-`6.0` vehicle
for D2 at 1.0, which is also what the instance's economics imply.)

**2. On a bundled instance.** `small/dataset_3`:

| Formulation | Time limit | Status | Objective | Gap |
| --- | ---: | --- | ---: | ---: |
| global fleet (before) | 604 s | incumbent only | 15.36 | 11.85% |
| route-local slots (after) | **~3 s** | **optimal** | **15.06** | 0% |

The new formulation reaches *and proves* an objective the old one could not
reach in ten minutes. This is not an equivalence violation: the final plan uses
15 vehicles against an old cap of 20, so the cap was never binding and both
formulations share the same optimum — the old one simply never got there.

### 7.2 Make the legacy fleet deterministic

> **Implemented (2026-10-16), minimal option.** `load()` itself now calls
> `random.seed(self.seed)` immediately before the draw, with the same default
> seed (`20260408`) the CPLEX runners publish. Seeding inside the loader - rather
> than at each call site - is what makes the guarantee hold for *every* caller,
> including `optLogicMultiple.run_model_from_paths` and any future entry point
> that forgets to seed. Regression tests are in both packages'
> `tests/test_cplex_api.py`. The "better" option below was **not** taken: the
> type mix is still instance data rather than a model decision (see Finding 2).

Whatever formulation is chosen, the vehicle-type draw must stop being an
unseeded global side effect:

- **Minimal**: call `random.seed(seed)` before `data.load()` in
  `optLogicSingle.py` / `optLogicMultiple.py`, mirroring the CPLEX runners, and
  record the realised type mix in the run metadata.
- **Better**: remove the draw from the data loader altogether. The problem says
  vehicle types are chosen by cost, so the type mix belongs to the model, not
  to the instance data.

The route-local monolithic models no longer use the loader's global fleet
fields. They are retained for compatibility with historical code paths; the
legacy fleet draw should still be made deterministic or removed when those
paths are retired.

> **Scope note.** Because the route-local models no longer read these fields,
> this fix makes the *legacy* data path deterministic. It does not by itself
> change the route-local models' solution, which derive slots from route demand
> and minimum vehicle capacity. The value of the fix is that the loader no
> longer mutates global RNG state unpredictably for whatever runs after it.

### 7.3 Correct the legacy bound

Use `math.ceil` for the two volume-based terms. Note that this **changes the
model size** (36 → 37 vehicles on `dataset_1`), so it belongs in a deliberate
model revision with the benchmark re-run, not a drive-by edit.

This remains a legacy-loader issue. It does not determine the size of the
current route-local monolithic models, whose slots are derived from route
demand and minimum vehicle capacity rather than from `max_veh_num`.

### 7.4 Align the documentation

The problem descriptions now distinguish the stated unlimited-fleet business
assumption from the legacy loader's pre-estimated, randomly typed fleet. The
route-local models use the stated assumption directly; historical loader
behavior is documented as compatibility behavior.

---

## 8. What checked out fine

Verified against the descriptions, no action needed:

- **Demand coverage across the horizon** — multi-period Constraint 6 sums over
  all plants, vehicles and periods and compares with `>=` the dealer's total
  demand, matching "the total quantity ... must satisfy the total demand".
- **Inventory-flow balance** — multi-period Constraint 7 is exactly
  `shipments = production + opening inventory − ending inventory`, per
  (plant, SKU, period), with `s_ikt[·, 0]` fixed to the initial inventory.
- **Plant inventory capacity** — multi-period Constraint 8 caps the summed
  ending inventory per plant and period; single-period Constraint 7 caps
  shipments per (plant, SKU) at the available inventory, matching the
  single-period "supply capacity" clause.
- **Vehicle capacity** — both models cap the loaded volume at the vehicle
  type's capacity (Constraint 5).
- **Minimum-load penalty** — Constraint 3 (`load + zz·BIG ≥ min_load`) plus
  Constraint 4 (`z + zz − 1 ≤ p`) together make `p_lt = 1` exactly when a
  *used* vehicle is below its minimum load, and `p_lt = 0` when the vehicle is
  unused. The penalty is charged as `α · cost` in the objective, as described.
- **Structure** — "multiple plants can supply the same SKU", "one plant serves
  many dealers", "one dealer is served by several plants" and "no horizontal
  transfers between plants" are all reflected in the `supply_chain` index set
  and the constraint indexing.
- **CPLEX API parity** — `cplex_api_monolithic/model.py` in both projects
  declares the same eight constraints, in the same order and with the same
  semantics as the Gurobi models, and its runners seed the RNG first.
- **Symmetry cuts** — Constraint 9 is disabled by default in *both* projects
  (single-period line 239, multi-period line 417), so this is consistent, not a
  discrepancy. It is worth reconsidering given §6.

---

## 9. Demand coverage is an inequality

The final thesis document and the implementation use the coverage form
`sum(x) >= demand`. This makes over-delivery legal, which allows an
underloaded but already-paid-for vehicle to be topped up without activating
another vehicle. The ALNS implementation relies on exactly that freedom
(`fill_minimum_load_repair`). An earlier audit note described the constraint
as an equality; that note was inaccurate and has been corrected here.
