# Mathematical Model for the Multi-Period Production-Driven Product Distribution Problem

## Purpose and Source

This document records the mathematical model used by the multi-period
monolithic implementation in this repository. It is derived from the final
master's thesis:

> Jiujin Wang, *Research on Models and Algorithms for Product Distribution
> Problem in a Two-Echelon Inventory System*, Shanghai University, 2026,
> Chapter 4, Sections 4.1 and 4.2.

The thesis presents a production-driven multi-period integer program in
Chapter 4. This document translates that model into English, explains the
legacy vehicle-indexed formulation and its defects, and then records the new
route-local monolithic formulation implemented here.

The corresponding implementation files are:

- data interface:
  `src/multi_period_delivery/monolithic/InputDataMultiple.py`
- Gurobi monolithic:
  `src/multi_period_delivery/monolithic/MonolithicModelMultiple.py`
- CPLEX monolithic:
  `src/multi_period_delivery/cplex_api_monolithic/model.py`

The repository-level audit that motivated the reformulation is
[`../../docs/model_implementation_audit.md`](../../docs/model_implementation_audit.md).

---

## 1. Problem Setting and Assumptions

The multi-period problem extends the single-period model to rolling
operations. Dealer demand can be covered cumulatively over the planning
horizon, while production is scheduled by plant, SKU, and period. The model
must therefore decide shipment quantities and ending inventory jointly.

Sets and indices:

- production plants: \(i\in I\);
- dealers: \(j\in J\);
- SKUs: \(k\in K\);
- vehicle types: \(f\in F\);
- planning periods: \(t\in\{1,\ldots,T\}\), with \(t=0\) representing the
  initial state.

The thesis retains the following assumptions:

1. **Sufficient supply over the horizon.** Total initial inventory plus
   scheduled production is at least total demand for every SKU.
2. **Exogenous production.** \(p_{ikt}\), the quantity of SKU \(k\) produced
   by plant \(i\) in period \(t\), is a parameter.
3. **One-dimensional volume.** SKU volume and vehicle capacity use a linear
   volume scale. Weight, stacking, and geometrical loading constraints are
   ignored.
4. **Unlimited vehicle availability.** The 3PL can provide as many vehicles
   as required, and vehicle-type selection is driven by cost.

As in the single-period model, the demand constraint is written as coverage:
over-delivery is feasible. This allows an activated vehicle to be filled to
its minimum-load threshold without adding another vehicle.

---

## 2. Notation

### 2.1 Sets and route data

| Symbol | Type | Meaning |
| --- | --- | --- |
| \(I\) | set | Production plants |
| \(J\) | set | Dealers |
| \(K\) | set | SKUs |
| \(F\) | set | Vehicle types |
| \(T\) | parameter | Number of planning periods |
| \(A \subseteq I\times J\) | set | Routes \((i,j)\) with at least one shareable SKU |
| \(K^S_{ij}\) | set | SKUs that can be shipped on route \((i,j)\) |
| \(D_{ij}\) | parameter | Aggregate route demand volume, \(D_{ij}=\sum_{k\in K^S_{ij}}v_k d_{jk}\) |
| \(u_{\min}\) | parameter | Smallest positive vehicle capacity |
| \(K_{ij}\) | parameter | Route-local slot count, \(K_{ij}=\lceil D_{ij}/u_{\min}\rceil+1\) |
| \(N_{ij}\) | set | Slot indices, \(N_{ij}=\{1,\ldots,K_{ij}\}\) |

The slot count uses aggregate demand over the full horizon. It is a safe
upper bound for every individual period because the volume on one route in
one period cannot exceed its total-horizon volume.

### 2.2 Parameters

| Symbol | Meaning |
| --- | --- |
| \(v_k\) | Volume per unit of SKU \(k\) |
| \(s_{ik0}\) | Initial inventory of SKU \(k\) at plant \(i\) |
| \(d_{jk}\) | Total demand of dealer \(j\) for SKU \(k\) over the horizon |
| \(p_{ikt}\) | Scheduled production of SKU \(k\) at plant \(i\) in period \(t\) |
| \(Q_i\) | Storage capacity of plant \(i\) |
| \(c_f\) | Fixed activation cost of vehicle type \(f\) |
| \(u_f\) | Capacity of vehicle type \(f\) |
| \(e_f\) | Minimum-load threshold of vehicle type \(f\) |
| \(\alpha\) | Minimum-load penalty multiplier, \(\alpha\ge 1\) |
| \(M\) | A sufficiently large positive constant |

### 2.3 New decision variables

| Symbol | Domain | Meaning |
| --- | --- | --- |
| \(z_{ijfn t}\) | \(\{0,1\}\) | Slot \(n\) of route \((i,j)\) uses type \(f\) in period \(t\) |
| \(z'_{ijn t}\) | \(\{0,1\}\) | Slot \(n\) is allowed to be below minimum load in period \(t\) |
| \(p_{ijfn t}\) | \(\{0,1\}\) | Slot \(n\) is used with type \(f\) and is below minimum load in period \(t\) |
| \(x_{ijkn t}\) | \(\mathbb{Z}_{+}\) | Units of SKU \(k\) loaded on slot \(n\) of route \((i,j)\) in period \(t\) |
| \(s_{ikt}\) | \(\mathbb{Z}_{+}\) | Ending inventory of SKU \(k\) at plant \(i\) in period \(t\), with \(s_{ik0}\) fixed to initial inventory |

The implementation uses `zz` for the thesis symbol \(z'\).

---

## 3. Legacy Thesis Integer Program

The thesis Section 4.2 model indexes loading decisions by an individual
vehicle \(l\in L\). It is reproduced here in English so that the motivation
for the new route-local model is explicit.

### 3.1 Legacy notation

| Symbol | Meaning |
| --- | --- |
| \(L\) | Fixed set of individual vehicles |
| \(c_l\) | Fixed usage cost of vehicle \(l\) |
| \(u_l\) | Capacity of vehicle \(l\) |
| \(e_l\) | Minimum-load threshold of vehicle \(l\) |
| \(z_{lt}\) | 1 if vehicle \(l\) is used in period \(t\) |
| \(z'_{lt}\) | 1 if vehicle \(l\) is allowed to be below minimum load in period \(t\) |
| \(p_{lt}\) | 1 if vehicle \(l\) is used and below minimum load in period \(t\) |
| \(y_{ijlt}\) | 1 if vehicle \(l\) serves route \((i,j)\) in period \(t\) |
| \(x_{ijklt}\) | Units of SKU \(k\) loaded on vehicle \(l\) on route \((i,j)\) in period \(t\) |

### 3.2 Legacy objective

$$
\min
\sum_{t=1}^{T}\sum_{l\in L}c_l z_{lt}
+
\alpha\sum_{t=1}^{T}\sum_{l\in L}c_l p_{lt}.
\tag{ML1}
$$

The first term charges vehicle activation cost in every period. The second
term charges minimum-load penalties for used vehicles that remain below their
threshold.

### 3.3 Legacy constraints

#### Vehicle use and route assignment

$$
\sum_{i\in I}\sum_{j\in J}y_{ijlt}=z_{lt},
\quad \forall l\in L,\ t=1,\ldots,T.
\tag{ML2}
$$

A vehicle is marked as used in period \(t\) when it is assigned to one route
in that period.

#### Load activation

$$
\sum_{k\in K}x_{ijklt}\le M y_{ijlt},
\quad \forall i,j,l,t.
\tag{ML3}
$$

SKUs can be loaded only when the vehicle-route assignment is active.

#### Minimum load

$$
\sum_{k\in K}v_k\sum_{i\in I}\sum_{j\in J}x_{ijklt}
+ M z'_{lt}\ge e_l,
\quad \forall l,t.
\tag{ML4}
$$

The loaded volume must satisfy the threshold unless the minimum-load slack is
activated.

#### Penalty linkage

$$
z_{lt}+z'_{lt}-1\le p_{lt},
\quad \forall l,t.
\tag{ML5}
$$

The penalty is forced only for used vehicles that activate the minimum-load
slack.

#### Vehicle capacity

$$
\sum_{k\in K}v_k\sum_{i\in I}\sum_{j\in J}x_{ijklt}\le u_l,
\quad \forall l,t.
\tag{ML6}
$$

Vehicle capacity cannot be exceeded.

#### Horizon demand coverage

$$
\sum_{t=1}^{T}\sum_{i\in I}\sum_{l\in L}x_{ijklt}\ge d_{jk},
\quad \forall j,k.
\tag{ML7}
$$

For each dealer and SKU, cumulative shipments over the horizon must cover
total demand.

#### Inventory-flow balance

$$
\sum_{j\in J}\sum_{l\in L}x_{ijklt}
=
p_{ikt}+s_{ik,t-1}-s_{ikt},
\quad \forall i,k,t.
\tag{ML8}
$$

The quantity shipped in period \(t\) equals production in period \(t\) plus
opening inventory minus ending inventory. This is the staircase coupling that
links adjacent periods.

#### Plant inventory capacity

$$
\sum_{k\in K}s_{ikt}\le Q_i,
\quad \forall i,t.
\tag{ML9}
$$

The total ending inventory at a plant cannot exceed its storage capacity.

#### Domains

$$
x_{ijklt}\in\mathbb{Z}_{+},\qquad
y_{ijlt},z_{lt},z'_{lt},p_{lt}\in\{0,1\},\qquad
s_{ikt}\in\mathbb{Z}_{+}.
\tag{ML10}
$$

### 3.4 Defects of the legacy multi-period model

1. **Finite fleet index versus unlimited fleet assumption.** Variables exist
   only for a pre-specified set \(L\). The fleet-size bound is therefore a
   modelling approximation, not an operational restriction stated by the
   problem.
2. **Vehicle type is fixed before optimization.** The parameters \(c_l\),
   \(u_l\), and \(e_l\) are attached to a vehicle identity. The solver cannot
   select a different type for the same activation opportunity. The archived
   loader generated the type mix randomly, which made otherwise identical
   runs solve different models.
3. **The global fleet couples unrelated routes.** A vehicle is shared by the
   entire route set, creating symmetry and an index of size
   \(|I|\times|J|\times|K|\times|L|\times T\).
4. **The fleet cap is per period.** No constraint links \(z_{lt}\) across
   periods. Consequently, a bound of \(L\) vehicles acts as a bound on
   vehicle-period activations rather than on a persistent fleet.
5. **Inventory precision is coupled to an inflated loading model.** The
   staircase inventory equations are small and well structured, but they are
   solved together with a loading model whose size is dominated by the
   artificial vehicle index.
6. **The cost and capacity of a vehicle are not decision variables.** The
   problem statement says that vehicle-type selection is driven by cost.
   The legacy formulation instead treats the fleet mix as data.

---

## 4. New Route-Local Multi-Period Monolithic Model

The new model uses a route-local slot in each period. A slot is a modelling
position on a route, not a permanent physical vehicle. This is consistent
with the stated assumption that the 3PL can provide an unlimited number of
vehicles and that the relevant decision is the cost-minimizing type of each
period-specific activation.

### 4.1 Slot definition

For route \((i,j)\), the implementation uses

$$
K_{ij}=
\left\lceil \frac{D_{ij}}{u_{\min}} \right\rceil+1.
$$

The route can be served by up to \(K_{ij}\) slots in any period. Since
\(D_{ij}\) is an upper bound on total route volume over the horizon, this
number is sufficient for every period. The extra slot is retained for plans
that split a load around the minimum-load threshold.

### 4.2 Objective

$$
\min
\sum_{t=1}^{T}
\sum_{(i,j)\in A}
\sum_{n\in N_{ij}}
\sum_{f\in F}
c_f z_{ijfn t}
+
\alpha
\sum_{t=1}^{T}
\sum_{(i,j)\in A}
\sum_{n\in N_{ij}}
\sum_{f\in F}
c_f p_{ijfn t}.
\tag{MN1}
$$

Fixed activation cost is charged per activated route-local slot and period.
The minimum-load penalty is charged per activated slot that remains below the
threshold of its selected type.

### 4.3 Constraints

#### One type per slot and period

$$
\sum_{f\in F}z_{ijfn t}\le 1,
\quad \forall (i,j)\in A,\ n\in N_{ij},\ t=1,\ldots,T.
\tag{MN2}
$$

The slot is either unused in period \(t\) or assigned one vehicle type.

#### Vehicle capacity

$$
\sum_{k\in K^S_{ij}}v_k x_{ijkn t}
\le
\sum_{f\in F}u_f z_{ijfn t},
\quad \forall i,j,n,t.
\tag{MN3}
$$

The volume loaded on the slot cannot exceed the capacity of the selected
type.

#### Minimum load

$$
\sum_{k\in K^S_{ij}}v_k x_{ijkn t}
+ M z'_{ijn t}
\ge
\sum_{f\in F}e_f z_{ijfn t},
\quad \forall i,j,n,t.
\tag{MN4}
$$

The selected type's minimum-load threshold must hold unless the slack
variable is activated.

#### Penalty linkage

$$
z_{ijfn t}+z'_{ijn t}-1\le p_{ijfn t},
\quad \forall i,j,f,n,t.
\tag{MN5}
$$

The penalty variable is one exactly when a used slot is below its selected
type's threshold.

#### Cumulative demand coverage

$$
\sum_{t=1}^{T}
\sum_{i\in I}
\sum_{n\in N_{ij}}
x_{ijkn t}
\ge d_{jk},
\quad \forall j\in J,\ k\in K.
\tag{MN6}
$$

Dealer demand can be fulfilled in any period, but it must be covered by the
end of the horizon.

#### Inventory-flow balance

$$
\sum_{j\in J}\sum_{n\in N_{ij}}x_{ijkn t}
=
p_{ikt}+s_{ik,t-1}-s_{ikt},
\quad \forall i,k,t.
\tag{MN7}
$$

The equation is unchanged from the thesis model. Shipments in period \(t\)
equal scheduled production plus opening inventory minus ending inventory.
The initial inventory is fixed:

$$
s_{ik0}=\text{initial inventory}_{ik}.
$$

#### Plant inventory capacity

$$
\sum_{k\in K}s_{ikt}\le Q_i,
\quad \forall i,\ t.
\tag{MN8}
$$

Total ending inventory at a plant cannot exceed its storage capacity.

#### Domains

$$
x_{ijkn t}\in\mathbb{Z}_{+},\qquad
z_{ijfn t},z'_{ijn t},p_{ijfn t}\in\{0,1\},\qquad
s_{ikt}\in\mathbb{Z}_{+}.
\tag{MN9}
$$

### 4.4 Constraint interpretation summary

| Constraint | Meaning |
| --- | --- |
| \((MN1)\) | Minimize vehicle activation and minimum-load penalties over the horizon |
| \((MN2)\) | Select at most one vehicle type for each route slot and period |
| \((MN3)\) | Enforce the selected vehicle type's capacity |
| \((MN4)\) | Enforce the selected minimum-load threshold unless penalized |
| \((MN5)\) | Link vehicle use, minimum-load slack, and the penalty |
| \((MN6)\) | Cover cumulative dealer demand over the horizon |
| \((MN7)\) | Maintain the staircase inventory balance across periods |
| \((MN8)\) | Respect plant storage capacity |
| \((MN9)\) | State integer and binary domains |

---

## 5. Relationship Between the New and Legacy Models

The route-local model changes the representation of vehicle availability,
not the physical shipment or inventory equations.

- Dealer demand and plant inventory remain the same economic restrictions.
- The inventory staircase \((MN7)\) is identical in meaning to \((ML8)\).
- The minimum-load penalty remains a per-vehicle charge. It is not replaced
  by a route-average penalty.
- A route-local slot is closed in every period in which it is not selected,
  so no shipment can enter an unused slot.
- The type decision \(z_{ijfn t}\) replaces the fixed parameters \(c_l\),
  \(u_l\), and \(e_l\).
- The global fleet index \(l\) and the assignment variable \(y_{ijlt}\) are
  removed.

If a future version adds a constraint that a physical vehicle must keep its
type across periods, the model would need cross-period type variables and
linking constraints. Such a requirement is not present in the thesis problem
or in the current implementation.

---

## 6. Why the New Model Is Simpler

The new formulation has three structural advantages.

1. **No arbitrary fleet roster.** The model does not need `max_veh_num`,
   randomly assigned vehicle types, or a precomputed finite set \(L\).
2. **No global route coupling through vehicle identity.** A slot belongs to
   one route. There is no symmetry between assigning the same physical
   vehicle to unrelated routes.
3. **A smaller and more structured variable set.** Only route-level loading
   variables \(x_{ijkn t}\), type choices \(z_{ijfn t}\), slack variables
   \(z'_{ijn t}\), penalties \(p_{ijfn t}\), and inventory variables
   \(s_{ikt}\) are required.

The multi-period inventory model remains a staircase model, but the loading
substructure no longer carries a global fleet dimension.

---

## 7. Implementation Mapping

| Mathematical object | Gurobi implementation | CPLEX implementation |
| --- | --- | --- |
| \(z_{ijfn t}\) | `z_ijfn_t` | `z[(i,j,f,n,t)]` |
| \(z'_{ijn t}\) | `zz_ijn_t` | `zz[(i,j,n,t)]` |
| \(p_{ijfn t}\) | `p_ijfn_t` | `p[(i,j,f,n,t)]` |
| \(x_{ijkn t}\) | `x_ijkn_t` | `x[(i,j,k,n,t)]` |
| \(s_{ikt}\) | `s_ikt` | `s[(i,k,t)]` |
| \(K_{ij}\), \(N_{ij}\) | `data.route_slots` | `data.route_slots` |

The data loader derives `route_slots`, `veh_type_cap`,
`veh_type_min_load`, and `veh_type_cost` so that both monolithic
implementations use the same route-local index set.

---

## 8. Validation and Operational Notes

The route-local multi-period reformulation was checked on a small synthetic
instance:

| Formulation | Status | Objective | Best bound |
| --- | --- | ---: | ---: |
| Legacy vehicle-indexed CPLEX model | optimal | 2.0 | 2.0 |
| Route-local CPLEX model | optimal | 2.0 | 2.0 |
| Legacy vehicle-indexed Gurobi model | optimal | 2.0 | 2.0 |
| Route-local Gurobi model | optimal | 2.0 | 2.0 |

The equality of the four objectives is a small-instance equivalence check.
The larger value of the reformulation is not only equivalence but also
scalability: the legacy model can become memory-bound because its loading
variables grow with the global fleet index and the horizon, whereas the new
model grows with route-local slots and periods.

The model is intended as the exact monolithic benchmark. ALNS and other
heuristics can be compared against it without inheriting the random
vehicle-type mixture or the artificial global fleet of the legacy
implementation.
