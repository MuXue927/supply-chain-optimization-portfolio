# Mathematical Models for the Single-Period Product Distribution Problem

## Purpose and Source

This document records the mathematical models used by the single-period
implementation in this repository. It is derived from the final master's
thesis:

> Jiujin Wang, *Research on Models and Algorithms for Product Distribution
> Problem in a Two-Echelon Inventory System*, Shanghai University, 2026,
> Chapter 3, Sections 3.2 and 3.3.

The thesis presents a vehicle-indexed integer program in Section 3.2 and a
column-generation decomposition in Section 3.3. This document translates both
models into English and then records the route-local monolithic reformulation
implemented here. The reformulation preserves the business problem and the
minimum-load penalty while removing the global vehicle index that causes the
original model to grow with the size of an artificial fleet.

The repository-level audit that motivated the reformulation is
[`../../docs/model_implementation_audit.md`](../../docs/model_implementation_audit.md).
The implementation paths are:

- Gurobi monolithic:
  `src/single_period_delivery/monolithic/MonolithicModelSingle.py`
- CPLEX monolithic:
  `src/single_period_delivery/cplex_api_monolithic/model.py`
- Gurobi column generation:
  `src/single_period_delivery/column_generation/`
- CPLEX column generation:
  `src/single_period_delivery/cplex_api_column_generation/model.py`

---

## 1. Problem Setting and Assumptions

Consider one planning period in a two-echelon inventory system.

- Production plants are indexed by $i \in I$.
- Dealers are indexed by $j \in J$.
- SKUs are indexed by $k \in K$.
- Vehicle types are indexed by $f \in F$.

The supply network is many-to-many: one SKU can be supplied by several plants,
one plant can serve several dealers, and one dealer can receive shipments from
several plants. Horizontal transfers between plants are not allowed.

The following assumptions are retained from the thesis:

1. **Sufficient aggregate supply.** For every SKU, total plant inventory is at
   least total dealer demand.
2. **Static production and inventory.** Production is not a decision in the
   single-period problem. Plant inventory is an exogenous parameter.
3. **One-dimensional volume.** SKU volume and vehicle capacity are measured on
   one linear volume scale. Weight, stacking, loading order, and geometrical
   packing restrictions are not modelled.
4. **Unlimited vehicle availability.** The third-party logistics provider
   can supply an unlimited number of vehicles. Vehicle-type selection is
   driven by cost, not by a fixed fleet roster.

The final thesis writes the dealer-demand constraint as a coverage
inequality. Over-delivery is therefore feasible. This matters commercially
because an already activated vehicle can be topped up to its minimum-load
threshold instead of activating another vehicle.

---

## 2. Parameter and Index Notation

The notation below follows the thesis Chapter 3 tables, with the additional
route-local symbols required by the reformulation.

| Symbol | Type | Meaning |
| --- | --- | --- |
| $I$ | set | Production plants |
| $J$ | set | Dealers |
| $K$ | set | SKUs |
| $F$ | set | Vehicle types |
| $A \subseteq I \times J$ | set | Routes $(i,j)$ for which plant $i$ can supply at least one SKU required by dealer $j$ |
| $K^S_{ij}$ | set | SKUs that can be shipped on route $(i,j)$: $K^S_{ij}=\{k:(i,k)\in \text{inventory and }(j,k)\in \text{demand}\}$ |
| $v_k$ | parameter | Volume per unit of SKU $k$ |
| $s_{ik}$ | parameter | Inventory of SKU $k$ at plant $i$ |
| $d_{jk}$ | parameter | Demand of dealer $j$ for SKU $k$ |
| $c_f$ | parameter | Fixed activation cost of vehicle type $f$ |
| $u_f$ | parameter | Capacity of vehicle type $f$ |
| $e_f$ | parameter | Minimum-load threshold of vehicle type $f$ |
| $\alpha$ | parameter | Minimum-load penalty multiplier, with $\alpha \ge 1$ |
| $M$ | parameter | A sufficiently large positive constant |
| $D_{ij}$ | parameter | Upper bound on route demand volume, $D_{ij}=\sum_{k\in K^S_{ij}} v_k d_{jk}$ |
| $u_{\min}$ | parameter | Smallest positive capacity, $u_{\min}=\min_{f\in F:u_f>0} u_f$ |
| $K_{ij}$ | parameter | Number of route-local slots, $K_{ij}=\lceil D_{ij}/u_{\min}\rceil+1$ |
| $N_{ij}$ | set | Route-local slot indices, $N_{ij}=\{1,\ldots,K_{ij}\}$ |
| $z_{ijfn}$ | binary | 1 if slot $n$ on route $(i,j)$ uses vehicle type $f$, 0 otherwise |
| $z'_{ijn}$ | binary | 1 if slot $n$ is allowed to be below the selected type's minimum load |
| $p_{ijfn}$ | binary | 1 if slot $n$ is used with type $f$ and is below its minimum load |
| $x_{ijkn}$ | integer | Units of SKU $k$ loaded on slot $n$ of route $(i,j)$ |

The legacy model uses the thesis notation $z_l$, $z'_l$, $p_l$,
$y_{ijl}$, and $x_{ijkl}$, where $l\in L$ is an individual vehicle.
The implementation uses `zz` for the thesis symbol $z'$.

---

## 3. Legacy Thesis Integer Program

The thesis Section 3.2 model is written below for reference. It is a valid
finite-fleet formulation, but it is the formulation whose scaling and
reproducibility defects motivated the route-local reformulation.

### 3.1 Legacy parameters

| Symbol | Meaning |
| --- | --- |
| $L$ | Fixed set of individual vehicles available to the carrier |
| $c_l$ | Fixed activation cost of vehicle $l$ |
| $u_l$ | Capacity of vehicle $l$ |
| $e_l$ | Minimum-load threshold of vehicle $l$ |
| $z_l$ | 1 if vehicle $l$ is used |
| $z'_l$ | 1 if vehicle $l$ is allowed to be below its minimum load |
| $p_l$ | 1 if vehicle $l$ is used and below its minimum load |
| $y_{ijl}$ | 1 if vehicle $l$ is assigned to route $(i,j)$ |
| $x_{ijkl}$ | Units of SKU $k$ loaded on vehicle $l$ on route $(i,j)$ |

### 3.2 Legacy objective and constraints

$$
\min \sum_{l\in L} c_l z_l
+ \alpha \sum_{l\in L} c_l p_l
\tag{L1}
$$

subject to

$$
\sum_{i\in I}\sum_{j\in J} y_{ijl}=z_l,
\quad \forall l\in L
\tag{L2}
$$

$$
\sum_{k\in K}x_{ijkl}\le M y_{ijl},
\quad \forall i\in I,\ j\in J,\ l\in L
\tag{L3}
$$

$$
\sum_{k\in K}v_k\sum_{i\in I}\sum_{j\in J}x_{ijkl}
+ M z'_l \ge e_l,
\quad \forall l\in L
\tag{L4}
$$

$$
z_l+z'_l-1\le p_l,
\quad \forall l\in L
\tag{L5}
$$

$$
\sum_{k\in K}v_k\sum_{i\in I}\sum_{j\in J}x_{ijkl}
\le u_l,
\quad \forall l\in L
\tag{L6}
$$

$$
\sum_{i\in I}\sum_{l\in L}x_{ijkl}\ge d_{jk},
\quad \forall j\in J,\ k\in K
\tag{L7}
$$

$$
\sum_{j\in J}\sum_{l\in L}x_{ijkl}\le s_{ik},
\quad \forall i\in I,\ k\in K
\tag{L8}
$$

$$
x_{ijkl}\in \mathbb{Z}_{+},
\quad y_{ijl}\in\{0,1\},
\quad z_l,z'_l,p_l\in\{0,1\}.
\tag{L9}
$$

Constraint interpretations:

- **(L1)** Minimizes fixed vehicle activation cost plus the penalty for used
  vehicles that do not reach their minimum-load threshold.
- **(L2)** Links the use indicator of vehicle $l$ to its route assignment.
  A vehicle can be assigned to at most one route because the left-hand side
  counts at most one selected route.
- **(L3)** Allows SKU quantities to be loaded only when the vehicle is
  assigned to the route.
- **(L4)** Enforces the minimum-load threshold unless the slack variable
  $z'_l$ is activated.
- **(L5)** Defines the penalty indicator $p_l$. Together with minimization,
  it becomes 1 exactly when a used vehicle is below its minimum load.
- **(L6)** Enforces vehicle capacity.
- **(L7)** Covers dealer demand. The inequality allows top-up shipments.
- **(L8)** Enforces plant inventory availability.
- **(L9)** States the variable domains.

### 3.3 Defects of the legacy formulation

The legacy model is mathematically coherent for a known, finite,
heterogeneous fleet. It becomes a defective modelling abstraction when the
actual problem assumption is an unlimited fleet whose types are chosen by
cost:

1. **The fleet index conflicts with unlimited availability.** The model
   requires a finite set $L$ before it can create variables. The size of
   $L$ is therefore a modelling approximation rather than a business
   parameter.
2. **Vehicle type is a parameter, not a decision.** Once $l$ is defined,
   $c_l$, $u_l$, and $e_l$ are fixed. The solver cannot choose an
   alternative type for that vehicle. The archived loader even generated the
   type mix randomly, so two runs of the same instance could solve different
   models.
3. **The global fleet couples unrelated routes.** Vehicle $l$ is shared by
   every route in the index set. Even though a route assignment consumes the
   vehicle only once, the single global fleet creates symmetry and makes the
   model larger as $I\times J\times K\times L$.
4. **Interchangeable vehicles create massive symmetry.** Many solutions are
   permutations of the same physical plan. Branch-and-bound must repeatedly
   rediscover equivalent assignments.
5. **The bound on $L$ is implementation-sensitive.** In the archived code,
   the fleet-size bound used floor division and a randomly typed fleet. This
   introduced both an unnecessarily loose or mismatched bound and
   non-reproducible parameter values.
6. **The column-generation model inherited the global vehicle index.** Its
   master model works at the vehicle-type level, but the original objective
   accounting did not align the master pattern cost and the pricing penalty.
   The corrected implementation uses one consistent column-cost definition.

---

## 4. New Route-Local Monolithic Model

The new model treats a vehicle as a type choice on a route-local slot rather
than as a member of a global fleet roster. A slot $(i,j,n)$ belongs to one
route. Its type is selected by $z_{ijfn}$. This removes $y_{ijl}$, removes
the global fleet index, and makes the vehicle-type decision explicit.

### 4.1 Why the slot count is safe

For route $(i,j)$, define

$$
D_{ij}=\sum_{k\in K^S_{ij}} v_k d_{jk}.
$$

This is an upper bound on the volume that could be assigned to the route. If
every activated slot uses the smallest positive vehicle type, at most

$$
\left\lceil \frac{D_{ij}}{u_{\min}} \right\rceil
$$

vehicles are required to carry the route volume. The model provides

$$
K_{ij}=\left\lceil \frac{D_{ij}}{u_{\min}} \right\rceil+1
$$

slots. The additional slot is deliberate slack for splitting a load around
the minimum-load threshold. Because every activation has nonnegative fixed
cost and has no benefit beyond carrying demand, a plan with more route
activations than this bound is dominated by a plan using the available slots.
When $D_{ij}=0$ or $u_{\min}=0$, the implementation uses one slot.

### 4.2 New decision variables

| Symbol | Domain | Meaning |
| --- | --- | --- |
| $z_{ijfn}$ | $\{0,1\}$ | Slot $n$ on route $(i,j)$ uses type $f$ |
| $z'_{ijn}$ | $\{0,1\}$ | Slot $n$ is allowed to be below minimum load |
| $p_{ijfn}$ | $\{0,1\}$ | Slot $n$ is used with type $f$ and is below minimum load |
| $x_{ijkn}$ | $\mathbb{Z}_{+}$ | Units of SKU $k$ loaded on slot $n$ of route $(i,j)$ |

All variables are defined only for $(i,j)\in A$, $n\in N_{ij}$,
$f\in F$, and $k\in K^S_{ij}$.

### 4.3 Objective

$$
\min
\sum_{(i,j)\in A}\sum_{n\in N_{ij}}\sum_{f\in F}
c_f z_{ijfn}
+
\alpha
\sum_{(i,j)\in A}\sum_{n\in N_{ij}}\sum_{f\in F}
c_f p_{ijfn}.
\tag{N1}
$$

The first term charges the fixed cost of each activated route-local slot. The
second term charges the minimum-load penalty only when a used slot is below
the selected type's threshold.

### 4.4 Constraints

#### One type per slot

$$
\sum_{f\in F} z_{ijfn}\le 1,
\quad \forall (i,j)\in A,\ n\in N_{ij}.
\tag{N2}
$$

Each route-local slot can be either unused or assigned exactly one vehicle
type.

#### Vehicle capacity

$$
\sum_{k\in K^S_{ij}} v_k x_{ijkn}
\le
\sum_{f\in F} u_f z_{ijfn},
\quad \forall (i,j)\in A,\ n\in N_{ij}.
\tag{N3}
$$

The loaded volume on a slot cannot exceed the capacity of its selected type.
If the slot is unused, both sides are zero, so no load can enter the slot.

#### Minimum load

$$
\sum_{k\in K^S_{ij}} v_k x_{ijkn}
+ M z'_{ijn}
\ge
\sum_{f\in F} e_f z_{ijfn},
\quad \forall (i,j)\in A,\ n\in N_{ij}.
\tag{N4}
$$

The load must satisfy the selected type's minimum threshold unless the
minimum-load slack is activated. The slack does not mean that the threshold
disappears from the economic problem; it triggers the penalty in $(N5)$.

#### Penalty linkage

$$
z_{ijfn}+z'_{ijn}-1\le p_{ijfn},
\quad \forall (i,j)\in A,\ n\in N_{ij},\ f\in F.
\tag{N5}
$$

If a slot is used and the minimum-load slack is active, the corresponding
penalty indicator must be one. If the slot's selected type is different from
$f$, or if the slot is unused, the left-hand side is non-positive and the
penalty is not forced.

#### Dealer demand coverage

$$
\sum_{i\in I}\sum_{n\in N_{ij}} x_{ijkn}
\ge d_{jk},
\quad \forall j\in J,\ k\in K.
\tag{N6}
$$

The total quantity of SKU $k$ delivered to dealer $j$ must cover its
demand. The inequality permits over-delivery, which is consistent with the
final thesis model and is useful for reaching a minimum-load threshold.

#### Plant inventory availability

$$
\sum_{j\in J}\sum_{n\in N_{ij}} x_{ijkn}
\le s_{ik},
\quad \forall i\in I,\ k\in K.
\tag{N7}
$$

The quantity shipped from plant $i$ cannot exceed the available inventory
of SKU $k$.

#### Variable domains

$$
x_{ijkn}\in\mathbb{Z}_{+},\qquad
z_{ijfn},z'_{ijn},p_{ijfn}\in\{0,1\}.
\tag{N8}
$$

### 4.5 Constraint interpretation summary

| Constraint | Economic and algorithmic meaning |
| --- | --- |
| $(N1)$ | Minimize fixed activation cost plus per-vehicle minimum-load penalties |
| $(N2)$ | Prevent assigning two vehicle types to one route-local slot |
| $(N3)$ | Enforce type-dependent vehicle capacity |
| $(N4)$ | Enforce the selected type's minimum-load threshold unless penalized |
| $(N5)$ | Make the penalty indicator exact under minimization |
| $(N6)$ | Cover each dealer-SKU demand over the period |
| $(N7)$ | Respect each plant's SKU inventory |
| $(N8)$ | State integer load and binary type, slack, and penalty variables |

---

## 5. Equivalence and Modelling Consequences

The reformulation is equivalent under the stated assumptions:

1. A vehicle is assigned to at most one route in the original model.
   Therefore a route-local slot can replace the global vehicle identity for
   that assignment.
2. The number of route-local slots is sufficient to reproduce every
   non-dominated shipment plan. The construction uses the smallest possible
   vehicle capacity and adds one slack slot.
3. Vehicle type becomes explicit through $z_{ijfn}$. There is no need to
   pre-draw a fleet type mix, so the solver chooses the type that minimizes
   total cost.
4. The $y_{ijl}$ assignment variable is unnecessary. A slot belongs to a
   route by construction.
5. The minimum-load penalty remains per activated vehicle. It is not
   aggregated across a route, so the objective keeps its original economic
   interpretation.

The reformulation deliberately removes the finite-fleet restriction. If a
future business case imposes a finite number of vehicles of each type, or
requires the same vehicle to retain its type across periods, those
requirements must be added explicitly as type-count, route-linking, or
cross-period linking constraints.

Empirical consequences recorded in this repository:

- On `small/dataset_1`, the route-local model reaches a 179.40 incumbent with
  a 0.27% gap, compared with the legacy reference result of 233.14.
- On `small/dataset_3`, the route-local model proves optimality at 15.06 in
  about 2.4 seconds, compared with a thesis reference incumbent of 15.36.
- The model has also been mirrored in the Gurobi and CPLEX monolithic
  implementations, so the published formulation and both solvers now use the
  same route-local structure.

The refreshed comparison is stored in
[`../../results/single-period/comparison.csv`](../../results/single-period/comparison.csv).

---

## 6. Column-Generation Algorithm Model

The column-generation model is the decomposition of the single-period
problem. It is based on thesis Section 3.3 and is implemented by the
`column_generation/` and `cplex_api_column_generation/` packages.

The idea is:

1. Represent a vehicle trip as a feasible load pattern.
2. Select the use of patterns in a restricted master problem.
3. Read the dual prices of demand and inventory constraints.
4. Price new patterns by solving a small integer problem for each
   $(i,j,f)$.
5. Add every pattern with positive reduced-cost contribution and repeat.
6. Solve the resulting pattern master as an integer program.

### 6.1 Additional notation

| Symbol | Meaning |
| --- | --- |
| $a_{ijkf}$ | Quantity of SKU $k$ in one load pattern on route $(i,j)$ using type $f$ |
| $\lambda_{ijf}$ | Number of times the corresponding pattern is selected |
| $\phi_{ijf}$ | Cost of one use of the pattern |
| $\delta_{ijf}$ | 1 if the pattern is below the type's minimum load |
| $\pi^D_{jk}$ | Dual price of the dealer-demand constraint |
| $\pi^S_k$ | Dual price of the total-SKU-supply constraint |
| $\pi^I_{ik}$ | Dual price of the plant-SKU inventory constraint |
| $M_1,M_2,M_3$ | Large penalties on artificial variables in the initial master |
| $r^D_{jk},r^S_k,r^I_{ik}$ | Nonnegative artificial variables for demand, total supply, and plant inventory |

The thesis uses the artificial variables $\alpha_{jk}$, $\beta_k$, and
$\gamma_{ik}$. They are renamed here to avoid a collision with the
minimum-load coefficient $\alpha$.

### 6.2 Thesis restricted master problem

For a generated set of feasible patterns:

$$
\min \sum_{i\in I}\sum_{j\in J}\sum_{f\in F} c_f \lambda_{ijf}
\tag{CG1}
$$

subject to

$$
\sum_{i\in I}\sum_{f\in F} a_{ijkf}\lambda_{ijf}
\ge d_{jk},
\quad \forall j\in J,\ k\in K
\tag{CG2}
$$

$$
\sum_{i\in I}\sum_{j\in J}\sum_{f\in F}
a_{ijkf}\lambda_{ijf}
\le \sum_{i\in I}s_{ik},
\quad \forall k\in K
\tag{CG3}
$$

$$
\sum_{j\in J}\sum_{f\in F}
a_{ijkf}\lambda_{ijf}
\le s_{ik},
\quad \forall i\in I,\ k\in K
\tag{CG4}
$$

$$
\lambda_{ijf}\ge 0.
\tag{CG5}
$$

Constraint $(CG3)$ is a global SKU inventory cut. Constraint $(CG4)$ is
stronger at the plant-SKU level and is the inventory row used by the
implementation. The final master fixes $\lambda_{ijf}\in\mathbb{Z}_{+}$;
during pricing, the restricted master is solved as a linear program.

### 6.3 Initial master with artificial coverage

To obtain a feasible initial linear program, the thesis adds artificial
variables:

$$
\min
\sum_{i,j,f} c_f\lambda_{ijf}
+ M_1\sum_{j,k}r^D_{jk}
+ M_2\sum_k r^S_k
+ M_3\sum_{i,k}r^I_{ik}.
\tag{CG6}
$$

subject to

$$
\sum_{i,f}a_{ijkf}\lambda_{ijf}+r^D_{jk}\ge d_{jk},
\quad \forall j,k
\tag{CG7}
$$

$$
\sum_{i,j,f}a_{ijkf}\lambda_{ijf}-r^S_k
\le \sum_i s_{ik},
\quad \forall k
\tag{CG8}
$$

$$
\sum_{j,f}a_{ijkf}\lambda_{ijf}-r^I_{ik}
\le s_{ik},
\quad \forall i,k
\tag{CG9}
$$

$$
r^D_{jk},r^S_k,r^I_{ik}\ge 0,\qquad
\lambda_{ijf}\ge 0.
\tag{CG10}
$$

The artificial variables make the restricted master feasible before enough
vehicle patterns have been generated. They are driven to zero by the large
penalties $M_1,M_2,M_3$.

### 6.4 Pricing problem

For fixed $(i,j,f)$, the pricing problem searches for a load pattern with
the largest reduced-cost contribution:

$$
\max
\sum_{k\in K^S_{ij}}
\left(\pi^D_{jk}+\pi^S_k+\pi^I_{ik}\right)a_{ijkf}
- c_f-\eta_{ijf}
\tag{CG11}
$$

subject to

$$
\sum_{k\in K^S_{ij}} v_k a_{ijkf}\le u_f
\tag{CG12}
$$

$$
\sum_{k\in K^S_{ij}} v_k a_{ijkf}
+ M\delta_{ijf}
\ge e_f
\tag{CG13}
$$

$$
0\le a_{ijkf}\le \min(d_{jk},s_{ik}),
\quad a_{ijkf}\in\mathbb{Z}_{+},
\quad \delta_{ijf}\in\{0,1\}.
\tag{CG14}
$$

Here $\eta_{ijf}$ is the minimum-load penalty of the pattern. The corrected
implementation uses

$$
\eta_{ijf}=\alpha c_f\delta_{ijf},
\qquad
\phi_{ijf}=c_f+\eta_{ijf}.
\tag{CG15}
$$

Thus the same column cost is used in the master, the pricing objective, and
the final objective recomputation. This corrects the historical mismatch in
which the master used only $c_f$ while the pricing model charged a
minimum-load penalty directly.

Constraint interpretations:

- $(CG11)$ is the pricing objective. A positive value means that adding the
  pattern improves the restricted master objective.
- $(CG12)$ enforces the type capacity.
- $(CG13)$ is the minimum-load threshold. The binary $\delta_{ijf}$
  activates the large-M relaxation when the pattern is below threshold.
- $(CG14)$ bounds each SKU quantity by demand and plant inventory and
  requires integer units.

### 6.5 Mapping between thesis and implementation

| Thesis concept | Repository representation |
| --- | --- |
| Pattern load $a_{ijkf}$ | `Pattern.load` |
| Pattern selection $\lambda_{ijf}$ | `pattern_vars[index]` |
| Minimum-load indicator | `Pattern.penalty_indicator` |
| Pattern cost $\phi_{ijf}$ | `Pattern.cost` |
| Demand dual $\pi^D_{jk}$ | `demand_duals[(dealer, sku)]` |
| Plant-inventory dual $\pi^I_{ik}$ | `supply_duals[(plant, sku)]` |
| Pricing problem | `solve_pricing_subproblem(...)` |
| Integer final master | `_build_master(integer_patterns=True)` |

The implementation uses the same route-specific inventory row as $(CG4)$
and omits the redundant aggregate row $(CG3)$. It also uses one artificial
variable family for demand coverage in the CPLEX API version; the original
Gurobi version retains the thesis's three artificial families for historical
comparison.

---

## 7. Model Selection Guidance

The route-local monolithic model is the mathematical benchmark for the
single-period problem because it describes the full decision space without a
pre-sized vehicle roster. It is also the correct model to use when the goal
is to validate another algorithm.

The column-generation model remains useful when:

- the monolithic model is too large to solve directly;
- a pattern decomposition can exploit the route and vehicle-type structure;
- a good initial set of patterns is available;
- the analyst needs a master-pricing decomposition for algorithmic study.

Both models represent the same minimum-load economics. The monolithic model
is the reference; the column-generation model is the decomposition used to
construct and certify columns for that reference problem.
