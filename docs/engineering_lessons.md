# Engineering Lessons and Verification Checklist

This document records the reusable engineering lessons discovered while
reviewing, reproducing, and refactoring this portfolio. It is intentionally
practical: each lesson states the symptom, the underlying cause, how the issue
is detected, and what evidence is required before calling the fix complete.

The file complements the mathematical model specifications and validation
reports. It is also a checklist for future code review, interview preparation,
and regression work.

---

## 1. Model and solver lessons

### 1.1 A global vehicle index can couple unrelated routes

**Symptom:** The monolithic model grows rapidly with plants, dealers, SKUs, and
the estimated fleet. Medium instances may return no integer solution, and
large instances may run out of memory.

**Root cause:** The original formulation indexes loading variables by an
individual vehicle in a global set. The model then uses an assignment variable
to connect that vehicle to one route. However, the physical decision allows
vehicles to be selected per route, and the stated problem assumes unlimited
vehicle availability.

**Remediation:** Replace the global vehicle identity with route-local,
type-agnostic slots:

```text
z_ijfn = 1 if slot n of route (i,j) uses type f
sum_f z_ijfn <= 1
```

The loading variable is then indexed by `(route, slot, SKU)` rather than
`(route, SKU, global vehicle)`.

**Evidence required:**

- the legacy and new models have equal optimal objective on a tiny instance;
- the route-local slot count is sufficient for every non-dominated plan;
- the refreshed model size and solve results are recorded;
- both Gurobi and CPLEX implementations use the same reformulation.

### 1.2 Vehicle-type mix must be a decision, not a random parameter

> **Status (2026-10-16).** The first evidence item below is now satisfied: the
> loader seeds its own draw (`random.seed(self.seed)`, default `20260408`), so
> two runs of the same benchmark solve the same model. The route-local
> monolithic models already choose the type per slot through `z_ijfn`; the
> loader's fleet fields remain only as legacy compatibility data (see
> `model_implementation_audit.md` §7.2).

**Symptom:** Two runs of the same benchmark can solve different models.
Reported objectives, bounds, and run times are not comparable.

**Root cause:** The legacy loader assigns a random type to each vehicle
identity before optimization. The Gurobi paths did not seed that draw, and the
model could not change a vehicle's type.

**Remediation:** Let the model choose the type for each activated route slot
through `z_ijfn`. Do not use the randomly generated fleet fields in the
route-local monolithic models.

**Evidence required:**

- a fixed seed or no random draw at all;
- the realized fleet fields are documented as legacy compatibility data if
  they remain in the loader;
- the Gurobi and CPLEX models produce identical objectives on the same tiny
  instance;
- the benchmark result can be reproduced from a clean checkout.

### 1.3 A minimum-load penalty is a per-vehicle cost

**Symptom:** Aggregating vehicles by route or type can make the model smaller
but can also lose the number of underloaded vehicles, changing the objective.

**Root cause:** The penalty is charged per activated vehicle whose loaded
volume is below its type threshold. A route-level aggregate does not determine
how many individual vehicles are underloaded.

**Remediation:** Keep one binary penalty indicator per route-local slot and
the selected type:

```text
z_ijfn + zz_ijn - 1 <= p_ijfn
```

**Evidence required:**

- an underloaded vehicle is penalized exactly once;
- an unused slot has no penalty;
- the objective can be recomputed from the output CSV files;
- the same accounting is used by monolithic, column generation, and ALNS.

### 1.4 Time-limit status is not optimality

**Symptom:** A solver reports a feasible objective, but the model is not
proven optimal within the time budget.

**Root cause:** MIP solvers can return a primal incumbent without closing the
dual gap. `time_limit_feasible` is not an optimum certificate.

**Remediation:** Always record the objective, best bound, gap, status, wall
time, and time limit separately.

**Evidence required:**

- the status and solver status code are saved;
- the gap is computed from the objective and bound;
- a zero-gap result is described as optimal only when the solver status says
  so;
- tiny equivalence tests use proven optimal solves rather than time-limit
  incumbents.

#### 1.4.1 Corollary: do not gate *result extraction* on optimality
A time-limited run still has a usable incumbent, so treating `OPTIMAL` as the
only status that may write the solution throws away the very result the run
produced - and does so precisely on the medium and large instances that reach
the time limit.

**Symptom (observed 2026-10-16):** the run was recorded as successful, but
`result.csv` contained only its header. Downstream, `result_state()` then read
that file and raised `KeyError: 'Index'`.

**Root cause:** two gates disagreed. `run_experiments.py` accepted
`time_limit` as a successful status, while
`optLogicMultiple.create_dataframe()` read variable values only when
`model.model.Status == GRB.OPTIMAL`. Gurobi's own search path had already been
exporting the incumbent for `TIME_LIMIT`/`INTERRUPTED`, so only the CSV was
lost.

**Remediation:** gate extraction on the presence of an incumbent, not on
proven optimality - `SolCount > 0`, mirroring what the solution-export path
already does - and skip individual values that are undefined for the current
status rather than raising. Statuses with no solution (infeasible, unbounded,
inf-or-unbounded) still produce no rows.

**Evidence required:**

- a test proves rows are extracted for a time-limit status with `SolCount > 0`;
- a test proves no rows are written when `SolCount == 0` or `SolCount` is absent;
- unreadable values (`NaN`, `inf`, `None`) are skipped instead of raising.

Regression tests:
`multi-period-delivery/tests/test_monolithic_result_output.py`.

#### 1.4.2 Do not mistake a swallowed solver error for a successful run

**Symptom (observed 2026-10-16):** a result file records
`status: "solver_error: CPLEX Error 1016: Community Edition."` together with
`objective: null`, yet the run is counted as successful - "it ran, the
objective is just missing".

**Root cause:** two individually defensible decisions combine badly. The solve
wrapper deliberately catches solver-level failures so one bad instance cannot
abort a batch, and the experiment harness accepted an allow-list of status
strings. A failure the harness had never seen before fell outside the list only
by accident; a failure whose status string looked benign would have been
accepted.

**Remediation:** have the producer state the outcome explicitly rather than let
consumers infer it from a status string. The CPLEX monolithic paths now return a
`solved` boolean derived from a shared `SOLVED_STATUSES` set, and the harness
prefers that field when present (falling back to the status allow-list only for
producers that predate it).

**Evidence required:**

- a solver-level failure produces `solved: false` and a non-null status;
- the harness records a failure for such a run;
- the failure status is distinguishable from every solved status.

Also release native solver objects: the CPLEX model was never ended, so every
solve leaked a handle - which matters in exactly the batch runs the catch was
written to protect, because CPLEX concurrency is licence-bound.

Regression tests: `multi-period-delivery/tests/test_p1_fixes.py`.

#### 1.4.3 Phase schedules must be measured against the budget, not an iteration ceiling

**Symptom (observed 2026-10-16):** phase-dependent search policies never
activated: actual-priority acceptance, late fleet priority, light
intensification, elite relinking and ML takeover were all dead code.

**Root cause:** those policies gate on `progress`, which was computed as
`iteration / max_iterations`, where `max_iterations` is a runtime-derived
ceiling `MAX_RUNTIME x iterations_per_second x safety_factor + buffer`. The
configured `iterations_per_second` was `1000.0`, an assumption three orders of
magnitude above measured throughput (0.1-40 iterations/second on the bundled
instances). The ceiling became 900,200 iterations, so `progress` stayed around
0.001 and no threshold in the 0.20-0.70 range could be reached.

**A first attempt, and why it was wrong.** The obvious repair was to calibrate
that constant to measured throughput. That is a dead end: it converts a
*statement about the run* ("the last 30% of the budget") into a statement about
the *host and instance* ("this many iterations"), and measured throughput varies
by two orders of magnitude across instance sizes on one machine - and again
across machines. Worse, sizing the ceiling to measured throughput makes the
ceiling **bind**: on `small/dataset_1` a value of `5.0` capped the run at 4,700
iterations, which it reached after ~152 s of a 600 s budget, so progress peaked
at ~0.25 and the 0.60/0.70 gates still never fired. A ceiling derived from
throughput cannot both be generous enough to let the schedule finish and tight
enough to be meaningful.

**Remediation:** measure progress against **elapsed runtime / `MAX_RUNTIME`**
(`alns/progress.py`), which is machine-independent by construction - a host
twice as fast executes twice the iterations in the same fraction of the budget -
and demote the iteration ceiling to a pure safety net that must not bind in
normal operation.

Four call sites shared the same faulty measure and now share one implementation:
`accept.DemandConstraintAccept` (phase gates), `ml_operator_selector`
(ML takeover **and** the epsilon-greedy exploration decay), `callbacks`
(the light-intensify window) and `param_tuner.ParamAutoTuner` (destroy-strength
decay).

**Corollary: one measure means no fallback measure.** Keeping an
iteration-based path "in case time is unavailable" looked safe and was not: two
measures meant two behaviours, and the fallback could silently reinstate the
machine-dependent schedule this lesson is about. The measure now has exactly one
definition and one failure mode, and both are loud:

- a tracker that exposes no clock raises `AttributeError` (a programming error);
- a non-positive `MAX_RUNTIME` raises `ValueError` (a configuration error).

Falling back was the wrong instinct: an unreachable branch still has to be read
and maintained, and a reachable one hides a defect.

**Corollary: one clock means one origin.** Two runtime measurements existed -
the stopping rule's stopwatch and the tracker's - starting at different moments
(optimization start versus first stop-rule evaluation, i.e. separated by data
loading and model build). Both were "machine-independent" and still put the same
gate at two different moments. The run now has a single stopwatch: the optimizer
hands it to the tracker and to the parameter tuner, and passes the origin into
the stopping rule.

Note the deliberate clock split, which is *not* the same thing: wall-clock time
(`time.time()`) is still used for log display, while the runtime budget uses the
monotonic clock (`time.perf_counter`) so a wall-clock step cannot shift the
phase schedule.

**Evidence required:**

- progress equals elapsed-runtime / `MAX_RUNTIME` on a real tracker;
- every documented gate is reachable within one runtime budget;
- the same elapsed point reports the same progress for hosts whose iteration
  counts differ by orders of magnitude;
- the iteration ceiling does **not** bind before `MAX_RUNTIME` at measured
  throughput (otherwise the run is truncated and later gates are skipped);
- the ceiling still respects its formula and never undercuts `MAX_ITERATIONS`;
- a tracker without a clock, and a non-positive `MAX_RUNTIME`, both raise;
- the stopping rule, the tracker and the parameter tuner report the same
  elapsed seconds in a real run (verified: agreement within 0.00 s).

Regression tests: `multi-period-delivery/tests/test_progress_measure.py`,
`test_phase_progress.py`, `test_p1_fixes.py`.

#### 1.4.4 A time-based measure must not be asserted with float equality

**Symptom (CI failure, 2026-10-16):** a new test asserted that a
just-started run reports progress `== 0.0` and failed on CI with

```
assert 7.54992167154948e-09 == 0.0
```

**Root cause:** time elapses between building the tracker and reading the
clock, so "just started" is a few nanoseconds of progress, not exactly zero.
The production clamp was correct; the assertion was not.

**Why local runs passed:** the suite was run 15 times locally without a failure.
The CI runner was faster (that step took 29 s versus 71 s locally), which
widened the gap between construction and measurement enough to expose it. A
green local run is not evidence that a timing-sensitive assertion is sound.

**Remediation:** assert the property, not a bit pattern - a lower bound plus a
tolerance (`0.0 <= progress < 1e-6`) or `pytest.approx` - and reserve exact
equality for values the code computes rather than measures. Where a test needs a
specific elapsed time, inject the clock (`set_elapsed_clock`) instead of relying
on wall-clock timing.

**Evidence required:**

- no assertion compares a measured duration to an exact float;
- tests that need a phase position set it explicitly;
- the timing-sensitive test passes when repeated in a loop.

Regression tests: `multi-period-delivery/tests/test_progress_measure.py`
(`test_progress_is_clamped`).

### 1.5 A bound variable is not automatically a valid plan

**Symptom:** A model reports a best bound lower than the incumbent but no
integer solution exists yet.

**Root cause:** A dual bound is not a delivery plan and cannot be written to
the result files.

**Remediation:** Distinguish `objective` from `best_bound`, and require
primal feasibility before exporting result rows.

**Evidence required:**

- no result CSV is produced from a bound alone;
- every exported shipment has a positive integer quantity;
- demand, inventory, capacity, and minimum-load validation passes.

### 1.6 Over-delivery is part of the model

**Symptom:** Reviewers may assume the demand constraint is an equality and
incorrectly flag over-delivery as a bug.

**Root cause:** The final thesis and implementation use a coverage inequality.
This allows an already activated vehicle to be topped up to the
minimum-load threshold without activating another vehicle.

**Remediation:** State the inequality explicitly in the model documentation
and validation reports.

**Evidence required:**

- the symbol table and model equations use `>=` consistently;
- over-delivery is reported separately from shortages;
- the validator distinguishes a shortage from legal over-delivery.

### 1.7 An objective value must apply its objective coefficients

**Symptom (observed 2026-10-16):** the column-generation master model printed a
`RealObj` that looked plausible but was a column count. Two pattern columns of
value 1 were reported as `2.0` regardless of what they cost.

**Root cause:** the value was computed as `sum(var.X for var in ...)`, which
sums solution values with an implicit coefficient of one. In a priced
formulation the coefficients carry all the economics, so the sum is only the
objective when every coefficient happens to be 1.

**Remediation:** compute the value from the variables and their coefficients
(`var.X * var.Obj`), which also picks up the artificial penalty terms. Keep the
derivation in a named method so it can be unit-tested without a solver licence.

**Evidence required:**

- two equal-valued columns with different coefficients produce the priced sum;
- unused columns contribute nothing;
- the printed summary changes when a coefficient changes.

Regression tests: `single-period-delivery/tests/test_master_objective.py`.

---

## 2. Data and reproducibility lessons

### 2.1 Synthetic data must be clearly separated from commercial data

**Symptom:** A public repository can unintentionally expose a real customer
network, parameters, or business identifiers.

**Root cause:** The model may be derived from a real case even when the
published numbers are simulated.

**Remediation:** Publish only synthetic instances and generic operational
structure. Keep company names, real parameters, confidential documents, and
potentially identifying network details out of the tracked repository.

**Evidence required:**

- data generation is deterministic from a documented seed;
- the public README states that the data are simulated;
- no real company names, credentials, or private files are committed;
- the confidentiality boundary is reviewed before publishing a license.

### 2.2 Relative paths must be resolved against a documented base

**Symptom:** A script works from one directory and fails from another, or
writes results into an unexpected nested folder.

**Root cause:** The script resolves relative paths against the module location
while the user expects them relative to the current working directory, or vice
versa.

**Remediation:** Define the base directory explicitly in the CLI, and test the
documented invocation from the repository root.

**Evidence required:**

- the same command produces the same output location from a clean checkout;
- the path-resolution rule is stated in the README;
- no generated results are written inside a package source folder.

### 2.3 Raw and curated results have different retention rules

**Symptom:** Large logs and solver dumps are committed, or a curated result
cannot be traced back to its run.

**Root cause:** Raw run artifacts and committed summary tables have different
purposes.

**Remediation:** Keep raw runs under ignored `results/*/raw/`; commit only
small, deterministic summaries and the command that regenerates them.

**Evidence required:**

- the raw directory is ignored;
- the summary has a stable schema;
- the report builder can regenerate the comparison table.

---

## 3. CI and quality lessons

### 3.1 Duplicate test filenames require an explicit import mode

**Symptom:** Running pytest from the repository root imports the wrong
`test_cli.py` or `test_cplex_api.py`.

**Root cause:** Both packages have tests with identical basenames.

**Remediation:** Configure `--import-mode=importlib` at the workspace root and
restrict collection to the two package test directories.

**Evidence required:**

- `pytest -q` from the repository root collects both projects;
- generated folders such as `outputs/` are not scanned;
- the CI command uses the same configuration as local development.

### 3.2 Optional solvers should skip, not fail

**Symptom:** CI fails because CPLEX or Gurobi is not installed.

**Root cause:** Solver-dependent tests are collected in the normal test suite,
while the optional runtime is intentionally absent from CI.

**Remediation:** Use `pytest.importorskip` for optional CPLEX tests and keep
the standard CI environment solver-free where possible.

**Evidence required:**

- skipped tests are visible in the summary;
- solver-independent paths still run in CI;
- the documented local command exercises the optional solver path.

### 3.3 Coverage gates must be meaningful but not brittle

**Symptom:** A coverage limit either blocks ordinary refactoring or is so low
that it provides no signal.

**Root cause:** Coverage is a baseline metric, not a proof of correctness.

**Remediation:** Measure branch and statement coverage over the two packages,
set the gate below the measured baseline, and raise it only after real
coverage improves.

**Evidence required:**

- the current baseline is recorded;
- CI uses the same command as the documented local check;
- solver-dependent modules are explained when they are not covered.

### 3.4 Lint rules should be adopted in controlled groups

**Symptom:** Turning on the full Ruff rule set creates hundreds of unrelated
changes and hides real defects.

**Root cause:** A clean-code baseline is being introduced into an existing
research codebase.

**Remediation:** Enable focused groups first (`E4`, `E7`, `E9`, `F`, `I`,
`UP`) and expand only when the current gate is clean.

**Evidence required:**

- the rule set is declared in each package;
- CI runs the same lint command;
- fixes are separated from behavior-changing refactors where practical.

### 3.5 uv lock and workspace pruning are features, not surprises

**Symptom:** A package installed manually disappears after `uv sync`.

**Root cause:** `uv sync` makes `.venv` match the lock file and prunes
undeclared packages.

**Remediation:** Declare every required dependency in the correct workspace
member, use the `cplex` extra when needed, or explicitly use `uv sync --inexact`
only for a deliberate local exception.

**Evidence required:**

- `pyproject.toml` and `uv.lock` agree;
- clean `uv sync` installs both members;
- the optional CPLEX flow is documented separately.

### 3.6 CPLEX Community Edition has a size limit

**Symptom:** A modest model fails with CPLEX error `1016`.

**Root cause:** The PyPI CPLEX wheel contains the Community runtime, which
limits model size.

**Remediation:** Install the matching Studio runtime and run
`docplex config --upgrade`, ensuring the wheel and runtime versions match.

**Evidence required:**

- the upgrade command is documented;
- the installed wheel and DLL versions match;
- a model beyond the Community limit is constructed successfully.

### 3.7 Gurobi license state must be explicit

**Symptom:** Gurobi fails with an error about a missing license file, or a
restricted license is used without the solver run being described correctly.

**Root cause:** Gurobi license lookup depends on environment variables and the
installed license type.

**Remediation:** Record the license environment used for local verification;
do not commit license files or credentials. Do not use a restricted license to
claim production-scale performance.

**Evidence required:**

- the validation report states the license mode;
- tiny tests are solved under the restricted mode when that is sufficient;
- large benchmark claims are made only with the appropriate runtime.

---

## 4. Git and repository hygiene lessons

### 4.1 Generated model dumps and logs belong in ignored output directories

**Symptom:** LP, ILP, console, and raw experiment files pollute the repository.

**Root cause:** Solver APIs write their own diagnostic artifacts by default.

**Remediation:** Write solver artifacts under the run output directory and
ignore `*.lp`, `*.ilp`, `*.log`, and raw result directories.

**Evidence required:**

- `git status` is clean after a solve;
- generated artifacts can be recreated by the documented command.

### 4.2 Line endings should be declared once

**Symptom:** Git reports the entire file as changed because of CRLF/LF churn.

**Root cause:** Windows editors and Git defaults can disagree.

**Remediation:** Keep a repository-level `.gitattributes` and avoid converting
line endings as part of unrelated fixes.

**Evidence required:**

- `git diff --check` is clean;
- line-ending warnings are understood before commit;
- behavioral diffs are not mixed with formatting-only churn.

### 4.3 Local-only material must remain local

**Symptom:** Interview notes, private resume drafts, or sensitive setup notes
are accidentally committed.

**Root cause:** Personal notes are stored next to the public codebase.

**Remediation:** Keep them under an ignored `local-only/` directory and verify
`git status --ignored` before publishing.

**Evidence required:**

- `local-only/` is ignored;
- public docs contain no private credentials or company data;
- local notes can still reference public files by relative path.

---

## 5. Mathematical documentation lessons

### 5.1 Keep legacy and new formulations side by side

The document should show the thesis model, the route-local model, and the
equivalence argument. A reader must be able to see what changed without
reverse-engineering the code.

Required sections:

- symbol tables;
- objective;
- every constraint and its business meaning;
- variable domains;
- legacy defects and motivation;
- new-model advantages;
- validation evidence;
- implementation file map.

### 5.2 Markdown math needs renderer-compatible inline syntax

Use `$...$` for inline math and `$$...$$` for display math. Do not mix
renderer-specific syntax such as `\(...\)` if the target renderer treats it as
plain text.

**Verification:** render the document in the target Markdown viewer and check
that set membership, subscripts, Greek letters, and fractions appear as
mathematical symbols rather than raw TeX.

### 5.3 Every result claim needs a provenance trail

For each headline result, record:

- the instance or whether it is synthetic;
- the solver and version;
- the seed;
- the time limit;
- the status;
- the objective and best bound;
- the source CSV or JSON;
- the command that regenerates it.

Without this trail, a result is a claim rather than evidence.

---

## 6. Final review checklist

Before describing the portfolio as complete, answer all of these:

1. Can I explain the legacy model, the route-local model, and the exact
   equivalence argument without reading code?
2. Can I explain why the old fleet index grows and why the new slot count is
   sufficient?
3. Can I explain why minimum-load penalties must remain per vehicle?
4. Can I explain the difference between an incumbent, a best bound, and an
   optimal solution?
5. Can I reproduce one small CPLEX model, one Gurobi model, and one ALNS run?
6. Can I run Ruff, pytest, coverage, and the CI workflow locally?
7. Can I explain what `uv sync`, the lock file, and the CPLEX extra do?
8. Can I explain the CPLEX Community/Studio upgrade and why it matters?
9. Can I identify every generated artifact that must not be committed?
10. Can I point to a document and a command for every line in my resume?
