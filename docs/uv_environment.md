# Shared uv Environment

The portfolio is one [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/):
the repository root declares both packages as members, pins the interpreter in
`.python-version` and locks every dependency in `uv.lock`. Everything installs
into a single virtual environment:

```text
supply-chain-optimization-portfolio/.venv
```

The interpreter is pinned to Python 3.10 because the official CPLEX Python API
wheel used here (`cplex==22.1.2.0`) only ships wheels for CPython 3.10-3.12.
Both packages are installed in editable mode from the same environment.

## Create the Environment

Run from the portfolio root:

```powershell
uv sync
```

That single command creates `.venv`, installs both workspace members in
editable mode, and installs the root dev group (`pytest`, `ruff`) at the
versions recorded in `uv.lock`. uv reads `.python-version` (3.10) to pick the
interpreter, so the CPLEX wheel stays installable for anyone who asks for it.

### Optional: the CPLEX runtime

```powershell
uv sync --extra cplex
.\.venv\Scripts\docplex.exe config --upgrade "C:\Program Files\IBM\ILOG\CPLEX_Studio2212"
```

The second command is required after any sync that reinstalls the `cplex`
wheel: the Community Edition runtime replaces the size-unlimited one that the
upgrade copies in (see the next section).

> **`uv sync` prunes whatever the lock does not list.** Packages that were
> installed into `.venv` by hand are removed on the next sync. Declare them,
> pass `--extra cplex`, or use `uv sync --inexact` to keep unrelated packages.

### The earlier uv-pip flow (kept for reference)

```powershell
$env:UV_CACHE_DIR = "$PWD\outputs\work\uv-cache"
uv venv --python 3.10 .venv
uv pip install --python .venv `
  numpy numba pandas matplotlib seaborn rich `
  alns scikit-learn openpyxl networkx plotly vrplib `
  pytest ruff setuptools wheel
uv pip install --python .venv cplex==22.1.2.0 docplex==2.32.264
uv pip install --python .venv --no-build-isolation --no-deps `
  -e single-period-delivery `
  -e multi-period-delivery
```

This path still works, but it does not pin versions and it needs
`setuptools`/`wheel` inside the environment for the editable installs.

## CPLEX Community Edition Size Limit

The PyPI `cplex` package ships the Community Edition runtime, which refuses
models beyond roughly 1,000 variables or constraints with CPLEX error `1016`.
The official way to remove that limit is the `docplex config --upgrade`
command, which copies the size-unlimited runtime from a locally installed
CPLEX Studio into the active Python environment:

```powershell
.\.venv\Scripts\docplex.exe config --upgrade "C:\Program Files\IBM\ILOG\CPLEX_Studio2212"
```

This copies `cplex2212.dll` into both the `cplex` package directory and
`.venv\Scripts`, and `cpoptimizer.exe` into `.venv\Scripts`. It was verified
against the installed Studio 22.1.2 runtime: after the upgrade, a 3,000-variable
LP solves normally instead of raising `1016`.

Two things must line up for the upgrade to work:

- The installed `cplex` wheel and the installed CPLEX Studio must share the same
  version. `cplex==22.1.2.0` matches Studio 22.1.2 (runtime library
  `cplex2212.dll`). Installing `cplex==22.1.0.0` makes `docplex` look for
  `cplex2210.dll`, which Studio 22.1.2 does not provide.
- `docplex` must be installed in the same environment, because the
  `docplex.exe` entry point is what ships the `config --upgrade` command.

No manual DLL copy is required. The upgrade is idempotent, so re-running it
after a `uv` dependency change that reinstalls the `cplex` wheel is safe.

> Superseded: an earlier version of this document copied
> `cplex2212.dll` over a bundled `cplex2210.dll` by hand. That workaround is no
> longer needed now that the wheel version matches the Studio version.

## Activate

```powershell
& .\.venv\Scripts\Activate.ps1
```

After activation, `python -m single_period_delivery.cplex_api_monolithic` and
`python -m multi_period_delivery.cplex_api_monolithic` use the Python API path.

## Running the Tests

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The root `pytest.ini` sets `--import-mode=importlib` (the two projects contain
tests with identical file names) and restricts collection to the two `tests/`
folders, so no extra flags are required and generated folders such as
`outputs/` are never scanned.

With coverage - the same command CI runs:

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  --cov=multi_period_delivery --cov=single_period_delivery `
  --cov-branch --cov-report=term-missing --cov-fail-under=54
```

`--cov-branch` reports branch coverage as well as statements, so a line that
runs but whose decision is never seen to go the other way no longer counts as
covered. The measured baseline is **55.36%** combined coverage over the two
packages with the four CPLEX tests skipping (the default, CI-equivalent
environment): 56.83% of statements and 50.50% of branches. The gate sits a
little below the combined figure so ordinary refactoring does not trip it;
raise it whenever coverage genuinely improves. The Gurobi-backed modules still
need a licence to solve, so those paths stay uncovered by design rather than
being mocked.

## Continuous Integration

`.github/workflows/ci.yml` runs the same checks on GitHub Actions for every
push to `main` and for every pull request:

1. install [uv](https://docs.astral.sh/uv/) and run `uv sync`, which reads
   `.python-version` and `uv.lock` and installs both workspace members plus the
   root dev group into `.venv`;
2. `ruff check` each project - each one declares its own baseline rule set in
   `[tool.ruff.lint]` inside its `pyproject.toml`;
3. run the combined test suite with coverage:
   `pytest -q --cov=multi_period_delivery --cov=single_period_delivery
   --cov-branch --cov-report=term-missing --cov-fail-under=54` - 186 passed +
   4 skipped, and
   55.36% coverage against the 54% gate (the CPLEX tests skip themselves when
   the optional extra is absent).

The optional `cplex` extra is deliberately **not** synced in CI: the CPLEX
tests skip themselves through `pytest.importorskip`, and the Community Edition
runtime carries a size limit that would make CI runs flaky.

## Network and Proxy Note

If `uv pip install` fails with `No connection could be made`, a stale
`HTTP_PROXY`/`HTTPS_PROXY` environment variable may be pointing at a proxy that
is not running. Clearing the proxy variables for the command restores direct
PyPI access:

```powershell
$env:HTTP_PROXY = ""; $env:HTTPS_PROXY = ""; $env:ALL_PROXY = ""
uv pip install --python .venv numba
```
