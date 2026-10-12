# Shared uv Environment

The portfolio uses one shared virtual environment for both subprojects:

```text
supply-chain-optimization-portfolio/.venv
```

The environment uses Python 3.10 because the official CPLEX Python API wheel
used here (`cplex==22.1.2.0`) is available for Python 3.10. Both packages are
installed in editable mode from the same environment.

## Create the Environment

Run from the portfolio root:

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

## Continuous Integration

`.github/workflows/ci.yml` runs the same checks on GitHub Actions for every
push to `main` and for every pull request:

1. install [uv](https://docs.astral.sh/uv/) and create the shared Python 3.10
   environment with `uv venv`;
2. install both packages in editable mode plus `pytest` and `ruff`:

   ```text
   uv pip install --python .venv -e ./single-period-delivery -e ./multi-period-delivery pytest ruff
   ```

3. `ruff check` each project - each one declares its own baseline rule set in
   `[tool.ruff.lint]` inside its `pyproject.toml`;
4. run the combined test suite (`pytest -q`, 30 cases).

The optional `cplex` extra is deliberately **not** installed in CI: the CPLEX
tests skip themselves through `pytest.importorskip`, and the Community Edition
runtime carries a size limit that would make CI runs flaky.

The repository is not a single uv project (there is no root `pyproject.toml`
and no `uv.lock`), so CI uses uv's pip-compatible interface instead of
`uv sync`. Moving the two packages into a uv workspace with a committed lock
file is the next step towards a one-command, fully reproducible environment.

## Network and Proxy Note

If `uv pip install` fails with `No connection could be made`, a stale
`HTTP_PROXY`/`HTTPS_PROXY` environment variable may be pointing at a proxy that
is not running. Clearing the proxy variables for the command restores direct
PyPI access:

```powershell
$env:HTTP_PROXY = ""; $env:HTTPS_PROXY = ""; $env:ALL_PROXY = ""
uv pip install --python .venv numba
```
