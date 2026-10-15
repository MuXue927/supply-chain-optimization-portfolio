"""CPLEX Python API support for the alternative single-period models.

This module holds the small model-builder interface shared by the single-period
CPLEX Python API models. It sends a model directly to the official ``cplex``
package.

The module deliberately contains only model construction, solve, and result
extraction helpers. Mathematical formulations remain in the ``cplex_api_*``
algorithm folders.
"""

from __future__ import annotations

import io
import re
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Number = int | float
ConstraintSense = Literal["<=", ">=", "="]

#: Statuses that carry a usable primal solution. A solver-level failure is
#: reported as ``solver_error: ...`` (see ``CplexApiModel.solve``); callers must
#: not treat that as a successful run merely because the process did not crash.
SOLVED_STATUSES: frozenset[str] = frozenset(
    {"optimal", "time_limit_feasible", "feasible"}
)


def _load_cplex() -> Any:
    """Import the official CPLEX Python package with an actionable error."""
    try:
        import cplex
    except ImportError as exc:
        raise ImportError(
            "The official IBM CPLEX Python API is required. Install the "
            "``cplex`` package in the active uv environment, for example "
            "``uv pip install cplex``."
        ) from exc
    return cplex


def _safe_name(name: str) -> str:
    """Convert an internal identifier to a CPLEX-safe symbol name."""
    value = re.sub(r"[^0-9A-Za-z_]", "_", str(name))
    if not value or not (value[0].isalpha() or value[0] == "_"):
        value = f"v_{value}"
    return value


@dataclass
class CplexApiSolution:
    """Solution and solver metadata returned by the CPLEX Python API."""

    status: str
    objective: float | None
    variables: dict[str, float] = field(default_factory=dict)
    duals: dict[str, float] = field(default_factory=dict)
    best_bound: float | None = None
    mip_gap: float | None = None
    solver_status: str = ""
    solver_status_code: int | None = None
    stdout: str = ""


class CplexApiModel:
    """Small deterministic model builder backed by the CPLEX Python API."""
    def __init__(
        self,
        name: str = "model",
        *,
        sense: Literal["min", "max"] = "min",
    ) -> None:
        self.name = name
        self.sense = sense
        self.objective_terms: MutableMapping[str, float] = {}
        self.constraints: list[
            tuple[str, dict[str, float], ConstraintSense, float]
        ] = []
        self.variables: MutableMapping[str, tuple[float, float, str]] = {}

    def add_variable(
        self,
        name: str,
        *,
        lb: float = 0.0,
        ub: float | None = None,
        kind: Literal["continuous", "binary", "integer"] = "continuous",
    ) -> str:
        """Create or validate a variable and return its safe symbol name."""
        safe = _safe_name(name)
        if kind == "binary":
            lb, ub = 0.0, 1.0
        elif ub is None:
            ub = float("inf")
        current = (float(lb), float(ub), kind)
        previous = self.variables.get(safe)
        if previous is not None and previous != current:
            raise ValueError(
                f"Variable {safe!r} was declared with conflicting definitions: "
                f"{previous!r} vs {current!r}"
            )
        self.variables[safe] = current
        return safe

    def set_objective(self, terms: Mapping[str, Number]) -> None:
        """Replace the objective with the supplied linear terms."""
        self.objective_terms = {
            variable: float(coefficient)
            for variable, coefficient in terms.items()
            if abs(float(coefficient)) > 1e-15
        }

    def add_objective_term(self, variable: str, coefficient: Number) -> None:
        """Add one objective coefficient."""
        self.objective_terms[variable] = (
            self.objective_terms.get(variable, 0.0) + float(coefficient)
        )

    def add_constraint(
        self,
        name: str,
        terms: Mapping[str, Number],
        sense: ConstraintSense,
        rhs: Number,
    ) -> str:
        """Add one linear constraint and return its safe symbol name."""
        safe = _safe_name(name)
        self.constraints.append(
            (
                safe,
                {
                    variable: float(coefficient)
                    for variable, coefficient in terms.items()
                    if abs(float(coefficient)) > 1e-15
                },
                sense,
                float(rhs),
            )
        )
        return safe

    def solve(
        self,
        work_dir: str | Path | None = None,
        *,
        time_limit: float | None = None,
        mip_gap: float | None = None,
        random_seed: int | None = None,
        parameter_overrides: Mapping[str, object] | None = None,
        write_model_path: str | Path | None = None,
    ) -> CplexApiSolution:
        """Solve the model through the CPLEX Python API."""
        cplex = _load_cplex()
        if work_dir is not None:
            Path(work_dir).mkdir(parents=True, exist_ok=True)
        if write_model_path is not None:
            Path(write_model_path).parent.mkdir(parents=True, exist_ok=True)
        problem = cplex.Cplex()
        log_stream = io.StringIO()
        problem.set_log_stream(log_stream)
        problem.set_results_stream(log_stream)
        problem.set_warning_stream(log_stream)
        problem.set_error_stream(log_stream)

        safe_names = list(self.variables)
        lower_bounds = []
        upper_bounds = []
        variable_types = []
        for name in safe_names:
            lower, upper, kind = self.variables[name]
            lower_bounds.append(lower)
            upper_bounds.append(
                cplex.infinity if upper == float("inf") else upper
            )
            variable_types.append(
                {
                    "continuous": problem.variables.type.continuous,
                    "binary": problem.variables.type.binary,
                    "integer": problem.variables.type.integer,
                }[kind]
            )

        try:
            problem.variables.add(
                lb=lower_bounds,
                ub=upper_bounds,
                types=variable_types,
                names=safe_names,
            )
        except cplex.exceptions.CplexError as exc:
            raise ValueError(
                "Unable to create CPLEX variables. Check for duplicate or "
                "invalid variable symbols."
            ) from exc

        name_to_index = {
            name: index for index, name in enumerate(safe_names)
        }
        problem.objective.set_sense(
            problem.objective.sense.minimize
            if self.sense == "min"
            else problem.objective.sense.maximize
        )
        problem.objective.set_linear(
            [
                (name_to_index[variable], coefficient)
                for variable, coefficient in self.objective_terms.items()
            ]
        )

        sense_map = {"<=": "L", ">=": "G", "=": "E"}
        expressions = []
        senses = []
        rhs_values = []
        constraint_names = []
        for name, terms, sense, rhs in self.constraints:
            indices = []
            coefficients = []
            for variable, coefficient in terms.items():
                indices.append(name_to_index[variable])
                coefficients.append(coefficient)
            expressions.append(cplex.SparsePair(ind=indices, val=coefficients))
            senses.append(sense_map[sense])
            rhs_values.append(rhs)
            constraint_names.append(name)
        if expressions:
            problem.linear_constraints.add(
                lin_expr=expressions,
                senses=senses,
                rhs=rhs_values,
                names=constraint_names,
            )

        if time_limit is not None:
            problem.parameters.timelimit.set(float(time_limit))
        if mip_gap is not None:
            problem.parameters.mip.tolerances.mipgap.set(float(mip_gap))
        if random_seed is not None:
            problem.parameters.randomseed.set(int(random_seed))
        for parameter_name, value in (parameter_overrides or {}).items():
            target: Any = problem.parameters
            for part in parameter_name.split("."):
                target = getattr(target, part)
            target.set(value)

        if write_model_path is not None:
            problem.write(str(Path(write_model_path)))

        # A solver-level failure (for example CPLEX Error 1001, out of memory on
        # a large model) must not crash the whole run. Report it as a status and
        # let the caller record a result without a primal solution.
        try:
            problem.solve()
        except Exception as exc:
            # Release the native model before reporting, otherwise a failing
            # solve leaks its CPLEX handle.
            try:
                problem.end()
            except Exception:
                pass
            return CplexApiSolution(
                status=f"solver_error: {exc}",
                objective=None,
                variables={},
                duals={},
                best_bound=None,
                mip_gap=None,
                solver_status=str(exc),
                solver_status_code=None,
                stdout=log_stream.getvalue(),
            )
        try:
            status_text = problem.solution.get_status_string()
            status_lower = status_text.lower()
            if "integer optimal" in status_lower or status_lower == "optimal":
                status = "optimal"
            elif "time limit" in status_lower and "feasible" in status_lower:
                status = "time_limit_feasible"
            elif "integer infeasible" in status_lower:
                status = "integer_infeasible"
            elif "infeasible" in status_lower:
                status = "infeasible"
            elif "unbounded" in status_lower:
                status = "unbounded"
            else:
                status = status_lower

            try:
                objective = float(problem.solution.get_objective_value())
            except Exception:
                objective = None
            try:
                values = problem.solution.get_values()
            except Exception:
                values = []
            try:
                dual_values = problem.solution.dual.get_values()
            except Exception:
                dual_values = []
            try:
                best_bound = float(problem.solution.MIP.get_best_objective())
            except Exception:
                best_bound = None
            try:
                mip_gap = float(problem.solution.MIP.get_mip_relative_gap())
            except Exception:
                mip_gap = None
            try:
                solver_status_code = int(problem.solution.get_status())
            except Exception:
                solver_status_code = None

            variable_values = {
                name: float(values[index])
                for index, name in enumerate(safe_names)
                if index < len(values)
            }
            dual_map = {
                name: float(dual_values[index])
                for index, name in enumerate(constraint_names)
                if index < len(dual_values)
            }
            return CplexApiSolution(
                status=status,
                objective=objective,
                variables=variable_values,
                duals=dual_map,
                best_bound=best_bound,
                mip_gap=mip_gap,
                solver_status=status_text,
                solver_status_code=solver_status_code,
                stdout=log_stream.getvalue(),
            )
        finally:
            # Release the native CPLEX model. Without this every solve leaks a
            # solver handle, which accumulates across the batch experiments (and
            # can exhaust CPLEX's concurrent-licence pool). Cleanup must not mask
            # the extracted solution, so failures here are ignored.
            try:
                problem.end()
            except Exception:
                pass
