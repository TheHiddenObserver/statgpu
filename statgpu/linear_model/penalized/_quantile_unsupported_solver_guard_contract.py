"""Fail-closed public boundaries for unsupported explicit Quantile solvers.

The maintained Quantile estimator routes are deliberately narrower than the
set of generic optimization engines exported by statgpu. In particular:

* FISTA-BB relies on smooth-gradient differences for BB curvature estimates;
* L-BFGS is a smooth quasi-Newton/line-search method;
* shared ADMM uses accelerated gradient descent for its w-subproblem.

Quantile/check loss has a step-function subgradient, so these explicit requests
must fail before backend or CV numerical work. Ordinary FISTA remains the
maintained sparse convex route; L2/no-penalty uses IRLS and SCAD/MCP use the
dedicated Proximal IRLS-CD route.
"""

from __future__ import annotations

from functools import wraps

from . import _quantile_solver_contract as _quantile_contract


_MARKER = "_statgpu_quantile_unsupported_solver_guard_contract"
_UNSUPPORTED = frozenset({"admm", "fista_bb", "lbfgs"})


def install_quantile_unsupported_solver_guard_contract() -> None:
    current = _quantile_contract._validate_quantile_solver_request
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def _validate_without_unsupported_quantile_solvers(
        *,
        loss_name,
        penalty_name,
        solver_name,
        allow_internal_nonconvex=False,
    ):
        resolved_loss = _quantile_contract._loss_name(loss_name)
        resolved_solver = str(solver_name or "").lower().strip()
        if resolved_loss == "quantile" and resolved_solver in _UNSUPPORTED:
            if resolved_solver == "fista_bb":
                reason = (
                    "Barzilai-Borwein step sizes require meaningful "
                    "smooth-gradient differences"
                )
            elif resolved_solver == "lbfgs":
                reason = (
                    "the quasi-Newton curvature and line-search contract "
                    "requires a smooth objective"
                )
            else:
                reason = (
                    "the shared ADMM w-update uses accelerated gradient descent "
                    "and requires a smooth loss gradient"
                )
            raise ValueError(
                f"solver='{resolved_solver}' is not a maintained Quantile route: "
                f"{reason}, while Quantile loss is non-smooth. Use ordinary "
                "FISTA for maintained sparse convex routes, solver='auto'/'irls' "
                "for L2 or no penalty, or solver='auto' for SCAD/MCP."
            )
        return current(
            loss_name=loss_name,
            penalty_name=penalty_name,
            solver_name=solver_name,
            allow_internal_nonconvex=allow_internal_nonconvex,
        )

    setattr(_validate_without_unsupported_quantile_solvers, _MARKER, True)
    _validate_without_unsupported_quantile_solvers._statgpu_original = current
    _quantile_contract._validate_quantile_solver_request = (
        _validate_without_unsupported_quantile_solvers
    )


install_quantile_unsupported_solver_guard_contract()
