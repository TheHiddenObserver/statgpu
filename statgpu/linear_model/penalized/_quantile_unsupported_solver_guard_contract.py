"""Fail-closed public boundaries for unsupported explicit Quantile solvers.

The maintained Quantile estimator routes are deliberately narrower than the
set of generic optimization engines exported by statgpu. This follow-up adds
only the missing all-penalty boundaries for:

* FISTA-BB, which relies on smooth-gradient differences for BB curvature;
* shared ADMM, whose generic w-subproblem uses accelerated gradient descent.

Existing Quantile validation remains authoritative first. That preserves the
more specific historical errors for smooth FISTA-BB, SCAD/MCP dedicated
Proximal IRLS-CD routing, L-BFGS, Newton, and other already-unsupported rows.
Only requests that the existing validator accepted can reach the new guard.
"""

from __future__ import annotations

from functools import wraps

from . import _quantile_solver_contract as _quantile_contract


_MARKER = "_statgpu_quantile_unsupported_solver_guard_contract"
_UNSUPPORTED = frozenset({"admm", "fista_bb"})


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
        # Preserve every existing Quantile rejection and its more specific
        # public error semantics. The new guard only closes rows that the
        # established validator previously allowed.
        current(
            loss_name=loss_name,
            penalty_name=penalty_name,
            solver_name=solver_name,
            allow_internal_nonconvex=allow_internal_nonconvex,
        )

        resolved_loss = _quantile_contract._loss_name(loss_name)
        resolved_solver = str(solver_name or "").lower().strip()
        if resolved_loss != "quantile" or resolved_solver not in _UNSUPPORTED:
            return None

        if resolved_solver == "fista_bb":
            reason = (
                "Barzilai-Borwein step sizes require meaningful "
                "smooth-gradient differences"
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

    setattr(_validate_without_unsupported_quantile_solvers, _MARKER, True)
    _validate_without_unsupported_quantile_solvers._statgpu_original = current
    _quantile_contract._validate_quantile_solver_request = (
        _validate_without_unsupported_quantile_solvers
    )


install_quantile_unsupported_solver_guard_contract()
