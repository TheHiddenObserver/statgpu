"""Fail-closed public Quantile boundary for the shared ADMM solver.

The shared ADMM implementation uses accelerated gradient descent for its
w-subproblem. Quantile/check loss has a non-smooth step-function subgradient,
so explicit Quantile ADMM requests are not a maintained route. Keep this guard
at the shared Quantile validation boundary so direct fits and CV reject the
request before backend or alpha-grid numerical work starts.
"""

from __future__ import annotations

from functools import wraps

from . import _quantile_solver_contract as _quantile_contract


_MARKER = "_statgpu_quantile_admm_guard_contract"


def install_quantile_admm_guard_contract() -> None:
    current = _quantile_contract._validate_quantile_solver_request
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def _validate_without_quantile_admm(
        *,
        loss_name,
        penalty_name,
        solver_name,
        allow_internal_nonconvex=False,
    ):
        resolved_loss = _quantile_contract._loss_name(loss_name)
        resolved_solver = str(solver_name or "").lower().strip()
        if resolved_loss == "quantile" and resolved_solver == "admm":
            raise ValueError(
                "solver='admm' is not a maintained Quantile route: the shared "
                "ADMM w-update requires a smooth loss gradient, while Quantile "
                "loss is non-smooth. Use solver='auto'/'irls' for L2 or no "
                "penalty, a maintained FISTA-family sparse route, or "
                "solver='auto' for SCAD/MCP."
            )
        return current(
            loss_name=loss_name,
            penalty_name=penalty_name,
            solver_name=solver_name,
            allow_internal_nonconvex=allow_internal_nonconvex,
        )

    setattr(_validate_without_quantile_admm, _MARKER, True)
    _validate_without_quantile_admm._statgpu_original = current
    _quantile_contract._validate_quantile_solver_request = (
        _validate_without_quantile_admm
    )


install_quantile_admm_guard_contract()
