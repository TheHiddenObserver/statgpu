"""Fail-closed public boundaries for unsupported explicit Quantile solvers.

The Quantile estimator surface is deliberately narrower than the set of generic
optimization engines exported by statgpu. This follow-up adds only the missing
all-penalty boundaries for:

* FISTA-BB, which relies on smooth-gradient differences for BB curvature;
* shared ADMM, whose generic w-subproblem uses accelerated gradient descent.

It also corrects the estimator/CV failure reason for explicit Quantile L-BFGS.
L-BFGS does not require a Hessian, but the shared implementation assumes a
smooth loss gradient; Quantile/check loss has a step-function subgradient. The
separate low-level omitted/uniform-weight Quantile L-BFGS compatibility surface
remains unchanged.

Existing Quantile validation remains authoritative first for all other rows.
That preserves the more specific historical errors for L2/no-penalty FISTA-BB,
SCAD/MCP dedicated Proximal IRLS-CD routing, Newton, and other already-
unsupported combinations. Only requests that the existing validator accepted
can reach the new FISTA-BB/ADMM guard.
"""

from __future__ import annotations

from functools import wraps

from . import _quantile_solver_contract as _quantile_contract


_MARKER = "_statgpu_quantile_unsupported_solver_guard_contract"
_ESTIMATOR_MARKER = "_statgpu_quantile_lbfgs_estimator_boundary_contract"
_UNSUPPORTED = frozenset({"admm", "fista_bb"})


def _reject_quantile_lbfgs() -> None:
    raise ValueError(
        "solver='lbfgs' is not supported for Quantile estimators or Quantile CV: "
        "the shared L-BFGS implementation assumes a smooth loss gradient, while "
        "Quantile loss has a step-function subgradient. Use solver='auto'/'irls' "
        "for L2 or no penalty, or ordinary FISTA for supported sparse objectives. "
        "Direct low-level lbfgs_solver(QuantileLoss, ...) retains its separate "
        "omitted/uniform-weight compatibility boundary."
    )


def _is_quantile_lbfgs(loss_name, solver_name) -> bool:
    return (
        _quantile_contract._loss_name(loss_name) == "quantile"
        and str(solver_name or "").lower().strip() == "lbfgs"
    )


def _install_quantile_validator_boundary() -> None:
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
        # L-BFGS is already rejected at estimator/CV level, but the older
        # Quantile validator grouped it with Hessian-based solvers. Correct the
        # public reason before that historical branch can emit a false claim.
        if _is_quantile_lbfgs(loss_name, solver_name):
            _reject_quantile_lbfgs()

        # Preserve every other existing Quantile rejection and its more
        # specific public error semantics. The new guard only closes rows that
        # the established validator previously allowed.
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
            f"solver='{resolved_solver}' does not support Quantile loss: "
            f"{reason}, while Quantile loss is non-smooth. Use ordinary "
            "FISTA for supported sparse convex objectives, solver='auto'/'irls' "
            "for L2 or no penalty, or solver='auto' for SCAD/MCP."
        )

    setattr(_validate_without_unsupported_quantile_solvers, _MARKER, True)
    _validate_without_unsupported_quantile_solvers._statgpu_original = current
    _quantile_contract._validate_quantile_solver_request = (
        _validate_without_unsupported_quantile_solvers
    )


def _install_estimator_lbfgs_boundary() -> None:
    current = _quantile_contract.PenalizedGeneralizedLinearModel._validate_solver_penalty
    if getattr(current, _ESTIMATOR_MARKER, False):
        return

    @wraps(current)
    def _validate_estimator_with_quantile_lbfgs_boundary(self):
        # The shared base validator historically rejects Quantile L-BFGS with
        # a Hessian-specific message. Preempt only this one row so direct
        # generic/typed estimator failure semantics match the CV boundary
        # without changing any numerical route.
        if _is_quantile_lbfgs(
            getattr(self, "loss", ""),
            getattr(self, "_solver", ""),
        ):
            _reject_quantile_lbfgs()
        return current(self)

    setattr(
        _validate_estimator_with_quantile_lbfgs_boundary,
        _ESTIMATOR_MARKER,
        True,
    )
    _validate_estimator_with_quantile_lbfgs_boundary._statgpu_original = current
    _quantile_contract.PenalizedGeneralizedLinearModel._validate_solver_penalty = (
        _validate_estimator_with_quantile_lbfgs_boundary
    )


def install_quantile_unsupported_solver_guard_contract() -> None:
    _install_quantile_validator_boundary()
    _install_estimator_lbfgs_boundary()


install_quantile_unsupported_solver_guard_contract()
