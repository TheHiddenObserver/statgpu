"""Truthful explicit FISTA support for smooth Quantile penalties.

Quantile L2/no-penalty ``solver='auto'`` deliberately prefers IRLS, but an
explicit ``solver='fista'`` request should remain authoritative.  Historically
the generic FISTA branch silently substituted ``QuantileLoss.irls()`` for those
smooth penalties; the earlier provenance repair therefore failed the request
closed rather than reporting a solver that did not execute.

This contract completes that capability without changing auto dispatch:

* ``auto`` / explicit ``irls`` on Quantile L2/no penalty still execute IRLS;
* explicit ``fista`` on the same objectives executes the maintained generic
  ``fista_solver`` on NumPy/CuPy/Torch;
* FISTA-BB, ADMM, L-BFGS, and non-convex Quantile boundaries remain unchanged.

The existing fit implementation contains the historical IRLS substitution in
its FISTA branch.  To avoid duplicating the large backend/intercept setup, this
narrow installer presents a loss view that preserves every Quantile numerical
primitive while intentionally hiding only the ``irls`` attribute during an
explicit smooth-Quantile FISTA call.  The original branch therefore reaches
``fista_solver`` with the same prepared design, selective intercept penalty,
warm start, analytic weights, dtype, and backend as every other FISTA route.
"""

from __future__ import annotations

from functools import wraps

from . import _fit_mixin
from . import _quantile_solver_contract as _quantile_contract


_VALIDATOR_MARKER = "_statgpu_quantile_smooth_fista_validator_contract"
_FIT_MARKER = "_statgpu_quantile_smooth_fista_execution_contract"
_SMOOTH_PENALTIES = _quantile_contract._SMOOTH_PENALTIES


class _QuantileFistaLossView:
    """Delegate Quantile numerics while hiding only the IRLS entry point."""

    __slots__ = ("_loss",)
    name = "quantile"

    def __init__(self, loss):
        self._loss = loss

    def __getattr__(self, name):
        if name == "irls":
            raise AttributeError(name)
        return getattr(self._loss, name)


def _wrapper_chain_has_marker(function, marker: str) -> bool:
    current = function
    seen = set()
    while callable(current) and id(current) not in seen:
        seen.add(id(current))
        if bool(getattr(current, marker, False)):
            return True
        wrapped = getattr(current, "_statgpu_original", None)
        if not callable(wrapped):
            wrapped = getattr(current, "__wrapped__", None)
        if not callable(wrapped):
            break
        current = wrapped
    return False


def _is_smooth_quantile_fista_request(
    *, loss_name, penalty_name, solver_name
) -> bool:
    return (
        _quantile_contract._loss_name(loss_name) == "quantile"
        and _quantile_contract._penalty_name(penalty_name) in _SMOOTH_PENALTIES
        and str(solver_name or "").lower().strip() == "fista"
    )


def _install_validator_allowance() -> None:
    current = _quantile_contract._validate_quantile_solver_request
    if _wrapper_chain_has_marker(current, _VALIDATOR_MARKER):
        return

    @wraps(current)
    def _validate_with_explicit_smooth_quantile_fista(
        *,
        loss_name,
        penalty_name,
        solver_name,
        allow_internal_nonconvex=False,
    ):
        if _is_smooth_quantile_fista_request(
            loss_name=loss_name,
            penalty_name=penalty_name,
            solver_name=solver_name,
        ):
            return None
        return current(
            loss_name=loss_name,
            penalty_name=penalty_name,
            solver_name=solver_name,
            allow_internal_nonconvex=allow_internal_nonconvex,
        )

    setattr(
        _validate_with_explicit_smooth_quantile_fista,
        _VALIDATOR_MARKER,
        True,
    )
    _validate_with_explicit_smooth_quantile_fista._statgpu_original = current
    _quantile_contract._validate_quantile_solver_request = (
        _validate_with_explicit_smooth_quantile_fista
    )


def _install_execution_route() -> None:
    current = _fit_mixin._PenalizedFitMixin._fit_loss_backend
    if _wrapper_chain_has_marker(current, _FIT_MARKER):
        return

    @wraps(current)
    def _fit_loss_backend_with_explicit_smooth_quantile_fista(
        self, X, y, sample_weight, solver_name, backend_name
    ):
        if not _is_smooth_quantile_fista_request(
            loss_name=getattr(self, "_loss", getattr(self, "loss", "")),
            penalty_name=getattr(self, "_penalty", getattr(self, "penalty", "")),
            solver_name=solver_name,
        ):
            return current(self, X, y, sample_weight, solver_name, backend_name)

        original_loss = self._loss
        self._loss = _QuantileFistaLossView(original_loss)
        try:
            return current(self, X, y, sample_weight, solver_name, backend_name)
        finally:
            self._loss = original_loss

    setattr(
        _fit_loss_backend_with_explicit_smooth_quantile_fista,
        _FIT_MARKER,
        True,
    )
    _fit_loss_backend_with_explicit_smooth_quantile_fista._statgpu_original = current
    _fit_mixin._PenalizedFitMixin._fit_loss_backend = (
        _fit_loss_backend_with_explicit_smooth_quantile_fista
    )


def install_quantile_smooth_fista_contract() -> None:
    """Install explicit smooth-Quantile FISTA support idempotently."""

    _install_validator_allowance()
    _install_execution_route()


install_quantile_smooth_fista_contract()
