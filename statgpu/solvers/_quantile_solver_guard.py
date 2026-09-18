"""Unsupported low-level solver combinations for non-smooth Quantile loss.

Two generic solver modules rely on smooth-gradient structure that the
Quantile/check loss does not provide:

* FISTA-BB estimates local curvature from smooth-gradient differences;
* ADMM solves its generic w-subproblem with Nesterov-accelerated gradient descent.

Ordinary Quantile FISTA is supported for the corresponding convex objectives,
while L2/no-penalty Quantile defaults to IRLS and SCAD/MCP use Proximal IRLS-CD.

Low-level ``lbfgs_solver(QuantileLoss, ...)`` retains its historical
unweighted/uniform numerical compatibility behavior. The public wrapper below
only validates supervised input shape before delegating to that unchanged
kernel. Estimator-level explicit Quantile L-BFGS remains unsupported, and
non-uniform direct weights are still rejected by the L-BFGS weight check.
"""

from __future__ import annotations

from functools import wraps

import numpy as np

from ._admm import admm_solver as _admm_solver
from ._fista import fista_solver as _fista_solver
from ._fista_bb import fista_bb_solver as _fista_bb_solver
from ._lbfgs import lbfgs_solver as _lbfgs_solver
from ._quantile_cd import quantile_cd_solver as _quantile_cd_solver


def _is_quantile(loss) -> bool:
    return str(getattr(loss, "name", "") or "").lower().strip() == "quantile"


def _validate_quantile_xy_shapes(loss, X, y, solver_name: str) -> None:
    """Validate supported low-level Quantile solver inputs before numerics."""
    if not _is_quantile(loss):
        return

    x_ndim = getattr(X, "ndim", None)
    x_shape = getattr(X, "shape", None)
    if x_ndim is None or x_shape is None:
        X_host = np.asarray(X)
        x_ndim = X_host.ndim
        x_shape = X_host.shape

    y_ndim = getattr(y, "ndim", None)
    y_shape = getattr(y, "shape", None)
    if y_ndim is None or y_shape is None:
        y_host = np.asarray(y)
        y_ndim = y_host.ndim
        y_shape = y_host.shape

    if int(x_ndim) != 2:
        raise ValueError(f"X must be two-dimensional for {solver_name}")
    if int(y_ndim) != 1:
        raise ValueError(f"y must be one-dimensional for {solver_name}")
    if int(y_shape[0]) != int(x_shape[0]):
        raise ValueError(
            f"y must have the same number of observations as X for {solver_name}"
        )


@wraps(_fista_solver)
def fista_solver(loss, penalty, X, y, *args, **kwargs):
    """Run ordinary FISTA with the public Quantile supervised-shape contract."""
    _validate_quantile_xy_shapes(loss, X, y, "fista_solver")
    return _fista_solver(loss, penalty, X, y, *args, **kwargs)


@wraps(_lbfgs_solver)
def lbfgs_solver(loss, penalty, X, y, *args, **kwargs):
    """Run L-BFGS while preserving its maintained Quantile compatibility row."""
    _validate_quantile_xy_shapes(loss, X, y, "lbfgs_solver")
    return _lbfgs_solver(loss, penalty, X, y, *args, **kwargs)


_quantile_supported_shape_doc = """Quantile supervised-input boundary

For direct public Quantile calls, X must be two-dimensional, y must be
one-dimensional, and their observation counts must agree. Malformed shapes are
rejected before loss preprocessing or numerical iteration.
"""
fista_solver.__doc__ = (
    _quantile_supported_shape_doc.rstrip()
    + "\n\n"
    + (_fista_solver.__doc__ or "").lstrip()
)
lbfgs_solver.__doc__ = (
    _quantile_supported_shape_doc.rstrip()
    + "\n\n"
    + (_lbfgs_solver.__doc__ or "").lstrip()
)

@wraps(_quantile_cd_solver)
def quantile_cd_solver(*args, **kwargs):
    """Fail closed for the retired experimental Quantile coordinate-descent path."""
    raise NotImplementedError(
        "quantile_cd_solver is retained only as a compatibility symbol and is "
        "not a maintained public numerical route. The historical implementation "
        "silently ignored sample_weight and could not represent an unpenalized "
        "intercept reliably. Use proximal_irls_quantile_solver for maintained "
        "scalar SCAD/MCP Quantile fitting, or ordinary fista_solver for supported "
        "convex Quantile objectives."
    )


_quantile_cd_solver_doc = """Compatibility boundary

The historical Quantile coordinate-descent implementation is not a maintained
public numerical route. It is retained only as an import-compatible symbol and
fails before numerical work. Use proximal_irls_quantile_solver for scalar
SCAD/MCP Quantile objectives or ordinary fista_solver for supported convex
Quantile objectives.
"""
quantile_cd_solver.__doc__ = _quantile_cd_solver_doc

def _reject_quantile(solver_name: str, reason: str) -> None:
    raise ValueError(
        f"{solver_name} does not support Quantile loss: {reason}. "
        "Use ordinary Quantile FISTA for supported convex objectives, "
        "IRLS for L2/no penalty, or Proximal IRLS-CD for SCAD/MCP."
    )


@wraps(_fista_bb_solver)
def fista_bb_solver(loss, *args, **kwargs):
    """Run FISTA-BB only when its smooth-gradient curvature requirements hold."""
    if _is_quantile(loss):
        _reject_quantile(
            "fista_bb_solver",
            "BB step sizes require meaningful smooth-gradient differences",
        )
    return _fista_bb_solver(loss, *args, **kwargs)


# ``functools.wraps`` preserves the generic solver signature and ``__wrapped__``
# chain while initially copying the complete generic docstring. Prefix the
# public boundary instead of replacing that documentation so ``help()`` keeps
# the established parameter/return reference as well as the Quantile contract.
_fista_boundary_doc = """Quantile compatibility

FISTA-BB estimates local curvature from smooth-gradient differences. Quantile
loss has a step-function subgradient, so FISTA-BB does not support Quantile;
public calls with ``QuantileLoss`` raise an error before numerical iteration.
Use ordinary FISTA for supported convex Quantile objectives.
"""
fista_bb_solver.__doc__ = (
    _fista_boundary_doc.rstrip()
    + "\n\n"
    + (_fista_bb_solver.__doc__ or "").lstrip()
)


@wraps(_admm_solver)
def admm_solver(loss, *args, **kwargs):
    """Run ADMM only when the shared smooth w-update requirements are satisfied."""
    if _is_quantile(loss):
        _reject_quantile(
            "admm_solver",
            "the shared w-update uses accelerated gradient descent and requires a smooth loss gradient",
        )
    return _admm_solver(loss, *args, **kwargs)


_admm_boundary_doc = """Quantile compatibility

The shared non-Cholesky w-update uses Nesterov-accelerated gradient descent and
therefore requires a smooth loss gradient. Quantile loss has a step-function
subgradient, so this ADMM implementation does not support Quantile; public calls
with ``QuantileLoss`` raise an error before numerical iteration.
"""
admm_solver.__doc__ = (
    _admm_boundary_doc.rstrip()
    + "\n\n"
    + (_admm_solver.__doc__ or "").lstrip()
)
