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

from . import _fista as _fista_module
from . import _lbfgs as _lbfgs_module
from . import _quantile_cd as _quantile_cd_module
from ._admm import admm_solver as _admm_solver
from ._fista import fista_solver as _fista_solver
from ._fista_bb import fista_bb_solver as _fista_bb_solver
from ._lbfgs import lbfgs_solver as _lbfgs_solver
from ._lbfgs_b import lbfgs_b_solver as _lbfgs_b_solver
from ._newton import newton_solver as _newton_solver
from ._proximal_newton import proximal_newton_solver as _proximal_newton_solver
from ._quantile_cd import quantile_cd_solver as _quantile_cd_solver


_FISTA_SHAPE_MARKER = "_statgpu_quantile_fista_shape_guard"
_LBFGS_SHAPE_MARKER = "_statgpu_quantile_lbfgs_shape_guard"
_QUANTILE_CD_TOMBSTONE_MARKER = "_statgpu_quantile_cd_tombstone"


def _unwrap_existing_guard(function, marker):
    """Return the numerical original and any already-installed guard."""
    if getattr(function, marker, False):
        original = getattr(function, "_statgpu_original", None)
        if callable(original):
            return original, function
    return function, None


_fista_solver, _existing_fista_guard = _unwrap_existing_guard(
    _fista_solver, _FISTA_SHAPE_MARKER
)
_lbfgs_solver, _existing_lbfgs_guard = _unwrap_existing_guard(
    _lbfgs_solver, _LBFGS_SHAPE_MARKER
)
_quantile_cd_solver, _existing_quantile_cd_guard = _unwrap_existing_guard(
    _quantile_cd_solver, _QUANTILE_CD_TOMBSTONE_MARKER
)

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
setattr(fista_solver, _FISTA_SHAPE_MARKER, True)
fista_solver._statgpu_original = _fista_solver
setattr(lbfgs_solver, _LBFGS_SHAPE_MARKER, True)
lbfgs_solver._statgpu_original = _lbfgs_solver
if _existing_fista_guard is not None:
    fista_solver = _existing_fista_guard
if _existing_lbfgs_guard is not None:
    lbfgs_solver = _existing_lbfgs_guard

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
setattr(quantile_cd_solver, _QUANTILE_CD_TOMBSTONE_MARKER, True)
quantile_cd_solver._statgpu_original = _quantile_cd_solver
if _existing_quantile_cd_guard is not None:
    quantile_cd_solver = _existing_quantile_cd_guard

def _reject_quantile(solver_name: str, reason: str) -> None:
    raise ValueError(
        f"{solver_name} does not support Quantile loss: {reason}. "
        "Use ordinary Quantile FISTA for supported convex objectives, "
        "IRLS for L2/no penalty, or Proximal IRLS-CD for SCAD/MCP."
    )

@wraps(_newton_solver)
def newton_solver(loss, *args, **kwargs):
    """Reject Quantile before entering a Hessian-based Newton kernel."""
    if _is_quantile(loss):
        _reject_quantile(
            "newton_solver",
            "Newton requires a maintained Hessian, which Quantile loss does not provide",
        )
    return _newton_solver(loss, *args, **kwargs)


@wraps(_proximal_newton_solver)
def proximal_newton_solver(loss, *args, **kwargs):
    """Reject Quantile rather than substituting generic FISTA delegation."""
    if _is_quantile(loss):
        _reject_quantile(
            "proximal_newton_solver",
            "the maintained Quantile routes do not expose a Hessian-metric proximal Newton method",
        )
    return _proximal_newton_solver(loss, *args, **kwargs)


@wraps(_lbfgs_b_solver)
def lbfgs_b_solver(loss, *args, **kwargs):
    """Reject non-smooth Quantile loss from the smooth L-BFGS-B route."""
    if _is_quantile(loss):
        _reject_quantile(
            "lbfgs_b_solver",
            "the projected quasi-Newton update requires a smooth objective gradient",
        )
    return _lbfgs_b_solver(loss, *args, **kwargs)

_newton_quantile_doc = """Quantile compatibility

Newton requires a Hessian. QuantileLoss does not provide one, so public
newton_solver calls with QuantileLoss raise before loss preprocessing or
numerical iteration.
"""
newton_solver.__doc__ = (
    _newton_quantile_doc.rstrip() + "\n\n" + (_newton_solver.__doc__ or "").lstrip()
)

_proximal_newton_quantile_doc = """Quantile compatibility

The public Proximal Newton route does not support QuantileLoss. Quantile calls
raise before the generic non-smooth-penalty FISTA delegation, so requesting
Proximal Newton never silently becomes ordinary Quantile FISTA.
"""
proximal_newton_solver.__doc__ = (
    _proximal_newton_quantile_doc.rstrip()
    + "\n\n"
    + (_proximal_newton_solver.__doc__ or "").lstrip()
)

_lbfgs_b_quantile_doc = """Quantile compatibility

L-BFGS-B is a projected smooth quasi-Newton route. QuantileLoss has a
step-function subgradient, so public lbfgs_b_solver Quantile calls raise before
loss preprocessing or numerical iteration.
"""
lbfgs_b_solver.__doc__ = (
    _lbfgs_b_quantile_doc.rstrip()
    + "\n\n"
    + (_lbfgs_b_solver.__doc__ or "").lstrip()
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
# Keep historical direct-module aliases aligned with guarded supported symbols.
# This prevents callers from bypassing Quantile shape/tombstone contracts through
# the long-standing solver module paths while preserving object identity.
_fista_module.fista_solver = fista_solver
_lbfgs_module.lbfgs_solver = lbfgs_solver
_quantile_cd_module.quantile_cd_solver = quantile_cd_solver

# A direct importlib.reload of this guard module reuses the existing supported
# wrappers above, then refreshes public package aliases so import order cannot
# split the public and historical module-path identities.
import sys as _sys
_solver_package = _sys.modules.get(__package__)
if _solver_package is not None:
    for _name in (
        "fista_solver", "fista_bb_solver", "newton_solver",
        "proximal_newton_solver", "lbfgs_solver", "lbfgs_b_solver",
        "admm_solver", "quantile_cd_solver",
    ):
        setattr(_solver_package, _name, globals()[_name])

_glm_core_module = _sys.modules.get("statgpu.glm_core")
if _glm_core_module is not None:
    for _name in ("fista_solver", "fista_bb_solver", "newton_solver", "lbfgs_solver", "admm_solver"):
        setattr(_glm_core_module, _name, globals()[_name])
