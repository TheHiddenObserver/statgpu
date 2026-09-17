"""Unsupported low-level solver combinations for non-smooth Quantile loss.

Two generic solver modules rely on smooth-gradient structure that the
Quantile/check loss does not provide:

* FISTA-BB estimates local curvature from smooth-gradient differences;
* ADMM solves its generic w-subproblem with Nesterov-accelerated gradient descent.

Ordinary Quantile FISTA is supported for the corresponding convex objectives,
while L2/no-penalty Quantile defaults to IRLS and SCAD/MCP use Proximal IRLS-CD.

Low-level ``lbfgs_solver(QuantileLoss, ...)`` is intentionally not wrapped
here: unweighted/uniform direct Quantile L-BFGS retains its historical
compatibility behavior. Estimator-level explicit Quantile L-BFGS is unsupported,
and non-uniform direct weights are rejected by the L-BFGS weight check.
"""

from __future__ import annotations

from functools import wraps

from ._admm import admm_solver as _admm_solver
from ._fista_bb import fista_bb_solver as _fista_bb_solver


def _is_quantile(loss) -> bool:
    return str(getattr(loss, "name", "") or "").lower().strip() == "quantile"


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
