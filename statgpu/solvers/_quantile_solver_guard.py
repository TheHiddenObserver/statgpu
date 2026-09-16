"""Fail-closed low-level solver boundaries for non-smooth Quantile loss.

Two generic solver modules rely on smooth-gradient structure that the
Quantile/check loss does not provide:

* FISTA-BB estimates local curvature from smooth-gradient differences;
* ADMM solves its generic w-subproblem with Nesterov-accelerated gradient descent.

Ordinary Quantile FISTA remains a maintained sparse route, while
L2/no-penalty Quantile uses IRLS and SCAD/MCP use Proximal IRLS-CD.

Low-level ``lbfgs_solver(QuantileLoss, ...)`` is intentionally not wrapped
here: unweighted/uniform direct Quantile L-BFGS is an existing maintained
compatibility surface with regression coverage. Estimator-level explicit
Quantile L-BFGS remains fail-closed in the penalized model validator, and
non-uniform direct weights remain fail-closed in the L-BFGS weight contract.
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
        "Use ordinary Quantile FISTA for maintained sparse convex routes, "
        "IRLS for L2/no penalty, or Proximal IRLS-CD for SCAD/MCP."
    )


@wraps(_fista_bb_solver)
def fista_bb_solver(loss, *args, **kwargs):
    """Run FISTA-BB only when its smooth-gradient curvature contract holds."""
    if _is_quantile(loss):
        _reject_quantile(
            "fista_bb_solver",
            "BB step sizes require meaningful smooth-gradient differences",
        )
    return _fista_bb_solver(loss, *args, **kwargs)


# ``functools.wraps`` intentionally preserves the generic solver signature and
# ``__wrapped__`` chain, but it also copies the generic docstring. Override only
# the public documentation after decoration so runtime help exposes the actual
# fail-closed Quantile boundary without changing introspection/signature.
fista_bb_solver.__doc__ = """Public FISTA-BB solver entrypoint.

FISTA-BB estimates local curvature from smooth-gradient differences. Quantile
loss has a step-function subgradient and is therefore not a maintained
FISTA-BB route; public calls with ``QuantileLoss`` fail before loss numerical
work. Use ordinary FISTA for maintained sparse Quantile routes.
"""


@wraps(_admm_solver)
def admm_solver(loss, *args, **kwargs):
    """Run ADMM only when the shared smooth w-update contract is satisfied."""
    if _is_quantile(loss):
        _reject_quantile(
            "admm_solver",
            "the shared w-update uses accelerated gradient descent and requires a smooth loss gradient",
        )
    return _admm_solver(loss, *args, **kwargs)


admm_solver.__doc__ = """Public ADMM solver entrypoint.

The shared non-Cholesky w-update uses Nesterov-accelerated gradient descent and
therefore requires a smooth loss gradient. Quantile loss has a step-function
subgradient and is not a maintained ADMM route; public calls with
``QuantileLoss`` fail before loss numerical work.
"""
