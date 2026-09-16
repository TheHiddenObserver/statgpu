"""Fail-closed low-level solver boundaries for non-smooth Quantile loss.

The generic solver modules below are maintained for algorithmic assumptions
that Quantile/check loss does not satisfy:

* L-BFGS expects a smooth objective for its curvature history and Armijo search;
* FISTA-BB estimates local curvature from smooth-gradient differences;
* ADMM solves its w-subproblem with Nesterov-accelerated gradient descent.

Ordinary Quantile FISTA remains a maintained sparse route, while smooth
L2/no-penalty Quantile uses IRLS and SCAD/MCP use Proximal IRLS-CD.
"""

from __future__ import annotations

from functools import wraps

from ._admm import admm_solver as _admm_solver
from ._fista_bb import fista_bb_solver as _fista_bb_solver
from ._lbfgs import lbfgs_solver as _lbfgs_solver


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


@wraps(_lbfgs_solver)
def lbfgs_solver(loss, *args, **kwargs):
    """Run L-BFGS only when its smooth-objective contract holds."""
    if _is_quantile(loss):
        _reject_quantile(
            "lbfgs_solver",
            "the quasi-Newton curvature and line-search contract requires a smooth objective",
        )
    return _lbfgs_solver(loss, *args, **kwargs)


@wraps(_admm_solver)
def admm_solver(loss, *args, **kwargs):
    """Run ADMM only when the shared smooth w-update contract is satisfied."""
    if _is_quantile(loss):
        _reject_quantile(
            "admm_solver",
            "the shared w-update uses accelerated gradient descent and requires a smooth loss gradient",
        )
    return _admm_solver(loss, *args, **kwargs)
