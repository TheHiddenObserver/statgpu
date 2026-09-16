"""Fail-closed ADMM boundary for non-smooth Quantile loss.

The maintained shared ADMM implementation solves its w-subproblem with
Nesterov-accelerated gradient descent and therefore requires a smooth loss
gradient. Quantile/check loss has a discontinuous subgradient, so routing it
through that implementation is not a maintained numerical algorithm.
"""

from __future__ import annotations

from functools import wraps

from ._admm import admm_solver as _admm_solver


@wraps(_admm_solver)
def admm_solver(loss, *args, **kwargs):
    """Run ADMM only when the shared smooth w-update contract is satisfied."""
    loss_name = str(getattr(loss, "name", "") or "").lower().strip()
    if loss_name == "quantile":
        raise ValueError(
            "admm_solver does not support Quantile loss: its shared w-update "
            "uses accelerated gradient descent and requires a smooth loss "
            "gradient. Use Quantile IRLS/FISTA-family routes, or the dedicated "
            "Proximal IRLS-CD route for SCAD/MCP."
        )
    return _admm_solver(loss, *args, **kwargs)
