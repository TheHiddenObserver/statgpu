"""Public contract wrapper for Proximal IRLS-CD Quantile calls.

The reviewed numerical kernel lives in ``_proximal_irls_quantile``. This module
adds only public-input validation and a defined meaning for the historical
``max_iter=None`` default, then delegates unchanged valid inputs to that kernel.
"""

from __future__ import annotations

from functools import wraps

from ._proximal_irls_quantile import (
    proximal_irls_quantile_solver as _proximal_irls_quantile_solver,
)


@wraps(_proximal_irls_quantile_solver)
def proximal_irls_quantile_solver(
    loss,
    penalty,
    X,
    y,
    alpha_path,
    max_lla_per_step=2,
    lla_tol=1e-6,
    max_iter=None,
    tol=1e-6,
    fit_intercept=True,
    sample_weight=None,
):
    """Validate the public boundary, then run the reviewed Quantile kernel."""
    if sample_weight is not None:
        # Import lazily to avoid a package-initialization cycle through
        # ``glm_core.__init__`` -> ``statgpu.solvers``.
        from statgpu.glm_core._validation import validate_glm_sample_weight

        sample_weight = validate_glm_sample_weight(
            sample_weight,
            int(X.shape[0]),
        )

    # ``None`` has historically appeared in the public signature even though
    # the kernel requires a concrete iteration budget. Give that default a
    # deterministic public meaning instead of reaching ``range(None)``.
    if max_iter is None:
        max_iter = 100

    return _proximal_irls_quantile_solver(
        loss,
        penalty,
        X,
        y,
        alpha_path,
        max_lla_per_step=max_lla_per_step,
        lla_tol=lla_tol,
        max_iter=max_iter,
        tol=tol,
        fit_intercept=fit_intercept,
        sample_weight=sample_weight,
    )


# ``functools.wraps`` retains signature/``__wrapped__`` compatibility. Extend
# runtime help with the one public-default clarification absent from the frozen
# kernel docstring.
_doc = _proximal_irls_quantile_solver.__doc__ or ""
_old = "    max_iter : int or list\n        Maximum IRLS iterations per continuation step."
_new = (
    "    max_iter : int or list, optional\n"
    "        Maximum IRLS iterations per continuation step. ``None`` uses 100 "
    "iterations per step."
)
if _old in _doc:
    proximal_irls_quantile_solver.__doc__ = _doc.replace(_old, _new)


__all__ = ["proximal_irls_quantile_solver"]
