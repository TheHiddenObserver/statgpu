"""Public contract wrapper for Proximal IRLS-CD Quantile calls.

The reviewed numerical kernel lives in ``_proximal_irls_quantile``. This module
adds only public-input validation and a defined meaning for the historical
``max_iter=None`` default, then delegates unchanged valid inputs to that kernel.
The wrapper is installed on the kernel module itself so the maintained top-level
export and the historically documented module path cannot diverge.
"""

from __future__ import annotations

from functools import wraps

from . import _proximal_irls_quantile as _kernel_module


_MARKER = "_statgpu_quantile_proximal_public_contract"


def install_quantile_proximal_public_contract():
    current = _kernel_module.proximal_irls_quantile_solver
    if getattr(current, _MARKER, False):
        return current

    @wraps(current)
    def _with_public_boundary(
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

        return current(
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

    setattr(_with_public_boundary, _MARKER, True)
    _with_public_boundary._statgpu_original = current

    # ``functools.wraps`` retains signature/``__wrapped__`` compatibility.
    # Extend runtime help with the one public-default clarification absent from
    # the frozen kernel docstring.
    doc = current.__doc__ or ""
    old = "    max_iter : int or list\n        Maximum IRLS iterations per continuation step."
    new = (
        "    max_iter : int or list, optional\n"
        "        Maximum IRLS iterations per continuation step. ``None`` uses 100 "
        "iterations per step."
    )
    if old in doc:
        _with_public_boundary.__doc__ = doc.replace(old, new)

    _kernel_module.proximal_irls_quantile_solver = _with_public_boundary
    return _with_public_boundary


proximal_irls_quantile_solver = install_quantile_proximal_public_contract()


__all__ = [
    "install_quantile_proximal_public_contract",
    "proximal_irls_quantile_solver",
]
