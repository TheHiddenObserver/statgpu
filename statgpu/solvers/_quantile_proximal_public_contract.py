"""Public contract wrapper for Proximal IRLS-CD Quantile calls.

The reviewed numerical kernel lives in ``_proximal_irls_quantile``. This module
adds public-input validation, a defined meaning for the historical
``max_iter=None`` default, and objective-consistent handling of estimator-
generated Quantile continuation starts before delegating to that kernel.
The wrapper is installed on the kernel module itself so the maintained top-level
export and the historically documented module path cannot diverge.
"""

from __future__ import annotations

from functools import wraps
from numbers import Integral, Real

import numpy as np

from . import _proximal_irls_quantile as _kernel_module
from ._quantile_continuation import resolve_auto_quantile_continuation_path


_MARKER = "_statgpu_quantile_proximal_public_contract"


def _validate_alpha_path(alpha_path):
    """Validate the public continuation path before any numerical work."""
    if isinstance(alpha_path, (list, tuple)):
        path_ndim = np.asarray(alpha_path, dtype=object).ndim
    else:
        path_ndim = getattr(alpha_path, "ndim", 1)
    if path_ndim != 1:
        raise ValueError("alpha_path must be a non-empty one-dimensional sequence")

    try:
        path_len = len(alpha_path)
    except TypeError as exc:
        raise ValueError(
            "alpha_path must be a non-empty one-dimensional sequence"
        ) from exc
    if path_len < 1:
        raise ValueError("alpha_path must be a non-empty one-dimensional sequence")

    path_values = []
    for value in alpha_path:
        if isinstance(value, (bool, np.bool_, str, bytes)):
            raise ValueError("alpha_path must contain finite positive numbers")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "alpha_path must contain finite positive numbers"
            ) from exc
        if not np.isfinite(numeric) or numeric <= 0.0:
            raise ValueError("alpha_path must contain finite positive numbers")
        path_values.append(numeric)

    if any(
        path_values[i + 1] > path_values[i]
        for i in range(len(path_values) - 1)
    ):
        raise ValueError(
            "alpha_path must be non-increasing from continuation start to target"
        )
    return alpha_path


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
        loss_name = str(getattr(loss, "name", "") or "").lower().strip()
        if loss_name != "quantile":
            raise ValueError(
                "proximal_irls_quantile_solver requires QuantileLoss"
            )
        penalty_name = str(
            getattr(penalty, "name", "") or ""
        ).lower().strip()
        if penalty_name not in {"scad", "mcp"}:
            raise ValueError(
                "proximal_irls_quantile_solver requires scalar SCAD or MCP penalty"
            )

        if not isinstance(fit_intercept, (bool, np.bool_)):
            raise ValueError("fit_intercept must be boolean")
        fit_intercept = bool(fit_intercept)

        if sample_weight is not None:
            # Import lazily to avoid a package-initialization cycle through
            # ``glm_core.__init__`` -> ``statgpu.solvers``.
            from statgpu.glm_core._validation import validate_glm_sample_weight

            sample_weight = validate_glm_sample_weight(
                sample_weight,
                len(X),
            )

        # Validate the public path before the objective-aware resolver or the
        # numerical kernel sees it. Plain low-level paths remain authoritative,
        # but they must still describe a finite positive continuation from a
        # no-smaller start to the target alpha.
        alpha_path = _validate_alpha_path(alpha_path)

        # Only estimator-generated Quantile paths carry the internal marker.
        # Non-uniform analytic weights define the data-fit objective, so their
        # automatic continuation start uses the same weighted intercept/score.
        # Direct low-level callers supplying a plain alpha_path remain
        # authoritative and are never rewritten here.
        alpha_path = resolve_auto_quantile_continuation_path(
            loss,
            X,
            y,
            alpha_path,
            sample_weight=sample_weight,
            fit_intercept=fit_intercept,
        )

        # ``None`` has historically appeared in the public signature even though
        # the kernel requires a concrete iteration budget. Give that default a
        # deterministic public meaning instead of reaching ``range(None)``.
        if max_iter is None:
            max_iter = 100

        if isinstance(max_lla_per_step, (bool, np.bool_)) or not isinstance(
            max_lla_per_step, Integral
        ) or int(max_lla_per_step) < 1:
            raise ValueError("max_lla_per_step must be a positive integer")
        max_lla_per_step = int(max_lla_per_step)

        for name, value in (("tol", tol), ("lla_tol", lla_tol)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise ValueError(f"{name} must be a finite positive number")
            value = float(value)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a finite positive number")
            if name == "tol":
                tol = value
            else:
                lla_tol = value

        if isinstance(max_iter, (list, tuple)):
            if len(max_iter) != len(alpha_path):
                raise ValueError(
                    "max_iter sequence must have one positive integer per alpha_path step"
                )
            normalized_max_iter = []
            for value in max_iter:
                if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                    raise ValueError(
                        "max_iter sequence must contain only positive integers"
                    )
                value = int(value)
                if value < 1:
                    raise ValueError(
                        "max_iter sequence must contain only positive integers"
                    )
                normalized_max_iter.append(value)
            max_iter = normalized_max_iter
        else:
            if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, Integral):
                raise ValueError("max_iter must be a positive integer or sequence")
            max_iter = int(max_iter)
            if max_iter < 1:
                raise ValueError("max_iter must be a positive integer or sequence")

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
    # Extend runtime help with the public-default and automatic weighted-path
    # clarifications absent from the frozen kernel docstring.
    doc = current.__doc__ or ""
    old = "    max_iter : int or list\n        Maximum IRLS iterations per continuation step."
    new = (
        "    max_iter : int or list, optional\n"
        "        Maximum IRLS iterations per continuation step. ``None`` uses 100 "
        "iterations per step."
    )
    if old in doc:
        doc = doc.replace(old, new)
    old_path = "    alpha_path : array\n        Continuation path from lambda_max to target alpha."
    new_path = (
        "    alpha_path : array\n"
        "        Non-empty one-dimensional continuation path of finite positive "
        "values in non-increasing order from lambda_max/start to target alpha. "
        "Estimator-generated paths use the declared analytic weights in the "
        "Quantile continuation start; an explicitly supplied low-level path is "
        "preserved subject to the same shape/value/order requirements."
    )
    if old_path in doc:
        doc = doc.replace(old_path, new_path)
    _with_public_boundary.__doc__ = doc

    _kernel_module.proximal_irls_quantile_solver = _with_public_boundary
    return _with_public_boundary


proximal_irls_quantile_solver = install_quantile_proximal_public_contract()


__all__ = [
    "install_quantile_proximal_public_contract",
    "proximal_irls_quantile_solver",
]
