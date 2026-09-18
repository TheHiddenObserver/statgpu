"""Public validation boundary for direct Quantile IRLS calls.

High-level penalized estimators validate analytic weights before dispatch, but
``QuantileLoss.irls()`` is also a callable low-level public method. Keep its
weighted contract fail-closed without changing the reviewed IRLS numerical
kernel itself.
"""

from __future__ import annotations

from functools import wraps
from numbers import Integral, Real

import numpy as np

from ._quantile import QuantileLoss


_MARKER = "_statgpu_quantile_irls_weight_validation_contract"


def _validate_quantile_xy_shapes(X, y) -> int:
    """Validate supervised array shapes without materializing GPU data."""
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
        raise ValueError("X must be two-dimensional for Quantile IRLS")
    if int(y_ndim) != 1:
        raise ValueError("y must be one-dimensional for Quantile IRLS")
    n_samples = int(x_shape[0])
    if int(y_shape[0]) != n_samples:
        raise ValueError(
            "y must have the same number of observations as X for Quantile IRLS"
        )
    return n_samples


def install_quantile_irls_validation_contract() -> None:
    current = QuantileLoss.irls
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def _irls_with_validated_weights(
        self,
        X,
        y,
        penalty=None,
        max_iter=100,
        tol=1e-6,
        init_coef=None,
        eps=1e-8,
        sample_weight=None,
        fit_intercept=False,
    ):
        if not isinstance(fit_intercept, (bool, np.bool_)):
            raise ValueError("fit_intercept must be boolean")
        fit_intercept = bool(fit_intercept)

        if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, Integral):
            raise ValueError("max_iter must be a positive integer")
        max_iter = int(max_iter)
        if max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        for name, value in (("tol", tol), ("eps", eps)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise ValueError(f"{name} must be a finite positive number")
            value = float(value)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a finite positive number")
            if name == "tol":
                tol = value
            else:
                eps = value

        n_samples = _validate_quantile_xy_shapes(X, y)

        if sample_weight is not None:
            # Import lazily so the losses package does not enter glm_core while
            # module initialization is still resolving the generic solver graph.
            from statgpu.glm_core._validation import validate_glm_sample_weight

            sample_weight = validate_glm_sample_weight(
                sample_weight,
                n_samples,
            )

        return current(
            self,
            X,
            y,
            penalty=penalty,
            max_iter=max_iter,
            tol=tol,
            init_coef=init_coef,
            eps=eps,
            sample_weight=sample_weight,
            fit_intercept=fit_intercept,
        )

    setattr(_irls_with_validated_weights, _MARKER, True)
    _irls_with_validated_weights._statgpu_original = current
    QuantileLoss.irls = _irls_with_validated_weights


install_quantile_irls_validation_contract()


__all__ = ["install_quantile_irls_validation_contract"]
