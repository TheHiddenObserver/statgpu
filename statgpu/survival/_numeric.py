"""Shared numerical boundaries for survival-model public outputs."""

from __future__ import annotations

from typing import Any, Type

import numpy as np

from statgpu.backends._array_ops import _xp as _get_xp, _xp_asarray
from statgpu.backends._utils import (
    _is_complex_array,
    _require_real_array,
    _to_float_scalar,
)


_LOG_FLOAT64_MAX = float(np.log(np.finfo(np.float64).max))
_LOG_FLOAT64_MIN_POSITIVE = float(
    np.log(np.nextafter(np.float64(0.0), np.float64(1.0)))
)


def _normalize_prediction_matrix(
    value: Any,
    *,
    backend: Any,
    n_features: int,
    name: str = "X",
):
    """Normalize a public prediction matrix with one shared Cox contract.

    A one-dimensional input is a vector of observations for a one-feature
    model, or one complete observation for a multi-feature model.  Ambiguous
    or higher-dimensional inputs fail before backend matmul so NumPy, CuPy,
    and Torch expose the same public error behavior.
    """
    _require_real_array(value, name)
    n_features = int(n_features)
    if n_features < 1:
        raise ValueError("n_features must be a positive integer")
    array = backend.asarray(value, dtype=backend.float64)
    if array.ndim == 1:
        if n_features == 1:
            array = array.reshape(-1, 1)
        elif int(array.shape[0]) == n_features:
            array = array.reshape(1, -1)
        else:
            raise ValueError(
                f"One-dimensional {name} must contain one complete "
                f"{n_features}-feature row"
            )
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    actual_features = int(array.shape[1])
    if actual_features != n_features:
        raise ValueError(
            f"{name} has {actual_features} features; expected {n_features}"
        )
    xp = backend.xp
    if bool(_to_float_scalar(xp.any(~xp.isfinite(array)))):
        raise ValueError(
            f"{name} must contain only finite values; NaN or infinite "
            "values are not allowed"
        )
    return array


def _safe_exp_linear_predictor(
    value: Any,
    *,
    error_type: Type[FloatingPointError] = FloatingPointError,
    name: str = "linear predictor",
):
    """Exponentiate only values representable as finite positive float64.

    Cox log-risk remains available through ``predict_risk_score`` without this
    transformation. Hazard-ratio APIs deliberately raise instead of silently
    clipping statistically meaningful log-risk values.
    """
    if _is_complex_array(value):
        raise ValueError(f"{name} must be real-valued")
    xp = _get_xp(value)
    array = _xp_asarray(value, dtype=xp.float64, ref_arr=value)
    if getattr(xp, "__name__", "") == "numpy":
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            result = xp.exp(array)
    else:
        result = xp.exp(array)
    invalid = xp.any(
        (~xp.isfinite(array))
        | (array > _LOG_FLOAT64_MAX)
        | (array < _LOG_FLOAT64_MIN_POSITIVE)
        | (~xp.isfinite(result))
        | (result <= 0)
    )
    if bool(_to_float_scalar(invalid)):
        raise error_type(
            f"{name} is outside the finite positive float64 exp range; "
            "use predict_risk_score() for unexponentiated log-risk"
        )
    return result


__all__ = ["_normalize_prediction_matrix", "_safe_exp_linear_predictor"]
