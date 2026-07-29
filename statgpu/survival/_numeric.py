"""Shared numerical boundaries for survival-model public outputs."""

from __future__ import annotations

from typing import Any, Type

import numpy as np

from statgpu.backends._array_ops import _xp as _get_xp, _xp_asarray
from statgpu.backends._utils import _is_complex_array, _to_float_scalar


_LOG_FLOAT64_MAX = float(np.log(np.finfo(np.float64).max))
_LOG_FLOAT64_MIN_POSITIVE = float(
    np.log(np.nextafter(np.float64(0.0), np.float64(1.0)))
)


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


__all__ = ["_safe_exp_linear_predictor"]
