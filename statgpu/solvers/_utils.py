"""Shared utility functions for solvers.

Validation helpers, penalty value/gradient/hessian utilities,
and objective function helpers. All work with generic loss/penalty interfaces.
"""

import numpy as np

from statgpu.backends import _resolve_backend, _to_numpy
from statgpu.backends._utils import _to_float_scalar, _get_xp
from statgpu.backends._array_ops import (
    _abs_sum,
    _abs_sum_dev,
    _copy_arr,
    _dot,
    _dot_dev,
    _eye_like,
    _norm2,
    _norm2_dev,
    _sum_sq,
    _sum_sq_dev,
    _sync_scalars,
    _zeros,
    _zeros_like,
)


def _validate_uniform_sample_weight(sample_weight, n_samples, solver_name):
    if sample_weight is None:
        return
    _sw = _to_numpy(sample_weight)
    if _sw.ndim != 1 or _sw.shape[0] != n_samples:
        raise ValueError("sample_weight must be a 1D array with length n_samples")
    if not np.all(np.isfinite(_sw)):
        raise ValueError("sample_weight must contain only finite values")
    if np.any(_sw < 0):
        raise ValueError("sample_weight must be non-negative")
    if np.sum(_sw) <= 0.0:
        raise ValueError("sample_weight must contain at least one positive value")
    if not np.allclose(_sw, _sw[0]):
        raise ValueError(
            f"{solver_name} does not support non-uniform sample_weight yet; "
            "use solver='irls' for weighted GLM fits."
        )


def _validate_sample_weight(sample_weight, n_samples):
    if sample_weight is None:
        return
    _sw = _to_numpy(sample_weight)
    if _sw.ndim != 1 or _sw.shape[0] != n_samples:
        raise ValueError("sample_weight must be 1D with length n_samples")
    if not np.all(np.isfinite(_sw)):
        raise ValueError("sample_weight must contain only finite values")
    if np.any(_sw < 0):
        raise ValueError("sample_weight must be non-negative")
    if np.sum(_sw) <= 0:
        raise ValueError("sample_weight must contain at least one positive value")


def _as_backend_vector(arr, backend, ref):
    from statgpu.backends._utils import xp_asarray
    xp = _get_xp(backend)
    dtype = getattr(ref, "dtype", np.float64)
    return xp_asarray(arr, dtype=dtype, xp=xp, ref_arr=ref)


def _call_with_weight(fn, *args, sample_weight=None, **kwargs):
    """Call fn with sample_weight if it accepts it, without otherwise.

    Avoids the repeated try/except TypeError pattern. Inspects the
    function signature once to decide whether to pass sample_weight.
    """
    import inspect
    try:
        sig = inspect.signature(fn)
        if 'sample_weight' in sig.parameters:
            return fn(*args, sample_weight=sample_weight, **kwargs)
    except (ValueError, TypeError):
        pass
    return fn(*args, **kwargs)


def _nesterov_momentum(t_k, beta_cap=None):
    """Compute Nesterov momentum parameters.

    Parameters
    ----------
    t_k : float
        Current momentum parameter.
    beta_cap : float, optional
        Maximum allowed momentum (e.g. 0.5 for CV stability).

    Returns
    -------
    beta : float
        Momentum coefficient.
    t_new : float
        Updated momentum parameter.
    """
    import math
    t_new = (1.0 + math.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
    beta = (t_k - 1.0) / t_new
    if beta_cap is not None:
        beta = min(beta, beta_cap)
    return beta, t_new


def _nesterov_update(coef, coef_old, t_k, beta_cap=None):
    """Nesterov momentum update: compute extrapolated point y_k and new t.

    Parameters
    ----------
    coef : array
        Current iterate.
    coef_old : array
        Previous iterate.
    t_k : float
        Current momentum parameter.
    beta_cap : float, optional
        Maximum allowed momentum (e.g. 0.5 for CV stability).

    Returns
    -------
    y_k : array
        Extrapolated point: coef + beta * (coef - coef_old).
    t_new : float
        Updated momentum parameter.
    """
    beta, t_new = _nesterov_momentum(t_k, beta_cap)
    y_k = coef + beta * (coef - coef_old)
    return y_k, t_new


def _penalty_name(penalty):
    return str(getattr(penalty, "name", "none")).lower()


def _smooth_penalty_value(penalty, coef):
    if penalty is None:
        return 0.0
    if hasattr(penalty, "smooth_value"):
        return float(_to_numpy(penalty.smooth_value(coef)))
    if _penalty_name(penalty) in ("none", "null"):
        return 0.0
    if _penalty_name(penalty) == "l2":
        return 0.5 * float(getattr(penalty, "alpha", 0.0)) * _sum_sq(coef)
    if _penalty_name(penalty) == "elasticnet":
        alpha = float(getattr(penalty, "alpha", 0.0))
        l1_ratio = float(getattr(penalty, "l1_ratio", 1.0))
        return 0.5 * alpha * (1.0 - l1_ratio) * _sum_sq(coef)
    raise ValueError(
        f"solver requires a smooth penalty, got penalty='{_penalty_name(penalty)}'."
    )


def _tracking_penalty_value(penalty, coef):
    pen_name = _penalty_name(penalty)
    if penalty is None or pen_name in ("none", "null"):
        return 0.0
    n_features = getattr(penalty, "n_features", None)
    if n_features is not None:
        coef_eval = coef[: int(n_features)]
        backend = _resolve_backend("auto", coef_eval)
        if pen_name == "l1":
            if backend in ("torch", "cupy"):
                abs_sum, = _sync_scalars(_abs_sum_dev(coef_eval), backend=backend)
            else:
                abs_sum = _abs_sum(coef_eval)
            return float(getattr(penalty, "alpha", 0.0)) * abs_sum
        if pen_name in ("elasticnet", "en"):
            alpha = float(getattr(penalty, "alpha", 0.0))
            l1_ratio = float(getattr(penalty, "l1_ratio", 1.0))
            if backend in ("torch", "cupy"):
                abs_sum, sum_sq = _sync_scalars(
                    _abs_sum_dev(coef_eval), _sum_sq_dev(coef_eval), backend=backend,
                )
            else:
                abs_sum = _abs_sum(coef_eval)
                sum_sq = _sum_sq(coef_eval)
            return alpha * (l1_ratio * abs_sum + 0.5 * (1.0 - l1_ratio) * sum_sq)
    try:
        return float(penalty.value(coef))
    except (ValueError, TypeError, AttributeError):
        pass
    try:
        return float(penalty.value(_to_numpy(coef)))
    except (ValueError, TypeError, AttributeError):
        pass
    return 0.0


def _abs_mean_max(y, backend):
    backend = _resolve_backend(backend, y)
    xp = _get_xp(backend)
    y_abs = xp.abs(y)
    mean_val, max_val = _sync_scalars(xp.mean(y_abs), xp.max(y_abs), backend=backend)
    return mean_val, max_val


def _smooth_penalty_gradient(penalty, coef):
    if penalty is None or _penalty_name(penalty) in ("none", "null"):
        return _zeros_like(coef)
    if hasattr(penalty, "smooth_gradient"):
        return penalty.smooth_gradient(coef)
    if _penalty_name(penalty) == "l2":
        return float(getattr(penalty, "alpha", 0.0)) * coef
    if _penalty_name(penalty) == "elasticnet":
        alpha = float(getattr(penalty, "alpha", 0.0))
        l1_ratio = float(getattr(penalty, "l1_ratio", 1.0))
        return alpha * (1.0 - l1_ratio) * coef
    raise ValueError(
        f"solver requires a smooth penalty, got penalty='{_penalty_name(penalty)}'."
    )


def _smooth_penalty_hessian(penalty, coef):
    if penalty is None or _penalty_name(penalty) in ("none", "null"):
        return 0.0
    n = coef.shape[0]
    if hasattr(penalty, "smooth_hessian"):
        return penalty.smooth_hessian(coef)
    if _penalty_name(penalty) == "l2":
        return float(getattr(penalty, "alpha", 0.0)) * _eye_like(n, coef)
    if _penalty_name(penalty) in ("elasticnet", "en"):
        # ElasticNet Hessian is the L2 component only (L1 is non-smooth)
        alpha = float(getattr(penalty, "alpha", 0.0))
        l1_ratio = float(getattr(penalty, "l1_ratio", 0.5))
        return alpha * (1.0 - l1_ratio) * _eye_like(n, coef)
    raise ValueError(
        f"solver requires a smooth penalty, got penalty='{_penalty_name(penalty)}'."
    )


def _objective_value(loss, penalty, X, y, coef):
    return float(_to_numpy(loss.value(X, y, coef))) + _smooth_penalty_value(penalty, coef)


def _objective_gradient(loss, penalty, X, y, coef):
    return loss.gradient(X, y, coef) + _smooth_penalty_gradient(penalty, coef)


def _smooth_penalty_lipschitz(penalty):
    if penalty is None:
        return 0.0
    _pname = _penalty_name(penalty)
    if _pname in ("none", "null", "l1", "scad", "mcp", "adaptive_l1", "adaptive_lasso",
                  "group_lasso", "group_mcp", "group_scad", "gl", "gmcp", "gscad", "en"):
        return 0.0
    alpha = float(getattr(penalty, 'alpha', 0.0))
    l1_ratio = float(getattr(penalty, 'l1_ratio', 0.0))
    return alpha * (1.0 - l1_ratio)


def _smooth_penalty_value_dev(penalty, coef):
    if penalty is None:
        return 0.0
    if hasattr(penalty, "smooth_value"):
        return penalty.smooth_value(coef)
    pname = _penalty_name(penalty)
    if pname in ("none", "null"):
        return 0.0
    if pname == "l2":
        return 0.5 * float(getattr(penalty, "alpha", 0.0)) * _sum_sq_dev(coef)
    if pname == "elasticnet":
        alpha = float(getattr(penalty, "alpha", 0.0))
        l1_ratio = float(getattr(penalty, "l1_ratio", 1.0))
        return 0.5 * alpha * (1.0 - l1_ratio) * _sum_sq_dev(coef)
    raise ValueError(
        f"smooth_penalty_value_dev requires a smooth penalty, got '{pname}'."
    )


def _objective_value_dev(loss, penalty, X, y, coef):
    val = loss.value(X, y, coef)
    pen_val = _smooth_penalty_value_dev(penalty, coef)
    if isinstance(pen_val, (int, float)) and pen_val == 0.0:
        return val
    return val + pen_val
