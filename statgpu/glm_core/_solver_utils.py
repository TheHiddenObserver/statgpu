"""
Shared helpers for GLM solvers.

Extracted from _solver.py to improve maintainability. Contains:
- Constants and convergence thresholds
- torch.compile lazy-loaders
- Validation helpers
- Penalty value/gradient/hessian utilities
- Fused GLM loss+gradient functions
"""

__all__ = ["ConvergenceWarning"]


import warnings
import numpy as np

from statgpu.backends import _resolve_backend, _to_numpy
from statgpu.backends._utils import _to_float_scalar, _get_xp
from statgpu.backends._array_ops import (
    _abs_sum,
    _abs_sum_dev,
    _clip_grad_on_device,
    _copy_arr,
    _device_gt,
    _device_leq,
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_LIPSCHITZ_SAFETY_LOGISTIC_CV = 2.0
_SLACK_TOLERANCE = 1e-14
_DIVERGE_COEF_NORM_CAP = 100.0
_DIVERGE_OBJ_RATIO = 100.0
_DIVERGE_OBJ_ABS = 10.0
_BB_RESTART_DOT_TOL = 1e-14
_LIPSCHITZ_FLOOR = 1e-30


class ConvergenceWarning(UserWarning):
    """Solver did not converge within the iteration limit."""
    pass


# ---------------------------------------------------------------------------
# torch.compile lazy-loaders
# ---------------------------------------------------------------------------
_FISTA_STEP_COMPILED = None
_NEWTON_STEP_COMPILED = None

from statgpu.backends._utils import torch_compile_supported as _torch_compile_supported


def _get_fista_step_compiled():
    global _FISTA_STEP_COMPILED
    if _FISTA_STEP_COMPILED is not None:
        return _FISTA_STEP_COMPILED
    import torch
    def _fista_step(y_k, grad, step, coef_old, coef, beta_t):
        w_tilde = y_k - step * grad
        y_k_new = coef + beta_t * (coef - coef_old)
        return w_tilde, y_k_new
    if _torch_compile_supported():
        try:
            _FISTA_STEP_COMPILED = torch.compile(_fista_step, dynamic=True, fullgraph=False)
        except RuntimeError:
            _FISTA_STEP_COMPILED = _fista_step
    else:
        _FISTA_STEP_COMPILED = _fista_step
    return _FISTA_STEP_COMPILED


def _fista_step_call(compiled_fn, *args):
    try:
        return compiled_fn(*args)
    except (RuntimeError, TypeError):
        def _fista_eager(y_k, grad, step, coef_old, coef, beta_t):
            w_tilde = y_k - step * grad
            y_k_new = coef + beta_t * (coef - coef_old)
            return w_tilde, y_k_new
        return _fista_eager(*args)


def _get_newton_step_compiled():
    global _NEWTON_STEP_COMPILED
    if _NEWTON_STEP_COMPILED is not None:
        return _NEWTON_STEP_COMPILED
    import torch
    def _newton_step(params, direction, params_old):
        params_new = params - direction
        diff_norm = torch.linalg.norm(params_new - params_old)
        return params_new, diff_norm
    if _torch_compile_supported():
        try:
            _NEWTON_STEP_COMPILED = torch.compile(_newton_step, dynamic=True, fullgraph=False)
        except RuntimeError:
            _NEWTON_STEP_COMPILED = _newton_step
    else:
        _NEWTON_STEP_COMPILED = _newton_step
    return _NEWTON_STEP_COMPILED


def _newton_step_call(compiled_fn, *args):
    try:
        return compiled_fn(*args)
    except (RuntimeError, TypeError):
        def _newton_eager(params, direction, params_old):
            params_new = params - direction
            diff_norm = torch.linalg.norm(params_new - params_old)
            return params_new, diff_norm
        return _newton_eager(*args)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Penalty utilities
# ---------------------------------------------------------------------------
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
    # SelectivePenalty or other wrappers without value() — return 0
    # (the penalty effect is captured in the proximal operator, not the value)
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
                  "group_lasso", "group_mcp", "group_scad", "gl", "gmcp", "gscad"):
        return 0.0
    alpha = float(getattr(penalty, 'alpha', 0.0))
    l1_ratio = float(getattr(penalty, 'l1_ratio', 0.0))
    return alpha * (1.0 - l1_ratio)


def _smooth_penalty_value_dev(penalty, coef):
    if penalty is None:
        return 0.0
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


# ---------------------------------------------------------------------------
# Fused GLM loss+gradient (avoids redundant X @ coef)
# ---------------------------------------------------------------------------
def _fused_logistic(eta, X, y, n, loss):
    from statgpu.backends._array_ops import _sigmoid, _softplus, _sum
    p = _sigmoid(eta)
    log1pexp = _softplus(eta)
    val = _sum(-y * eta + log1pexp) / n
    grad = X.T @ (p - y) / n
    return val, grad


def _fused_poisson(eta, X, y, n, loss):
    from statgpu.backends._array_ops import _exp, _log, _clip, _sum
    mu = _exp(_clip(eta, -30, 30))
    mu_c = _clip(mu, 1e-10, None)
    val = _sum(mu - y * _log(mu_c)) / n
    grad = X.T @ (mu - y) / n
    return val, grad


def _fused_gamma(eta, X, y, n, loss):
    from statgpu.backends._array_ops import _exp, _log, _clip, _sum
    gamma_link = getattr(loss, 'link_name', getattr(loss, 'link', 'log'))
    if gamma_link == 'inverse_power':
        eta_lo = float(getattr(loss, '_ETA_LO', 1e-2))
        eta_hi = float(getattr(loss, '_ETA_HI', 1e3))
        eta_c = _clip(eta, eta_lo, eta_hi)
        mu = 1.0 / eta_c
        val = _sum(y * eta_c - _log(eta_c)) / n
        grad = X.T @ (y - mu) / n
        return val, grad
    mu = _exp(_clip(eta, -30, 30))
    mu_c = _clip(mu, 1e-10, None)
    val = _sum(y / mu_c + _log(mu_c)) / n
    grad = X.T @ ((mu_c - y) / mu_c) / n
    return val, grad


def _fused_negative_binomial(eta, X, y, n, loss):
    from statgpu.backends._array_ops import _exp, _log, _clip, _sum
    a = float(getattr(loss, 'alpha', 1.0))
    mu = _exp(_clip(eta, -30, 30))
    mu_c = _clip(mu, 1e-300, None)
    one_plus_a_mu = 1.0 + a * mu_c
    val = _sum(-y * _log(mu_c / one_plus_a_mu) + (1.0 / a) * _log(one_plus_a_mu)) / n
    grad = X.T @ ((mu_c - y) / one_plus_a_mu) / n
    return val, grad


def _fused_tweedie(eta, X, y, n, loss):
    from statgpu.backends._array_ops import _exp, _clip, _sum, _log
    pw = float(getattr(loss, 'power', 1.5))
    mu = _exp(_clip(eta, -50, 50))
    mu_c = _clip(mu, 1e-10, 1e6)
    log_mu = _log(mu_c)
    d1 = 1.0 - pw
    d2 = 2.0 - pw
    if abs(d1) < 0.01:
        term1 = -y * log_mu
    else:
        term1 = -y * mu_c ** d1 / d1
    if abs(d2) < 0.01:
        term2 = log_mu
    else:
        term2 = mu_c ** d2 / d2
    val = _sum(term1 + term2) / n
    grad = X.T @ (mu_c ** d1 * (mu_c - y)) / n
    return val, grad


def _fused_inverse_gaussian(eta, X, y, n, loss):
    from statgpu.backends._array_ops import _exp, _clip, _sum
    mu = _exp(_clip(eta, -30, 30))
    mu_c = _clip(mu, 5e-2, 1e3)
    val = _sum(y / (2.0 * mu_c * mu_c) - 1.0 / mu_c) / n
    grad = X.T @ ((mu_c - y) / (mu_c * mu_c)) / n
    return val, grad


def _fused_glm_value_and_gradient(loss, X, y, coef):
    n = X.shape[0]
    eta = X @ coef
    loss_name = getattr(loss, 'name', '')
    _fused_map = {
        'logistic': _fused_logistic,
        'poisson': _fused_poisson,
        'gamma': _fused_gamma,
        'negative_binomial': _fused_negative_binomial,
        'tweedie': _fused_tweedie,
        'inverse_gaussian': _fused_inverse_gaussian,
    }
    if loss_name in _fused_map:
        return _fused_map[loss_name](eta, X, y, n, loss)
    return loss.value(X, y, coef), loss.gradient(X, y, coef)


def _weighted_loss_and_grad(loss, X, y, coef, sample_weight):
    n = X.shape[0]
    _backend = _resolve_backend("auto", X)
    xp = _get_xp(_backend)
    _sw_np = _to_numpy(sample_weight)
    if hasattr(X, 'device'):
        _sw = xp.asarray(_sw_np, dtype=X.dtype, device=X.device)
    else:
        _sw = xp.asarray(_sw_np, dtype=X.dtype)
    sw_sum = _to_float_scalar(xp.sum(_sw))

    loss_name = getattr(loss, 'name', '')
    if loss_name == 'squared_error':
        resid = X @ coef - y
        grad = X.T @ (_sw * resid) / sw_sum
        val = 0.5 * _to_float_scalar(xp.sum(_sw * resid * resid)) / sw_sum
        return val, grad

    if hasattr(loss, 'fused_value_and_gradient'):
        try:
            return loss.fused_value_and_gradient(X, y, coef, sample_weight=sample_weight)
        except TypeError:
            pass

    try:
        val = loss.value(X, y, coef, sample_weight=sample_weight)
        grad = loss.gradient(X, y, coef, sample_weight=sample_weight)
        return val, grad
    except TypeError:
        val = loss.value(X, y, coef)
        grad = loss.gradient(X, y, coef)
        return val, grad
