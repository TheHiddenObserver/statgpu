"""GLM-specific fused loss+gradient functions.

These avoid redundant X @ coef computation by computing value and gradient
in a single pass. They are called by GLMLoss subclasses' fused_value_and_gradient()
methods for performance.

NOT part of the generic solver interface — these are GLM internal optimizations.
"""

import numpy as np

from statgpu.backends import _resolve_backend
from statgpu.backends._utils import _to_float_scalar, _get_xp
from statgpu.backends._array_ops import _to_numpy


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
    gamma_link = getattr(loss, "link_name", getattr(loss, "link", "log"))
    if gamma_link == "inverse_power":
        eta_lo = float(getattr(loss, "_ETA_LO", 1e-2))
        eta_hi = float(getattr(loss, "_ETA_HI", 1e3))
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
    a = float(getattr(loss, "alpha", 1.0))
    mu = _exp(_clip(eta, -30, 30))
    mu_c = _clip(mu, 1e-300, None)
    one_plus_a_mu = 1.0 + a * mu_c
    val = (
        _sum(-y * _log(mu_c / one_plus_a_mu) + (1.0 / a) * _log(one_plus_a_mu)) / n
    )
    grad = X.T @ ((mu_c - y) / one_plus_a_mu) / n
    return val, grad


def _fused_tweedie(eta, X, y, n, loss):
    from statgpu.backends._array_ops import _exp, _clip, _sum, _log
    pw = float(getattr(loss, "power", 1.5))
    mu = _exp(_clip(eta, -50, 50))
    mu_c = _clip(mu, 1e-10, 1e6)
    log_mu = _log(mu_c)
    d1 = 1.0 - pw
    d2 = 2.0 - pw
    if abs(d1) < 0.01:
        term1 = -y * log_mu
    else:
        term1 = -y * mu_c**d1 / d1
    if abs(d2) < 0.01:
        term2 = log_mu
    else:
        term2 = mu_c**d2 / d2
    val = _sum(term1 + term2) / n
    grad = X.T @ (mu_c**d1 * (mu_c - y)) / n
    return val, grad


def _fused_inverse_gaussian(eta, X, y, n, loss):
    from statgpu.backends._array_ops import _exp, _clip, _sum
    mu = _exp(_clip(eta, -30, 30))
    mu_c = _clip(mu, 5e-2, 1e3)
    val = _sum(y / (2.0 * mu_c * mu_c) - 1.0 / mu_c) / n
    grad = X.T @ ((mu_c - y) / (mu_c * mu_c)) / n
    return val, grad


def _fused_glm_value_and_gradient(loss, X, y, coef):
    """Dispatch to fused kernel based on loss name (GLM-specific)."""
    n = X.shape[0]
    eta = X @ coef
    loss_name = getattr(loss, "name", "")
    _fused_map = {
        "logistic": _fused_logistic,
        "poisson": _fused_poisson,
        "gamma": _fused_gamma,
        "negative_binomial": _fused_negative_binomial,
        "tweedie": _fused_tweedie,
        "inverse_gaussian": _fused_inverse_gaussian,
    }
    if loss_name in _fused_map:
        return _fused_map[loss_name](eta, X, y, n, loss)
    return loss.value(X, y, coef), loss.gradient(X, y, coef)


def _weighted_loss_and_grad(loss, X, y, coef, sample_weight):
    """Weighted loss+gradient (GLM-specific fast paths)."""
    n = X.shape[0]
    _backend = _resolve_backend("auto", X)
    xp = _get_xp(_backend)
    _sw_np = _to_numpy(sample_weight)
    if hasattr(X, "device"):
        _sw = xp.asarray(_sw_np, dtype=X.dtype, device=X.device)
    else:
        _sw = xp.asarray(_sw_np, dtype=X.dtype)
    sw_sum = _to_float_scalar(xp.sum(_sw))

    loss_name = getattr(loss, "name", "")
    if loss_name == "squared_error":
        resid = X @ coef - y
        grad = X.T @ (_sw * resid) / sw_sum
        val = 0.5 * _to_float_scalar(xp.sum(_sw * resid * resid)) / sw_sum
        return val, grad

    if hasattr(loss, "fused_value_and_gradient"):
        try:
            return loss.fused_value_and_gradient(
                X, y, coef, sample_weight=sample_weight
            )
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
