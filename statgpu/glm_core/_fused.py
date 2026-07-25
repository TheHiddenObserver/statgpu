"""GLM-specific fused loss+gradient functions.

These avoid redundant X @ coef computation by computing value and gradient
in a single pass. They are called by GLMLoss subclasses' fused_value_and_gradient()
methods for performance.

NOT part of the generic solver interface — these are GLM internal optimizations.
"""

import numpy as np

from statgpu.backends import _resolve_backend
from statgpu.backends._utils import _to_float_scalar, _get_xp
from statgpu.backends import _to_numpy


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
    eta_lo = float(getattr(loss, "_LOG_ETA_LO", -30.0))
    eta_hi = float(getattr(loss, "_LOG_ETA_HI", 30.0))
    eta_c = _clip(eta, eta_lo, eta_hi)
    ratio = y * _exp(-eta_c)
    val = _sum(eta_c + ratio) / n
    grad = X.T @ (1.0 - ratio) / n
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
    z_clip = float(getattr(loss, "_Z_CLIP", 50.0))
    mu_c = _exp(_clip(eta, -z_clip, z_clip))
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
    """Weighted loss+gradient using per_sample_value/gradient directly.

    Uses per_sample_value() and per_sample_gradient() to compute
    weighted results without calling back into fused_value_and_gradient(),
    avoiding infinite recursion when fused_value_and_gradient itself
    delegates to this helper for weighted computations.
    """
    from statgpu.backends import xp_asarray

    _backend = _resolve_backend("auto", X)
    xp = _get_xp(_backend)

    sw = xp_asarray(sample_weight, dtype=X.dtype, xp=xp, ref_arr=X).reshape(-1)

    if sw.ndim != 1 or sw.shape[0] != X.shape[0]:
        raise ValueError(
            "sample_weight must be one-dimensional with length n_samples"
        )

    loss_name = getattr(loss, "name", "")
    if loss_name == "squared_error":
        resid = X @ coef - y
        weight_sum = _to_float_scalar(xp.sum(sw))
        grad = X.T @ (sw * resid) / weight_sum
        val = 0.5 * _to_float_scalar(xp.sum(sw * resid * resid)) / weight_sum
        return val, grad

    # Compute per-sample loss and gradient, then apply weights.
    # This avoids calling fused_value_and_gradient() which would recurse
    # back into this same function when sample_weight is present.
    eta = X @ coef
    per_value = loss.per_sample_value(eta, y)
    per_gradient = loss.per_sample_gradient(eta, y)

    weight_sum = xp.sum(sw)
    val = xp.sum(sw * per_value) / weight_sum
    grad = X.T @ (sw * per_gradient) / weight_sum

    return val, grad
