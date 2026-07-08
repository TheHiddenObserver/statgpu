"""
Dispersion (scale) parameter helpers for M-estimation inference.

Provides model-based dispersion estimates for GLM families
and robust (M-estimation) losses.
"""

__all__ = [
    "glm_pearson_dispersion",
    "robust_scale_dispersion",
]

import numpy as np


def glm_pearson_dispersion(loss, X, y, coef, df_resid):
    """Pearson chi-squared dispersion for GLM families.

    φ = sum_i (y_i - mu_i)^2 / V(mu_i) / df_resid

    Uses ``loss._mu_from_eta`` (GLMLoss) or identity for Gaussian.

    Parameters
    ----------
    loss : LossBase (typically GLMLoss)
    X : ndarray (n, p)
    y : ndarray (n,)
    coef : ndarray (p,)
    df_resid : int
        Residual degrees of freedom (n - k).

    Returns
    -------
    phi : float
        Pearson dispersion, or 1.0 if the loss has no mu_from_eta.
    """
    from statgpu.backends import _to_numpy

    if not hasattr(loss, '_mu_from_eta'):
        # Non-GLM loss (Huber, quantile, Cox, etc.) — use default
        return 1.0

    name = getattr(loss, 'name', '')
    if name in ("squared_error", "logistic"):
        # Gaussian: variance = 1. Binomial: variance = p(1-p).
        return 1.0

    X_np = np.asarray(_to_numpy(X), dtype=float)
    y_np = np.asarray(_to_numpy(y), dtype=float).ravel()
    coef_np = np.asarray(_to_numpy(coef), dtype=float)

    eta = X_np @ coef_np
    mu = loss._mu_from_eta(eta)

    # Compute variance function V(mu) per family
    if name == "poisson":
        V = np.clip(mu, 1e-10, None)
    elif name == "gamma":
        V = mu ** 2
    elif name == "inverse_gaussian":
        V = mu ** 3
    elif name == "negative_binomial":
        alpha = getattr(loss, 'alpha', 1.0)
        V = mu + alpha * (mu ** 2)
    elif name == "tweedie":
        p = getattr(loss, 'power', 1.5)
        V = mu ** p
    else:
        return 1.0

    resid_sq = (y_np - mu) ** 2
    pearson = np.sum(resid_sq / np.maximum(V, 1e-10))
    df = max(df_resid, 1)
    return float(pearson / df)


def robust_scale_dispersion(loss):
    """Return the scale estimate from a robust loss (Huber, Bisquare, Fair).

    These losses estimate scale during fitting (MAD or Huber Proposal 2).
    """
    if hasattr(loss, '_scale'):
        return float(loss._scale)
    if hasattr(loss, 'scale'):
        return float(loss.scale)
    return 1.0
