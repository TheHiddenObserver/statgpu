"""
Negative Binomial loss: negative log-likelihood with log link.

For overdispersed count data:
    Var(Y) = mu + alpha * mu^2
where mu = exp(X @ coef), alpha is the dispersion parameter.

Supports numpy / cupy / torch backends via _array_ops helpers.
"""
import numpy as np

from statgpu.backends._array_ops import _clip, _exp, _log, _sum, _max_eigval_power
from ._base import GLMLoss, register_glm_loss


@register_glm_loss('negative_binomial')
class NegativeBinomialLoss(GLMLoss):
    name = "negative_binomial"
    y_type = "count"
    smooth_gradient = True
    has_hessian = True

    _MU_LO = 1e-300

    def __init__(self, alpha=1.0):
        if not np.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("alpha must be a finite positive scalar for negative binomial loss")
        self.alpha = alpha

    def _mu_from_eta(self, eta):
        return _clip(_exp(_clip(eta, -30, 30)), self._MU_LO, None)

    # ── Per-sample formulas (single source of truth) ──────────────────

    def per_sample_value(self, eta, y):
        mu = self._mu_from_eta(eta)
        a = self.alpha
        one_plus_a_mu = 1.0 + a * mu
        return -y * _log(mu / one_plus_a_mu) + (1.0 / a) * _log(one_plus_a_mu)

    def per_sample_gradient(self, eta, y):
        mu = self._mu_from_eta(eta)
        return (mu - y) / (1.0 + self.alpha * mu)

    def hessian(self, X, y, coef, sample_weight=None):
        z = _clip(X @ coef, -30, 30)
        mu = _exp(z)
        W = _clip(mu, 1e-10, None) / (1.0 + self.alpha * _clip(mu, 1e-10, None))
        if sample_weight is not None:
            W = W * sample_weight
        n_eff = float(sample_weight.sum()) if sample_weight is not None else X.shape[0]
        return X.T @ (X * W[:, None]) / n_eff

    _lipschitz_safety = 2.0  # NB Hessian varies moderately with mu

    def lipschitz(self, X, coef, y=None, sample_weight=None):
        z = _clip(X @ coef, -30, 30)
        mu = _exp(z)
        W = _clip(mu, 1e-10, 1e6) / (1.0 + self.alpha * _clip(mu, 1e-10, 1e6))
        if sample_weight is not None:
            W = W * sample_weight
        n_eff = float(sample_weight.sum()) if sample_weight is not None else X.shape[0]
        XtWX = X.T @ (X * W[:, None])
        L = _max_eigval_power(XtWX) / n_eff
        return max(L, 1e-8)  # Safety factor applied by solver via _lipschitz_safety

    def predict(self, X, coef):
        return _exp(X @ coef)
