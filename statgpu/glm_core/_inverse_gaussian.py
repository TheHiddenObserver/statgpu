"""
Inverse Gaussian loss: negative log-likelihood with log link.

For positive right-skewed outcomes:
    loss = (1/n) * sum(y/(2*mu^2) - 1/mu)
where mu = exp(X @ coef).

Supports numpy / cupy / torch backends via _array_ops helpers.
"""
from statgpu.backends._array_ops import _clip, _exp, _sum, _max_eigval_power
from ._base import GLMLoss, register_glm_loss


@register_glm_loss('inverse_gaussian')
class InverseGaussianLoss(GLMLoss):
    name = "inverse_gaussian"
    y_type = "positive"
    smooth_gradient = True
    has_hessian = True
    _lipschitz_uses_y = True
    _lipschitz_safety = 3.0  # 1/mu^3 gradient scaling requires safety factor

    _MU_LO = 5e-2
    _MU_HI = 1e3

    def _mu_from_eta(self, eta):
        return _clip(_exp(_clip(eta, -30, 30)), self._MU_LO, self._MU_HI)

    # ── Per-sample formulas (single source of truth) ──────────────────

    def per_sample_value(self, eta, y):
        mu = self._mu_from_eta(eta)
        return y / (2.0 * mu * mu) - 1.0 / mu

    def per_sample_gradient(self, eta, y):
        mu = self._mu_from_eta(eta)
        return (mu - y) / (mu * mu)

    def hessian(self, X, y, coef, sample_weight=None):
        z = _clip(X @ coef, -30, 30)
        mu = _clip(_exp(z), self._MU_LO, self._MU_HI)
        W = 1.0 / mu
        if sample_weight is not None:
            W = W * sample_weight
        n_eff = float(sample_weight.sum()) if sample_weight is not None else X.shape[0]
        return X.T @ (X * W[:, None]) / n_eff

    def lipschitz(self, X, coef, y=None, sample_weight=None):
        z = _clip(X @ coef, -30, 30)
        mu = _clip(_exp(z), self._MU_LO, self._MU_HI)
        W = 1.0 / mu
        if sample_weight is not None:
            W = W * sample_weight
        n_eff = float(sample_weight.sum()) if sample_weight is not None else X.shape[0]
        XtWX = X.T @ (X * W[:, None])
        L = _max_eigval_power(XtWX) / n_eff
        return max(L, 1e-8)

    def predict(self, X, coef):
        return _exp(X @ coef)
