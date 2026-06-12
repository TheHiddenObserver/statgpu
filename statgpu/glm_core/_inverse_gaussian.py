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

    def hessian(self, X, y, coef):
        z = _clip(X @ coef, -30, 30)
        mu = _clip(_exp(z), 5e-2, 1e3)
        # Expected Fisher: W(mu) = 1/mu.
        W = 1.0 / mu
        return X.T @ (X * W[:, None]) / X.shape[0]

    def lipschitz(self, X, coef, y=None):
        z = _clip(X @ coef, -30, 30)
        mu = _clip(_exp(z), 5e-2, 1e3)
        W = 1.0 / mu
        XtWX = X.T @ (X * W[:, None])
        L = _max_eigval_power(XtWX) / X.shape[0]
        # Dynamic upper bound: scale with data to avoid underestimating
        # curvature for large y values.
        import numpy as np
        _y_np = y if y is not None else None
        if _y_np is not None:
            from statgpu.backends import _to_numpy
            _y_abs_mean = float(np.mean(np.abs(_to_numpy(_y_np))))
            upper = max(1e6, 100.0 * _y_abs_mean)
        else:
            upper = 1e6
        return max(min(L, upper), 1e-8)

    def predict(self, X, coef):
        return _exp(X @ coef)
