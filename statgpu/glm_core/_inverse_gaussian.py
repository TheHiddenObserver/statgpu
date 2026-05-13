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

    def value(self, X, y, coef):
        z = _clip(X @ coef, -30, 30)
        mu = _clip(_exp(z), 5e-2, 1e3)
        return _sum(y / (2.0 * mu * mu) - 1.0 / mu) / X.shape[0]

    def gradient(self, X, y, coef):
        z = _clip(X @ coef, -30, 30)
        mu = _clip(_exp(z), 5e-2, 1e3)
        return X.T @ ((mu - y) / (mu * mu)) / X.shape[0]

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
        return max(min(L, 1e3), 1e-8)

    def predict(self, X, coef):
        return _exp(X @ coef)
