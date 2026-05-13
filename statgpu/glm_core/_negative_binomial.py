"""
Negative Binomial loss: negative log-likelihood with log link.

For overdispersed count data:
    Var(Y) = mu + alpha * mu^2
where mu = exp(X @ coef), alpha is the dispersion parameter.

Supports numpy / cupy / torch backends via _array_ops helpers.
"""
from statgpu.backends._array_ops import _clip, _exp, _log, _sum, _max_eigval_power
from ._base import GLMLoss, register_glm_loss


@register_glm_loss('negative_binomial')
class NegativeBinomialLoss(GLMLoss):
    name = "negative_binomial"
    y_type = "count"
    smooth_gradient = True
    has_hessian = True

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def value(self, X, y, coef):
        z = _clip(X @ coef, -30, 30)
        mu = _exp(z)
        mu_c = _clip(mu, 1e-300, None)
        a_plus_mu = self.alpha + mu_c
        return _sum(
            -y * _log(mu_c / a_plus_mu)
            - (1.0 / self.alpha) * _log(self.alpha / a_plus_mu)
        ) / X.shape[0]

    def gradient(self, X, y, coef):
        z = _clip(X @ coef, -30, 30)
        mu = _exp(z)
        return X.T @ ((mu - y) / (1.0 + self.alpha * mu)) / X.shape[0]

    def hessian(self, X, y, coef):
        z = _clip(X @ coef, -30, 30)
        mu = _exp(z)
        W = _clip(mu, 1e-10, None) / (1.0 + self.alpha * _clip(mu, 1e-10, None))
        return X.T @ (X * W[:, None]) / X.shape[0]

    def lipschitz(self, X, coef, y=None):
        z = _clip(X @ coef, -30, 30)
        mu = _exp(z)
        W = _clip(mu, 1e-10, 1e6) / (1.0 + self.alpha * _clip(mu, 1e-10, 1e6))
        XtWX = X.T @ (X * W[:, None])
        L = _max_eigval_power(XtWX) / X.shape[0]
        return max(L, 1e-8)

    def predict(self, X, coef):
        return _exp(X @ coef)
