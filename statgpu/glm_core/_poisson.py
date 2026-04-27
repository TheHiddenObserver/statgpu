"""
Poisson loss: negative Poisson log-likelihood.

For count data:
    loss = (1/n) * sum(mu - y*log(mu))
where mu = exp(X @ coef).

Supports numpy / cupy / torch backends via _backend helpers.
"""
from statgpu.backends._array_ops import _clip, _exp, _log, _eigvalsh, _sum
from ._base import GLMLoss, register_glm_loss


@register_glm_loss('poisson')
class PoissonLoss(GLMLoss):
    name = "poisson"
    y_type = "count"
    smooth_gradient = True
    has_hessian = True

    def value(self, X, y, coef):
        """Negative Poisson log-likelihood (to minimize)."""
        z = _clip(X @ coef, -500, 500)
        mu = _exp(z)
        return _sum(mu - y * _log(_clip(mu, 1e-300, None))) / X.shape[0]

    def gradient(self, X, y, coef):
        """Gradient = X'(mu - y) / n."""
        z = _clip(X @ coef, -500, 500)
        mu = _exp(z)
        return X.T @ (mu - y) / X.shape[0]

    def hessian(self, X, y, coef):
        """Hessian = X'WX / n, W = diag(mu)."""
        z = _clip(X @ coef, -500, 500)
        mu = _exp(z)
        W = _clip(mu, 1e-10, None)
        return X.T @ (X * W[:, None]) / X.shape[0]

    def lipschitz(self, X, coef):
        z = _clip(X @ coef, -500, 500)
        mu = _exp(z)
        # Cap mu to prevent explosion
        W = _clip(mu, 1e-10, 1e6)
        XtWX = X.T @ (X * W[:, None])
        L = float(_eigvalsh(XtWX)[-1]) / X.shape[0]
        return max(L, 1e-8)

    def predict(self, X, coef):
        return _exp(X @ coef)
