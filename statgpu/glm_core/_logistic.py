"""
Logistic loss: negative Bernoulli log-likelihood.

For binary classification:
    loss = (1/n) * sum(-y*z + log(1 + exp(z)))
where z = X @ coef.

Supports numpy / cupy / torch backends via _backend helpers.
"""
from statgpu.backends._array_ops import _clip, _log1p, _exp, _sigmoid, _sum, _max_eigval_power
from ._base import GLMLoss, register_glm_loss


@register_glm_loss('logistic')
class LogisticLoss(GLMLoss):
    name = "logistic"
    y_type = "binary"
    smooth_gradient = True
    has_hessian = True

    def value(self, X, y, coef):
        """Negative Bernoulli log-likelihood (to minimize)."""
        z = X @ coef
        return _sum(-y * z + _log1p(_exp(_clip(z, -500, 500)))) / X.shape[0]

    def gradient(self, X, y, coef):
        z = X @ coef
        p = _sigmoid(z)
        return X.T @ (p - y) / X.shape[0]

    def hessian(self, X, y, coef):
        z = X @ coef
        p = _sigmoid(z)
        W = _clip(p * (1.0 - p), 1e-10, 1.0 - 1e-10)
        return X.T @ (X * W[:, None]) / X.shape[0]

    def lipschitz(self, X, coef, y=None):
        # Global Lipschitz: L = lambda_max(X'X) / (4n)
        XtX = X.T @ X
        return _max_eigval_power(XtX) / (4.0 * X.shape[0])

    def predict(self, X, coef):
        z = X @ coef
        p = _sigmoid(z)
        if hasattr(p, 'numpy'):
            return (p > 0.5).cpu().numpy()
        elif hasattr(p, 'get'):
            return (p > 0.5).get()
        return p > 0.5
