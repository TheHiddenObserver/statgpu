"""
Gamma loss: negative Gamma log-likelihood with log link.

For positive continuous outcomes:
    loss = (1/n) * sum(y/mu + log(mu))
where mu = exp(X @ coef).

Supports numpy / cupy / torch backends via _array_ops helpers.
"""
from statgpu.backends._array_ops import _clip, _exp, _log, _sum, _max_eigval_power
from ._base import GLMLoss, register_glm_loss


@register_glm_loss('gamma')
class GammaLoss(GLMLoss):
    name = "gamma"
    y_type = "positive"
    smooth_gradient = True
    has_hessian = True
    _lipschitz_uses_y = True

    _MU_LO = 1e-3
    _MU_HI = 1e4

    def value(self, X, y, coef):
        z = _clip(X @ coef, -30, 30)
        mu = _clip(_exp(z), self._MU_LO, self._MU_HI)
        return _sum(y / mu + _log(mu)) / X.shape[0]

    def gradient(self, X, y, coef):
        z = _clip(X @ coef, -30, 30)
        mu = _clip(_exp(z), self._MU_LO, self._MU_HI)
        return X.T @ (1.0 - y / mu) / X.shape[0]

    def hessian(self, X, y, coef):
        # Expected Fisher: W(mu) = 1 for Gamma with log link.
        return X.T @ X / X.shape[0]

    def lipschitz(self, X, coef, y=None):
        if y is not None:
            z = _clip(X @ coef, -30, 30)
            mu = _clip(_exp(z), self._MU_LO, self._MU_HI)
            W = y / mu
            XtWX = X.T @ (X * W[:, None])
            L = _max_eigval_power(XtWX) / X.shape[0]
        else:
            XtX = X.T @ X
            L = _max_eigval_power(XtX) / X.shape[0]
        return max(L, 1e-8)

    def predict(self, X, coef):
        return _exp(X @ coef)
