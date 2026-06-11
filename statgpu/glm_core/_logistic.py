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

    def value(self, X, y, coef, sample_weight=None):
        """Negative Bernoulli log-likelihood (to minimize)."""
        z = X @ coef
        xp = __import__(type(z).__module__.split(".")[0])
        if xp.__name__ == "torch":
            log1pexp = _log1p(_exp(-xp.abs(z))) + xp.clamp(z, min=0)
        else:
            log1pexp = _log1p(_exp(-xp.abs(z))) + xp.maximum(z, 0)
        nll = -y * z + log1pexp
        if sample_weight is not None:
            return _sum(sample_weight * nll) / sample_weight.sum()
        return _sum(nll) / X.shape[0]

    def gradient(self, X, y, coef, sample_weight=None):
        z = X @ coef
        p = _sigmoid(z)
        resid = p - y
        if sample_weight is not None:
            return X.T @ (sample_weight * resid) / sample_weight.sum()
        return X.T @ resid / X.shape[0]

    def hessian(self, X, y, coef, sample_weight=None):
        z = X @ coef
        p = _sigmoid(z)
        W = _clip(p * (1.0 - p), 1e-10, 1.0 - 1e-10)
        if sample_weight is not None:
            W = W * sample_weight
            return X.T @ (X * W[:, None]) / sample_weight.sum()
        return X.T @ (X * W[:, None]) / X.shape[0]

    def lipschitz(self, X, coef, y=None, sample_weight=None):
        # Global bound: L_global = lambda_max(X'X) / (4n)
        n_eff = sample_weight.sum() if sample_weight is not None else X.shape[0]
        if sample_weight is not None:
            sw = sample_weight[:, None] if hasattr(sample_weight, '__len__') else sample_weight
            XtWX = X.T @ (X * sw)
            L_global = _max_eigval_power(XtWX) / (4.0 * n_eff)
        else:
            XtX = X.T @ X
            L_global = _max_eigval_power(XtX) / (4.0 * n_eff)
        if coef is not None:
            z = X @ coef
            p = _sigmoid(z)
            W = _clip(p * (1.0 - p), 1e-10, 0.25)
            XtWX = X.T @ (X * W[:, None])
            L_iter = _max_eigval_power(XtWX) / X.shape[0]
            # Floor at 10% of global bound to prevent overshoot near optimum
            return max(L_iter, L_global * 0.1)
        return L_global

    def predict(self, X, coef):
        z = X @ coef
        p = _sigmoid(z)
        if hasattr(p, 'numpy'):
            return (p > 0.5).cpu().numpy()
        elif hasattr(p, 'get'):
            return (p > 0.5).get()
        return p > 0.5
