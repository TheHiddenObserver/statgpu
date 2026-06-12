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

    # ── Per-sample formulas (single source of truth) ──────────────────

    def per_sample_value(self, eta, y):
        """Negative Bernoulli log-likelihood per sample."""
        xp = __import__(type(eta).__module__.split(".")[0])
        if xp.__name__ == "torch":
            max_eta = xp.clamp(eta, min=0)
        else:
            max_eta = xp.maximum(eta, 0)
        log1pexp = _log1p(_exp(-xp.abs(eta))) + max_eta
        return -y * eta + log1pexp

    def per_sample_gradient(self, eta, y):
        return _sigmoid(eta) - y

    # ── Hessian / Lipschitz (override for weighted support) ───────────

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
        n_eff = float(sample_weight.sum()) if sample_weight is not None else X.shape[0]
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
            if sample_weight is not None:
                W = W * (sample_weight if sample_weight.ndim == 1 else sample_weight.ravel())
            XtWX = X.T @ (X * W[:, None])
            L_iter = _max_eigval_power(XtWX) / n_eff
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
