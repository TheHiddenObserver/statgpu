"""
Tweedie loss: negative Tweedie log-likelihood with log link.

For compound Poisson-Gamma (1 < p < 2) outcomes:
    loss = (1/n) * sum(-y * mu^(1-p)/(1-p) + mu^(2-p)/(2-p))
where mu = exp(X @ coef), p is the Tweedie power parameter.

Supports numpy / cupy / torch backends via _array_ops helpers.
"""
from statgpu.backends._array_ops import _clip, _exp, _sum, _max_eigval_power
from statgpu.glm_core._base import GLMLoss, register_glm_loss


@register_glm_loss('tweedie')
class TweedieLoss(GLMLoss):
    name = "tweedie"
    y_type = "nonnegative"
    smooth_gradient = True
    has_hessian = True
    _lipschitz_uses_y = True
    _lipschitz_safety = 5.0  # Tweedie variance function requires large safety

    # Clip z to [-50, 50] instead of [-500, 500] to prevent
    # mu^(-0.5) explosion: mu >= exp(-50) ~ 1.9e-22 -> mu^(-0.5) <= 2.3e10.
    _Z_CLIP = 50.0

    _MU_LO = 1e-3
    _MU_HI = 1e4

    def __init__(self, power=1.5):
        if not 1.0 < power < 2.0:
            raise ValueError(f"Tweedie power must be in (1, 2), got {power}")
        self.power = power

    def _mu_from_eta(self, eta):
        return _clip(_exp(_clip(eta, -self._Z_CLIP, self._Z_CLIP)), self._MU_LO, self._MU_HI)

    # ── Per-sample formulas (single source of truth) ──────────────────

    def per_sample_value(self, eta, y):
        mu = self._mu_from_eta(eta)
        p = self.power
        return -y * mu ** (1.0 - p) / (1.0 - p) + mu ** (2.0 - p) / (2.0 - p)

    def per_sample_gradient(self, eta, y):
        mu = self._mu_from_eta(eta)
        p = self.power
        return mu ** (1.0 - p) * (mu - y)

    def hessian(self, X, y, coef, sample_weight=None):
        z = _clip(X @ coef, -self._Z_CLIP, self._Z_CLIP)
        mu = _clip(_exp(z), self._MU_LO, self._MU_HI)
        p = self.power
        W = mu ** (2.0 - p)
        if sample_weight is not None:
            W = W * sample_weight
        n_eff = float(sample_weight.sum()) if sample_weight is not None else X.shape[0]
        return X.T @ (X * W[:, None]) / n_eff

    def lipschitz(self, X, coef, y=None, sample_weight=None):
        z = _clip(X @ coef, -self._Z_CLIP, self._Z_CLIP)
        mu = _clip(_exp(z), self._MU_LO, self._MU_HI)
        p = self.power
        W = mu ** (2.0 - p)
        if sample_weight is not None:
            W = W * sample_weight
        n_eff = float(sample_weight.sum()) if sample_weight is not None else X.shape[0]
        XtWX = X.T @ (X * W[:, None])
        L = _max_eigval_power(XtWX) / n_eff
        return max(L, 1e-8)

    def predict(self, X, coef):
        return _exp(X @ coef)
