"""
Poisson loss: negative Poisson log-likelihood.

For count data:
    loss = (1/n) * sum(mu - y*log(mu))
where mu = exp(X @ coef).

Supports numpy / cupy / torch backends via _backend helpers.
"""
from statgpu.backends._array_ops import _clip, _exp, _log, _sum, _max_eigval_power
from ._base import GLMLoss, register_glm_loss


@register_glm_loss('poisson')
class PoissonLoss(GLMLoss):
    name = "poisson"
    y_type = "count"
    smooth_gradient = True
    has_hessian = True

    _MU_LO = 1e-10
    _MU_HI = 1e6  # must exceed typical max(y); clip prevents extreme weights
    _ETA_LO = -30
    _ETA_HI = 30

    # ── Per-sample formulas (single source of truth) ──────────────────

    def _mu_from_eta(self, eta):
        return _clip(_exp(_clip(eta, self._ETA_LO, self._ETA_HI)), self._MU_LO, self._MU_HI)

    def per_sample_value(self, eta, y):
        mu = self._mu_from_eta(eta)
        return mu - y * _log(mu)

    def per_sample_gradient(self, eta, y):
        mu = self._mu_from_eta(eta)
        return mu - y

    # ── Hessian / Lipschitz ───────────────────────────────────────────

    def hessian(self, X, y, coef):
        z = _clip(X @ coef, -30, 30)
        mu = _clip(_exp(z), self._MU_LO, self._MU_HI)
        return X.T @ (X * mu[:, None]) / X.shape[0]

    def lipschitz(self, X, coef, y=None):
        z = _clip(X @ coef, -30, 30)
        mu = _clip(_exp(z), self._MU_LO, self._MU_HI)
        XtWX = X.T @ (X * mu[:, None])
        L = _max_eigval_power(XtWX) / X.shape[0]
        return max(L, 1e-8)

    def predict(self, X, coef):
        return _exp(X @ coef)
