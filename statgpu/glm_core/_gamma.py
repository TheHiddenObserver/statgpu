"""
Gamma loss: negative Gamma log-likelihood.

For positive continuous outcomes:
    loss = (1/n) * sum(y/mu + log(mu))
where mu is determined by the configured link:
    - log: mu = exp(X @ coef)
    - inverse_power: mu = 1 / (X @ coef)

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
    _lipschitz_safety = 3.0  # Gamma Hessian varies with mu

    _MU_LO = 1e-3
    _MU_HI = 1e4
    _ETA_LO = 1e-4
    _ETA_HI = 1e3

    def __init__(self, link="log"):
        if link not in ("log", "inverse_power"):
            raise ValueError(
                "GammaLoss link must be 'log' or 'inverse_power', "
                f"got {link!r}."
            )
        self.link = link
        self.link_name = link
        self._lipschitz_at_init = link == "inverse_power"

    def _eta_mu(self, X, coef):
        eta = X @ coef
        if self.link == "inverse_power":
            eta_c = _clip(eta, self._ETA_LO, self._ETA_HI)
            return eta_c, 1.0 / eta_c
        z = _clip(eta, -30, 30)
        return z, _clip(_exp(z), self._MU_LO, self._MU_HI)

    def _mu_from_eta(self, eta):
        if self.link == "inverse_power":
            eta_c = _clip(eta, self._ETA_LO, self._ETA_HI)
            return 1.0 / eta_c
        return _clip(_exp(_clip(eta, -30, 30)), self._MU_LO, self._MU_HI)

    # ── Per-sample formulas (single source of truth) ──────────────────

    def per_sample_value(self, eta, y):
        if self.link == "inverse_power":
            eta_c = _clip(eta, self._ETA_LO, self._ETA_HI)
            return y * eta_c - _log(eta_c)
        mu = self._mu_from_eta(eta)
        return y / mu + _log(mu)

    def per_sample_gradient(self, eta, y):
        if self.link == "inverse_power":
            mu = self._mu_from_eta(eta)
            return y - mu
        mu = self._mu_from_eta(eta)
        return 1.0 - y / mu

    def hessian(self, X, y, coef):
        if self.link == "inverse_power":
            eta, _ = self._eta_mu(X, coef)
            W = 1.0 / (eta * eta)
            return X.T @ (X * W[:, None]) / X.shape[0]
        # Expected Fisher: W(mu) = 1 for Gamma with log link.
        return X.T @ X / X.shape[0]

    def lipschitz(self, X, coef, y=None):
        if self.link == "inverse_power":
            eta, _ = self._eta_mu(X, coef)
            W = 1.0 / (eta * eta)
            XtWX = X.T @ (X * W[:, None])
            L = _max_eigval_power(XtWX) / X.shape[0]
            return max(L, 1e-8)
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
        if self.link == "inverse_power":
            eta = _clip(X @ coef, self._ETA_LO, self._ETA_HI)
            return 1.0 / eta
        return _exp(X @ coef)
