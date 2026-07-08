"""
Gamma loss: negative Gamma log-likelihood.

For positive continuous outcomes:
    loss = (1/n) * sum(y/mu + log(mu))
where mu is determined by the configured link:
    - log: mu = exp(X @ coef)
    - inverse_power: mu = 1 / (X @ coef)

Supports numpy / cupy / torch backends via _array_ops helpers.
"""
from statgpu.backends._array_ops import _clip, _exp, _log, _max_eigval_power, _xp
from statgpu.glm_core._base import GLMLoss, register_glm_loss


@register_glm_loss('gamma')
class GammaLoss(GLMLoss):
    name = "gamma"
    y_type = "positive"
    smooth_gradient = True
    has_hessian = True
    _lipschitz_uses_y = True
    _lipschitz_safety = 3.0  # Gamma Hessian varies with mu
    _conservative_momentum_with_nonsmooth = True
    _gamma_like = True

    _LOG_ETA_LO = -30.0
    _LOG_ETA_HI = 30.0
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
        # The observed Gamma-log Hessian depends on y / mu.
        self._has_constant_hessian = False

    def preprocess(self, X, y):
        xp = _xp(y)
        invalid = xp.any(~xp.isfinite(y)) | xp.any(y <= 0)
        if bool(invalid.item() if hasattr(invalid, "item") else invalid):
            raise ValueError("Gamma loss requires finite, strictly positive y values.")
        return X, y

    def _eta_mu(self, X, coef):
        eta = X @ coef
        if self.link == "inverse_power":
            eta_c = _clip(eta, self._ETA_LO, self._ETA_HI)
            return eta_c, 1.0 / eta_c
        eta_c = _clip(eta, self._LOG_ETA_LO, self._LOG_ETA_HI)
        return eta_c, _exp(eta_c)

    def _mu_from_eta(self, eta):
        if self.link == "inverse_power":
            eta_c = _clip(eta, self._ETA_LO, self._ETA_HI)
            return 1.0 / eta_c
        eta_c = _clip(eta, self._LOG_ETA_LO, self._LOG_ETA_HI)
        return _exp(eta_c)

    # ── Per-sample formulas (single source of truth) ──────────────────

    def per_sample_value(self, eta, y):
        if self.link == "inverse_power":
            eta_c = _clip(eta, self._ETA_LO, self._ETA_HI)
            return y * eta_c - _log(eta_c)
        eta_c = _clip(eta, self._LOG_ETA_LO, self._LOG_ETA_HI)
        return eta_c + y * _exp(-eta_c)

    def per_sample_gradient(self, eta, y):
        if self.link == "inverse_power":
            mu = self._mu_from_eta(eta)
            return y - mu
        eta_c = _clip(eta, self._LOG_ETA_LO, self._LOG_ETA_HI)
        return 1.0 - y * _exp(-eta_c)

    def hessian(self, X, y, coef, sample_weight=None):
        n_eff = float(sample_weight.sum()) if sample_weight is not None else X.shape[0]
        eta, mu = self._eta_mu(X, coef)
        if self.link == "inverse_power":
            W = 1.0 / (eta * eta)
        else:
            # Exact observed Hessian.  Since y / mu is positive, X'WX is
            # positive semidefinite (positive definite for full-rank X).
            W = y / mu
        if sample_weight is not None:
            W = W * sample_weight
        return X.T @ (X * W[:, None]) / n_eff

    def fisher_information(self, X, coef, sample_weight=None):
        n_eff = float(sample_weight.sum()) if sample_weight is not None else X.shape[0]
        _, mu = self._eta_mu(X, coef)
        if self.link == "inverse_power":
            # Canonical link: Fisher = observed Hessian.  W = 1/eta² = mu².
            # Since eta = 1/mu, 1/eta² = mu².  We compute as 1/eta² for stability.
            eta = X @ coef
            W = 1.0 / (eta * eta)
        else:
            # Log link: Fisher weight = 1/(V(mu)*g'(mu)²) = 1/(mu² * 1/mu²) = 1
            from statgpu.backends._utils import xp_ones
            from statgpu.backends._array_ops import _xp as _get_xp
            xp = _get_xp(mu)
            W = xp_ones(X.shape[0], mu.dtype, xp, ref_arr=mu)
        if sample_weight is not None:
            W = W * sample_weight
        return X.T @ (X * W[:, None]) / n_eff

    def lipschitz(self, X, coef, y=None, sample_weight=None):
        n_eff = float(sample_weight.sum()) if sample_weight is not None else X.shape[0]
        if self.link == "inverse_power":
            eta, _ = self._eta_mu(X, coef)
            W = 1.0 / (eta * eta)
        elif y is not None:
            z = _clip(X @ coef, self._LOG_ETA_LO, self._LOG_ETA_HI)
            mu = _exp(z)
            W = y / mu
        else:
            XtX = X.T @ X
            return max(_max_eigval_power(XtX) / n_eff, 1e-8)
        if sample_weight is not None:
            W = W * sample_weight
        XtWX = X.T @ (X * W[:, None])
        L = _max_eigval_power(XtWX) / n_eff
        return max(L, 1e-8)

    def predict(self, X, coef):
        if self.link == "inverse_power":
            eta = _clip(X @ coef, self._ETA_LO, self._ETA_HI)
            return 1.0 / eta
        return _exp(X @ coef)
