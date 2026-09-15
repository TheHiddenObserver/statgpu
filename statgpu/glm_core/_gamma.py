"""
Gamma loss: negative Gamma log-likelihood.

For positive continuous outcomes:
    loss = (1/n) * sum(y/mu + log(mu))
where mu is determined by the configured link:
    - log: mu = exp(X @ coef)
    - inverse_power: mu = 1 / (X @ coef)

Supports numpy / cupy / torch backends via _array_ops helpers.
"""
import numpy as np

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

    # Frozen before the next physical acceptance run.  These are numerical
    # certification controls for the explicit Newton/L-BFGS path, not new
    # statistical restrictions on the Gamma inverse link itself.
    _DOMAIN_MARGIN_ULPS = 32.0
    _DOMAIN_SEPARATOR_ULPS = 64.0
    _DOMAIN_STEP_SAFETY = 0.99
    _DOMAIN_FW_MAX_ITER = 512

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

    # ── Private explicit-smooth optimization domain ──────────────────

    @staticmethod
    def _scalar(value):
        return float(value.item() if hasattr(value, "item") else value)

    @staticmethod
    def _truth(value):
        return bool(value.item() if hasattr(value, "item") else value)

    def _domain_eps(self, X):
        xp = _xp(X)
        if xp.__name__ == "torch":
            import torch
            dtype = X.dtype if torch.is_floating_point(X) else torch.float64
            return float(torch.finfo(dtype).eps)
        try:
            return float(np.finfo(X.dtype).eps)
        except (TypeError, ValueError):
            return float(np.finfo(np.float64).eps)

    def _loss_domain_bounds(self, X):
        eps = self._domain_eps(X)
        lo_margin = self._DOMAIN_MARGIN_ULPS * eps * max(1.0, abs(self._ETA_LO))
        hi_margin = self._DOMAIN_MARGIN_ULPS * eps * max(1.0, abs(self._ETA_HI))
        lo = float(self._ETA_LO + lo_margin)
        hi = float(self._ETA_HI - hi_margin)
        if not lo < hi:
            raise RuntimeError("Gamma inverse_power smooth-domain bounds collapsed.")
        return lo, hi

    def _domain_active_data(self, X, y=None, sample_weight=None):
        from statgpu.backends._utils import xp_asarray

        xp = _xp(X)
        if sample_weight is None:
            X_active = X
            y_active = None if y is None else xp_asarray(
                y, dtype=X.dtype, xp=xp, ref_arr=X
            ).reshape(-1)
            w_active = None
        else:
            w = xp_asarray(
                sample_weight, dtype=X.dtype, xp=xp, ref_arr=X
            ).reshape(-1)
            mask = w > 0
            if not self._truth(xp.any(mask)):
                raise ValueError("Gamma inverse_power requires at least one active row.")
            X_active = X[mask]
            y_active = None if y is None else xp_asarray(
                y, dtype=X.dtype, xp=xp, ref_arr=X
            ).reshape(-1)[mask]
            w_active = w[mask]

        if not self._truth(xp.all(xp.isfinite(X_active))):
            raise ValueError(
                "Gamma inverse_power active design rows must be finite."
            )
        return X_active, y_active, w_active

    def _scale_safe_row_norms(self, X):
        xp = _xp(X)
        if xp.__name__ == "torch":
            import torch
            scale = torch.amax(torch.abs(X), dim=1)
            if self._truth(torch.any(scale == 0)):
                raise ValueError(
                    "Gamma inverse_power active design contains an exact zero row; "
                    "no strictly positive linear predictor exists for that row."
                )
            unit = X / scale[:, None]
            return scale * torch.sqrt(torch.sum(unit * unit, dim=1))

        scale = xp.max(xp.abs(X), axis=1)
        if self._truth(xp.any(scale == 0)):
            raise ValueError(
                "Gamma inverse_power active design contains an exact zero row; "
                "no strictly positive linear predictor exists for that row."
            )
        unit = X / scale[:, None]
        return scale * xp.sqrt(xp.sum(unit * unit, axis=1))

    def _zeros_direction(self, X):
        xp = _xp(X)
        if xp.__name__ == "torch":
            import torch
            return torch.zeros(X.shape[1], dtype=X.dtype, device=X.device)
        return xp.zeros(X.shape[1], dtype=X.dtype)

    def _scale_domain_direction(self, X, y, weights, direction):
        """Scale a positive separator into the maintained smooth band."""
        xp = _xp(X)
        a = X @ direction
        if not self._truth(xp.all(xp.isfinite(a))):
            return None
        a_min = self._scalar(xp.min(a))
        a_max = self._scalar(xp.max(a))
        if not (a_min > 0.0 and np.isfinite(a_max) and a_max > 0.0):
            return None

        lo, hi = self._loss_domain_bounds(X)
        c_lo = lo / a_min
        c_hi = hi / a_max
        eps = self._domain_eps(X)
        frac = max(self._DOMAIN_SEPARATOR_ULPS * eps, 1e-12)
        c_lo_safe = c_lo * (1.0 + frac)
        c_hi_safe = c_hi * (1.0 - frac)
        if not (
            np.isfinite(c_lo_safe)
            and np.isfinite(c_hi_safe)
            and c_lo_safe < c_hi_safe
        ):
            return None

        if weights is None:
            normalization = float(X.shape[0])
            denom = self._scalar(xp.sum(y * a))
        else:
            normalization = self._scalar(xp.sum(weights))
            denom = self._scalar(xp.sum(weights * y * a))
        if not (np.isfinite(denom) and denom > 0.0):
            return None

        c_ray = normalization / denom
        c = min(max(c_ray, c_lo_safe), c_hi_safe)
        beta = direction * c
        if self._loss_domain_is_feasible(X, beta, sample_weight=weights):
            return beta
        return None

    def _loss_domain_initial_point(self, X, y, sample_weight=None):
        if self.link != "inverse_power":
            return None

        X_active, y_active, w_active = self._domain_active_data(
            X, y, sample_weight
        )
        norms = self._scale_safe_row_norms(X_active)
        xp = _xp(X_active)
        U = X_active / norms[:, None]

        # Fast candidate: a column with one strict sign over all active rows.
        if xp.__name__ == "torch":
            import torch
            positive = torch.all(X_active > 0, dim=0)
            negative = torch.all(X_active < 0, dim=0)
            pos_idx = torch.nonzero(positive, as_tuple=False).reshape(-1)
            neg_idx = torch.nonzero(negative, as_tuple=False).reshape(-1)
        else:
            positive = xp.all(X_active > 0, axis=0)
            negative = xp.all(X_active < 0, axis=0)
            pos_idx = xp.nonzero(positive)[0]
            neg_idx = xp.nonzero(negative)[0]

        for indices, sign in ((pos_idx, 1.0), (neg_idx, -1.0)):
            if int(indices.shape[0]) > 0:
                j = int(indices[0].item() if hasattr(indices[0], "item") else indices[0])
                d = self._zeros_direction(X_active)
                d[j] = sign
                beta = self._scale_domain_direction(
                    X_active, y_active, w_active, d
                )
                if beta is not None:
                    return beta

        # Deterministic Gilbert / Frank-Wolfe minimum-norm convex-hull search.
        if xp.__name__ == "torch":
            import torch
            c = torch.mean(U, dim=0)
        else:
            c = xp.mean(U, axis=0)
        margin = self._DOMAIN_SEPARATOR_ULPS * self._domain_eps(X_active)
        denom_floor = max(margin * margin, 1e-30)

        for _ in range(self._DOMAIN_FW_MAX_ITER):
            dots = U @ c
            m = self._scalar(xp.min(dots))
            if m > margin:
                beta = self._scale_domain_direction(
                    X_active, y_active, w_active, c
                )
                if beta is not None:
                    return beta

            j_dev = xp.argmin(dots)
            j = int(j_dev.item() if hasattr(j_dev, "item") else j_dev)
            s = U[j]
            diff = c - s
            c2 = self._scalar(xp.sum(c * c))
            denom = self._scalar(xp.sum(diff * diff))
            gap = c2 - m
            if denom <= denom_floor or gap <= margin * max(1.0, c2):
                break
            gamma = min(max(gap / denom, 0.0), 1.0)
            if gamma <= margin:
                break
            c = (1.0 - gamma) * c + gamma * s

        from statgpu.solvers._smooth_domain import _LossDomainError
        raise _LossDomainError(
            "Gamma inverse_power has no numerically certified smooth-domain "
            "start for the active design; the geometry may be infeasible or "
            "too ill-conditioned for the maintained certification procedure."
        )

    def _loss_domain_is_feasible(self, X, coef, sample_weight=None):
        if self.link != "inverse_power":
            return True
        X_active, _, _ = self._domain_active_data(
            X, None, sample_weight
        )
        xp = _xp(X_active)
        eta = X_active @ coef
        lo, hi = self._loss_domain_bounds(X_active)
        ok = xp.all(xp.isfinite(eta)) & xp.all(eta > lo) & xp.all(eta < hi)
        return self._truth(ok)

    def _loss_domain_max_step(self, X, coef, delta, sample_weight=None):
        if self.link != "inverse_power":
            return None
        X_active, _, _ = self._domain_active_data(
            X, None, sample_weight
        )
        xp = _xp(X_active)
        eta = X_active @ coef
        move = X_active @ delta
        lo, hi = self._loss_domain_bounds(X_active)

        if not self._loss_domain_is_feasible(
            X, coef, sample_weight=sample_weight
        ):
            raise ValueError(
                "Gamma inverse_power current iterate is outside the maintained "
                "smooth training domain."
            )

        # Divide only on rows that actually move toward a boundary.  ``where``
        # would eagerly evaluate division on move==0 rows for NumPy/CuPy,
        # producing spurious divide-by-zero warnings even though those rows do
        # not constrain the admissible step.
        if xp.__name__ == "torch":
            import torch

            lower = torch.full_like(eta, float("inf"))
            upper = torch.full_like(eta, float("inf"))
            lower_mask = move < 0
            upper_mask = move > 0
            lower[lower_mask] = (eta[lower_mask] - lo) / (-move[lower_mask])
            upper[upper_mask] = (hi - eta[upper_mask]) / move[upper_mask]
            boundary = torch.minimum(torch.min(lower), torch.min(upper))
            boundary_value = self._scalar(boundary)
        else:
            lower = xp.full_like(eta, float("inf"))
            upper = xp.full_like(eta, float("inf"))
            lower_mask = move < 0
            upper_mask = move > 0
            lower[lower_mask] = (eta[lower_mask] - lo) / (-move[lower_mask])
            upper[upper_mask] = (hi - eta[upper_mask]) / move[upper_mask]
            boundary_value = self._scalar(xp.minimum(xp.min(lower), xp.min(upper)))

        if not np.isfinite(boundary_value):
            return None
        return boundary_value * self._DOMAIN_STEP_SAFETY

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
            # Use the same clipped predictor as the maintained Hessian.  This
            # also keeps zero-weight rows finite before their weight is applied.
            eta = _clip(X @ coef, self._ETA_LO, self._ETA_HI)
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