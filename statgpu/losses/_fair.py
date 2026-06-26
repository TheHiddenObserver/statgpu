"""
Fair loss for robust regression.

Loss = (1/n) * sum(rho_c(y_i - eta_i))
where rho_c(u) = c^2 * (|u|/c - log(1 + |u|/c))

Supports numpy / cupy / torch backends via _xp dispatch.

Matches R's MASS::rlm() with psi='fair'.
"""

import numpy as np
from statgpu.backends._array_ops import _xp as _get_xp
from ._base import LossBase
from ._registry import register_loss


@register_loss('fair')
class FairLoss(LossBase):
    """Fair loss for robust regression.

    Smooth psi function with gradual down-weighting.  More smooth
    than Huber but less aggressive than bisquare.

    Parameters
    ----------
    delta : float, optional
        Fixed threshold. If provided, ``epsilon`` and ``method`` are ignored.
    epsilon : float, default=1.35
        Robustness tuning parameter.
    method : str, default="MAD"
        Scale estimation method: ``"MAD"`` or ``"huber_prop2"``.
    """

    name = "fair"
    y_type = "continuous"
    smooth_gradient = True
    has_hessian = True

    _lipschitz_safety = 1.5

    def __init__(self, delta=None, epsilon: float = 1.35, method: str = "MAD"):
        if delta is not None:
            if delta <= 0:
                raise ValueError(f"delta must be positive, got {delta}")
            self.delta = float(delta)
            self.epsilon = None
            self._method = "fixed"
            self._auto_scale = False
            self._scale_estimated = True
        else:
            if epsilon <= 0:
                raise ValueError(f"epsilon must be positive, got {epsilon}")
            self.epsilon = float(epsilon)
            method = method.lower()
            if method not in ("mad", "huber_prop2"):
                raise ValueError(f"method must be 'MAD' or 'huber_prop2', got '{method}'")
            self._method = method
            self.delta = 1.0
            self._auto_scale = True
            self._scale_estimated = False

    # ── Scale estimation ─────────────────────────────────────────────

    def estimate_scale(self, X, y, coef=None):
        """Estimate scale from residuals via MAD: scale = MAD(y - X@coef) / 0.6745."""
        y_np = self._to_numpy(y)
        X_np = self._to_numpy(X)
        if coef is not None:
            coef_np = self._to_numpy(coef)
        else:
            coef_np = np.linalg.lstsq(X_np, y_np, rcond=None)[0]
        r = y_np - X_np @ coef_np
        mad = float(np.median(np.abs(r)))
        scale = max(mad / 0.6745, 1e-10)
        self.delta = self.epsilon * scale
        self._scale_estimated = True
        return scale

    def _update_scale_prop2(self, X, y, coef):
        y_np = self._to_numpy(y)
        X_np = self._to_numpy(X)
        coef_np = self._to_numpy(coef)
        r = y_np - X_np @ coef_np
        n = len(r)
        eps = self.epsilon
        eps2 = eps * eps
        sigma = self.delta / eps if eps > 0 else 1.0
        for _ in range(20):
            abs_r = np.abs(r)
            inliers = abs_r <= eps * sigma
            n_outliers = n - np.sum(inliers)
            sum_sq_in = np.sum(r[inliers] ** 2)
            sigma_new = np.sqrt((sum_sq_in + sigma**2 * eps2 * n_outliers) / n)
            if abs(sigma_new - sigma) < 1e-10 * max(sigma, 1e-10):
                break
            sigma = sigma_new
        self.delta = eps * max(sigma, 1e-10)
        if hasattr(self, '_prev_sigma') and self._prev_sigma > 0:
            if abs(sigma - self._prev_sigma) / self._prev_sigma < 0.01:
                self._scale_estimated = True
        self._prev_sigma = sigma

    def _ensure_scale(self, X, y, coef):
        if not self._auto_scale or self._scale_estimated:
            return
        if self._method == "mad":
            self.estimate_scale(X, y, coef)
        elif self._method == "huber_prop2":
            self._update_scale_prop2(X, y, coef)

    @staticmethod
    def _to_numpy(arr):
        if hasattr(arr, 'cpu'):
            return arr.cpu().numpy()
        elif hasattr(arr, 'get'):
            return arr.get()
        return np.asarray(arr)

    # ── Solver-facing methods ────────────────────────────────────────

    def gradient(self, X, y, coef, sample_weight=None):
        self._ensure_scale(X, y, coef)
        return super().gradient(X, y, coef, sample_weight=sample_weight)

    def fused_value_and_gradient(self, X, y, coef, sample_weight=None):
        self._ensure_scale(X, y, coef)
        return self._fused_impl(X, y, coef, sample_weight=sample_weight)

    def hessian(self, X, y, coef, sample_weight=None):
        self._ensure_scale(X, y, coef)
        return self._hessian_impl(X, y, coef, sample_weight=sample_weight)

    # ── Core computation ─────────────────────────────────────────────

    def _fused_impl(self, X, y, coef, sample_weight=None):
        """Fair loss: rho(u) = c^2 * (|u|/c - log(1 + |u|/c)).

        Gradient: psi(u) = u / (1 + |u|/c)
        """
        xp = _get_xp(X)
        eta = X @ coef
        u = y - eta
        c = self.delta
        abs_u = xp.abs(u)
        t = abs_u / c

        # Value
        ps = c * c * (t - xp.log(1.0 + t))

        # Gradient: d rho(y-eta)/d eta = -rho'(y-eta) = -u / (1 + |u|/c)
        resid = -u / (1.0 + t)

        # Aggregate
        if sample_weight is not None:
            sw_sum = float(xp.dot(sample_weight, ps).item()) if xp.__name__ == "torch" else float(xp.dot(sample_weight, ps))
            val = sw_sum / float(sample_weight.sum())
            grad = X.T @ (sample_weight * resid) / float(sample_weight.sum())
        else:
            n = X.shape[0]
            if xp.__name__ == "torch":
                val = float(xp.sum(ps).item()) / n
            else:
                val = float(xp.sum(ps)) / n
            grad = X.T @ resid / n
        return val, grad

    def _hessian_impl(self, X, y, coef, sample_weight=None):
        """Hessian: X' W X / n, where w_i = 1/(1 + |u_i|/c)^2."""
        xp = _get_xp(X)
        eta = X @ coef
        u = y - eta
        c = self.delta
        abs_u = xp.abs(u)
        t = abs_u / c
        w = 1.0 / ((1.0 + t) * (1.0 + t))
        if sample_weight is not None:
            w = w * sample_weight
            return X.T @ (X * w[:, None]) / float(sample_weight.sum())
        return X.T @ (X * w[:, None]) / X.shape[0]

    # ── Per-sample formulas ──────────────────────────────────────────

    def per_sample_value(self, eta, y):
        u = y - eta
        xp = _get_xp(u)
        c = self.delta
        t = xp.abs(u) / c
        return c * c * (t - xp.log(1.0 + t))

    def per_sample_gradient(self, eta, y):
        u = y - eta
        xp = _get_xp(u)
        c = self.delta
        return -u / (1.0 + xp.abs(u) / c)

    def irls(self, X, y, penalty=None, max_iter=100, tol=1e-6, init_coef=None, eps=1e-12):
        """IRLS for Fair loss."""
        # Ensure scale is estimated before running IRLS
        if not self._scale_estimated:
            if init_coef is not None:
                self.estimate_scale(X, y, init_coef)
            else:
                self.estimate_scale(X, y)  # uses OLS internally

        xp = _get_xp(X)
        X_dev = xp.asarray(X, dtype=xp.float64)
        y_dev = xp.asarray(y, dtype=xp.float64)
        n, p = int(X_dev.shape[0]), int(X_dev.shape[1])
        c = self.delta

        if init_coef is not None:
            beta = xp.asarray(init_coef, dtype=xp.float64).copy()
        else:
            if xp.__name__ == "torch":
                beta = xp.linalg.lstsq(X_dev, y_dev).solution
            else:
                beta = xp.linalg.lstsq(X_dev, y_dev, rcond=None)[0]

        for iteration in range(max_iter):
            eta = X_dev @ beta
            r = y_dev - eta
            abs_r = xp.abs(r)
            t = abs_r / c

            # IRLS weights: w = psi(r)/r = 1/(1+|r|/c)
            w = 1.0 / (1.0 + t)
            w = xp.maximum(w, eps)

            WX = X_dev * w[:, None]
            XtWX = X_dev.T @ WX
            XtWy = X_dev.T @ (w * y_dev)

            ridge = eps * (xp.eye(p, dtype=xp.float64) if xp.__name__ != "torch"
                           else xp.eye(p, dtype=xp.float64, device=X_dev.device))
            A = XtWX + ridge

            if penalty is not None and hasattr(penalty, 'alpha'):
                alpha = float(penalty.alpha)
                I = xp.eye(p, dtype=xp.float64) if xp.__name__ != "torch" else xp.eye(p, dtype=xp.float64, device=X_dev.device)
                if hasattr(penalty, 'l1_ratio'):
                    A = A + n * alpha * (1.0 - float(penalty.l1_ratio)) * I
                else:
                    A = A + n * alpha * I

            beta_new = xp.linalg.solve(A, XtWy)
            diff = beta_new - beta
            delta = float(xp.linalg.norm(diff).item()) if xp.__name__ == "torch" else float(xp.linalg.norm(diff))
            if delta < tol:
                return beta_new, iteration + 1
            beta = beta_new

        return beta, max_iter

    def lipschitz(self, X, coef, y=None, sample_weight=None):
        from statgpu.backends._array_ops import _max_eigval_power
        cache_key = id(X)
        if not hasattr(self, '_lipschitz_cache'):
            self._lipschitz_cache = {}
        if cache_key in self._lipschitz_cache:
            return self._lipschitz_cache[cache_key]
        XtX = X.T @ X
        L = _max_eigval_power(XtX) / X.shape[0]
        self._lipschitz_cache[cache_key] = L
        return L
