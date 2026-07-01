"""
Bisquare (Tukey biweight) loss for robust regression.

Loss = (1/n) * sum(rho_c(y_i - eta_i))
where rho_c(u) = (c^2/6) * [1 - (1 - (u/c)^2)^3]  if |u| <= c
                 = c^2/6                              otherwise

Supports numpy / cupy / torch backends via _xp dispatch.

Matches R's MASS::rlm() with psi='bisquare' (k=4.685).
"""

import numpy as np
from statgpu.backends._array_ops import _xp as _get_xp
from ._robust_base import RobustLossBase
from ._registry import register_loss


@register_loss('bisquare')
class BisquareLoss(RobustLossBase):
    """Bisquare (Tukey biweight) loss for robust regression.

    Completely ignores residuals beyond threshold (gradient = 0),
    giving higher breakdown point than Huber.

    Parameters
    ----------
    delta : float, optional
        Fixed threshold. If provided, ``epsilon`` and ``method`` are ignored.
    epsilon : float, default=4.685
        Robustness tuning parameter.  Default 4.685 gives 95% efficiency
        at the normal distribution (R MASS default).
    method : str, default="MAD"
        Scale estimation method: ``"MAD"`` or ``"huber_prop2"``.
    """

    name = "bisquare"
    y_type = "continuous"
    smooth_gradient = True    # gradient is continuous (0 at boundary)
    has_hessian = True        # Hessian weight is continuous (0 at boundary)
    _supports_irls = True     # has irls() method

    _lipschitz_safety = 2.0  # bisquare gradient is more aggressive; extra safety

    def __init__(self, delta=None, epsilon: float = 4.685, method: str = "MAD"):
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

    # ── Core computation ─────────────────────────────────────────────

    def _fused_impl(self, X, y, coef, sample_weight=None):
        xp = _get_xp(X)
        eta = X @ coef
        u = y - eta
        c = self.delta
        t = u / c  # normalized residual

        # Value: rho(t) = (c^2/6) * [1 - (1-t^2)^3] for |t|<=1, c^2/6 otherwise
        abs_t = xp.abs(t)
        in_range = abs_t <= 1.0
        if xp.__name__ == "torch":
            in_f = in_range.to(u.dtype)
        else:
            in_f = in_range.astype(u.dtype)
        one_minus_t2 = xp.clip(1.0 - t * t, 0.0, None)
        ps = in_f * (c * c / 6.0) * (1.0 - one_minus_t2 ** 3) + (1.0 - in_f) * (c * c / 6.0)

        # Gradient: d rho(y-eta)/d eta = -rho'(y-eta) = -c*t*(1-t^2)^2
        resid = -c * in_f * t * one_minus_t2 * one_minus_t2

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
        """Hessian: X' W X / n, where w_i = max(0, (1-5t^2)(1-t^2)) for |t|<=1, 0 otherwise.

        Weight is clamped to non-negative to ensure positive semi-definite Hessian
        for Newton-type solvers. Without clamping, w < 0 when |t| > 1/sqrt(5),
        making the Hessian indefinite.
        """
        xp = _get_xp(X)
        eta = X @ coef
        u = y - eta
        c = self.delta
        t = u / c
        abs_t = xp.abs(t)
        in_range = abs_t <= 1.0
        if xp.__name__ == "torch":
            in_f = in_range.to(u.dtype)
        else:
            in_f = in_range.astype(u.dtype)
        t2 = t * t
        one_minus_t2 = xp.clip(1.0 - t2, 0.0, None)
        # Hessian weight: max(0, (1 - t^2)(1 - 5t^2)) for |t|<=1
        w = in_f * xp.clip(one_minus_t2 * (1.0 - 5.0 * t2), 0.0, None)
        if sample_weight is not None:
            w = w * sample_weight
            return X.T @ (X * w[:, None]) / float(sample_weight.sum())
        return X.T @ (X * w[:, None]) / X.shape[0]

    def _fused_grad_hess_impl(self, X, y, coef, sample_weight=None):
        """Fused gradient + Hessian: one X@coef, shared residuals."""
        xp = _get_xp(X)
        eta = X @ coef
        u = y - eta
        c = self.delta
        t = u / c
        abs_t = xp.abs(t)
        in_range = abs_t <= 1.0
        if xp.__name__ == "torch":
            in_f = in_range.to(u.dtype)
        else:
            in_f = in_range.astype(u.dtype)
        t2 = t * t
        one_minus_t2 = xp.clip(1.0 - t2, 0.0, None)

        # Gradient: resid = -c * in_f * t * (1-t^2)^2
        resid = -c * in_f * t * one_minus_t2 * one_minus_t2

        # Hessian weight: max(0, (1-t^2)(1-5t^2)) for |t|<=1
        w = in_f * xp.clip(one_minus_t2 * (1.0 - 5.0 * t2), 0.0, None)

        if sample_weight is not None:
            sw_sum = float(sample_weight.sum())
            grad = X.T @ (sample_weight * resid) / sw_sum
            w = w * sample_weight
            hess = X.T @ (X * w[:, None]) / sw_sum
        else:
            n = X.shape[0]
            grad = X.T @ resid / n
            hess = X.T @ (X * w[:, None]) / n
        return grad, hess

    # ── Per-sample formulas ──────────────────────────────────────────

    def per_sample_value(self, eta, y):
        u = y - eta
        xp = _get_xp(u)
        c = self.delta
        t = u / c
        abs_t = xp.abs(t)
        in_range = abs_t <= 1.0
        if xp.__name__ == "torch":
            in_f = in_range.to(u.dtype)
        else:
            in_f = in_range.astype(u.dtype)
        one_minus_t2 = xp.clip(1.0 - t * t, 0.0, None)
        return in_f * (c * c / 6.0) * (1.0 - one_minus_t2 ** 3) + (1.0 - in_f) * (c * c / 6.0)

    def per_sample_gradient(self, eta, y):
        u = y - eta
        xp = _get_xp(u)
        c = self.delta
        t = u / c
        abs_t = xp.abs(t)
        in_range = abs_t <= 1.0
        if xp.__name__ == "torch":
            in_f = in_range.to(u.dtype)
        else:
            in_f = in_range.astype(u.dtype)
        one_minus_t2 = xp.clip(1.0 - t * t, 0.0, None)
        return -c * in_f * t * one_minus_t2 * one_minus_t2

    def irls(self, X, y, penalty=None, max_iter=100, tol=1e-6, init_coef=None, eps=1e-12):
        """IRLS for bisquare loss. Bisquare needs IRLS (not Newton) because
        Hessian weight = 0 for large residuals."""
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

            # IRLS weights: w = (1 - t^2)^2 for |t|<=1, 0 otherwise
            in_range = t <= 1.0
            if xp.__name__ == "torch":
                in_f = in_range.to(r.dtype)
            else:
                in_f = in_range.astype(r.dtype)
            one_minus_t2 = xp.clip(1.0 - t * t, 0.0, None)
            w = in_f * one_minus_t2 * one_minus_t2
            if xp.__name__ == "torch":
                w = xp.maximum(w, xp.tensor(eps, dtype=w.dtype, device=w.device))
            else:
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
