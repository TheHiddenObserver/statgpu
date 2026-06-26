"""
Huber loss for robust regression.

Loss = (1/n) * sum(rho_delta(y_i - eta_i))
where rho_delta(u) = 0.5 * u^2           if |u| <= delta
                   = delta * (|u| - 0.5*delta)  otherwise

Supports numpy / cupy / torch backends via _xp dispatch.

Matches R's MASS::rlm() with method='M' and psi='huber'.
"""

import numpy as np
from statgpu.backends._array_ops import _xp as _get_xp
from ._base import LossBase
from ._registry import register_loss


@register_loss('huber')
class HuberLoss(LossBase):
    """Huber loss for robust regression.

    Parameters
    ----------
    delta : float, optional
        Fixed threshold. If provided, ``epsilon`` and ``method`` are ignored.
    epsilon : float, default=1.35
        Robustness tuning parameter (dimensionless).  The effective threshold
        is ``epsilon * scale`` where ``scale`` depends on ``method``.
    method : str, default="MAD"
        Scale estimation method:

        - ``"MAD"``: one-shot ``scale = MAD(y) / 0.6745`` (R MASS::rlm default).
          Scale is estimated once from ``y`` and fixed for the entire optimization.
        - ``"huber_prop2"``: Huber's Proposal 2.  Scale is re-estimated at each
          ``gradient()`` call via fixed-point iteration (R MASS::rlm scale.est="Huber").
    """

    name = "huber"
    y_type = "continuous"
    smooth_gradient = True    # gradient is continuous everywhere
    has_hessian = True        # Piecewise Hessian (discontinuous at |u|=delta, works in practice)

    # Optimization hints
    _lipschitz_safety = 1.5  # Huber gradient is piecewise; extra safety for BB step

    def __init__(self, delta=None, epsilon: float = 1.35, method: str = "MAD"):
        if delta is not None:
            # Fixed-delta mode (backward compatible)
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
            if method not in ("mad", "huber_prop2", "joint"):
                raise ValueError(f"method must be 'MAD', 'huber_prop2', or 'joint', got '{method}'")
            self._method = method
            if method == "joint":
                # Joint: solver optimizes [coef, sigma] as one vector
                self.delta = 1.0
                self._auto_scale = False
                self._scale_estimated = False
                self._joint_sigma_idx = None  # set in _prepare_joint
            else:
                # MAD / prop2: placeholder delta, estimated on first gradient() call
                self.delta = 1.0
                self._auto_scale = True
                self._scale_estimated = False

    # ── Scale estimation ─────────────────────────────────────────────

    def estimate_scale(self, X, y, coef=None):
        """Estimate scale from residuals via MAD: scale = MAD(y - X@coef) / 0.6745.

        Uses OLS residuals if coef is not provided.
        Matches R's MASS::rlm behavior (MAD of residuals, not MAD of y).

        References:
        - Huber (1981), Robust Statistics
        - Maronna, Martin, Yohai (2006), Robust Statistics: Theory and Methods
        """
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
        """Huber Proposal 2: re-estimate sigma via fixed-point iteration.

        sigma^2 = (1/n) * [sum_{inliers} r_i^2 + sigma^2 * eps^2 * n_outliers]

        Freezes (sets _scale_estimated=True) when sigma changes < 1% between calls.
        """
        y_np = self._to_numpy(y)
        if hasattr(X, 'cpu'):
            X_np = X.cpu().numpy()
        elif hasattr(X, 'get'):
            X_np = X.get()
        else:
            X_np = np.asarray(X)
        if hasattr(coef, 'cpu'):
            coef_np = coef.cpu().numpy()
        elif hasattr(coef, 'get'):
            coef_np = coef.get()
        else:
            coef_np = np.asarray(coef)

        r = y_np - X_np @ coef_np
        n = len(r)
        eps = self.epsilon
        eps2 = eps * eps

        # Initialize sigma from current delta
        sigma = self.delta / eps if eps > 0 else 1.0
        old_sigma = sigma

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

        # Freeze when sigma stabilizes (< 1% change between calls)
        if hasattr(self, '_prev_sigma') and self._prev_sigma > 0:
            if abs(sigma - self._prev_sigma) / self._prev_sigma < 0.01:
                self._scale_estimated = True
        self._prev_sigma = sigma

    def _ensure_scale(self, X, y, coef):
        """Lazy scale estimation."""
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

    # ── Joint optimization helpers ───────────────────────────────────

    def n_params(self, p):
        """Return parameter dimension: p for fixed/MAD/prop2, p+1 for joint."""
        return p + 1 if self._method == "joint" else p

    def _unpack_joint(self, coef):
        """Unpack [coef, log_sigma] for joint mode.

        Uses log_sigma to ensure sigma > 0 and improve conditioning.
        """
        p = len(coef) - 1
        return coef[:p], float(np.exp(coef[p]))

    def _joint_sigma_grad_hess(self, X, y, coef_vec, sigma):
        """Compute sigma gradient and Hessian cross/terms for joint mode.

        Loss (per-sample, matching sklearn):
            L = (1/n) * sum rho(r_i) + sigma
            where rho is Huber with delta = epsilon * sigma

        Gradient and Hessian are also per-sample (divided by n).
        """
        eps = self.epsilon
        self.delta = eps * max(sigma, 1e-10)
        n = X.shape[0]
        r = y - X @ coef_vec
        abs_r = np.abs(r)
        d = self.delta
        in_quad = abs_r <= d
        r_in = r[in_quad]
        X_in = X[in_quad]
        n_in = int(np.sum(in_quad))
        n_out = n - n_in
        sq_loss = float(np.sum(r_in ** 2))  # sum of r^2 for inliers

        # Per-sample sigma gradient: 1 - (1/n)*n_out*eps^2 - (1/n)*sq_loss/sigma^2
        grad_s = 1.0 - (n_out * eps**2 + sq_loss / (sigma**2)) / n

        # Per-sample sigma hessian: (2/n) * sq_loss / sigma^3
        hess_s = 2.0 * sq_loss / (n * sigma**3)

        # Cross term: d²L/d_coef d_sigma = -(2/(n*sigma^2)) * X_in^T @ r_in
        cross = -(2.0 / (n * sigma**2)) * (X_in.T @ r_in)

        return grad_s, hess_s, cross

    # ── Solver-facing methods (with lazy scale / joint) ──────────────

    def fused_value_and_gradient(self, X, y, coef, sample_weight=None):
        if self._method == "joint":
            return self._joint_fused(X, y, coef, sample_weight=sample_weight)
        self._ensure_scale(X, y, coef)
        return self._fused_impl(X, y, coef, sample_weight=sample_weight)

    def hessian(self, X, y, coef, sample_weight=None):
        if self._method == "joint":
            return self._joint_hessian(X, y, coef, sample_weight=sample_weight)
        self._ensure_scale(X, y, coef)
        return self._hessian_impl(X, y, coef, sample_weight=sample_weight)

    def value(self, X, y, coef, sample_weight=None):
        if self._method == "joint":
            coef_vec, sigma = self._unpack_joint(coef)
            val, _ = self._joint_fused(X, y, coef, sample_weight=sample_weight)
            return val
        return super().value(X, y, coef, sample_weight=sample_weight)

    def gradient(self, X, y, coef, sample_weight=None):
        if self._method == "joint":
            _, grad = self._joint_fused(X, y, coef, sample_weight=sample_weight)
            return grad
        self._ensure_scale(X, y, coef)
        return super().gradient(X, y, coef, sample_weight=sample_weight)

    def _joint_fused(self, X, y, coef, sample_weight=None):
        """Fused value+gradient for joint [coef, sigma] optimization.

        Matches sklearn loss: n*sigma + squared_loss/sigma + outlier_loss
        (NOT divided by n).
        """
        coef_vec, sigma = self._unpack_joint(coef)
        eps = self.epsilon
        self.delta = eps * max(sigma, 1e-10)
        n, p = X.shape
        eta = X @ coef_vec
        r = y - eta
        abs_r = np.abs(r)
        d = self.delta
        in_quad = abs_r <= d

        r_in = r[in_quad]
        X_in = X[in_quad]
        sq_loss = float(np.sum(r_in ** 2)) / sigma

        r_out = abs_r[~in_quad]
        X_out = X[~in_quad]
        n_out = len(r_out)
        out_loss = 2.0 * eps * float(np.sum(r_out)) - sigma * n_out * eps**2

        val = n * sigma + sq_loss + out_loss

        sign_out = np.sign(r[~in_quad])
        grad_coef = (2.0 / sigma) * (X_in.T @ r_in) - 2.0 * eps * (X_out.T @ sign_out)
        # dL/d(log_sigma) = sigma * dL/dsigma
        grad_sigma = sigma * (n - n_out * eps**2 - sq_loss / sigma)

        full_grad = np.zeros(p + 1)
        full_grad[:p] = grad_coef
        full_grad[p] = grad_sigma
        return val, full_grad

    def _joint_hessian(self, X, y, coef, sample_weight=None):
        """Hessian for joint [coef, sigma] optimization."""
        coef_vec, sigma = self._unpack_joint(coef)
        eps = self.epsilon
        self.delta = eps * max(sigma, 1e-10)
        n, p = X.shape
        eta = X @ coef_vec
        r = y - eta
        abs_r = np.abs(r)
        d = self.delta
        in_quad = abs_r <= d

        r_in = r[in_quad]
        X_in = X[in_quad]
        sq_loss = float(np.sum(r_in ** 2)) / sigma

        # Hessian w.r.t. coef: (2/sigma) * X_in^T @ X_in
        hess_coef = (2.0 / sigma) * (X_in.T @ X_in)

        # Cross term: d²L/d_coef d(log_sigma) = sigma * d²L/d_coef d_sigma
        cross = sigma * (-(2.0 / (sigma ** 2)) * (X_in.T @ r_in))

        # d²L/d(log_sigma)² = sigma * dL/dsigma + sigma² * d²L/dsigma²
        grad_sigma_raw = n - n_out * eps**2 - sq_loss / sigma
        hess_sigma = sigma * grad_sigma_raw + sigma**2 * (2.0 * sq_loss / (sigma ** 3))

        full = np.zeros((p + 1, p + 1))
        full[:p, :p] = hess_coef
        full[:p, p] = cross
        full[p, :p] = cross
        full[p, p] = hess_sigma
        return full

    def _fused_impl(self, X, y, coef, sample_weight=None):
        """Actual fused value+gradient computation (original code)."""
        xp = _get_xp(X)
        eta = X @ coef
        u = y - eta
        d = self.delta
        abs_u = xp.abs(u)

        # Value
        in_quad = (abs_u <= d)
        if xp.__name__ == "torch":
            in_quad_f = in_quad.to(u.dtype)
            min_abs_d = xp.tensor(d, dtype=u.dtype, device=u.device)
        else:
            in_quad_f = in_quad.astype(u.dtype)
            min_abs_d = d
        ps = in_quad_f * 0.5 * u * u + (1.0 - in_quad_f) * d * (abs_u - 0.5 * d)

        # Gradient
        resid = -xp.sign(u) * xp.minimum(abs_u, min_abs_d)

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
        """Actual Hessian computation (original code)."""
        xp = _get_xp(X)
        eta = X @ coef
        u = y - eta
        d = self.delta
        in_quad = (xp.abs(u) <= d)
        if xp.__name__ == "torch":
            w = in_quad.to(u.dtype)
        else:
            w = in_quad.astype(u.dtype)
        if sample_weight is not None:
            w = w * sample_weight
            return X.T @ (X * w[:, None]) / float(sample_weight.sum())
        return X.T @ (X * w[:, None]) / X.shape[0]

    # ── Per-sample formulas (backend-aware, dtype-safe) ──────────────

    def per_sample_value(self, eta, y):
        """Huber loss: rho_delta(y - eta).

        rho_delta(u) = 0.5 * u^2                  if |u| <= delta
                     = delta * (|u| - 0.5 * delta)  otherwise
        """
        u = y - eta
        xp = _get_xp(u)
        d = self.delta
        in_quad = (xp.abs(u) <= d)
        if xp.__name__ == "torch":
            in_quad = in_quad.to(u.dtype)
        else:
            in_quad = in_quad.astype(u.dtype)
        return in_quad * 0.5 * u * u + (1.0 - in_quad) * d * (xp.abs(u) - 0.5 * d)

    def per_sample_gradient(self, eta, y):
        """Gradient w.r.t. eta: -u if |u|<=delta, -delta*sign(u) otherwise.

        Computed as: -sign(u) * min(|u|, delta)
        """
        u = y - eta
        xp = _get_xp(u)
        d = self.delta
        abs_u = xp.abs(u)
        if xp.__name__ == "torch":
            return -xp.sign(u) * xp.minimum(abs_u, xp.tensor(d, dtype=u.dtype, device=u.device))
        return -xp.sign(u) * xp.minimum(abs_u, d)

    def lipschitz(self, X, coef, y=None, sample_weight=None):
        """Lipschitz constant: lambda_max(X'X) / n.

        Cached — X'X and its max eigenvalue are computed once and reused.
        """
        from statgpu.backends._array_ops import _max_eigval_power
        xp = _get_xp(X)
        # Cache key: use X's identity (same object = same lipschitz)
        cache_key = id(X)
        if not hasattr(self, '_lipschitz_cache'):
            self._lipschitz_cache = {}
        if cache_key in self._lipschitz_cache:
            return self._lipschitz_cache[cache_key]
        XtX = X.T @ X
        L = _max_eigval_power(XtX) / X.shape[0]
        self._lipschitz_cache[cache_key] = L
        return L
