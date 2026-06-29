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
from ._robust_base import RobustLossBase
from ._registry import register_loss


@register_loss('huber')
class HuberLoss(RobustLossBase):
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
        - ``"joint"``: Joint optimization of [coef, log_sigma].  The solver
          optimizes coefficients and scale simultaneously.
    """

    name = "huber"
    y_type = "continuous"
    smooth_gradient = True    # gradient is continuous everywhere
    has_hessian = True        # Piecewise Hessian (discontinuous at |u|=delta, works in practice)
    # Note: Huber has irls() method but Newton is preferred — IRLS converges to
    # a fixed point of the weighted LS problem, not the true Huber loss minimum.
    # Newton uses the true Hessian and converges to the global minimum.
    _supports_irls = False

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

    # ── Joint optimization helpers ───────────────────────────────────

    def n_params(self, p):
        """Return parameter dimension: p for fixed/MAD/prop2, p+1 for joint."""
        return p + 1 if self._method == "joint" else p

    def _unpack_joint(self, coef):
        """Unpack [coef, log_sigma] for joint mode.

        Uses log_sigma to ensure sigma > 0 and improve conditioning.
        Backend-aware: works with numpy/cupy/torch.
        """
        from statgpu.backends._array_ops import _xp as _get_xp
        xp = _get_xp(coef)
        p = len(coef) - 1
        sigma_val = coef[p]
        if hasattr(sigma_val, 'item'):
            sigma = float(xp.exp(sigma_val).item())
        else:
            sigma = float(xp.exp(sigma_val))
        return coef[:p], sigma

    def _joint_sigma_grad_hess(self, X, y, coef_vec, sigma):
        """Compute sigma gradient and Hessian cross/terms for joint mode.

        Loss (per-sample, matching sklearn):
            L = (1/n) * sum rho(r_i) + sigma
            where rho is Huber with delta = epsilon * sigma

        Gradient and Hessian are also per-sample (divided by n).
        """
        xp = _get_xp(X)
        eps = self.epsilon
        self.delta = eps * max(sigma, 1e-10)
        n = X.shape[0]
        r = y - X @ coef_vec
        abs_r = xp.abs(r)
        d = self.delta
        in_quad = abs_r <= d
        r_in = r[in_quad]
        X_in = X[in_quad]
        n_in = int(xp.sum(in_quad.astype(xp.int64)))
        n_out = n - n_in
        sq_loss = float(xp.sum(r_in ** 2))  # sum of r^2 for inliers

        # Per-sample sigma gradient: 1 - (1/n)*n_out*eps^2 - (1/n)*sq_loss/sigma^2
        grad_s = 1.0 - (n_out * eps**2 + sq_loss / (sigma**2)) / n

        # Per-sample sigma hessian: (2/n) * sq_loss / sigma^3
        hess_s = 2.0 * sq_loss / (n * sigma**3)

        # Cross term: d²L/d_coef d_sigma = +(2/(n*sigma^2)) * X_in^T @ r_in
        cross = (2.0 / (n * sigma**2)) * (X_in.T @ r_in)

        return grad_s, hess_s, cross

    # ── Solver-facing methods (with joint mode dispatch) ──────────────

    def value(self, X, y, coef, sample_weight=None):
        if self._method == "joint":
            val, _ = self._joint_fused(X, y, coef, sample_weight=sample_weight)
            return val
        return super().value(X, y, coef, sample_weight=sample_weight)

    def gradient(self, X, y, coef, sample_weight=None):
        if self._method == "joint":
            _, grad = self._joint_fused(X, y, coef, sample_weight=sample_weight)
            return grad
        return super().gradient(X, y, coef, sample_weight=sample_weight)

    def fused_value_and_gradient(self, X, y, coef, sample_weight=None):
        if self._method == "joint":
            return self._joint_fused(X, y, coef, sample_weight=sample_weight)
        return super().fused_value_and_gradient(X, y, coef, sample_weight=sample_weight)

    def hessian(self, X, y, coef, sample_weight=None):
        if self._method == "joint":
            return self._joint_hessian(X, y, coef, sample_weight=sample_weight)
        return super().hessian(X, y, coef, sample_weight=sample_weight)

    # ── Joint mode implementations ───────────────────────────────────

    def _joint_fused(self, X, y, coef, sample_weight=None):
        """Fused value+gradient for joint [coef, sigma] optimization.

        Matches sklearn loss: n*sigma + squared_loss/sigma + outlier_loss
        (NOT divided by n).
        """
        xp = _get_xp(X)
        coef_vec, sigma = self._unpack_joint(coef)
        eps = self.epsilon
        self.delta = eps * max(sigma, 1e-10)
        n, p = X.shape
        eta = X @ coef_vec
        r = y - eta
        abs_r = xp.abs(r)
        d = self.delta
        in_quad = abs_r <= d

        r_in = r[in_quad]
        X_in = X[in_quad]
        sq_loss = float(xp.sum(r_in ** 2)) / sigma

        r_out = abs_r[~in_quad]
        X_out = X[~in_quad]
        n_out = len(r_out)
        out_loss = 2.0 * eps * float(xp.sum(r_out)) - sigma * n_out * eps**2

        val = n * sigma + sq_loss + out_loss

        sign_out = xp.sign(r[~in_quad])
        grad_coef = -(2.0 / sigma) * (X_in.T @ r_in) - 2.0 * eps * (X_out.T @ sign_out)
        # dL/d(log_sigma) = sigma * dL/dsigma
        grad_sigma = sigma * (n - n_out * eps**2 - sq_loss / sigma)

        full_grad = xp.zeros(p + 1, dtype=X.dtype)
        full_grad[:p] = grad_coef
        full_grad[p] = grad_sigma
        return val, full_grad

    def _joint_hessian(self, X, y, coef, sample_weight=None):
        """Hessian for joint [coef, sigma] optimization."""
        xp = _get_xp(X)
        coef_vec, sigma = self._unpack_joint(coef)
        eps = self.epsilon
        self.delta = eps * max(sigma, 1e-10)
        n, p = X.shape
        eta = X @ coef_vec
        r = y - eta
        abs_r = xp.abs(r)
        d = self.delta
        in_quad = abs_r <= d

        r_in = r[in_quad]
        X_in = X[in_quad]
        sq_loss = float(xp.sum(r_in ** 2)) / sigma
        n_out = n - int(xp.sum(in_quad.astype(xp.int64)))

        # Hessian w.r.t. coef: (2/sigma) * X_in^T @ X_in
        hess_coef = (2.0 / sigma) * (X_in.T @ X_in)

        # Cross term: d²L/d_coef d(log_sigma) = sigma * d/d_sigma[dL/d_coef]
        # dL/d_coef = -(2/sigma)*X_in^T@r_in, so d/d_sigma = +(2/sigma^2)*X_in^T@r_in
        # d²L/d_coef d(log_sigma) = sigma * (2/sigma^2)*X_in^T@r_in = (2/sigma)*X_in^T@r_in
        cross = (2.0 / sigma) * (X_in.T @ r_in)

        # d²L/d(log_sigma)² = sigma * dL/dsigma + sigma² * d²L/dsigma²
        grad_sigma_raw = n - n_out * eps**2 - sq_loss / sigma
        # d²L/d(sigma²) = 2*sum_in(r^2)/sigma^3 = 2*sq_loss/sigma^2
        # sigma² * d²L/d(sigma²) = 2*sq_loss
        hess_sigma = sigma * grad_sigma_raw + 2.0 * sq_loss

        full = xp.zeros((p + 1, p + 1), dtype=X.dtype)
        full[:p, :p] = hess_coef
        full[:p, p] = cross
        full[p, :p] = cross
        full[p, p] = hess_sigma
        return full

    # ── Core computation ─────────────────────────────────────────────

    def _fused_impl(self, X, y, coef, sample_weight=None):
        """Actual fused value+gradient computation."""
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
        """Actual Hessian computation."""
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

    def _fused_grad_hess_impl(self, X, y, coef, sample_weight=None):
        """Fused gradient + Hessian: one X@coef, shared residuals.

        Computes eta = X @ coef once, then derives both gradient and Hessian.
        Saves one X @ coef vs separate gradient() + hessian() calls.
        """
        # Joint mode: delegate to joint-specific methods
        if self._method == "joint":
            _, grad = self._joint_fused(X, y, coef, sample_weight=sample_weight)
            hess = self._joint_hessian(X, y, coef, sample_weight=sample_weight)
            return grad, hess

        xp = _get_xp(X)
        eta = X @ coef
        u = y - eta
        d = self.delta
        abs_u = xp.abs(u)

        # Gradient: resid = -sign(u) * min(|u|, delta)
        if xp.__name__ == "torch":
            min_abs_d = xp.tensor(d, dtype=u.dtype, device=u.device)
        else:
            min_abs_d = d
        resid = -xp.sign(u) * xp.minimum(abs_u, min_abs_d)

        # Hessian weights: w = 1 if |u| <= delta, 0 otherwise
        in_quad = (abs_u <= d)
        if xp.__name__ == "torch":
            w = in_quad.to(u.dtype)
        else:
            w = in_quad.astype(u.dtype)

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

    def irls(self, X, y, penalty=None, max_iter=100, tol=1e-6, init_coef=None, eps=1e-12):
        """Not supported for Huber loss.

        IRLS converges to a fixed point of the weighted least squares problem
        (smooth weights w = min(1, delta/|r|)), which is NOT the true minimum
        of the Huber loss. The gradient at the IRLS solution is typically ~1e-2,
        far from zero.

        Use ``solver='newton'`` instead — Newton uses the true Hessian
        (binary weights w = 1 if |r|<=delta, 0 otherwise) and converges
        to the global minimum (gradient ≈ 0).

        Raises
        ------
        NotImplementedError
            Always. Use solver='newton' or solver='fista' for Huber loss.
        """
        raise NotImplementedError(
            "IRLS is not supported for Huber loss. "
            "IRLS converges to a fixed point of the weighted LS problem, "
            "not the true Huber loss minimum. "
            "Use solver='newton' (recommended) or solver='fista' instead."
        )
