"""
Huber loss for robust regression.

Loss = (1/n) * sum(rho_delta(y_i - eta_i))
where rho_delta(u) = 0.5 * u^2           if |u| <= delta
                   = delta * (|u| - 0.5*delta)  otherwise

Supports numpy / cupy / torch backends via _xp dispatch.

Matches R's MASS::rlm() with method='M' and psi='huber'.
"""

from statgpu.backends._array_ops import _xp as _get_xp
from ._base import LossBase
from ._registry import register_loss


@register_loss('huber')
class HuberLoss(LossBase):
    """Huber loss for robust regression.

    Parameters
    ----------
    delta : float, default=1.0
        Threshold parameter. Residuals with |u| <= delta use quadratic
        loss; larger residuals use linear loss.
    """

    name = "huber"
    y_type = "continuous"
    smooth_gradient = True    # gradient is continuous everywhere
    has_hessian = True        # Piecewise Hessian (discontinuous at |u|=delta, works in practice)

    # Optimization hints
    _lipschitz_safety = 1.0

    def __init__(self, delta: float = 1.0):
        if delta <= 0:
            raise ValueError(f"delta must be positive, got {delta}")
        self.delta = float(delta)

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

    def hessian(self, X, y, coef, sample_weight=None):
        """Hessian: X' W X / n, where w_i = 1 if |u_i| <= delta else 0.

        Piecewise constant — discontinuous at |u|=delta but works
        well in practice for Newton-type solvers.
        """
        xp = _get_xp(X)
        eta = X @ coef
        u = y - eta
        d = self.delta
        # Weight: 1 in quadratic region, 0 in linear region
        in_quad = (xp.abs(u) <= d)
        if xp.__name__ == "torch":
            w = in_quad.to(u.dtype)
        else:
            w = in_quad.astype(u.dtype)
        if sample_weight is not None:
            w = w * sample_weight
            return X.T @ (X * w[:, None]) / float(sample_weight.sum())
        return X.T @ (X * w[:, None]) / X.shape[0]
