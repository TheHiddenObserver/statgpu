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
    has_hessian = False       # Hessian is discontinuous at |u|=delta

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

        Computed as: 0.5 * min(u^2, delta*(2*|u| - delta))
        """
        u = y - eta
        xp = _get_xp(u)
        d = self.delta
        abs_u = xp.abs(u)
        quad = u * u
        linear = d * (2.0 * abs_u - d)
        if xp.__name__ == "torch":
            return 0.5 * xp.minimum(quad, linear)
        return 0.5 * xp.minimum(quad, linear)

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
