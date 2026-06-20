"""
Quantile loss (pinball loss) for quantile regression.

Loss = (1/n) * sum(rho_tau(y_i - eta_i))
where rho_tau(u) = u * (tau - 1{u < 0})

For tau=0.5 this reduces to the absolute loss (median regression).

Supports numpy / cupy / torch backends via _xp dispatch.

Matches R's quantreg::rq() interface.
"""

from statgpu.backends._array_ops import _xp as _get_xp
from ._base import LossBase
from ._registry import register_loss


@register_loss('quantile')
class QuantileLoss(LossBase):
    """Quantile regression loss (pinball loss).

    Parameters
    ----------
    quantile : float, default=0.5
        Target quantile in (0, 1).
    """

    name = "quantile"
    y_type = "continuous"
    smooth_gradient = False   # non-smooth at u=0; FISTA handles via proximal
    has_hessian = False

    # Optimization hints
    _lipschitz_safety = 1.0
    _prefer_fista_over_bb = False

    def __init__(self, quantile: float = 0.5):
        if not 0.0 < quantile < 1.0:
            raise ValueError(f"quantile must be in (0, 1), got {quantile}")
        self.quantile = float(quantile)
        self._tau = self.quantile

    # ── Per-sample formulas (backend-aware, dtype-safe) ──────────────

    def per_sample_value(self, eta, y):
        """Pinball loss: rho_tau(y - eta).

        rho_tau(u) = tau * max(u, 0) + (1 - tau) * max(-u, 0)
        """
        u = y - eta
        tau = self._tau
        xp = _get_xp(u)
        if xp.__name__ == "torch":
            pos = xp.clamp(u, min=0)
            neg = xp.clamp(-u, min=0)
        else:
            pos = xp.maximum(u, 0)
            neg = xp.maximum(-u, 0)
        return tau * pos + (1.0 - tau) * neg

    def per_sample_gradient(self, eta, y):
        """Gradient w.r.t. eta: -tau + (1-tau) * (u < 0).

        At u=0 (y=eta), returns -tau (arbitrary subgradient choice).
        """
        u = y - eta
        tau = self._tau
        xp = _get_xp(u)
        if xp.__name__ == "torch":
            neg_mask = (u < 0).to(u.dtype)
        else:
            neg_mask = (u < 0).astype(u.dtype)
        return -tau + (1.0 - tau) * neg_mask

    def irls(self, X, y, max_iter=100, tol=1e-6, init_coef=None, eps=1e-8):
        """IRLS (Iteratively Reweighted Least Squares) for quantile regression.

        Same algorithm as statsmodels QuantReg (Frisch-Newton variant).
        Much faster convergence than FISTA for quantile loss.
        Supports numpy / cupy / torch backends.

        Parameters
        ----------
        X : array of shape (n, p)
        y : array of shape (n,)
        max_iter : int
        tol : float
        init_coef : array of shape (p,), optional
        eps : float
            Small constant to avoid division by zero in weights.

        Returns
        -------
        coef : array of shape (p,)
        n_iter : int
        """
        xp = _get_xp(X)
        X_dev = xp.asarray(X, dtype=xp.float64)
        y_dev = xp.asarray(y, dtype=xp.float64)
        n, p = int(X_dev.shape[0]), int(X_dev.shape[1])
        tau = self._tau

        if init_coef is not None:
            beta = xp.asarray(init_coef, dtype=xp.float64).copy()
        else:
            # OLS initial estimate
            if xp.__name__ == "torch":
                beta = xp.linalg.lstsq(X_dev, y_dev).solution
            else:
                beta = xp.linalg.lstsq(X_dev, y_dev, rcond=None)[0]

        for iteration in range(max_iter):
            eta = X_dev @ beta
            r = y_dev - eta

            # IRLS weights
            abs_r = xp.abs(r)
            if xp.__name__ == "torch":
                abs_r_safe = xp.maximum(abs_r, xp.tensor(eps, dtype=abs_r.dtype, device=abs_r.device))
                neg_mask = (r < 0).to(abs_r.dtype)
            else:
                abs_r_safe = xp.maximum(abs_r, eps)
                neg_mask = (r < 0).astype(abs_r.dtype)
            w = (tau + (1.0 - 2.0 * tau) * neg_mask) / abs_r_safe

            # Weighted least squares: beta = (X'WX + ridge)^{-1} X'Wy
            WX = X_dev * w[:, None]
            XtWX = X_dev.T @ WX
            XtWy = X_dev.T @ (w * y_dev)

            # Ridge for numerical stability
            if xp.__name__ == "torch":
                ridge = eps * xp.eye(p, dtype=xp.float64, device=X_dev.device)
            else:
                ridge = eps * xp.eye(p)

            beta_new = xp.linalg.solve(XtWX + ridge, XtWy)

            # Convergence check
            diff = beta_new - beta
            if xp.__name__ == "torch":
                delta = float(xp.linalg.norm(diff).item())
            else:
                delta = float(xp.linalg.norm(diff))

            if delta < tol:
                return beta_new, iteration + 1

            beta = beta_new

        return beta, max_iter
