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

    def lipschitz(self, X, coef, y=None, sample_weight=None):
        """Lipschitz constant: lambda_max(X'X) / n. Cached."""
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

    def fused_value_and_gradient(self, X, y, coef, sample_weight=None):
        """Fused value+gradient: single X@coef, shared intermediate results.

        Reduces kernel launches vs separate value() + gradient() calls.
        """
        xp = _get_xp(X)
        eta = X @ coef
        u = y - eta
        tau = self._tau

        # Value: tau * max(u,0) + (1-tau) * max(-u,0)
        if xp.__name__ == "torch":
            pos = xp.clamp(u, min=0)
            neg = xp.clamp(-u, min=0)
        else:
            pos = xp.maximum(u, 0)
            neg = xp.maximum(-u, 0)
        ps = tau * pos + (1.0 - tau) * neg

        # Gradient: -tau + (1-tau) * (u < 0)
        if xp.__name__ == "torch":
            neg_mask = (u < 0).to(u.dtype)
        else:
            neg_mask = (u < 0).astype(u.dtype)
        resid = -tau + (1.0 - tau) * neg_mask

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

    def irls(self, X, y, penalty=None, max_iter=100, tol=1e-6, init_coef=None, eps=1e-8):
        """IRLS (Iteratively Reweighted Least Squares) for quantile regression.

        Same algorithm as statsmodels QuantReg (Frisch-Newton variant).
        Much faster convergence than FISTA for quantile loss.
        Supports numpy / cupy / torch backends.

        Parameters
        ----------
        X : array of shape (n, p)
        y : array of shape (n,)
        penalty : Penalty, optional
            Smooth penalty (L2, ElasticNet). Non-smooth penalties (L1, SCAD, MCP)
            are not supported — use FISTA instead.
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

        # Check penalty compatibility
        if penalty is not None:
            pen_name = type(penalty).__name__.lower()
            if 'l1' in pen_name and 'elastic' not in pen_name and 'adaptive' not in pen_name:
                raise NotImplementedError(
                    "IRLS does not support L1 penalty. Use FISTA instead."
                )
            if 'scad' in pen_name or 'mcp' in pen_name or 'group' in pen_name:
                raise NotImplementedError(
                    "IRLS does not support non-smooth penalties. Use FISTA instead."
                )

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

            # Weighted least squares + penalty
            WX = X_dev * w[:, None]
            XtWX = X_dev.T @ WX
            XtWy = X_dev.T @ (w * y_dev)

            # Add penalty contribution to XtWX
            ridge = eps * xp.eye(p, dtype=xp.float64) if xp.__name__ != "torch" else eps * xp.eye(p, dtype=xp.float64, device=X_dev.device)
            A = XtWX + ridge

            if penalty is not None:
                # For L2: A += n * alpha * I, b += 0
                # For ElasticNet: A += n * alpha * (1-l1_ratio) * I
                if hasattr(penalty, 'alpha'):
                    alpha = float(penalty.alpha)
                    if hasattr(penalty, 'l1_ratio'):
                        l1r = float(penalty.l1_ratio)
                        A = A + n * alpha * (1.0 - l1r) * (xp.eye(p, dtype=xp.float64) if xp.__name__ != "torch" else xp.eye(p, dtype=xp.float64, device=X_dev.device))
                    else:
                        A = A + n * alpha * (xp.eye(p, dtype=xp.float64) if xp.__name__ != "torch" else xp.eye(p, dtype=xp.float64, device=X_dev.device))

            beta_new = xp.linalg.solve(A, XtWy)

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
