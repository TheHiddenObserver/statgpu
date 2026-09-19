"""
Quantile loss (pinball loss) for quantile regression.

Loss = (1/n) * sum(rho_tau(y_i - eta_i))
where rho_tau(u) = u * (tau - 1{u < 0})

For tau=0.5 this reduces to the absolute loss (median regression).

Supports numpy / cupy / torch backends via _xp dispatch.

Matches R's quantreg::rq() interface.
"""

from numbers import Real

import numpy as np

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

    _lipschitz_static = True
    name = "quantile"
    y_type = "continuous"
    smooth_gradient = False   # non-smooth at u=0; first-order routes use a subgradient
    has_hessian = False
    _supports_irls = True     # has irls() method (Frisch-Newton)

    # Optimization hints
    _lipschitz_safety = 1.0
    _prefer_fista_over_bb = False

    def __init__(self, quantile: float = 0.5):
        if (
            isinstance(quantile, (bool, np.bool_))
            or not isinstance(quantile, Real)
            or not np.isfinite(float(quantile))
            or not 0.0 < float(quantile) < 1.0
        ):
            raise ValueError(
                f"quantile must be a finite real number in (0, 1), got {quantile}"
            )
        self.quantile = float(quantile)
        self._tau = self.quantile

    def validate_response(self, y):
        """Validate a continuous Quantile response on its current backend."""
        xp = _get_xp(y)
        if xp.__name__ == "torch":
            import torch

            values = y if torch.is_tensor(y) else torch.as_tensor(y)
        else:
            try:
                values = xp.asarray(y)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "quantile response must be a real numeric array-like"
                ) from exc

        ndim = int(values.ndim)
        if ndim == 2 and int(values.shape[1]) == 1:
            values = values.reshape(-1)
        elif ndim != 1:
            raise ValueError(
                "quantile response must be one-dimensional; a single-column "
                "(n_samples, 1) response is also accepted"
            )
        if int(values.shape[0]) == 0:
            raise ValueError("quantile response must contain at least one observation")

        if xp.__name__ == "torch":
            import torch

            nonreal = torch.is_complex(values)
        else:
            nonreal = getattr(values.dtype, "kind", "") not in "biuf"
        if bool(nonreal.item() if hasattr(nonreal, "item") else nonreal):
            raise ValueError("quantile response must contain real numeric values")

        try:
            invalid = xp.any(~xp.isfinite(values))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "quantile response must contain real numeric finite values"
            ) from exc
        if bool(invalid.item() if hasattr(invalid, "item") else invalid):
            raise ValueError("quantile response requires finite values")
        return values

    def lipschitz(self, X, coef, y=None, sample_weight=None):
        """Return the design-scaled step parameter for first-order routes.

        Quantile/check loss has a discontinuous subgradient and therefore does
        **not** have a classical smooth-gradient Lipschitz constant. The shared
        solver interface nevertheless asks losses for a positive ``lipschitz``
        scale. For an unweighted objective we use

        ``max(tau, 1-tau) * lambda_max(X'X / n)``.

        For normalized analytic weights, the same design scale follows the
        fitted objective and uses

        ``max(tau, 1-tau) * lambda_max(X' W X / sum(w))``.

        This is an initialization/fixed-step design scale for a non-smooth
        subgradient route; it must not be interpreted as a proof that textbook
        smooth-FISTA assumptions hold for pinball loss.
        """
        from statgpu.backends._array_ops import _max_eigval_power

        if sample_weight is None:
            gram = (X.T @ X) / X.shape[0]
        else:
            xp = _get_xp(X)
            if xp.__name__ == "torch":
                import torch

                weight_dtype = (
                    X.dtype
                    if getattr(X.dtype, "is_floating_point", False)
                    else torch.float64
                )
                if torch.is_tensor(sample_weight):
                    sw = sample_weight.to(
                        dtype=weight_dtype,
                        device=X.device,
                    )
                else:
                    sw = torch.as_tensor(
                        sample_weight,
                        dtype=weight_dtype,
                        device=X.device,
                    )
            else:
                try:
                    design_kind = np.dtype(X.dtype).kind
                except (TypeError, ValueError):
                    design_kind = "f"
                weight_dtype = (
                    X.dtype
                    if design_kind in "fc"
                    else xp.float64
                )
                sw = xp.asarray(sample_weight, dtype=weight_dtype)
            sw = sw.reshape(-1)
            gram = X.T @ (X * sw[:, None]) / xp.sum(sw)

        grad_bound = max(self._tau, 1.0 - self._tau)
        return grad_bound * _max_eigval_power(gram)

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
        """Gradient w.r.t. eta: -tau + 1.0 * (u < 0).

        Thus u < 0 gives 1 - tau, while u >= 0 gives -tau. At u=0
        (y=eta), the implementation uses -tau as its subgradient choice.
        """
        u = y - eta
        tau = self._tau
        xp = _get_xp(u)
        if xp.__name__ == "torch":
            neg_mask = (u < 0).to(u.dtype)
        else:
            neg_mask = (u < 0).astype(u.dtype)
        return -tau + 1.0 * neg_mask

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

        # Gradient: -tau + 1.0 * (u < 0)
        if xp.__name__ == "torch":
            neg_mask = (u < 0).to(u.dtype)
        else:
            neg_mask = (u < 0).astype(u.dtype)
        resid = -tau + 1.0 * neg_mask

        # Aggregate. Keep the objective scalar on the active backend so async
        # FISTA does not synchronize GPU->CPU on every iteration merely to
        # obtain a value that is consumed by device-side line-search algebra.
        if sample_weight is not None:
            sw_total = xp.sum(sample_weight)
            val = xp.sum(sample_weight * ps) / sw_total
            grad = X.T @ (sample_weight * resid) / sw_total
        else:
            n = X.shape[0]
            val = xp.sum(ps) / n
            grad = X.T @ resid / n
        return val, grad

    def irls(self, X, y, penalty=None, max_iter=100, tol=1e-6, init_coef=None, eps=1e-8,
             sample_weight=None, fit_intercept=False):
        """IRLS (Iteratively Reweighted Least Squares) for quantile regression.

        Same algorithm as statsmodels QuantReg (Frisch-Newton variant).
        Much faster convergence than FISTA for quantile loss.
        Supports numpy / cupy / torch backends.

        Parameters
        ----------
        X : array of shape (n, p)
        y : array of shape (n,)
        penalty : Penalty, optional
            L2 penalty only. Non-smooth penalties, including ElasticNet, L1,
            adaptive/group penalties, SCAD, and MCP, are not supported by this
            IRLS subproblem; use FISTA or the maintained dedicated non-convex
            Quantile route instead.
        max_iter : int
        tol : float
        init_coef : array of shape (p,), optional
        eps : float
            Small constant to avoid division by zero in weights.
        sample_weight : array of shape (n,), optional
            Sample weights for weighted quantile regression.
        fit_intercept : bool
            If True, X includes an intercept column (last column) which should
            not be penalized.

        Returns
        -------
        coef : array of shape (p,)
        n_iter : int
        """
        if penalty is not None:
            pen_name = str(getattr(penalty, "name", "")).lower().strip()
            if pen_name != "l2":
                display_name = pen_name or type(penalty).__name__
                raise ValueError(
                    "QuantileLoss.irls() supports only L2 or no penalty; "
                    f"got penalty='{display_name}'. Use FISTA or the dedicated "
                    "Quantile non-convex solver for non-smooth penalties."
                )

        xp = _get_xp(X)
        X_dev = xp.asarray(X, dtype=xp.float64)
        y_dev = xp.asarray(y, dtype=xp.float64)
        n, p = int(X_dev.shape[0]), int(X_dev.shape[1])
        tau = self._tau

        # Handle sample_weight
        if sample_weight is not None:
            sw = xp.asarray(sample_weight, dtype=xp.float64)
            # Ensure sw is on same device as X for torch CUDA
            if hasattr(X_dev, 'device') and hasattr(sw, 'to'):
                sw = sw.to(device=X_dev.device)
            sw_sum = float(xp.sum(sw))
            sw = sw * (n / sw_sum)  # normalize so sum(sw) = n
        else:
            sw = None

        if init_coef is not None:
            beta = xp.asarray(init_coef, dtype=xp.float64)
            if xp.__name__ == "torch":
                beta = beta.to(device=X_dev.device).clone()
            else:
                beta = beta.copy()
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

            # Apply sample weights
            if sw is not None:
                w = w * sw

            # Weighted least squares + L2 penalty
            WX = X_dev * w[:, None]
            XtWX = X_dev.T @ WX
            XtWy = X_dev.T @ (w * y_dev)

            # Add numerical ridge plus optional L2 curvature. The public
            # objective uses average-loss scaling, so this unnormalized normal
            # equation receives n * alpha on penalized coordinates.
            ridge = eps * xp.eye(p, dtype=xp.float64) if xp.__name__ != "torch" else eps * xp.eye(p, dtype=xp.float64, device=X_dev.device)
            A = XtWX + ridge

            if penalty is not None:
                alpha = float(penalty.alpha)
                pen_diag = xp.ones(p, dtype=xp.float64) if xp.__name__ != "torch" else xp.ones(p, dtype=xp.float64, device=X_dev.device)
                if fit_intercept and p > 1:
                    pen_diag[-1] = 0.0  # don't penalize intercept
                A = A + n * alpha * xp.diag(pen_diag)

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

        import warnings
        from statgpu.solvers._convergence import ConvergenceWarning

        warnings.warn(
            "QuantileLoss.irls() did not converge within "
            f"{max_iter} iterations; returning the final iterate.",
            ConvergenceWarning,
            stacklevel=2,
        )
        return beta, max_iter
