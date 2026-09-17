"""Penalized quantile regression.

Quantile loss is NOT a GLM loss — it inherits from LossBase, not GLMLoss.
This class provides a clean API with quantile-specific parameters and scoring.
"""

__all__ = ["PenalizedQuantileRegression"]

import numpy as np
from ._base import PenalizedGeneralizedLinearModel


class PenalizedQuantileRegression(PenalizedGeneralizedLinearModel):
    """Penalized quantile regression.

    Minimizes: quantile_loss(X, y, coef) + penalty(coef)

    The quantile loss (pinball loss) estimates conditional quantiles:
    - quantile=0.5 gives the conditional median (robust to outliers)
    - quantile=0.9 gives the 90th percentile
    - quantile=0.1 gives the 10th percentile

    Supports all penalties (L1, L2, ElasticNet, SCAD, MCP, group, adaptive).

    Parameters
    ----------
    quantile : float, default=0.5
        Quantile to estimate, must be in (0, 1).
    penalty : str or Penalty, default='l2'
        Penalty type.
    alpha : float, default=1.0
        Regularization strength.
    solver : str, default='auto'
        Solver policy. For L2/no-penalty Quantile objectives, ``'auto'``
        resolves to the maintained Quantile IRLS algorithm. Explicit
        ``solver='irls'`` requests the same route, while explicit ordinary
        ``solver='fista'`` is also supported and executes the generic FISTA
        engine rather than being silently substituted by IRLS. IRLS remains
        the preferred automatic route for these smooth penalties. Convex sparse
        L1/ElasticNet objectives use ordinary FISTA, while SCAD/MCP use the
        dedicated Proximal IRLS-CD continuation path. FISTA-BB and shared ADMM
        are not maintained Quantile routes. Estimator/CV ``solver='lbfgs'``
        also fails closed because the maintained shared L-BFGS route assumes a
        smooth loss gradient; the separate low-level omitted/uniform-weight
        Quantile L-BFGS compatibility surface is not an estimator-level solver
        option.
    max_iter : int, default=1000
        Maximum iterations.
    tol : float, default=1e-4
        Convergence tolerance. Quantile IRLS uses a tighter internal tolerance
        (at most 1e-8) on the maintained L2/no-penalty route.
    fit_intercept : bool, default=True
        Whether to fit an intercept.
    device : str, default='auto'
        Device: 'auto', 'cpu', 'cuda', 'torch'.
    loss_kwargs : dict, optional
        Advanced low-level Quantile-loss overrides. For historical
        compatibility, ``loss_kwargs={'quantile': q}`` takes precedence over
        the typed ``quantile=`` argument for fitting and scoring, while the
        public ``quantile`` attribute retains the outer constructor value for
        clone identity. Prefer the typed ``quantile=`` argument for ordinary
        use and avoid supplying conflicting values unless this compatibility
        behavior is intentionally required.

    Examples
    --------
    >>> from statgpu.linear_model import PenalizedQuantileRegression
    >>> # Median regression; auto resolves to IRLS for this L2 objective.
    >>> model = PenalizedQuantileRegression(quantile=0.5, penalty='l2', alpha=0.01)
    >>> model.fit(X, y)
    >>> pred = model.predict(X_test)

    >>> # 90th percentile with L1 penalty (ordinary FISTA route)
    >>> model = PenalizedQuantileRegression(quantile=0.9, penalty='l1', alpha=0.05)
    """

    def __init__(self, quantile=0.5, penalty='l2', alpha=1.0, *,
                 solver='auto', max_iter=1000, tol=1e-4,
                 fit_intercept=True, l1_ratio=0.5,
                 penalty_kwargs=None, device='auto',
                 loss_kwargs=None, **kwargs):
        if not 0.0 < quantile < 1.0:
            raise ValueError(f"quantile must be in (0, 1), got {quantile}")

        _lk = {'quantile': quantile}
        if loss_kwargs:
            _lk.update(loss_kwargs)

        super().__init__(
            loss='quantile', penalty=penalty, alpha=alpha,
            solver=solver, max_iter=max_iter, tol=tol,
            fit_intercept=fit_intercept, l1_ratio=l1_ratio,
            penalty_kwargs=penalty_kwargs, device=device,
            loss_kwargs=_lk, **kwargs,
        )
        self.quantile = quantile

    def _resolved_quantile_loss_kwargs(self) -> dict:
        """Build effective Quantile kwargs without mutating clone-safe state.

        ``BaseEstimator`` restores public constructor attributes to the exact
        values supplied to the most-derived wrapper. Consequently
        ``self.loss_kwargs`` legitimately remains ``None`` when omitted even
        though this typed wrapper owns a separate ``quantile`` parameter.
        Numerical resolution must recombine those public controls. An explicit
        ``loss_kwargs['quantile']`` retains the historical precedence used by
        this wrapper; otherwise the typed ``quantile`` value is authoritative.
        """
        kwargs = {"quantile": float(getattr(self, "quantile", 0.5))}
        if self.loss_kwargs:
            kwargs.update(dict(self.loss_kwargs))
        return kwargs

    def _resolve_loss(self):
        from statgpu.losses import get_loss

        kwargs = self._resolved_quantile_loss_kwargs()
        # The shared fit preamble mirrors clone-safe public ``loss_kwargs`` into
        # ``_loss_kwargs``. Restore the resolved internal kwargs here so every
        # downstream numerical helper sees the same quantile as the loss object
        # without changing ``get_params()`` / sklearn-clone constructor state.
        self._loss_kwargs = dict(kwargs)
        return get_loss("quantile", **kwargs)

    def _fit_initial(self, X, y, backend_name="numpy"):
        """Preserve the typed quantile in adaptive-L1 initialization."""
        penalty_name = str(getattr(self._penalty, "name", "")).lower()
        if penalty_name not in ("adaptive_l1", "adaptive_lasso"):
            return super()._fit_initial(X, y, backend_name=backend_name)

        from statgpu.backends import get_backend
        from statgpu.backends._utils import _to_numpy
        from statgpu.linear_model.penalized._fit_mixin import _irls_ridge_init

        if backend_name in ("torch", "cupy"):
            backend = get_backend(backend=backend_name, device="cuda")
            X_b = backend.asarray(X, dtype=backend.float64)
            y_b = backend.asarray(y, dtype=backend.float64)
        else:
            X_b = np.asarray(_to_numpy(X), dtype=np.float64)
            y_b = np.asarray(_to_numpy(y), dtype=np.float64)

        return _irls_ridge_init(
            X_b,
            y_b,
            loss_name="quantile",
            alpha=0.01,
            max_iter=100,
            tol=1e-4,
            loss_kwargs=self._resolved_quantile_loss_kwargs(),
        )

    def predict(self, X, return_cpu=True):
        """Predict using fitted model (identity link).

        Returns X @ coef + intercept.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        return_cpu : bool, default=True

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        X = self._prepare_predict_X(X)
        backend_name = self._prediction_backend_name()

        if backend_name == "cupy":
            import cupy as cp
            Xb = cp.asarray(self._to_array(X, Device.CUDA))
            coef = cp.asarray(self.coef_)
            raw = Xb @ coef
            if self._effective_intercept:
                raw += cp.asarray(self.intercept_, dtype=raw.dtype)
            return _to_numpy(raw) if return_cpu else raw

        if backend_name == "torch":
            import torch
            Xb = self._to_array(X, Device.TORCH, backend="torch").to(torch.float64)
            coef = torch.as_tensor(self.coef_, dtype=Xb.dtype, device=Xb.device)
            raw = Xb @ coef
            if self._effective_intercept:
                raw = raw + torch.as_tensor(self.intercept_, dtype=raw.dtype, device=raw.device)
            return _to_numpy(raw) if return_cpu else raw

        raw = X @ self.coef_
        if self._effective_intercept:
            raw += self.intercept_
        return raw

    def score(self, X, y, sample_weight=None):
        """Return negative pinball loss on test data; higher is better.

        For quantile=0.5, this is negative mean absolute error / 2.
        """
        y_pred = self.predict(X, return_cpu=True)
        y = np.asarray(y)
        u = y - y_pred
        q = float(self._resolved_quantile_loss_kwargs()["quantile"])
        per_sample = np.where(u >= 0, q * u, (q - 1.0) * u)
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=np.float64)
            pinball = float(np.average(per_sample, weights=sw))
        else:
            pinball = float(np.mean(per_sample))
        return -pinball  # Negative because higher is "better" in sklearn convention


# Import needed for predict()
from statgpu._config import Device
from statgpu.backends._utils import _to_numpy
