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
        Solver: 'auto', 'fista', 'fista_bb'.
        'auto' selects FISTA (quantile loss has no Hessian).
    max_iter : int, default=1000
        Maximum iterations.
    tol : float, default=1e-4
        Convergence tolerance.  For quantile regression, tighter
        tolerance (1e-8) is used internally for IRLS convergence.
    fit_intercept : bool, default=True
        Whether to fit an intercept.
    device : str, default='auto'
        Device: 'auto', 'cpu', 'cuda', 'torch'.

    Examples
    --------
    >>> from statgpu.linear_model import PenalizedQuantileRegression
    >>> # Median regression
    >>> model = PenalizedQuantileRegression(quantile=0.5, penalty='l2', alpha=0.01)
    >>> model.fit(X, y)
    >>> pred = model.predict(X_test)

    >>> # 90th percentile with L1 penalty (sparse)
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
        """Pinball loss (quantile loss) on test data. Lower is better.

        For quantile=0.5, this is the mean absolute error / 2.
        """
        y_pred = self.predict(X, return_cpu=True)
        y = np.asarray(y)
        u = y - y_pred
        q = self.quantile
        pinball = np.mean(np.where(u >= 0, q * u, (q - 1.0) * u))
        return -pinball  # Negative because higher is "better" in sklearn convention


# Import needed for predict()
from statgpu._config import Device
from statgpu.backends._utils import _to_numpy
