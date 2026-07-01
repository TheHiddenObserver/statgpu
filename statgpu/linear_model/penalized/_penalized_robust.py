"""Penalized robust regression (Huber, Bisquare, Fair).

These losses are NOT GLM losses — they inherit from LossBase, not GLMLoss.
This class provides a clean API with loss-specific parameters exposed directly.
"""

__all__ = ["PenalizedRobustRegression"]

from ._base import PenalizedGeneralizedLinearModel


class PenalizedRobustRegression(PenalizedGeneralizedLinearModel):
    """Penalized robust regression using Huber, Bisquare (Tukey), or Fair loss.

    Minimizes: loss(X, y, coef) + penalty(coef)

    Unlike OLS (squared_error), robust losses down-weight outliers,
    giving higher breakdown point.  Supports all penalties (L1, L2,
    ElasticNet, SCAD, MCP, group, adaptive).

    Parameters
    ----------
    loss : str, default='huber'
        Robust loss function: 'huber', 'bisquare', or 'fair'.
    penalty : str or Penalty, default='l2'
        Penalty type.
    alpha : float, default=1.0
        Regularization strength.
    epsilon : float, default=1.35
        Robustness tuning parameter (dimensionless).  The effective
        threshold is ``epsilon * scale`` where ``scale`` depends on
        ``method``.  For bisquare, the default is 4.685.
    method : str, default='MAD'
        Scale estimation method:
        - ``'MAD'``: one-shot ``scale = MAD(residuals) / 0.6745``.
        - ``'huber_prop2'``: Huber's Proposal 2 (iterative re-estimation).
    solver : str, default='auto'
        Solver: 'auto', 'fista', 'fista_bb', 'newton', 'lbfgs'.
        'auto' selects Newton for smooth penalties, FISTA for non-smooth.
    max_iter : int, default=1000
        Maximum iterations.
    tol : float, default=1e-4
        Convergence tolerance.
    fit_intercept : bool, default=True
        Whether to fit an intercept.
    device : str, default='auto'
        Device: 'auto', 'cpu', 'cuda', 'torch'.

    Examples
    --------
    >>> from statgpu.linear_model import PenalizedRobustRegression
    >>> model = PenalizedRobustRegression(loss='huber', penalty='l2', alpha=0.01)
    >>> model.fit(X, y)
    >>> pred = model.predict(X_test)

    >>> # Bisquare with SCAD penalty
    >>> model = PenalizedRobustRegression(loss='bisquare', penalty='scad', alpha=0.05, epsilon=4.685)

    >>> # Huber with Proposal 2 scale estimation (matches sklearn)
    >>> model = PenalizedRobustRegression(loss='huber', method='huber_prop2')
    """

    def __init__(self, loss='huber', penalty='l2', alpha=1.0, *,
                 epsilon=None, method='MAD',
                 solver='auto', max_iter=1000, tol=1e-4,
                 fit_intercept=True, l1_ratio=0.5,
                 penalty_kwargs=None, device='auto',
                 loss_kwargs=None, **kwargs):
        # Set default epsilon per loss type
        if epsilon is None:
            _default_epsilon = {
                'huber': 1.35,
                'bisquare': 4.685,
                'fair': 1.35,
            }
            epsilon = _default_epsilon.get(loss, 1.35)

        # Build loss_kwargs from explicit parameters
        _lk = {'epsilon': epsilon, 'method': method}
        if loss_kwargs:
            _lk.update(loss_kwargs)

        super().__init__(
            loss=loss, penalty=penalty, alpha=alpha,
            solver=solver, max_iter=max_iter, tol=tol,
            fit_intercept=fit_intercept, l1_ratio=l1_ratio,
            penalty_kwargs=penalty_kwargs, device=device,
            loss_kwargs=_lk, **kwargs,
        )
        # Expose parameters for introspection
        self.epsilon = epsilon
        self.method = method

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
        """R² score on response scale."""
        return super().score(X, y, sample_weight=sample_weight)


# Import needed for predict()
from statgpu._config import Device
from statgpu.backends._utils import _to_numpy
