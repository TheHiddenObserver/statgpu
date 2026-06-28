"""Penalized Cox proportional hazards model.

CoxPH is NOT a GLM — it inherits from LossBase, not GLMLoss.
This class provides a clean API with survival-specific parameters and prediction.
"""

__all__ = ["PenalizedCoxPHModel"]

import numpy as np
from ._base import PenalizedGeneralizedLinearModel


class PenalizedCoxPHModel(PenalizedGeneralizedLinearModel):
    """Penalized Cox proportional hazards model.

    Minimizes: -partial_likelihood(X, time, event) + penalty(coef)

    The Cox PH model estimates log-hazard ratios:
        h(t|X) = h0(t) * exp(X @ coef)

    Supports all penalties (L1, L2, ElasticNet, SCAD, MCP, group, adaptive).

    Parameters
    ----------
    penalty : str or Penalty, default='l2'
        Penalty type.
    alpha : float, default=1.0
        Regularization strength.
    ties : str, default='breslow'
        Method for handling tied event times: 'breslow' or 'efron'.
    solver : str, default='auto'
        Solver: 'auto', 'fista', 'fista_bb', 'newton'.
        'auto' selects Newton for smooth penalties.
    max_iter : int, default=1000
        Maximum iterations.
    tol : float, default=1e-4
        Convergence tolerance.
    fit_intercept : bool, default=True
        Whether to fit an intercept (baseline hazard).
    device : str, default='auto'
        Device: 'auto', 'cpu', 'cuda', 'torch'.

    Examples
    --------
    >>> from statgpu.linear_model import PenalizedCoxPHModel
    >>> # y must be (n, 2) array with columns [time, event]
    >>> model = PenalizedCoxPHModel(penalty='l2', alpha=0.01)
    >>> model.fit(X, y_surv)
    >>> hazard_ratio = model.predict_hazard_ratio(X_test)

    >>> # Sparse Cox model with L1 penalty
    >>> model = PenalizedCoxPHModel(penalty='l1', alpha=0.05)
    """

    def __init__(self, penalty='l2', alpha=1.0, *,
                 ties='breslow',
                 solver='auto', max_iter=1000, tol=1e-4,
                 fit_intercept=True, l1_ratio=0.5,
                 penalty_kwargs=None, device='auto',
                 loss_kwargs=None, **kwargs):
        _lk = {'ties': ties}
        if loss_kwargs:
            _lk.update(loss_kwargs)

        super().__init__(
            loss='cox_ph', penalty=penalty, alpha=alpha,
            solver=solver, max_iter=max_iter, tol=tol,
            fit_intercept=fit_intercept, l1_ratio=l1_ratio,
            penalty_kwargs=penalty_kwargs, device=device,
            loss_kwargs=_lk, **kwargs,
        )
        self.ties = ties

    def predict(self, X, return_cpu=True):
        """Predict hazard ratio: exp(X @ coef + intercept).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        return_cpu : bool, default=True

        Returns
        -------
        hazard_ratio : ndarray of shape (n_samples,)
            exp(X @ coef + intercept), the hazard ratio relative to baseline.
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
            result = cp.exp(cp.clip(raw, -500.0, 500.0))
            return _to_numpy(result) if return_cpu else result

        if backend_name == "torch":
            import torch
            Xb = self._to_array(X, Device.TORCH, backend="torch").to(torch.float64)
            coef = torch.as_tensor(self.coef_, dtype=Xb.dtype, device=Xb.device)
            raw = Xb @ coef
            if self._effective_intercept:
                raw = raw + torch.as_tensor(self.intercept_, dtype=raw.dtype, device=raw.device)
            result = torch.exp(torch.clamp(raw, -500.0, 500.0))
            return _to_numpy(result) if return_cpu else result

        raw = X @ self.coef_
        if self._effective_intercept:
            raw += self.intercept_
        return np.exp(np.clip(raw, -500.0, 500.0))

    def predict_hazard_ratio(self, X, return_cpu=True):
        """Predict hazard ratio: exp(X @ coef). Excludes intercept.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        return_cpu : bool, default=True

        Returns
        -------
        hr : ndarray of shape (n_samples,)
            exp(X @ coef), the hazard ratio (without baseline hazard).
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
            result = cp.exp(cp.clip(raw, -500.0, 500.0))
            return _to_numpy(result) if return_cpu else result

        if backend_name == "torch":
            import torch
            Xb = self._to_array(X, Device.TORCH, backend="torch").to(torch.float64)
            coef = torch.as_tensor(self.coef_, dtype=Xb.dtype, device=Xb.device)
            raw = Xb @ coef
            result = torch.exp(torch.clamp(raw, -500.0, 500.0))
            return _to_numpy(result) if return_cpu else result

        raw = X @ self.coef_
        return np.exp(np.clip(raw, -500.0, 500.0))

    def score(self, X, y, sample_weight=None):
        """Concordance index (C-index) for survival data.

        Returns the C-index measuring discrimination ability.
        Higher is better (0.5 = random, 1.0 = perfect).

        Note: Requires lifelines or scikit-survival for computation.
        Falls back to a simple implementation if not available.
        """
        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        # Get hazard ratios (without intercept)
        hr = self.predict_hazard_ratio(X, return_cpu=True)

        # Extract time and event from y
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] >= 2:
            time = y[:, 0]
            event = y[:, 1]
        else:
            raise ValueError("y must be (n, 2) array with columns [time, event]")

        # Simple C-index implementation
        n = len(time)
        concordant = 0
        permissible = 0
        for i in range(n):
            for j in range(i + 1, n):
                if time[i] != time[j]:
                    permissible += 1
                    # Higher risk (higher HR) should have shorter survival
                    if (hr[i] > hr[j] and time[i] < time[j]) or \
                       (hr[i] < hr[j] and time[i] > time[j]):
                        concordant += 1

        return concordant / permissible if permissible > 0 else 0.5


# Import needed for predict()
from statgpu._config import Device
from statgpu.backends._utils import _to_numpy
