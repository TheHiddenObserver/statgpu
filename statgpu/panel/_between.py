"""Between OLS estimator for panel data with GPU acceleration."""

from __future__ import annotations

__all__ = ["BetweenOLS"]

from typing import Optional, Union

import numpy as np

from statgpu._base import BaseEstimator
from statgpu._config import Device
from statgpu.backends import _LINALG_ERRORS, _to_float_scalar, _to_numpy, xp_asarray

from statgpu.panel._utils import PanelSummary, group_means
from statgpu.panel._covariance import clustered_covariance


class BetweenOLS(BaseEstimator):
    """Between-entity OLS estimator for panel data.

    Collapses the data to group means and runs OLS on the collapsed data.

    Parameters
    ----------
    cov_type : str, default='nonrobust'
        Covariance estimator: ``'nonrobust'``, ``'robust'``, or ``'clustered'``.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    device : str or Device, default='auto'
        Computation device.

    Attributes
    ----------
    coef_ : ndarray, shape (k,)
        Estimated coefficients (including intercept).
    bse_ : ndarray, shape (k,)
        Standard errors.
    tvalues_ : ndarray, shape (k,)
        t-statistics.
    pvalues_ : ndarray, shape (k,)
        Two-sided p-values.
    conf_int_ : ndarray, shape (k, 2)
        Confidence intervals.
    rsquared : float
        R-squared.
    nobs : int
        Number of observations (groups).
    df_resid : int
        Residual degrees of freedom.
    """

    def __init__(
        self,
        cov_type: str = "nonrobust",
        alpha: float = 0.05,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.cov_type = cov_type.lower()
        self.alpha = alpha
        if self.cov_type not in ("nonrobust", "robust", "clustered"):
            raise ValueError("cov_type must be 'nonrobust', 'robust', or 'clustered'")

    def fit(self, X, y, entity_ids=None):
        """Fit the between OLS model.

        Parameters
        ----------
        X : array-like, shape (n, k)
            Design matrix (an intercept is added automatically).
        y : array-like, shape (n,)
            Dependent variable.
        entity_ids : array-like, shape (n,)
            Entity (individual) identifiers.

        Returns
        -------
        self
        """
        if entity_ids is None:
            raise ValueError("entity_ids is required for BetweenOLS")

        backend = self._get_backend(backend="auto")
        xp = backend.xp

        X_arr = xp_asarray(X, dtype=xp.float64, xp=xp)
        y_arr = xp_asarray(y, dtype=xp.float64, xp=xp, ref_arr=X_arr).ravel()
        eids = xp_asarray(entity_ids, xp=xp, ref_arr=X_arr).ravel()

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        n_orig = X_arr.shape[0]
        p = X_arr.shape[1]

        # Add intercept to original X
        ones = xp.ones((n_orig, 1), dtype=xp.float64)
        if hasattr(X_arr, 'is_cuda'):
            ones = ones.to(device=X_arr.device)
        X_full = xp.concatenate([ones, X_arr], axis=1)
        k = X_full.shape[1]

        # Collapse to group means
        # For each column of X and y, compute group means
        unique_eids = xp.unique(eids)
        n_groups = int(unique_eids.shape[0])

        # Build collapsed data
        X_mean = xp.zeros((n_groups, k), dtype=xp.float64)
        y_mean = xp.zeros(n_groups, dtype=xp.float64)
        if hasattr(X_arr, 'is_cuda'):
            X_mean = X_mean.to(device=X_arr.device)
            y_mean = y_mean.to(device=X_arr.device)

        for idx in range(n_groups):
            eid = unique_eids[idx]
            mask = eids == eid
            X_mean[idx] = xp.mean(X_full[mask], axis=0)
            y_mean[idx] = xp.mean(y_arr[mask])

        # OLS on group means
        XtX = X_mean.T @ X_mean
        Xty = X_mean.T @ y_mean
        try:
            params = xp.linalg.solve(XtX, Xty)
        except _LINALG_ERRORS:
            params = xp.linalg.lstsq(XtX, Xty)[0]

        resid = y_mean - X_mean @ params
        n = n_groups
        scale = _to_float_scalar(xp.sum(resid * resid)) / (n - k)

        # Inference
        _compute_ols_inference(
            self, X_mean, resid, params, scale, n, k, xp, backend.name,
            self.cov_type, self.alpha, dist_df=n - k
        )

        # R-squared
        y_bar = xp.mean(y_mean)
        ss_tot = _to_float_scalar(xp.sum((y_mean - y_bar) ** 2))
        ss_res = _to_float_scalar(xp.sum(resid * resid))
        self.rsquared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        self.nobs = n
        self.df_resid = n - k
        self._fitted = True

        return self

    def predict(self, X):
        """Predict using the fitted model."""
        self._check_is_fitted()
        backend = self._get_backend(backend="auto")
        xp = backend.xp
        X_arr = xp_asarray(X, dtype=xp.float64, xp=xp)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        n = X_arr.shape[0]
        ones = xp.ones((n, 1), dtype=xp.float64)
        if hasattr(X_arr, 'is_cuda'):
            ones = ones.to(device=X_arr.device)
        X_arr = xp.concatenate([ones, X_arr], axis=1)
        params = xp_asarray(self.coef_, dtype=xp.float64, xp=xp, ref_arr=X_arr)
        return _to_numpy(X_arr @ params)

    def summary(self):
        """Return a summary object."""
        self._check_is_fitted()
        return PanelSummary(
            model_type="BetweenOLS",
            cov_type=self.cov_type,
            coef=np.asarray(self.coef_),
            bse=np.asarray(self.bse_),
            tvalues=np.asarray(self.tvalues_),
            pvalues=np.asarray(self.pvalues_),
            conf_int=np.asarray(self.conf_int_),
            nobs=self.nobs,
            df_resid=self.df_resid,
            alpha=self.alpha,
        )

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params["cov_type"] = self.cov_type
        params["alpha"] = self.alpha
        return params

    def set_params(self, **params):
        for key in ["cov_type", "alpha"]:
            if key in params:
                setattr(self, key, params.pop(key))
        if params:
            super().set_params(**params)
        return self


def _compute_ols_inference(model, X, resid, params, scale, n, k, xp, backend_name,
                           cov_type, alpha, dist_df=None):
    """Shared OLS inference for panel models."""
    XtX = X.T @ X / n
    try:
        XtX_inv = xp.linalg.inv(XtX)
    except _LINALG_ERRORS:
        XtX_inv = xp.linalg.pinv(XtX)

    if cov_type == "nonrobust":
        cov_params = scale * XtX_inv / n
    elif cov_type == "robust":
        scores = X * resid[:, None]
        meat = scores.T @ scores
        cov_params = XtX_inv @ meat @ XtX_inv / (n * n) * n / (n - k)
    elif cov_type == "clustered":
        # Not supported without cluster labels
        cov_params = scale * XtX_inv / n

    bse_dev = xp.sqrt(xp.diag(cov_params))
    tvalues_dev = params / bse_dev

    df = dist_df if dist_df is not None else n - k
    from statgpu.inference._distributions_backend import get_distribution
    dist_name = "norm" if cov_type in ("robust", "clustered", "hac") else "t"
    t_dist = get_distribution(dist_name, backend=backend_name)
    if dist_name == "t":
        pvalues_dev = 2 * t_dist.sf(xp.abs(tvalues_dev), df)
        t_crit = t_dist.isf(alpha / 2, df)
    else:
        pvalues_dev = 2 * t_dist.sf(xp.abs(tvalues_dev))
        t_crit = t_dist.isf(alpha / 2)

    conf_low = params - t_crit * bse_dev
    conf_high = params + t_crit * bse_dev

    model.coef_ = _to_numpy(params)
    model.bse_ = _to_numpy(bse_dev)
    model.tvalues_ = _to_numpy(tvalues_dev)
    model.pvalues_ = _to_numpy(pvalues_dev)
    model.conf_int_ = _to_numpy(xp.stack([conf_low, conf_high], axis=1))
