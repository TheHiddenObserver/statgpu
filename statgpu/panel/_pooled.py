"""Pooled OLS panel data model with GPU acceleration."""

from __future__ import annotations

__all__ = ["PooledOLS"]

from typing import Optional, Union

import numpy as np

from statgpu._base import BaseEstimator
from statgpu._config import Device
from statgpu.backends import _LINALG_ERRORS, _to_float_scalar, _to_numpy, xp_asarray, xp_zeros

from statgpu.panel._utils import PanelSummary, validate_panel_alpha, validate_panel_numeric_data
from statgpu.panel._covariance import clustered_covariance, hac_covariance


class PooledOLS(BaseEstimator):
    """Pooled OLS estimator for panel data.

    Runs OLS on the pooled (stacked) panel data without any demeaning
    or transformation.  Supports multiple covariance estimators.

    Parameters
    ----------
    cov_type : str, default='nonrobust'
        Covariance estimator: ``'nonrobust'``, ``'robust'`` (HC1),
        ``'clustered'``, or ``'hac'``.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    bandwidth : int or None, default=None
        HAC bandwidth (only used when ``cov_type='hac'``).
    kernel : str, default='bartlett'
        HAC kernel (only used when ``cov_type='hac'``).
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
        Confidence intervals at the specified alpha level.
    rsquared : float
        R-squared.
    nobs : int
        Number of observations.
    df_resid : int
        Residual degrees of freedom.
    """

    def __init__(
        self,
        cov_type: str = "nonrobust",
        alpha: float = 0.05,
        bandwidth: Optional[int] = None,
        kernel: str = "bartlett",
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.cov_type = cov_type.lower()
        self.alpha = alpha
        self.bandwidth = bandwidth
        self.kernel = kernel
        if self.cov_type not in ("nonrobust", "robust", "clustered", "hac"):
            raise ValueError("cov_type must be 'nonrobust', 'robust', 'clustered', or 'hac'")

    def fit(self, X=None, y=None, cluster=None, time_index=None, formula=None, data=None):
        """Fit the pooled OLS model.

        Parameters
        ----------
        X : array-like, shape (n, k), optional
            Design matrix (an intercept is added automatically).
            Required if ``formula`` is None.
        y : array-like, shape (n,), optional
            Dependent variable.  Required if ``formula`` is None.
        cluster : array-like, shape (n,), optional
            Cluster labels (required when ``cov_type='clustered'``).
        time_index : array-like, shape (n,), optional
            Time index for HAC estimation.  Data should be sorted by time.
        formula : str, optional
            R-style formula string (e.g. ``"y ~ x1 + x2"``).
        data : DataFrame, optional
            DataFrame for formula parsing.

        Returns
        -------
        self
        """
        from statgpu.panel._formula import _align_formula_side_array, _prepare_formula_fit
        (y_arr, X_arr, self._design_info, self._feature_names, self._formula_has_intercept,
         _fe_eids, _fe_tids, _fe_entity, _fe_time) = \
            _prepare_formula_fit(formula, data, X, y, model_has_intercept=True)
        if formula is not None:
            cluster = _align_formula_side_array(cluster, self._design_info, len(y_arr), "cluster")
            time_index = _align_formula_side_array(time_index, self._design_info, len(y_arr), "time_index")

        backend = self._get_backend(backend="auto")
        xp = backend.xp

        X_arr = xp_asarray(X_arr, dtype=xp.float64, xp=xp)
        y_arr = xp_asarray(y_arr, dtype=xp.float64, xp=xp, ref_arr=X_arr).ravel()
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        validate_panel_alpha(self.alpha)
        validate_panel_numeric_data(X_arr, y_arr, xp)

        # Add intercept
        n = X_arr.shape[0]
        ones = xp.ones((n, 1), dtype=xp.float64)
        if hasattr(X_arr, 'is_cuda'):
            ones = ones.to(device=X_arr.device)
        X_arr = xp.concatenate([ones, X_arr], axis=1)

        n, k = X_arr.shape

        # OLS: beta = (X'X)^{-1} X'y
        XtX = X_arr.T @ X_arr
        Xty = X_arr.T @ y_arr
        try:
            params = xp.linalg.solve(XtX, Xty)
        except _LINALG_ERRORS:
            params = xp.linalg.pinv(X_arr) @ y_arr

        if n <= k:
            raise ValueError(f"positive residual degrees of freedom required; n={n}, k={k}")
        resid = y_arr - X_arr @ params
        scale = _to_float_scalar(xp.sum(resid * resid)) / (n - k)

        # Inference
        self._compute_inference(X_arr, resid, params, scale, n, k, xp, backend.name,
                                cluster=cluster)

        # R-squared
        y_mean = xp.mean(y_arr)
        ss_tot = _to_float_scalar(xp.sum((y_arr - y_mean) ** 2))
        ss_res = _to_float_scalar(xp.sum(resid * resid))
        self.rsquared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        self.nobs = n
        self.df_resid = n - k
        self._fitted = True

        return self

    def predict(self, X):
        """Predict using the fitted model.

        Parameters
        ----------
        X : array-like, shape (n, k) or DataFrame
            If the model was fitted with a formula, pass a DataFrame.

        Returns
        -------
        y_pred : ndarray, shape (n,)
        """
        self._check_is_fitted()
        from statgpu.panel._formula import _formula_predict
        X_arr = _formula_predict(X, getattr(self, '_design_info', None),
                                 getattr(self, '_formula_has_intercept', None),
                                 model_has_intercept=True)

        backend = self._get_backend(backend="auto")
        xp = backend.xp
        X_arr = xp_asarray(X_arr, dtype=xp.float64, xp=xp)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        ones = xp.ones((X_arr.shape[0], 1), dtype=xp.float64)
        if hasattr(X_arr, 'is_cuda'):
            ones = ones.to(device=X_arr.device)
        X_arr = xp.concatenate([ones, X_arr], axis=1)
        params = xp_asarray(self.coef_, dtype=xp.float64, xp=xp, ref_arr=X_arr)
        return _to_numpy(X_arr @ params)

    def summary(self):
        """Return a summary object."""
        self._check_is_fitted()
        from statgpu.panel._formula import _get_feature_names
        feature_names = _get_feature_names(
            getattr(self, '_feature_names', None),
            len(self.coef_),
            prefix="x"
        )
        return PanelSummary(
            model_type="PooledOLS",
            cov_type=self.cov_type,
            coef=np.asarray(self.coef_),
            bse=np.asarray(self.bse_),
            tvalues=np.asarray(self.tvalues_),
            pvalues=np.asarray(self.pvalues_),
            conf_int=np.asarray(self.conf_int_),
            nobs=self.nobs,
            df_resid=self.df_resid,
            alpha=self.alpha,
            feature_names=feature_names,
        )

    def _compute_inference(self, X, resid, params, scale, n, k, xp, backend_name, cluster=None):
        """Compute standard errors, t-stats, p-values, and CIs."""
        # X'X inverse
        XtX = X.T @ X / n
        try:
            XtX_inv = xp.linalg.inv(XtX)
        except _LINALG_ERRORS:
            XtX_inv = xp.linalg.pinv(XtX)

        if self.cov_type == "nonrobust":
            cov_params = scale * XtX_inv / n
        elif self.cov_type == "robust":
            # HC1: (X'X)^{-1} X' diag(e^2) X (X'X)^{-1} * n/(n-k)
            scores = X * resid[:, None]
            meat = scores.T @ scores
            cov_params = XtX_inv @ meat @ XtX_inv / (n * n) * n / (n - k)
        elif self.cov_type == "clustered":
            if cluster is None:
                raise ValueError("cluster is required for cov_type='clustered'")
            cluster_arr = xp_asarray(cluster, xp=xp, ref_arr=X).ravel()
            cov_params = clustered_covariance(X, resid, cluster_arr, xp)
        elif self.cov_type == "hac":
            cov_params = hac_covariance(X, resid, bandwidth=self.bandwidth,
                                        kernel=self.kernel, xp=xp)

        # SE, t, p, CI
        bse_dev = xp.sqrt(xp.diag(cov_params))
        tvalues_dev = params / bse_dev
        df = n - k

        from statgpu.inference._distributions_backend import get_distribution
        dist_name = "norm" if self.cov_type in ("robust", "clustered", "hac") else "t"
        t_dist = get_distribution(dist_name, backend=backend_name)
        if dist_name == "t":
            pvalues_dev = 2 * t_dist.sf(xp.abs(tvalues_dev), df)
            t_crit = t_dist.isf(self.alpha / 2, df)
        else:
            pvalues_dev = 2 * t_dist.sf(xp.abs(tvalues_dev))
            t_crit = t_dist.isf(self.alpha / 2)

        # Ensure t_crit is on the same device as params (distribution may return CPU scalar).
        t_crit = xp_asarray(t_crit, dtype=params.dtype, xp=xp, ref_arr=params)

        conf_low = params - t_crit * bse_dev
        conf_high = params + t_crit * bse_dev

        self.coef_ = _to_numpy(params)
        self.bse_ = _to_numpy(bse_dev)
        self.tvalues_ = _to_numpy(tvalues_dev)
        self.pvalues_ = _to_numpy(pvalues_dev)
        self.conf_int_ = _to_numpy(xp.stack([conf_low, conf_high], axis=1))

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params["cov_type"] = self.cov_type
        params["alpha"] = self.alpha
        params["bandwidth"] = self.bandwidth
        params["kernel"] = self.kernel
        return params

    def set_params(self, **params):
        for key in ["cov_type", "alpha", "bandwidth", "kernel"]:
            if key in params:
                setattr(self, key, params.pop(key))
        if params:
            super().set_params(**params)
        return self
