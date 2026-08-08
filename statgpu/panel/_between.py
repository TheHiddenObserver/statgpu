"""Between OLS estimator for panel data with GPU acceleration."""

from __future__ import annotations

__all__ = ["BetweenOLS"]

from typing import Optional, Union

import numpy as np

from statgpu._config import Device
from statgpu.backends import _LINALG_ERRORS, _to_float_scalar, _to_numpy, xp_asarray
from statgpu.panel._base import BasePanelModel
from statgpu.panel._utils import factorize_panel_labels, group_means


class BetweenOLS(BasePanelModel):
    """Between-entity OLS estimator for panel data.

    Collapses the data to group means and runs OLS on the collapsed data.

    Parameters
    ----------
    cov_type : str, default='nonrobust'
        Covariance estimator: ``'nonrobust'`` or ``'robust'`` (HC1).
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
        if self.cov_type not in ("nonrobust", "robust"):
            raise ValueError("cov_type must be 'nonrobust' or 'robust'")

    def fit(self, X=None, y=None, entity_ids=None, time_ids=None, formula=None, data=None):
        """Fit the between OLS model."""
        # Preserve the pre-Stage-A requirement: BetweenOLS always requires an
        # explicit entity_ids side array, including for formula-based fitting.
        if entity_ids is None:
            raise ValueError("entity_ids is required for BetweenOLS")

        (
            y_data,
            X_data,
            _fe_eids,
            _fe_tids,
            _fe_entity,
            _fe_time,
            aligned,
        ) = self._panel_prepare_formula_fit(
            formula,
            data,
            X,
            y,
            model_has_intercept=True,
            side_arrays={"entity_ids": entity_ids},
        )
        entity_ids = aligned["entity_ids"]

        backend, xp, X_arr, y_arr = self._panel_prepare_numeric(X_data, y_data)
        self._panel_set_index_info(X_arr.shape[0], entity_ids=entity_ids)
        eids, unique_eids = factorize_panel_labels(
            entity_ids,
            xp,
            ref_arr=X_arr,
            name="entity_ids",
            expected_n=X_arr.shape[0],
        )

        n_orig = X_arr.shape[0]

        # Add intercept exactly as before.
        ones = xp.ones((n_orig, 1), dtype=xp.float64)
        if hasattr(X_arr, "is_cuda"):
            ones = ones.to(device=X_arr.device)
        X_full = xp.concatenate([ones, X_arr], axis=1)
        k = X_full.shape[1]

        n_groups = len(unique_eids)
        first_idx_np = np.unique(_to_numpy(eids).ravel(), return_index=True)[1]
        first_idx = xp_asarray(first_idx_np, dtype=xp.int64, xp=xp, ref_arr=X_arr)
        y_mean = group_means(y_arr, eids, xp=xp)[first_idx]
        X_mean_aligned = xp.zeros_like(X_full)
        for j in range(k):
            X_mean_aligned[:, j] = group_means(X_full[:, j], eids, xp=xp)
        X_mean = X_mean_aligned[first_idx]

        XtX = X_mean.T @ X_mean
        Xty = X_mean.T @ y_mean
        try:
            params = xp.linalg.solve(XtX, Xty)
        except _LINALG_ERRORS:
            params = xp.linalg.pinv(X_mean) @ y_mean

        resid = y_mean - X_mean @ params
        n = n_groups
        if n <= k:
            raise ValueError(
                f"positive residual degrees of freedom required; groups={n}, parameters={k}"
            )
        df_resid = n - k
        scale = _to_float_scalar(xp.sum(resid * resid)) / df_resid

        self._panel_store_ols_inference(
            X_mean,
            resid,
            params,
            scale=scale,
            df_resid=df_resid,
            backend=backend,
            cov_type=self._cov_type,
            allowed=("nonrobust", "robust"),
            hc1_correction=n / df_resid if self._cov_type == "robust" else None,
            distribution_df=df_resid,
            diag_floor=1e-30,
        )

        y_bar = xp.mean(y_mean)
        ss_tot = _to_float_scalar(xp.sum((y_mean - y_bar) ** 2))
        ss_res = _to_float_scalar(xp.sum(resid * resid))
        self.rsquared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        self.nobs = n
        self.df_resid = df_resid
        self._fitted = True
        return self

    def predict(self, X):
        """Predict using the fitted model."""
        return self._panel_predict_linear(
            X,
            model_has_intercept=True,
            add_intercept=True,
            return_numpy=True,
        )

    def summary(self):
        """Return a summary object."""
        return self._panel_summary(
            model_type="BetweenOLS",
            cov_type=self._cov_type,
        )

    def get_params(self, deep=True):
        """Return the shared exact-constructor parameter contract."""
        return super().get_params(deep)

    def set_params(self, **params):
        """Delegate parameter updates to the shared estimator contract."""
        return super().set_params(**params)
