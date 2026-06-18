"""First-difference OLS estimator for panel data with GPU acceleration."""

from __future__ import annotations

__all__ = ["FirstDifferenceOLS"]

from typing import Optional, Union

import numpy as np

from statgpu._base import BaseEstimator
from statgpu._config import Device
from statgpu.backends import _LINALG_ERRORS, _to_float_scalar, _to_numpy, xp_asarray

from statgpu.panel._utils import PanelSummary
from statgpu.panel._utils import compute_panel_inference as _compute_ols_inference


class FirstDifferenceOLS(BaseEstimator):
    """First-difference OLS estimator for panel data.

    Transforms the data by taking first differences within each entity:
    ``Δy_t = y_t - y_{t-1}``, ``ΔX_t = X_t - X_{t-1}``, then runs OLS
    on the differenced data.

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
        Estimated coefficients (no intercept -- differencing removes it).
    bse_ : ndarray, shape (k,)
        Standard errors.
    tvalues_ : ndarray, shape (k,)
        t-statistics.
    pvalues_ : ndarray, shape (k,)
        Two-sided p-values.
    conf_int_ : ndarray, shape (k, 2)
        Confidence intervals.
    rsquared : float
        R-squared of the differenced regression.
    nobs : int
        Number of observations after differencing.
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
        """Fit the first-difference OLS model.

        Parameters
        ----------
        X : array-like, shape (n, k), optional
            Design matrix (without intercept -- it is removed by differencing).
        y : array-like, shape (n,), optional
            Dependent variable.
        entity_ids : array-like, shape (n,)
            Entity identifiers.
        time_ids : array-like, shape (n,), optional
            Time identifiers.  If None, assumes data is already sorted by
            time within each entity.
        formula : str, optional
            R-style formula string (e.g. ``"y ~ x1 + x2 - 1"``).
        data : DataFrame, optional
            DataFrame for formula parsing.

        Returns
        -------
        self
        """
        if entity_ids is None:
            raise ValueError("entity_ids is required for FirstDifferenceOLS")

        from statgpu.panel._formula import _prepare_formula_fit
        (y_arr, X_arr, self._design_info, self._feature_names, self._formula_has_intercept,
         _fe_eids, _fe_tids, _fe_entity, _fe_time) = \
            _prepare_formula_fit(formula, data, X, y, model_has_intercept=False)

        backend = self._get_backend(backend="auto")
        xp = backend.xp

        X_arr = xp_asarray(X_arr, dtype=xp.float64, xp=xp)
        y_arr = xp_asarray(y_arr, dtype=xp.float64, xp=xp, ref_arr=X_arr).ravel()
        eids = xp_asarray(entity_ids, xp=xp, ref_arr=X_arr).ravel()

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        # First differencing: sort by entity and time, then diff
        X_diff, y_diff = _first_diff_transform(X_arr, y_arr, eids, time_ids, xp)

        n, k = X_diff.shape

        # OLS on differenced data (no intercept)
        XtX = X_diff.T @ X_diff
        Xty = X_diff.T @ y_diff
        try:
            params = xp.linalg.solve(XtX, Xty)
        except _LINALG_ERRORS:
            params = xp.linalg.lstsq(XtX, Xty)[0]

        resid = y_diff - X_diff @ params
        scale = _to_float_scalar(xp.sum(resid * resid)) / (n - k)

        _compute_ols_inference(
            self, X_diff, resid, params, scale, n, k, xp, backend.name,
            self.cov_type, self.alpha, dist_df=n - k
        )

        y_bar = xp.mean(y_diff)
        ss_tot = _to_float_scalar(xp.sum((y_diff - y_bar) ** 2))
        ss_res = _to_float_scalar(xp.sum(resid * resid))
        self.rsquared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        self.nobs = n
        self.df_resid = n - k
        self._fitted = True

        return self

    def predict(self, X):
        """Predict using the fitted model."""
        self._check_is_fitted()
        from statgpu.panel._formula import _formula_predict
        X_arr = _formula_predict(X, getattr(self, '_design_info', None),
                                 getattr(self, '_formula_has_intercept', None),
                                 model_has_intercept=False)
        backend = self._get_backend(backend="auto")
        xp = backend.xp
        X_arr = xp_asarray(X_arr, dtype=xp.float64, xp=xp)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        params = xp_asarray(self.coef_, dtype=xp.float64, xp=xp, ref_arr=X_arr)
        return _to_numpy(X_arr @ params)

    def summary(self):
        """Return a summary object."""
        self._check_is_fitted()
        from statgpu.panel._formula import _get_feature_names
        feature_names = _get_feature_names(
            getattr(self, '_feature_names', None), len(self.coef_), prefix="x"
        )
        return PanelSummary(
            model_type="FirstDifferenceOLS",
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


def _first_diff_transform(X, y, entity_ids, time_ids, xp):
    """Apply first differencing within each entity.

    Returns X_diff, y_diff (differenced data, potentially shorter than input).
    """
    # Work in numpy for indexing
    eids_np = _to_numpy(entity_ids).ravel()
    X_np = _to_numpy(X)
    y_np = _to_numpy(y).ravel()

    if time_ids is not None:
        tids_np = _to_numpy(time_ids).ravel()
        # Sort by entity then time
        sort_idx = np.lexsort((tids_np, eids_np))
    else:
        # Assume already sorted by entity and time
        sort_idx = np.argsort(eids_np, kind='stable')

    X_sorted = X_np[sort_idx]
    y_sorted = y_np[sort_idx]
    eids_sorted = eids_np[sort_idx]

    # First diff within each entity
    X_diff_list = []
    y_diff_list = []
    unique_eids = np.unique(eids_sorted)

    for eid in unique_eids:
        mask = eids_sorted == eid
        X_ent = X_sorted[mask]
        y_ent = y_sorted[mask]
        if X_ent.shape[0] < 2:
            continue
        X_diff_list.append(np.diff(X_ent, axis=0))
        y_diff_list.append(np.diff(y_ent))

    if not X_diff_list:
        raise ValueError("No entities with 2+ observations for differencing")

    X_diff_np = np.vstack(X_diff_list)
    y_diff_np = np.concatenate(y_diff_list)

    return (
        xp.asarray(X_diff_np, dtype=xp.float64),
        xp.asarray(y_diff_np, dtype=xp.float64),
    )
