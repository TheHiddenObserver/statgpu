"""Fama-MacBeth two-pass regression for panel data with GPU acceleration."""

from __future__ import annotations

__all__ = ["FamaMacBeth"]

from typing import Optional, Union

import numpy as np

from statgpu._base import BaseEstimator
from statgpu._config import Device
from statgpu.backends import _LINALG_ERRORS, _to_float_scalar, _to_numpy, xp_asarray

from statgpu.panel._utils import PanelSummary
from statgpu.panel._covariance import hac_covariance


class FamaMacBeth(BaseEstimator):
    """Fama-MacBeth two-pass regression estimator.

    Step 1: For each time period, run a cross-sectional OLS regression
    to obtain time-series of coefficient estimates β_t.
    Step 2: Average the β_t and compute standard errors using the
    time-series of β_t (optionally with Newey-West HAC correction).

    Parameters
    ----------
    cov_type : str, default='newey-west'
        Covariance estimator: ``'nonrobust'`` (simple time-series SE)
        or ``'newey-west'`` (HAC).
    bandwidth : int or None, default=None
        Newey-West bandwidth.  If None, uses the Newey-West (1994) rule.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    min_obs_per_period : int, default=1
        Minimum observations per time period to include that period.
    device : str or Device, default='auto'
        Computation device.

    Attributes
    ----------
    coef_ : ndarray, shape (k,)
        Average coefficients across time periods (including intercept).
    bse_ : ndarray, shape (k,)
        Standard errors.
    tvalues_ : ndarray, shape (k,)
        t-statistics.
    pvalues_ : ndarray, shape (k,)
        Two-sided p-values.
    conf_int_ : ndarray, shape (k, 2)
        Confidence intervals.
    betas_ : ndarray, shape (T, k)
        Time-series of coefficient estimates from Step 1.
    nobs : int
        Total number of observations.
    n_periods : int
        Number of time periods used.
    df_resid : int
        Residual degrees of freedom (T - 1).
    """

    def __init__(
        self,
        cov_type: str = "newey-west",
        bandwidth: Optional[int] = None,
        alpha: float = 0.05,
        min_obs_per_period: int = 1,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.cov_type = cov_type.lower()
        self.bandwidth = bandwidth
        self.alpha = alpha
        self.min_obs_per_period = min_obs_per_period
        if self.cov_type not in ("nonrobust", "newey-west"):
            raise ValueError("cov_type must be 'nonrobust' or 'newey-west'")

    def fit(self, X=None, y=None, time_ids=None, formula=None, data=None):
        """Fit the Fama-MacBeth model.

        Parameters
        ----------
        X : array-like, shape (n, k), optional
            Design matrix (an intercept is added automatically).
        y : array-like, shape (n,), optional
            Dependent variable.
        time_ids : array-like, shape (n,)
            Time period identifiers.
        formula : str, optional
            R-style formula string (e.g. ``"y ~ x1 + x2"``).
        data : DataFrame, optional
            DataFrame for formula parsing.

        Returns
        -------
        self
        """
        if time_ids is None:
            raise ValueError("time_ids is required for FamaMacBeth")

        from statgpu.panel._formula import _align_formula_side_array, _prepare_formula_fit
        (y_np, X_np, self._design_info, self._feature_names, self._formula_has_intercept,
         _fe_eids, _fe_tids, _fe_entity, _fe_time) = \
            _prepare_formula_fit(formula, data, X, y, model_has_intercept=True)
        if formula is not None:
            time_ids = _align_formula_side_array(time_ids, self._design_info, len(y_np), "time_ids")

        backend = self._get_backend(backend="auto")
        X_np = np.asarray(_to_numpy(X_np), dtype=np.float64)
        y_np = np.asarray(_to_numpy(y_np), dtype=np.float64).ravel()
        tids_np = np.asarray(_to_numpy(time_ids)).ravel()

        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)

        # Add intercept
        n_orig = X_np.shape[0]
        X_np = np.column_stack([np.ones(n_orig), X_np])
        k = X_np.shape[1]

        # Step 1: Cross-sectional regressions for each time period
        unique_times = np.unique(tids_np)
        betas_list = []

        for t in unique_times:
            mask = tids_np == t
            n_t = mask.sum()
            if n_t < self.min_obs_per_period:
                continue
            if n_t < k + 1:
                continue  # Not enough observations for OLS

            X_t = X_np[mask]
            y_t = y_np[mask]

            # OLS
            try:
                beta_t = np.linalg.solve(X_t.T @ X_t, X_t.T @ y_t)
            except np.linalg.LinAlgError:
                beta_t = np.linalg.pinv(X_t) @ y_t
            betas_list.append(beta_t)

        if not betas_list:
            raise ValueError("No time periods with enough observations")

        betas = np.array(betas_list)  # (T, k)
        T = betas.shape[0]
        if T < 2:
            raise ValueError("FamaMacBeth requires at least 2 time periods after filtering")

        # Step 2: Time-series averages and SEs
        avg_beta = betas.mean(axis=0)

        # Covariance of the time-series mean
        if self.cov_type == "nonrobust":
            # Simple: var(beta_bar) = var(beta_t) / T
            beta_centered = betas - avg_beta
            S = beta_centered.T @ beta_centered / (T - 1)
            cov_params = S / T
        elif self.cov_type == "newey-west":
            # Newey-West on the beta_t time series
            beta_centered = betas - avg_beta  # (T, k)
            bandwidth = self.bandwidth
            if bandwidth is None:
                bandwidth = int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))
            bandwidth = max(0, min(bandwidth, T - 1))

            # Gamma_0
            S = beta_centered.T @ beta_centered / T
            # Gamma_h
            for h in range(1, bandwidth + 1):
                w = 1.0 - h / (bandwidth + 1.0)
                Gamma_h = beta_centered[h:].T @ beta_centered[:T - h] / T
                S = S + w * (Gamma_h + Gamma_h.T)
            cov_params = S / T

        # SE, t, p, CI
        bse = np.sqrt(np.diag(cov_params))
        tvalues = avg_beta / bse
        df = T - 1

        from statgpu.inference._distributions_backend import get_distribution
        dist_name = "norm" if self.cov_type == "newey-west" else "t"
        t_dist = get_distribution(dist_name, backend=backend.name)
        abs_t = np.abs(tvalues)
        if dist_name == "t":
            pvalues = np.asarray([_to_float_scalar(t_dist.sf(float(t), df)) * 2 for t in abs_t])
            t_crit = _to_float_scalar(t_dist.isf(self.alpha / 2, df))
        else:
            pvalues = np.asarray([_to_float_scalar(t_dist.sf(float(t))) * 2 for t in abs_t])
            t_crit = _to_float_scalar(t_dist.isf(self.alpha / 2))

        conf_int = np.column_stack([avg_beta - t_crit * bse, avg_beta + t_crit * bse])

        # Store results
        self.coef_ = avg_beta
        self.bse_ = bse
        self.tvalues_ = tvalues
        self.pvalues_ = pvalues
        self.conf_int_ = conf_int
        self.betas_ = betas
        self.nobs = n_orig
        self.n_periods = T
        self.df_resid = df
        self._fitted = True

        return self

    def predict(self, X):
        """Predict using the fitted model."""
        self._check_is_fitted()
        from statgpu.panel._formula import _formula_predict
        X_np = _formula_predict(X, getattr(self, '_design_info', None),
                                getattr(self, '_formula_has_intercept', None),
                                model_has_intercept=True)
        X_np = np.asarray(X_np, dtype=np.float64)
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)
        X_np = np.column_stack([np.ones(X_np.shape[0]), X_np])
        return X_np @ self.coef_

    def summary(self):
        """Return a summary object."""
        self._check_is_fitted()
        from statgpu.panel._formula import _get_feature_names
        feature_names = _get_feature_names(
            getattr(self, '_feature_names', None), len(self.coef_), prefix="x"
        )
        return PanelSummary(
            model_type="FamaMacBeth",
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
        params["bandwidth"] = self.bandwidth
        params["alpha"] = self.alpha
        params["min_obs_per_period"] = self.min_obs_per_period
        return params

    def set_params(self, **params):
        for key in ["cov_type", "bandwidth", "alpha", "min_obs_per_period"]:
            if key in params:
                setattr(self, key, params.pop(key))
        if params:
            super().set_params(**params)
        return self
