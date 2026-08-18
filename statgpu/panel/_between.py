"""Between OLS estimator for panel data with GPU acceleration."""

from __future__ import annotations

__all__ = ["BetweenOLS"]

from typing import Optional, Union

import numpy as np

from statgpu._config import Device
from statgpu.backends import _to_float_scalar, _to_numpy, xp_asarray
from statgpu.panel._base import BasePanelModel
from statgpu.panel._linalg import panel_lstsq
from statgpu.panel._utils import factorize_panel_labels, group_means


class BetweenOLS(BasePanelModel):
    """Between-entity OLS estimator for panel data.

    Collapses the data to entity means and runs OLS on the collapsed data.
    An intercept is added automatically. Stage C adds transformed-fit-space
    HC0/HC2/HC3 covariance while preserving the historical HC1 ``robust`` path.

    Parameters
    ----------
    cov_type : {'nonrobust', 'robust', 'hc0', 'hc1', 'hc2', 'hc3'}, default='nonrobust'
        Covariance estimator. ``robust`` and ``hc1`` are the same historical
        HC1 contract; HC2/HC3 use leverage from the entity-mean fit-space design.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    device : str or Device, default='auto'
        Computation device.
    n_jobs : int or None, default=None
        Optional parallelism hint retained by the shared estimator contract.

    Attributes
    ----------
    coef_ : ndarray, shape (k,)
        Estimated coefficients, including the automatically added intercept.
    bse_ : ndarray, shape (k,)
        Standard errors.
    tvalues_ : ndarray, shape (k,)
        Coefficient test statistics.
    pvalues_ : ndarray, shape (k,)
        Coefficient p-values.
    conf_int_ : ndarray, shape (k, 2)
        Confidence intervals.
    rsquared : float
        R-squared of the entity-mean regression.
    nobs : int
        Number of entity-mean observations (groups).
    df_resid : int
        Residual degrees of freedom of the legacy between regression.
    fit_statistics_ : PanelFitStatistics or None
        Standardized Stage-B panel fit statistics populated after ``fit``.
    """

    def __init__(
        self,
        cov_type: str = "nonrobust",
        alpha: float = 0.05,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        from statgpu.panel._covariance import normalize_covariance_type

        self.cov_type = normalize_covariance_type(cov_type)
        self.alpha = alpha
        if self.cov_type not in ("nonrobust", "robust", "hc0", "hc2", "hc3"):
            raise ValueError(
                "cov_type must be one of 'nonrobust', 'robust', 'hc0', 'hc1', 'hc2', or 'hc3'"
            )
        self.fit_statistics_ = None

    def fit(self, X=None, y=None, entity_ids=None, time_ids=None, formula=None, data=None):
        """Fit the between OLS model."""
        self._reset_fit_state()
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
        if formula is not None:
            if not bool(self._formula_has_intercept):
                raise ValueError(
                    "BetweenOLS always includes an intercept; explicit no-intercept "
                    "formulas are not supported"
                )
            self._feature_names = [
                "Intercept",
                *list(self._feature_names or []),
            ]

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

        params, rank_mean = panel_lstsq(X_mean, y_mean, xp)

        resid = y_mean - X_mean @ params
        n = n_groups
        if n <= rank_mean:
            raise ValueError(
                "positive residual degrees of freedom required; "
                f"groups={n}, rank={rank_mean}"
            )
        df_resid = n - int(rank_mean)
        from statgpu.panel._diagnostics import (
            _restore_squared_scale,
            _scaled_mean,
            _scaled_residual_r2,
            _scaled_residual_variance,
            _scaled_unit_values,
        )

        scale = _scaled_residual_variance(resid, df_resid, xp)

        self._panel_store_ols_inference(
            X_mean,
            resid,
            params,
            scale=scale,
            df_resid=df_resid,
            backend=backend,
            fit_rank=rank_mean,
            cov_type=self._cov_type,
            allowed=("nonrobust", "robust", "hc0", "hc2", "hc3"),
            hc1_correction=n / df_resid if self._cov_type == "robust" else None,
            distribution_df=df_resid,
            diag_floor=1e-30,
        )

        y_centered = y_mean - _scaled_mean(y_mean, xp)
        self.rsquared, r2_degenerate = _scaled_residual_r2(
            resid, y_centered, xp
        )
        if r2_degenerate:
            self.rsquared = float("nan")
        resid_scale = xp.max(xp.abs(resid))
        centered_scale = xp.max(xp.abs(y_centered))
        resid_unit = _scaled_unit_values(resid, resid_scale, xp)
        centered_unit = _scaled_unit_values(y_centered, centered_scale, xp)
        ss_res = _restore_squared_scale(
            _to_float_scalar(xp.sum(resid_unit * resid_unit)),
            _to_float_scalar(resid_scale),
        )
        ss_tot = _restore_squared_scale(
            _to_float_scalar(xp.sum(centered_unit * centered_unit)),
            _to_float_scalar(centered_scale),
        )
        self.nobs = n
        self.df_resid = df_resid

        from statgpu.panel._diagnostic_context import build_model_fit_statistics

        diagnostic_df = n - rank_mean
        self.fit_statistics_ = build_model_fit_statistics(
            y_arr,
            X_full,
            params,
            xp=xp,
            entity_codes=eids,
            has_constant=True,
            rss_fit=ss_res,
            tss_fit=ss_tot,
            df_resid=diagnostic_df,
            df_total=n - 1,
            f_y=y_mean,
            f_X=X_mean,
            f_params=params,
            f_has_constant=True,
            metadata={
                "fit_space": "entity-mean between regression",
                "legacy_df_resid": int(df_resid),
                "diagnostic_df_resid": int(diagnostic_df),
                "diagnostic_rank": int(rank_mean),
                "legacy_rsquared": self.rsquared,
            },
        )

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