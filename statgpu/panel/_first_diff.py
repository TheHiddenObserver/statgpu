"""First-difference OLS estimator for panel data with GPU acceleration."""

from __future__ import annotations

__all__ = ["FirstDifferenceOLS"]

from typing import Optional, Union

import numpy as np

from statgpu._config import Device
from statgpu.backends import _LINALG_ERRORS, _to_float_scalar, _to_numpy, xp_asarray
from statgpu.panel._base import BasePanelModel
from statgpu.panel._utils import factorize_panel_labels


class FirstDifferenceOLS(BasePanelModel):
    """First-difference OLS estimator for panel data.

    Transforms the data by taking first differences within each entity:
    ``Δy_t = y_t - y_{t-1}``, ``ΔX_t = X_t - X_{t-1}``, then runs OLS
    on the differenced data.
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
        """Fit the first-difference OLS model."""
        # Preserve the existing contract: an explicit entity side array is
        # required even when the numerical design is supplied by a formula.
        if entity_ids is None:
            raise ValueError("entity_ids is required for FirstDifferenceOLS")

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
            model_has_intercept=False,
            side_arrays={"entity_ids": entity_ids, "time_ids": time_ids},
        )
        entity_ids = aligned["entity_ids"]
        time_ids = aligned["time_ids"]

        backend, xp, X_arr, y_arr = self._panel_prepare_numeric(X_data, y_data)
        self._panel_set_index_info(
            X_arr.shape[0], entity_ids=entity_ids, time_ids=time_ids
        )
        eids, _entity_labels = factorize_panel_labels(
            entity_ids,
            xp,
            ref_arr=X_arr,
            name="entity_ids",
            expected_n=X_arr.shape[0],
        )

        X_diff, y_diff = _first_diff_transform(
            X_arr, y_arr, eids, time_ids, xp
        )
        n, k = X_diff.shape

        XtX = X_diff.T @ X_diff
        Xty = X_diff.T @ y_diff
        try:
            params = xp.linalg.solve(XtX, Xty)
        except _LINALG_ERRORS:
            params = xp.linalg.pinv(X_diff) @ y_diff

        if n <= k:
            raise ValueError(
                f"positive residual degrees of freedom required; n={n}, k={k}"
            )
        df_resid = n - k
        resid = y_diff - X_diff @ params
        scale = _to_float_scalar(xp.sum(resid * resid)) / df_resid

        self._panel_store_ols_inference(
            X_diff,
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

        y_bar = xp.mean(y_diff)
        ss_tot = _to_float_scalar(xp.sum((y_diff - y_bar) ** 2))
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
            model_has_intercept=False,
            add_intercept=False,
            return_numpy=True,
        )

    def summary(self):
        """Return a summary object."""
        return self._panel_summary(
            model_type="FirstDifferenceOLS",
            cov_type=self._cov_type,
        )

    def get_params(self, deep=True):
        """Return the shared exact-constructor parameter contract."""
        return super().get_params(deep)

    def set_params(self, **params):
        """Delegate parameter updates to the shared estimator contract."""
        return super().set_params(**params)


def _first_diff_transform(X, y, entity_ids, time_ids, xp):
    """Apply first differencing within each entity.

    Sorting remains metadata-only on CPU; numerical X/y stay on the selected
    backend and are reordered there.
    """
    eids_np = _to_numpy(entity_ids).ravel()
    if time_ids is not None:
        tids_np = np.asarray(_to_numpy(time_ids)).ravel()
        if tids_np.shape[0] != eids_np.shape[0]:
            raise ValueError("time_ids must have the same length as entity_ids")
        sort_idx_np = np.lexsort((tids_np, eids_np))
    else:
        sort_idx_np = np.argsort(eids_np, kind="stable")

    sort_idx = xp_asarray(sort_idx_np, dtype=xp.int64, xp=xp, ref_arr=X)
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]
    eids_sorted = entity_ids[sort_idx]

    same_entity = eids_sorted[1:] == eids_sorted[:-1]
    X_diff = (X_sorted[1:] - X_sorted[:-1])[same_entity]
    y_diff = (y_sorted[1:] - y_sorted[:-1])[same_entity]
    if int(X_diff.shape[0]) == 0:
        raise ValueError("No entities with 2+ observations for differencing")
    return X_diff, y_diff
