"""First-difference OLS estimator for panel data with GPU acceleration."""

from __future__ import annotations

__all__ = ["FirstDifferenceOLS"]

from typing import Optional, Union

import numpy as np

from statgpu._config import Device
from statgpu.backends import _LINALG_ERRORS, _to_float_scalar, _to_numpy, xp_asarray
from statgpu.panel._base import BasePanelModel
from statgpu.panel._linalg import panel_lstsq, panel_matrix_rank
from statgpu.panel._utils import factorize_panel_labels, factorize_panel_metadata


class FirstDifferenceOLS(BasePanelModel):
    """First-difference OLS estimator for panel data.

    Transforms the data by taking first differences within each entity and runs
    OLS on the retained differenced sample. Stage C adds transformed-fit-space
    HC0/HC2/HC3 covariance while preserving historical HC1 ``robust`` behavior.
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
        """Fit the first-difference OLS model."""
        self._reset_fit_state()
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

        rank_diff = panel_matrix_rank(X_diff, xp)
        if rank_diff < int(X_diff.shape[1]):
            params, _ = panel_lstsq(X_diff, y_diff, xp)
        else:
            XtX = X_diff.T @ X_diff
            Xty = X_diff.T @ y_diff
            try:
                params = xp.linalg.solve(XtX, Xty)
            except _LINALG_ERRORS:
                params, _ = panel_lstsq(X_diff, y_diff, xp)

        if n <= rank_diff:
            raise ValueError(
                "positive residual degrees of freedom required; "
                f"n={n}, rank={rank_diff}"
            )
        df_resid = n - int(rank_diff)
        resid = y_diff - X_diff @ params
        scale = _to_float_scalar(xp.sum(resid * resid)) / df_resid

        self._panel_store_ols_inference(
            X_diff,
            resid,
            params,
            scale=scale,
            df_resid=df_resid,
            backend=backend,
            fit_rank=rank_diff,
            cov_type=self._cov_type,
            allowed=("nonrobust", "robust", "hc0", "hc2", "hc3"),
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

        from statgpu.panel._diagnostic_context import build_model_fit_statistics

        diagnostic_df = n - rank_diff
        ss_tot_diag = _to_float_scalar(xp.sum(y_diff * y_diff))
        self.fit_statistics_ = build_model_fit_statistics(
            y_arr,
            X_arr,
            params,
            xp=xp,
            entity_codes=eids,
            has_constant=False,
            rss_fit=ss_res,
            tss_fit=ss_tot_diag,
            df_resid=diagnostic_df,
            df_total=n,
            f_y=y_diff,
            f_X=X_diff,
            f_params=params,
            f_has_constant=False,
            metadata={
                "fit_space": "first-difference regression",
                "legacy_df_resid": int(df_resid),
                "diagnostic_df_resid": int(diagnostic_df),
                "diagnostic_rank": int(rank_diff),
                "legacy_rsquared": self.rsquared,
            },
        )

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
        _time_labels, time_codes = factorize_panel_metadata(
            time_ids, name="time_ids", expected_n=eids_np.shape[0]
        )
        pairs = np.column_stack(
            [np.asarray(eids_np, dtype=np.int64), np.asarray(time_codes, dtype=np.int64)]
        )
        if np.unique(pairs, axis=0).shape[0] != pairs.shape[0]:
            raise ValueError(
                "FirstDifferenceOLS requires unique (entity_id, time_id) observations"
            )
        # Differences are taken between consecutive observed times within each
        # entity. Internal calendar gaps are therefore allowed and are not filled.
        sort_idx_np = np.lexsort((time_codes, eids_np))
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