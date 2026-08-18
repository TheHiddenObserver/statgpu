"""Pooled OLS panel data model with GPU acceleration."""

from __future__ import annotations

__all__ = ["PooledOLS"]

from typing import Optional, Union

import numpy as np

from statgpu._config import Device
from statgpu.backends import _to_float_scalar, _to_numpy, xp_asarray
from statgpu.panel._base import BasePanelModel
from statgpu.panel._linalg import panel_lstsq
from statgpu.panel._utils import factorize_panel_labels, factorize_panel_metadata


def _panel_lstsq(X, y, xp):
    """Use the shared panel SVD rank and working-scale policy."""
    return panel_lstsq(X, y, xp)


class PooledOLS(BasePanelModel):
    """Pooled OLS estimator for panel data.

    Runs OLS on the pooled stacked panel. ``robust`` preserves the historical
    HC1 contract, while Stage C adds HC0/HC2/HC3 and Driscoll-Kraay. ``hac``
    remains the legacy row-order Newey-West covariance.
    """

    def __init__(
        self,
        cov_type: str = "nonrobust",
        alpha: float = 0.05,
        bandwidth: Optional[int] = None,
        kernel: str = "bartlett",
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
        *,
        group_debias: bool = False,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        from statgpu.panel._covariance import normalize_covariance_type

        self.cov_type = normalize_covariance_type(cov_type)
        self.alpha = alpha
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.group_debias = group_debias
        allowed = {
            "nonrobust",
            "robust",
            "hc0",
            "hc2",
            "hc3",
            "clustered",
            "hac",
            "driscoll-kraay",
        }
        if self.cov_type not in allowed:
            raise ValueError(
                "cov_type must be one of 'nonrobust', 'robust', 'hc0', 'hc1', "
                "'hc2', 'hc3', 'clustered', 'hac', 'driscoll-kraay', 'dk', or 'kernel'"
            )
        if not isinstance(group_debias, (bool, np.bool_)):
            raise ValueError("group_debias must be boolean")
        self.fit_statistics_ = None

    def fit(
        self,
        X=None,
        y=None,
        cluster=None,
        time_index=None,
        formula=None,
        data=None,
        entity_ids=None,
    ):
        """Fit the pooled OLS model.

        ``entity_ids`` is optional and does not affect coefficients. When
        supplied it enables Stage-B within/between R² and the one-way panel
        Breusch-Pagan random-effects LM diagnostic. ``time_index`` supplies the
        legacy row-HAC order and the Stage-C Driscoll-Kraay time grouping.
        """
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
            side_arrays={
                "cluster": cluster,
                "time_index": time_index,
                "entity_ids": entity_ids,
            },
        )
        cluster = aligned["cluster"]
        time_index = aligned["time_index"]
        entity_ids = aligned["entity_ids"]
        if formula is not None:
            if not bool(self._formula_has_intercept):
                raise ValueError(
                    "PooledOLS always includes an intercept; explicit no-intercept "
                    "formulas are not supported"
                )
            self._feature_names = [
                "Intercept",
                *list(self._feature_names or []),
            ]

        backend, xp, X_arr, y_arr = self._panel_prepare_numeric(X_data, y_data)
        entity_arr = None
        if entity_ids is not None:
            entity_arr, _ = factorize_panel_labels(
                entity_ids,
                xp,
                ref_arr=X_arr,
                name="entity_ids",
                expected_n=X_arr.shape[0],
            )

        if self._cov_type == "clustered" and cluster is None:
            raise ValueError("cluster is required for cov_type='clustered'")
        if self._cov_type == "driscoll-kraay" and time_index is None:
            raise ValueError("time_index is required for Driscoll-Kraay covariance")
        if bool(self.group_debias) and self._cov_type != "clustered":
            raise ValueError("group_debias=True requires cov_type='clustered'")

        # Preserve the legacy ordered-sequence HAC contract, but derive the
        # chronological ordering through the same metadata factorizer used by
        # the other panel time-indexed paths. In particular, an ordered pandas
        # categorical must follow its declared category order rather than the
        # lexical order of the materialized labels.
        if self._cov_type == "hac" and time_index is not None:
            _time_labels, time_codes = factorize_panel_metadata(
                time_index,
                name="time_index",
                expected_n=X_arr.shape[0],
            )
            order_np = np.argsort(time_codes, kind="stable")
            order = xp_asarray(order_np, dtype=xp.int64, xp=xp, ref_arr=X_arr)
            X_arr = X_arr[order]
            y_arr = y_arr[order]
            if entity_arr is not None:
                entity_arr = entity_arr[order]

        n = X_arr.shape[0]
        ones = xp.ones((n, 1), dtype=xp.float64)
        if hasattr(X_arr, "is_cuda"):
            ones = ones.to(device=X_arr.device)
        X_arr = xp.concatenate([ones, X_arr], axis=1)

        n, _ = X_arr.shape
        params, rank = _panel_lstsq(X_arr, y_arr, xp)
        df_resid = n - rank
        if df_resid <= 0:
            raise ValueError(
                f"positive residual degrees of freedom required; n={n}, rank={rank}"
            )
        resid = y_arr - X_arr @ params
        from statgpu.panel._diagnostics import (
            _restore_squared_scale,
            _scaled_mean,
            _scaled_residual_r2,
            _scaled_residual_variance,
        )

        scale = _scaled_residual_variance(resid, df_resid, xp)

        cluster_for_cov = cluster
        if self._cov_type == "clustered":
            cluster_np = np.asarray(_to_numpy(cluster))
            if cluster_np.ndim not in (1, 2) or cluster_np.shape[0] != n:
                raise ValueError(
                    "cluster must have n_samples rows and one or two cluster dimensions"
                )
            if cluster_np.ndim == 2 and cluster_np.shape[1] not in (1, 2):
                raise ValueError("cluster must contain one or two cluster dimensions")
            cluster_for_cov = cluster_np

        self._panel_store_ols_inference(
            X_arr,
            resid,
            params,
            scale=scale,
            df_resid=df_resid,
            backend=backend,
            fit_rank=rank,
            cov_type=self._cov_type,
            cluster=cluster_for_cov,
            time_ids=time_index,
            bandwidth=self.bandwidth,
            kernel=self.kernel,
            group_debias=bool(self.group_debias),
            extra_df=0,
            allowed=(
                "nonrobust",
                "robust",
                "hc0",
                "hc2",
                "hc3",
                "clustered",
                "hac",
                "driscoll-kraay",
            ),
            hc1_correction=n / df_resid if self._cov_type == "robust" else None,
            distribution_df=df_resid,
            # PooledOLS historically used sqrt(diag(V)) without clipping.
            diag_floor=None,
        )

        y_centered = y_arr - _scaled_mean(y_arr, xp)
        self.rsquared, r2_degenerate = _scaled_residual_r2(
            resid, y_centered, xp
        )
        if r2_degenerate:
            self.rsquared = float("nan")
        resid_scale = xp.max(xp.abs(resid))
        centered_scale = xp.max(xp.abs(y_centered))
        resid_unit = resid / xp.where(
            resid_scale > 0.0, resid_scale, xp.ones_like(resid_scale)
        )
        centered_unit = y_centered / xp.where(
            centered_scale > 0.0, centered_scale, xp.ones_like(centered_scale)
        )
        ss_res = _restore_squared_scale(
            _to_float_scalar(xp.sum(resid_unit * resid_unit)),
            _to_float_scalar(resid_scale),
        )
        ss_tot = _restore_squared_scale(
            _to_float_scalar(xp.sum(centered_unit * centered_unit)),
            _to_float_scalar(centered_scale),
        )
        self.nobs = n
        self.rank_ = rank
        self.df_resid = df_resid

        from statgpu.panel._diagnostic_context import (
            bp_lm_from_residuals,
            build_model_fit_statistics,
        )

        self.fit_statistics_ = build_model_fit_statistics(
            y_arr,
            X_arr,
            params,
            xp=xp,
            entity_codes=entity_arr,
            has_constant=True,
            rss_fit=ss_res,
            tss_fit=ss_tot,
            df_resid=df_resid,
            df_total=n - 1,
            metadata={
                "fit_space": "pooled level regression",
                "legacy_rsquared": self.rsquared,
                "diagnostic_df_resid": int(df_resid),
            },
        )
        self._bp_lm_result = (
            None
            if entity_arr is None
            else bp_lm_from_residuals(resid, entity_arr, xp=xp)
        )
        self._fitted = True
        return self

    def breusch_pagan_lm_test(self):
        """Test pooled OLS against a one-way entity random-effects component."""
        from statgpu.panel._diagnostics import breusch_pagan_lm_test

        return breusch_pagan_lm_test(self)

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
            model_type="PooledOLS",
            cov_type=self._cov_type,
        )

    def get_params(self, deep=True):
        """Return the shared exact-constructor parameter contract."""
        return super().get_params(deep)

    def set_params(self, **params):
        """Delegate parameter updates to the shared estimator contract."""
        return super().set_params(**params)