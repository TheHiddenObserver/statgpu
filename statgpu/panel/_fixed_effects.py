"""
Fixed effects panel data model (PanelOLS).

Implements one-way and two-way fixed effects estimation with support for the
existing non-robust, HC1 robust, and clustered covariance estimators.
"""

from __future__ import annotations

__all__ = ["PanelOLS"]

from typing import Optional, Union

import numpy as np

from statgpu._config import Device
from statgpu.backends import (
    _LINALG_ERRORS,
    _to_float_scalar,
    _to_numpy,
    xp_cholesky_solve,
    xp_maximum,
)
from statgpu.panel._base import BasePanelModel
from statgpu.panel._utils import (
    _scatter_add,
    demean_variables,
    factorize_panel_labels,
)


class PanelOLS(BasePanelModel):
    """Fixed effects estimator for panel data."""

    def __init__(
        self,
        entity_effects: bool = False,
        time_effects: bool = False,
        cov_type: str = "nonrobust",
        alpha: float = 0.05,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.entity_effects = entity_effects
        self.time_effects = time_effects
        self.cov_type = cov_type.lower()
        self.alpha = alpha
        if self.cov_type not in ("nonrobust", "robust", "clustered"):
            raise ValueError(
                "cov_type must be 'nonrobust', 'robust', or 'clustered'"
            )

        self.coef_ = None
        self.bse_ = None
        self.tvalues_ = None
        self.pvalues_ = None
        self.conf_int_ = None
        self.rsquared_within = None
        self.nobs = None
        self.df_resid = None

        self._params = None
        self._scale = None
        self._entity_effects_map = {}
        self._time_effects_map = {}

    def fit(
        self,
        X=None,
        y=None,
        entity_ids=None,
        time_ids=None,
        cluster=None,
        formula=None,
        data=None,
    ):
        """Fit the fixed effects model."""
        (
            y_data,
            X_data,
            fe_entity_ids,
            fe_time_ids,
            fe_entity_effects,
            fe_time_effects,
            aligned,
        ) = self._panel_prepare_formula_fit(
            formula,
            data,
            X,
            y,
            model_has_intercept=False,
            support_pipe=True,
            side_arrays={
                "entity_ids": entity_ids,
                "time_ids": time_ids,
                "cluster": cluster,
            },
        )

        # Preserve the existing formula contract: formula-extracted effects
        # enable constructor flags, and formula identifiers fill only missing
        # explicit side arrays.
        if fe_entity_effects:
            self.entity_effects = True
        if fe_time_effects:
            self.time_effects = True
        entity_ids = aligned["entity_ids"]
        time_ids = aligned["time_ids"]
        cluster = aligned["cluster"]
        if entity_ids is None and fe_entity_ids is not None:
            entity_ids = fe_entity_ids
        if time_ids is None and fe_time_ids is not None:
            time_ids = fe_time_ids

        backend, xp, X_arr, y_arr = self._panel_prepare_numeric(X_data, y_data)
        n, k = X_arr.shape
        self.nobs = n

        if self.entity_effects and entity_ids is None:
            raise ValueError("entity_ids is required when entity_effects=True")
        if self.time_effects and time_ids is None:
            raise ValueError("time_ids is required when time_effects=True")
        if self._cov_type == "clustered" and cluster is None:
            raise ValueError("cluster is required when cov_type='clustered'")

        self._panel_set_index_info(
            n, entity_ids=entity_ids, time_ids=time_ids
        )

        entity_arr = None
        time_arr = None
        entity_labels = None
        time_labels = None
        if entity_ids is not None:
            entity_arr, entity_labels = factorize_panel_labels(
                entity_ids,
                xp,
                ref_arr=X_arr,
                name="entity_ids",
                expected_n=n,
            )
        if time_ids is not None:
            time_arr, time_labels = factorize_panel_labels(
                time_ids,
                xp,
                ref_arr=X_arr,
                name="time_ids",
                expected_n=n,
            )

        if self.entity_effects or self.time_effects:
            y_d, X_d = demean_variables(
                y_arr,
                X_arr,
                entity_ids=entity_arr if self.entity_effects else None,
                time_ids=time_arr if self.time_effects else None,
                xp=xp,
            )
        else:
            y_d = y_arr
            X_d = X_arr

        XtX = X_d.T @ X_d
        Xty = X_d.T @ y_d
        try:
            coef = xp_cholesky_solve(XtX, Xty, xp)
        except _LINALG_ERRORS:
            coef = xp.linalg.solve(XtX, Xty)

        n_entities = len(entity_labels) if entity_labels is not None else 0
        n_times = len(time_labels) if time_labels is not None else 0
        n_effects = 0
        if self.entity_effects:
            n_effects += n_entities - 1
        if self.time_effects:
            n_effects += n_times - 1
        self.df_resid = n - k - n_effects
        if self.df_resid <= 0:
            raise ValueError(
                f"Not enough observations: n={n}, k={k}, n_effects={n_effects}, "
                f"df_resid={self.df_resid}.  Check that N*T >> k + effects."
            )

        y_pred = X_d @ coef
        resid = y_d - y_pred
        scale = _to_float_scalar(xp.sum(resid ** 2)) / self.df_resid
        self._scale = scale

        # Effect recovery remains model-specific and byte-for-byte equivalent in
        # meaning to the pre-Stage-A implementation. The stored grand mean is
        # intentionally not added by predict(); that historical behavior is
        # frozen by test_panel_stage_a_golden.py.
        self._entity_effects_map = {}
        self._time_effects_map = {}
        resid_orig = y_arr - X_arr @ coef
        grand_mean = float(xp.mean(resid_orig))
        resid_centered = resid_orig - grand_mean
        self._grand_mean = grand_mean

        if self.entity_effects and entity_arr is not None:
            ent_sums = _scatter_add(
                xp, entity_arr, resid_centered, len(entity_labels)
            )
            ent_counts = _scatter_add(
                xp,
                entity_arr,
                xp.ones_like(resid_centered),
                len(entity_labels),
            )
            ent_effects = _to_numpy(
                ent_sums / xp_maximum(ent_counts, 1.0, xp)
            ).ravel()
            for i, eid in enumerate(entity_labels):
                self._entity_effects_map[eid] = float(ent_effects[i])

        if self.time_effects and time_arr is not None:
            time_sums = _scatter_add(
                xp, time_arr, resid_centered, len(time_labels)
            )
            time_counts = _scatter_add(
                xp,
                time_arr,
                xp.ones_like(resid_centered),
                len(time_labels),
            )
            time_effect_values = _to_numpy(
                time_sums / xp_maximum(time_counts, 1.0, xp)
            ).ravel()
            for i, tid in enumerate(time_labels):
                self._time_effects_map[tid] = float(time_effect_values[i])

        cluster_for_cov = cluster
        if self._cov_type == "clustered":
            cluster_np = np.asarray(_to_numpy(cluster))
            if len(cluster_np) != X_d.shape[0]:
                raise ValueError(
                    f"cluster length ({len(cluster_np)}) does not match "
                    f"data length ({X_d.shape[0]})"
                )
            cluster_for_cov = cluster_np

        self._panel_store_ols_inference(
            X_d,
            resid,
            coef,
            scale=scale,
            df_resid=self.df_resid,
            backend=backend,
            cov_type=self._cov_type,
            cluster=cluster_for_cov,
            allowed=("nonrobust", "robust", "clustered"),
            hc1_correction=(
                n / self.df_resid if self._cov_type == "robust" else None
            ),
            distribution_df=self.df_resid,
            diag_floor=0.0,
        )

        ss_res = _to_float_scalar(xp.sum(resid ** 2))
        y_d_mean = _to_float_scalar(xp.mean(y_d))
        ss_tot = _to_float_scalar(xp.sum((y_d - y_d_mean) ** 2))
        self.rsquared_within = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        self._params = np.asarray(self.coef_).ravel()
        self.coef_ = self._params
        self._fitted = True
        return self

    def predict(self, X, entity_ids=None, time_ids=None):
        """Predict using the fitted model, preserving existing effect semantics."""
        self._check_is_fitted()
        if getattr(self, "_design_info", None) is not None and hasattr(X, "columns"):
            from statgpu.panel._formula import _formula_predict

            X_arr = _formula_predict(
                X,
                self._design_info,
                self._formula_has_intercept,
                model_has_intercept=False,
            )
        else:
            X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if X_arr.shape[1] + 1 == self.coef_.shape[0]:
            X_arr = np.column_stack([np.ones(X_arr.shape[0]), X_arr])
        y_pred = X_arr @ self.coef_

        if self._entity_effects_map and entity_ids is not None:
            ent_arr = np.asarray(entity_ids).ravel()
            ent_effects = np.vectorize(
                self._entity_effects_map.get, otypes=[np.float64]
            )(ent_arr, 0.0)
            y_pred = y_pred + ent_effects

        if self._time_effects_map and time_ids is not None:
            time_arr = np.asarray(time_ids).ravel()
            time_effects = np.vectorize(
                self._time_effects_map.get, otypes=[np.float64]
            )(time_arr, 0.0)
            y_pred = y_pred + time_effects

        return y_pred

    def summary(self):
        """Print and return the existing structured coefficient summary."""
        k = len(self._params)
        return self._panel_summary(
            model_type="PanelOLS",
            cov_type=self._cov_type,
            rsquared_within=self.rsquared_within,
            entity_effects=self.entity_effects,
            time_effects=self.time_effects,
            feature_names_override=[f"x{i + 1}" for i in range(k)],
            print_result=True,
        )

    def get_params(self, deep=True):
        """Return the shared exact-constructor parameter contract."""
        return super().get_params(deep)

    def set_params(self, **params):
        """Delegate parameter updates to the shared estimator contract."""
        return super().set_params(**params)


FixedEffects = PanelOLS
