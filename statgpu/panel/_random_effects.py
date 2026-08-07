"""Random effects panel data model using the Swamy-Arora feasible GLS path."""
from __future__ import annotations

__all__ = ["RandomEffects"]

import warnings
from typing import Optional, Union

import numpy as np

from statgpu._config import Device
from statgpu.backends import (
    _LINALG_ERRORS,
    _to_float_scalar,
    _to_numpy,
    xp_asarray,
    xp_cholesky_solve,
    xp_zeros,
)
from statgpu.panel._base import BasePanelModel
from statgpu.panel._utils import (
    factorize_panel_labels,
    group_means,
    group_sizes,
    within_transform,
)


class RandomEffects(BasePanelModel):
    """Random effects estimator for panel data.

    The Swamy-Arora variance-component and quasi-demeaning calculations remain
    model-specific; Stage A only shares neutral lifecycle and nonrobust
    inference infrastructure.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.alpha = alpha
        self.coef_ = None
        self.bse_ = None
        self.tvalues_ = None
        self.pvalues_ = None
        self.conf_int_ = None
        self.theta_ = None
        self.variance_components_ = None
        self.nobs = None
        self.df_resid = None
        self._params = None
        self._scale = None

    def fit(
        self,
        X=None,
        y=None,
        entity_ids=None,
        time_ids=None,
        formula=None,
        data=None,
    ):
        """Fit the random effects model."""
        (
            y_data,
            X_data,
            fe_entity_ids,
            fe_time_ids,
            _fe_entity,
            _fe_time,
            aligned,
        ) = self._panel_prepare_formula_fit(
            formula,
            data,
            X,
            y,
            model_has_intercept=False,
            support_pipe=True,
            side_arrays={"entity_ids": entity_ids, "time_ids": time_ids},
        )
        entity_ids = aligned["entity_ids"]
        time_ids = aligned["time_ids"]
        if entity_ids is None and fe_entity_ids is not None:
            entity_ids = fe_entity_ids
        if time_ids is None and fe_time_ids is not None:
            time_ids = fe_time_ids
        if entity_ids is None:
            raise ValueError("entity_ids is required for RandomEffects")

        backend, xp, X_arr, y_arr = self._panel_prepare_numeric(X_data, y_data)
        self._backend_name = backend.name

        # Preserve the old validation order/message for entity_ids rather than
        # letting the new metadata substrate introduce a different rejection.
        entity_arr, _entity_labels = factorize_panel_labels(
            entity_ids, xp, ref_arr=X_arr, name="entity_ids"
        )
        n, k = X_arr.shape
        self.nobs = n
        if entity_arr.shape[0] != n:
            raise ValueError(
                f"entity_ids has {entity_arr.shape[0]} observations but X has {n} rows"
            )
        # time_ids is currently reserved/unused by RandomEffects; do not add a
        # new array-interface validation rule for it in this refactor.
        self._panel_set_index_info(n, entity_ids=entity_ids)

        # --- Step 1: Between estimation ---
        y_bar_i = group_means(y_arr, entity_arr, xp=xp)
        X_bar_i = xp.zeros_like(X_arr)
        for j in range(k):
            X_bar_i[:, j] = group_means(X_arr[:, j], entity_arr, xp=xp)

        entity_np = _to_numpy(entity_arr).ravel()
        unique_entities, first_idx = np.unique(entity_np, return_index=True)
        n_groups = len(unique_entities)
        first_idx_dev = xp_asarray(
            first_idx, dtype=xp.int64, xp=xp, ref_arr=X_arr
        )
        y_bar_unique = y_bar_i[first_idx_dev]
        X_bar_unique = X_bar_i[first_idx_dev]

        XtX_b = X_bar_unique.T @ X_bar_unique
        Xty_b = X_bar_unique.T @ y_bar_unique
        try:
            beta_between = xp.linalg.solve(XtX_b, Xty_b)
        except _LINALG_ERRORS:
            beta_between = xp.linalg.pinv(XtX_b) @ Xty_b
        resid_between = y_bar_unique - X_bar_unique @ beta_between
        rss_between = float(xp.sum(resid_between ** 2))

        # --- Step 2: Within estimation ---
        y_within = within_transform(y_arr, entity_arr, xp=xp)
        X_within = xp.zeros_like(X_arr)
        for j in range(k):
            X_within[:, j] = within_transform(X_arr[:, j], entity_arr, xp=xp)

        XtX_w = X_within.T @ X_within
        Xty_w = X_within.T @ y_within
        try:
            beta_within = xp.linalg.solve(XtX_w, Xty_w)
        except _LINALG_ERRORS:
            beta_within = xp.linalg.pinv(XtX_w) @ Xty_w
        resid_within = y_within - X_within @ beta_within
        rss_within = float(xp.sum(resid_within ** 2))

        # --- Step 3: Swamy-Arora variance components ---
        n_entities = len(xp.unique(entity_arr))
        T_i = group_sizes(entity_arr, xp=xp)
        T_i_np = _to_numpy(T_i)
        entity_np = _to_numpy(entity_arr).ravel()
        _, first_idx = np.unique(entity_np, return_index=True)
        per_entity_sizes = T_i_np[first_idx]
        T_bar = float(n_entities) / float(np.sum(1.0 / per_entity_sizes))

        df_within = n - k - (n_entities - 1)
        if df_within <= 0:
            raise ValueError(
                f"Not enough observations for within df: n={n}, k={k}, "
                f"n_entities={n_entities}, df_within={df_within}"
            )
        sigma2_e = rss_within / df_within

        df_between = n_entities - k
        if df_between <= 0:
            warnings.warn(
                f"Between estimator under-identified: n_entities={n_entities} <= k={k}. "
                "Variance component sigma2_a may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
            df_between = max(df_between, 1)
        s_b_sq = rss_between / df_between
        sigma2_a_raw = (s_b_sq - sigma2_e) / T_bar
        sigma2_a = max(0.0, sigma2_a_raw)
        self.variance_components_ = {
            "sigma2_e": sigma2_e,
            "sigma2_a": sigma2_a,
        }

        # --- Step 4: GLS transformation ---
        T_i_unique = np.unique(T_i_np)
        theta_map = {}
        for Ti in T_i_unique:
            denom = sigma2_e + Ti * sigma2_a
            if denom > 0:
                theta_map[Ti] = 1.0 - np.sqrt(sigma2_e / denom)
            else:
                theta_map[Ti] = 0.0

        theta_arr = xp_zeros(n, xp.float64, xp, X_arr)
        for Ti, th in theta_map.items():
            theta_arr[T_i == Ti] = th

        entity_counts = {}
        for Ti in T_i_unique:
            entity_counts[Ti] = int(np.sum(T_i_np[first_idx] == Ti))
        total_entities = sum(entity_counts.values())
        self.theta_ = sum(
            theta_map[Ti] * entity_counts[Ti] / total_entities
            for Ti in T_i_unique
        )

        y_star = y_arr - theta_arr * y_bar_i
        X_star = xp.zeros_like(X_arr)
        for j in range(k):
            X_star[:, j] = X_arr[:, j] - theta_arr * X_bar_i[:, j]

        # --- Step 5: OLS on transformed data ---
        XtX_s = X_star.T @ X_star
        Xty_s = X_star.T @ y_star
        try:
            beta_gls = xp_cholesky_solve(XtX_s, Xty_s, xp)
        except _LINALG_ERRORS:
            beta_gls = xp.linalg.solve(XtX_s, Xty_s)

        resid_gls = y_star - X_star @ beta_gls
        df_resid = n - k
        self.df_resid = df_resid
        self._scale = _to_float_scalar(xp.sum(resid_gls ** 2)) / df_resid

        # Existing RandomEffects inference is nonrobust OLS inference on the
        # quasi-demeaned design. Reuse exactly that residual-sandwich context;
        # robust RE covariance remains Stage C.
        self._panel_store_ols_inference(
            X_star,
            resid_gls,
            beta_gls,
            scale=self._scale,
            df_resid=df_resid,
            backend=backend,
            cov_type="nonrobust",
            allowed=("nonrobust",),
            distribution_df=df_resid,
            diag_floor=0.0,
        )

        self._params = np.asarray(self.coef_).ravel()
        self.coef_ = self._params
        self._fitted = True
        return self

    def predict(self, X):
        """Predict using the fitted model, preserving current NumPy output."""
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
        return X_arr @ self.coef_

    def summary(self):
        """Print and return the existing structured coefficient summary."""
        k = len(self._params)
        return self._panel_summary(
            model_type="RandomEffects",
            variance_components=self.variance_components_,
            theta=self.theta_,
            feature_names_override=[f"x{i + 1}" for i in range(k)],
            print_result=True,
        )

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        params = super().get_params(deep)
        params.update({"alpha": self.alpha})
        return params

    def set_params(self, **params):
        """Set parameters for this estimator."""
        if "alpha" in params:
            self.alpha = params.pop("alpha")
        super().set_params(**params)
        return self


RandomEffectsOLS = RandomEffects
