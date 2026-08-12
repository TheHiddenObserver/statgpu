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
from statgpu.panel._linalg import panel_lstsq, panel_matrix_rank
from statgpu.panel._utils import (
    factorize_panel_labels,
    group_means,
    group_sizes,
    within_transform,
)


class RandomEffects(BasePanelModel):
    """Random effects estimator for panel data.

    Swamy-Arora variance components and coefficients are covariance-invariant.
    Stage C adds HC, clustered and Driscoll-Kraay inference on the quasi-demeaned
    GLS fit space.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
        *,
        cov_type: str = "nonrobust",
        bandwidth: Optional[int] = None,
        kernel: str = "bartlett",
        group_debias: bool = False,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        from statgpu.panel._covariance import normalize_covariance_type

        self.alpha = alpha
        # Follow the repository constructor-capture contract used by the other
        # panel estimators: expose the canonical value during __init__ so the
        # wrapper stores it in _cov_type, then let the wrapper restore the exact
        # raw public constructor argument after construction.
        self.cov_type = normalize_covariance_type(cov_type)
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
            "driscoll-kraay",
        }
        if self.cov_type not in allowed:
            raise ValueError(
                "cov_type must be one of 'nonrobust', 'robust', 'hc0', 'hc1', "
                "'hc2', 'hc3', 'clustered', 'driscoll-kraay', 'dk', or 'kernel'"
            )
        if not isinstance(group_debias, (bool, np.bool_)):
            raise ValueError("group_debias must be boolean")
        self.coef_ = None
        self.bse_ = None
        self.tvalues_ = None
        self.pvalues_ = None
        self.conf_int_ = None
        self.theta_ = None
        self.variance_components_ = None
        self.fit_statistics_ = None
        self.nobs = None
        self.df_resid = None
        self._params = None
        self._scale = None
        self._panel_diagnostic_identity = None

    def fit(
        self,
        X=None,
        y=None,
        entity_ids=None,
        time_ids=None,
        formula=None,
        data=None,
        cluster=None,
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
            side_arrays={
                "entity_ids": entity_ids,
                "time_ids": time_ids,
                "cluster": cluster,
            },
        )

        # The shared formula parser strips Patsy's Intercept because most panel
        # estimators either add their own constant or transform it away. Random
        # effects does neither. Restore the formula-generated constant here so
        # standard R-style ``y ~ x | entity`` has its declared intercept, while
        # ``0 +``/``- 1`` remains an explicit no-intercept model.
        if formula is not None and bool(self._formula_has_intercept):
            X_data = np.column_stack(
                [
                    np.ones(len(y_data), dtype=np.float64),
                    np.asarray(X_data, dtype=np.float64),
                ]
            )
            self._feature_names = [
                "Intercept",
                *list(self._feature_names or []),
            ]

        entity_ids = aligned["entity_ids"]
        time_ids = aligned["time_ids"]
        cluster = aligned["cluster"]
        if entity_ids is None and fe_entity_ids is not None:
            entity_ids = fe_entity_ids
        if time_ids is None and fe_time_ids is not None:
            time_ids = fe_time_ids
        if entity_ids is None:
            raise ValueError("entity_ids is required for RandomEffects")
        if self._cov_type == "clustered" and cluster is None:
            raise ValueError("cluster is required when cov_type='clustered'")
        if self._cov_type == "driscoll-kraay" and time_ids is None:
            raise ValueError("time_ids is required for Driscoll-Kraay covariance")
        if bool(self.group_debias) and self._cov_type != "clustered":
            raise ValueError("group_debias=True requires cov_type='clustered'")

        backend, xp, X_arr, y_arr = self._panel_prepare_numeric(X_data, y_data)

        entity_arr, _entity_labels = factorize_panel_labels(
            entity_ids, xp, ref_arr=X_arr, name="entity_ids"
        )
        n, k = X_arr.shape
        self.nobs = n
        if entity_arr.shape[0] != n:
            raise ValueError(
                f"entity_ids has {entity_arr.shape[0]} observations but X has {n} rows"
            )
        # Preserve the Stage-B behavior for otherwise-unused time_ids. Only DK
        # promotes time metadata into a validated model-index contract.
        self._panel_set_index_info(
            n,
            entity_ids=entity_ids,
            time_ids=time_ids if self._cov_type == "driscoll-kraay" else None,
        )

        from statgpu.panel._diagnostic_context import (
            build_diagnostic_identity,
            explicit_constant_column,
        )

        constant_index = explicit_constant_column(X_arr, xp=xp)
        has_constant = constant_index is not None

        self._panel_diagnostic_identity = (
            build_diagnostic_identity(
                X_arr,
                y_arr,
                xp=xp,
                entity_codes=entity_arr,
                feature_names=self._feature_names,
                has_constant=has_constant,
            )
            if self._cov_type == "nonrobust"
            else None
        )

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

        rank_between = panel_matrix_rank(X_bar_unique, xp)
        if rank_between < int(X_bar_unique.shape[1]):
            beta_between, _ = panel_lstsq(X_bar_unique, y_bar_unique, xp)
        else:
            XtX_b = X_bar_unique.T @ X_bar_unique
            Xty_b = X_bar_unique.T @ y_bar_unique
            try:
                beta_between = xp.linalg.solve(XtX_b, Xty_b)
            except _LINALG_ERRORS:
                beta_between, _ = panel_lstsq(X_bar_unique, y_bar_unique, xp)
        resid_between = y_bar_unique - X_bar_unique @ beta_between
        rss_between = float(xp.sum(resid_between ** 2))

        # --- Step 2: Within estimation ---
        y_within = within_transform(y_arr, entity_arr, xp=xp)
        X_within = xp.zeros_like(X_arr)
        for j in range(k):
            X_within[:, j] = within_transform(X_arr[:, j], entity_arr, xp=xp)

        if constant_index is not None:
            slope_indices = np.asarray(
                [j for j in range(k) if j != int(constant_index)],
                dtype=np.int64,
            )
            if slope_indices.size == 0:
                rank_within = 0
                resid_within = y_within
            else:
                slope_idx_dev = xp_asarray(
                    slope_indices,
                    dtype=xp.int64,
                    xp=xp,
                    ref_arr=X_arr,
                )
                X_within_fit = X_within[:, slope_idx_dev]
                rank_within = panel_matrix_rank(X_within_fit, xp)
                if rank_within < int(X_within_fit.shape[1]):
                    beta_within, _ = panel_lstsq(X_within_fit, y_within, xp)
                else:
                    XtX_w = X_within_fit.T @ X_within_fit
                    Xty_w = X_within_fit.T @ y_within
                    beta_within = xp.linalg.pinv(XtX_w) @ Xty_w
                resid_within = y_within - X_within_fit @ beta_within
        else:
            rank_within = panel_matrix_rank(X_within, xp)
            if rank_within < int(X_within.shape[1]):
                beta_within, _ = panel_lstsq(X_within, y_within, xp)
            else:
                XtX_w = X_within.T @ X_within
                Xty_w = X_within.T @ y_within
                try:
                    beta_within = xp.linalg.solve(XtX_w, Xty_w)
                except _LINALG_ERRORS:
                    beta_within, _ = panel_lstsq(X_within, y_within, xp)
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

        # Preserve the historical Swamy-Arora parameterization at full
        # column rank, but count only identified auxiliary-regression directions
        # in the rank-deficient extension. With an explicit level constant, the
        # within regression drops that annihilated column and the entity nuisance
        # rank is N; without a constant it is N-1. These formulas reduce exactly
        # to n-k-(N-1) and N-k on the historical full-rank paths.
        within_effect_df = n_entities if has_constant else n_entities - 1
        df_within = n - int(rank_within) - int(within_effect_df)
        if df_within <= 0:
            raise ValueError(
                "Not enough observations for within df: "
                f"n={n}, rank_within={rank_within}, "
                f"effect_df={within_effect_df}, df_within={df_within}"
            )
        sigma2_e = rss_within / df_within

        df_between = n_entities - int(rank_between)
        if df_between <= 0:
            warnings.warn(
                "Between estimator under-identified: "
                f"n_entities={n_entities} <= rank_between={rank_between}. "
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
        rank_star = panel_matrix_rank(X_star, xp)
        if rank_star < int(X_star.shape[1]):
            beta_gls, _ = panel_lstsq(X_star, y_star, xp)
        else:
            XtX_s = X_star.T @ X_star
            Xty_s = X_star.T @ y_star
            try:
                beta_gls = xp_cholesky_solve(XtX_s, Xty_s, xp)
            except _LINALG_ERRORS:
                try:
                    beta_gls = xp.linalg.solve(XtX_s, Xty_s)
                except _LINALG_ERRORS:
                    beta_gls, _ = panel_lstsq(X_star, y_star, xp)

        resid_gls = y_star - X_star @ beta_gls
        # Full-rank behavior remains n-k. The rank-deficient extension uses
        # identified quasi-demeaned rank so redundant columns cannot change
        # scale, HC1 correction, or Student-t degrees of freedom.
        df_resid = n - int(rank_star)
        if df_resid <= 0:
            raise ValueError(
                "positive residual degrees of freedom required; "
                f"n={n}, rank={rank_star}"
            )
        self.df_resid = df_resid
        self._scale = _to_float_scalar(xp.sum(resid_gls ** 2)) / df_resid

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
            X_star,
            resid_gls,
            beta_gls,
            scale=self._scale,
            df_resid=df_resid,
            backend=backend,
            fit_rank=rank_star,
            cov_type=self._cov_type,
            cluster=cluster_for_cov,
            time_ids=time_ids,
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
                "driscoll-kraay",
            ),
            hc1_correction=n / df_resid if self._cov_type == "robust" else None,
            distribution_df=df_resid,
            diag_floor=0.0,
        )

        from statgpu.panel._diagnostic_context import build_model_fit_statistics

        diagnostic_df_resid = n - rank_star
        ss_res_diag = _to_float_scalar(xp.sum(resid_gls * resid_gls))

        restricted_X = None
        restricted_rank = 0
        if has_constant:
            restricted_X = X_star[:, constant_index : constant_index + 1]
            restricted_rank = panel_matrix_rank(restricted_X, xp)
            if restricted_rank < int(restricted_X.shape[1]):
                restricted_params, _ = panel_lstsq(restricted_X, y_star, xp)
            else:
                restricted_params = xp.linalg.pinv(restricted_X) @ y_star
            restricted_resid = y_star - restricted_X @ restricted_params
            ss_tot_diag = _to_float_scalar(
                xp.sum(restricted_resid * restricted_resid)
            )
        else:
            ss_tot_diag = _to_float_scalar(xp.sum(y_star * y_star))

        self.fit_statistics_ = build_model_fit_statistics(
            y_arr,
            X_arr,
            beta_gls,
            xp=xp,
            entity_codes=entity_arr,
            has_constant=has_constant,
            rss_fit=ss_res_diag,
            tss_fit=ss_tot_diag,
            df_resid=diagnostic_df_resid,
            df_total=n - restricted_rank,
            f_y=y_star,
            f_X=X_star,
            f_params=beta_gls,
            f_has_constant=has_constant,
            f_restricted_X=restricted_X,
            metadata={
                "fit_space": "Swamy-Arora quasi-demeaned GLS regression",
                "legacy_df_resid": int(self.df_resid),
                "diagnostic_df_resid": int(diagnostic_df_resid),
                "diagnostic_rank": int(rank_star),
                "has_explicit_constant": bool(has_constant),
                "constant_column_index": constant_index,
                "restricted_rank": int(restricted_rank),
            },
        )

        self._params = np.asarray(self.coef_).ravel()
        self.coef_ = self._params
        self._fitted = True
        return self

    def hausman_test(self, fixed_effects_model):
        """Compare this RE fit with a matched one-way entity PanelOLS fit."""
        from statgpu.panel._diagnostics import hausman_test

        return hausman_test(fixed_effects_model, self)

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
        """Print and return the structured coefficient summary."""
        k = len(self._params)
        feature_names_override = (
            None
            if self._feature_names is not None
            else [f"x{i + 1}" for i in range(k)]
        )
        return self._panel_summary(
            model_type="RandomEffects",
            cov_type=self._cov_type,
            variance_components=self.variance_components_,
            theta=self.theta_,
            feature_names_override=feature_names_override,
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
