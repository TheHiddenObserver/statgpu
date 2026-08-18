"""
Fixed effects panel data model (PanelOLS).

Implements one-way and two-way fixed effects estimation with Stage-C covariance
support while preserving the historical nonrobust, HC1 robust, and clustered
contracts.
"""

from __future__ import annotations

__all__ = ["PanelOLS"]

from typing import Optional, Union

import numpy as np

from statgpu._config import Device
from statgpu.backends import (
    _to_float_scalar,
    _to_numpy,
    xp_maximum,
    xp_asarray,
)
from statgpu.panel._base import BasePanelModel
from statgpu.panel._linalg import panel_lstsq
from statgpu.panel._utils import (
    _recover_two_way_effects,
    _scatter_add,
    demean_variables,
    factorize_panel_labels,
)


def _two_way_component_maps(entity_codes, time_codes, entity_labels, time_labels):
    """Return fitted entity/time incidence-component maps for prediction guards."""
    entity = np.asarray(_to_numpy(entity_codes), dtype=np.int64).ravel()
    time = np.asarray(_to_numpy(time_codes), dtype=np.int64).ravel()
    if entity.shape != time.shape:
        raise ValueError("entity/time codes must have matching observation counts")
    n_entities = int(len(entity_labels))
    n_times = int(len(time_labels))
    total = n_entities + n_times
    parent = np.arange(total, dtype=np.int64)
    rank = np.zeros(total, dtype=np.int8)

    def find(node):
        node = int(node)
        root = node
        while int(parent[root]) != root:
            root = int(parent[root])
        while int(parent[node]) != node:
            nxt = int(parent[node])
            parent[node] = root
            node = nxt
        return root

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        if rank[left_root] == rank[right_root]:
            rank[left_root] += 1

    for entity_code, time_code in zip(entity, time):
        union(int(entity_code), n_entities + int(time_code))

    root_to_component = {}

    def component(node):
        root = find(node)
        if root not in root_to_component:
            root_to_component[root] = len(root_to_component)
        return int(root_to_component[root])

    entity_map = {
        label: component(index)
        for index, label in enumerate(entity_labels)
    }
    time_map = {
        label: component(n_entities + index)
        for index, label in enumerate(time_labels)
    }
    return entity_map, time_map, len(root_to_component)


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
        *,
        bandwidth: Optional[int] = None,
        kernel: str = "bartlett",
        group_debias: bool = False,
        demean_max_iter: int = 1_000_000,
        demean_tol: float = 1e-10,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        from statgpu.panel._covariance import normalize_covariance_type

        self.entity_effects = entity_effects
        self.time_effects = time_effects
        self.cov_type = normalize_covariance_type(cov_type)
        self.alpha = alpha
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.group_debias = group_debias
        if isinstance(demean_max_iter, (bool, np.bool_)) or not isinstance(
            demean_max_iter, (int, np.integer)
        ) or int(demean_max_iter) <= 0:
            raise ValueError("demean_max_iter must be a positive integer")
        if not np.isfinite(float(demean_tol)) or float(demean_tol) <= 0.0:
            raise ValueError("demean_tol must be finite and positive")
        self.demean_max_iter = int(demean_max_iter)
        self.demean_tol = float(demean_tol)
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
        self.rsquared_within = None
        self.fit_statistics_ = None
        self.nobs = None
        self.df_resid = None

        self._params = None
        self._scale = None
        self._entity_effects_map = {}
        self._time_effects_map = {}
        self._two_way_entity_components = {}
        self._two_way_time_components = {}
        self._pooling_f_result = None
        self._panel_diagnostic_identity = None
        self._predict_constant_index = None
        self._predict_constant_value = None

    def _reset_fit_state(self):
        """Clear derived fit state so failed refits cannot expose stale results."""
        configured = getattr(self, "_constructor_params_raw", {})
        self.entity_effects = bool(
            configured.get("entity_effects", self.entity_effects)
        )
        self.time_effects = bool(
            configured.get("time_effects", self.time_effects)
        )
        self._fitted = False
        for name in (
            "coef_",
            "bse_",
            "tvalues_",
            "pvalues_",
            "conf_int_",
            "rsquared_within",
            "fit_statistics_",
            "nobs",
            "df_resid",
            "_params",
            "_scale",
            "_pooling_f_result",
            "_panel_diagnostic_identity",
            "_predict_constant_index",
            "_predict_constant_value",
            "_grand_mean",
            "_design_info",
            "_feature_names",
            "_formula_has_intercept",
            "_backend_name",
            "_inference_backend_name",
            "_predict_backend_name",
            "_panel_index_info",
            "_panel_cov_params_raw",
            "_coefficient_inference_reason",
            "_inference_result",
            "_bse",
            "_tvalues",
            "_zvalues",
            "_pvalues",
            "_conf_int",
        ):
            setattr(self, name, None)
        self._coefficient_inference_available = False
        self._covariance_metadata = {}
        self._entity_effects_map = {}
        self._time_effects_map = {}
        self._two_way_entity_components = {}
        self._two_way_time_components = {}

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
        """Fit the fixed effects model transactionally."""
        self._reset_fit_state()
        try:
            return self._fit_impl(
                X=X,
                y=y,
                entity_ids=entity_ids,
                time_ids=time_ids,
                cluster=cluster,
                formula=formula,
                data=data,
            )
        except BaseException:
            # A failed fit must not expose partially written state from
            # either the previous result or the failed new request.
            self._reset_fit_state()
            raise

    def _fit_impl(
        self,
        X=None,
        y=None,
        entity_ids=None,
        time_ids=None,
        cluster=None,
        formula=None,
        data=None,
    ):
        """Internal implementation; :meth:`fit` owns state cleanup."""
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

        # Formula effects are fit-time requests, not persistent constructor
        # mutations.  Rebuild the effective FE configuration from the captured
        # constructor parameters on every fit so a formula-enabled FE fit cannot
        # leak into a later array/no-FE refit of the same estimator.
        configured = getattr(self, "_constructor_params_raw", {})
        configured_entity_effects = bool(
            configured.get("entity_effects", self.entity_effects)
        )
        configured_time_effects = bool(
            configured.get("time_effects", self.time_effects)
        )
        self.entity_effects = configured_entity_effects or bool(fe_entity_effects)
        self.time_effects = configured_time_effects or bool(fe_time_effects)

        # Patsy/R formulas include an intercept by default.  Fixed-effect fits
        # absorb that common intercept into the nuisance-effect space, but a
        # no-FE PanelOLS fit is an ordinary level regression and must retain it.
        if (
            formula is not None
            and bool(self._formula_has_intercept)
            and not (self.entity_effects or self.time_effects)
        ):
            X_data = np.column_stack(
                [np.ones(len(y_data), dtype=np.float64), np.asarray(X_data)]
            )
            self._feature_names = ["Intercept", *list(self._feature_names or [])]

        if (
            formula is not None
            and (self.entity_effects or self.time_effects)
            and int(np.asarray(X_data).shape[1]) == 0
        ):
            raise ValueError(
                "PanelOLS fixed-effect formulas require at least one "
                "non-intercept regressor; effects-only formulas are not supported"
            )

        entity_ids = aligned["entity_ids"]
        time_ids = aligned["time_ids"]
        cluster = aligned["cluster"]
        entity_ids, time_ids, _pipe_vars = self._panel_resolve_formula_ids(
            formula,
            entity_ids,
            time_ids,
            fe_entity_ids,
            fe_time_ids,
        )

        backend, xp, X_arr, y_arr = self._panel_prepare_numeric(X_data, y_data)
        n, k = X_arr.shape
        self.nobs = n

        if self.entity_effects and entity_ids is None:
            raise ValueError("entity_ids is required when entity_effects=True")
        if self.time_effects and time_ids is None:
            raise ValueError("time_ids is required when time_effects=True")
        if self._cov_type == "clustered" and cluster is None:
            raise ValueError("cluster is required when cov_type='clustered'")
        if self._cov_type == "driscoll-kraay" and time_ids is None:
            raise ValueError("time_ids is required for Driscoll-Kraay covariance")
        if bool(self.group_debias) and self._cov_type != "clustered":
            raise ValueError("group_debias=True requires cov_type='clustered'")

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
                max_iter=self.demean_max_iter,
                tol=self.demean_tol,
            )
        else:
            y_d = y_arr
            X_d = X_arr

        # Solve and rank from the same SVD policy. Full numerical rank does
        # not certify normal equations: forming X'X squares the condition number
        # and can silently corrupt coefficients well before the SVD rank cutoff.
        coef, fit_rank = panel_lstsq(X_d, y_d, xp)

        n_entities = len(entity_labels) if entity_labels is not None else 0
        n_times = len(time_labels) if time_labels is not None else 0

        from statgpu.panel._diagnostic_context import (
            build_diagnostic_identity,
            build_model_fit_statistics,
            explicit_constant_column,
            fixed_effect_diagnostic_df,
            pooling_f_from_level_arrays,
        )

        constant_index = explicit_constant_column(X_arr, xp=xp)
        self._predict_constant_index = constant_index
        self._predict_constant_value = (
            None
            if constant_index is None
            else _to_float_scalar(X_arr[0, int(constant_index)])
        )
        # A level constant is identified only when no FE space absorbs it.
        # FE-transformed constants are rank-zero columns and continue to use the
        # historical/standard nuisance-rank conventions below.
        has_level_constant = (
            constant_index is not None
            and not self.entity_effects
            and not self.time_effects
        )

        diagnostic_df = fixed_effect_diagnostic_df(
            X_d,
            xp=xp,
            nobs=n,
            n_entities=n_entities,
            n_times=n_times,
            entity_effects=self.entity_effects,
            time_effects=self.time_effects,
            has_constant=has_level_constant,
            entity_codes=entity_arr,
            time_codes=time_arr,
            rank_x=fit_rank,
        )

        self._two_way_entity_components = {}
        self._two_way_time_components = {}
        if (
            self.entity_effects
            and self.time_effects
            and entity_arr is not None
            and time_arr is not None
        ):
            (
                self._two_way_entity_components,
                self._two_way_time_components,
                component_count,
            ) = _two_way_component_maps(
                entity_arr,
                time_arr,
                entity_labels,
                time_labels,
            )
            if component_count != int(diagnostic_df["incidence_components"]):
                raise RuntimeError(
                    "two-way incidence component metadata disagrees with diagnostic rank"
                )

        n_effects = 0
        if self.entity_effects:
            n_effects += n_entities - 1
        if self.time_effects:
            n_effects += n_times - 1
        legacy_df_resid = n - int(fit_rank) - n_effects
        standard_df_resid = int(diagnostic_df["df_resid"])
        if standard_df_resid <= 0:
            raise ValueError(
                "Not enough observations after fixed-effect rank adjustment: "
                f"n={n}, k={k}, legacy_n_effects={n_effects}, "
                f"legacy_df_resid={legacy_df_resid}, "
                f"effect_rank={diagnostic_df['effect_rank']}, "
                f"incidence_components={diagnostic_df['incidence_components']}, "
                f"df_resid={standard_df_resid}."
            )
        # Public inference must count the full identified nuisance-effect rank.
        # The older N-1/T-1 shortcut omitted one nuisance direction whenever FE
        # were represented by within transformation without an explicit level
        # constant, understating nonrobust scale and the HC1 correction.
        self.df_resid = standard_df_resid
        public_df_basis = "standard"

        y_pred = X_d @ coef
        resid = y_d - y_pred
        from statgpu.panel._diagnostics import (
            _restore_squared_scale,
            _scaled_mean,
            _scaled_residual_r2,
            _scaled_residual_variance,
        )

        scale = _scaled_residual_variance(resid, self.df_resid, xp)
        self._scale = scale

        self._entity_effects_map = {}
        self._time_effects_map = {}
        resid_orig = y_arr - X_arr @ coef
        grand_mean = _to_float_scalar(_scaled_mean(resid_orig, xp))
        resid_centered = resid_orig - grand_mean
        self._grand_mean = grand_mean

        if (
            self.entity_effects
            and self.time_effects
            and entity_arr is not None
            and time_arr is not None
        ):
            ent_effects_dev, time_effects_dev = _recover_two_way_effects(
                resid_centered,
                entity_arr,
                time_arr,
                xp,
                max_iter=self.demean_max_iter,
                tol=self.demean_tol,
            )
            ent_effects = np.asarray(_to_numpy(ent_effects_dev)).ravel()
            time_effect_values = np.asarray(_to_numpy(time_effects_dev)).ravel()
            for i, eid in enumerate(entity_labels):
                self._entity_effects_map[eid] = float(ent_effects[i])
            for i, tid in enumerate(time_labels):
                self._time_effects_map[tid] = float(time_effect_values[i])
        else:
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
            fit_rank=fit_rank,
            cov_type=self._cov_type,
            cluster=cluster_for_cov,
            time_ids=time_ids,
            bandwidth=self.bandwidth,
            kernel=self.kernel,
            group_debias=bool(self.group_debias),
            extra_df=int(diagnostic_df["effect_rank"]),
            allowed=(
                "nonrobust",
                "robust",
                "hc0",
                "hc2",
                "hc3",
                "clustered",
                "driscoll-kraay",
            ),
            hc1_correction=(
                n / self.df_resid if self._cov_type == "robust" else None
            ),
            distribution_df=self.df_resid,
            diag_floor=0.0,
        )

        if has_level_constant or self.entity_effects or self.time_effects:
            y_d_centered = y_d - _scaled_mean(y_d, xp)
        else:
            y_d_centered = y_d
        self.rsquared_within, _r2_degenerate = _scaled_residual_r2(
            resid, y_d_centered, xp
        )
        resid_scale = xp.max(xp.abs(resid))
        total_scale = xp.max(xp.abs(y_d_centered))
        resid_unit = resid / xp.where(
            resid_scale > 0.0, resid_scale, xp.ones_like(resid_scale)
        )
        total_unit = y_d_centered / xp.where(
            total_scale > 0.0, total_scale, xp.ones_like(total_scale)
        )
        ss_res = _restore_squared_scale(
            _to_float_scalar(xp.sum(resid_unit * resid_unit)),
            _to_float_scalar(resid_scale),
        )
        ss_tot_diag = _restore_squared_scale(
            _to_float_scalar(xp.sum(total_unit * total_unit)),
            _to_float_scalar(total_scale),
        )
        self.fit_statistics_ = build_model_fit_statistics(
            y_arr,
            X_arr,
            coef,
            xp=xp,
            entity_codes=entity_arr,
            has_constant=has_level_constant,
            rss_fit=ss_res,
            tss_fit=ss_tot_diag,
            df_resid=diagnostic_df["df_resid"],
            df_total=diagnostic_df["df_total"],
            f_y=y_d,
            f_X=X_d,
            f_params=coef,
            f_has_constant=has_level_constant,
            metadata={
                "fit_space": (
                    "fixed-effect transformed regression"
                    if self.entity_effects or self.time_effects
                    else "level regression"
                ),
                "legacy_df_resid": int(legacy_df_resid),
                "public_df_resid_basis": public_df_basis,
                "diagnostic_df": dict(diagnostic_df),
                "legacy_rsquared_within": float(self.rsquared_within),
            },
        )
        hausman_compatible = (
            bool(self.entity_effects)
            and not bool(self.time_effects)
            and str(self._cov_type).lower() == "nonrobust"
        )
        self._panel_diagnostic_identity = (
            build_diagnostic_identity(
                X_arr,
                y_arr,
                xp=xp,
                entity_codes=entity_arr,
                feature_names=self._feature_names,
                has_constant=False,
            )
            if hausman_compatible
            else None
        )
        self._pooling_f_result = (
            pooling_f_from_level_arrays(
                y_arr,
                X_arr,
                xp=xp,
                rss_effects=ss_res,
                df_resid_effects=diagnostic_df["df_resid"],
                has_constant=False,
                resid_effects=resid,
            )
            if self.entity_effects or self.time_effects
            else None
        )

        self._params = np.asarray(self.coef_).ravel()
        self.coef_ = self._params
        self._fitted = True
        return self

    def pooling_f_test(self):
        """Return the classical test that included fixed effects are jointly zero."""
        from statgpu.panel._diagnostics import pooling_f_test

        return pooling_f_test(self)

    def hausman_test(self, random_effects_model):
        """Compare one-way entity FE with a matched classical RandomEffects fit."""
        from statgpu.panel._diagnostics import hausman_test

        return hausman_test(self, random_effects_model)

    def predict(self, X, entity_ids=None, time_ids=None):
        """Predict on the selected numerical backend and return NumPy output."""
        self._check_is_fitted()
        backend = self._get_backend(backend="auto")
        xp = backend.xp
        prediction = self._panel_predict_linear(
            X,
            model_has_intercept=False,
            add_intercept=False,
            return_numpy=False,
            omitted_constant_index=self._predict_constant_index,
            omitted_constant_value=self._predict_constant_value,
        )
        self._predict_backend_name = backend.name

        entity_ids_np = None if entity_ids is None else np.asarray(_to_numpy(entity_ids)).ravel()
        time_ids_np = None if time_ids is None else np.asarray(_to_numpy(time_ids)).ravel()
        if entity_ids_np is not None and entity_ids_np.shape[0] != int(prediction.shape[0]):
            raise ValueError("entity_ids must have one value per prediction row")
        if time_ids_np is not None and time_ids_np.shape[0] != int(prediction.shape[0]):
            raise ValueError("time_ids must have one value per prediction row")

        if self._two_way_entity_components and self._two_way_time_components:
            n_rows = int(prediction.shape[0])
            for row in range(n_rows):
                entity_value = None if entity_ids_np is None else entity_ids_np[row]
                time_value = None if time_ids_np is None else time_ids_np[row]
                entity_component = (
                    None
                    if entity_value is None
                    else self._two_way_entity_components.get(entity_value)
                )
                time_component = (
                    None
                    if time_value is None
                    else self._two_way_time_components.get(time_value)
                )
                # Even on a connected graph, entity/time effects are separately
                # identified only up to an opposite normalization shift.  A FE
                # prediction therefore requires both known labels in one
                # incidence component.  If both labels are unseen, no fitted FE
                # is used and the documented linear-only fallback remains.
                if entity_component is None and time_component is None:
                    continue
                if (
                    entity_component is None
                    or time_component is None
                    or entity_component != time_component
                ):
                    raise ValueError(
                        "two-way fixed-effect prediction is not identified unless "
                        "both entity and time labels are known and belong to the "
                        "same incidence component"
                    )

        n_rows = int(prediction.shape[0])
        uses_fitted_effect = np.zeros(n_rows, dtype=bool)

        def _effect_values(ids_np, mapping):
            if not mapping or ids_np is None:
                return None, None
            known = np.fromiter(
                (value in mapping for value in ids_np),
                dtype=bool,
                count=ids_np.shape[0],
            )
            values = np.fromiter(
                (float(mapping.get(value, 0.0)) for value in ids_np),
                dtype=np.float64,
                count=ids_np.shape[0],
            )
            return (
                xp_asarray(values, dtype=xp.float64, xp=xp, ref_arr=prediction),
                known,
            )

        entity_effect, entity_known = _effect_values(
            entity_ids_np, self._entity_effects_map
        )
        if entity_effect is not None:
            prediction = prediction + entity_effect
            uses_fitted_effect |= entity_known
        time_effect, time_known = _effect_values(
            time_ids_np, self._time_effects_map
        )
        if time_effect is not None:
            prediction = prediction + time_effect
            uses_fitted_effect |= time_known

        # Fixed-effect maps are recovered after centering the level residual by
        # its grand mean.  Whenever a row actually uses a stored fitted effect,
        # restore that common level component exactly once.  Rows whose labels
        # are wholly unseen preserve the documented linear-only fallback.
        if np.any(uses_fitted_effect):
            grand_mean = xp_asarray(
                uses_fitted_effect.astype(np.float64) * float(self._grand_mean),
                dtype=xp.float64,
                xp=xp,
                ref_arr=prediction,
            )
            prediction = prediction + grand_mean

        return np.asarray(_to_numpy(prediction), dtype=np.float64)

    def summary(self):
        """Print and return the existing structured coefficient summary."""
        self._check_is_fitted()
        k = len(self._params)
        feature_names_override = (
            None
            if self._feature_names is not None
            else [f"x{i + 1}" for i in range(k)]
        )
        return self._panel_summary(
            model_type="PanelOLS",
            cov_type=self._cov_type,
            rsquared_within=self.rsquared_within,
            entity_effects=self.entity_effects,
            time_effects=self.time_effects,
            feature_names_override=feature_names_override,
            print_result=True,
        )

    def get_params(self, deep=True):
        """Return the shared exact-constructor parameter contract."""
        return super().get_params(deep)

    def set_params(self, **params):
        """Validate FE convergence controls and delegate estimator updates."""
        if "demean_max_iter" in params:
            value = params["demean_max_iter"]
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ) or int(value) <= 0:
                raise ValueError("demean_max_iter must be a positive integer")
        if "demean_tol" in params:
            value = params["demean_tol"]
            if not np.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError("demean_tol must be finite and positive")
        return super().set_params(**params)


FixedEffects = PanelOLS