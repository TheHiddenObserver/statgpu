"""Shared internal lifecycle helpers for panel estimators.

The Stage-A abstraction is intentionally conservative: it centralizes formula
state, backend preparation, panel metadata, ordinary linear prediction,
residual-based inference finalization, and summary construction without choosing
an econometric transformation for subclasses.
"""

from __future__ import annotations

import functools
from typing import Dict, Optional

import numpy as np

from statgpu._base import BaseEstimator
from statgpu.backends import _to_float_scalar, _to_numpy, xp_asarray, xp_maximum, xp_ones
from statgpu.inference._reference_distribution import two_sided_reference_inference
from statgpu.panel._results import build_panel_index_info
from statgpu.panel._utils import (
    PanelSummary,
    validate_panel_alpha,
    validate_panel_numeric_data,
    _zero_safe_statistic_ratio,
)


class BasePanelModel(BaseEstimator):
    """Internal base class for statistically neutral panel-model lifecycle code."""

    # Formula-facing panel estimators align these observation-level metadata
    # arrays only after Patsy has established the retained fit rows.  The public
    # finite-input guard must therefore defer validation until after alignment:
    # a non-finite label on a row Patsy drops is irrelevant, while retained-row
    # missing labels are still rejected by the panel metadata validators.
    _FORMULA_OWNED_SIDE_ARRAYS = frozenset(
        {"entity_ids", "time_ids", "time_index", "cluster"}
    )

    def __init_subclass__(cls, **kwargs):
        """Wrap every public panel ``fit`` in a fail-closed state transaction."""
        super().__init_subclass__(**kwargs)
        original_fit = cls.__dict__.get("fit")
        if original_fit is None or getattr(
            original_fit, "__statgpu_panel_transactional_fit__", False
        ):
            return

        @functools.wraps(original_fit)
        def transactional_fit(self, *args, **kwargs):
            self._reset_fit_state()
            try:
                return original_fit(self, *args, **kwargs)
            except BaseException:
                # A failed fit must expose neither the previous successful fit
                # nor partially written outputs from the failed new request.
                backend_name = getattr(self, "_backend_name", None)
                self._reset_fit_state()
                if backend_name is not None:
                    # The failure executed on a concrete backend; retain the
                    # execution provenance so fail-closed audits can attribute
                    # the failure without reconstructing it from the requested
                    # device (which cannot detect a silent fallback).
                    self._backend_name = backend_name
                raise

        transactional_fit.__statgpu_panel_transactional_fit__ = True
        cls.fit = transactional_fit

    def _reset_fit_state(self):
        """Invalidate common panel fit/inference outputs before a new fit."""
        self._fitted = False
        for name in (
            "coef_",
            "bse_",
            "tvalues_",
            "pvalues_",
            "conf_int_",
            "betas_",
            "cov_params_",
            "nobs",
            "n_periods",
            "df_resid",
            "rank_",
            "rsquared",
            "rsquared_within",
            "theta_",
            "variance_components_",
            "_params",
            "_bse",
            "_tvalues",
            "_zvalues",
            "_pvalues",
            "_conf_int",
            "_inference_result",
            "_backend_name",
            "_inference_backend_name",
            "_predict_backend_name",
            "_panel_index_info",
            "_design_info",
            "_feature_names",
            "_formula_has_intercept",
            "_panel_cov_params_raw",
            "_covariance_metadata",
            "_coefficient_inference_available",
            "_coefficient_inference_reason",
            "_bp_lm_result",
            "_panel_diagnostic_identity",
            "_predict_constant_index",
            "_predict_constant_value",
            "_scale",
        ):
            self.__dict__.pop(name, None)
        self.fit_statistics_ = None

    def _panel_resolve_formula_ids(
        self,
        formula,
        explicit_entity_ids,
        explicit_time_ids,
        formula_entity_ids,
        formula_time_ids,
    ):
        """Reconcile explicit panel IDs with pipe-named formula metadata."""
        if formula is None:
            return explicit_entity_ids, explicit_time_ids, ()

        from statgpu.panel._formula import _split_panel_formula

        _main_formula, pipe_vars = _split_panel_formula(formula)
        pipe_vars = tuple(pipe_vars)

        def resolve(name, explicit, parsed, pipe_index):
            if pipe_index < len(pipe_vars):
                pipe_name = pipe_vars[pipe_index]
                if parsed is None:
                    raise ValueError(
                        f"Formula pipe fixed-effect variable {pipe_name!r} was not found in data"
                    )
                if explicit is not None:
                    try:
                        matches = np.array_equal(
                            np.asarray(explicit), np.asarray(parsed)
                        )
                    except (TypeError, ValueError):
                        matches = False
                    if not matches:
                        raise ValueError(
                            f"{name} conflicts with formula pipe fixed-effect variable "
                            f"{pipe_name!r}; remove the explicit argument or make "
                            "it match the formula column"
                        )
                return parsed
            return explicit if explicit is not None else parsed

        return (
            resolve(
                "entity_ids", explicit_entity_ids, formula_entity_ids, 0
            ),
            resolve(
                "time_ids", explicit_time_ids, formula_time_ids, 1
            ),
            pipe_vars,
        )

    def _panel_prepare_formula_fit(
        self,
        formula,
        data,
        X,
        y,
        *,
        model_has_intercept: bool,
        support_pipe: bool = False,
        side_arrays: Optional[Dict[str, object]] = None,
    ):
        from statgpu.panel._formula import (
            _align_formula_side_array,
            _prepare_formula_fit,
        )

        (
            y_data,
            X_data,
            design_info,
            feature_names,
            formula_has_intercept,
            fe_entity_ids,
            fe_time_ids,
            fe_entity_effects,
            fe_time_effects,
        ) = _prepare_formula_fit(
            formula,
            data,
            X,
            y,
            model_has_intercept=model_has_intercept,
            support_pipe=support_pipe,
        )

        self._design_info = design_info
        self._feature_names = feature_names
        self._formula_has_intercept = formula_has_intercept

        aligned = dict(side_arrays or {})
        if formula is not None:
            from statgpu.backends._validation import check_finite

            for name, value in list(aligned.items()):
                aligned_value = _align_formula_side_array(
                    value, design_info, len(y_data), name
                )
                if aligned_value is not None:
                    # Public finite validation is intentionally deferred until
                    # after Patsy row filtering for formula-owned metadata.
                    # Reapply it here so retained rows keep the same finite-input
                    # contract even when a later estimator path does not consume
                    # the optional metadata.
                    check_finite(aligned_value, name=name)
                aligned[name] = aligned_value

        return (
            y_data,
            X_data,
            fe_entity_ids,
            fe_time_ids,
            bool(fe_entity_effects),
            bool(fe_time_effects),
            aligned,
        )

    def _panel_prepare_numeric(self, X, y, *, validate_alpha: bool = True):
        """Convert X/y to backend float64 while preserving explicit device use."""
        backend = self._get_backend(backend="auto")
        xp = backend.xp
        X_device = self._to_array(X, backend=backend.name)
        y_device = self._to_array(y, backend=backend.name)
        X_arr = xp_asarray(X_device, dtype=xp.float64, xp=xp)
        y_arr = xp_asarray(
            y_device, dtype=xp.float64, xp=xp, ref_arr=X_arr
        ).ravel()
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if validate_alpha and hasattr(self, "alpha"):
            validate_panel_alpha(self.alpha)
        validate_panel_numeric_data(X_arr, y_arr, xp)
        # Persist the backend actually selected at the numerical fit boundary.
        # Physical validation must not reconstruct this provenance later from
        # the requested device, since that cannot detect a silent fallback.
        self._backend_name = backend.name
        return backend, xp, X_arr, y_arr

    def _panel_set_index_info(self, nobs, *, entity_ids=None, time_ids=None):
        info = build_panel_index_info(
            nobs, entity_ids=entity_ids, time_ids=time_ids
        )
        self._panel_index_info = info
        return info

    def set_params(self, **params):
        """Set estimator parameters and refresh panel covariance aliases.

        ``BaseEstimator`` deliberately preserves the exact user-supplied public
        constructor value for sklearn constructor identity.  Panel covariance
        dispatch, however, uses the normalized private ``_cov_type`` runtime
        value.  Refresh only that private value after sklearn-style parameter
        updates so ``hc1``/``dk``/``kernel`` behave exactly like direct
        construction without changing the public raw parameter.
        """
        if "group_debias" in params and not isinstance(
            params["group_debias"], (bool, np.bool_)
        ):
            raise ValueError("group_debias must be boolean")
        result = super().set_params(**params)
        if "cov_type" in params and hasattr(self, "cov_type"):
            from statgpu.panel._covariance import normalize_covariance_type

            self._cov_type = normalize_covariance_type(self.cov_type)
        return result

    @property
    def _panel_cov_params(self):
        """Return the small covariance matrix used by Stage-B diagnostics.

        Existing Stage-A inference is computed and stored before this property is
        consulted, so rescaling here cannot change public bse/t/p/CI values.
        Current PanelOLS fits use the standard fixed-effect residual df directly.
        The legacy branch is retained only for compatibility with an already
        materialized result whose metadata explicitly records a legacy-scaled
        homoskedastic covariance.
        """
        raw = getattr(self, "_panel_cov_params_raw", None)
        if raw is None:
            return None
        result = getattr(self, "fit_statistics_", None)
        metadata = getattr(result, "metadata", {}) if result is not None else {}
        diagnostic_df = metadata.get("diagnostic_df")
        legacy_df = metadata.get("legacy_df_resid")
        public_df_basis = metadata.get("public_df_resid_basis")
        cov_type = str(getattr(self, "_cov_type", "nonrobust")).lower()
        if (
            cov_type == "nonrobust"
            and public_df_basis == "legacy"
            and isinstance(diagnostic_df, dict)
            and legacy_df is not None
        ):
            standard_df = diagnostic_df.get("df_resid")
            if standard_df is not None and int(standard_df) > 0:
                factor = float(legacy_df) / float(standard_df)
                return raw * factor
        return raw

    def _panel_predict_linear(
        self,
        X,
        *,
        model_has_intercept: bool,
        add_intercept: bool,
        return_numpy: bool,
        params=None,
        omitted_constant_index=None,
        omitted_constant_value=None,
    ):
        """Shared formula-aware linear prediction with explicit return contract.

        A one-column-short prediction matrix is accepted only when ``fit``
        persisted an actual explicit constant column.  The exact fitted constant
        value and its original position are restored; arbitrary shape mismatches
        never silently turn into an intercept model.
        """
        self._check_is_fitted()
        from statgpu.panel._formula import _formula_predict

        # A formula-generated prediction matrix carries column identity through
        # Patsy.  When the fitted formula had an intercept, _formula_predict()
        # deterministically strips that intercept just as fit() did, so a slope
        # column that happens to be constant on this prediction batch must not be
        # mistaken for an explicitly supplied intercept.
        formula_omitted_intercept = (
            getattr(self, "_design_info", None) is not None
            and hasattr(X, "columns")
            and bool(getattr(self, "_formula_has_intercept", False))
        )

        X_data = _formula_predict(
            X,
            getattr(self, "_design_info", None),
            getattr(self, "_formula_has_intercept", None),
            model_has_intercept=model_has_intercept,
        )
        backend = self._get_backend(backend="auto")
        xp = backend.xp
        X_device = self._to_array(X_data, backend=backend.name)
        X_arr = xp_asarray(X_device, dtype=xp.float64, xp=xp)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if X_arr.ndim != 2:
            raise ValueError("X must be one- or two-dimensional")

        if add_intercept:
            ones = xp_ones((int(X_arr.shape[0]), 1), xp.float64, xp, X_arr)
            if getattr(xp, "__name__", "") == "torch":
                X_arr = xp.cat([ones, X_arr], dim=1)
            else:
                X_arr = xp.concatenate([ones, X_arr], axis=1)

        value = self.coef_ if params is None else params
        params_dev = xp_asarray(
            value, dtype=xp.float64, xp=xp, ref_arr=X_arr
        ).ravel()
        target_columns = int(params_dev.shape[0])
        if (
            int(X_arr.shape[1]) == target_columns - 1
            and omitted_constant_index is not None
        ):
            constant_index = int(omitted_constant_index)
            if not 0 <= constant_index < target_columns:
                raise ValueError("stored prediction constant index is out of range")
            if omitted_constant_value is None or not np.isfinite(
                float(omitted_constant_value)
            ):
                raise ValueError("stored prediction constant value must be finite")

            # Shape alone cannot prove which training column was omitted.  If
            # the supplied short matrix already contains the fitted constant,
            # inserting another constant would silently reinterpret a missing
            # slope as an omitted intercept.  Reject that ambiguous case.
            if int(X_arr.shape[1]) > 0 and not formula_omitted_intercept:
                delta = xp.abs(X_arr - float(omitted_constant_value))
                if getattr(xp, "__name__", "") == "torch":
                    max_delta = xp.max(delta, dim=0).values
                else:
                    max_delta = xp.max(delta, axis=0)
                constant_tol = (
                    256.0
                    * np.finfo(np.float64).eps
                    * abs(float(omitted_constant_value))
                )
                contains_fitted_constant = _to_float_scalar(
                    xp.any(max_delta <= constant_tol)
                ) > 0.0
                if contains_fitted_constant:
                    raise ValueError(
                        "X has an incompatible feature count; the supplied short "
                        "matrix already contains the fitted constant, so the "
                        "omitted training feature is ambiguous"
                    )

            constant = xp_ones(
                (int(X_arr.shape[0]), 1), xp.float64, xp, X_arr
            ) * float(omitted_constant_value)
            left = X_arr[:, :constant_index]
            right = X_arr[:, constant_index:]
            if getattr(xp, "__name__", "") == "torch":
                X_arr = xp.cat([left, constant, right], dim=1)
            else:
                X_arr = xp.concatenate([left, constant, right], axis=1)

        if int(X_arr.shape[1]) != target_columns:
            raise ValueError("X has an incompatible feature count")
        prediction = X_arr @ params_dev
        return _to_numpy(prediction) if return_numpy else prediction

    def _panel_store_ols_inference(
        self,
        X,
        resid,
        params,
        *,
        scale,
        df_resid,
        backend,
        cov_type,
        cluster=None,
        time_ids=None,
        bandwidth=None,
        kernel="bartlett",
        group_debias: bool = False,
        extra_df: int = 0,
        allowed=None,
        hc1_correction=None,
        distribution_df=None,
        fit_rank=None,
        diag_floor=0.0,
    ):
        """Store residual-based OLS inference from the shared covariance registry."""
        from statgpu.inference._results import BaseInferenceResult, ParameterInferenceResult
        from statgpu.panel._covariance import (
            normalize_covariance_type,
            ols_covariance,
        )

        xp = backend.xp
        canonical_cov_type = normalize_covariance_type(cov_type)
        if fit_rank is None:
            from statgpu.panel._linalg import panel_matrix_rank

            fit_rank = panel_matrix_rank(X, xp)
        fit_rank = int(fit_rank)
        design_columns = int(X.shape[1])
        if fit_rank <= 0 or fit_rank > design_columns:
            raise ValueError(
                "fit_rank must identify a positive subspace no larger than the design"
            )
        rank_deficient = fit_rank < design_columns

        covariance_metadata: dict = {}
        # HC2/HC3 divide by (1-h_i).  A rank-deficient design can have h_i=1
        # while still having positive residual df, so the coefficient covariance
        # is undefined even though fitted values remain perfectly well defined.
        # Since coordinate inference is unavailable for every rank-deficient fit
        # anyway, do not let this secondary covariance invalidate the fit itself.
        skip_rank_deficient_hc = rank_deficient and canonical_cov_type in {"hc2", "hc3"}
        if skip_rank_deficient_hc:
            covariance_metadata.update(
                {
                    "covariance": canonical_cov_type,
                    "rank_deficient_covariance_unavailable": True,
                }
            )
            cov_params = None
            self._panel_cov_params_raw = None
        else:
            cov_params = ols_covariance(
                X,
                resid,
                cov_type=canonical_cov_type,
                scale=scale,
                df_resid=df_resid,
                cluster=cluster,
                time_ids=time_ids,
                bandwidth=bandwidth,
                kernel=kernel,
                group_debias=group_debias,
                extra_df=extra_df,
                xp=xp,
                allowed=allowed,
                hc1_correction=hc1_correction,
                metadata=covariance_metadata,
            )
            self._panel_cov_params_raw = np.asarray(
                _to_numpy(cov_params), dtype=np.float64
            )
            cov_np = self._panel_cov_params_raw
            if not np.all(np.isfinite(cov_np)):
                raise ValueError(
                    "covariance contains non-finite values; inference is not numerically valid"
                )
        self._covariance_metadata = covariance_metadata
        self._coefficient_inference_available = not rank_deficient
        self._coefficient_inference_reason = None
        self._covariance_metadata["design_rank"] = fit_rank
        self._covariance_metadata["design_columns"] = design_columns
        self._covariance_metadata["coefficient_inference_applicable"] = not rank_deficient

        if rank_deficient:
            reason = (
                "coefficient-level inference is unavailable because the fit-space design "
                f"is rank deficient (rank={fit_rank}, columns={design_columns}); "
                "fitted values and identified fit-space quantities remain available"
            )
            self._coefficient_inference_reason = reason
            self._covariance_metadata["coefficient_inference_reason"] = reason
            self.coef_ = np.asarray(_to_numpy(params), dtype=np.float64).ravel()
            self.bse_ = None
            self.tvalues_ = None
            self.pvalues_ = None
            self.conf_int_ = None
            self._params = self.coef_.copy()
            self._bse = None
            self._tvalues = None
            self._zvalues = None
            self._pvalues = None
            self._conf_int = None
            feature_names = getattr(self, "_feature_names", None)
            if feature_names is not None and len(feature_names) != len(self.coef_):
                feature_names = None
            BaseInferenceResult(
                method="panel_ols_unavailable",
                feature_names=feature_names,
                metadata={
                    "applicable": False,
                    "reason": reason,
                    "covariance": dict(covariance_metadata),
                    "fit_rank": fit_rank,
                    "design_columns": design_columns,
                },
            ).apply_to(self)
            return cov_params

        diag = xp.diag(cov_params)
        diag_np = np.diag(cov_np).astype(np.float64, copy=False)
        # A variance is invalid whenever it is strictly negative. There is no
        # dimensionally valid generic tolerance that can distinguish a small
        # negative variance from cancellation using only the final covariance:
        # outcome and regressor rescaling transform covariance entries at
        # different rates. IEEE negative zero compares equal to zero and is
        # normalized below without weakening this fail-closed rule.
        if np.any(diag_np < 0.0):
            raise ValueError(
                "covariance has materially negative diagonal variance; "
                "inference is not numerically valid"
            )
        # ``diag_floor`` is retained only for private-call compatibility.
        # A positive absolute floor is dimensionful and would break outcome-scale
        # equivariance, so it is intentionally ignored. Exact zero is handled by
        # the explicit statistic-ratio convention below.
        _ = diag_floor
        diag = xp_maximum(diag, 0.0, xp)
        bse_dev = xp.sqrt(diag)
        tvalues_dev = _zero_safe_statistic_ratio(params, bse_dev, xp)

        dist_name = "t" if canonical_cov_type == "nonrobust" else "normal"
        df = int(df_resid if distribution_df is None else distribution_df)
        inference_device = (
            str(params.device)
            if backend.name == "torch" and hasattr(params, "device")
            else None
        )
        pvalues_dev, critical = two_sided_reference_inference(
            xp.abs(tvalues_dev),
            distribution=dist_name,
            alpha=self.alpha,
            backend=backend.name,
            xp=xp,
            df=df if dist_name == "t" else None,
            device=inference_device,
        )
        self._inference_backend_name = backend.name
        critical = xp_asarray(
            critical, dtype=params.dtype, xp=xp, ref_arr=params
        )
        conf_low = params - critical * bse_dev
        conf_high = params + critical * bse_dev

        self.coef_ = np.asarray(_to_numpy(params)).ravel()
        self.bse_ = np.asarray(_to_numpy(bse_dev)).ravel()
        self.tvalues_ = np.asarray(_to_numpy(tvalues_dev)).ravel()
        self.pvalues_ = np.asarray(_to_numpy(pvalues_dev)).ravel()
        self.conf_int_ = np.asarray(
            _to_numpy(xp.stack([conf_low, conf_high], axis=1))
        )

        feature_names = getattr(self, "_feature_names", None)
        if feature_names is not None and len(feature_names) != len(self.coef_):
            feature_names = None
        inference = ParameterInferenceResult(
            method="panel_ols",
            feature_names=feature_names,
            metadata={
                "covariance": dict(covariance_metadata),
                "inference_backend": backend.name,
            },
            params=self.coef_,
            bse=self.bse_,
            statistic=self.tvalues_,
            statistic_name="t" if dist_name == "t" else "z",
            pvalues=self.pvalues_,
            conf_int=self.conf_int_,
            cov_type=canonical_cov_type,
            distribution="t" if dist_name == "t" else "normal",
            df=float(df) if dist_name == "t" else None,
        )
        inference.apply_to(self)
        return cov_params

    def _panel_summary(
        self,
        *,
        model_type: str,
        cov_type=None,
        rsquared_within=None,
        entity_effects=None,
        time_effects=None,
        variance_components=None,
        theta=None,
        extra=None,
        prefix: str = "x",
        feature_names_override=None,
        print_result: bool = False,
    ):
        """Construct the existing PanelSummary with explicit compatibility knobs."""
        self._check_is_fitted()
        from statgpu.panel._formula import _get_feature_names

        if getattr(self, "_coefficient_inference_available", True) is False:
            raise ValueError(
                getattr(
                    self,
                    "_coefficient_inference_reason",
                    "coefficient-level inference is unavailable for this fit",
                )
            )

        coef_np = np.asarray(_to_numpy(self.coef_)).ravel()
        if feature_names_override is None:
            feature_names = _get_feature_names(
                getattr(self, "_feature_names", None),
                len(coef_np),
                prefix=prefix,
            )
        else:
            feature_names = list(feature_names_override)
        summary = PanelSummary(
            model_type=model_type,
            cov_type=cov_type,
            coef=coef_np,
            bse=np.asarray(_to_numpy(self.bse_)).ravel(),
            tvalues=np.asarray(_to_numpy(self.tvalues_)).ravel(),
            pvalues=np.asarray(_to_numpy(self.pvalues_)).ravel(),
            conf_int=np.asarray(_to_numpy(self.conf_int_)),
            nobs=int(self.nobs),
            df_resid=int(self.df_resid),
            alpha=float(self.alpha),
            feature_names=feature_names,
            rsquared_within=rsquared_within,
            entity_effects=entity_effects,
            time_effects=time_effects,
            variance_components=variance_components,
            theta=theta,
            extra={} if extra is None else dict(extra),
        )
        if print_result:
            print(summary)
        return summary