"""Shared internal lifecycle helpers for panel estimators.

The Stage-A abstraction is intentionally conservative: it centralizes formula
state, backend preparation, panel metadata, ordinary linear prediction,
residual-based inference finalization, and summary construction without choosing
an econometric transformation for subclasses.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from statgpu._base import BaseEstimator
from statgpu.backends import _to_numpy, xp_asarray, xp_maximum, xp_ones
from statgpu.panel._results import build_panel_index_info
from statgpu.panel._utils import (
    PanelSummary,
    validate_panel_alpha,
    validate_panel_numeric_data,
)


class BasePanelModel(BaseEstimator):
    """Internal base class for statistically neutral panel-model lifecycle code."""

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
            for name, value in list(aligned.items()):
                aligned[name] = _align_formula_side_array(
                    value, design_info, len(y_data), name
                )

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
        PanelOLS preserves a historical residual-df convention that is one rank
        parameterization away from the standard fixed-effect model df used by
        classical Hausman tests.  When Stage-B fit metadata provides both the
        legacy and standard diagnostic df, convert only this internal covariance
        copy to the standard homoskedastic scale.
        """
        raw = getattr(self, "_panel_cov_params_raw", None)
        if raw is None:
            return None
        result = getattr(self, "fit_statistics_", None)
        metadata = getattr(result, "metadata", {}) if result is not None else {}
        diagnostic_df = metadata.get("diagnostic_df")
        legacy_df = metadata.get("legacy_df_resid")
        cov_type = str(getattr(self, "_cov_type", "nonrobust")).lower()
        if (
            cov_type == "nonrobust"
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
    ):
        """Shared formula-aware linear prediction with explicit return contract."""
        self._check_is_fitted()
        from statgpu.panel._formula import _formula_predict

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
        if int(X_arr.shape[1]) != int(params_dev.shape[0]):
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
        diag_floor=1e-30,
    ):
        """Store residual-based OLS inference from the shared covariance registry."""
        from statgpu.inference._distributions_backend import get_distribution
        from statgpu.inference._results import ParameterInferenceResult
        from statgpu.panel._covariance import (
            normalize_covariance_type,
            ols_covariance,
        )

        xp = backend.xp
        canonical_cov_type = normalize_covariance_type(cov_type)
        covariance_metadata: dict = {}
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
        self._covariance_metadata = covariance_metadata
        self._panel_cov_params_raw = np.asarray(
            _to_numpy(cov_params), dtype=np.float64
        )

        diag = xp.diag(cov_params)
        cov_np = self._panel_cov_params_raw
        diag_np = np.diag(cov_np).astype(np.float64, copy=False)
        row_scale = np.max(np.abs(cov_np), axis=1)
        col_scale = np.max(np.abs(cov_np), axis=0)
        local_scale = np.maximum(row_scale, col_scale)
        negative_tol = 4096.0 * np.finfo(np.float64).eps * local_scale
        if np.any(diag_np < -negative_tol):
            raise ValueError(
                "covariance has materially negative diagonal variance; "
                "inference is not numerically valid"
            )
        # Only suppress elementwise roundoff-scale negative zeros.  A material
        # negative variance must fail closed before any historical diagonal floor
        # is used, even when another coefficient has a much larger variance.
        diag = xp_maximum(diag, 0.0, xp)
        if diag_floor is not None:
            diag = xp_maximum(diag, float(diag_floor), xp)
        bse_dev = xp.sqrt(diag)
        if diag_floor is None:
            tvalues_dev = params / bse_dev
        else:
            denominator = xp_maximum(
                bse_dev, np.finfo(np.float64).tiny, xp
            )
            tvalues_dev = params / denominator

        dist_name = "t" if canonical_cov_type == "nonrobust" else "norm"
        df = int(df_resid if distribution_df is None else distribution_df)
        if dist_name == "t" and df == 1:
            distribution = get_distribution("cauchy", backend=backend.name)
            pvalues_dev = 2 * distribution.sf(xp.abs(tvalues_dev))
            critical = distribution.isf(float(self.alpha) / 2)
        elif dist_name == "t":
            distribution = get_distribution("t", backend=backend.name)
            pvalues_dev = 2 * distribution.sf(xp.abs(tvalues_dev), df)
            critical = distribution.isf(float(self.alpha) / 2, df)
        else:
            distribution = get_distribution("norm", backend=backend.name)
            pvalues_dev = 2 * distribution.sf(xp.abs(tvalues_dev))
            critical = distribution.isf(float(self.alpha) / 2)
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
            metadata={"covariance": dict(covariance_metadata)},
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
