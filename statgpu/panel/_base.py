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
from statgpu.panel._utils import PanelSummary, validate_panel_alpha, validate_panel_numeric_data


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
        from statgpu.panel._formula import _align_formula_side_array, _prepare_formula_fit

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
        y_arr = xp_asarray(y_device, dtype=xp.float64, xp=xp, ref_arr=X_arr).ravel()
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if validate_alpha and hasattr(self, "alpha"):
            validate_panel_alpha(self.alpha)
        validate_panel_numeric_data(X_arr, y_arr, xp)
        return backend, xp, X_arr, y_arr

    def _panel_set_index_info(self, nobs, *, entity_ids=None, time_ids=None):
        info = build_panel_index_info(
            nobs, entity_ids=entity_ids, time_ids=time_ids
        )
        self._panel_index_info = info
        return info

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
        params_dev = xp_asarray(value, dtype=xp.float64, xp=xp, ref_arr=X_arr).ravel()
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
        bandwidth=None,
        kernel="bartlett",
        allowed=None,
        hc1_correction=None,
        distribution_df=None,
        diag_floor=1e-30,
    ):
        """Store existing residual-based OLS inference from the shared registry."""
        from statgpu.inference._distributions_backend import get_distribution
        from statgpu.panel._covariance import ols_covariance

        xp = backend.xp
        cov_params = ols_covariance(
            X,
            resid,
            cov_type=cov_type,
            scale=scale,
            df_resid=df_resid,
            cluster=cluster,
            bandwidth=bandwidth,
            kernel=kernel,
            xp=xp,
            allowed=allowed,
            hc1_correction=hc1_correction,
        )
        diag = xp.diag(cov_params)
        if diag_floor is not None:
            diag = xp_maximum(diag, float(diag_floor), xp)
        bse_dev = xp.sqrt(diag)
        if diag_floor is None:
            tvalues_dev = params / bse_dev
        else:
            denominator = xp_maximum(bse_dev, np.finfo(np.float64).tiny, xp)
            tvalues_dev = params / denominator

        dist_name = "t" if str(cov_type).lower() == "nonrobust" else "norm"
        distribution = get_distribution(dist_name, backend=backend.name)
        df = int(df_resid if distribution_df is None else distribution_df)
        if dist_name == "t":
            pvalues_dev = 2 * distribution.sf(xp.abs(tvalues_dev), df)
            critical = distribution.isf(float(self.alpha) / 2, df)
        else:
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
                getattr(self, "_feature_names", None), len(coef_np), prefix=prefix
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
