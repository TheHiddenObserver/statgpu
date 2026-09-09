"""PR #138 review-fix contract for weighted sparse Gaussian GPU paths.

This module closes two cross-path regressions found by the canonical code-review
loop after the first physical acceptance:

* the weighted average-loss fix must apply to the public generic
  ``PenalizedGeneralizedLinearModel`` entry point as well as typed
  ``PenalizedLinearRegression``/Lasso/ElasticNet wrappers;
* weighted GPU fits that request debiased inference must keep that numerical
  inference on the executed CuPy/Torch backend instead of falling through to the
  CPU-oriented post-fit helper.

The implementation deliberately reuses the maintained fused unweighted GPU
solver. Weighted, centered rows are rescaled by
``sqrt(n_samples / sum(sample_weight))`` so that its ``1 / n_samples`` objective
is exactly the declared weighted ``1 / sum(sample_weight)`` objective.
"""

from __future__ import annotations

import functools

import numpy as np

from statgpu.backends import _to_numpy
from statgpu.backends._utils import _get_xp, xp_asarray
from statgpu.inference._results import DebiasedInferenceResult
from statgpu.linear_model._gaussian_inference import _inverse_or_pinv
from statgpu.linear_model.penalized._base import PenalizedGeneralizedLinearModel
from statgpu.linear_model.penalized._fit_mixin import _validate_sample_weight_backend
from statgpu.linear_model.penalized._penalized_linear import PenalizedLinearRegression

from . import _penalized_inference_api_contract as _inference_contract


_REVIEW_FIX_MARKER = "__statgpu_pr138_weighted_sparse_review_fix__"
_BASE_GPU_FIT = PenalizedGeneralizedLinearModel._fit_gpu_backend


def _runtime_inference_method(model) -> str:
    return str(
        getattr(
            model,
            "_inference_method",
            getattr(model, "inference_method", ""),
        )
    ).strip().lower()


def _is_weighted_sparse_fast_path(model, sample_weight, backend_name: str) -> bool:
    if sample_weight is None:
        return False
    # The centered/sqrt-weight working transform below is an algebraic identity
    # only for the quadratic Gaussian objective.  The wrapper is installed on
    # the generic PGLM class, so fail this predicate before looking at the
    # sparse penalty/solver for Logistic/Poisson/Gamma/etc. Those losses must
    # keep their own sample_weight-aware backend objective unchanged.
    loss_obj = getattr(model, "_loss", None)
    loss_name = str(
        getattr(loss_obj, "name", getattr(model, "loss", ""))
    ).strip().lower()
    if loss_name != "squared_error":
        return False
    penalty_name = str(
        getattr(
            getattr(model, "_penalty", None),
            "name",
            getattr(model, "penalty", ""),
        )
    ).strip().lower()
    solver_name = getattr(model, "_selected_solver", None)
    if not solver_name:
        solver_name = model._select_solver(model._loss, backend_name=backend_name)
    return penalty_name in {"l1", "elasticnet", "en"} and str(solver_name).lower() in {
        "fista",
        "fista_bb",
    }


def _normal_intercept_report(
    z_native,
    se_native,
    intercept: float,
    backend_name: str,
    ref_arr,
):
    """Return normal-reference intercept p-value and CI after device computation."""
    from statgpu.inference._distributions_backend import norm as _norm

    if backend_name == "torch":
        import torch

        one = torch.tensor(1.0, dtype=ref_arr.dtype, device=ref_arr.device)
        p_native = torch.minimum(one, 2.0 * _norm.sf(torch.abs(z_native)))
        critical = _norm.ppf(0.975)
        intercept_native = torch.tensor(
            intercept,
            dtype=ref_arr.dtype,
            device=ref_arr.device,
        )
        ci_native = torch.stack(
            [
                intercept_native - critical * se_native,
                intercept_native + critical * se_native,
            ]
        )
    elif backend_name == "cupy":
        import cupy as cp

        one = cp.asarray(1.0, dtype=ref_arr.dtype)
        p_native = cp.minimum(one, 2.0 * _norm.sf(cp.abs(z_native)))
        critical = _norm.ppf(0.975)
        intercept_native = cp.asarray(intercept, dtype=ref_arr.dtype)
        ci_native = cp.stack(
            [
                intercept_native - critical * se_native,
                intercept_native + critical * se_native,
            ]
        )
    else:
        from scipy import stats

        z_value = float(np.asarray(_to_numpy(z_native), dtype=np.float64))
        se_value = float(np.asarray(_to_numpy(se_native), dtype=np.float64))
        p_value = float(2.0 * stats.norm.sf(abs(z_value)))
        critical = float(stats.norm.ppf(0.975))
        ci = np.asarray(
            [
                intercept - critical * se_value,
                intercept + critical * se_value,
            ],
            dtype=np.float64,
        )
        return p_value, ci

    return (
        float(np.asarray(_to_numpy(p_native), dtype=np.float64)),
        np.asarray(_to_numpy(ci_native), dtype=np.float64),
    )


def _finalize_weighted_debiased_result(
    model,
    *,
    X_arr,
    y_work,
    X_work,
    row_scale,
    coef_native,
    backend_name: str,
    original_intercept: bool,
    simultaneous_requested: bool,
    sample_weighted: bool = True,
):
    """Restore intercept/layout after centered backend-native debiased inference."""
    base_result = getattr(model, "_inference_result", None)
    if base_result is None or str(getattr(base_result, "method", "")).lower() != "debiased":
        raise RuntimeError(
            "sparse debiased inference did not produce the required "
            "backend-native result; refusing an inference fallback"
        )

    xp = _get_xp(backend_name)
    n = int(X_work.shape[0])
    p = int(X_work.shape[1])
    resid_work = y_work - X_work @ coef_native
    s_hat = int(
        float(
            np.asarray(
                _to_numpy(xp.sum(xp.abs(coef_native) > 0)),
                dtype=np.float64,
            )
        )
    )
    scale_native = xp.sum(resid_work * resid_work) / float(max(n - s_hat, 1))
    scale = float(np.asarray(_to_numpy(scale_native), dtype=np.float64))

    params = np.asarray(base_result.params, dtype=np.float64).reshape(-1)
    bse = np.asarray(base_result.bse, dtype=np.float64).reshape(-1)
    statistic = np.asarray(base_result.statistic, dtype=np.float64).reshape(-1)
    pvalues = np.asarray(base_result.pvalues, dtype=np.float64).reshape(-1)
    conf_int = np.asarray(base_result.conf_int, dtype=np.float64)

    if original_intercept:
        feature_block = X_arr * row_scale.reshape(-1, 1)
        intercept_block = row_scale.reshape(-1, 1)
        if backend_name == "torch":
            full_design = xp.cat([intercept_block, feature_block], dim=1)
        else:
            full_design = xp.concatenate(
                [intercept_block, feature_block],
                axis=1,
            )
        bread_inv = _inverse_or_pinv(full_design.T @ full_design, backend_name)
        se_intercept_native = xp.sqrt(xp.abs(scale_native * bread_inv[0, 0]))
        intercept_native = xp_asarray(
            [float(model.intercept_)],
            dtype=X_arr.dtype,
            xp=xp,
            ref_arr=X_arr,
        ).reshape(-1)[0]
        z_intercept_native = intercept_native / (se_intercept_native + 1e-30)
        p_intercept, ci_intercept = _normal_intercept_report(
            z_intercept_native,
            se_intercept_native,
            float(model.intercept_),
            backend_name,
            X_arr,
        )
        se_intercept = float(
            np.asarray(_to_numpy(se_intercept_native), dtype=np.float64)
        )
        z_intercept = float(
            np.asarray(_to_numpy(z_intercept_native), dtype=np.float64)
        )
        params = np.concatenate([[float(model.intercept_)], params])
        bse = np.concatenate([[se_intercept], bse])
        statistic = np.concatenate([[z_intercept], statistic])
        pvalues = np.concatenate([[p_intercept], pvalues])
        conf_int = np.vstack([ci_intercept.reshape(1, 2), conf_int])
        model._X_design = np.column_stack(
            [
                np.asarray(_to_numpy(row_scale), dtype=np.float64),
                np.asarray(_to_numpy(X_work), dtype=np.float64),
            ]
        )
    else:
        model._X_design = np.asarray(_to_numpy(X_work), dtype=np.float64)

    model._y = np.asarray(_to_numpy(y_work), dtype=np.float64).reshape(-1)
    model._resid = np.asarray(_to_numpy(resid_work), dtype=np.float64).reshape(-1)
    model._scale = scale
    model._nobs = n
    model._df_resid = n - (p + int(original_intercept))

    model._params = params
    model._bse = bse
    model._tvalues = statistic
    model._zvalues = statistic
    model._pvalues = pvalues
    model._conf_int = conf_int

    if simultaneous_requested:
        model._compute_simultaneous_ci_maxz_bootstrap()

    metadata = dict(getattr(base_result, "metadata", {}) or {})
    if sample_weighted:
        metadata["backend_path"] = f"{backend_name}_debiased_weighted"
    metadata.update(
        {
            "sample_weighted": bool(sample_weighted),
            "numerical_backend": backend_name,
            "numerical_device": getattr(model, "_selected_backend_device", None),
            "reporting_backend": "numpy",
            "reporting_boundary": "post_numerical_inference",
        }
    )
    result = DebiasedInferenceResult(
        method="debiased",
        feature_names=model._inference_feature_names(),
        params=params,
        bse=bse,
        statistic=statistic,
        statistic_name="z",
        pvalues=pvalues,
        conf_int=conf_int,
        distribution="normal",
        precision_method=getattr(base_result, "precision_method", "nodewise_lasso"),
        metadata=metadata,
        simultaneous_conf_int=getattr(model, "_conf_int_simultaneous", None),
        simultaneous_method=getattr(model, "simultaneous_method", None),
        simultaneous_alpha=getattr(model, "simultaneous_alpha", None),
        simultaneous_n_bootstrap=getattr(model, "simultaneous_n_bootstrap", None),
        simultaneous_critical_value=getattr(model, "_simultaneous_critical_value", None),
        simultaneous_target_mask=getattr(model, "_simultaneous_target_mask", None),
    )
    result.apply_to(model)
    return result


def _review_fit_gpu_backend(self, X, y, sample_weight=None, backend_name="cupy"):
    """Shared weighted sparse GPU fit with strict inference/backend preservation."""
    if not _is_weighted_sparse_fast_path(self, sample_weight, backend_name):
        return _BASE_GPU_FIT(self, X, y, sample_weight, backend_name=backend_name)

    xp = _get_xp(backend_name)
    X_arr = xp_asarray(X, dtype=np.float64, xp=xp, ref_arr=X)
    y_arr = xp_asarray(y, dtype=np.float64, xp=xp, ref_arr=y).reshape(-1)
    sw_arr = xp_asarray(
        sample_weight,
        dtype=X_arr.dtype,
        xp=xp,
        ref_arr=X_arr,
    ).reshape(-1)
    n_samples, n_features = X_arr.shape
    n_eff = _validate_sample_weight_backend(
        sw_arr,
        n_samples,
        backend_name,
    )
    original_intercept = bool(self._effective_intercept)
    X_work, y_work, X_mean, y_mean = (
        PenalizedLinearRegression._weighted_sparse_gpu_working_data(
            X_arr,
            y_arr,
            sw_arr,
            fit_intercept=original_intercept,
            xp=xp,
            n_eff=n_eff,
        )
    )
    row_scale = xp.sqrt(sw_arr * (float(n_samples) / float(n_eff)))

    saved_use_intercept = self._use_intercept
    saved_compute_inference = self._compute_inference_enabled
    runtime_method = _runtime_inference_method(self)
    run_debiased = saved_compute_inference and "debiased" in runtime_method
    simultaneous_attr = hasattr(self, "enable_simultaneous_inference")
    saved_simultaneous = (
        bool(getattr(self, "enable_simultaneous_inference", False))
        if simultaneous_attr
        else False
    )
    cache_sentinel = object()
    saved_cv_cache = getattr(self, "_cv_cache", cache_sentinel)

    self._use_intercept = False
    if run_debiased and simultaneous_attr:
        self.enable_simultaneous_inference = False
    if saved_cv_cache is not cache_sentinel:
        del self._cv_cache
    try:
        _BASE_GPU_FIT(
            self,
            X_work,
            y_work,
            None,
            backend_name=backend_name,
        )
    finally:
        self._use_intercept = saved_use_intercept
        self._compute_inference_enabled = saved_compute_inference
        if simultaneous_attr:
            self.enable_simultaneous_inference = saved_simultaneous
        if saved_cv_cache is not cache_sentinel:
            self._cv_cache = saved_cv_cache

    coef = np.asarray(self.coef_, dtype=np.float64)
    self.coef_ = coef
    if original_intercept:
        X_mean_np = np.asarray(_to_numpy(X_mean), dtype=np.float64)
        y_mean_value = float(np.asarray(_to_numpy(y_mean), dtype=np.float64))
        self.intercept_ = float(y_mean_value - X_mean_np @ coef)
        self._params = np.concatenate([[self.intercept_], coef])
    else:
        self.intercept_ = 0.0
        self._params = coef.copy()
    self._nobs = int(n_samples)
    self._df_resid = int(n_samples - (n_features + int(original_intercept)))

    if run_debiased:
        coef_native = xp_asarray(
            coef,
            dtype=X_arr.dtype,
            xp=xp,
            ref_arr=X_arr,
        )
        _finalize_weighted_debiased_result(
            self,
            X_arr=X_arr,
            y_work=y_work,
            X_work=X_work,
            row_scale=row_scale,
            coef_native=coef_native,
            backend_name=backend_name,
            original_intercept=original_intercept,
            simultaneous_requested=saved_simultaneous,
        )
    return None


def install_post_selection_ols_review_fix_contract():
    """Install the post-acceptance review fixes once."""
    current = PenalizedGeneralizedLinearModel._fit_gpu_backend
    if getattr(current, _REVIEW_FIX_MARKER, False):
        return

    _inference_contract._install_constructor_contract(
        PenalizedGeneralizedLinearModel
    )
    _inference_contract._install_fit_device_contract(
        PenalizedGeneralizedLinearModel
    )

    wrapped = functools.wraps(_BASE_GPU_FIT)(_review_fit_gpu_backend)
    setattr(wrapped, _REVIEW_FIX_MARKER, True)
    PenalizedGeneralizedLinearModel._fit_gpu_backend = wrapped
    PenalizedLinearRegression._fit_gpu_backend = wrapped


__all__ = ["install_post_selection_ols_review_fix_contract"]
