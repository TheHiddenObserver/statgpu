"""Coherent original-coordinate intercept inference for centered debiased Lasso.

The sparse-Gaussian debiased paths estimate feature coefficients on a centered
working design.  A reported intercept must therefore be transformed with the
same debiased feature vector.  Publishing the penalized-fit intercept next to
debiased slopes breaks the elementary feature-translation identity and gives
intercept marginal/simultaneous inference a different parameterization.

This contract keeps ``coef_``/``intercept_`` owned by the penalized prediction
fit, while the inference result reports

    intercept_db = intercept_pen - xbar_w @ (theta_db - beta_pen).

Its influence is derived from the same centered nodewise precision ``M`` used by
the debiased slopes.  CuPy/Torch keep the local native ``M`` and ``theta_db``
only until the reporting finalizer consumes them, then release those references.
"""

from __future__ import annotations

from contextvars import ContextVar
import functools

import numpy as np

from statgpu.backends import _to_numpy
from statgpu.backends._utils import _get_xp, xp_asarray
from statgpu.linear_model import (
    _post_selection_ols_review_fix_contract as _weighted_contract,
)
from statgpu.linear_model.penalized._base import PenalizedGeneralizedLinearModel


_STATS_MARKER = "__statgpu_pr138_debiased_native_stats_capture__"
_GPU_MARKER = "__statgpu_pr138_debiased_native_gpu_capture__"
_FIT_MARKER = "__statgpu_pr138_debiased_native_fit_lifetime__"
_FINALIZER_MARKER = "__statgpu_pr138_debiased_intercept_parameterization__"
_CAPTURE_OWNER: ContextVar[object | None] = ContextVar(
    "statgpu_pr138_debiased_capture_owner",
    default=None,
)

_ORIGINAL_STATS = PenalizedGeneralizedLinearModel._debiased_stats_from_M
_ORIGINAL_GPU = PenalizedGeneralizedLinearModel._compute_inference_debiased_gpu
_ORIGINAL_TORCH = PenalizedGeneralizedLinearModel._compute_inference_debiased_torch
_ORIGINAL_FIT_GPU = PenalizedGeneralizedLinearModel._fit_gpu_backend
_ORIGINAL_FINALIZER = _weighted_contract._finalize_weighted_debiased_result

_NATIVE_M = "_statgpu_debiased_M_native_work"
_NATIVE_THETA = "_statgpu_debiased_theta_native_work"
_DEFER_CLEAR = "_statgpu_debiased_native_defer_clear"


def _clear_native_work(estimator) -> None:
    estimator.__dict__.pop(_NATIVE_M, None)
    estimator.__dict__.pop(_NATIVE_THETA, None)


@functools.wraps(_ORIGINAL_STATS)
def _debiased_stats_from_M(X, resid, beta_hat, M, backend_name):
    result = _ORIGINAL_STATS(X, resid, beta_hat, M, backend_name)
    owner = _CAPTURE_OWNER.get()
    if owner is not None and backend_name in {"cupy", "torch"}:
        setattr(owner, _NATIVE_M, M)
        setattr(owner, _NATIVE_THETA, result[0])
    return result


def _capture_native_gpu_call(current):
    @functools.wraps(current)
    def wrapped(self, *args, **kwargs):
        token = _CAPTURE_OWNER.set(self)
        try:
            return current(self, *args, **kwargs)
        finally:
            _CAPTURE_OWNER.reset(token)
            if not bool(getattr(self, _DEFER_CLEAR, False)):
                _clear_native_work(self)

    setattr(wrapped, _GPU_MARKER, True)
    return wrapped


@functools.wraps(_ORIGINAL_FIT_GPU)
def _fit_gpu_backend(self, X, y, sample_weight=None, backend_name="cupy"):
    """Keep weighted native debiasing state alive through the outer finalizer."""
    defer = sample_weight is not None
    previous = self.__dict__.get(_DEFER_CLEAR, None)
    if defer:
        setattr(self, _DEFER_CLEAR, True)
    try:
        return _ORIGINAL_FIT_GPU(
            self,
            X,
            y,
            sample_weight,
            backend_name=backend_name,
        )
    finally:
        if defer:
            if previous is None:
                self.__dict__.pop(_DEFER_CLEAR, None)
            else:
                setattr(self, _DEFER_CLEAR, previous)
        _clear_native_work(self)


def _native_debiased_state(model, backend_name: str, *, X_work, params_feature):
    xp = _get_xp(backend_name)
    if backend_name == "numpy":
        M_native = xp_asarray(
            model._debiased_M_cpu,
            dtype=np.float64,
            xp=xp,
            ref_arr=X_work,
        )
        theta_native = xp_asarray(
            params_feature,
            dtype=np.float64,
            xp=xp,
            ref_arr=X_work,
        ).reshape(-1)
        return M_native, theta_native

    M_native = getattr(model, _NATIVE_M, None)
    theta_native = getattr(model, _NATIVE_THETA, None)
    if M_native is None or theta_native is None:
        raise RuntimeError(
            "backend-native debiased intercept inference is missing the native "
            "nodewise precision/feature estimate; refusing a host-side fallback"
        )
    return M_native, theta_native


def _correct_intercept_result(
    model,
    result,
    *,
    X_arr,
    y_work,
    X_work,
    row_scale,
    coef_native,
    backend_name: str,
    simultaneous_requested: bool,
):
    xp = _get_xp(backend_name)
    n = int(X_work.shape[0])
    feature_params = np.asarray(result.params, dtype=np.float64).reshape(-1)[1:]
    M_native, theta_native = _native_debiased_state(
        model,
        backend_name,
        X_work=X_work,
        params_feature=feature_params,
    )
    coef_native = xp_asarray(
        coef_native,
        dtype=X_work.dtype,
        xp=xp,
        ref_arr=X_work,
    ).reshape(-1)
    row_scale = xp_asarray(
        row_scale,
        dtype=X_work.dtype,
        xp=xp,
        ref_arr=X_work,
    ).reshape(-1)

    # row_scale^2 = w * n / sum(w), so its sum is n and this is the
    # unweighted/weighted original-design mean in one expression.
    x_mean = xp.sum(X_arr * (row_scale * row_scale).reshape(-1, 1), axis=0) / float(n)
    penalized_intercept_native = xp_asarray(
        [float(model.intercept_)],
        dtype=X_work.dtype,
        xp=xp,
        ref_arr=X_work,
    ).reshape(-1)[0]
    intercept_native = penalized_intercept_native - x_mean @ (
        theta_native.reshape(-1) - coef_native
    )

    # theta_db = beta_pen + M X_work' r_work / n. Combining that score with
    # ybar_w - xbar_w theta_db gives the original-coordinate intercept score.
    q_native = row_scale - X_work @ (M_native.T @ x_mean)
    influence_native = q_native / float(n)

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
    se_native = xp.sqrt(
        xp.abs(
            scale_native
            * xp.sum(q_native * q_native)
            / float(max(n * n, 1))
        )
    )
    z_native = intercept_native / (se_native + 1e-30)
    p_intercept, ci_intercept = _weighted_contract._normal_intercept_report(
        z_native,
        se_native,
        intercept_native,
        backend_name,
        X_work,
    )

    intercept = float(np.asarray(_to_numpy(intercept_native), dtype=np.float64))
    se_intercept = float(np.asarray(_to_numpy(se_native), dtype=np.float64))
    z_intercept = float(np.asarray(_to_numpy(z_native), dtype=np.float64))

    params = np.asarray(result.params, dtype=np.float64).copy()
    bse = np.asarray(result.bse, dtype=np.float64).copy()
    statistic = np.asarray(result.statistic, dtype=np.float64).copy()
    pvalues = np.asarray(result.pvalues, dtype=np.float64).copy()
    conf_int = np.asarray(result.conf_int, dtype=np.float64).copy()
    params[0] = intercept
    bse[0] = se_intercept
    statistic[0] = z_intercept
    pvalues[0] = p_intercept
    conf_int[0] = np.asarray(ci_intercept, dtype=np.float64).reshape(2)

    model._debiased_intercept_influence_cpu = np.asarray(
        _to_numpy(influence_native),
        dtype=np.float64,
    ).reshape(-1)
    model._params = params
    model._bse = bse
    model._tvalues = statistic
    model._zvalues = statistic
    model._pvalues = pvalues
    model._conf_int = conf_int

    result.params = params
    result.bse = bse
    result.statistic = statistic
    result.pvalues = pvalues
    result.conf_int = conf_int
    metadata = dict(getattr(result, "metadata", {}) or {})
    metadata.update(
        {
            "intercept_estimator": "centered_debiased",
            "intercept_influence": "centered_nodewise",
        }
    )
    result.metadata = metadata

    if simultaneous_requested:
        model._compute_simultaneous_ci_maxz_bootstrap()
        result.simultaneous_conf_int = np.asarray(
            model._conf_int_simultaneous,
            dtype=np.float64,
        ).copy()
        result.simultaneous_method = getattr(model, "simultaneous_method", None)
        result.simultaneous_alpha = getattr(model, "simultaneous_alpha", None)
        result.simultaneous_n_bootstrap = getattr(
            model,
            "simultaneous_n_bootstrap",
            None,
        )
        result.simultaneous_critical_value = getattr(
            model,
            "_simultaneous_critical_value",
            None,
        )
        result.simultaneous_target_mask = np.asarray(
            model._simultaneous_target_mask,
            dtype=bool,
        ).copy()

    result.apply_to(model)
    return result


@functools.wraps(_ORIGINAL_FINALIZER)
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
    # Suppress the older intercept-inclusive simultaneous call until the
    # coherent intercept parameter/influence below has replaced the historical
    # penalized-intercept reporting slot.
    result = _ORIGINAL_FINALIZER(
        model,
        X_arr=X_arr,
        y_work=y_work,
        X_work=X_work,
        row_scale=row_scale,
        coef_native=coef_native,
        backend_name=backend_name,
        original_intercept=original_intercept,
        simultaneous_requested=(simultaneous_requested and not original_intercept),
        sample_weighted=sample_weighted,
    )
    try:
        if not original_intercept:
            return result
        return _correct_intercept_result(
            model,
            result,
            X_arr=X_arr,
            y_work=y_work,
            X_work=X_work,
            row_scale=row_scale,
            coef_native=coef_native,
            backend_name=backend_name,
            simultaneous_requested=simultaneous_requested,
        )
    finally:
        _clear_native_work(model)


def install_debiased_intercept_parameterization_contract() -> None:
    current_stats = PenalizedGeneralizedLinearModel._debiased_stats_from_M
    if not getattr(current_stats, _STATS_MARKER, False):
        setattr(_debiased_stats_from_M, _STATS_MARKER, True)
        PenalizedGeneralizedLinearModel._debiased_stats_from_M = staticmethod(
            _debiased_stats_from_M
        )

    current_gpu = PenalizedGeneralizedLinearModel._compute_inference_debiased_gpu
    if not getattr(current_gpu, _GPU_MARKER, False):
        PenalizedGeneralizedLinearModel._compute_inference_debiased_gpu = (
            _capture_native_gpu_call(current_gpu)
        )
        PenalizedGeneralizedLinearModel._compute_inference_debiased_torch = (
            _capture_native_gpu_call(
                PenalizedGeneralizedLinearModel._compute_inference_debiased_torch
            )
        )

    current_fit = PenalizedGeneralizedLinearModel._fit_gpu_backend
    if not getattr(current_fit, _FIT_MARKER, False):
        setattr(_fit_gpu_backend, _FIT_MARKER, True)
        PenalizedGeneralizedLinearModel._fit_gpu_backend = _fit_gpu_backend

    current_finalizer = _weighted_contract._finalize_weighted_debiased_result
    if not getattr(current_finalizer, _FINALIZER_MARKER, False):
        setattr(_finalize_weighted_debiased_result, _FINALIZER_MARKER, True)
        _weighted_contract._finalize_weighted_debiased_result = (
            _finalize_weighted_debiased_result
        )


__all__ = ["install_debiased_intercept_parameterization_contract"]
