"""Coherent original-coordinate intercept inference for centered debiased Lasso.

The sparse-Gaussian debiased paths estimate feature coefficients on a centered
working design. A reported intercept must therefore be transformed with the
same debiased feature vector. Publishing the penalized-fit intercept next to
debiased slopes breaks the elementary feature-translation identity and gives
intercept marginal/simultaneous inference a different parameterization.

This contract keeps ``coef_``/``intercept_`` owned by the penalized prediction
fit, while inference reporting uses

    intercept_db = intercept_pen - xbar_w @ (theta_db - beta_pen).

The intercept influence is derived from the same centered nodewise precision
matrix ``M`` used by the debiased slopes. CuPy/Torch retain the local native
``M`` and ``theta_db`` only through the reporting finalizer, so intercept
numerics stay on the executed backend before the established NumPy reporting
snapshot. Scalar reference-distribution calls made by the maintained GPU
implementation are routed through the same context-local backend/device rather
than letting a Python scalar re-resolve them to NumPy.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import functools

import numpy as np

from statgpu.backends import _to_numpy
from statgpu.backends._utils import _get_xp, xp_asarray
import statgpu.inference._distributions_backend as _distribution_module
from statgpu.linear_model import (
    _post_selection_ols_review_fix_contract as _weighted_contract,
)
from statgpu.linear_model.penalized._base import PenalizedGeneralizedLinearModel
from statgpu.linear_model.penalized._penalized_linear import PenalizedLinearRegression


_STATS_MARKER = "__statgpu_pr138_debiased_native_stats_capture__"
_GPU_MARKER = "__statgpu_pr138_debiased_native_gpu_capture__"
_FIT_MARKER = "__statgpu_pr138_debiased_native_fit_lifetime__"
_FINALIZER_MARKER = "__statgpu_pr138_debiased_intercept_parameterization__"
_DISTRIBUTION_MARKER = "__statgpu_pr138_debiased_distribution_backend__"
_CAPTURE_OWNER = ContextVar(
    "statgpu_pr138_debiased_capture_owner",
    default=None,
)
_DISTRIBUTION_ROUTE = ContextVar(
    "statgpu_pr138_debiased_distribution_route",
    default=None,
)

_ORIGINAL_STATS = PenalizedGeneralizedLinearModel._debiased_stats_from_M
_ORIGINAL_FINALIZER = _weighted_contract._finalize_weighted_debiased_result
_ORIGINAL_DISTRIBUTION_RESOLVE = _distribution_module.DistributionProxy._resolve

_NATIVE_M = "_statgpu_debiased_M_native_work"
_NATIVE_THETA = "_statgpu_debiased_theta_native_work"
_DEFER_CLEAR = "_statgpu_debiased_native_defer_clear"
_MISSING = object()


def _clear_native_work(estimator) -> None:
    estimator.__dict__.pop(_NATIVE_M, None)
    estimator.__dict__.pop(_NATIVE_THETA, None)


@functools.wraps(_ORIGINAL_DISTRIBUTION_RESOLVE)
def _distribution_resolve(self, kwargs, *arrays):
    """Honor the active debiased backend for scalar AUTO distribution calls."""
    route = _DISTRIBUTION_ROUTE.get()
    if route is not None and "backend" not in kwargs:
        backend_name, device_name = route
        kwargs["backend"] = backend_name
        if backend_name == "torch" and device_name is not None and "device" not in kwargs:
            kwargs["device"] = device_name
    return _ORIGINAL_DISTRIBUTION_RESOLVE(self, kwargs, *arrays)


@contextmanager
def _distribution_backend_context(backend_name: str, ref_arr):
    """Bind scalar distribution calls to the executed debiased backend/device."""
    backend_name = str(backend_name).strip().lower()
    device_name = None
    if backend_name == "torch":
        device_name = str(ref_arr.device)
    token = _DISTRIBUTION_ROUTE.set((backend_name, device_name))
    try:
        if backend_name == "cupy":
            import cupy as cp

            with cp.cuda.Device(int(ref_arr.device.id)):
                yield
        else:
            yield
    finally:
        _DISTRIBUTION_ROUTE.reset(token)


@functools.wraps(_ORIGINAL_STATS)
def _debiased_stats_from_M(
    M,
    Sigma_hat,
    sigma2,
    coef,
    X,
    y,
    intercept,
    fit_intercept,
    n,
    xp,
    arr_norm,
):
    """Capture native precision/feature estimates while GPU debiasing is active."""
    result = _ORIGINAL_STATS(
        M,
        Sigma_hat,
        sigma2,
        coef,
        X,
        y,
        intercept,
        fit_intercept,
        n,
        xp,
        arr_norm,
    )
    owner = _CAPTURE_OWNER.get()
    backend_name = str(getattr(xp, "__name__", "")).lower()
    if owner is not None and backend_name in {"cupy", "torch"}:
        setattr(owner, _NATIVE_M, M)
        setattr(owner, _NATIVE_THETA, result[0])
    return result


def _capture_native_gpu_call(current, backend_name: str):
    @functools.wraps(current)
    def wrapped(self, *args, **kwargs):
        ref_arr = args[0] if args else kwargs.get("X_gpu", kwargs.get("X_torch"))
        if ref_arr is None:
            raise RuntimeError(
                "debiased GPU inference requires a concrete design array for backend routing"
            )
        owner_token = _CAPTURE_OWNER.set(self)
        try:
            with _distribution_backend_context(backend_name, ref_arr):
                return current(self, *args, **kwargs)
        finally:
            _CAPTURE_OWNER.reset(owner_token)
            if not bool(getattr(self, _DEFER_CLEAR, False)):
                _clear_native_work(self)

    setattr(wrapped, _GPU_MARKER, True)
    wrapped._statgpu_original = current
    return wrapped


def _wrap_fit_gpu_backend(current):
    @functools.wraps(current)
    def wrapped(self, X, y, sample_weight=None, backend_name="cupy"):
        method = str(
            getattr(
                self,
                "_inference_method",
                getattr(self, "inference_method", ""),
            )
        ).strip().lower()
        defer = bool(
            sample_weight is not None
            and bool(getattr(self, "_compute_inference_enabled", False))
            and "debiased" in method
            and bool(getattr(self, "_effective_intercept", False))
        )
        previous = self.__dict__.get(_DEFER_CLEAR, _MISSING)
        if defer:
            setattr(self, _DEFER_CLEAR, True)
        try:
            return current(
                self,
                X,
                y,
                sample_weight,
                backend_name=backend_name,
            )
        finally:
            if previous is _MISSING:
                self.__dict__.pop(_DEFER_CLEAR, None)
            else:
                setattr(self, _DEFER_CLEAR, previous)
            _clear_native_work(self)

    setattr(wrapped, _FIT_MARKER, True)
    wrapped._statgpu_original = current
    return wrapped


def _native_debiased_state(model, backend_name: str, *, X_work, feature_params):
    if backend_name == "numpy":
        M_native = np.asarray(model._debiased_M_cpu, dtype=np.float64)
        theta_native = np.asarray(feature_params, dtype=np.float64).reshape(-1)
        return M_native, theta_native

    M_native = getattr(model, _NATIVE_M, None)
    theta_native = getattr(model, _NATIVE_THETA, None)
    if M_native is None or theta_native is None:
        raise RuntimeError(
            "backend-native debiased intercept inference is missing the native "
            "nodewise precision/feature estimate; refusing a host-side fallback"
        )
    return M_native, theta_native


def _native_intercept_report(
    z_native,
    se_native,
    intercept_value: float,
    backend_name: str,
    ref_arr,
):
    """Run intercept reference-distribution work on the executed array device."""
    with _distribution_backend_context(backend_name, ref_arr):
        return _weighted_contract._normal_intercept_report(
            z_native,
            se_native,
            intercept_value,
            backend_name,
            ref_arr,
        )


def _validate_finite_intercept_report(
    intercept_value: float,
    se_intercept: float,
    p_intercept: float,
    ci_intercept,
) -> np.ndarray:
    """Fail closed before publishing a non-representable intercept report."""
    ci = np.asarray(ci_intercept, dtype=np.float64).reshape(2)
    values = np.asarray(
        [intercept_value, se_intercept, p_intercept, ci[0], ci[1]],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise FloatingPointError(
            "debiased intercept inference produced non-finite estimate, standard "
            "error, p-value, or confidence interval"
        )
    return ci


def _publish_coherent_intercept(
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
    """Insert one coherent debiased intercept into a feature-only result."""
    xp = _get_xp(backend_name)
    n = int(X_work.shape[0])
    p = int(X_work.shape[1])
    feature_params = np.asarray(result.params, dtype=np.float64).reshape(-1)
    if feature_params.shape[0] != p:
        raise RuntimeError(
            "debiased feature result does not match the centered working design"
        )

    M_native, theta_native = _native_debiased_state(
        model,
        backend_name,
        X_work=X_work,
        feature_params=feature_params,
    )
    if tuple(M_native.shape) != (p, p):
        raise RuntimeError(
            "debiased precision matrix does not match the centered working design"
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
    if backend_name != "numpy":
        theta_native = xp_asarray(
            theta_native,
            dtype=X_work.dtype,
            xp=xp,
            ref_arr=X_work,
        ).reshape(-1)

    # row_scale**2 = w * n / sum(w), so a vector-matrix reduction divided by n
    # recovers xbar_w without materializing an n-by-p weighted design temporary.
    row_weight = row_scale * row_scale
    x_mean = (row_weight @ X_arr) / float(n)
    intercept_pen_native = xp_asarray(
        [float(model.intercept_)],
        dtype=X_work.dtype,
        xp=xp,
        ref_arr=X_work,
    ).reshape(-1)[0]
    intercept_native = intercept_pen_native - x_mean @ (
        theta_native - coef_native
    )

    # theta_db = beta_pen + M X_work' r_work / n. Combining this score with
    # ybar_w - xbar_w theta_db gives q_i/n as the original-coordinate
    # intercept influence against the centered working residual r_work_i.
    q_native = row_scale - X_work @ (M_native.T @ x_mean)
    influence_native = q_native / float(max(n, 1))

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
        xp.abs(scale_native * xp.sum(q_native * q_native))
    ) / float(max(n, 1))
    z_native = intercept_native / (se_native + 1e-30)

    intercept_value = float(
        np.asarray(_to_numpy(intercept_native), dtype=np.float64)
    )
    p_intercept, ci_intercept = _native_intercept_report(
        z_native,
        se_native,
        intercept_value,
        backend_name,
        X_work,
    )
    se_intercept = float(np.asarray(_to_numpy(se_native), dtype=np.float64))
    z_intercept = float(np.asarray(_to_numpy(z_native), dtype=np.float64))
    ci_intercept = _validate_finite_intercept_report(
        intercept_value,
        se_intercept,
        p_intercept,
        ci_intercept,
    )

    feature_bse = np.asarray(result.bse, dtype=np.float64).reshape(-1)
    feature_statistic = np.asarray(result.statistic, dtype=np.float64).reshape(-1)
    feature_pvalues = np.asarray(result.pvalues, dtype=np.float64).reshape(-1)
    feature_conf_int = np.asarray(result.conf_int, dtype=np.float64)
    if (
        feature_bse.shape[0] != p
        or feature_statistic.shape[0] != p
        or feature_pvalues.shape[0] != p
        or feature_conf_int.shape != (p, 2)
    ):
        raise RuntimeError(
            "debiased feature inference layout does not match the centered design"
        )

    params = np.concatenate([[intercept_value], feature_params])
    bse = np.concatenate([[se_intercept], feature_bse])
    statistic = np.concatenate([[z_intercept], feature_statistic])
    pvalues = np.concatenate([[p_intercept], feature_pvalues])
    conf_int = np.vstack([
        ci_intercept.reshape(1, 2),
        feature_conf_int,
    ])

    include_intercept_simultaneous = bool(
        simultaneous_requested
        and getattr(
            model,
            "simultaneous_include_intercept",
            getattr(model, "_simultaneous_include_intercept", False),
        )
    )
    if include_intercept_simultaneous:
        model._debiased_intercept_influence_cpu = np.asarray(
            _to_numpy(influence_native),
            dtype=np.float64,
        ).reshape(-1)
    else:
        model.__dict__.pop("_debiased_intercept_influence_cpu", None)

    # The feature-only finalizer has already created the reporting snapshot of
    # X_work. Reuse it and transfer only the n-vector row_scale; copying X_work
    # again would duplicate the largest GPU-to-host reporting transfer.
    feature_design_cpu = np.asarray(model._X_design, dtype=np.float64)
    if feature_design_cpu.shape != (n, p):
        raise RuntimeError(
            "debiased feature reporting design does not match the centered work design"
        )
    model._X_design = np.column_stack([
        np.asarray(_to_numpy(row_scale), dtype=np.float64).reshape(-1),
        feature_design_cpu,
    ])
    model._df_resid = n - (p + 1)
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
        target_mask = getattr(model, "_simultaneous_target_mask", None)
        result.simultaneous_target_mask = (
            None
            if target_mask is None
            else np.asarray(target_mask, dtype=bool).copy()
        )

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
    """Finalize debiased reporting with one coherent intercept parameterization."""
    if not original_intercept:
        model.__dict__.pop("_debiased_intercept_influence_cpu", None)
        return _ORIGINAL_FINALIZER(
            model,
            X_arr=X_arr,
            y_work=y_work,
            X_work=X_work,
            row_scale=row_scale,
            coef_native=coef_native,
            backend_name=backend_name,
            original_intercept=False,
            simultaneous_requested=simultaneous_requested,
            sample_weighted=sample_weighted,
        )

    try:
        # The maintained finalizer owns feature debiasing/reporting. Ask it for
        # the feature-only layout so the historical penalized-intercept/raw-OLS
        # calculation never runs; then insert the coherent intercept exactly once.
        result = _ORIGINAL_FINALIZER(
            model,
            X_arr=X_arr,
            y_work=y_work,
            X_work=X_work,
            row_scale=row_scale,
            coef_native=coef_native,
            backend_name=backend_name,
            original_intercept=False,
            simultaneous_requested=False,
            sample_weighted=sample_weighted,
        )
        return _publish_coherent_intercept(
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
    current_resolve = _distribution_module.DistributionProxy._resolve
    if not getattr(current_resolve, _DISTRIBUTION_MARKER, False):
        setattr(_distribution_resolve, _DISTRIBUTION_MARKER, True)
        _distribution_module.DistributionProxy._resolve = _distribution_resolve

    current_stats = PenalizedGeneralizedLinearModel._debiased_stats_from_M
    if not getattr(current_stats, _STATS_MARKER, False):
        setattr(_debiased_stats_from_M, _STATS_MARKER, True)
        PenalizedGeneralizedLinearModel._debiased_stats_from_M = staticmethod(
            _debiased_stats_from_M
        )

    current_gpu = PenalizedGeneralizedLinearModel._compute_inference_debiased_gpu
    if not getattr(current_gpu, _GPU_MARKER, False):
        PenalizedGeneralizedLinearModel._compute_inference_debiased_gpu = (
            _capture_native_gpu_call(current_gpu, "cupy")
        )
        PenalizedGeneralizedLinearModel._compute_inference_debiased_torch = (
            _capture_native_gpu_call(
                PenalizedGeneralizedLinearModel._compute_inference_debiased_torch,
                "torch",
            )
        )

    for cls in (PenalizedGeneralizedLinearModel, PenalizedLinearRegression):
        current_fit = cls.__dict__.get("_fit_gpu_backend")
        if current_fit is None:
            current_fit = getattr(cls, "_fit_gpu_backend")
        if not getattr(current_fit, _FIT_MARKER, False):
            cls._fit_gpu_backend = _wrap_fit_gpu_backend(current_fit)

    current_finalizer = _weighted_contract._finalize_weighted_debiased_result
    if not getattr(current_finalizer, _FINALIZER_MARKER, False):
        setattr(_finalize_weighted_debiased_result, _FINALIZER_MARKER, True)
        _weighted_contract._finalize_weighted_debiased_result = (
            _finalize_weighted_debiased_result
        )


__all__ = ["install_debiased_intercept_parameterization_contract"]
