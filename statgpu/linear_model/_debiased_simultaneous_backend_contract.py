"""Backend-native simultaneous inference for centered debiased sparse Gaussian fits.

PR #138 centers sparse-Gaussian debiased inference when an intercept is fitted.
The maintained CuPy/Torch debiased implementations historically snapshotted the
working design/residual to NumPy before invoking the generic max-|Z| bootstrap.
That is not an acceptable execution path once the centered/intercept-capable
simultaneous procedure is part of the explicit GPU closure.

This focused contract intercepts only the centered fit-intercept finalizer.  The
marginal debiased calculation remains owned by the existing implementation; when
simultaneous inference is requested on CuPy/Torch, multiplier scores, max-|Z|,
quantile calibration, and target confidence intervals are computed on the same
concrete device before the established NumPy reporting snapshot is published.
CPU behavior and historical no-intercept simultaneous behavior are unchanged.
"""

from __future__ import annotations

import functools

import numpy as np

from statgpu.backends import _to_numpy
from statgpu.backends._utils import _get_xp, xp_asarray
from statgpu.linear_model import (
    _debiased_intercept_parameterization_contract as _intercept_contract,
)
from statgpu.linear_model import (
    _post_selection_ols_review_fix_contract as _weighted_contract,
)


_FINALIZER_MARKER = "__statgpu_pr138_debiased_simultaneous_backend__"
_ORIGINAL_FINALIZER = _weighted_contract._finalize_weighted_debiased_result


def _device_name(backend_name: str, ref_arr) -> str:
    if backend_name == "torch":
        return str(ref_arr.device)
    if backend_name == "cupy":
        return f"cuda:{int(ref_arr.device.id)}"
    return "cpu"


def _random_normal(backend_name: str, *, shape, ref_arr, rng):
    xp = _get_xp(backend_name)
    if backend_name == "torch":
        import torch

        kwargs = {
            "dtype": ref_arr.dtype,
            "device": ref_arr.device,
        }
        if rng is not None:
            kwargs["generator"] = rng
        return torch.randn(shape, **kwargs)
    values = rng.standard_normal(size=shape)
    return values.astype(ref_arr.dtype, copy=False)


def _make_rng(backend_name: str, *, random_state, ref_arr):
    if backend_name == "torch":
        import torch

        if random_state is None:
            return None
        generator = torch.Generator(device=ref_arr.device)
        generator.manual_seed(int(random_state))
        return generator

    import cupy as cp

    return cp.random.RandomState(random_state)


def _row_max_abs(values, backend_name: str):
    xp = _get_xp(backend_name)
    if backend_name == "torch":
        return xp.amax(xp.abs(values), dim=1)
    return xp.max(xp.abs(values), axis=1)


def _native_simultaneous_maxz(
    model,
    result,
    *,
    X_arr,
    y_work,
    X_work,
    row_scale,
    coef_native,
    M_native,
    backend_name: str,
):
    """Compute centered max-|Z| inference on one concrete GPU backend."""
    xp = _get_xp(backend_name)
    n = int(X_work.shape[0])
    p = int(X_work.shape[1])
    if p <= 0:
        raise RuntimeError("simultaneous debiased inference requires at least one feature")

    X_arr = xp_asarray(
        X_arr,
        dtype=X_work.dtype,
        xp=xp,
        ref_arr=X_work,
    )
    row_scale = xp_asarray(
        row_scale,
        dtype=X_work.dtype,
        xp=xp,
        ref_arr=X_work,
    ).reshape(-1)
    coef_native = xp_asarray(
        coef_native,
        dtype=X_work.dtype,
        xp=xp,
        ref_arr=X_work,
    ).reshape(-1)
    M_native = xp_asarray(
        M_native,
        dtype=X_work.dtype,
        xp=xp,
        ref_arr=X_work,
    )
    if tuple(M_native.shape) != (p, p):
        raise RuntimeError(
            "debiased precision matrix does not match the simultaneous working design"
        )

    params_np = np.asarray(result.params, dtype=np.float64).reshape(-1)
    bse_np = np.asarray(result.bse, dtype=np.float64).reshape(-1)
    if params_np.shape[0] != p + 1 or bse_np.shape[0] != p + 1:
        raise RuntimeError(
            "centered simultaneous inference requires intercept plus feature reporting rows"
        )
    params_native = xp_asarray(
        params_np,
        dtype=X_work.dtype,
        xp=xp,
        ref_arr=X_work,
    )
    bse_native = xp_asarray(
        bse_np,
        dtype=X_work.dtype,
        xp=xp,
        ref_arr=X_work,
    )

    resid_work = y_work - X_work @ coef_native
    row_weight = row_scale * row_scale
    x_mean = (row_weight @ X_arr) / float(n)
    q_native = row_scale - X_work @ (M_native.T @ x_mean)
    influence_native = q_native / float(max(n, 1))

    B = int(
        getattr(
            model,
            "simultaneous_n_bootstrap",
            getattr(model, "_simultaneous_n_bootstrap", 1000),
        )
    )
    if B <= 0:
        raise ValueError("simultaneous_n_bootstrap must be positive")
    alpha = float(
        getattr(
            model,
            "simultaneous_alpha",
            getattr(model, "_simultaneous_alpha", 0.05),
        )
    )
    if not (0.0 < alpha < 1.0):
        raise ValueError("simultaneous_alpha must lie strictly between 0 and 1")
    random_state = getattr(
        model,
        "simultaneous_random_state",
        getattr(model, "_simultaneous_random_state", None),
    )
    include_intercept = bool(
        getattr(
            model,
            "simultaneous_include_intercept",
            getattr(model, "_simultaneous_include_intercept", False),
        )
    )

    rng = _make_rng(
        backend_name,
        random_state=random_state,
        ref_arr=X_work,
    )
    max_stats = xp.empty((B,), dtype=X_work.dtype, device=X_work.device) if backend_name == "torch" else xp.empty((B,), dtype=X_work.dtype)
    chunk = min(256, B)
    filled = 0
    while filled < B:
        bsz = min(chunk, B - filled)
        xi = _random_normal(
            backend_name,
            shape=(bsz, n),
            ref_arr=X_work,
            rng=rng,
        )
        multiplier_resid = xi * resid_work.reshape(1, -1)
        feature_score = (multiplier_resid @ X_work) @ M_native.T / float(max(n, 1))
        z_feature = feature_score / (bse_native[1:].reshape(1, -1) + 1e-30)
        batch_max = _row_max_abs(z_feature, backend_name)
        if include_intercept:
            intercept_score = multiplier_resid @ influence_native
            z_intercept = intercept_score / (bse_native[0] + 1e-30)
            batch_max = xp.maximum(batch_max, xp.abs(z_intercept))
        max_stats[filled : filled + bsz] = batch_max
        filled += bsz

    critical_native = xp.quantile(max_stats, 1.0 - alpha)
    critical = float(np.asarray(_to_numpy(critical_native), dtype=np.float64))
    if not np.isfinite(critical) or critical < 0.0:
        raise FloatingPointError(
            "backend-native simultaneous debiased inference produced a non-finite critical value"
        )

    conf_sim = np.asarray(result.conf_int, dtype=np.float64).copy()
    if include_intercept:
        lower = params_native - critical_native * bse_native
        upper = params_native + critical_native * bse_native
        conf_sim[:, 0] = np.asarray(_to_numpy(lower), dtype=np.float64)
        conf_sim[:, 1] = np.asarray(_to_numpy(upper), dtype=np.float64)
        target_mask = np.ones(p + 1, dtype=bool)
        model._debiased_intercept_influence_cpu = np.asarray(
            _to_numpy(influence_native),
            dtype=np.float64,
        ).reshape(-1)
    else:
        lower = params_native[1:] - critical_native * bse_native[1:]
        upper = params_native[1:] + critical_native * bse_native[1:]
        conf_sim[1:, 0] = np.asarray(_to_numpy(lower), dtype=np.float64)
        conf_sim[1:, 1] = np.asarray(_to_numpy(upper), dtype=np.float64)
        target_mask = np.zeros(p + 1, dtype=bool)
        target_mask[1:] = True
        model.__dict__.pop("_debiased_intercept_influence_cpu", None)

    if not np.all(np.isfinite(conf_sim)):
        raise FloatingPointError(
            "backend-native simultaneous debiased inference produced non-finite confidence intervals"
        )

    result.simultaneous_conf_int = conf_sim
    result.simultaneous_method = getattr(model, "simultaneous_method", None)
    result.simultaneous_alpha = alpha
    result.simultaneous_n_bootstrap = B
    result.simultaneous_critical_value = critical
    result.simultaneous_target_mask = target_mask
    metadata = dict(getattr(result, "metadata", {}) or {})
    metadata.update(
        {
            "simultaneous_numerical_backend": backend_name,
            "simultaneous_numerical_device": _device_name(backend_name, X_work),
            "simultaneous_reporting_backend": "numpy",
            "simultaneous_reporting_boundary": "post_numerical_inference",
        }
    )
    result.metadata = metadata
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
    backend_name = str(backend_name).strip().lower()
    if not (
        simultaneous_requested
        and original_intercept
        and backend_name in {"cupy", "torch"}
    ):
        return _ORIGINAL_FINALIZER(
            model,
            X_arr=X_arr,
            y_work=y_work,
            X_work=X_work,
            row_scale=row_scale,
            coef_native=coef_native,
            backend_name=backend_name,
            original_intercept=original_intercept,
            simultaneous_requested=simultaneous_requested,
            sample_weighted=sample_weighted,
        )

    M_native = getattr(model, _intercept_contract._NATIVE_M, None)
    if M_native is None:
        raise RuntimeError(
            "backend-native simultaneous debiased inference is missing the native precision matrix"
        )

    # Publish the coherent marginal result without invoking the NumPy
    # simultaneous helper; keep a local native M reference across the marginal
    # finalizer's cleanup boundary, then compute the joint procedure on device.
    result = _ORIGINAL_FINALIZER(
        model,
        X_arr=X_arr,
        y_work=y_work,
        X_work=X_work,
        row_scale=row_scale,
        coef_native=coef_native,
        backend_name=backend_name,
        original_intercept=original_intercept,
        simultaneous_requested=False,
        sample_weighted=sample_weighted,
    )
    return _native_simultaneous_maxz(
        model,
        result,
        X_arr=X_arr,
        y_work=y_work,
        X_work=X_work,
        row_scale=row_scale,
        coef_native=coef_native,
        M_native=M_native,
        backend_name=backend_name,
    )


def install_debiased_simultaneous_backend_contract() -> None:
    current = _weighted_contract._finalize_weighted_debiased_result
    if getattr(current, _FINALIZER_MARKER, False):
        return
    setattr(_finalize_weighted_debiased_result, _FINALIZER_MARKER, True)
    _weighted_contract._finalize_weighted_debiased_result = (
        _finalize_weighted_debiased_result
    )


__all__ = [
    "install_debiased_simultaneous_backend_contract",
    "_native_simultaneous_maxz",
]
