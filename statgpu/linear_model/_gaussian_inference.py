"""Shared Gaussian linear-model inference helpers.

Numerical covariance and reference-distribution inference stay on the selected
NumPy/CuPy/Torch backend. Conversion to NumPy is deliberately delayed until all
numerical inference for the call has completed and ``GaussianInferenceResult``
is constructed for the established reporting API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from statgpu.backends import _resolve_backend, _to_numpy
from statgpu.backends._array_ops import _linalg_exception_is_rank_failure
from statgpu.backends._utils import _cupy_asarray_on_device
from statgpu.inference._reference_distribution import two_sided_reference_inference
from statgpu.inference._results import GaussianInferenceResult


@dataclass
class GaussianFitState:
    X_design: Any
    y: Any
    resid: Any
    scale: Any
    nobs: int
    df_resid: int
    params: Any
    backend: str = "numpy"
    device: str = "cpu"
    normalization: float = 0.0


def validate_cov_type(cov_type: str) -> str:
    """Validate and normalize cov_type. Preserve string identity for clone()."""
    normalized = str(cov_type).lower()
    if normalized not in ("nonrobust", "hc0", "hc1", "hc2", "hc3", "hac"):
        raise ValueError(
            "cov_type must be one of: 'nonrobust', 'hc0', 'hc1', 'hc2', 'hc3', 'hac'"
        )
    return cov_type if str(cov_type) == normalized else normalized


def validate_hac_maxlags(hac_maxlags: Optional[int]) -> Optional[int]:
    if hac_maxlags is not None and int(hac_maxlags) < 0:
        raise ValueError("hac_maxlags must be a non-negative integer or None")
    return None if hac_maxlags is None else int(hac_maxlags)


def resolve_hac_maxlags(n_obs: int, hac_maxlags: Optional[int]) -> int:
    if n_obs <= 1:
        return 0
    if hac_maxlags is None:
        maxlags = int(np.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0)))
    else:
        maxlags = int(hac_maxlags)
    return max(0, min(maxlags, n_obs - 1))


def _namespace(backend: str):
    if backend == "torch":
        import torch

        return torch
    if backend == "cupy":
        import cupy as cp

        return cp
    return np


def _device_label(value, backend: str) -> str:
    if backend == "torch":
        return str(value.device)
    if backend == "cupy":
        return f"cuda:{int(value.device.id)}"
    return "cpu"


def _first_native_device(backend: str, *values) -> Optional[str]:
    """Return the concrete device of the first operand native to ``backend``."""
    if backend == "torch":
        import torch

        for value in values:
            if isinstance(value, torch.Tensor):
                return str(value.device)
        return None
    if backend == "cupy":
        import cupy as cp

        for value in values:
            if isinstance(value, cp.ndarray):
                return f"cuda:{int(value.device.id)}"
        return None
    return "cpu"


def _is_floating_dtype(value, backend: str) -> bool:
    if backend == "torch":
        return bool(value.dtype.is_floating_point)
    return bool(np.issubdtype(value.dtype, np.floating))


def _as_backend_array(value, backend: str, *, like=None, device: Optional[str] = None):
    """Convert without moving an existing native array off its concrete device.

    The historical NumPy shared-inference path always promoted numerical state
    to float64; preserve that precision contract. CuPy/Torch preserve the
    floating dtype selected by the fitted backend.
    """
    if backend == "torch":
        import torch

        if isinstance(like, torch.Tensor):
            target_device = like.device
            target_dtype = like.dtype if like.dtype.is_floating_point else torch.float64
        elif isinstance(value, torch.Tensor):
            target_device = value.device
            target_dtype = value.dtype if value.dtype.is_floating_point else torch.float64
        else:
            target_device = device or "cuda"
            target_dtype = torch.float64
        if isinstance(value, torch.Tensor):
            return value.to(device=target_device, dtype=target_dtype)
        return torch.as_tensor(value, dtype=target_dtype, device=target_device)

    if backend == "cupy":
        import cupy as cp

        target_dtype = None
        target_device = None
        if isinstance(like, cp.ndarray):
            target_dtype = like.dtype if _is_floating_dtype(like, "cupy") else cp.float64
            target_device = int(like.device.id)
        elif isinstance(value, cp.ndarray):
            target_dtype = value.dtype if _is_floating_dtype(value, "cupy") else cp.float64
            target_device = int(value.device.id)
        elif device is not None:
            device_label = str(device)
            if device_label == "cuda":
                target_device = int(cp.cuda.runtime.getDevice())
            elif device_label.startswith("cuda:"):
                try:
                    target_device = int(device_label.split(":", 1)[1])
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid CuPy CUDA device label: {device!r}"
                    ) from exc
            else:
                raise ValueError(
                    f"CuPy inference requires a CUDA device label, got {device!r}"
                )
        if target_device is None:
            target_device = int(cp.cuda.runtime.getDevice())
        return _cupy_asarray_on_device(
            value, target_device, dtype=target_dtype or cp.float64
        )

    return np.asarray(_to_numpy(value), dtype=np.float64)


def _sum(value, backend: str, axis=None):
    if backend == "torch":
        import torch

        return torch.sum(value) if axis is None else torch.sum(value, dim=axis)
    return value.sum() if axis is None else value.sum(axis=axis)


def _sqrt(value, backend: str):
    return _namespace(backend).sqrt(value)


def _abs(value, backend: str):
    if backend == "torch":
        import torch

        return torch.abs(value)
    return _namespace(backend).abs(value)


def _diag(value, backend: str):
    return _namespace(backend).diag(value)


def _stack(values, backend: str, axis=0):
    if backend == "torch":
        import torch

        return torch.stack(list(values), dim=axis)
    return _namespace(backend).stack(list(values), axis=axis)


def _maximum(value, floor: float, backend: str):
    if backend == "torch":
        import torch

        return torch.clamp(value, min=floor)
    return _namespace(backend).maximum(value, floor)


def _scalar(value, backend: str) -> float:
    if backend == "torch":
        return float(value.detach().item())
    if backend == "cupy":
        return float(value.item())
    return float(np.asarray(value))


def _contains_nan(value, backend: str) -> bool:
    xp = _namespace(backend)
    if backend == "torch":
        return bool(xp.any(xp.isnan(value)).item())
    if backend == "cupy":
        return bool(xp.any(xp.isnan(value)).item())
    return bool(np.any(np.isnan(np.asarray(value))))


def _to_reporting_numpy(value) -> np.ndarray:
    return np.asarray(_to_numpy(value), dtype=float)


def _constant_first_column(X, backend: str) -> bool:
    if int(X.shape[1]) == 0:
        return False
    first = X[0, 0]
    diff = _abs(X[:, 0] - first, backend)
    max_diff = diff.max()
    scale = _abs(first, backend)
    return _scalar(max_diff, backend) <= 1e-8 + 1e-5 * _scalar(scale, backend)


def build_gaussian_fit_state(
    X,
    y,
    coef,
    intercept,
    fit_intercept: bool,
    sample_weight=None,
    *,
    backend: str = "auto",
    device: Optional[str] = None,
) -> GaussianFitState:
    """Build backend-native Gaussian numerical state.

    ``sample_weight`` follows the existing analytic-weight convention by
    applying ``sqrt(weight)`` to design, response, and residual state.
    """
    backend_name = _resolve_backend(backend, X, y, coef, intercept, sample_weight)
    conversion_device = device or _first_native_device(
        backend_name, X, y, coef, intercept, sample_weight
    )
    X_arr = _as_backend_array(X, backend_name, device=conversion_device)
    y_arr = _as_backend_array(
        y, backend_name, like=X_arr, device=conversion_device
    )
    if y_arr.ndim == 2 and int(y_arr.shape[1]) == 1:
        y_arr = y_arr.reshape(-1)

    coef_arr = _as_backend_array(
        coef, backend_name, like=X_arr, device=conversion_device
    )
    intercept_arr = _as_backend_array(
        intercept, backend_name, like=X_arr, device=conversion_device
    )

    if fit_intercept:
        if backend_name == "torch":
            import torch

            intercept_col = torch.ones(
                (int(X_arr.shape[0]), 1), dtype=X_arr.dtype, device=X_arr.device
            )
            X_design_raw = torch.cat([intercept_col, X_arr], dim=1)
            if coef_arr.ndim == 1:
                params = torch.cat([intercept_arr.reshape(1), coef_arr])
            else:
                params = torch.cat([intercept_arr.reshape(1, -1), coef_arr], dim=0)
        elif backend_name == "cupy":
            import cupy as cp

            device_id = int(X_arr.device.id)
            with cp.cuda.Device(device_id):
                intercept_col = cp.ones(
                    (int(X_arr.shape[0]), 1), dtype=X_arr.dtype
                )
                X_design_raw = cp.concatenate([intercept_col, X_arr], axis=1)
                if coef_arr.ndim == 1:
                    params = cp.concatenate(
                        [intercept_arr.reshape(1), coef_arr], axis=0
                    )
                else:
                    params = cp.concatenate(
                        [intercept_arr.reshape(1, -1), coef_arr], axis=0
                    )
        else:
            intercept_col = np.ones((int(X_arr.shape[0]), 1), dtype=X_arr.dtype)
            X_design_raw = np.concatenate([intercept_col, X_arr], axis=1)
            if coef_arr.ndim == 1:
                params = np.concatenate([intercept_arr.reshape(1), coef_arr], axis=0)
            else:
                params = np.concatenate(
                    [intercept_arr.reshape(1, -1), coef_arr], axis=0
                )
    else:
        X_design_raw = X_arr
        params = coef_arr.clone() if backend_name == "torch" else coef_arr.copy()

    y_pred = X_arr @ coef_arr
    if fit_intercept:
        y_pred = y_pred + intercept_arr
    resid_raw = y_arr - y_pred

    normalization = float(int(X_arr.shape[0]))
    if sample_weight is None:
        X_design = X_design_raw
        y_state = y_arr
        resid = resid_raw
    else:
        sw = _as_backend_array(
            sample_weight,
            backend_name,
            like=X_arr,
            device=conversion_device,
        ).reshape(-1)
        if int(sw.shape[0]) != int(X_arr.shape[0]):
            raise ValueError(
                "sample_weight must be one-dimensional with length n_samples."
            )
        sqrt_sw = _sqrt(sw, backend_name)
        X_design = X_design_raw * sqrt_sw.reshape(-1, 1)
        y_state = (
            y_arr * sqrt_sw
            if y_arr.ndim == 1
            else y_arr * sqrt_sw.reshape(-1, 1)
        )
        resid = (
            resid_raw * sqrt_sw
            if resid_raw.ndim == 1
            else resid_raw * sqrt_sw.reshape(-1, 1)
        )
        normalization = _scalar(_sum(sw, backend_name), backend_name)

    nobs = int(X_design.shape[0])
    df_resid = nobs - int(X_design.shape[1])
    rss = _sum(resid**2, backend_name, axis=0)
    if df_resid > 0:
        scale = rss / float(df_resid)
    elif backend_name == "torch":
        import torch

        scale = torch.full_like(rss, float("nan"))
    else:
        xp = _namespace(backend_name)
        scale = xp.full_like(rss, xp.nan, dtype=X_arr.dtype)

    return GaussianFitState(
        X_design=X_design,
        y=y_state,
        resid=resid,
        scale=scale,
        nobs=nobs,
        df_resid=df_resid,
        params=params,
        backend=backend_name,
        device=_device_label(X_arr, backend_name),
        normalization=normalization,
    )


def _hac_meat_numpy(scores: np.ndarray, maxlags: int) -> np.ndarray:
    meat = scores.T @ scores
    for lag in range(1, maxlags + 1):
        weight = 1.0 - lag / (maxlags + 1.0)
        gamma = scores[lag:].T @ scores[:-lag]
        meat += weight * (gamma + gamma.T)
    return meat


def robust_covariance_numpy(
    X: np.ndarray,
    resid: np.ndarray,
    bread_inv: np.ndarray,
    cov_type: str,
    hac_maxlags: Optional[int] = None,
    df_resid: Optional[int] = None,
) -> np.ndarray:
    cov_type = validate_cov_type(cov_type)
    n, k = X.shape
    resid = np.asarray(resid, dtype=float)

    if cov_type == "hac":
        scores = X * resid[:, None]
        maxlags = resolve_hac_maxlags(n, hac_maxlags)
        meat = _hac_meat_numpy(scores, maxlags)
        return bread_inv @ meat @ bread_inv

    leverage = None
    if cov_type in ("hc2", "hc3"):
        leverage = np.sum(X * (X @ bread_inv), axis=1)
        leverage = np.clip(leverage, 0.0, 1.0 - 1e-12)

    if cov_type == "hc2":
        omega = resid**2 / np.maximum(1.0 - leverage, 1e-12)
    elif cov_type == "hc3":
        omega = resid**2 / np.maximum((1.0 - leverage) ** 2, 1e-12)
    else:
        omega = resid**2

    meat = X.T @ (X * omega[:, None])
    if cov_type == "hc1":
        correction_df = df_resid if df_resid is not None else (n - k)
        if correction_df > 0:
            meat *= n / correction_df
    return bread_inv @ meat @ bread_inv


def _hac_meat_gpu(scores, maxlags):
    meat = scores.T @ scores
    for lag in range(1, maxlags + 1):
        weight = 1.0 - lag / (maxlags + 1.0)
        gamma = scores[lag:].T @ scores[:-lag]
        meat = meat + weight * (gamma + gamma.T)
    return meat


def robust_covariance_gpu(
    X,
    resid,
    bread_inv,
    cov_type,
    xp,
    hac_maxlags=None,
    df_resid=None,
):
    """Backend-native robust/HAC covariance for CuPy or Torch.

    The historical ``xp`` parameter remains for compatibility with the
    existing exact-L2 GPU inference helpers.
    """
    backend = "torch" if getattr(xp, "__name__", "") == "torch" else "cupy"
    cov_type = validate_cov_type(cov_type)
    n, k = X.shape

    if cov_type == "hac":
        scores = X * resid.reshape(-1, 1)
        maxlags = resolve_hac_maxlags(int(n), hac_maxlags)
        meat = _hac_meat_gpu(scores, maxlags)
        return bread_inv @ meat @ bread_inv

    leverage = None
    if cov_type in ("hc2", "hc3"):
        if backend == "torch":
            leverage = xp.sum(X * (X @ bread_inv), dim=1)
            leverage = xp.clamp(leverage, min=0.0, max=1.0 - 1e-12)
        else:
            leverage = xp.sum(X * (X @ bread_inv), axis=1)
            leverage = xp.clip(leverage, 0.0, 1.0 - 1e-12)

    if cov_type == "hc2":
        omega = resid.reshape(-1) ** 2 / _maximum(
            1.0 - leverage, 1e-12, backend
        )
    elif cov_type == "hc3":
        omega = resid.reshape(-1) ** 2 / _maximum(
            (1.0 - leverage) ** 2, 1e-12, backend
        )
    else:
        omega = resid.reshape(-1) ** 2

    meat = X.T @ (X * omega.reshape(-1, 1))
    if cov_type == "hc1":
        correction_df = df_resid if df_resid is not None else (int(n) - int(k))
        if correction_df > 0:
            meat = meat * (int(n) / correction_df)
    return bread_inv @ meat @ bread_inv


def _inverse_or_pinv(matrix, backend: str):
    try:
        if backend == "torch":
            import torch

            return torch.linalg.inv(matrix)
        if backend == "cupy":
            import cupy as cp
            import cupyx

            with cupyx.errstate(linalg="raise"):
                return cp.linalg.inv(matrix)
        return np.linalg.inv(matrix)
    except Exception as exc:
        if not _linalg_exception_is_rank_failure(exc):
            raise
        if backend == "torch":
            import torch

            return torch.linalg.pinv(matrix)
        if backend == "cupy":
            import cupy as cp

            return cp.linalg.pinv(matrix)
        return np.linalg.pinv(matrix)


def _distribution_metadata(
    backend: str, device: str, ridge_alpha: float, alpha: float
):
    return {
        "ridge_alpha": float(ridge_alpha),
        "alpha": float(alpha),
        "numerical_backend": backend,
        "numerical_device": device,
        "reporting_backend": "numpy",
        "reporting_boundary": "post_numerical_inference",
    }


def _reference_inference(statistic, *, distribution, alpha, backend, device, df=None):
    xp = _namespace(backend)
    statistic_abs = _abs(statistic, backend)
    if backend == "cupy":
        with xp.cuda.Device(int(statistic_abs.device.id)):
            return two_sided_reference_inference(
                statistic_abs,
                distribution=distribution,
                alpha=alpha,
                backend=backend,
                xp=xp,
                df=df,
                device=None,
            )
    return two_sided_reference_inference(
        statistic_abs,
        distribution=distribution,
        alpha=alpha,
        backend=backend,
        xp=xp,
        df=df,
        device=device if backend == "torch" else None,
    )


def _compute_single_native(
    X,
    params,
    resid,
    scale,
    *,
    XtX,
    bread_inv,
    backend: str,
    device: str,
    df_resid: int,
    cov_type: str,
    hac_maxlags: Optional[int],
    ridge_alpha: float,
    alpha: float,
):
    """Return native bse/statistic/pvalue/CI for one response target."""
    if cov_type == "nonrobust":
        if ridge_alpha:
            cov_params = scale * (bread_inv @ XtX @ bread_inv)
        else:
            cov_params = scale * bread_inv
        bse = _sqrt(_diag(cov_params, backend), backend)
        statistic = params / (bse + 1e-30)
        pvalues, critical = _reference_inference(
            statistic,
            distribution="t",
            alpha=alpha,
            backend=backend,
            device=device,
            df=df_resid,
        )
        method = "classical"
        distribution = "t"
    else:
        if backend == "numpy":
            cov_params = robust_covariance_numpy(
                X,
                resid,
                bread_inv,
                cov_type,
                hac_maxlags=hac_maxlags,
                df_resid=df_resid,
            )
        else:
            xp = _namespace(backend)
            cov_params = robust_covariance_gpu(
                X,
                resid,
                bread_inv,
                cov_type,
                xp,
                hac_maxlags=hac_maxlags,
                df_resid=df_resid,
            )
        bse = _sqrt(_maximum(_diag(cov_params, backend), 0.0, backend), backend)
        statistic = params / (bse + 1e-30)
        pvalues, critical = _reference_inference(
            statistic,
            distribution="normal",
            alpha=alpha,
            backend=backend,
            device=device,
        )
        method = "sandwich"
        distribution = "normal"

    conf_int = _stack(
        [params - critical * bse, params + critical * bse], backend, axis=1
    )
    return bse, statistic, pvalues, conf_int, distribution, method


def compute_gaussian_inference(
    X_design,
    params,
    resid,
    scale,
    df_resid: int,
    cov_type: str,
    hac_maxlags: Optional[int] = None,
    ridge_alpha: float = 0.0,
    alpha: float = 0.05,
    ridge_penalize_intercept: Optional[bool] = None,
    *,
    backend: str = "auto",
    device: Optional[str] = None,
) -> Optional[GaussianInferenceResult]:
    """Compute Gaussian inference natively, then take one reporting snapshot."""
    if X_design is None or scale is None:
        return None

    backend_name = _resolve_backend(backend, X_design, params, resid, scale)
    conversion_device = device or _first_native_device(
        backend_name, X_design, params, resid, scale
    )
    X = _as_backend_array(X_design, backend_name, device=conversion_device)
    params_arr = _as_backend_array(
        params, backend_name, like=X, device=conversion_device
    )
    resid_arr = _as_backend_array(
        resid, backend_name, like=X, device=conversion_device
    )
    scale_arr = _as_backend_array(
        scale, backend_name, like=X, device=conversion_device
    )
    if _contains_nan(scale_arr, backend_name):
        return None

    numerical_device = _device_label(X, backend_name)
    k = int(X.shape[1])
    XtX = X.T @ X
    xp = _namespace(backend_name)

    if backend_name == "torch":
        penalty_diag = xp.zeros(k, dtype=X.dtype, device=X.device)
    elif backend_name == "cupy":
        with xp.cuda.Device(int(X.device.id)):
            penalty_diag = xp.zeros(k, dtype=X.dtype)
    else:
        penalty_diag = xp.zeros(k, dtype=X.dtype)

    if ridge_alpha:
        penalty_diag[:] = float(ridge_alpha)
        if ridge_penalize_intercept is None:
            unpenalized_intercept = k > 0 and _constant_first_column(
                X, backend_name
            )
        else:
            unpenalized_intercept = k > 0 and not bool(ridge_penalize_intercept)
        if unpenalized_intercept:
            penalty_diag[0] = 0.0

    bread = XtX + _diag(penalty_diag, backend_name)
    bread_inv = _inverse_or_pinv(bread, backend_name)
    cov_type_norm = validate_cov_type(cov_type)

    if params_arr.ndim == 2:
        n_targets = int(params_arr.shape[1])
        native = [
            _compute_single_native(
                X,
                params_arr[:, target],
                resid_arr[:, target],
                scale_arr.reshape(-1)[target],
                XtX=XtX,
                bread_inv=bread_inv,
                backend=backend_name,
                device=numerical_device,
                df_resid=df_resid,
                cov_type=cov_type_norm,
                hac_maxlags=hac_maxlags,
                ridge_alpha=ridge_alpha,
                alpha=alpha,
            )
            for target in range(n_targets)
        ]
        bse = _stack([value[0] for value in native], backend_name, axis=1)
        statistic = _stack([value[1] for value in native], backend_name, axis=1)
        pvalues = _stack([value[2] for value in native], backend_name, axis=1)
        conf_int = _stack([value[3] for value in native], backend_name, axis=1)
        distribution = native[0][4]
        method = native[0][5]
        metadata = {
            **_distribution_metadata(
                backend_name, numerical_device, ridge_alpha, alpha
            ),
            "n_targets": n_targets,
        }
    else:
        (
            bse,
            statistic,
            pvalues,
            conf_int,
            distribution,
            method,
        ) = _compute_single_native(
            X,
            params_arr,
            resid_arr,
            scale_arr,
            XtX=XtX,
            bread_inv=bread_inv,
            backend=backend_name,
            device=numerical_device,
            df_resid=df_resid,
            cov_type=cov_type_norm,
            hac_maxlags=hac_maxlags,
            ridge_alpha=ridge_alpha,
            alpha=alpha,
        )
        metadata = _distribution_metadata(
            backend_name, numerical_device, ridge_alpha, alpha
        )

    # This is the sole full-result reporting boundary for the shared helper.
    return GaussianInferenceResult(
        params=_to_reporting_numpy(params_arr),
        bse=_to_reporting_numpy(bse),
        statistic=_to_reporting_numpy(statistic),
        pvalues=_to_reporting_numpy(pvalues),
        conf_int=_to_reporting_numpy(conf_int),
        cov_type=cov_type_norm,
        distribution=distribution,
        df=df_resid,
        method=method,
        metadata=metadata,
    )
