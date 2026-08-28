"""Shared Gaussian linear-model inference helpers.

Numerical covariance and reference-distribution inference stay on the selected
NumPy/CuPy/Torch backend.  Conversion to NumPy is intentionally delayed until
``GaussianInferenceResult`` is constructed for the established reporting API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from statgpu.backends import _resolve_backend, _to_numpy, get_backend
from statgpu.backends._array_ops import _linalg_exception_is_rank_failure
from statgpu.inference._distributions_backend import get_distribution
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
    """Validate and normalize cov_type. Preserves string identity for sklearn clone()."""
    _ct = str(cov_type).lower()
    if _ct not in ("nonrobust", "hc0", "hc1", "hc2", "hc3", "hac"):
        raise ValueError(
            "cov_type must be one of: 'nonrobust', 'hc0', 'hc1', 'hc2', 'hc3', 'hac'"
        )
    return cov_type if str(cov_type) == _ct else _ct


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


def _device_label(value, backend: str) -> str:
    if backend == "torch":
        return str(value.device)
    if backend == "cupy":
        try:
            return f"cuda:{int(value.device.id)}"
        except Exception:
            return "cuda"
    return "cpu"


def _is_floating_dtype(value, backend: str) -> bool:
    if backend == "torch":
        return bool(value.dtype.is_floating_point)
    return bool(np.issubdtype(value.dtype, np.floating))


def _as_backend_array(value, backend: str, *, like=None, device: Optional[str] = None):
    """Convert without moving an existing native array off its concrete device."""
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
        if target_device is not None:
            with cp.cuda.Device(target_device):
                return cp.asarray(value, dtype=target_dtype)
        return get_backend("cupy").asarray(value, dtype=target_dtype or cp.float64)

    arr = np.asarray(value)
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float64)
    return arr


def _sum(value, backend: str, axis=None):
    if backend == "torch":
        import torch
        return torch.sum(value) if axis is None else torch.sum(value, dim=axis)
    return value.sum() if axis is None else value.sum(axis=axis)


def _sqrt(value, backend: str):
    if backend == "torch":
        import torch
        return torch.sqrt(value)
    return np.sqrt(value) if backend == "numpy" else __import__("cupy").sqrt(value)


def _abs(value, backend: str):
    if backend == "torch":
        import torch
        return torch.abs(value)
    return np.abs(value) if backend == "numpy" else __import__("cupy").abs(value)


def _diag(value, backend: str):
    if backend == "torch":
        import torch
        return torch.diag(value)
    return np.diag(value) if backend == "numpy" else __import__("cupy").diag(value)


def _stack(values, backend: str, axis=0):
    if backend == "torch":
        import torch
        return torch.stack(list(values), dim=axis)
    xp = np if backend == "numpy" else __import__("cupy")
    return xp.stack(list(values), axis=axis)


def _maximum(value, floor: float, backend: str):
    if backend == "torch":
        import torch
        return torch.clamp(value, min=floor)
    xp = np if backend == "numpy" else __import__("cupy")
    return xp.maximum(value, floor)


def _clip(value, lo: float, hi: float, backend: str):
    if backend == "torch":
        import torch
        return torch.clamp(value, min=lo, max=hi)
    xp = np if backend == "numpy" else __import__("cupy")
    return xp.clip(value, lo, hi)


def _scalar(value, backend: str) -> float:
    if backend == "torch":
        return float(value.detach().item())
    if backend == "cupy":
        return float(value.item())
    return float(np.asarray(value))


def _contains_nan(value, backend: str) -> bool:
    if backend == "torch":
        import torch
        return bool(torch.any(torch.isnan(value)).item())
    if backend == "cupy":
        import cupy as cp
        return bool(cp.any(cp.isnan(value)).item())
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
    applying ``sqrt(weight)`` to the design, response, and residual state.
    """
    backend_name = _resolve_backend(backend, X, y)
    X_arr = _as_backend_array(X, backend_name, device=device)
    y_arr = _as_backend_array(y, backend_name, like=X_arr, device=device)
    if y_arr.ndim == 2 and int(y_arr.shape[1]) == 1:
        y_arr = y_arr.reshape(-1)

    coef_arr = _as_backend_array(coef, backend_name, like=X_arr, device=device)
    intercept_arr = _as_backend_array(intercept, backend_name, like=X_arr, device=device)

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
        else:
            xp = np if backend_name == "numpy" else __import__("cupy")
            intercept_col = xp.ones((int(X_arr.shape[0]), 1), dtype=X_arr.dtype)
            X_design_raw = xp.concatenate([intercept_col, X_arr], axis=1)
            if coef_arr.ndim == 1:
                params = xp.concatenate([intercept_arr.reshape(1), coef_arr], axis=0)
            else:
                params = xp.concatenate([intercept_arr.reshape(1, -1), coef_arr], axis=0)
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
        sw = _as_backend_array(sample_weight, backend_name, like=X_arr, device=device).reshape(-1)
        if int(sw.shape[0]) != int(X_arr.shape[0]):
            raise ValueError("sample_weight must be one-dimensional with length n_samples.")
        sqrt_sw = _sqrt(sw, backend_name)
        X_design = X_design_raw * sqrt_sw.reshape(-1, 1)
        y_state = y_arr * sqrt_sw if y_arr.ndim == 1 else y_arr * sqrt_sw.reshape(-1, 1)
        resid = resid_raw * sqrt_sw if resid_raw.ndim == 1 else resid_raw * sqrt_sw.reshape(-1, 1)
        normalization = _scalar(_sum(sw, backend_name), backend_name)

    nobs = int(X_design.shape[0])
    df_resid = nobs - int(X_design.shape[1])
    rss = _sum(resid ** 2, backend_name, axis=0)
    if df_resid > 0:
        scale = rss / float(df_resid)
    else:
        if backend_name == "torch":
            import torch
            scale = torch.full_like(rss, float("nan"))
        else:
            xp = np if backend_name == "numpy" else __import__("cupy")
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
        omega = resid ** 2 / np.maximum(1.0 - leverage, 1e-12)
    elif cov_type == "hc3":
        omega = resid ** 2 / np.maximum((1.0 - leverage) ** 2, 1e-12)
    else:
        omega = resid ** 2

    meat = X.T @ (X * omega[:, None])
    if cov_type == "hc1":
        correction_df = df_resid if df_resid is not None else (n - k)
        if correction_df > 0:
            meat *= n / correction_df
    return bread_inv @ meat @ bread_inv


def _hac_meat_gpu(scores, maxlags, backend: str):
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

    The historical ``xp`` parameter is retained for compatibility with the
    exact-L2 GPU inference helpers.
    """
    backend = "torch" if getattr(xp, "__name__", "") == "torch" else "cupy"
    cov_type = validate_cov_type(cov_type)
    n, k = X.shape

    if cov_type == "hac":
        scores = X * resid.reshape(-1, 1)
        maxlags = resolve_hac_maxlags(int(n), hac_maxlags)
        meat = _hac_meat_gpu(scores, maxlags, backend)
        return bread_inv @ meat @ bread_inv

    leverage = None
    if cov_type in ("hc2", "hc3"):
        if backend == "torch":
            import torch
            leverage = torch.sum(X * (X @ bread_inv), dim=1)
            leverage = torch.clamp(leverage, min=0.0, max=1.0 - 1e-12)
        else:
            import cupy as cp
            leverage = cp.sum(X * (X @ bread_inv), axis=1)
            leverage = cp.clip(leverage, 0.0, 1.0 - 1e-12)

    if cov_type == "hc2":
        omega = resid.reshape(-1) ** 2 / _maximum(1.0 - leverage, 1e-12, backend)
    elif cov_type == "hc3":
        omega = resid.reshape(-1) ** 2 / _maximum((1.0 - leverage) ** 2, 1e-12, backend)
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


def _distribution_metadata(backend: str, device: str, ridge_alpha: float, alpha: float):
    return {
        "ridge_alpha": float(ridge_alpha),
        "alpha": float(alpha),
        "numerical_backend": backend,
        "numerical_device": device,
        "reporting_backend": "numpy",
        "reporting_boundary": "post_numerical_inference",
    }


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
    """Compute Gaussian inference natively, then snapshot result arrays to NumPy."""
    if X_design is None or scale is None:
        return None

    backend_name = _resolve_backend(backend, X_design, params, resid)
    X = _as_backend_array(X_design, backend_name, device=device)
    params_arr = _as_backend_array(params, backend_name, like=X, device=device)
    resid_arr = _as_backend_array(resid, backend_name, like=X, device=device)
    scale_arr = _as_backend_array(scale, backend_name, like=X, device=device)
    if _contains_nan(scale_arr, backend_name):
        return None

    numerical_device = _device_label(X, backend_name)
    n, k = int(X.shape[0]), int(X.shape[1])
    XtX = X.T @ X

    if backend_name == "torch":
        import torch
        penalty_diag = torch.zeros(k, dtype=X.dtype, device=X.device)
    else:
        xp = np if backend_name == "numpy" else __import__("cupy")
        penalty_diag = xp.zeros(k, dtype=X.dtype)

    if ridge_alpha:
        penalty_diag[:] = float(ridge_alpha)
        if ridge_penalize_intercept is None:
            unpenalized_intercept = k > 0 and _constant_first_column(X, backend_name)
        else:
            unpenalized_intercept = k > 0 and not bool(ridge_penalize_intercept)
        if unpenalized_intercept:
            penalty_diag[0] = 0.0

    bread = XtX + _diag(penalty_diag, backend_name)
    bread_inv = _inverse_or_pinv(bread, backend_name)

    if params_arr.ndim == 2:
        n_targets = int(params_arr.shape[1])
        bse_out = np.empty(tuple(params_arr.shape), dtype=float)
        t_out = np.empty(tuple(params_arr.shape), dtype=float)
        p_out = np.empty(tuple(params_arr.shape), dtype=float)
        ci_out = np.empty((int(params_arr.shape[0]), n_targets, 2), dtype=float)
        for j in range(n_targets):
            scale_j = scale_arr.reshape(-1)[j]
            result = compute_gaussian_inference(
                X,
                params_arr[:, j],
                resid_arr[:, j],
                scale_j,
                df_resid,
                cov_type,
                hac_maxlags=hac_maxlags,
                ridge_alpha=ridge_alpha,
                alpha=alpha,
                ridge_penalize_intercept=ridge_penalize_intercept,
                backend=backend_name,
                device=numerical_device,
            )
            if result is None:
                return None
            bse_out[:, j] = result.bse
            t_out[:, j] = result.tvalues
            p_out[:, j] = result.pvalues
            ci_out[:, j, :] = result.conf_int
        cov_type_norm = validate_cov_type(cov_type)
        return GaussianInferenceResult(
            params=_to_reporting_numpy(params_arr),
            bse=bse_out,
            statistic=t_out,
            pvalues=p_out,
            conf_int=ci_out,
            cov_type=cov_type_norm,
            distribution="t" if cov_type_norm == "nonrobust" else "normal",
            df=df_resid,
            method="classical" if cov_type_norm == "nonrobust" else "sandwich",
            metadata={
                **_distribution_metadata(
                    backend_name, numerical_device, ridge_alpha, alpha
                ),
                "n_targets": n_targets,
            },
        )

    cov_type_norm = validate_cov_type(cov_type)
    if cov_type_norm == "nonrobust":
        if ridge_alpha:
            cov_params = scale_arr * (bread_inv @ XtX @ bread_inv)
        else:
            cov_params = scale_arr * bread_inv
        bse = _sqrt(_diag(cov_params, backend_name), backend_name)
        statistic = params_arr / (bse + 1e-30)
        dist = get_distribution(
            "t",
            backend=backend_name,
            device=numerical_device if backend_name == "torch" else None,
        )
        pvalues = dist.two_sided_pvalue(statistic, df=df_resid)
        critical = dist.two_sided_critical_value(alpha, df=df_resid)
        conf_int = _stack(
            [params_arr - critical * bse, params_arr + critical * bse],
            backend_name,
            axis=1,
        )
        method = "classical"
        distribution = "t"
    else:
        if backend_name == "numpy":
            cov_params = robust_covariance_numpy(
                X,
                resid_arr,
                bread_inv,
                cov_type_norm,
                hac_maxlags=hac_maxlags,
                df_resid=df_resid,
            )
        else:
            xp = __import__("torch") if backend_name == "torch" else __import__("cupy")
            cov_params = robust_covariance_gpu(
                X,
                resid_arr,
                bread_inv,
                cov_type_norm,
                xp,
                hac_maxlags=hac_maxlags,
                df_resid=df_resid,
            )
        bse = _sqrt(_maximum(_diag(cov_params, backend_name), 0.0, backend_name), backend_name)
        statistic = params_arr / (bse + 1e-30)
        dist = get_distribution(
            "norm",
            backend=backend_name,
            device=numerical_device if backend_name == "torch" else None,
        )
        pvalues = _clip(2.0 * dist.sf(_abs(statistic, backend_name)), 0.0, 1.0, backend_name)
        critical = dist.ppf(1.0 - alpha / 2.0)
        conf_int = _stack(
            [params_arr - critical * bse, params_arr + critical * bse],
            backend_name,
            axis=1,
        )
        method = "sandwich"
        distribution = "normal"

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
        metadata=_distribution_metadata(
            backend_name, numerical_device, ridge_alpha, alpha
        ),
    )
