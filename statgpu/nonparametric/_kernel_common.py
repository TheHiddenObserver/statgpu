"""Shared utilities for kernel-based nonparametric estimators."""

from __future__ import annotations

import math
from typing import Any, Union

import numpy as np

from statgpu.backends import (
    _get_torch_device_str,
    _get_xp,
    _resolve_backend,
    _to_float_scalar,
    _to_numpy,
)

# Re-export for backward compatibility
__all__ = [
    "_auto_backend_from_device",
    "_as_points_2d",
    "_as_samples_2d",
    "_bandwidth_factor",
    "_bandwidth_factor_1d_nrd",
    "_effective_sample_size",
    "_get_xp",
    "_kernel_values_from_quad",
    "_normalize_kernel_name",
    "_normalize_regression_name",
    "_normalize_weights",
    "_resolve_backend",
    "_stable_inv_and_det",
    "_to_float_scalar",
    "_to_numpy",
    "_weighted_covariance",
]


def _torch_device_from_data(data) -> str:
    """Extract device string from a torch tensor, or return 'cpu' for others."""
    try:
        import torch
        if isinstance(data, torch.Tensor):
            return str(data.device)
    except (ImportError, AttributeError):
        pass
    return "cpu"


def _auto_backend_from_device(device: str, prefer_torch: bool = False) -> str:
    d = str(device).strip().lower()
    if d in ("numpy", "cpu"):
        return "numpy"
    if d == "torch":
        return "torch"
    if d in ("cuda", "gpu"):
        # Check if Torch is available and has CUDA
        if prefer_torch:
            try:
                import torch
                if torch.cuda.is_available():
                    return "torch"
            except Exception:
                pass
        # Otherwise try CuPy
        try:
            import cupy as cp
            _ = int(cp.cuda.runtime.getDeviceCount())
            return "cupy"
        except Exception:
            # Fallback to Torch if CuPy unavailable
            if not prefer_torch:
                try:
                    import torch
                    if torch.cuda.is_available():
                        return "torch"
                except Exception:
                    pass
            return "numpy"
    # Default: prefer CuPy, then Torch, then NumPy
    try:
        import cupy as cp
        _ = int(cp.cuda.runtime.getDeviceCount())
        return "cupy"
    except Exception:
        try:
            import torch
            if torch.cuda.is_available():
                return "torch"
        except Exception:
            pass
        return "numpy"


def _normalize_kernel_name(kernel: str) -> str:
    name = str(kernel).strip().lower()
    aliases = {
        "gaussian": "gaussian",
        "normal": "gaussian",
        "rectangular": "rectangular",
        "uniform": "rectangular",
        "box": "rectangular",
        "triangular": "triangular",
        "epanechnikov": "epanechnikov",
        "epa": "epanechnikov",
        "biweight": "biweight",
        "quartic": "biweight",
        "triweight": "triweight",
        "cosine": "cosine",
        "optcosine": "optcosine",
    }
    normalized = aliases.get(name)
    if normalized is None:
        raise ValueError(
            "kernel must be one of: 'gaussian', 'rectangular', 'triangular', "
            "'epanechnikov', 'biweight', 'triweight', 'cosine', 'optcosine'"
        )
    return normalized


def _normalize_regression_name(regression: str) -> str:
    name = str(regression).strip().lower()
    aliases = {
        "nw": "nw",
        "nadaraya_watson": "nw",
        "nadaraya-watson": "nw",
        "local_linear": "local_linear",
        "local-linear": "local_linear",
        "ll": "local_linear",
    }
    normalized = aliases.get(name)
    if normalized is None:
        raise ValueError(
            "regression must be one of: 'nw', 'nadaraya_watson', 'local_linear', 'll'"
        )
    return normalized


def _kernel_values_from_quad(quad, kernel_name: str, xp):
    if kernel_name == "gaussian":
        return xp.exp(-0.5 * quad)

    support_mask = quad <= 1.0
    if kernel_name == "rectangular":
        return support_mask.astype(xp.float64)

    if kernel_name == "triangular":
        return xp.maximum(1.0 - xp.sqrt(xp.maximum(quad, 0.0)), 0.0)

    one_minus_quad = xp.maximum(1.0 - quad, 0.0)
    if kernel_name == "epanechnikov":
        return one_minus_quad
    if kernel_name == "biweight":
        return one_minus_quad * one_minus_quad
    if kernel_name == "triweight":
        return one_minus_quad * one_minus_quad * one_minus_quad
    if kernel_name == "cosine":
        r = xp.sqrt(xp.maximum(quad, 0.0))
        return xp.where(support_mask, 0.5 * (1.0 + xp.cos(math.pi * r)), 0.0)
    if kernel_name == "optcosine":
        r = xp.sqrt(xp.maximum(quad, 0.0))
        return xp.where(support_mask, xp.cos(0.5 * math.pi * r), 0.0)

    raise ValueError(f"Unsupported kernel: {kernel_name}")


def _as_samples_2d(samples, xp):
    arr = xp.asarray(samples, dtype=xp.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim != 2:
        raise ValueError("samples must be 1D or 2D")

    n_samples = int(arr.shape[0])
    if n_samples < 2:
        raise ValueError("samples must contain at least 2 observations")
    return arr


def _as_points_2d(points, n_features: int, xp):
    arr = xp.asarray(points, dtype=xp.float64)
    if arr.ndim == 1:
        if n_features == 1:
            arr = arr.reshape(-1, 1)
        elif int(arr.size) == n_features:
            arr = arr.reshape(1, n_features)
        else:
            raise ValueError("points shape is incompatible with sample dimensionality")
    elif arr.ndim != 2:
        raise ValueError("points must be 1D or 2D")

    if int(arr.shape[1]) != int(n_features):
        raise ValueError("points feature dimension does not match samples")
    return arr


def _normalize_weights(weights, n_samples: int, xp, device: str = "cpu"):
    if weights is None:
        fill_val = 1.0 / float(n_samples)
        try:
            import torch
            if xp is torch:
                return xp.full((n_samples,), fill_val, dtype=xp.float64, device=device)
        except ImportError:
            pass
        return xp.full((n_samples,), fill_val, dtype=xp.float64)

    w = xp.asarray(weights, dtype=xp.float64).reshape(-1)
    if int(w.size) != int(n_samples):
        raise ValueError("weights must have the same length as samples")
    if _to_float_scalar(xp.min(w)) < 0.0:
        raise ValueError("weights must be non-negative")

    w_sum = xp.sum(w)
    if _to_float_scalar(w_sum) <= 0.0:
        raise ValueError("weights must sum to a positive value")

    return w / w_sum


def _effective_sample_size(weights, xp) -> float:
    w2 = xp.sum(weights * weights)
    denom = _to_float_scalar(w2)
    if denom <= 0.0:
        raise ValueError("invalid weights: effective sample size denominator is non-positive")
    return 1.0 / denom


def _bandwidth_factor_1d_nrd(
    method: str,
    *,
    n_eff: float,
    samples_2d,
    data_cov,
    xp,
) -> float:
    method_n = str(method).strip().lower()
    if method_n not in ("nrd0", "nrd"):
        raise ValueError("method must be one of: 'nrd0', 'nrd'")

    x = np.asarray(_to_numpy(samples_2d[:, 0]), dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 2:
        raise ValueError("need at least 2 finite samples for 'nrd0'/'nrd' bandwidth")

    sd = float(np.std(x, ddof=1))
    q75, q25 = np.quantile(x, [0.75, 0.25])
    robust = float((q75 - q25) / 1.34)

    scale = min(sd, robust) if np.isfinite(robust) and robust > 0.0 else sd
    if (not np.isfinite(scale)) or scale <= 0.0:
        scale = float(np.std(x, ddof=0))
    if (not np.isfinite(scale)) or scale <= 0.0:
        raise ValueError("unable to compute positive scale for 'nrd0'/'nrd' bandwidth")

    coeff = 0.9 if method_n == "nrd0" else 1.06
    bw_abs = float(coeff * scale * (float(n_eff) ** (-1.0 / 5.0)))
    if (not np.isfinite(bw_abs)) or bw_abs <= 0.0:
        raise ValueError("automatic bandwidth rule produced a non-positive value")

    data_sd = math.sqrt(max(_to_float_scalar(data_cov[0, 0]), 0.0))
    if data_sd <= 0.0 or (not np.isfinite(data_sd)):
        data_sd = max(float(np.finfo(np.float64).tiny), sd)

    factor = float(bw_abs / data_sd)
    if (not np.isfinite(factor)) or factor <= 0.0:
        raise ValueError("bandwidth factor must be a finite positive scalar")
    return factor


def _bandwidth_factor(
    bandwidth: Union[str, float, int],
    *,
    n_eff: float,
    n_features: int,
) -> float:
    if isinstance(bandwidth, str):
        method = bandwidth.strip().lower()
        if method == "scott":
            factor = n_eff ** (-1.0 / (n_features + 4.0))
        elif method == "silverman":
            factor = (n_eff * (n_features + 2.0) / 4.0) ** (-1.0 / (n_features + 4.0))
        else:
            raise ValueError(
                "bandwidth must be one of: 'scott', 'silverman', 'nrd0', 'nrd', "
                "'ucv', 'bcv', 'sj', 'sj-ste', 'sj-dpi', 'cv', 'cv_ls', 'cv-nw', 'cv-ll', "
                "or a positive scalar"
            )
    else:
        factor = float(bandwidth)

    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError("bandwidth factor must be a finite positive scalar")
    return float(factor)


def _weighted_covariance(samples_2d, weights_1d, xp):
    n_features = int(samples_2d.shape[1])

    mean = xp.sum(samples_2d * weights_1d[:, None], axis=0)
    centered = samples_2d - mean

    denom = 1.0 - xp.sum(weights_1d * weights_1d)
    denom_f = _to_float_scalar(denom)
    if denom_f <= 1e-15:
        raise ValueError("effective degrees of freedom is too small for covariance estimation")

    cov = (centered.T * weights_1d[None, :]) @ centered / denom
    cov = 0.5 * (cov + cov.T)

    trace = _to_float_scalar(xp.trace(cov))
    base = trace / float(max(1, n_features)) if np.isfinite(trace) else 1.0
    jitter = max(base * 1e-12, 1e-12)
    cov = cov + jitter * xp.eye(n_features, dtype=xp.float64)
    return cov


def _stable_inv_and_det(cov, xp):
    n_features = int(cov.shape[0])
    cov_work = cov.astype(xp.float64, copy=True)

    trace = _to_float_scalar(xp.trace(cov_work))
    base = trace / float(max(1, n_features)) if np.isfinite(trace) else 1.0
    jitter = max(base * 1e-12, 1e-12)

    last_err = None
    for _ in range(8):
        try:
            inv_cov = xp.linalg.inv(cov_work)
            det_cov = _to_float_scalar(xp.linalg.det(cov_work))
            if np.isfinite(det_cov) and det_cov > 0.0:
                return inv_cov, det_cov, cov_work
        except Exception as exc:
            last_err = exc

        cov_work = cov_work + jitter * xp.eye(n_features, dtype=xp.float64)
        jitter *= 10.0

    if last_err is not None:
        raise ValueError("covariance inversion failed") from last_err
    raise ValueError("covariance matrix is not positive definite")
