"""Kernel density estimation with NumPy/CuPy backends."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import NormalDist
from typing import Any, Dict, Optional, Union

import numpy as np

from statgpu._base import BaseEstimator
from ._bandwidth_selection import select_bandwidth

from ._kernel_common import (
    _auto_backend_from_device,
    _as_points_2d,
    _as_samples_2d,
    _effective_sample_size,
    _get_xp,
    _kernel_values_from_quad,
    _normalize_kernel_name,
    _normalize_weights,
    _stable_inv_and_det,
    _to_float_scalar,
    _to_numpy,
    _weighted_covariance,
)


def _unit_ball_volume(n_features: int) -> float:
    d = int(n_features)
    if d <= 0:
        raise ValueError("n_features must be a positive integer")
    return float((math.pi ** (0.5 * d)) / math.gamma(0.5 * d + 1.0))


def _kernel_norm_const(kernel_name: str, n_features: int) -> float:
    d = int(n_features)
    if kernel_name == "gaussian":
        return float((2.0 * math.pi) ** (-0.5 * d))

    volume = _unit_ball_volume(d)
    if kernel_name == "rectangular":
        return float(1.0 / volume)
    if kernel_name == "triangular":
        return float((d + 1.0) / volume)
    if kernel_name == "epanechnikov":
        return float((d + 2.0) / (2.0 * volume))
    if kernel_name == "biweight":
        return float(((d + 2.0) * (d + 4.0)) / (8.0 * volume))
    if kernel_name == "triweight":
        return float(((d + 2.0) * (d + 4.0) * (d + 6.0)) / (48.0 * volume))
    if kernel_name == "cosine":
        if d != 1:
            raise ValueError("kernel='cosine' currently supports only 1D samples")
        return 1.0
    if kernel_name == "optcosine":
        if d != 1:
            raise ValueError("kernel='optcosine' currently supports only 1D samples")
        return float(math.pi / 4.0)

    raise ValueError(f"Unsupported kernel: {kernel_name}")


class KernelDensityEstimator(BaseEstimator):
    """sklearn-style kernel density estimator with class-owned fit/predict API."""

    def __init__(
        self,
        *,
        bandwidth: Union[str, float, int] = "scott",
        weights=None,
        kernel: str = "gaussian",
        backend: str = "auto",
        device: str = "auto",
        n_jobs: Optional[int] = None,
        gpu_memory_cleanup: bool = False,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.bandwidth = bandwidth
        self.weights = weights
        self.kernel = kernel
        self.backend = backend
        self.gpu_memory_cleanup = gpu_memory_cleanup

    def _resolve_backend_name(self, X) -> str:
        name = str(self.backend).strip().lower()
        if name != "auto":
            return name
        return _auto_backend_from_device(self._get_compute_device().value)

    def fit(self, X, y=None):
        backend_name = self._resolve_backend_name(X)
        xp = _get_xp(backend_name)

        samples_2d = _as_samples_2d(X, xp)
        n_samples, n_features = int(samples_2d.shape[0]), int(samples_2d.shape[1])

        weights_1d = _normalize_weights(self.weights, n_samples, xp)
        n_eff = _effective_sample_size(weights_1d, xp)
        kernel_name = _normalize_kernel_name(self.kernel)

        data_cov = _weighted_covariance(samples_2d, weights_1d, xp)
        if kernel_name in ("cosine", "optcosine") and n_features != 1:
            raise ValueError(f"kernel='{kernel_name}' currently supports only 1D samples")

        bw_result = None
        if isinstance(self.bandwidth, str):
            bw_result = select_bandwidth(
                self.bandwidth,
                n_eff=n_eff,
                n_features=n_features,
                samples_2d=samples_2d,
                weights_1d=weights_1d,
                data_cov=data_cov,
                xp=xp,
                enable_r_selectors=True,
                estimator="kde",
            )
            factor = float(bw_result.factor)
        else:
            factor = float(self.bandwidth)
            if (not np.isfinite(factor)) or factor <= 0.0:
                raise ValueError("bandwidth factor must be a finite positive scalar")

        scaled_cov = data_cov * (factor**2)
        inv_cov, det_cov, stable_cov = _stable_inv_and_det(scaled_cov, xp)

        kernel_norm_const = _kernel_norm_const(kernel_name, n_features)
        if not np.isfinite(kernel_norm_const) or kernel_norm_const <= 0.0:
            raise ValueError("kernel normalization constant must be finite and positive")
        norm_const = np.sqrt(det_cov) / kernel_norm_const

        self.samples_ = samples_2d
        self.weights_ = weights_1d
        self.bandwidth_factor_ = factor
        self.bandwidth_info_ = bw_result
        self.covariance_ = stable_cov
        self.inv_covariance_ = inv_cov
        self.norm_const_ = float(norm_const)
        self.inv_norm_const_ = float(1.0 / self.norm_const_)
        self.kernel_ = kernel_name
        self.n_features_ = n_features
        self.n_samples_ = n_samples
        self.backend_ = backend_name
        # Cache these terms for repeated evaluations to avoid recomputation in hot paths.
        self._samples_proj_ = self.samples_ @ self.inv_covariance_
        self._samples_quad_ = xp.sum(self._samples_proj_ * self.samples_, axis=1)
        self._fitted = True
        return self

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Estimator not fitted. Call fit() first.")

    def _cleanup_cuda_memory(self):
        if not self.gpu_memory_cleanup:
            return
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass

    def _cleanup_torch_memory(self):
        if not self.gpu_memory_cleanup:
            return
        try:
            import torch
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            pass

    def __del__(self):
        try:
            self._cleanup_cuda_memory()
            self._cleanup_torch_memory()
        except Exception:
            pass

    def _evaluate_density(self, points_2d, *, batch_size: int, xp):
        n_points = int(points_2d.shape[0])
        n_samples = int(self.samples_.shape[0])
        n_features = int(self.samples_.shape[1])

        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        out = xp.empty(n_points, dtype=xp.float64)

        if n_features == 1:
            samples_1d = self.samples_[:, 0]
            inv_scalar = self.inv_covariance_[0, 0]

            if xp is np and self.kernel_ == "gaussian" and (n_points * n_samples) <= 8_000_000:
                q_1d = points_2d[:, 0]
                diff = q_1d[:, None] - samples_1d[None, :]
                diff *= diff
                diff *= (-0.5 * inv_scalar)
                np.exp(diff, out=diff)
                out[:] = (diff @ self.weights_) * self.inv_norm_const_
                return out

            if xp is np and (n_points * n_samples) <= 8_000_000:
                q_1d = points_2d[:, 0]
                diff = q_1d[:, None] - samples_1d[None, :]
                quad = (diff * diff) * inv_scalar
                kernels = _kernel_values_from_quad(quad, self.kernel_, xp)
                out[:] = (kernels @ self.weights_) * self.inv_norm_const_
                return out

            for start in range(0, n_points, int(batch_size)):
                stop = min(start + int(batch_size), n_points)
                q_1d = points_2d[start:stop, 0]
                diff = q_1d[:, None] - samples_1d[None, :]
                if self.kernel_ == "gaussian":
                    diff *= diff
                    diff *= (-0.5 * inv_scalar)
                    xp.exp(diff, out=diff)
                    out[start:stop] = (diff @ self.weights_) * self.inv_norm_const_
                else:
                    quad = (diff * diff) * inv_scalar
                    kernels = _kernel_values_from_quad(quad, self.kernel_, xp)
                    out[start:stop] = (kernels @ self.weights_) * self.inv_norm_const_
            return out

        s_proj = self._samples_proj_
        s_quad = self._samples_quad_
        is_gaussian = self.kernel_ == "gaussian"
        use_log_sum_exp = n_features >= 8

        for start in range(0, n_points, int(batch_size)):
            stop = min(start + int(batch_size), n_points)
            q = points_2d[start:stop]

            q_proj = q @ self.inv_covariance_
            q_quad = xp.sum(q_proj * q, axis=1)
            cross = q_proj @ self.samples_.T
            quad = q_quad[:, None] + s_quad[None, :] - 2.0 * cross
            quad = xp.maximum(quad, 0.0)

            if use_log_sum_exp:
                if is_gaussian:
                    log_kernels = -0.5 * quad
                else:
                    kernels = _kernel_values_from_quad(quad, self.kernel_, xp)
                    log_kernels = xp.log(xp.maximum(kernels, np.finfo(np.float64).tiny))
                log_kernels_max = xp.max(log_kernels, axis=1, keepdims=True)
                log_sum = log_kernels_max[:, 0] + xp.log(
                    xp.sum(xp.exp(log_kernels - log_kernels_max) * self.weights_[None, :], axis=1)
                )
                out[start:stop] = xp.exp(log_sum) * self.inv_norm_const_
            else:
                kernels = _kernel_values_from_quad(quad, self.kernel_, xp)
                out[start:stop] = (kernels @ self.weights_) * self.inv_norm_const_

        return out

    def pdf(self, points, *, batch_size: int = 1024):
        self._require_fitted()
        xp = _get_xp(self.backend_)
        points_2d = _as_points_2d(points, self.n_features_, xp)
        result = self._evaluate_density(points_2d, batch_size=int(batch_size), xp=xp)
        self._cleanup_cuda_memory()
        self._cleanup_torch_memory()
        return result

    def _evaluate_log_density(self, points_2d, *, batch_size: int, xp):
        """Evaluate log-density in log domain (avoids underflow for high dimensions)."""
        n_points = int(points_2d.shape[0])
        n_features = int(points_2d.shape[1])

        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        if n_features == 1:
            density = self._evaluate_density(points_2d, batch_size=batch_size, xp=xp)
            tiny = np.finfo(np.float64).tiny
            return xp.log(xp.maximum(density, tiny))

        s_proj = self._samples_proj_
        s_quad = self._samples_quad_
        log_norm = math.log(self.inv_norm_const_) if self.inv_norm_const_ > 0.0 else float("-inf")
        is_gaussian = self.kernel_ == "gaussian"

        out = xp.empty(n_points, dtype=xp.float64)

        for start in range(0, n_points, int(batch_size)):
            stop = min(start + int(batch_size), n_points)
            q = points_2d[start:stop]

            q_proj = q @ self.inv_covariance_
            q_quad = xp.sum(q_proj * q, axis=1)
            cross = q_proj @ self.samples_.T
            quad = q_quad[:, None] + s_quad[None, :] - 2.0 * cross
            quad = xp.maximum(quad, 0.0)

            if is_gaussian:
                log_kernels = -0.5 * quad
            else:
                kernels = _kernel_values_from_quad(quad, self.kernel_, xp)
                log_kernels = xp.log(xp.maximum(kernels, np.finfo(np.float64).tiny))

            log_kernels_max = xp.max(log_kernels, axis=1, keepdims=True)
            log_kernels_shifted = log_kernels - log_kernels_max
            log_sum = log_kernels_max[:, 0] + xp.log(
                xp.sum(xp.exp(log_kernels_shifted) * self.weights_[None, :], axis=1)
            )
            out[start:stop] = log_sum + log_norm

        return out

    def logpdf(self, points, *, batch_size: int = 1024):
        self._require_fitted()
        xp = _get_xp(self.backend_)
        points_2d = _as_points_2d(points, self.n_features_, xp)
        result = self._evaluate_log_density(points_2d, batch_size=int(batch_size), xp=xp)
        self._cleanup_cuda_memory()
        self._cleanup_torch_memory()
        return result

    def __call__(self, points, *, batch_size: int = 1024):
        return self.pdf(points, batch_size=batch_size)

    def to_numpy_metadata(self):
        self._require_fitted()
        bandwidth_selection = None
        if hasattr(self.bandwidth_info_, "to_dict"):
            bandwidth_selection = self.bandwidth_info_.to_dict()
        return {
            "bandwidth_factor": float(self.bandwidth_factor_),
            "bandwidth_selection": bandwidth_selection,
            "n_samples": int(self.n_samples_),
            "n_features": int(self.n_features_),
            "backend": self.backend_,
            "kernel": self.kernel_,
            "covariance": _to_numpy(self.covariance_),
            "inv_covariance": _to_numpy(self.inv_covariance_),
            "weights": _to_numpy(self.weights_),
        }

    def predict(self, X):
        return self.pdf(X)

    def score_samples(self, X):
        return self.logpdf(X)

    def score(self, X, y=None):
        vals = self.score_samples(X)
        return float(np.mean(_to_numpy(vals)))


class KDE(KernelDensityEstimator):
    """Alias class for KernelDensityEstimator."""


def fit_kde(
    samples,
    *,
    bandwidth: Union[str, float, int] = "scott",
    weights=None,
    kernel: str = "gaussian",
    backend: str = "auto",
) -> KDE:
    """Fit a KDE model.

    Parameters
    ----------
    samples : array-like of shape (n_samples,) or (n_samples, n_features)
        Training observations.
    bandwidth : {'scott', 'silverman', 'nrd0', 'nrd', 'ucv', 'bcv', 'sj', 'sj-ste', 'sj-dpi'} or float, default='scott'
        Bandwidth scaling factor mode or explicit positive factor.
    weights : array-like of shape (n_samples,), optional
        Non-negative sample weights. If omitted, uniform weights are used.
    kernel : {'gaussian', 'rectangular', 'triangular', 'epanechnikov',
              'biweight', 'triweight', 'cosine', 'optcosine'}, default='gaussian'
        Kernel function used for density estimation.
    backend : {'auto', 'numpy', 'cupy'}, default='auto'
        Compute backend. 'auto' selects from the estimator's configured
        device/backend rather than inferring from input array types.

    Returns
    -------
    KDE
        Fitted KDE object.
    """
    model = KDE(
        bandwidth=bandwidth,
        weights=weights,
        kernel=kernel,
        backend=backend,
    )
    return model.fit(samples)


def kde_pdf(
    samples,
    points,
    *,
    bandwidth: Union[str, float, int] = "scott",
    weights=None,
    kernel: str = "gaussian",
    backend: str = "auto",
    return_log: bool = False,
    batch_size: int = 1024,
):
    """One-shot Gaussian KDE evaluation.

    This helper fits a KDE model and evaluates it at `points`.
    """
    model = fit_kde(
        samples,
        bandwidth=bandwidth,
        weights=weights,
        kernel=kernel,
        backend=backend,
    )
    if return_log:
        return model.logpdf(points, batch_size=batch_size)
    return model.pdf(points, batch_size=batch_size)


@dataclass
class KDEBootstrapResult:
    """Pointwise bootstrap confidence intervals for KDE estimates."""

    points: np.ndarray
    estimate: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    confidence_level: float
    n_resamples: int
    random_state: Optional[int]
    kernel: str
    backend: str
    metadata: Dict[str, Any]
    bootstrap_samples: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "points": np.asarray(self.points).tolist(),
            "estimate": np.asarray(self.estimate).tolist(),
            "lower": np.asarray(self.lower).tolist(),
            "upper": np.asarray(self.upper).tolist(),
            "confidence_level": float(self.confidence_level),
            "n_resamples": int(self.n_resamples),
            "random_state": self.random_state,
            "kernel": self.kernel,
            "backend": self.backend,
            "metadata": self.metadata,
        }
        if self.bootstrap_samples is not None:
            payload["bootstrap_samples"] = np.asarray(self.bootstrap_samples).tolist()
        return payload


def _validate_confidence_level(confidence_level: float) -> float:
    level = float(confidence_level)
    if level <= 0.0 or level >= 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    return level


def _validate_n_resamples(n_resamples: int) -> int:
    n = int(n_resamples)
    if n <= 0:
        raise ValueError("n_resamples must be a positive integer")
    return n


def kde_bootstrap_confidence_interval(
    samples,
    points,
    *,
    bandwidth: Union[str, float, int] = "scott",
    weights=None,
    kernel: str = "gaussian",
    backend: str = "auto",
    n_resamples: int = 200,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None,
    method: str = "percentile",
    return_bootstrap_samples: bool = False,
    batch_size: int = 1024,
) -> KDEBootstrapResult:
    """Backward-compatible bootstrap CI wrapper for KDE.

    This wrapper preserves the original API and delegates to
    ``kde_confidence_interval(method='bootstrap')``.
    """
    return kde_confidence_interval(
        samples,
        points,
        bandwidth=bandwidth,
        weights=weights,
        kernel=kernel,
        backend=backend,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        random_state=random_state,
        method="bootstrap",
        bootstrap_method=method,
        return_bootstrap_samples=return_bootstrap_samples,
        batch_size=batch_size,
    )


def kde_confidence_interval(
    samples,
    points,
    *,
    bandwidth: Union[str, float, int] = "scott",
    weights=None,
    kernel: str = "gaussian",
    backend: str = "auto",
    n_resamples: int = 200,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None,
    method: str = "normal",
    bootstrap_method: str = "percentile",
    return_bootstrap_samples: bool = False,
    batch_size: int = 1024,
) -> KDEBootstrapResult:
    """Estimate pointwise KDE confidence intervals.

    Supported methods:
    - ``normal``: asymptotic normal approximation (fast path, 1D Gaussian).
    - ``bootstrap``: non-parametric bootstrap percentile intervals.
    """
    method_name = str(method).strip().lower()
    if method_name not in ("normal", "bootstrap"):
        raise ValueError("method must be one of: 'normal', 'bootstrap'")

    bootstrap_method_name = str(bootstrap_method).strip().lower()
    if bootstrap_method_name != "percentile":
        raise ValueError("bootstrap_method must be 'percentile'")

    level = _validate_confidence_level(confidence_level)
    n_boot = _validate_n_resamples(n_resamples)
    model = fit_kde(
        samples,
        bandwidth=bandwidth,
        weights=weights,
        kernel=kernel,
        backend=backend,
    )

    xp = _get_xp(model.backend_)
    points_2d = _as_points_2d(points, model.n_features_, xp)
    estimate = np.asarray(_to_numpy(model.pdf(points_2d, batch_size=batch_size)), dtype=np.float64)

    points_np = np.asarray(_to_numpy(points_2d), dtype=np.float64)
    if model.n_features_ == 1 and points_np.ndim == 2 and int(points_np.shape[1]) == 1:
        points_np = points_np.reshape(-1)

    if method_name == "normal":
        if model.n_features_ != 1 or model.kernel_ != "gaussian":
            raise ValueError("method='normal' currently supports only 1D Gaussian KDE")

        n_eff = float(_effective_sample_size(model.weights_, xp))
        if (not np.isfinite(n_eff)) or n_eff <= 0.0:
            raise ValueError("effective sample size must be finite and positive")

        cov11 = float(_to_float_scalar(model.covariance_[0, 0]))
        if (not np.isfinite(cov11)) or cov11 <= 0.0:
            raise ValueError("covariance must be positive for normal CI")

        h = math.sqrt(cov11)
        if (not np.isfinite(h)) or h <= 0.0:
            raise ValueError("bandwidth scale must be positive for normal CI")

        r_kernel = 1.0 / (2.0 * math.sqrt(math.pi))
        var = np.maximum(estimate, 0.0) * (r_kernel / (n_eff * h))
        var = np.maximum(var, 0.0)
        se = np.sqrt(var)

        z = float(NormalDist().inv_cdf(0.5 + 0.5 * level))
        lower = np.maximum(estimate - z * se, 0.0)
        upper = estimate + z * se

        return KDEBootstrapResult(
            points=points_np,
            estimate=estimate,
            lower=lower,
            upper=upper,
            confidence_level=level,
            n_resamples=0,
            random_state=random_state,
            kernel=_normalize_kernel_name(kernel),
            backend=model.backend_,
            metadata={
                "method": method_name,
                "bandwidth": bandwidth,
                "batch_size": int(batch_size),
                "n_features": int(model.n_features_),
                "n_eff": float(n_eff),
            },
            bootstrap_samples=None,
        )

    samples_np = np.asarray(_to_numpy(_as_samples_2d(samples, xp)), dtype=np.float64)
    n_samples = int(samples_np.shape[0])

    weights_np = np.asarray(_to_numpy(_normalize_weights(weights, n_samples, xp)), dtype=np.float64)
    rng = np.random.default_rng(random_state)
    boot_samples = np.empty((n_boot, estimate.size), dtype=np.float64)

    use_fast_1d_numpy = (
        (xp is np)
        and (model.n_features_ == 1)
        and (model.kernel_ == "gaussian")
        and np.isfinite(float(model.bandwidth_factor_))
        and (float(model.bandwidth_factor_) > 0.0)
    )

    if use_fast_1d_numpy:
        samples_1d = samples_np.reshape(-1)
        points_1d = points_np.reshape(-1)
        bw_factor = float(model.bandwidth_factor_)
        sqrt_2pi = math.sqrt(2.0 * math.pi)

        for i in range(n_boot):
            idx = rng.choice(n_samples, size=n_samples, replace=True, p=weights_np)
            sampled_x = samples_1d[idx]
            sampled_w = weights_np[idx]

            sampled_w_sum = float(np.sum(sampled_w))
            if sampled_w_sum <= 0.0:
                raise ValueError("bootstrap weights must sum to a positive value")
            sampled_w = sampled_w / sampled_w_sum

            mean = float(np.sum(sampled_w * sampled_x))
            centered = sampled_x - mean
            denom = 1.0 - float(np.sum(sampled_w * sampled_w))
            if denom <= 1e-15:
                raise ValueError("effective degrees of freedom is too small for covariance estimation")

            var = float(np.sum(sampled_w * (centered * centered)) / denom)
            if (not np.isfinite(var)) or var <= 0.0:
                var = float(np.finfo(np.float64).tiny)
            jitter = max(var * 1e-12, 1e-12)
            var = var + jitter

            scaled_var = var * (bw_factor * bw_factor)
            inv_scalar = 1.0 / scaled_var
            inv_norm_const = 1.0 / (math.sqrt(scaled_var) * sqrt_2pi)

            diff = points_1d[:, None] - sampled_x[None, :]
            diff *= diff
            diff *= (-0.5 * inv_scalar)
            np.exp(diff, out=diff)
            boot_samples[i, :] = (diff @ sampled_w) * inv_norm_const
    else:
        for i in range(n_boot):
            idx = rng.choice(n_samples, size=n_samples, replace=True, p=weights_np)
            sampled_data = samples_np[idx]
            sampled_weights = weights_np[idx]
            sampled_weight_sum = float(np.sum(sampled_weights))
            if sampled_weight_sum <= 0.0:
                raise ValueError("bootstrap weights must sum to a positive value")
            sampled_weights = sampled_weights / sampled_weight_sum

            sampled_data_backend = sampled_data if xp is np else xp.asarray(sampled_data, dtype=xp.float64)
            sampled_weights_backend = sampled_weights if xp is np else xp.asarray(sampled_weights, dtype=xp.float64)

            boot_model = fit_kde(
                sampled_data_backend,
                bandwidth=bandwidth,
                weights=sampled_weights_backend,
                kernel=kernel,
                backend=model.backend_,
            )
            boot_samples[i, :] = np.asarray(_to_numpy(boot_model.pdf(points_2d, batch_size=batch_size)), dtype=np.float64)

    alpha = 1.0 - level
    lower = np.quantile(boot_samples, alpha / 2.0, axis=0)
    upper = np.quantile(boot_samples, 1.0 - alpha / 2.0, axis=0)

    return KDEBootstrapResult(
        points=points_np,
        estimate=estimate,
        lower=lower,
        upper=upper,
        confidence_level=level,
        n_resamples=n_boot,
        random_state=random_state,
        kernel=_normalize_kernel_name(kernel),
        backend=model.backend_,
        metadata={
            "method": method_name,
            "bootstrap_method": bootstrap_method_name,
            "bandwidth": bandwidth,
            "batch_size": int(batch_size),
            "n_features": int(model.n_features_),
        },
        bootstrap_samples=boot_samples if return_bootstrap_samples else None,
    )


__all__ = [
    "KernelDensityEstimator",
    "KDE",
    "KDEBootstrapResult",
    "fit_kde",
    "kde_pdf",
    "kde_confidence_interval",
    "kde_bootstrap_confidence_interval",
]
