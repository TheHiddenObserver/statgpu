"""Kernel density estimation with NumPy/CuPy backends."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import NormalDist
from typing import Any, Dict, Optional, Union

import numpy as np

from statgpu._base import BaseEstimator
from statgpu.backends import xp_asarray, xp_empty, xp_maximum


def _xp_max(x, **kwargs):
    """Backend-safe max that returns values only (torch.max returns (values, indices))."""
    # Use amax if available (returns values only, works for torch/cupy/numpy)
    if hasattr(x, 'amax'):
        return x.amax(**kwargs)
    # Fallback: check if max returns (values, indices) tuple
    result = x.max(**kwargs)
    if hasattr(result, 'values'):
        return result.values
    return result
from statgpu.nonparametric.kernel_smoothing._bandwidth_selection import select_bandwidth

from statgpu.nonparametric.kernel_smoothing._kernel_common import (
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
    _torch_device_from_data,
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
    """Weighted kernel density estimation for continuous observations.

    Parameters
    ----------
    bandwidth : str or float, default='scott'
        Positive finite dimensionless covariance factor b: H=b**2*Sigma.
        A number is not an absolute width in input units. String selectors:
        scott, silverman, nrd0, nrd, ucv, bcv, sj, sj-ste, sj-dpi.
    weights : array-like, shape (n_samples,), optional
        Finite nonnegative training weights with positive sum, normalized
        internally. Covariance needs more than one effectively weighted row.
        Supply weights here, not as a keyword to fit.
    kernel : str, default='gaussian'
        gaussian, rectangular, triangular, epanechnikov, biweight, triweight,
        cosine, or optcosine. The last two require one-dimensional samples.
    backend : {'auto', 'numpy', 'cupy', 'torch'}, default='auto'
        Array library. auto consults estimator/global device settings, but an
        explicit library can override device. See current exceptions below.
    device : {'auto', 'cpu', 'cuda', 'torch'}, default='auto'
        Requested computation device. The intended convention requires an
        explicit accelerator or an error; current routing does not always
        enforce it, even with matching backend and device settings.
    n_jobs : int or None, default=None
        Shared estimator option; does not create a parallel query pool.
    gpu_memory_cleanup : bool, default=False
        Best-effort GPU-cache cleanup, preserving arrays needed for prediction.

    Attributes
    ----------
    samples_ : backend array, shape (n_samples, n_features)
        Float64 training observations.
    weights_ : backend array, shape (n_samples,)
        Normalized training weights.
    bandwidth_factor_ : float
        Selected covariance factor.
    bandwidth_info_ : BandwidthSelectionResult or None
        Selector diagnostics; None for an explicit numeric factor.
    covariance_, inv_covariance_ : backend arrays, shape (n_features, n_features)
        Stabilized kernel covariance H and its inverse.
    norm_const_, inv_norm_const_ : float
        Kernel-volume normalization and its reciprocal.
    n_samples_, n_features_ : int
        Training dimensions.
    kernel_, backend_ : str
        Resolved kernel and array library.

    Notes
    -----
    With NumPy or Torch CPU inputs, device='torch' and backend='auto' or
    'torch' can fit and predict on Torch CPU. Explicit backend='torch' can
    also run on CPU with device='cuda'; backend='numpy' overrides either
    accelerator request. Inspect samples_ and prediction arrays, using
    Torch .device/.is_cuda or CuPy .device; NumPy arrays are on CPU. Neither
    the configured device nor backend_ proves CUDA placement. For a
    predictable CPU path, use NumPy inputs with device='cpu', backend='numpy'.

    fit accepts finite real samples (n,) or (n,p), n >= 2; numeric calculations
    use float64. pdf/predict/__call__ return backend-native (n_query,) density;
    logpdf/score_samples return log density. score is mean log density, not R2.
    For large coordinate offsets, center samples and queries by the same
    training-derived offset. Remove zero-weight rows before fitting when
    using Gaussian log density: zero-weight rows can currently destabilize
    its log-sum-exp evaluation in the tails. Rerunning a string bandwidth
    selector after deletion can change its chosen factor. Choose new smoothing
    on the positive-weight rows, or preserve a previously chosen factor by
    refitting them and their retained weights with numeric
    bandwidth=original.bandwidth_factor_. This preserves the specified kernel
    covariance and density, not the scientific suitability of the original
    selector choice. Density interval helpers return
    host NumPy arrays and have separate pointwise coverage restrictions.
    Torch currently rejects explicit weights with TypeError, including weights
    supplied internally by bootstrap intervals. Use backend='numpy' and CPU
    arrays for weighted fitting or bootstrap. Multivariate Torch queries must
    be two-dimensional, including (1,p) for one point; the vector check fails.
    """

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
        """Fit from finite real samples with shape (n,) or (n,p), n >= 2.

        Returns self. Optional y is unused for density estimation, but is still
        subject to shared finite-input validation. Training weights belong to the
        constructor. Calls after a failed refit must not be mistaken for results
        of that attempted fit; successfully refit before using the new data.
        """
        backend_name = self._resolve_backend_name(X)
        xp = _get_xp(backend_name)

        samples_2d = _as_samples_2d(X, xp)
        n_samples, n_features = int(samples_2d.shape[0]), int(samples_2d.shape[1])

        device = _torch_device_from_data(samples_2d)
        self._torch_device = device
        weights_1d = _normalize_weights(self.weights, n_samples, xp, device=device, ref_arr=samples_2d)
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
        if not self._gpu_memory_cleanup:
            return
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass

    def _cleanup_torch_memory(self):
        if not self._gpu_memory_cleanup:
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

        out = xp_empty((n_points,), xp.float64, xp, ref_arr=points_2d)

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
            quad = xp_maximum(quad, 0.0, xp)

            if use_log_sum_exp:
                if is_gaussian:
                    log_kernels = -0.5 * quad
                    log_kernels_max = _xp_max(log_kernels, axis=1, keepdims=True)
                    log_sum = log_kernels_max[:, 0] + xp.log(
                        xp.sum(xp.exp(log_kernels - log_kernels_max) * self.weights_[None, :], axis=1)
                    )
                    out[start:stop] = xp.exp(log_sum) * self.inv_norm_const_
                else:
                    kernels = _kernel_values_from_quad(quad, self.kernel_, xp)
                    log_sum = self._log_weighted_kernel_sum(kernels, xp)
                    out[start:stop] = xp.where(
                        xp.isfinite(log_sum),
                        xp.exp(log_sum) * self.inv_norm_const_,
                        0.0,
                    )
            else:
                kernels = _kernel_values_from_quad(quad, self.kernel_, xp)
                out[start:stop] = (kernels @ self.weights_) * self.inv_norm_const_

        return out

    def _log_weighted_kernel_sum(self, kernels, xp):
        """Compute row-wise log(weighted kernel sum) with exact zero-density handling.

        Parameters
        ----------
        kernels : array-like
            Kernel values for each query/sample pair.
        xp : module
            Backend array module used for the computation.

        Returns
        -------
        array-like
            Per-row log-sum values, or ``-inf`` when all weighted kernel terms are zero.
        """
        positive_weight_mask = self.weights_[None, :] > 0.0
        positive_term_mask = (kernels > 0.0) & positive_weight_mask
        safe_kernels = xp.where(positive_term_mask, kernels, 1.0)
        safe_weights = xp.where(positive_weight_mask, self.weights_[None, :], 1.0)
        log_terms = xp.where(
            positive_term_mask,
            xp.log(safe_kernels) + xp.log(safe_weights),
            float("-inf"),
        )
        log_terms_max = _xp_max(log_terms, axis=1, keepdims=True)
        finite_rows = xp.isfinite(log_terms_max[:, 0])
        # ``where`` evaluates both branches on NumPy/CuPy, so protect the
        # subtraction and logarithm explicitly to avoid inf-inf and log(0)
        # warnings for valid zero-density rows of compact-support kernels.
        safe_max = xp.where(finite_rows[:, None], log_terms_max, 0.0)
        shifted = xp.where(
            finite_rows[:, None],
            log_terms - safe_max,
            float("-inf"),
        )
        exp_sum = xp.sum(xp.exp(shifted), axis=1)
        safe_exp_sum = xp.where(finite_rows, exp_sum, 1.0)
        finite_result = safe_max[:, 0] + xp.log(safe_exp_sum)
        return xp.where(finite_rows, finite_result, float("-inf"))

    def pdf(self, points, *, batch_size: int = 1024):
        """Evaluate density, returning a backend-native (n_query,) array.

        points must be finite with shape (n_query,n_features). A vector means
        many one-feature queries, or one multivariate query of matching length.
        Multivariate Torch queries must instead use an explicit (n_query,p)
        matrix; the vector shape check currently raises TypeError.
        batch_size is a positive integer query-batch size (default 1024).
        Compact-support kernels return zero where no sample contributes.
        """
        self._require_fitted()
        xp = _get_xp(self.backend_)
        points_2d = _as_points_2d(points, self.n_features_, xp, ref_arr=self.samples_)
        result = self._evaluate_density(points_2d, batch_size=int(batch_size), xp=xp)
        self._cleanup_cuda_memory()
        self._cleanup_torch_memory()
        return result

    def _evaluate_log_density(self, points_2d, *, batch_size: int, xp):
        """Evaluate log-density in log domain (avoids underflow for high dimensions)."""
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        n_points = int(points_2d.shape[0])

        s_quad = self._samples_quad_
        log_norm = math.log(self.inv_norm_const_) if self.inv_norm_const_ > 0.0 else float("-inf")
        is_gaussian = self.kernel_ == "gaussian"

        out = xp_empty((n_points,), xp.float64, xp, ref_arr=points_2d)

        for start in range(0, points_2d.shape[0], int(batch_size)):
            stop = min(start + int(batch_size), points_2d.shape[0])
            q = points_2d[start:stop]

            q_proj = q @ self.inv_covariance_
            q_quad = xp.sum(q_proj * q, axis=1)
            cross = q_proj @ self.samples_.T
            quad = q_quad[:, None] + s_quad[None, :] - 2.0 * cross
            quad = xp_maximum(quad, 0.0, xp)

            if is_gaussian:
                log_kernels = -0.5 * quad
                log_kernels_max = _xp_max(log_kernels, axis=1, keepdims=True)
                log_kernels_shifted = log_kernels - log_kernels_max
                log_sum = log_kernels_max[:, 0] + xp.log(
                    xp.sum(xp.exp(log_kernels_shifted) * self.weights_[None, :], axis=1)
                )
                out[start:stop] = log_sum + log_norm
            else:
                kernels = _kernel_values_from_quad(quad, self.kernel_, xp)
                log_sum = self._log_weighted_kernel_sum(kernels, xp)
                out[start:stop] = xp.where(
                    xp.isfinite(log_sum),
                    log_sum + log_norm,
                    float("-inf"),
                )

        return out

    def logpdf(self, points, *, batch_size: int = 1024):
        """Evaluate log density with the same query and batch contract as pdf.

        Returns backend-native (n_query,). Compact-support kernels return -inf
        where density is zero. Gaussian tails use log-domain evaluation; remove
        zero-weight training rows first and center large-offset coordinates.
        Rerunning a string selector after deletion can change the bandwidth.
        To preserve existing smoothing, refit positive-weight rows and their
        weights with numeric bandwidth=original.bandwidth_factor_; otherwise
        choose a new bandwidth on the filtered data. Preserving the factor
        does not validate the original selector choice.
        """
        self._require_fitted()
        xp = _get_xp(self.backend_)
        points_2d = _as_points_2d(points, self.n_features_, xp, ref_arr=self.samples_)
        result = self._evaluate_log_density(points_2d, batch_size=int(batch_size), xp=xp)
        self._cleanup_cuda_memory()
        self._cleanup_torch_memory()
        return result

    def __call__(self, points, *, batch_size: int = 1024):
        """Alias for pdf(points, batch_size=batch_size), returning density."""
        return self.pdf(points, batch_size=batch_size)

    def to_numpy_metadata(self):
        """Return fitted bandwidth, covariance, weights, and dimension metadata.

        Array values are transferred to host NumPy arrays. Requires fitting.
        """
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
        """Return density at X; alias for pdf using its default batch size.
        """
        return self.pdf(X)

    def score_samples(self, X):
        """Return log density at X using the default evaluation batch size.
        """
        return self.logpdf(X)

    def score(self, X, y=None):
        """Return the unweighted mean query log density as a Python float.

        Higher values are better on the same held-out data and measurement scale.
        Optional y is unused but remains subject to finite-input validation.
        """
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
    backend : {'auto', 'numpy', 'cupy', 'torch'}, default='auto'
        Compute backend. 'auto' selects from the estimator's configured
        device/backend rather than inferring from input array types. An
        explicit Torch backend selects the library, not necessarily CUDA.

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
    """Fit a KDE and evaluate density or log density at points.

    samples, bandwidth, weights, kernel, and backend have the fit_kde
    meanings; all supported kernels are accepted. points follows the fitted
    estimator query-shape contract. return_log=False returns density;
    True returns log density. batch_size=1024 is a positive query-batch size.
    The returned array is backend-native with shape (n_query,). Reuse fit_kde
    for repeated queries rather than fitting again on each helper call.
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
    """NumPy results for pointwise normal or bootstrap KDE intervals.

    Attributes
    ----------
    points : numpy.ndarray, shape (n_query,) or (n_query,n_features)
        Evaluation points; a one-feature result uses a vector.
    estimate, lower, upper : numpy.ndarray, shape (n_query,)
        Original density estimate and pointwise confidence bounds. These are
        density units, not probabilities or simultaneous confidence bands.
    confidence_level : float
        Requested marginal confidence level.
    n_resamples : int
        Actual bootstrap count, or 0 for a normal-method interval.
    random_state : int or None
        Supplied bootstrap seed; unused by the normal method.
    kernel : str
        Resolved KDE kernel name.
    backend : str
        Fitted array-library label, not result-array placement; all result
        arrays are NumPy even when the fit used a GPU library.
    metadata : dict
        Method, bandwidth, batch size and feature count, with method-specific
        diagnostics such as normal-method n_eff or bootstrap_method.
    bootstrap_samples : numpy.ndarray or None, default=None
        Shape (n_resamples,n_query) when requested for bootstrap. Always None
        for normal intervals, even if return_bootstrap_samples=True.
    """

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
        """Convert result arrays to lists; omit absent bootstrap_samples.

        Other result fields are retained, including the metadata dictionary.
        """
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
    """Return pointwise percentile bootstrap intervals for a KDE.

    Arguments have the kde_confidence_interval meanings, except this wrapper
    accepts only method='percentile' and selects the general function's
    method='bootstrap'. It returns KDEBootstrapResult with NumPy arrays.
    The intervals are neither bias-corrected nor simultaneous. See the
    selector-refitting and nonuniform-weight limitations in that function.
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
    """Estimate pointwise confidence intervals for a smoothed density.

    Parameters
    ----------
    samples : array-like, shape (n_samples,) or (n_samples, n_features)
        Finite real observations, at least two.
    points : array-like
        Finite query points following KernelDensityEstimator.pdf shapes.
    bandwidth : str or float, default='scott'
        Selector or positive covariance factor, with the fit_kde meaning.
    weights : array-like, shape (n_samples,), optional
        Nonnegative normalized fitting weights. The bootstrap uses weights
        both as sampling probabilities and again on sampled rows; establish
        that this scheme fits the study design before using unequal weights.
    kernel : str, default='gaussian'
        KDE kernel. The normal method requires a one-dimensional Gaussian KDE.
    backend : {'auto', 'numpy', 'cupy', 'torch'}, default='auto'
        Fitting array library; result arrays are always host NumPy arrays.
    n_resamples : int, default=200
        Positive bootstrap replicate count, validated even for normal intervals.
    confidence_level : float, default=0.95
        Finite value strictly between zero and one; do not pass NaN.
    random_state : int or None, default=None
        Seed for bootstrap sampling; unused by the normal method.
    method : {'normal', 'bootstrap'}, default='normal'
        Asymptotic pointwise normal interval (lower bound clipped to zero),
        or percentile bootstrap interval.
    bootstrap_method : {'percentile'}, default='percentile'
        Only supported bootstrap procedure, required even on the normal path.
    return_bootstrap_samples : bool, default=False
        Include the (n_resamples,n_query) replicate matrix for bootstrap.
        Normal intervals always have bootstrap_samples=None and n_resamples=0.
    batch_size : int, default=1024
        Positive query batch size; a NumPy bootstrap fast path can evaluate
        all queries together.

    Returns
    -------
    result : KDEBootstrapResult
        NumPy points, estimate, lower, upper, optional bootstrap_samples, and
        method/bandwidth/backend metadata. Bounds are pointwise, not a
        simultaneous band or a prediction interval for future observations.

    Notes
    -----
    Neither method corrects smoothing bias or dependence. The NumPy 1D
    Gaussian bootstrap fast path holds the original selected factor fixed;
    other paths refit string selectors in each replicate. A numeric factor
    is fixed on every path, but sample covariance and absolute width still
    change with each resample. Equal-weight independent observations give
    the introductory interpretation; do not assume all weighted bootstrap
    designs or preceding tuning uncertainty are handled.
    Bootstrap currently raises TypeError on Torch even when weights=None,
    because each resampled fit supplies explicit weights. Use backend='numpy'
    with CPU arrays for bootstrap. Unweighted one-dimensional Gaussian normal
    intervals are a separate supported Torch path.
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

            sampled_data_backend = sampled_data if xp is np else xp_asarray(sampled_data, dtype=xp.float64, xp=xp, ref_arr=points_2d)
            sampled_weights_backend = sampled_weights if xp is np else xp_asarray(sampled_weights, dtype=xp.float64, xp=xp, ref_arr=points_2d)

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
