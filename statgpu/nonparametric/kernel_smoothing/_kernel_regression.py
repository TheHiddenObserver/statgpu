"""Kernel regression with NumPy/CuPy backends."""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np

from statgpu._base import BaseEstimator
from statgpu.backends import (
    _torch_dev,
    xp_asarray,
    xp_empty,
    xp_eye,
    xp_full,
    xp_maximum,
    xp_ones,
)
from statgpu.nonparametric.kernel_smoothing._bandwidth_selection import select_bandwidth

from statgpu.nonparametric.kernel_smoothing._kernel_common import (
    _auto_backend_from_device,
    _as_points_2d,
    _as_samples_2d,
    _effective_sample_size,
    _get_xp,
    _kernel_values_from_quad,
    _normalize_kernel_name,
    _normalize_regression_name,
    _normalize_weights,
    _stable_inv_and_det,
    _to_float_scalar,
    _to_numpy,
    _weighted_covariance,
)


class KernelRegression(BaseEstimator):
    """sklearn-style kernel regression model (Nadaraya-Watson or local-linear)."""

    def __init__(
        self,
        *,
        bandwidth: Union[str, float, int] = "scott",
        weights=None,
        kernel: str = "gaussian",
        regression: str = "nw",
        kernel_metric: str = "full",
        bandwidth_per_feature=None,
        backend: str = "auto",
        device: str = "auto",
        n_jobs: Optional[int] = None,
        batch_size: int = 1024,
        min_effective_weight: float = 1e-12,
        gpu_memory_cleanup: bool = False,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.bandwidth = bandwidth
        self.weights = weights
        self.kernel = kernel
        self.regression = regression
        self.kernel_metric = kernel_metric
        self.bandwidth_per_feature = bandwidth_per_feature
        self.backend = backend
        self.batch_size = int(batch_size)
        self.min_effective_weight = float(min_effective_weight)
        self.gpu_memory_cleanup = gpu_memory_cleanup

    def _resolve_backend_name(self, X, y) -> str:
        backend_name = str(self.backend).strip().lower()
        if backend_name != "auto":
            return backend_name
        return _auto_backend_from_device(self._get_compute_device().value)

    def fit(self, X, y):
        """Fit kernel regression and cache model state on this instance."""
        backend_name = self._resolve_backend_name(X, y)
        xp = _get_xp(backend_name)

        samples_2d = _as_samples_2d(X, xp)
        n_samples, n_features = int(samples_2d.shape[0]), int(samples_2d.shape[1])

        targets_2d, target_was_1d = _as_targets_2d(y, n_samples, xp, ref_arr=samples_2d)
        n_targets = int(targets_2d.shape[1])

        weights_1d = _normalize_weights(self.weights, n_samples, xp, ref_arr=samples_2d)
        n_eff = _effective_sample_size(weights_1d, xp)

        kernel_name = _normalize_kernel_name(self.kernel)
        regression_name = _normalize_regression_name(self.regression)
        metric_name = _normalize_kernel_metric_name(self.kernel_metric)
        if kernel_name in ("cosine", "optcosine") and n_features != 1:
            raise ValueError(f"kernel='{kernel_name}' currently supports only 1D samples")

        data_cov = _weighted_covariance(samples_2d, weights_1d, xp)

        bandwidth_vec = _as_bandwidth_per_feature(self.bandwidth_per_feature, n_features, xp, ref_arr=data_cov)
        if bandwidth_vec is not None and metric_name != "diagonal":
            raise ValueError("bandwidth_per_feature requires kernel_metric='diagonal'")

        bw_result = None
        if bandwidth_vec is None:
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
                    estimator="kernel_regression",
                    targets=targets_2d,
                    regression=regression_name,
                    kernel=kernel_name,
                )
                factor = float(bw_result.factor)
            else:
                factor = float(self.bandwidth)
                if (not np.isfinite(factor)) or factor <= 0.0:
                    raise ValueError("bandwidth factor must be a finite positive scalar")
            base_cov = data_cov if metric_name == "full" else xp.diag(xp.diag(data_cov))
            scaled_cov = base_cov * (factor**2)
        else:
            tiny = float(np.finfo(np.float64).tiny)
            diag_cov = xp.diag(data_cov)
            diag_sd = xp.sqrt(xp_maximum(diag_cov, tiny, xp))
            rel = bandwidth_vec / diag_sd
            factor = float(_to_float_scalar(xp.mean(rel)))
            if (not np.isfinite(factor)) or factor <= 0.0:
                factor = 1.0
            scaled_cov = xp.diag(bandwidth_vec * bandwidth_vec)

        inv_cov, _, stable_cov = _stable_inv_and_det(scaled_cov, xp)
        target_mean = xp.sum(targets_2d * weights_1d[:, None], axis=0)

        self.samples_ = samples_2d
        self.targets_ = targets_2d
        self.weights_ = weights_1d
        self.bandwidth_factor_ = factor
        self.bandwidth_info_ = bw_result
        self.covariance_ = stable_cov
        self.inv_covariance_ = inv_cov
        self.kernel_ = kernel_name
        self.kernel_metric_ = metric_name
        self.bandwidth_per_feature_ = bandwidth_vec
        self.n_features_ = n_features
        self.n_targets_ = n_targets
        self.n_samples_ = n_samples
        self.backend_ = backend_name
        self.regression_ = regression_name
        self.target_mean_ = target_mean
        self.target_was_1d_ = target_was_1d
        # Cache reusable terms for prediction hot paths.
        self._samples_proj_ = self.samples_ @ self.inv_covariance_
        self._samples_quad_ = xp.sum(self._samples_proj_ * self.samples_, axis=1)

        self._ll_use_vectorized_moments_ = bool(self.n_features_ <= 24)
        self._ll_sample_xx_flat_ = None
        self._ll_sample_xy_flat_ = None
        self._ll_eye_p1_ = None
        self._ll_ones_col_ = None
        if self.regression_ == "local_linear":
            if self._ll_use_vectorized_moments_:
                self._ll_sample_xx_flat_ = (
                    (self.samples_[:, :, None] * self.samples_[:, None, :]).reshape(
                        self.n_samples_,
                        self.n_features_ * self.n_features_,
                    )
                )
                self._ll_sample_xy_flat_ = (
                    (self.samples_[:, :, None] * self.targets_[:, None, :]).reshape(
                        self.n_samples_,
                        self.n_features_ * self.n_targets_,
                    )
                )
                self._ll_eye_p1_ = xp_eye(self.n_features_ + 1, xp.float64, xp, ref_arr=self.samples_)
            else:
                self._ll_ones_col_ = xp_ones((self.n_samples_, 1), xp.float64, xp, ref_arr=self.samples_)
        self._fitted = True
        return self

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

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

    def _evaluate_nadaraya_watson(
        self,
        points_2d,
        *,
        batch_size: int,
        min_effective_weight: float,
        xp,
    ):
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be a positive integer")

        min_weight = float(min_effective_weight)
        if (not np.isfinite(min_weight)) or min_weight <= 0.0:
            raise ValueError("min_effective_weight must be a finite positive scalar")

        min_weight = max(min_weight, float(np.finfo(np.float64).tiny))

        samples_2d = self.samples_
        targets_2d = self.targets_
        weights_1d = self.weights_
        inv_cov = self.inv_covariance_
        fallback = self.target_mean_
        kernel_name = self.kernel_

        n_points = int(points_2d.shape[0])
        n_features = int(samples_2d.shape[1])
        n_targets = int(targets_2d.shape[1])

        out = xp_empty((n_points, n_targets), xp.float64, xp, ref_arr=points_2d)

        if n_features == 1:
            samples_1d = samples_2d[:, 0]
            inv_scalar = inv_cov[0, 0]

            if kernel_name == "gaussian":
                scale = -0.5 * inv_scalar
                for start in range(0, n_points, int(batch_size)):
                    stop = min(start + int(batch_size), n_points)
                    q_1d = points_2d[start:stop, 0]

                    diff = q_1d[:, None] - samples_1d[None, :]
                    diff *= diff
                    diff *= scale
                    xp.exp(diff, out=diff)

                    weighted_kernels = diff * weights_1d[None, :]
                    denom = xp.sum(weighted_kernels, axis=1)
                    numer = weighted_kernels @ targets_2d

                    denom_safe = xp_maximum(denom, min_weight, xp)
                    pred = numer / denom_safe[:, None]
                    out[start:stop] = xp.where(denom[:, None] > min_weight, pred, fallback[None, :])

                return out

            for start in range(0, n_points, int(batch_size)):
                stop = min(start + int(batch_size), n_points)
                q_1d = points_2d[start:stop, 0]

                diff = q_1d[:, None] - samples_1d[None, :]
                quad = (diff * diff) * inv_scalar
                kernels = _kernel_values_from_quad(quad, kernel_name, xp)

                weighted_kernels = kernels * weights_1d[None, :]
                denom = xp.sum(weighted_kernels, axis=1)
                numer = weighted_kernels @ targets_2d

                denom_safe = xp_maximum(denom, min_weight, xp)
                pred = numer / denom_safe[:, None]
                out[start:stop] = xp.where(denom[:, None] > min_weight, pred, fallback[None, :])

            return out

        s_proj = self._samples_proj_
        s_quad = self._samples_quad_

        for start in range(0, n_points, int(batch_size)):
            stop = min(start + int(batch_size), n_points)
            q = points_2d[start:stop]

            q_proj = q @ inv_cov
            q_quad = xp.sum(q_proj * q, axis=1)
            cross = q_proj @ samples_2d.T
            quad = q_quad[:, None] + s_quad[None, :] - 2.0 * cross
            quad = xp_maximum(quad, 0.0, xp)

            kernels = _kernel_values_from_quad(quad, kernel_name, xp)
            weighted_kernels = kernels * weights_1d[None, :]

            denom = xp.sum(weighted_kernels, axis=1)
            numer = weighted_kernels @ targets_2d

            denom_safe = xp_maximum(denom, min_weight, xp)
            pred = numer / denom_safe[:, None]
            out[start:stop] = xp.where(denom[:, None] > min_weight, pred, fallback[None, :])

        return out

    def _evaluate_local_linear(
        self,
        points_2d,
        *,
        batch_size: int,
        min_effective_weight: float,
        xp,
    ):
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be a positive integer")

        min_weight = float(min_effective_weight)
        if (not np.isfinite(min_weight)) or min_weight <= 0.0:
            raise ValueError("min_effective_weight must be a finite positive scalar")

        min_weight = max(min_weight, float(np.finfo(np.float64).tiny))

        samples_2d = self.samples_
        targets_2d = self.targets_
        weights_1d = self.weights_
        inv_cov = self.inv_covariance_
        fallback = self.target_mean_
        kernel_name = self.kernel_

        n_points = int(points_2d.shape[0])
        n_samples = int(samples_2d.shape[0])
        n_features = int(samples_2d.shape[1])
        n_targets = int(targets_2d.shape[1])

        out = xp_empty((n_points, n_targets), xp.float64, xp, ref_arr=points_2d)

        if n_features == 1:
            samples_1d = samples_2d[:, 0]
            inv_scalar = inv_cov[0, 0]

            if kernel_name == "gaussian":
                scale = -0.5 * inv_scalar
                for start in range(0, n_points, int(batch_size)):
                    stop = min(start + int(batch_size), n_points)
                    q_1d = points_2d[start:stop, 0]

                    delta = q_1d[:, None] - samples_1d[None, :]
                    quad = delta * delta
                    quad *= scale
                    xp.exp(quad, out=quad)

                    weighted_kernels = quad * weights_1d[None, :]
                    s0 = xp.sum(weighted_kernels, axis=1)
                    s1 = xp.sum(weighted_kernels * delta, axis=1)
                    s2 = xp.sum(weighted_kernels * delta * delta, axis=1)

                    t0 = weighted_kernels @ targets_2d
                    t1 = (weighted_kernels * delta) @ targets_2d

                    det = s0 * s2 - s1 * s1
                    det_thresh = min_weight * min_weight
                    use_ll = (s0 > min_weight) & (xp.abs(det) > det_thresh)

                    det_safe = xp.where(use_ll, det, 1.0)
                    pred_ll = (s2[:, None] * t0 - s1[:, None] * t1) / det_safe[:, None]

                    denom_safe = xp_maximum(s0, min_weight, xp)
                    pred_nw = t0 / denom_safe[:, None]

                    pred = xp.where(use_ll[:, None], pred_ll, pred_nw)
                    out[start:stop] = xp.where(s0[:, None] > min_weight, pred, fallback[None, :])

                return out

            for start in range(0, n_points, int(batch_size)):
                stop = min(start + int(batch_size), n_points)
                q_1d = points_2d[start:stop, 0]

                diff = q_1d[:, None] - samples_1d[None, :]
                quad = (diff * diff) * inv_scalar
                kernels = _kernel_values_from_quad(quad, kernel_name, xp)

                weighted_kernels = kernels * weights_1d[None, :]
                s0 = xp.sum(weighted_kernels, axis=1)
                s1 = xp.sum(weighted_kernels * diff, axis=1)
                s2 = xp.sum(weighted_kernels * diff * diff, axis=1)

                t0 = weighted_kernels @ targets_2d
                t1 = (weighted_kernels * diff) @ targets_2d

                det = s0 * s2 - s1 * s1
                det_thresh = min_weight * min_weight
                use_ll = (s0 > min_weight) & (xp.abs(det) > det_thresh)

                det_safe = xp.where(use_ll, det, 1.0)
                pred_ll = (s2[:, None] * t0 - s1[:, None] * t1) / det_safe[:, None]

                denom_safe = xp_maximum(s0, min_weight, xp)
                pred_nw = t0 / denom_safe[:, None]

                pred = xp.where(use_ll[:, None], pred_ll, pred_nw)
                out[start:stop] = xp.where(s0[:, None] > min_weight, pred, fallback[None, :])

            return out

        s_proj = self._samples_proj_
        s_quad = self._samples_quad_

        use_vectorized_moments = bool(self._ll_use_vectorized_moments_)
        sample_xx_flat = self._ll_sample_xx_flat_
        sample_xy_flat = self._ll_sample_xy_flat_
        eye_p1 = self._ll_eye_p1_
        ones_col = self._ll_ones_col_

        for start in range(0, n_points, int(batch_size)):
            stop = min(start + int(batch_size), n_points)
            q = points_2d[start:stop]

            q_proj = q @ inv_cov
            q_quad = xp.sum(q_proj * q, axis=1)
            cross = q_proj @ samples_2d.T
            quad = q_quad[:, None] + s_quad[None, :] - 2.0 * cross
            quad = xp_maximum(quad, 0.0, xp)

            kernels = _kernel_values_from_quad(quad, kernel_name, xp)
            weighted_kernels = kernels * weights_1d[None, :]
            denom = xp.sum(weighted_kernels, axis=1)
            numer_nw = weighted_kernels @ targets_2d

            denom_safe = xp_maximum(denom, min_weight, xp)
            pred_nw = numer_nw / denom_safe[:, None]

            if use_vectorized_moments:
                b = int(stop - start)

                wx = weighted_kernels @ samples_2d
                s1 = wx - q * denom[:, None]

                s2 = (weighted_kernels @ sample_xx_flat).reshape(b, n_features, n_features)
                q_outer = q[:, :, None] * q[:, None, :]
                s2 = (
                    s2
                    - wx[:, :, None] * q[:, None, :]
                    - q[:, :, None] * wx[:, None, :]
                    + denom[:, None, None] * q_outer
                )
                s2 = 0.5 * (s2 + xp.swapaxes(s2, 1, 2))

                t1 = (weighted_kernels @ sample_xy_flat).reshape(b, n_features, n_targets)
                t1 = t1 - q[:, :, None] * numer_nw[:, None, :]

                p1 = n_features + 1
                A_batch = xp_empty((b, p1, p1), xp.float64, xp, ref_arr=weighted_kernels)
                A_batch[:, 0, 0] = denom
                A_batch[:, 0, 1:] = s1
                A_batch[:, 1:, 0] = s1
                A_batch[:, 1:, 1:] = s2

                B_batch = xp_empty((b, p1, n_targets), xp.float64, xp, ref_arr=weighted_kernels)
                B_batch[:, 0, :] = numer_nw
                B_batch[:, 1:, :] = t1

                if _torch_dev(A_batch) is not None:
                    trace_batch = xp.sum(xp.diagonal(A_batch, dim1=1, dim2=2), axis=1)
                else:
                    trace_batch = xp.sum(xp.diagonal(A_batch, axis1=1, axis2=2), axis=1)
                ridge = xp_maximum(trace_batch / float(max(1, p1)) * 1e-10, 1e-10, xp)

                solved = False
                beta0 = None
                A_work = A_batch
                ridge_work = ridge
                for _ in range(6):
                    try:
                        beta = xp.linalg.solve(A_work, B_batch)
                        beta0 = beta[:, 0, :]
                        solved = True
                        break
                    except Exception:
                        A_work = A_work + ridge_work[:, None, None] * eye_p1[None, :, :]
                        ridge_work = ridge_work * 10.0

                if solved and beta0 is not None:
                    finite_mask = xp.all(xp.isfinite(beta0), axis=1)
                    use_ll = (denom > min_weight) & finite_mask
                    pred = xp.where(use_ll[:, None], beta0, pred_nw)
                else:
                    pred = pred_nw

                out[start:stop] = xp.where(denom[:, None] > min_weight, pred, fallback[None, :])
                continue

            for i in range(int(stop - start)):
                denom_i = _to_float_scalar(denom[i])
                if denom_i <= min_weight:
                    out[start + i] = fallback
                    continue

                wi = weighted_kernels[i]
                Xc = samples_2d - q[i]
                Z = xp.concatenate((ones_col, Xc), axis=1)

                zw = Z * wi[:, None]
                A = Z.T @ zw
                B = Z.T @ (wi[:, None] * targets_2d)

                beta = _solve_linear_system_with_ridge(A, B, xp)
                if beta is None:
                    out[start + i] = pred_nw[i]
                else:
                    out[start + i] = beta[0]

        return out

    def predict(
        self,
        points,
        *,
        batch_size: Optional[int] = None,
        min_effective_weight: Optional[float] = None,
    ):
        self._require_fitted()
        if batch_size is None:
            batch_size = int(self.batch_size)
        if min_effective_weight is None:
            min_effective_weight = float(self.min_effective_weight)

        xp = _get_xp(self.backend_)
        points_2d = _as_points_2d(points, self.n_features_, xp, ref_arr=self.samples_)
        if self.regression_ == "local_linear":
            preds_2d = self._evaluate_local_linear(
                points_2d,
                batch_size=int(batch_size),
                min_effective_weight=float(min_effective_weight),
                xp=xp,
            )
        else:
            preds_2d = self._evaluate_nadaraya_watson(
                points_2d,
                batch_size=int(batch_size),
                min_effective_weight=float(min_effective_weight),
                xp=xp,
            )

        self._cleanup_cuda_memory()
        self._cleanup_torch_memory()
        if self.target_was_1d_:
            return preds_2d.reshape(-1)
        return preds_2d

    def __call__(
        self,
        points,
        *,
        batch_size: Optional[int] = None,
        min_effective_weight: Optional[float] = None,
    ):
        return self.predict(
            points,
            batch_size=batch_size,
            min_effective_weight=min_effective_weight,
        )

    def score(self, X, y):
        pred = _to_numpy(self.predict(X)).reshape(-1)
        target = _to_numpy(y).reshape(-1)
        if pred.shape[0] != target.shape[0]:
            raise ValueError("X and y have incompatible lengths")

        ss_res = float(np.sum((target - pred) ** 2))
        y_mean = float(np.mean(target))
        ss_tot = float(np.sum((target - y_mean) ** 2))
        if ss_tot <= 0.0:
            return 0.0
        return 1.0 - (ss_res / ss_tot)

    def to_numpy_metadata(self):
        self._require_fitted()
        bandwidth_selection = None
        if hasattr(self.bandwidth_info_, "to_dict"):
            bandwidth_selection = self.bandwidth_info_.to_dict()
        return {
            "bandwidth_factor": float(self.bandwidth_factor_),
            "bandwidth_selection": bandwidth_selection,
            "bandwidth_per_feature": (
                None
                if self.bandwidth_per_feature_ is None
                else _to_numpy(self.bandwidth_per_feature_)
            ),
            "n_samples": int(self.n_samples_),
            "n_features": int(self.n_features_),
            "n_targets": int(self.n_targets_),
            "backend": self.backend_,
            "kernel": self.kernel_,
            "kernel_metric": self.kernel_metric_,
            "regression": self.regression_,
            "covariance": _to_numpy(self.covariance_),
            "inv_covariance": _to_numpy(self.inv_covariance_),
            "weights": _to_numpy(self.weights_),
            "target_mean": _to_numpy(self.target_mean_),
        }


class KernelRegressionRegressor(KernelRegression):
    """Alias class with sklearn-like naming for explicit regressor semantics."""


def fit_kernel_regression(
    samples,
    targets,
    *,
    bandwidth: Union[str, float, int] = "scott",
    weights=None,
    kernel: str = "gaussian",
    regression: str = "nw",
    kernel_metric: str = "full",
    bandwidth_per_feature=None,
    backend: str = "auto",
) -> KernelRegression:
    """Fit a kernel regressor (Nadaraya-Watson or local-linear)."""
    model = KernelRegression(
        bandwidth=bandwidth,
        weights=weights,
        kernel=kernel,
        regression=regression,
        kernel_metric=kernel_metric,
        bandwidth_per_feature=bandwidth_per_feature,
        backend=backend,
    )
    return model.fit(samples, targets)


def kernel_regression_predict(
    samples,
    targets,
    points,
    *,
    bandwidth: Union[str, float, int] = "scott",
    weights=None,
    kernel: str = "gaussian",
    regression: str = "nw",
    kernel_metric: str = "full",
    bandwidth_per_feature=None,
    backend: str = "auto",
    batch_size: int = 1024,
    min_effective_weight: float = 1e-12,
):
    """One-shot kernel regression prediction."""
    model = fit_kernel_regression(
        samples,
        targets,
        bandwidth=bandwidth,
        weights=weights,
        kernel=kernel,
        regression=regression,
        kernel_metric=kernel_metric,
        bandwidth_per_feature=bandwidth_per_feature,
        backend=backend,
    )
    return model.predict(
        points,
        batch_size=batch_size,
        min_effective_weight=min_effective_weight,
    )


def _as_targets_2d(targets, n_samples: int, xp, ref_arr=None):
    arr = xp_asarray(targets, dtype=xp.float64, xp=xp, ref_arr=ref_arr)
    if arr.ndim == 1:
        if int(arr.shape[0]) != int(n_samples):
            raise ValueError("targets must have the same number of rows as samples")
        return arr.reshape(-1, 1), True

    if arr.ndim == 2:
        if int(arr.shape[0]) != int(n_samples):
            raise ValueError("targets must have the same number of rows as samples")
        return arr, False

    raise ValueError("targets must be 1D or 2D")


def _normalize_kernel_metric_name(kernel_metric: str) -> str:
    name = str(kernel_metric).strip().lower()
    aliases = {
        "full": "full",
        "full_covariance": "full",
        "full-covariance": "full",
        "covariance": "full",
        "diag": "diagonal",
        "diagonal": "diagonal",
        "axis_aligned": "diagonal",
        "axis-aligned": "diagonal",
    }
    normalized = aliases.get(name)
    if normalized is None:
        raise ValueError("kernel_metric must be one of: 'full', 'diagonal'")
    return normalized


def _as_bandwidth_per_feature(bandwidth_per_feature, n_features: int, xp, ref_arr=None):
    if bandwidth_per_feature is None:
        return None

    bw = xp_asarray(bandwidth_per_feature, dtype=xp.float64, xp=xp, ref_arr=ref_arr).reshape(-1)
    if int(bw.size) == 1 and int(n_features) > 1:
        bw = xp_full(int(n_features), _to_float_scalar(bw[0]), xp.float64, xp, ref_arr=ref_arr)

    if int(bw.size) != int(n_features):
        raise ValueError("bandwidth_per_feature must match sample feature dimension")
    if _to_float_scalar(xp.sum(~xp.isfinite(bw))) > 0.0:
        raise ValueError("bandwidth_per_feature must contain only finite values")
    if _to_float_scalar(xp.min(bw)) <= 0.0:
        raise ValueError("bandwidth_per_feature must be strictly positive")

    return bw


def _solve_linear_system_with_ridge(A, B, xp):
    p1 = int(A.shape[0])
    eye = xp_eye(p1, xp.float64, xp, ref_arr=A)

    trace = _to_float_scalar(xp.trace(A))
    base = trace / float(max(1, p1)) if np.isfinite(trace) else 1.0
    ridge = max(base * 1e-10, 1e-10)

    A_work = A
    for _ in range(6):
        try:
            return xp.linalg.solve(A_work, B)
        except Exception:
            A_work = A_work + ridge * eye
            ridge *= 10.0
    return None


__all__ = [
    "KernelRegression",
    "KernelRegressionRegressor",
    "fit_kernel_regression",
    "kernel_regression_predict",
]
