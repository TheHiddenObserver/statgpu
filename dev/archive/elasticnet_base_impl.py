"""
Elastic Net Base Implementation - Reference Archive

This file contains the original base implementation of Elastic Net with FISTA solver.
It is kept for reference and educational purposes.

The current production implementation is in:
    statgpu/linear_model/_elasticnet.py

Key differences from the optimized version:
1. No fused CuPy kernels (@cp.fuse)
2. No torch.compile() optimization
3. Uses lazy import pattern for optional backends (removed in production)

This implementation is slower but may be useful for:
- Understanding the core algorithm
- Debugging issues with optimized kernels
- Running on systems without CuPy or with old PyTorch versions

Status: ARCHIVED (2026-04-18)
"""

from typing import Optional, Union
import numpy as np


class ElasticNetBaseImpl:
    """
    Reference implementation of Elastic Net base solver.

    This is NOT the production class - it's archived for reference.
    The production ElasticNet class is in statgpu/linear_model/_elasticnet.py
    """

    def __init__(self):
        self.lipschitz_L = None
        self.stopping = 'coef_delta'
        self.alpha = 1.0
        self.l1_ratio = 0.5
        self.max_iter = 1000
        self.tol = 1e-4
        self.fit_intercept = True
        self.n_iter_ = 0

    def _soft_threshold(self, x, gamma):
        """Standard soft thresholding operator."""
        return np.sign(x) * np.maximum(np.abs(x) - gamma, 0)

    def _soft_threshold_elastic(self, x, gamma, l2_scale):
        """Elastic Net soft thresholding with L2 scaling."""
        return self._soft_threshold(x, gamma) / l2_scale

    def _fit_cpu_base(self, X, y, sample_weight=None):
        """
        Base CPU FISTA solver (slower, reference implementation).
        """
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
            sqrt_sw = np.sqrt(sample_weight)
            X = X * sqrt_sw[:, np.newaxis]
            y = y * sqrt_sw

        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)
            X_centered = X - X_mean
            y_centered = y - y_mean
        else:
            X_centered = X
            y_mean = 0.0
            y_centered = y

        if y.ndim == 1:
            y_centered = y_centered.reshape(-1, 1)

        # Precompute XtX and Xty
        XtX = X_centered.T @ X_centered
        Xty = X_centered.T @ y_centered.flatten()

        # Elastic Net parameters
        alpha = float(self.alpha)
        l1_ratio = float(self.l1_ratio)
        l2_ratio = 1.0 - l1_ratio

        # Lipschitz constant
        if self.lipschitz_L is not None:
            L = float(self.lipschitz_L)
        else:
            try:
                eig_max = np.linalg.eigvalsh(XtX)[-1]
                L = float(eig_max / n_samples)
            except Exception:
                L_frob = float(np.sum(X_centered ** 2) / n_samples)
                L = L_frob

        if L <= 0:
            thresh = alpha * l1_ratio
            l2_scale = 1.0 + alpha * l2_ratio
            coef = self._soft_threshold_elastic(np.zeros(n_features), thresh, l2_scale)
            self.n_iter_ = 0
        else:
            step = 1.0 / L
            thresh = alpha * l1_ratio * step
            l2_scale = 1.0 + alpha * l2_ratio * step

            # FISTA variables
            coef = np.zeros(n_features)
            y_k = coef.copy()
            t_k = 1.0

            for iteration in range(self.max_iter):
                coef_old = coef.copy()

                # Gradient of RSS ONLY
                grad = (XtX @ y_k - Xty) / n_samples

                # Proximal step
                w_tilde = y_k - step * grad
                coef = self._soft_threshold_elastic(w_tilde, thresh, l2_scale)

                # Momentum update
                t_new = (1.0 + np.sqrt(1.0 + 4.0 * (t_k ** 2))) / 2.0
                beta = (t_k - 1.0) / t_new
                y_k = coef + beta * (coef - coef_old)
                t_k = t_new

                # Convergence check
                if self.stopping == "kkt":
                    grad_rss = (XtX @ coef - Xty) / n_samples
                    grad_l2 = alpha * l2_ratio * coef
                    sign_coef = np.sign(coef)
                    sign_coef[coef == 0] = 0
                    kkt_violation = np.zeros(n_features)
                    for j in range(n_features):
                        if coef[j] != 0:
                            kkt_violation[j] = np.abs(grad_rss[j] + grad_l2[j] + alpha * l1_ratio * sign_coef[j])
                        else:
                            kkt_violation[j] = max(0, np.abs(grad_rss[j] + grad_l2[j]) - alpha * l1_ratio)
                    violation = np.max(kkt_violation)
                    if violation < self.tol:
                        self.n_iter_ = iteration + 1
                        break
                else:
                    if np.sum(np.abs(coef - coef_old)) < self.tol:
                        self.n_iter_ = iteration + 1
                        break
            else:
                self.n_iter_ = self.max_iter

        # Compute intercept
        if self.fit_intercept:
            intercept = float(y_mean - X_mean @ coef)
        else:
            intercept = 0.0

        return coef, intercept

    def _fit_gpu_base(self, X_centered, y_centered, n_samples, n_features):
        """
        Base GPU implementation without fused kernels (fallback).
        """
        import cupy as cp

        # Precompute XtX / Xty
        XtX = X_centered.T @ X_centered
        Xty = X_centered.T @ y_centered

        # Elastic Net parameters
        alpha = float(self.alpha)
        l1_ratio = float(self.l1_ratio)
        l2_ratio = 1.0 - l1_ratio

        # Lipschitz constant
        if self.lipschitz_L is not None:
            L = cp.array(float(self.lipschitz_L), dtype=X_centered.dtype)
        else:
            try:
                eigvals = cp.linalg.eigvalsh(XtX)
                L = eigvals[-1] / n_samples
            except Exception:
                L_frob = cp.sum(X_centered ** 2) / n_samples
                L = L_frob

        if L <= 0:
            return cp.zeros(n_features, dtype=X_centered.dtype), 0

        step = 1.0 / L
        thresh = alpha * l1_ratio * step
        l2_scale = 1.0 + alpha * l2_ratio * step

        # FISTA variables
        coef = cp.zeros(n_features, dtype=X_centered.dtype)
        y_k = coef.copy()
        t_k = cp.array(1.0, dtype=X_centered.dtype)

        for iteration in range(self.max_iter):
            coef_old = coef.copy()

            # Gradient of RSS ONLY
            grad = (XtX @ y_k - Xty) / n_samples

            # Proximal step
            w_tilde = y_k - step * grad
            coef = self._soft_threshold_elastic_cupy(w_tilde, thresh, l2_scale)

            # Momentum update
            t_new = (1 + cp.sqrt(1 + 4 * (t_k ** 2))) / 2
            beta = (t_k - 1) / t_new
            y_k = coef + beta * (coef - coef_old)
            t_k = t_new

            # Convergence check
            if self.stopping == "kkt":
                grad_rss = (XtX @ coef - Xty) / n_samples
                grad_l2 = alpha * l2_ratio * coef
                sign_coef = cp.sign(coef)
                sign_coef[coef == 0] = 0
                kkt_nonzero = cp.abs(grad_rss + grad_l2 + alpha * l1_ratio * sign_coef)
                kkt_zero = cp.maximum(cp.abs(grad_rss + grad_l2) - alpha * l1_ratio, 0)
                kkt_violation = cp.where(coef != 0, kkt_nonzero, kkt_zero)
                violation = cp.max(kkt_violation)
                if violation < self.tol:
                    return coef, iteration + 1
            else:
                if cp.sum(cp.abs(coef - coef_old)) < self.tol:
                    return coef, iteration + 1

        return coef, self.max_iter

    def _soft_threshold_elastic_cupy(self, x, gamma, l2_scale):
        """Elastic Net soft thresholding for CuPy."""
        import cupy as cp
        return cp.sign(x) * cp.maximum(cp.abs(x) - gamma, 0) / l2_scale

    def _fit_torch_base(self, X_centered, y_centered, n_samples, n_features):
        """
        Base Torch implementation without torch.compile() (fallback).
        """
        import torch

        # Precompute XtX / Xty
        XtX = X_centered.T @ X_centered
        Xty = X_centered.T @ y_centered

        # Elastic Net parameters
        alpha = float(self.alpha)
        l1_ratio = float(self.l1_ratio)
        l2_ratio = 1.0 - l1_ratio

        # Lipschitz constant
        if self.lipschitz_L is not None:
            L = torch.tensor(float(self.lipschitz_L), dtype=X_centered.dtype, device=X_centered.device)
        else:
            try:
                eigvals = torch.linalg.eigvalsh(XtX)
                L = eigvals[-1] / n_samples
            except Exception:
                L_frob = torch.sum(X_centered ** 2) / n_samples
                L = L_frob

        if L <= 0:
            return torch.zeros(n_features, dtype=X_centered.dtype, device=X_centered.device), 0

        step = 1.0 / L
        thresh = alpha * l1_ratio * step
        l2_scale = 1.0 + alpha * l2_ratio * step

        # FISTA variables
        coef = torch.zeros(n_features, dtype=X_centered.dtype, device=X_centered.device)
        y_k = coef.clone()
        t_k = torch.tensor(1.0, dtype=X_centered.dtype, device=X_centered.device)

        for iteration in range(self.max_iter):
            coef_old = coef.clone()

            # Gradient of RSS ONLY
            grad = (XtX @ y_k - Xty) / n_samples

            # Proximal step
            w_tilde = y_k - step * grad
            coef = self._soft_threshold_elastic_torch(w_tilde, thresh, l2_scale)

            # Momentum update
            t_new = (1.0 + torch.sqrt(1.0 + 4.0 * (t_k ** 2))) / 2.0
            beta = (t_k - 1.0) / t_new
            y_k = coef + beta * (coef - coef_old)
            t_k = t_new

            # Convergence check
            if self.stopping == "kkt":
                grad_rss = (XtX @ coef - Xty) / n_samples
                grad_l2 = alpha * l2_ratio * coef
                sign_coef = torch.sign(coef)
                sign_coef[coef == 0] = 0
                kkt_nonzero = torch.abs(grad_rss + grad_l2 + alpha * l1_ratio * sign_coef)
                kkt_zero = torch.maximum(torch.abs(grad_rss + grad_l2) - alpha * l1_ratio, torch.tensor(0.0, dtype=X_centered.dtype, device=X_centered.device))
                violation = torch.max(torch.where(coef != 0, kkt_nonzero, kkt_zero))
                if violation < self.tol:
                    return coef, iteration + 1
            else:
                if torch.sum(torch.abs(coef - coef_old)) < self.tol:
                    return coef, iteration + 1

        return coef, self.max_iter

    def _soft_threshold_elastic_torch(self, x, gamma, l2_scale):
        """Elastic Net soft thresholding for Torch."""
        import torch
        zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        return torch.sign(x) * torch.maximum(torch.abs(x) - gamma, zero) / l2_scale
