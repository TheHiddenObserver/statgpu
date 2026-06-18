"""Kernel PCA with GPU acceleration."""

from __future__ import annotations

__all__ = ["KernelPCA"]

from typing import Optional, Union

import numpy as np

from statgpu._base import BaseEstimator
from statgpu._config import Device
from statgpu.backends import _LINALG_ERRORS, _to_float_scalar, _to_numpy, xp_asarray
from statgpu.nonparametric.kernel_methods._kernels import pairwise_kernels


class KernelPCA(BaseEstimator):
    """Kernel Principal Component Analysis.

    Performs nonlinear dimensionality reduction by computing the
    eigendecomposition of a kernel matrix in feature space.

    Parameters
    ----------
    n_components : int, default=2
        Number of components to extract.
    kernel : str or callable, default='rbf'
        Kernel function name or callable.
    gamma : float, optional
        Kernel coefficient (for rbf, poly, etc.).
    degree : int, default=3
        Polynomial degree (for poly kernel).
    coef0 : float, default=1
        Independent term (for poly and sigmoid kernels).
    alpha : float, default=1.0
        Regularization parameter.  Adds ``alpha * I`` to the kernel
        matrix before eigendecomposition for numerical stability.
    eigen_solver : str, default='auto'
        Eigensolver to use: ``'auto'`` or ``'dense'``.
    device : str or Device, default='auto'
        Computation device.

    Attributes
    ----------
    lambdas_ : ndarray, shape (n_components,)
        Eigenvalues of the centered kernel matrix.
    alphas_ : ndarray, shape (n_samples, n_components)
        Eigenvectors of the centered kernel matrix (normalized).
    X_fit_ : ndarray, shape (n_samples, n_features)
        Training data (stored for transform).
    n_samples_ : int
        Number of training samples.
    n_features_in_ : int
        Number of input features.
    """

    def __init__(
        self,
        n_components: int = 2,
        kernel: str = "rbf",
        gamma: Optional[float] = None,
        degree: int = 3,
        coef0: float = 1,
        alpha: float = 1.0,
        eigen_solver: str = "auto",
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.alpha = alpha
        self.eigen_solver = eigen_solver

    def fit(self, X, y=None):
        """Fit the Kernel PCA model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : ignored

        Returns
        -------
        self
        """
        backend = self._get_backend(backend="auto")
        xp = backend.xp

        X_arr = xp_asarray(X, dtype=xp.float64, xp=xp)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        n_samples = int(X_arr.shape[0])
        n_features = int(X_arr.shape[1])
        self.n_features_in_ = n_features
        self.n_samples_ = n_samples

        n_comp = min(self.n_components, n_samples)

        kernel_params = self._get_kernel_params()

        # Compute kernel matrix K
        K = pairwise_kernels(X_arr, metric=self.kernel, xp=xp, **kernel_params)

        # Center the kernel matrix efficiently using row/column means
        K_col_means = xp.mean(K, axis=0)  # (n,)
        K_row_means = xp.mean(K, axis=1, keepdims=True)  # (n, 1)
        K_mean = _to_float_scalar(xp.mean(K))
        K_centered = K - K_col_means[None, :] - K_row_means + K_mean

        # Cache training kernel statistics for transform()
        self._K_train_col_means_ = _to_numpy(K_col_means)
        self._K_train_mean_ = float(K_mean)

        # Add regularization
        if self.alpha > 0:
            eye = xp.eye(n_samples, dtype=xp.float64)
            if hasattr(K, 'is_cuda'):
                eye = eye.to(device=K.device)
            K_centered = K_centered + self.alpha * eye

        # Eigendecomposition
        try:
            eigenvalues, eigenvectors = xp.linalg.eigh(K_centered)
        except _LINALG_ERRORS:
            K_np = _to_numpy(K_centered)
            eigvals_np, eigvecs_np = np.linalg.eigh(K_np)
            eigenvalues = xp.asarray(eigvals_np, dtype=xp.float64)
            eigenvectors = xp.asarray(eigvecs_np, dtype=xp.float64)

        # Sort by descending eigenvalue
        idx = xp.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Keep top n_components
        eigenvalues = eigenvalues[:n_comp]
        eigenvectors = eigenvectors[:, :n_comp]

        # Normalize eigenvectors: alpha_k = v_k / sqrt(lambda_k)
        # (only for positive eigenvalues)
        norms = xp.sqrt(xp.maximum(eigenvalues, 1e-12))
        alphas = eigenvectors / norms[None, :]

        self.lambdas_ = _to_numpy(eigenvalues)
        self.alphas_ = _to_numpy(alphas)
        self.X_fit_ = _to_numpy(X_arr)
        self._kernel_params = kernel_params
        self._fitted = True
        return self

    def transform(self, X):
        """Project X into the kernel PCA space.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        X_transformed : ndarray, shape (n_samples, n_components)
        """
        self._check_is_fitted()
        backend = self._get_backend(backend="auto")
        xp = backend.xp

        X_arr = xp_asarray(X, dtype=xp.float64, xp=xp)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        X_fit_arr = xp.asarray(self.X_fit_, dtype=xp.float64)
        if hasattr(X_arr, 'is_cuda'):
            X_fit_arr = X_fit_arr.to(device=X_arr.device)

        # Compute kernel between X and training data
        K_test = pairwise_kernels(X_arr, X_fit_arr, metric=self.kernel,
                                  xp=xp, **self._kernel_params)

        # Center the test kernel using cached training kernel statistics
        K_train_col_means = xp.asarray(self._K_train_col_means_, dtype=xp.float64)
        K_train_mean = self._K_train_mean_
        if hasattr(X_arr, 'is_cuda'):
            K_train_col_means = K_train_col_means.to(device=X_arr.device)

        # Correct out-of-sample centering:
        # K_test_centered = K_test - mean(K_train, axis=0) - mean(K_test, axis=1) + mean(K_train)
        K_test_centered = (
            K_test
            - K_train_col_means[None, :]
            - xp.mean(K_test, axis=1, keepdims=True)
            + K_train_mean
        )

        # Project
        alphas = xp.asarray(self.alphas_, dtype=xp.float64)
        if hasattr(X_arr, 'is_cuda'):
            alphas = alphas.to(device=X_arr.device)

        X_transformed = K_test_centered @ alphas

        # GPU in → GPU out, CPU in → CPU out
        return X_transformed

    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        self.fit(X, y)
        # For training data: K_centered @ alphas_ = V * sqrt(lambda)
        # alphas_ = V / sqrt(lambda), so alphas_ * lambda = V * sqrt(lambda)
        backend = self._get_backend(backend="auto")
        xp = backend.xp
        result = np.asarray(self.alphas_) * np.maximum(self.lambdas_, 0)[None, :]
        return xp.asarray(result, dtype=xp.float64)

    def predict(self, X):
        """Alias for transform (required by BaseEstimator)."""
        return self.transform(X)

    def _get_kernel_params(self):
        """Collect kernel parameters."""
        params = {}
        if self.gamma is not None:
            params["gamma"] = self.gamma
        if self.kernel in ("poly", "polynomial"):
            params["degree"] = self.degree
            params["coef0"] = self.coef0
        elif self.kernel == "sigmoid":
            params["coef0"] = self.coef0
        return params

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params["n_components"] = self.n_components
        params["kernel"] = self.kernel
        params["gamma"] = self.gamma
        params["degree"] = self.degree
        params["coef0"] = self.coef0
        params["alpha"] = self.alpha
        params["eigen_solver"] = self.eigen_solver
        return params

    def set_params(self, **params):
        for key in ["n_components", "kernel", "gamma", "degree", "coef0", "alpha", "eigen_solver"]:
            if key in params:
                setattr(self, key, params.pop(key))
        if params:
            super().set_params(**params)
        return self
