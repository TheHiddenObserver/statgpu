"""Principal component analysis."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from statgpu._base import BaseEstimator
from statgpu._config import Device
from statgpu.unsupervised._utils import backend_random_normal, check_2d_array, scalar_to_float, svd_flip_components


class PCA(BaseEstimator):
    """
    Principal component analysis with NumPy, CuPy, or Torch backends.

    Parameters
    ----------
    n_components : int or None, default=None
        Number of components to keep. ``None`` keeps all components.
    svd_solver : {'auto', 'full', 'covariance', 'randomized'}, default='auto'
        Solver used for the decomposition. ``'auto'`` uses covariance/eigh
        when ``n_samples >= n_features`` and full SVD otherwise. ``'randomized'``
        computes an approximate truncated SVD and is useful when only a small
        number of components is needed.
    whiten : bool, default=False
        When True, scale transformed components to unit variance.
    copy : bool, default=True
        Kept for sklearn-style API compatibility. Inputs are not modified.
    device : {'auto', 'cpu', 'cuda', 'torch'}, default='auto'
        Compute device.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        svd_solver: str = "auto",
        whiten: bool = False,
        copy: bool = True,
        random_state: Optional[int] = None,
        n_oversamples: int = 10,
        iterated_power: int = 2,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.n_components = n_components
        self.svd_solver = svd_solver
        self.whiten = whiten
        self.copy = copy
        self.random_state = random_state
        self.n_oversamples = n_oversamples
        self.iterated_power = iterated_power

    def _validate_params(self, n_samples: int, n_features: int):
        if self.svd_solver not in ("auto", "full", "covariance", "randomized"):
            raise ValueError("svd_solver must be one of: 'auto', 'full', 'covariance', 'randomized'")
        if not isinstance(self.n_oversamples, (int, np.integer)) or int(self.n_oversamples) < 0:
            raise ValueError("n_oversamples must be a non-negative integer")
        if not isinstance(self.iterated_power, (int, np.integer)) or int(self.iterated_power) < 0:
            raise ValueError("iterated_power must be a non-negative integer")
        max_components = min(n_samples, n_features)
        if self.n_components is None:
            n_components = max_components
        else:
            if not isinstance(self.n_components, (int, np.integer)):
                raise ValueError("n_components must be None or a positive integer")
            n_components = int(self.n_components)
            if n_components < 1 or n_components > max_components:
                raise ValueError(
                    f"n_components must be in [1, {max_components}] for the input shape"
                )
        solver = self.svd_solver
        if solver == "auto":
            solver = "covariance" if n_samples >= n_features else "full"
        return n_components, solver

    def _randomized_svd(self, backend, X_centered, n_components: int):
        n_samples, n_features = X_centered.shape
        n_random = min(n_features, n_components + int(self.n_oversamples))
        omega = backend_random_normal(backend, self.random_state, size=(n_features, n_random))
        Q, _ = backend.qr(backend.matmul(X_centered, omega))
        for _ in range(int(self.iterated_power)):
            Q, _ = backend.qr(backend.matmul(X_centered.T, Q))
            Q, _ = backend.qr(backend.matmul(X_centered, Q))
        B = backend.matmul(Q.T, X_centered)
        _, singular_values_all, vh = backend.svd(B, full_matrices=False)
        return singular_values_all[:n_components], svd_flip_components(backend, vh[:n_components])

    def fit(self, X, y=None):
        backend = self._get_backend()
        X_arr = backend.asarray(X, dtype=backend.float64)
        check_2d_array(X_arr)
        n_samples, n_features = X_arr.shape
        if n_samples < 2:
            raise ValueError("PCA requires at least two samples")

        n_components, solver = self._validate_params(n_samples, n_features)
        mean = backend.mean(X_arr, axis=0, keepdims=False)

        if solver == "covariance":
            gram = backend.matmul(X_arr.T, X_arr)
            mean_col = backend.reshape(mean, (n_features, 1))
            cov = (gram - float(n_samples) * backend.matmul(mean_col, mean_col.T)) / float(n_samples - 1)
            eigenvalues, eigenvectors = backend.eigh(cov)
            order = backend.flip(backend.argsort(eigenvalues, axis=0), axis=0)
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]
            explained_variance = backend.maximum(eigenvalues[:n_components], 0.0)
            components = eigenvectors[:, :n_components].T
            components = svd_flip_components(backend, components)
            singular_values = backend.sqrt(explained_variance * float(n_samples - 1))
            total_var = backend.sum(backend.diag(cov))
        elif solver == "randomized":
            X_centered = X_arr - mean
            singular_values, components = self._randomized_svd(backend, X_centered, n_components)
            explained_variance = (singular_values ** 2) / float(n_samples - 1)
            total_var = backend.sum(X_centered * X_centered) / float(n_samples - 1)
        else:
            X_centered = X_arr - mean
            _, singular_values_all, vh = backend.svd(X_centered, full_matrices=False)
            components = svd_flip_components(backend, vh[:n_components])
            singular_values = singular_values_all[:n_components]
            explained_variance = (singular_values ** 2) / float(n_samples - 1)
            total_var = backend.sum(X_centered * X_centered) / float(n_samples - 1)

        if scalar_to_float(total_var) > 0.0:
            explained_variance_ratio = explained_variance / total_var
        else:
            explained_variance_ratio = explained_variance * 0.0

        self.components_ = components
        self.mean_ = mean
        self.explained_variance_ = explained_variance
        self.explained_variance_ratio_ = explained_variance_ratio
        self.singular_values_ = singular_values
        self.n_components_ = int(n_components)
        self.n_features_in_ = int(n_features)
        self._backend_name = backend.name
        self._fitted = True
        return self

    def transform(self, X):
        self._check_is_fitted()
        backend = self._get_backend()
        X_arr = backend.asarray(X, dtype=backend.float64)
        check_2d_array(X_arr)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X_arr.shape[1]} features, expected {self.n_features_in_}")
        X_transformed = backend.matmul(X_arr - self.mean_, self.components_.T)
        if self.whiten:
            scale = backend.sqrt(self.explained_variance_)
            X_transformed = X_transformed / scale
        return X_transformed

    def fit_transform(self, X, y=None):
        return self.fit(X, y=y).transform(X)

    def inverse_transform(self, X):
        self._check_is_fitted()
        backend = self._get_backend()
        X_arr = backend.asarray(X, dtype=backend.float64)
        if getattr(X_arr, "ndim", None) != 2:
            raise ValueError("X must be a 2D array")
        if X_arr.shape[1] != self.n_components_:
            raise ValueError(f"X has {X_arr.shape[1]} components, expected {self.n_components_}")
        if self.whiten:
            X_arr = X_arr * backend.sqrt(self.explained_variance_)
        return backend.matmul(X_arr, self.components_) + self.mean_

    def predict(self, X):
        """Alias for transform, provided for BaseEstimator compatibility."""
        return self.transform(X)

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params.update(
            {
                "n_components": self.n_components,
                "svd_solver": self.svd_solver,
                "whiten": self.whiten,
                "copy": self.copy,
                "random_state": self.random_state,
                "n_oversamples": self.n_oversamples,
                "iterated_power": self.iterated_power,
            }
        )
        return params
