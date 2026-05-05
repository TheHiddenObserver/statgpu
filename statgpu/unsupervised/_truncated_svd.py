"""Truncated singular value decomposition."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from statgpu._base import BaseEstimator
from statgpu._config import Device
from statgpu.unsupervised._utils import check_2d_array, randomized_svd, reject_sparse, scalar_to_float


class TruncatedSVD(BaseEstimator):
    """
    Dense truncated SVD with NumPy, CuPy, or Torch backends.

    Unlike PCA, this estimator does not center the input matrix.
    """

    def __init__(
        self,
        n_components: int = 2,
        algorithm: str = "randomized",
        n_iter: int = 5,
        n_oversamples: int = 10,
        random_state: Optional[int] = None,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.n_components = n_components
        self.algorithm = algorithm
        self.n_iter = n_iter
        self.n_oversamples = n_oversamples
        self.random_state = random_state

    def _validate_params(self, n_samples: int, n_features: int):
        if not isinstance(self.n_components, (int, np.integer)):
            raise ValueError("n_components must be a positive integer")
        n_components = int(self.n_components)
        max_components = min(n_samples, n_features)
        if n_components < 1 or n_components > max_components:
            raise ValueError(f"n_components must be in [1, {max_components}] for the input shape")
        if self.algorithm not in ("randomized", "full"):
            raise ValueError("algorithm must be one of: 'randomized', 'full'")
        if not isinstance(self.n_iter, (int, np.integer)) or int(self.n_iter) < 0:
            raise ValueError("n_iter must be a non-negative integer")
        if not isinstance(self.n_oversamples, (int, np.integer)) or int(self.n_oversamples) < 0:
            raise ValueError("n_oversamples must be a non-negative integer")
        return n_components

    def fit(self, X, y=None):
        reject_sparse(X, "TruncatedSVD")
        backend = self._get_backend()
        X_arr = backend.asarray(X, dtype=backend.float64)
        check_2d_array(X_arr)
        n_samples, n_features = X_arr.shape
        n_components = self._validate_params(n_samples, n_features)

        if self.algorithm == "full":
            _, singular_values_all, vh = backend.svd(X_arr, full_matrices=False)
            singular_values = singular_values_all[:n_components]
            components = vh[:n_components]
        else:
            singular_values, components = randomized_svd(
                backend,
                X_arr,
                n_components=n_components,
                n_oversamples=int(self.n_oversamples),
                n_iter=int(self.n_iter),
                random_state=self.random_state,
            )

        transformed = backend.matmul(X_arr, components.T)
        transformed_mean = backend.mean(transformed, axis=0, keepdims=True)
        explained_variance = backend.mean((transformed - transformed_mean) ** 2, axis=0)
        feature_mean = backend.mean(X_arr, axis=0, keepdims=True)
        total_variance = backend.sum(backend.mean((X_arr - feature_mean) ** 2, axis=0))
        if scalar_to_float(total_variance) > 0.0:
            explained_variance_ratio = explained_variance / total_variance
        else:
            explained_variance_ratio = explained_variance * 0.0

        self.components_ = components
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
        reject_sparse(X, "TruncatedSVD")
        backend = self._get_backend()
        X_arr = backend.asarray(X, dtype=backend.float64)
        check_2d_array(X_arr)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X_arr.shape[1]} features, expected {self.n_features_in_}")
        return backend.matmul(X_arr, self.components_.T)

    def fit_transform(self, X, y=None):
        return self.fit(X, y=y).transform(X)

    def inverse_transform(self, X):
        self._check_is_fitted()
        backend = self._get_backend()
        X_arr = backend.asarray(X, dtype=backend.float64)
        check_2d_array(X_arr)
        if X_arr.shape[1] != self.n_components_:
            raise ValueError(f"X has {X_arr.shape[1]} components, expected {self.n_components_}")
        return backend.matmul(X_arr, self.components_)

    def predict(self, X):
        """Alias for transform, provided for BaseEstimator compatibility."""
        return self.transform(X)

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params.update(
            {
                "n_components": self.n_components,
                "algorithm": self.algorithm,
                "n_iter": self.n_iter,
                "n_oversamples": self.n_oversamples,
                "random_state": self.random_state,
            }
        )
        return params
