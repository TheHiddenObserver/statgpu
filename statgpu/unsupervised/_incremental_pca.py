"""Incremental principal component analysis."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from statgpu._base import BaseEstimator
from statgpu._config import Device
from statgpu.unsupervised._utils import check_2d_array, reject_sparse, scalar_to_float, svd_flip_components


class IncrementalPCA(BaseEstimator):
    """Dense incremental PCA with NumPy, CuPy, or Torch backends."""

    def __init__(
        self,
        n_components: Optional[int] = None,
        batch_size: Optional[int] = None,
        whiten: bool = False,
        copy: bool = True,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.n_components = n_components
        self.batch_size = batch_size
        self.whiten = whiten
        self.copy = copy

    def _validate_params(self, n_samples: int, n_features: int, first_pass: bool):
        if self.n_components is None:
            n_components = n_features
        else:
            if not isinstance(self.n_components, (int, np.integer)) or int(self.n_components) < 1:
                raise ValueError("n_components must be None or a positive integer")
            n_components = int(self.n_components)
        if n_components > n_features:
            raise ValueError("n_components must be less than or equal to n_features")
        if first_pass and n_samples < n_components:
            raise ValueError("first partial_fit batch must contain at least n_components samples")
        if self.batch_size is not None:
            if not isinstance(self.batch_size, (int, np.integer)) or int(self.batch_size) < 1:
                raise ValueError("batch_size must be None or a positive integer")
        return n_components

    def _update_mean_var(self, backend, batch, batch_mean, batch_var):
        batch_count = int(batch.shape[0])
        if not getattr(self, "_fitted", False):
            return batch_mean, batch_var, batch_count
        old_count = int(self.n_samples_seen_)
        new_count = old_count + batch_count
        old_mean = self.mean_
        old_var = self.var_
        new_mean = (float(old_count) * old_mean + float(batch_count) * batch_mean) / float(new_count)
        old_ss = float(old_count) * (old_var + (old_mean - new_mean) ** 2)
        batch_ss = float(batch_count) * (batch_var + (batch_mean - new_mean) ** 2)
        new_var = (old_ss + batch_ss) / float(new_count)
        return new_mean, new_var, new_count

    def partial_fit(self, X, y=None):
        reject_sparse(X, "IncrementalPCA")
        backend = self._get_backend()
        X_arr = backend.asarray(X, dtype=backend.float64)
        check_2d_array(X_arr)
        n_samples, n_features = X_arr.shape
        first_pass = not getattr(self, "_fitted", False)
        n_components = self._validate_params(n_samples, n_features, first_pass=first_pass)
        if not first_pass and n_features != self.n_features_in_:
            raise ValueError(f"X has {n_features} features, expected {self.n_features_in_}")

        batch_mean = backend.mean(X_arr, axis=0)
        batch_var = backend.mean((X_arr - batch_mean) ** 2, axis=0)
        new_mean, new_var, new_count = self._update_mean_var(backend, X_arr, batch_mean, batch_var)
        X_centered = X_arr - batch_mean

        if first_pass:
            matrix = X_centered
        else:
            old_count = int(self.n_samples_seen_)
            old_basis = self.singular_values_[:, None] * self.components_
            mean_correction = np.sqrt(float(old_count * n_samples) / float(new_count)) * (self.mean_ - batch_mean)
            matrix = backend.concatenate([old_basis, X_centered, backend.reshape(mean_correction, (1, n_features))], axis=0)

        _, singular_values_all, vh = backend.svd(matrix, full_matrices=False)
        components = svd_flip_components(backend, vh[:n_components])
        singular_values = singular_values_all[:n_components]
        if new_count > 1:
            explained_variance = (singular_values ** 2) / float(new_count - 1)
            total_var = backend.sum(new_var) * float(new_count) / float(new_count - 1)
        else:
            explained_variance = singular_values * 0.0
            total_var = backend.sum(new_var)
        if scalar_to_float(total_var) > 0.0:
            explained_variance_ratio = explained_variance / total_var
        else:
            explained_variance_ratio = explained_variance * 0.0

        self.components_ = components
        self.mean_ = new_mean
        self.var_ = new_var
        self.explained_variance_ = explained_variance
        self.explained_variance_ratio_ = explained_variance_ratio
        self.singular_values_ = singular_values
        self.n_components_ = int(n_components)
        self.n_features_in_ = int(n_features)
        self.n_samples_seen_ = int(new_count)
        self._backend_name = backend.name
        self._fitted = True
        return self

    def fit(self, X, y=None):
        reject_sparse(X, "IncrementalPCA")
        backend = self._get_backend()
        X_arr = backend.asarray(X, dtype=backend.float64)
        check_2d_array(X_arr)
        n_samples, n_features = X_arr.shape
        batch_size = int(self.batch_size) if self.batch_size is not None else min(n_samples, max(1, 5 * n_features))
        self._fitted = False
        for start in range(0, n_samples, batch_size):
            self.partial_fit(X_arr[start : start + batch_size])
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
            X_transformed = X_transformed / backend.sqrt(self.explained_variance_)
        return X_transformed

    def fit_transform(self, X, y=None):
        return self.fit(X, y=y).transform(X)

    def inverse_transform(self, X):
        self._check_is_fitted()
        backend = self._get_backend()
        X_arr = backend.asarray(X, dtype=backend.float64)
        check_2d_array(X_arr)
        if X_arr.shape[1] != self.n_components_:
            raise ValueError(f"X has {X_arr.shape[1]} components, expected {self.n_components_}")
        if self.whiten:
            X_arr = X_arr * backend.sqrt(self.explained_variance_)
        return backend.matmul(X_arr, self.components_) + self.mean_

    def predict(self, X):
        return self.transform(X)

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params.update(
            {
                "n_components": self.n_components,
                "batch_size": self.batch_size,
                "whiten": self.whiten,
                "copy": self.copy,
            }
        )
        return params
