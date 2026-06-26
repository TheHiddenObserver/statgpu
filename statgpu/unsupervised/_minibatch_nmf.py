"""Mini-batch non-negative matrix factorization."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from statgpu._base import BaseEstimator
from statgpu._config import Device
from statgpu.unsupervised._utils import backend_random_normal, check_2d_array, reject_sparse, scalar_to_float


class MiniBatchNMF(BaseEstimator):
    """Dense mini-batch NMF with multiplicative updates and Frobenius loss."""

    def __init__(
        self,
        n_components: Optional[int] = None,
        init: str = "random",
        batch_size: Optional[int] = None,
        max_iter: int = 200,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.n_components = n_components
        self.init = init
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _validate_params(self, n_samples: int, n_features: int):
        if self.n_components is None:
            n_components = min(n_samples, n_features)
        else:
            if not isinstance(self.n_components, (int, np.integer)) or int(self.n_components) < 1:
                raise ValueError("n_components must be None or a positive integer")
            n_components = int(self.n_components)
        if self.init != "random":
            raise NotImplementedError("MiniBatchNMF v1 only supports init='random'")
        if self.batch_size is not None:
            if not isinstance(self.batch_size, (int, np.integer)) or int(self.batch_size) < 1:
                raise ValueError("batch_size must be a positive integer or None")
        if not isinstance(self.max_iter, (int, np.integer)) or int(self.max_iter) < 1:
            raise ValueError("max_iter must be a positive integer")
        if float(self.tol) < 0.0:
            raise ValueError("tol must be non-negative")
        return n_components

    def _check_nonnegative(self, backend, X):
        if scalar_to_float(backend.min(X)) < 0.0:
            raise ValueError("MiniBatchNMF input X must be non-negative")

    def _init_matrix(self, backend, shape, scale, seed):
        return backend.abs(backend_random_normal(backend, seed, size=shape, scale=scale)) + 1e-8

    def _init_components(self, backend, X, n_components):
        mean = max(scalar_to_float(backend.mean(X)), np.finfo(np.float64).eps)
        scale = np.sqrt(mean / float(n_components))
        return self._init_matrix(backend, (n_components, X.shape[1]), scale, self.random_state)

    def _init_w(self, backend, X, seed=None):
        mean = max(scalar_to_float(backend.mean(X)), np.finfo(np.float64).eps)
        scale = np.sqrt(mean / float(self.n_components_))
        if seed is None:
            seed = self.random_state
        return self._init_matrix(backend, (X.shape[0], self.n_components_), scale, seed)

    def _init_w_from_data(self, backend, X, H, eps):
        numerator = backend.matmul(X, H.T)
        denominator = backend.reshape(backend.sum(H * H, axis=1) + eps, (1, H.shape[0]))
        return backend.maximum(numerator / denominator, eps)

    def _update_h(self, backend, X, W, H, eps):
        numerator = backend.matmul(W.T, X)
        denominator = backend.matmul(backend.matmul(W.T, W), H) + eps
        return H * numerator / denominator

    def _update_h_from_stats(self, backend, H, A, B, eps):
        denominator = backend.matmul(A, H) + eps
        return H * B / denominator

    def _update_h_from_stats_steps(self, backend, H, A, B, eps, n_steps):
        for _ in range(int(n_steps)):
            H = self._update_h_from_stats(backend, H, A, B, eps)
        return H

    def _update_w(self, backend, X, W, H, HtH, eps):
        numerator = backend.matmul(X, H.T)
        denominator = backend.matmul(W, HtH) + eps
        W *= numerator
        W /= denominator
        W = backend.maximum(W, eps)
        return W

    def _fit_batch_w(self, backend, X, H, HtH, n_steps):
        eps = np.finfo(np.float64).eps
        W = self._init_w_from_data(backend, X, H, eps)
        for _ in range(int(n_steps)):
            W = self._update_w(backend, X, W, H, HtH, eps)
        return W

    def _batch_stats(self, backend, X, W):
        return backend.matmul(W.T, W), backend.matmul(W.T, X)

    def _reconstruction_error(self, backend, X, W, H):
        residual = X - backend.matmul(W, H)
        return scalar_to_float(backend.sqrt(backend.sum(residual * residual)))

    def _reconstruction_error_from_stats(self, backend, x_sq, A, B, H):
        cross = backend.sum(B * H)
        quadratic = backend.sum(backend.matmul(A, H) * H)
        value = backend.maximum(x_sq - 2.0 * cross + quadratic, 0.0)
        return scalar_to_float(backend.sqrt(value))

    def partial_fit(self, X, y=None):
        reject_sparse(X, "MiniBatchNMF")
        backend = self._get_backend()
        X_arr = backend.asarray(X, dtype=backend.float64)
        check_2d_array(X_arr)
        self._check_nonnegative(backend, X_arr)
        n_samples, n_features = X_arr.shape
        if not getattr(self, "_fitted", False):
            n_components = self._validate_params(n_samples, n_features)
            self.n_components_ = int(n_components)
            self.n_features_in_ = int(n_features)
            self.components_ = self._init_components(backend, X_arr, n_components)
            self.n_iter_ = 0
            self._n_batches_seen_ = 0
            self._A_accum = backend.zeros((self.n_components_, self.n_components_), dtype=backend.float64)
            self._B_accum = backend.zeros((self.n_components_, self.n_features_in_), dtype=backend.float64)
            self._backend_name = backend.name
            self._fitted = True
        elif n_features != self.n_features_in_:
            raise ValueError(f"X has {n_features} features, expected {self.n_features_in_}")

        eps = np.finfo(np.float64).eps
        HtH = backend.matmul(self.components_, self.components_.T)
        W = self._fit_batch_w(backend, X_arr, self.components_, HtH, n_steps=3)
        A_batch, B_batch = self._batch_stats(backend, X_arr, W)
        self._A_accum = self._A_accum + A_batch
        self._B_accum = self._B_accum + B_batch
        self._n_batches_seen_ = int(self._n_batches_seen_) + 1
        self.components_ = self._update_h_from_stats_steps(
            backend, self.components_, self._A_accum, self._B_accum, eps, n_steps=3
        )
        self.n_iter_ = int(self.n_iter_) + 1
        self.reconstruction_err_ = self._reconstruction_error(backend, X_arr, W, self.components_)
        return self

    def fit(self, X, y=None):
        reject_sparse(X, "MiniBatchNMF")
        backend = self._get_backend()
        X_arr = backend.asarray(X, dtype=backend.float64)
        check_2d_array(X_arr)
        self._check_nonnegative(backend, X_arr)
        n_samples, n_features = X_arr.shape
        n_components = self._validate_params(n_samples, n_features)
        self.n_components_ = int(n_components)
        self.n_features_in_ = int(n_features)
        self.components_ = self._init_components(backend, X_arr, n_components)
        self.n_iter_ = 0
        self._n_batches_seen_ = 0
        self._backend_name = backend.name
        self._fitted = True

        # Auto-size batch: large enough for GPU efficiency, small enough for mini-batch benefit
        if self.batch_size is not None:
            batch_size = min(int(self.batch_size), n_samples)
        else:
            batch_size = min(n_samples, max(20000, n_samples // 5))

        eps = np.finfo(np.float64).eps
        previous_delta = None
        last_A = None
        last_B = None
        # Throttle convergence check: every 5 epochs on GPU, every epoch on CPU
        check_interval = 5 if backend.name != "numpy" else 1
        last_W = None

        for epoch in range(1, int(self.max_iter) + 1):
            A_epoch = backend.zeros((self.n_components_, self.n_components_), dtype=backend.float64)
            B_epoch = backend.zeros((self.n_components_, self.n_features_in_), dtype=backend.float64)
            # Pre-compute HtH once per epoch (H is frozen within epoch)
            HtH = backend.matmul(self.components_, self.components_.T)
            for start in range(0, n_samples, batch_size):
                X_batch = X_arr[start : start + batch_size]
                W_batch = self._fit_batch_w(backend, X_batch, self.components_, HtH, n_steps=3)
                A_batch, B_batch = self._batch_stats(backend, X_batch, W_batch)
                A_epoch = A_epoch + A_batch
                B_epoch = B_epoch + B_batch
                self._n_batches_seen_ = int(self._n_batches_seen_) + 1
                last_W = W_batch

            old_components = self.components_
            new_components = self._update_h_from_stats_steps(
                backend, old_components, A_epoch, B_epoch, eps, n_steps=3
            )
            self.components_ = new_components
            self.n_iter_ = int(epoch)
            last_A = A_epoch
            last_B = B_epoch

            # Throttled convergence check
            if epoch % check_interval == 0 or epoch == int(self.max_iter):
                delta = scalar_to_float(backend.xp.linalg.norm(new_components - old_components) / (backend.xp.linalg.norm(old_components) + eps))
                if previous_delta is not None and delta <= float(self.tol):
                    break
                previous_delta = delta
        else:
            epoch = int(self.max_iter)
        self.n_iter_ = int(epoch)
        if last_A is None or last_B is None:
            W_full = self.transform(X_arr)
            self.reconstruction_err_ = self._reconstruction_error(backend, X_arr, W_full, self.components_)
            self._A_accum = backend.zeros((self.n_components_, self.n_components_), dtype=backend.float64)
            self._B_accum = backend.zeros((self.n_components_, self.n_features_in_), dtype=backend.float64)
        else:
            self.reconstruction_err_ = self._reconstruction_error_from_stats(
                backend, backend.sum(X_arr * X_arr), last_A, last_B, self.components_
            )
            self._A_accum = backend.copy(last_A)
            self._B_accum = backend.copy(last_B)
        return self

    def transform(self, X):
        self._check_is_fitted()
        reject_sparse(X, "MiniBatchNMF")
        backend = self._get_backend()
        X_arr = backend.asarray(X, dtype=backend.float64)
        check_2d_array(X_arr)
        self._check_nonnegative(backend, X_arr)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X_arr.shape[1]} features, expected {self.n_features_in_}")
        eps = np.finfo(np.float64).eps
        HtH = backend.matmul(self.components_, self.components_.T)
        W = self._init_w_from_data(backend, X_arr, self.components_, eps)
        n_steps = max(100, min(300, int(self.max_iter) * 5))
        for _ in range(n_steps):
            W = self._update_w(backend, X_arr, W, self.components_, HtH, eps)
        return W

    def fit_transform(self, X, y=None):
        # Run fit, then compute W directly (avoid redundant transform call)
        self.fit(X, y=y)
        backend = self._get_backend()
        X_arr = backend.asarray(X, dtype=backend.float64)
        eps = np.finfo(np.float64).eps
        HtH = backend.matmul(self.components_, self.components_.T)
        W = self._init_w_from_data(backend, X_arr, self.components_, eps)
        n_steps = max(100, min(300, int(self.max_iter) * 5))
        for _ in range(n_steps):
            W = self._update_w(backend, X_arr, W, self.components_, HtH, eps)
        return W

    def inverse_transform(self, X):
        self._check_is_fitted()
        backend = self._get_backend()
        X_arr = backend.asarray(X, dtype=backend.float64)
        check_2d_array(X_arr)
        if X_arr.shape[1] != self.n_components_:
            raise ValueError(f"X has {X_arr.shape[1]} components, expected {self.n_components_}")
        return backend.matmul(X_arr, self.components_)

    def predict(self, X):
        return self.transform(X)

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params.update(
            {
                "n_components": self.n_components,
                "init": self.init,
                "batch_size": self.batch_size,
                "max_iter": self.max_iter,
                "tol": self.tol,
                "random_state": self.random_state,
            }
        )
        return params
