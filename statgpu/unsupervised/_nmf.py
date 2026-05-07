"""Non-negative matrix factorization."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from statgpu._base import BaseEstimator
from statgpu._config import Device
from statgpu.unsupervised._utils import backend_random_normal, check_2d_array, reject_sparse, scalar_to_float


class NMF(BaseEstimator):
    """NMF with multiplicative updates and Frobenius loss."""

    def __init__(
        self,
        n_components: Optional[int] = None,
        init: str = "random",
        solver: str = "mu",
        beta_loss: str = "frobenius",
        max_iter: int = 200,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.n_components = n_components
        self.init = init
        self.solver = solver
        self.beta_loss = beta_loss
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
            raise NotImplementedError("NMF v1 only supports init='random'")
        if self.solver != "mu":
            raise NotImplementedError("NMF v1 only supports solver='mu'")
        if self.beta_loss != "frobenius":
            raise NotImplementedError("NMF v1 only supports beta_loss='frobenius'")
        if not isinstance(self.max_iter, (int, np.integer)) or int(self.max_iter) < 1:
            raise ValueError("max_iter must be a positive integer")
        if float(self.tol) < 0.0:
            raise ValueError("tol must be non-negative")
        return n_components

    def _check_nonnegative(self, backend, X):
        if scalar_to_float(backend.min(X)) < 0.0:
            raise ValueError("NMF input X must be non-negative")

    def _init_factors(self, backend, X, n_components, seed):
        eps = np.finfo(np.float64).eps
        rng = np.random.default_rng(seed)
        if X.shape[0] >= n_components:
            indices = rng.choice(int(X.shape[0]), size=int(n_components), replace=False)
            indices = backend.asarray(indices, dtype=backend.int64)
            H = backend.maximum(X[indices], eps) + 1e-8
        else:
            mean = max(scalar_to_float(backend.mean(X)), np.finfo(np.float64).eps)
            scale = np.sqrt(mean / float(n_components))
            H = backend.abs(backend_random_normal(backend, seed, size=(n_components, X.shape[1]), scale=scale)) + 1e-8
        W = self._init_w_from_data(backend, X, H, eps)
        return W, H

    def _init_w_from_data(self, backend, X, H, eps):
        numerator = backend.matmul(X, H.T)
        denominator = backend.reshape(backend.sum(H * H, axis=1) + eps, (1, H.shape[0]))
        return backend.maximum(numerator / denominator, eps)

    def _reconstruction_error(self, backend, X, W, H):
        residual = X - backend.matmul(W, H)
        return scalar_to_float(backend.sqrt(backend.sum(residual * residual)))

    def _update_h(self, backend, X, W, H, eps):
        numerator = backend.matmul(W.T, X)
        denominator = backend.matmul(backend.matmul(W.T, W), H) + eps
        H *= numerator
        H /= denominator
        return H

    def _update_w(self, backend, X, W, H, eps):
        numerator = backend.matmul(X, H.T)
        denominator = backend.matmul(W, backend.matmul(H, H.T)) + eps
        W *= numerator
        W /= denominator
        return W

    def fit(self, X, y=None):
        reject_sparse(X, "NMF")
        backend = self._get_backend()
        X_arr = backend.asarray(X, dtype=backend.float64)
        check_2d_array(X_arr)
        self._check_nonnegative(backend, X_arr)
        n_samples, n_features = X_arr.shape
        n_components = self._validate_params(n_samples, n_features)

        W, H = self._init_factors(backend, X_arr, n_components, self.random_state)
        eps = np.finfo(np.float64).eps
        previous_error = None
        error = None
        n_iter = 0
        if backend.name == "numpy":
            error_check_interval = 10
        else:
            error_check_interval = max(1, min(25, int(self.max_iter) // 5))
        for n_iter in range(1, int(self.max_iter) + 1):
            W = self._update_w(backend, X_arr, W, H, eps)
            H = self._update_h(backend, X_arr, W, H, eps)
            if n_iter % error_check_interval == 0 or n_iter == int(self.max_iter):
                error = self._reconstruction_error(backend, X_arr, W, H)
                if previous_error is not None:
                    if abs(previous_error - error) / max(previous_error, eps) <= float(self.tol):
                        break
                previous_error = error

        if error is None:
            error = self._reconstruction_error(backend, X_arr, W, H)

        self.components_ = H
        self._fit_W = W
        self.reconstruction_err_ = float(error if error is not None else 0.0)
        self.n_iter_ = int(n_iter)
        self.n_components_ = int(n_components)
        self.n_features_in_ = int(n_features)
        self._backend_name = backend.name
        self._fitted = True
        return self

    def transform(self, X):
        self._check_is_fitted()
        reject_sparse(X, "NMF")
        backend = self._get_backend()
        X_arr = backend.asarray(X, dtype=backend.float64)
        check_2d_array(X_arr)
        self._check_nonnegative(backend, X_arr)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X_arr.shape[1]} features, expected {self.n_features_in_}")
        eps = np.finfo(np.float64).eps
        W = self._init_w_from_data(backend, X_arr, self.components_, eps)
        for _ in range(int(self.max_iter)):
            W = self._update_w(backend, X_arr, W, self.components_, eps)
        return W

    def fit_transform(self, X, y=None):
        return self.fit(X, y=y)._fit_W

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
                "solver": self.solver,
                "beta_loss": self.beta_loss,
                "max_iter": self.max_iter,
                "tol": self.tol,
                "random_state": self.random_state,
            }
        )
        return params
