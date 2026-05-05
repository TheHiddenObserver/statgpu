"""Exact dense t-SNE."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from statgpu._base import BaseEstimator
from statgpu._config import Device
from statgpu.unsupervised._pca import PCA
from statgpu.unsupervised._utils import (
    check_2d_array,
    eye,
    random_normal,
    reject_sparse,
    scalar_to_float,
    squared_euclidean_distances,
)


class TSNE(BaseEstimator):
    """Dense exact t-SNE with backend-native probability and gradient steps."""

    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        early_exaggeration: float = 12.0,
        learning_rate: Union[str, float] = "auto",
        max_iter: int = 1000,
        init: str = "pca",
        random_state: Optional[int] = None,
        metric: str = "euclidean",
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.init = init
        self.random_state = random_state
        self.metric = metric

    def _validate_params(self, n_samples: int):
        if self.metric != "euclidean":
            raise NotImplementedError("TSNE v1 only supports metric='euclidean'")
        if not isinstance(self.n_components, (int, np.integer)) or int(self.n_components) < 1:
            raise ValueError("n_components must be a positive integer")
        if int(self.n_components) >= n_samples:
            raise ValueError("n_components must be less than n_samples")
        if float(self.perplexity) <= 0.0 or float(self.perplexity) >= n_samples:
            raise ValueError("perplexity must be in (0, n_samples)")
        if float(self.early_exaggeration) <= 0.0:
            raise ValueError("early_exaggeration must be positive")
        if not isinstance(self.max_iter, (int, np.integer)) or int(self.max_iter) < 250:
            raise ValueError("max_iter must be an integer >= 250")
        if self.init not in ("pca", "random"):
            raise ValueError("init must be one of: 'pca', 'random'")

    def _joint_probabilities(self, backend, X):
        n_samples = X.shape[0]
        distances = squared_euclidean_distances(backend, X)
        distances = distances * (1.0 - eye(backend, n_samples, dtype=backend.float64))
        beta = backend.ones((n_samples, 1), dtype=backend.float64)
        beta_min = backend.zeros((n_samples, 1), dtype=backend.float64)
        beta_max = backend.full((n_samples, 1), np.inf, dtype=backend.float64)
        target = float(np.log(self.perplexity))
        off_diag = 1.0 - eye(backend, n_samples, dtype=backend.float64)

        for _ in range(50):
            P = backend.exp(-distances * beta) * off_diag
            sumP = backend.maximum(backend.sum(P, axis=1, keepdims=True), 1e-300)
            H = backend.log(sumP) + beta * backend.sum(distances * P, axis=1, keepdims=True) / sumP
            too_high = H > target
            beta_min = backend.where(too_high, beta, beta_min)
            beta_max = backend.where(too_high, beta_max, beta)
            doubled = beta * 2.0
            halved = beta / 2.0
            averaged_high = (beta + beta_max) / 2.0
            averaged_low = (beta + beta_min) / 2.0
            beta = backend.where(
                too_high,
                backend.where(beta_max == np.inf, doubled, averaged_high),
                backend.where(beta_min == 0.0, halved, averaged_low),
            )

        P = backend.exp(-distances * beta) * off_diag
        P = P / backend.maximum(backend.sum(P, axis=1, keepdims=True), 1e-300)
        P = (P + P.T) / (2.0 * float(n_samples))
        return backend.maximum(P, 1e-300)

    def _initial_embedding(self, backend, X):
        n_samples = X.shape[0]
        if self.init == "random":
            init = random_normal(self.random_state, size=(n_samples, int(self.n_components)), scale=1e-4)
            return backend.asarray(init, dtype=backend.float64)
        pca = PCA(
            n_components=int(self.n_components),
            svd_solver="auto",
            random_state=self.random_state,
            device=self.device,
            n_jobs=self.n_jobs,
        )
        init = pca.fit_transform(X)
        first = init[:, :1]
        first_centered = first - backend.mean(first, axis=0, keepdims=True)
        scale = backend.sqrt(backend.maximum(backend.mean(first_centered * first_centered), 1e-300))
        return init / scale * 1e-4

    def _learning_rate(self, n_samples: int) -> float:
        if self.learning_rate == "auto":
            return max(float(n_samples) / float(self.early_exaggeration) / 4.0, 10.0)
        lr = float(self.learning_rate)
        if lr <= 0.0:
            raise ValueError("learning_rate must be 'auto' or a positive number")
        return lr

    def fit(self, X, y=None):
        reject_sparse(X, "TSNE")
        backend = self._get_backend()
        X_arr = backend.asarray(X, dtype=backend.float64)
        check_2d_array(X_arr)
        n_samples, n_features = X_arr.shape
        self._validate_params(n_samples)

        P = self._joint_probabilities(backend, X_arr)
        Y = self._initial_embedding(backend, X_arr)
        lr = self._learning_rate(n_samples)
        momentum = 0.5
        velocity = backend.zeros_like(Y)
        gains = backend.ones_like(Y)
        off_diag = 1.0 - eye(backend, n_samples, dtype=backend.float64)
        exaggeration_iters = min(250, int(self.max_iter) // 2)

        kl = None
        for it in range(int(self.max_iter)):
            P_use = P * float(self.early_exaggeration) if it < exaggeration_iters else P
            dist_sq = squared_euclidean_distances(backend, Y)
            inv = (1.0 / (1.0 + dist_sq)) * off_diag
            Q = inv / backend.maximum(backend.sum(inv), 1e-300)
            forces = (P_use - Q) * inv
            row_force = backend.sum(forces, axis=1, keepdims=True)
            grad = 4.0 * (row_force * Y - backend.matmul(forces, Y))
            sign_changed = (grad * velocity) < 0.0
            gains = backend.where(sign_changed, gains + 0.2, gains * 0.8)
            gains = backend.maximum(gains, 0.01)
            velocity = momentum * velocity - lr * gains * grad
            Y = Y + velocity
            Y = Y - backend.mean(Y, axis=0, keepdims=True)
            if it == exaggeration_iters:
                momentum = 0.8

        dist_sq = squared_euclidean_distances(backend, Y)
        inv = (1.0 / (1.0 + dist_sq)) * off_diag
        Q = backend.maximum(inv / backend.maximum(backend.sum(inv), 1e-300), 1e-300)
        kl = backend.sum(P * (backend.log(P) - backend.log(Q)))

        self.embedding_ = Y
        self.kl_divergence_ = scalar_to_float(kl)
        self.n_iter_ = int(self.max_iter)
        self.n_features_in_ = int(n_features)
        self._backend_name = backend.name
        self._fitted = True
        return self

    def fit_transform(self, X, y=None):
        return self.fit(X, y=y).embedding_

    def transform(self, X):
        raise NotImplementedError("TSNE v1 does not support transforming new data")

    def predict(self, X):
        raise NotImplementedError("TSNE v1 does not support prediction")

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params.update(
            {
                "n_components": self.n_components,
                "perplexity": self.perplexity,
                "early_exaggeration": self.early_exaggeration,
                "learning_rate": self.learning_rate,
                "max_iter": self.max_iter,
                "init": self.init,
                "random_state": self.random_state,
                "metric": self.metric,
            }
        )
        return params
