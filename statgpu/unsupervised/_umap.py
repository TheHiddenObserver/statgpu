"""Dense exact UMAP."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from statgpu._base import BaseEstimator
from statgpu._config import Device
from statgpu.unsupervised._utils import (
    check_2d_array,
    eye,
    reject_sparse,
    squared_euclidean_distances,
    topk_smallest,
)


class UMAP(BaseEstimator):
    """
    Dense exact UMAP with NumPy, CuPy, or Torch backends.

    Version 1 builds an exact dense Euclidean neighbor graph and does not
    implement approximate NNDescent or transforming new data.
    """

    def __init__(
        self,
        n_neighbors: int = 15,
        n_components: int = 2,
        metric: str = "euclidean",
        min_dist: float = 0.1,
        spread: float = 1.0,
        n_epochs: Optional[int] = None,
        learning_rate: float = 1.0,
        init: str = "spectral",
        negative_sample_rate: int = 5,
        repulsion_strength: float = 1.0,
        random_state: Optional[int] = None,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.metric = metric
        self.min_dist = min_dist
        self.spread = spread
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.init = init
        self.negative_sample_rate = negative_sample_rate
        self.repulsion_strength = repulsion_strength
        self.random_state = random_state

    def _validate_params(self, n_samples: int):
        if self.metric != "euclidean":
            raise NotImplementedError("UMAP v1 only supports metric='euclidean'")
        if not isinstance(self.n_neighbors, (int, np.integer)) or int(self.n_neighbors) < 2:
            raise ValueError("n_neighbors must be an integer >= 2")
        if int(self.n_neighbors) >= n_samples:
            raise ValueError("n_neighbors must be less than n_samples")
        if not isinstance(self.n_components, (int, np.integer)) or int(self.n_components) < 1:
            raise ValueError("n_components must be a positive integer")
        if int(self.n_components) >= n_samples:
            raise ValueError("n_components must be less than n_samples")
        if float(self.min_dist) < 0.0:
            raise ValueError("min_dist must be non-negative")
        if float(self.spread) <= 0.0:
            raise ValueError("spread must be positive")
        if self.init not in ("spectral", "random"):
            raise ValueError("init must be one of: 'spectral', 'random'")
        if self.n_epochs is not None:
            if not isinstance(self.n_epochs, (int, np.integer)) or int(self.n_epochs) < 1:
                raise ValueError("n_epochs must be None or a positive integer")
        if float(self.learning_rate) <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not isinstance(self.negative_sample_rate, (int, np.integer)) or int(self.negative_sample_rate) < 1:
            raise ValueError("negative_sample_rate must be a positive integer")
        if float(self.repulsion_strength) <= 0.0:
            raise ValueError("repulsion_strength must be positive")

    def _smooth_knn_membership(self, backend, neighbor_distances):
        rho = neighbor_distances[:, :1]
        adjusted = backend.maximum(neighbor_distances - rho, 0.0)
        sigma = backend.maximum(backend.mean(adjusted, axis=1, keepdims=True), 1e-12)
        membership = backend.exp(-adjusted / sigma)
        membership[:, 0] = 1.0
        return membership

    def _fuzzy_graph(self, backend, X):
        n_samples = X.shape[0]
        distances = backend.sqrt(squared_euclidean_distances(backend, X))
        distances = distances + eye(backend, n_samples, dtype=backend.float64) * 1e12
        neighbor_distances, neighbor_indices = topk_smallest(backend, distances, int(self.n_neighbors))
        membership = self._smooth_knn_membership(backend, neighbor_distances)
        graph = backend.zeros((n_samples, n_samples), dtype=backend.float64)
        for i in range(n_samples):
            graph[i, neighbor_indices[i]] = membership[i]
        graph = graph + graph.T - graph * graph.T
        graph = graph * (1.0 - eye(backend, n_samples, dtype=backend.float64))
        return graph

    def _initial_embedding(self, backend, graph):
        n_samples = graph.shape[0]
        rng = np.random.default_rng(self.random_state)
        if self.init == "random":
            init = 1e-4 * rng.normal(size=(n_samples, int(self.n_components)))
            return backend.asarray(init, dtype=backend.float64)

        degree = backend.sum(graph, axis=1)
        laplacian = backend.diag(degree) - graph
        eigenvalues, eigenvectors = backend.eigh(laplacian)
        order = backend.argsort(eigenvalues, axis=0)
        components = eigenvectors[:, order[1 : int(self.n_components) + 1]]
        jitter = 1e-4 * rng.normal(size=(n_samples, int(self.n_components)))
        return components + backend.asarray(jitter, dtype=backend.float64)

    def _epochs(self, n_samples: int) -> int:
        if self.n_epochs is not None:
            return int(self.n_epochs)
        return 500 if n_samples <= 10_000 else 200

    def fit(self, X, y=None):
        reject_sparse(X, "UMAP")
        backend = self._get_backend()
        X_arr = backend.asarray(X, dtype=backend.float64)
        check_2d_array(X_arr)
        n_samples, n_features = X_arr.shape
        self._validate_params(n_samples)

        graph = self._fuzzy_graph(backend, X_arr)
        Y = self._initial_embedding(backend, graph)
        n_epochs = self._epochs(n_samples)
        off_diag = 1.0 - eye(backend, n_samples, dtype=backend.float64)
        graph = backend.clip(graph, 0.0, 1.0)
        repulsion = float(self.repulsion_strength) / float(self.negative_sample_rate)

        for epoch in range(n_epochs):
            alpha = float(self.learning_rate) * (1.0 - (epoch / max(n_epochs, 1)))
            diff = backend.expand_dims(Y, 1) - backend.expand_dims(Y, 0)
            dist_sq = backend.sum(diff * diff, axis=2)
            inv = (1.0 / (1.0 + dist_sq)) * off_diag
            attractive = graph
            repulsive = (1.0 - graph) * inv * repulsion
            forces = (attractive - repulsive) * inv
            grad = 2.0 * backend.sum(backend.expand_dims(forces, 2) * diff, axis=1)
            Y = Y - alpha * grad
            Y = Y - backend.mean(Y, axis=0, keepdims=True)

        self.embedding_ = Y
        self.graph_ = graph
        self.n_epochs_ = int(n_epochs)
        self.n_features_in_ = int(n_features)
        self._backend_name = backend.name
        self._fitted = True
        return self

    def fit_transform(self, X, y=None):
        return self.fit(X, y=y).embedding_

    def transform(self, X):
        raise NotImplementedError("UMAP v1 does not support transforming new data")

    def predict(self, X):
        raise NotImplementedError("UMAP v1 does not support prediction")

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params.update(
            {
                "n_neighbors": self.n_neighbors,
                "n_components": self.n_components,
                "metric": self.metric,
                "min_dist": self.min_dist,
                "spread": self.spread,
                "n_epochs": self.n_epochs,
                "learning_rate": self.learning_rate,
                "init": self.init,
                "negative_sample_rate": self.negative_sample_rate,
                "repulsion_strength": self.repulsion_strength,
                "random_state": self.random_state,
            }
        )
        return params
