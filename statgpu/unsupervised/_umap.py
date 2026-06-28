"""Dense exact UMAP."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from statgpu._base import BaseEstimator
from statgpu._config import Device
from statgpu.backends._utils import _to_numpy, scatter_add_2d
from statgpu.unsupervised._utils import (
    backend_random_normal,
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
        nn_method: str = "auto",
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
        self.nn_method = nn_method

    def _validate_params(self, n_samples: int):
        if self.metric != "euclidean":
            raise NotImplementedError("UMAP v1 only supports metric='euclidean'")
        if self.nn_method not in ("auto", "exact", "nndescent"):
            raise ValueError("nn_method must be one of: 'auto', 'exact', 'nndescent'")
        if self.nn_method == "nndescent":
            try:
                from pynndescent import NNDescent  # noqa: F401
            except ImportError:
                raise ImportError(
                    "nn_method='nndescent' requires pynndescent: pip install pynndescent"
                )
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

    def _resolve_nn_method(self, n_samples):
        """Resolve 'auto' to 'exact' (NNDescent slower for UMAP due to graph quality)."""
        if self.nn_method != "auto":
            return self.nn_method
        # NNDescent is slower for UMAP because approximate graphs hurt optimization
        # Exact neighbors give better graph quality → faster optimization loop
        return "exact"

    def _fuzzy_graph(self, backend, X):
        n_samples = X.shape[0]
        k = int(self.n_neighbors)
        method = self._resolve_nn_method(n_samples)

        if method == "nndescent":
            # Approximate NN via NNDescent (O(n log n) vs O(n²))
            from statgpu.unsupervised._nndescent import nndescent_torch, nndescent_cupy, nndescent_numpy
            seed = self.random_state if self.random_state is not None else 42
            if hasattr(X, 'device') and not hasattr(X, 'get'):  # torch
                neighbor_indices, neighbor_distances_sq = nndescent_torch(
                    X, k=k, max_iter=10, seed=seed
                )
                neighbor_distances = backend.sqrt(backend.maximum(neighbor_distances_sq, 0.0))
            elif hasattr(X, 'get'):  # cupy
                neighbor_indices, neighbor_distances_sq = nndescent_cupy(
                    X, k=k, max_iter=10, seed=seed
                )
                neighbor_distances = backend.sqrt(backend.maximum(neighbor_distances_sq, 0.0))
            else:  # numpy
                neighbor_indices, neighbor_distances_sq = nndescent_numpy(
                    X, k=k, max_iter=10, seed=seed
                )
                neighbor_distances = backend.sqrt(backend.maximum(neighbor_distances_sq, 0.0))
        else:
            # Exact NN via dense distance matrix
            distances = backend.sqrt(squared_euclidean_distances(backend, X))
            distances = distances + eye(backend, n_samples, dtype=backend.float64) * 1e12
            neighbor_distances, neighbor_indices = topk_smallest(backend, distances, k)

        membership = self._smooth_knn_membership(backend, neighbor_distances)
        graph = backend.zeros((n_samples, n_samples), dtype=backend.float64)
        rows = backend.reshape(backend.arange(n_samples, dtype=backend.int64), (n_samples, 1))
        graph[rows, backend.astype(neighbor_indices, backend.int64)] = membership
        graph = graph + graph.T - graph * graph.T
        graph = graph * (1.0 - eye(backend, n_samples, dtype=backend.float64))
        return graph

    def _initial_embedding(self, backend, graph):
        n_samples = graph.shape[0]
        if self.init == "random":
            return backend_random_normal(backend, self.random_state, size=(n_samples, int(self.n_components)), scale=1e-4)

        degree = backend.sum(graph, axis=1)
        laplacian = backend.diag(degree) - graph
        eigenvalues, eigenvectors = backend.eigh(laplacian)
        order = backend.argsort(eigenvalues, axis=0)
        components = eigenvectors[:, order[1 : int(self.n_components) + 1]]
        jitter = backend_random_normal(backend, self.random_state, size=(n_samples, int(self.n_components)), scale=1e-4)
        return components + jitter

    def _epochs(self, n_samples: int) -> int:
        if self.n_epochs is not None:
            return int(self.n_epochs)
        # Fewer epochs for larger data (quality is similar, much faster)
        if n_samples <= 2_000:
            return 500
        elif n_samples <= 10_000:
            return 200
        else:
            return 100

    def _attraction_curve_params(self):
        """
        Fit UMAP's (a, b) curve parameters from min_dist and spread.

        This mirrors the reference approach used by umap-learn:
        target(d) = 1                      if d <= min_dist
                    exp(-(d-min_dist)/spread) otherwise
        and we fit 1 / (1 + a * d^(2b)) to that target.
        """
        min_dist = float(self.min_dist)
        spread = float(self.spread)
        xv = np.linspace(0.0, spread * 3.0, 300, dtype=np.float64)
        yv = np.where(xv <= min_dist, 1.0, np.exp(-(xv - min_dist) / max(spread, 1e-12)))

        def curve(d, a, b):
            return 1.0 / (1.0 + a * np.power(d, 2.0 * b))

        try:
            # Optional dependency: keep UMAP functional even when SciPy is absent.
            from scipy.optimize import curve_fit

            params, _ = curve_fit(
                curve,
                xv,
                yv,
                p0=(1.0, 1.0),
                bounds=((1e-12, 1e-12), (1e6, 10.0)),
                maxfev=20000,
            )
            a, b = float(params[0]), float(params[1])
            if np.isfinite(a) and np.isfinite(b) and a > 0.0 and b > 0.0:
                return a, b
        except Exception:
            pass

        # Conservative fallback to ensure training can proceed.
        return 1.0, 1.0 / max(spread, 1e-12)

    def fit(self, X, y=None):
        reject_sparse(X, "UMAP")
        backend = self._get_backend()
        X_arr = backend.asarray(X, dtype=backend.float64)
        check_2d_array(X_arr)
        n_samples, n_features = X_arr.shape
        self._validate_params(n_samples)

        # Use float32 for distance computations (2x faster, 2x less memory)
        X_f32 = backend.asarray(X_arr, dtype=backend.float32)
        graph = self._fuzzy_graph(backend, X_f32)
        graph = backend.asarray(graph, dtype=backend.float64)

        Y = self._initial_embedding(backend, graph)
        n_epochs = self._epochs(n_samples)
        a, b = self._attraction_curve_params()
        off_diag = 1.0 - eye(backend, n_samples, dtype=backend.float64)
        graph = backend.clip(graph, 0.0, 1.0)
        repulsion = float(self.repulsion_strength) / float(self.negative_sample_rate)
        a_f = float(a)
        b_f = float(b)
        lr = float(self.learning_rate)
        neg_rate = int(self.negative_sample_rate)

        # Extract sparse edges on GPU (no CPU transfer)
        graph_mask = graph > 0
        if hasattr(graph_mask, 'cpu'):  # torch
            import torch
            nz = torch.nonzero(graph_mask, as_tuple=False)
            edge_rows_b = nz[:, 0]
            edge_cols_b = nz[:, 1]
            edge_weights_b = graph[edge_rows_b, edge_cols_b]
        elif hasattr(graph_mask, 'get'):  # cupy
            import cupy as cp
            nz = cp.argwhere(graph_mask)
            edge_rows_b = nz[:, 0]
            edge_cols_b = nz[:, 1]
            edge_weights_b = graph[edge_rows_b, edge_cols_b]
        else:  # numpy
            edge_rows, edge_cols = np.nonzero(_to_numpy(graph_mask))
            edge_rows_b = backend.asarray(edge_rows, dtype=backend.int64)
            edge_cols_b = backend.asarray(edge_cols, dtype=backend.int64)
            edge_weights_b = graph[edge_rows_b, edge_cols_b]

        n_edges = len(edge_rows_b)

        rng = np.random.RandomState(self.random_state)

        for epoch in range(n_epochs):
            alpha = lr * (1.0 - (epoch / max(n_epochs, 1)))

            # === Attractive forces (sparse: only graph edges) ===
            Y_src = Y[edge_rows_b]
            Y_dst = Y[edge_cols_b]
            diff_attr = Y_src - Y_dst
            dist_sq_attr = backend.sum(diff_attr * diff_attr, axis=1)
            w_attr = 1.0 / (1.0 + a_f * backend.maximum(dist_sq_attr, 1e-10) ** b_f)
            force_attr = edge_weights_b * w_attr

            # Gradient: scatter-add forces to source nodes
            force_attr_2d = backend.expand_dims(force_attr, 1) * diff_attr
            grad_attr = scatter_add_2d(Y, edge_rows_b, force_attr_2d)

            # === Repulsive forces (negative sampling) ===
            neg_indices = rng.randint(0, n_samples, size=n_samples * neg_rate)
            neg_dst_indices = rng.randint(0, n_samples, size=n_samples * neg_rate)
            if hasattr(Y, 'device') and not hasattr(graph_mask, 'get'):  # torch
                import torch
                neg_src = torch.tensor(neg_indices, device=Y.device, dtype=torch.int64)
                neg_dst = torch.tensor(neg_dst_indices, device=Y.device, dtype=torch.int64)
            elif hasattr(graph_mask, 'get'):  # cupy
                import cupy as cp
                neg_src = cp.asarray(neg_indices, dtype=cp.int64)
                neg_dst = cp.asarray(neg_dst_indices, dtype=cp.int64)
            else:  # numpy
                neg_src = backend.asarray(neg_indices, dtype=backend.int64)
                neg_dst = backend.asarray(neg_dst_indices, dtype=backend.int64)

            Y_neg_src = Y[neg_src]
            Y_neg_dst = Y[neg_dst]
            diff_rep = Y_neg_src - Y_neg_dst
            dist_sq_rep = backend.sum(diff_rep * diff_rep, axis=1)
            w_rep = 1.0 / (1.0 + a_f * backend.maximum(dist_sq_rep, 1e-10) ** b_f)
            force_rep = repulsion * w_rep * w_rep

            force_rep_2d = backend.expand_dims(force_rep, 1) * diff_rep
            grad_rep = scatter_add_2d(Y, neg_src, force_rep_2d)

            # Update
            grad = 2.0 * (grad_attr - grad_rep)
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
                "nn_method": self.nn_method,
            }
        )
        return params
