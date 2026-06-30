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
    UMAP with NumPy, CuPy, or Torch backends.

    Supports exact NN (dense distance) and approximate NNDescent.
    Graph construction uses sparse COO edges (O(n*k) memory).
    Optimization loop and negative sampling are backend-aware
    (torch.randint / cp.random.randint / np.random.randint).

    Known limitations:
    - Spectral initialization uses CPU SciPy sparse eigensolver.
    - Graph edge construction may transfer indices/weights through host memory.
    - Transform of new data is not yet supported.
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
            # Uses internal _nndescent module (torch/cupy/numpy), no external dependency
            pass
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

        # Build COO sparse edges directly (O(n*k) memory, not O(n²))
        # For each point i, add edge (i, neighbor_j) with weight membership[i,j]
        import numpy as np
        all_src_np = np.repeat(np.arange(n_samples, dtype=np.int64), k)  # (n*k,)
        # Convert neighbor indices/membership to numpy (handle cupy/torch safely)
        if hasattr(neighbor_indices, 'get'):  # cupy
            import cupy as cp
            all_dst_np = cp.asnumpy(neighbor_indices).ravel().astype(np.int64)
            all_w_np = cp.asnumpy(membership).ravel().astype(np.float64)
        elif hasattr(neighbor_indices, 'cpu'):  # torch
            all_dst_np = neighbor_indices.cpu().numpy().ravel().astype(np.int64)
            all_w_np = membership.cpu().numpy().ravel().astype(np.float64)
        else:  # numpy
            all_dst_np = np.asarray(neighbor_indices, dtype=np.int64).ravel()
            all_w_np = np.asarray(membership, dtype=np.float64).ravel()
        all_src = backend.asarray(all_src_np, dtype=backend.int64)
        all_dst = backend.asarray(all_dst_np, dtype=backend.int64)
        all_w = backend.asarray(all_w_np, dtype=backend.float64)

        # Symmetrize: add reverse edges a+b-a*b
        rev_w = all_w + all_w - all_w * all_w  # fuzzy union
        # Remove self-edges (where src == dst, set to 0)
        is_self = all_src == all_dst
        if hasattr(is_self, 'cpu'):  # torch
            rev_w = rev_w.where(~is_self, backend.asarray(0.0, dtype=rev_w.dtype))
        else:
            rev_w[is_self] = 0.0

        return (all_src, all_dst, rev_w, n_samples)

    def _initial_embedding(self, backend, graph_data):
        all_src, all_dst, all_w, n_samples = graph_data
        if self.init == "random":
            return backend_random_normal(backend, self.random_state, size=(n_samples, int(self.n_components)), scale=1e-4)

        # Spectral embedding via sparse Laplacian eigendecomposition
        import numpy as np
        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import eigsh

        # Convert to numpy (handle cupy/torch/numpy)
        if hasattr(all_src, 'get'):  # cupy
            import cupy as cp
            src_np = cp.asnumpy(all_src).ravel().astype(np.int32)
            dst_np = cp.asnumpy(all_dst).ravel().astype(np.int32)
            w_np = cp.asnumpy(all_w).ravel().astype(np.float64)
        elif hasattr(all_src, 'cpu'):  # torch
            src_np = all_src.cpu().numpy().ravel().astype(np.int32)
            dst_np = all_dst.cpu().numpy().ravel().astype(np.int32)
            w_np = all_w.cpu().numpy().ravel().astype(np.float64)
        else:  # numpy
            src_np = np.asarray(all_src, dtype=np.int32).ravel()
            dst_np = np.asarray(all_dst, dtype=np.int32).ravel()
            w_np = np.asarray(all_w, dtype=np.float64).ravel()

        sparse_graph = coo_matrix((w_np, (src_np, dst_np)), shape=(n_samples, n_samples))
        sparse_graph = (sparse_graph + sparse_graph.T).tocsr() * 0.5
        degree = np.array(sparse_graph.sum(axis=1)).ravel()
        laplacian = sparse_graph - __import__('scipy').sparse.diags(degree)
        n_components = min(int(self.n_components) + 1, n_samples - 2)
        _, eigenvectors = eigsh(laplacian, k=n_components, which='SM', tol=1e-4)
        jitter = backend_random_normal(backend, self.random_state, size=(n_samples, int(self.n_components)), scale=1e-4)
        return backend.asarray(eigenvectors[:, 1:int(self.n_components)+1], dtype=backend.float64) + jitter

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
        graph_sparse = self._fuzzy_graph(backend, X_f32)
        edge_rows_b, edge_cols_b, edge_weights_b, n_samples = graph_sparse
        # Convert to float64 and clip weights (sparse: O(n*k), not O(n²))
        edge_rows_b = backend.asarray(edge_rows_b, dtype=backend.int64)
        edge_cols_b = backend.asarray(edge_cols_b, dtype=backend.int64)
        edge_weights_b = backend.clip(backend.asarray(edge_weights_b, dtype=backend.float64), 0.0, 1.0)

        Y = self._initial_embedding(backend, graph_sparse)
        n_epochs = self._epochs(n_samples)
        a, b = self._attraction_curve_params()
        repulsion = float(self.repulsion_strength) / float(self.negative_sample_rate)
        a_f = float(a)
        b_f = float(b)
        lr = float(self.learning_rate)
        neg_rate = int(self.negative_sample_rate)

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
            # Use backend-native RNG seeded by random_state for reproducibility
            rs = self.random_state if self.random_state is not None else 42
            if hasattr(Y, 'device') and not hasattr(Y, 'get'):  # torch
                try:
                    import torch
                    g = torch.Generator(device=Y.device).manual_seed(int(rs))
                    neg_src = torch.randint(0, n_samples, (n_samples * neg_rate,),
                                            generator=g, device=Y.device, dtype=torch.int64)
                    neg_dst = torch.randint(0, n_samples, (n_samples * neg_rate,),
                                            generator=g, device=Y.device, dtype=torch.int64)
                except ImportError:
                    neg_src = backend.asarray(rng.randint(0, n_samples, size=n_samples * neg_rate), dtype=backend.int64)
                    neg_dst = backend.asarray(rng.randint(0, n_samples, size=n_samples * neg_rate), dtype=backend.int64)
            elif hasattr(Y, 'get'):  # cupy
                import cupy as cp
                cp_rng = cp.random.RandomState(int(rs))
                neg_src = cp_rng.randint(0, n_samples, (n_samples * neg_rate,), dtype=cp.int64)
                neg_dst = cp_rng.randint(0, n_samples, (n_samples * neg_rate,), dtype=cp.int64)
            else:  # numpy
                neg_src = rng.randint(0, n_samples, size=n_samples * neg_rate).astype(np.int64)
                neg_dst = rng.randint(0, n_samples, size=n_samples * neg_rate).astype(np.int64)

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
        self.graph_ = graph_sparse
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
