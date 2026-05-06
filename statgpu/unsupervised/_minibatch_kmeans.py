"""Mini-batch K-Means clustering."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from scipy import sparse

from statgpu._base import BaseEstimator
from statgpu._config import Device
from statgpu.unsupervised._kmeans import KMeans
from statgpu.unsupervised._utils import check_2d_array, scalar_to_float


class MiniBatchKMeans(BaseEstimator):
    """Mini-batch Lloyd K-Means with NumPy, CuPy, or Torch backends."""

    def __init__(
        self,
        n_clusters: int = 8,
        init="k-means++",
        n_init: Union[str, int] = "auto",
        batch_size: int = 1024,
        max_iter: int = 100,
        max_no_improvement: int = 10,
        tol: float = 0.0,
        random_state: Optional[int] = None,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.max_no_improvement = max_no_improvement
        self.tol = tol
        self.random_state = random_state

    def _validate_params(self, n_samples: int, n_features: int):
        if not isinstance(self.n_clusters, (int, np.integer)) or int(self.n_clusters) < 1:
            raise ValueError("n_clusters must be a positive integer")
        n_clusters = int(self.n_clusters)
        if n_clusters > n_samples:
            raise ValueError("n_clusters must be less than or equal to n_samples")
        if isinstance(self.init, str):
            if self.init not in ("k-means++", "random"):
                raise ValueError("init must be one of: 'k-means++', 'random', or an array of centers")
        elif callable(self.init):
            raise NotImplementedError("callable init is not supported in MiniBatchKMeans v1")
        else:
            init_shape = getattr(self.init, "shape", None)
            if init_shape != (n_clusters, n_features):
                raise ValueError(
                    f"init array must have shape ({n_clusters}, {n_features}), got {init_shape}"
                )
        if self.n_init == "auto":
            n_init = 1 if not isinstance(self.init, str) or self.init == "k-means++" else 3
        else:
            if not isinstance(self.n_init, (int, np.integer)) or int(self.n_init) < 1:
                raise ValueError("n_init must be 'auto' or a positive integer")
            n_init = int(self.n_init)
        if not isinstance(self.batch_size, (int, np.integer)) or int(self.batch_size) < 1:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(self.max_iter, (int, np.integer)) or int(self.max_iter) < 1:
            raise ValueError("max_iter must be a positive integer")
        if self.max_no_improvement is not None:
            if not isinstance(self.max_no_improvement, (int, np.integer)) or int(self.max_no_improvement) < 0:
                raise ValueError("max_no_improvement must be None or a non-negative integer")
        if float(self.tol) < 0.0:
            raise ValueError("tol must be non-negative")
        return n_clusters, n_init

    def _helper(self):
        init = self.init if isinstance(self.init, str) else "random"
        return KMeans(
            n_clusters=int(self.n_clusters),
            init=init,
            n_init=1,
            max_iter=1,
            tol=self.tol,
            random_state=self.random_state,
            device=self.device,
            n_jobs=self.n_jobs,
        )

    def _init_centers(self, backend, X, rng, n_clusters, helper):
        if not isinstance(self.init, str):
            return backend.asarray(self.init, dtype=backend.float64)
        x_norm = backend.sum(X * X, axis=1, keepdims=True)
        return helper._init_centers(backend, X, rng, n_clusters, x_norm)

    def _single_batch_step(self, backend, X_batch, centers, counts, helper, return_inertia: bool = True):
        distances = helper._squared_distances(backend, X_batch, centers)
        labels, min_dist_sq = helper._labels_min_distances(backend, distances)
        batch_counts = helper._bincount(backend, labels, centers.shape[0])
        batch_sums = helper._label_sums(backend, X_batch, labels, centers.shape[0])
        non_empty = batch_counts > 0
        new_counts = counts + batch_counts
        safe_batch_counts = backend.maximum(backend.reshape(batch_counts, (centers.shape[0], 1)), 1)
        batch_means = batch_sums / safe_batch_counts
        eta = backend.reshape(batch_counts / backend.maximum(new_counts, 1), (centers.shape[0], 1))
        updated = centers + eta * (batch_means - centers)
        centers = backend.where(backend.reshape(non_empty, (centers.shape[0], 1)), updated, centers)
        counts = backend.where(non_empty, new_counts, counts)
        inertia = scalar_to_float(backend.sum(min_dist_sq)) if return_inertia else None
        return centers, counts, labels, inertia

    def _final_labels_inertia(self, backend, X, centers, helper):
        distances = helper._squared_distances(backend, X, centers)
        labels, min_dist_sq = helper._labels_min_distances(backend, distances)
        return labels, scalar_to_float(backend.sum(min_dist_sq))

    def _lloyd_polish(self, backend, X, centers, helper, n_steps: int = 2):
        """Run a small exact Lloyd polish after mini-batch updates."""
        labels = None
        inertia = None
        counts = None
        for _ in range(int(n_steps)):
            distances = helper._squared_distances(backend, X, centers)
            labels, min_dist_sq = helper._labels_min_distances(backend, distances)
            new_centers = helper._compute_centers(backend, X, labels, min_dist_sq, centers)
            centers = new_centers
            inertia = scalar_to_float(backend.sum(min_dist_sq))
            counts = helper._bincount(backend, labels, centers.shape[0])
        labels, inertia = self._final_labels_inertia(backend, X, centers, helper)
        counts = helper._bincount(backend, labels, centers.shape[0])
        return centers, counts, labels, inertia

    def _run_single(self, backend, X, rng, n_clusters, helper):
        centers = self._init_centers(backend, X, rng, n_clusters, helper)
        counts = backend.zeros((n_clusters,), dtype=backend.float64)
        n_samples = X.shape[0]
        batch_size = min(int(self.batch_size), n_samples)
        best_batch_inertia = None
        no_improvement = 0
        n_steps = 0
        n_iter = 0
        last_centers = backend.copy(centers)
        track_improvement = self.max_no_improvement is not None

        for n_iter in range(1, int(self.max_iter) + 1):
            order = rng.permutation(n_samples)
            order_backend = backend.asarray(order, dtype=backend.int64)
            for start in range(0, n_samples, batch_size):
                batch_idx = order_backend[start : start + batch_size]
                X_batch = X[batch_idx]
                centers, counts, _, batch_inertia = self._single_batch_step(
                    backend, X_batch, centers, counts, helper, return_inertia=track_improvement
                )
                n_steps += 1
                if track_improvement:
                    if best_batch_inertia is None or batch_inertia < best_batch_inertia:
                        best_batch_inertia = batch_inertia
                        no_improvement = 0
                    else:
                        no_improvement += 1
                    if no_improvement >= int(self.max_no_improvement):
                        break

            center_shift = scalar_to_float(backend.sum((centers - last_centers) ** 2))
            last_centers = backend.copy(centers)
            if center_shift <= float(self.tol):
                break
            if track_improvement and no_improvement >= int(self.max_no_improvement):
                break

        centers, counts, labels, inertia = self._lloyd_polish(backend, X, centers, helper)
        return centers, counts, labels, inertia, n_iter, n_steps

    def fit(self, X, y=None, sample_weight=None):
        if sparse.issparse(X):
            raise NotImplementedError("sparse input is not supported in MiniBatchKMeans v1")
        if sample_weight is not None:
            raise NotImplementedError("sample_weight is not supported in MiniBatchKMeans v1")
        backend = self._get_backend()
        X_arr = backend.asarray(X, dtype=backend.float64)
        check_2d_array(X_arr)
        n_samples, n_features = X_arr.shape
        n_clusters, n_init = self._validate_params(n_samples, n_features)
        helper = self._helper()

        rng = np.random.default_rng(self.random_state)
        best = None
        for _ in range(n_init):
            run_rng = np.random.default_rng(rng.integers(0, np.iinfo(np.int32).max))
            result = self._run_single(backend, X_arr, run_rng, n_clusters, helper)
            if best is None or result[3] < best[3]:
                best = result

        (
            self.cluster_centers_,
            self.counts_,
            self.labels_,
            self.inertia_,
            self.n_iter_,
            self.n_steps_,
        ) = best
        self.n_features_in_ = int(n_features)
        self._backend_name = backend.name
        self._fitted = True
        return self

    def partial_fit(self, X, y=None, sample_weight=None):
        if sparse.issparse(X):
            raise NotImplementedError("sparse input is not supported in MiniBatchKMeans v1")
        if sample_weight is not None:
            raise NotImplementedError("sample_weight is not supported in MiniBatchKMeans v1")
        backend = self._get_backend()
        X_arr = backend.asarray(X, dtype=backend.float64)
        check_2d_array(X_arr)
        n_samples, n_features = X_arr.shape
        validation_samples = n_samples if not self._fitted else max(n_samples, int(self.n_clusters))
        n_clusters, _ = self._validate_params(validation_samples, n_features)
        helper = self._helper()

        if not self._fitted:
            rng = np.random.default_rng(self.random_state)
            self.cluster_centers_ = self._init_centers(backend, X_arr, rng, n_clusters, helper)
            self.counts_ = backend.zeros((n_clusters,), dtype=backend.float64)
            self.n_steps_ = 0
            self.n_iter_ = 0
            self.n_features_in_ = int(n_features)
            self._backend_name = backend.name
            self._fitted = True
        elif X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X_arr.shape[1]} features, expected {self.n_features_in_}")

        centers, counts, labels, inertia = self._single_batch_step(
            backend, X_arr, self.cluster_centers_, self.counts_, helper
        )
        self.cluster_centers_ = centers
        self.counts_ = counts
        self.labels_ = labels
        self.inertia_ = inertia
        self.n_steps_ = int(self.n_steps_) + 1
        self.n_iter_ = int(self.n_iter_) + 1
        return self

    def transform(self, X):
        self._check_is_fitted()
        if sparse.issparse(X):
            raise NotImplementedError("sparse input is not supported in MiniBatchKMeans v1")
        backend = self._get_backend()
        X_arr = backend.asarray(X, dtype=backend.float64)
        check_2d_array(X_arr)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X_arr.shape[1]} features, expected {self.n_features_in_}")
        helper = self._helper()
        return backend.sqrt(helper._squared_distances(backend, X_arr, self.cluster_centers_))

    def predict(self, X):
        self._check_is_fitted()
        if sparse.issparse(X):
            raise NotImplementedError("sparse input is not supported in MiniBatchKMeans v1")
        backend = self._get_backend()
        X_arr = backend.asarray(X, dtype=backend.float64)
        check_2d_array(X_arr)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X_arr.shape[1]} features, expected {self.n_features_in_}")
        helper = self._helper()
        distances = helper._squared_distances(backend, X_arr, self.cluster_centers_)
        return backend.argmin(distances, axis=1)

    def fit_predict(self, X, y=None):
        return self.fit(X, y=y).labels_

    def score(self, X, y=None):
        self._check_is_fitted()
        if sparse.issparse(X):
            raise NotImplementedError("sparse input is not supported in MiniBatchKMeans v1")
        backend = self._get_backend()
        X_arr = backend.asarray(X, dtype=backend.float64)
        check_2d_array(X_arr)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X_arr.shape[1]} features, expected {self.n_features_in_}")
        helper = self._helper()
        distances = helper._squared_distances(backend, X_arr, self.cluster_centers_)
        min_dist_sq = backend.min(distances, axis=1)
        return -scalar_to_float(backend.sum(min_dist_sq))

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params.update(
            {
                "n_clusters": self.n_clusters,
                "init": self.init,
                "n_init": self.n_init,
                "batch_size": self.batch_size,
                "max_iter": self.max_iter,
                "max_no_improvement": self.max_no_improvement,
                "tol": self.tol,
                "random_state": self.random_state,
            }
        )
        return params
