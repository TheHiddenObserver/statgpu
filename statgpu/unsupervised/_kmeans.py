"""K-Means clustering."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from scipy import sparse

from statgpu._base import BaseEstimator
from statgpu._config import Device
from statgpu.unsupervised._utils import check_2d_array, scalar_to_float, scalar_to_int


class KMeans(BaseEstimator):
    """
    Lloyd K-Means clustering with NumPy, CuPy, or Torch backends.

    Parameters
    ----------
    n_clusters : int, default=8
        Number of clusters.
    init : {'k-means++', 'random'}, default='k-means++'
        Initialization strategy. ``'k-means++'`` uses greedy local trials,
        matching the practical variant used by scikit-learn.
    n_init : 'auto' or int, default='auto'
        Number of initializations. ``'auto'`` means 1 for k-means++ and 10
        for random initialization.
    max_iter : int, default=300
        Maximum Lloyd iterations per initialization.
    tol : float, default=1e-4
        Absolute convergence tolerance on center movement.
    random_state : int or None, default=None
        Random seed for deterministic initialization.
    device : {'auto', 'cpu', 'cuda', 'torch'}, default='auto'
        Compute device.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        init: str = "k-means++",
        n_init: Union[str, int] = "auto",
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _validate_params(self, n_samples: int):
        if not isinstance(self.n_clusters, (int, np.integer)) or int(self.n_clusters) < 1:
            raise ValueError("n_clusters must be a positive integer")
        n_clusters = int(self.n_clusters)
        if n_clusters > n_samples:
            raise ValueError("n_clusters must be less than or equal to n_samples")
        if self.init not in ("k-means++", "random"):
            if callable(self.init):
                raise NotImplementedError("callable init is not supported in KMeans v1")
            raise ValueError("init must be one of: 'k-means++', 'random'")
        if self.n_init == "auto":
            n_init = 1 if self.init == "k-means++" else 10
        else:
            if not isinstance(self.n_init, (int, np.integer)) or int(self.n_init) < 1:
                raise ValueError("n_init must be 'auto' or a positive integer")
            n_init = int(self.n_init)
        if not isinstance(self.max_iter, (int, np.integer)) or int(self.max_iter) < 1:
            raise ValueError("max_iter must be a positive integer")
        if float(self.tol) < 0.0:
            raise ValueError("tol must be non-negative")
        return n_clusters, n_init

    def _squared_distances(self, backend, X, centers):
        x_norm = backend.sum(X * X, axis=1, keepdims=True)
        return self._squared_distances_with_x_norm(backend, X, x_norm, centers)

    def _squared_distances_with_x_norm(self, backend, X, x_norm, centers):
        c_norm = backend.sum(centers * centers, axis=1, keepdims=False)
        c_norm = backend.reshape(c_norm, (1, centers.shape[0]))
        distances = x_norm + c_norm - 2.0 * backend.matmul(X, centers.T)
        return backend.maximum(distances, 0.0)

    def _init_centers(self, backend, X, rng, n_clusters, x_norm):
        n_samples = X.shape[0]
        if self.init == "random":
            indices = rng.choice(n_samples, size=n_clusters, replace=False)
            indices_backend = backend.asarray(indices, dtype=backend.int64)
            return backend.copy(X[indices_backend])

        first = int(rng.integers(0, n_samples))
        centers = [backend.copy(X[first])]
        closest_dist_sq = self._squared_distances_with_x_norm(backend, X, x_norm, backend.reshape(centers[0], (1, X.shape[1])))[:, 0]
        selected = [first]
        n_local_trials = 2 + int(np.log(n_clusters))
        for _ in range(1, n_clusters):
            probs = backend.to_numpy(closest_dist_sq).astype(np.float64, copy=False)
            total = float(np.sum(probs))
            if total <= 0.0 or not np.isfinite(total):
                candidates = np.setdiff1d(np.arange(n_samples), np.asarray(selected))
                next_idx = int(rng.choice(candidates)) if candidates.size else first
                centers.append(backend.copy(X[next_idx]))
                new_dist_sq = self._squared_distances_with_x_norm(backend, X, x_norm, backend.reshape(centers[-1], (1, X.shape[1])))[:, 0]
                closest_dist_sq = backend.minimum(closest_dist_sq, new_dist_sq)
            else:
                candidate_ids = rng.choice(n_samples, size=n_local_trials, replace=True, p=probs / total)
                candidate_ids_backend = backend.asarray(candidate_ids, dtype=backend.int64)
                candidate_centers = X[candidate_ids_backend]
                candidate_dist_sq = self._squared_distances_with_x_norm(backend, X, x_norm, candidate_centers)
                trial_dist_sq = backend.minimum(backend.expand_dims(closest_dist_sq, axis=1), candidate_dist_sq)
                potentials = backend.sum(trial_dist_sq, axis=0)
                best_trial = scalar_to_int(backend.argmin(potentials, axis=0))
                next_idx = int(candidate_ids[best_trial])
                centers.append(backend.copy(candidate_centers[best_trial]))
                closest_dist_sq = trial_dist_sq[:, best_trial]
            selected.append(next_idx)
        return backend.stack(centers, axis=0)

    def _labels_min_distances(self, backend, distances):
        labels = backend.argmin(distances, axis=1)
        gathered = backend.take_along_axis(distances, backend.expand_dims(labels, axis=1), axis=1)
        return labels, backend.reshape(gathered, (distances.shape[0],))

    def _bincount(self, backend, labels, n_clusters):
        if backend.name == "torch":
            return backend.xp.bincount(labels, minlength=n_clusters)
        return backend.xp.bincount(labels, minlength=n_clusters)

    def _label_sums(self, backend, X, labels, n_clusters):
        if backend.name == "numpy":
            if X.shape[0] * n_clusters <= 5_000_000:
                indicator = np.zeros((X.shape[0], n_clusters), dtype=X.dtype)
                indicator[np.arange(X.shape[0]), labels] = 1.0
                return indicator.T @ X
            return np.stack(
                [np.bincount(labels, weights=X[:, j], minlength=n_clusters) for j in range(X.shape[1])],
                axis=1,
            )
        if backend.name == "torch":
            sums = backend.zeros((n_clusters, X.shape[1]), dtype=X.dtype)
            return sums.index_add(0, labels, X)
        sums = backend.zeros((n_clusters, X.shape[1]), dtype=X.dtype)
        backend.xp.add.at(sums, labels, X)
        return sums

    def _compute_centers(self, backend, X, labels, min_dist_sq, centers):
        n_clusters = int(centers.shape[0])
        counts = self._bincount(backend, labels, n_clusters)
        sums = self._label_sums(backend, X, labels, n_clusters)
        safe_counts = backend.maximum(backend.reshape(counts, (n_clusters, 1)), 1)
        new_centers = sums / safe_counts

        empty = backend.to_numpy(counts == 0)
        if np.any(empty):
            replacement = scalar_to_int(backend.argmax(min_dist_sq, axis=0))
            bool_dtype = getattr(backend, "bool", getattr(backend.xp, "bool_", bool))
            empty_backend = backend.asarray(empty, dtype=bool_dtype)
            new_centers[empty_backend] = X[replacement]
        return new_centers

    def _run_single(self, backend, X, rng, n_clusters):
        x_norm = backend.sum(X * X, axis=1, keepdims=True)
        centers = self._init_centers(backend, X, rng, n_clusters, x_norm)
        labels = None
        min_dist_sq = None
        n_iter = 0
        for n_iter in range(1, int(self.max_iter) + 1):
            distances = self._squared_distances_with_x_norm(backend, X, x_norm, centers)
            labels, min_dist_sq = self._labels_min_distances(backend, distances)
            new_centers = self._compute_centers(backend, X, labels, min_dist_sq, centers)
            center_shift = backend.sum((new_centers - centers) ** 2)
            centers = new_centers
            if scalar_to_float(center_shift) <= float(self.tol):
                break
        distances = self._squared_distances_with_x_norm(backend, X, x_norm, centers)
        labels, min_dist_sq = self._labels_min_distances(backend, distances)
        inertia = scalar_to_float(backend.sum(min_dist_sq))
        return centers, labels, inertia, n_iter

    def fit(self, X, y=None, sample_weight=None):
        if sparse.issparse(X):
            raise NotImplementedError("sparse input is not supported in KMeans v1")
        if sample_weight is not None:
            raise NotImplementedError("sample_weight is not supported in KMeans v1")
        backend = self._get_backend()
        X_arr = backend.asarray(X, dtype=backend.float64)
        check_2d_array(X_arr)
        n_samples, n_features = X_arr.shape
        n_clusters, n_init = self._validate_params(n_samples)

        rng = np.random.default_rng(self.random_state)
        best = None
        for _ in range(n_init):
            centers, labels, inertia, n_iter = self._run_single(backend, X_arr, rng, n_clusters)
            if best is None or inertia < best[2]:
                best = (centers, labels, inertia, n_iter)

        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = best
        self.n_features_in_ = int(n_features)
        self._backend_name = backend.name
        self._fitted = True
        return self

    def transform(self, X):
        self._check_is_fitted()
        if sparse.issparse(X):
            raise NotImplementedError("sparse input is not supported in KMeans v1")
        backend = self._get_backend()
        X_arr = backend.asarray(X, dtype=backend.float64)
        check_2d_array(X_arr)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X_arr.shape[1]} features, expected {self.n_features_in_}")
        return backend.sqrt(self._squared_distances(backend, X_arr, self.cluster_centers_))

    def predict(self, X):
        self._check_is_fitted()
        if sparse.issparse(X):
            raise NotImplementedError("sparse input is not supported in KMeans v1")
        backend = self._get_backend()
        X_arr = backend.asarray(X, dtype=backend.float64)
        check_2d_array(X_arr)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X_arr.shape[1]} features, expected {self.n_features_in_}")
        distances = self._squared_distances(backend, X_arr, self.cluster_centers_)
        return backend.argmin(distances, axis=1)

    def fit_predict(self, X, y=None):
        return self.fit(X, y=y).labels_

    def score(self, X, y=None):
        self._check_is_fitted()
        backend = self._get_backend()
        X_arr = backend.asarray(X, dtype=backend.float64)
        distances = self._squared_distances(backend, X_arr, self.cluster_centers_)
        min_dist_sq = backend.min(distances, axis=1)
        return -scalar_to_float(backend.sum(min_dist_sq))

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params.update(
            {
                "n_clusters": self.n_clusters,
                "init": self.init,
                "n_init": self.n_init,
                "max_iter": self.max_iter,
                "tol": self.tol,
                "random_state": self.random_state,
            }
        )
        return params
