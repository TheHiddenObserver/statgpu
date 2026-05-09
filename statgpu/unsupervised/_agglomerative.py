"""Agglomerative clustering."""

from __future__ import annotations

import os
from typing import Optional, Union

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from statgpu._base import BaseEstimator
from statgpu._config import Device
from statgpu.unsupervised._utils import check_2d_array, reject_sparse, squared_euclidean_distances


class AgglomerativeClustering(BaseEstimator):
    """Exact dense agglomerative clustering."""

    _GPU_DISTANCE_LIMIT_BYTES = int(
        os.environ.get("STATGPU_AGGLOMERATIVE_GPU_MAX_BYTES", str(1 << 30))
    )

    def __init__(
        self,
        n_clusters: int = 2,
        linkage: str = "single",
        metric: str = "euclidean",
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.metric = metric

    def _validate_params(self, n_samples: int):
        if not isinstance(self.n_clusters, (int, np.integer)) or int(self.n_clusters) < 1:
            raise ValueError("n_clusters must be a positive integer")
        if int(self.n_clusters) > n_samples:
            raise ValueError("n_clusters must be less than or equal to n_samples")
        if self.linkage not in ("single", "complete", "average", "ward"):
            raise ValueError("linkage must be one of: 'single', 'complete', 'average', 'ward'")
        if self.metric != "euclidean":
            raise NotImplementedError("AgglomerativeClustering only supports metric='euclidean'")

    def _use_gpu_path(self) -> bool:
        return self.device in (Device.CUDA, Device.TORCH)

    def _check_gpu_memory(self, n_samples: int):
        required = int(n_samples) * int(n_samples) * 8
        if required > self._GPU_DISTANCE_LIMIT_BYTES:
            limit_mb = self._GPU_DISTANCE_LIMIT_BYTES / (1024**2)
            required_mb = required / (1024**2)
            raise MemoryError(
                "AgglomerativeClustering GPU exact path requires a dense "
                f"distance matrix of about {required_mb:.1f} MiB, exceeding "
                f"the configured limit {limit_mb:.1f} MiB. Use device='cpu' "
                "or raise STATGPU_AGGLOMERATIVE_GPU_MAX_BYTES explicitly."
            )

    @staticmethod
    def _labels_from_children(n_samples: int, n_clusters: int, children: np.ndarray) -> np.ndarray:
        clusters = {i: [i] for i in range(n_samples)}
        next_id = n_samples
        merges_to_apply = max(0, n_samples - int(n_clusters))
        for left, right in children[:merges_to_apply]:
            members = clusters.pop(int(left)) + clusters.pop(int(right))
            clusters[next_id] = members
            next_id += 1

        labels = np.empty(n_samples, dtype=np.int64)
        for label, members in enumerate(clusters.values()):
            labels[np.asarray(members, dtype=np.int64)] = label
        return labels

    def _fit_gpu(self, X):
        backend = self._get_backend()
        X_arr = self._to_array(X, backend=backend.name)
        X_arr = backend.asarray(X_arr, dtype=backend.float64)
        check_2d_array(X_arr)
        n_samples, n_features = X_arr.shape
        self._validate_params(n_samples)
        self._check_gpu_memory(n_samples)

        if n_samples == 1:
            self.labels_ = np.zeros(1, dtype=np.int64)
            self.children_ = np.empty((0, 2), dtype=np.int64)
            self.distances_ = np.empty((0,), dtype=np.float64)
            self.n_features_in_ = int(n_features)
            self._backend_name = backend.name
            self._fitted = True
            return self

        D = squared_euclidean_distances(backend, X_arr)
        if self.linkage != "ward":
            D = backend.sqrt(D)
        inf = float("inf")
        indices = backend.arange(n_samples, dtype=backend.int64)
        D[indices, indices] = inf

        children = np.empty((n_samples - 1, 2), dtype=np.int64)
        distances = np.empty(n_samples - 1, dtype=np.float64)
        cluster_ids = list(range(n_samples))
        cluster_sizes = [1.0] * n_samples
        active = [True] * n_samples

        for step in range(n_samples - 1):
            flat_idx = int(backend.item(backend.argmin(D)))
            a = flat_idx // n_samples
            b = flat_idx % n_samples
            if b < a:
                a, b = b, a

            merge_value = float(backend.item(D[a, b]))
            children[step] = (cluster_ids[a], cluster_ids[b])
            distances[step] = np.sqrt(max(merge_value, 0.0)) if self.linkage == "ward" else merge_value

            active_ids = [idx for idx, is_active in enumerate(active) if is_active and idx not in (a, b)]
            if active_ids:
                idx_arr = backend.asarray(active_ids, dtype=backend.int64)
                da = D[a, idx_arr]
                db = D[b, idx_arr]
                size_a = cluster_sizes[a]
                size_b = cluster_sizes[b]

                if self.linkage == "single":
                    updated = backend.minimum(da, db)
                elif self.linkage == "complete":
                    updated = backend.maximum(da, db)
                elif self.linkage == "average":
                    updated = (size_a * da + size_b * db) / (size_a + size_b)
                else:
                    size_k = backend.asarray([cluster_sizes[idx] for idx in active_ids], dtype=backend.float64)
                    total = size_a + size_b + size_k
                    updated = (
                        ((size_k + size_a) / total) * da
                        + ((size_k + size_b) / total) * db
                        - (size_k / total) * merge_value
                    )
                    updated = backend.maximum(updated, 0.0)

                D[a, idx_arr] = updated
                D[idx_arr, a] = updated

            active[b] = False
            cluster_ids[a] = n_samples + step
            cluster_sizes[a] += cluster_sizes[b]
            cluster_sizes[b] = 0.0
            D[b, :] = inf
            D[:, b] = inf
            D[a, a] = inf

        self.children_ = children
        self.distances_ = distances
        self.labels_ = self._labels_from_children(n_samples, int(self.n_clusters), children)
        self.n_features_in_ = int(n_features)
        self._backend_name = backend.name
        self._fitted = True
        return self

    def fit(self, X, y=None):
        reject_sparse(X, "AgglomerativeClustering")
        if self._use_gpu_path():
            return self._fit_gpu(X)

        X_arr = np.asarray(X, dtype=np.float64)
        check_2d_array(X_arr)
        n_samples, n_features = X_arr.shape
        self._validate_params(n_samples)

        if n_samples == 1:
            children = np.empty((0, 2), dtype=np.int64)
            distances = np.empty((0,), dtype=np.float64)
            labels = np.zeros(1, dtype=np.int64)
        else:
            Z = linkage(X_arr, method=self.linkage, metric="euclidean")
            children = Z[:, :2].astype(np.int64, copy=False)
            distances = Z[:, 2].astype(np.float64, copy=False)
            labels = fcluster(Z, t=int(self.n_clusters), criterion="maxclust").astype(np.int64) - 1

        self.labels_ = labels
        self.children_ = children
        self.distances_ = distances
        self.n_features_in_ = int(n_features)
        self._backend_name = "numpy"
        self._fitted = True
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X, y=y).labels_

    def predict(self, X):
        raise NotImplementedError("AgglomerativeClustering does not support predict for unseen samples")

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params.update(
            {
                "n_clusters": self.n_clusters,
                "linkage": self.linkage,
                "metric": self.metric,
            }
        )
        return params
