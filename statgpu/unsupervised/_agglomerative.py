"""Agglomerative clustering."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from statgpu._base import BaseEstimator
from statgpu._config import Device
from statgpu.unsupervised._utils import check_2d_array, reject_sparse


class AgglomerativeClustering(BaseEstimator):
    """Exact CPU single-linkage agglomerative clustering."""

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
        if self.linkage != "single":
            raise NotImplementedError("AgglomerativeClustering v1 only supports linkage='single'")
        if self.metric != "euclidean":
            raise NotImplementedError("AgglomerativeClustering v1 only supports metric='euclidean'")

    def _ensure_cpu_supported(self):
        if self.device in (Device.CUDA, Device.TORCH):
            raise NotImplementedError(
                "AgglomerativeClustering v1 only supports CPU execution; "
                "explicit device='cuda' or device='torch' is not silently downgraded"
            )

    def fit(self, X, y=None):
        reject_sparse(X, "AgglomerativeClustering")
        self._ensure_cpu_supported()
        X_arr = np.asarray(X, dtype=np.float64)
        check_2d_array(X_arr)
        n_samples, n_features = X_arr.shape
        self._validate_params(n_samples)

        if n_samples == 1:
            children = np.empty((0, 2), dtype=np.int64)
            distances = np.empty((0,), dtype=np.float64)
            labels = np.zeros(1, dtype=np.int64)
        else:
            Z = linkage(X_arr, method="single", metric="euclidean")
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
