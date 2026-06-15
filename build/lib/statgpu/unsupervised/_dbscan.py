"""Density-based spatial clustering."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

from statgpu._base import BaseEstimator
from statgpu._config import Device
from statgpu.unsupervised._utils import check_2d_array, reject_sparse, scalar_to_int

try:
    from statgpu.unsupervised._dbscan_cpu import dbscan_dense_pairwise
except Exception:  # pragma: no cover - optional compiled extension
    dbscan_dense_pairwise = None


class DBSCAN(BaseEstimator):
    """DBSCAN clustering for dense Euclidean data."""

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = "euclidean",
        batch_size: Optional[int] = None,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.batch_size = batch_size

    def _validate_params(self, n_samples: int):
        if float(self.eps) <= 0.0:
            raise ValueError("eps must be positive")
        if not isinstance(self.min_samples, (int, np.integer)) or int(self.min_samples) < 1:
            raise ValueError("min_samples must be a positive integer")
        if self.metric != "euclidean":
            raise NotImplementedError("DBSCAN v1 only supports metric='euclidean'")
        if self.batch_size is not None:
            if not isinstance(self.batch_size, (int, np.integer)) or int(self.batch_size) < 1:
                raise ValueError("batch_size must be None or a positive integer")
        if n_samples < 1:
            raise ValueError("DBSCAN requires at least one sample")

    def _fit_numpy(self, X):
        X_np = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X_np.shape
        tree = cKDTree(X_np)
        workers = 1 if self.n_jobs is None else int(self.n_jobs)

        sample_size = min(n_samples, 512)
        sample_idx = np.linspace(0, n_samples - 1, sample_size, dtype=np.int64)
        sample_counts = self._query_counts(tree, X_np[sample_idx], sample_size, workers)
        sample_core_fraction = float(np.mean(sample_counts >= int(self.min_samples)))
        dense_threshold = max(64.0, float(self.min_samples) * 8.0)
        if float(np.mean(sample_counts)) >= dense_threshold:
            dense_pairwise_limit = 10000 if dbscan_dense_pairwise is not None else 0
            condensed_bytes = n_samples * (n_samples - 1) // 2 * np.dtype(np.float64).itemsize
            if n_samples <= dense_pairwise_limit and sample_core_fraction >= 0.75:
                labels, core_indices = dbscan_dense_pairwise(
                    np.ascontiguousarray(X_np, dtype=np.float64),
                    float(self.eps),
                    int(self.min_samples),
                )
            elif condensed_bytes <= 256 * 1024 * 1024:
                labels, core_indices = self._fit_numpy_dense_pdist(X_np)
            else:
                labels, core_indices = self._fit_numpy_dense(tree, n_samples)
        else:
            labels, core_indices = self._fit_numpy_sparse(tree, X_np, n_samples, workers)

        self.labels_ = labels
        self.core_sample_indices_ = core_indices
        self.components_ = X_np[core_indices] if core_indices.size else X_np[:0]
        self.n_features_in_ = int(n_features)
        self._backend_name = "numpy"
        self._fitted = True
        return self

    def _query_counts(self, tree, X, n_rows, workers):
        try:
            counts = tree.query_ball_point(X, r=float(self.eps), workers=workers, return_length=True)
        except TypeError:
            counts = np.fromiter(
                (len(row) for row in tree.query_ball_point(X, r=float(self.eps))),
                dtype=np.int64,
                count=n_rows,
            )
        return np.asarray(counts, dtype=np.int64)

    def _fit_numpy_dense(self, tree, n_samples):
        neighbors = tree.sparse_distance_matrix(tree, float(self.eps), output_type="coo_matrix")
        row_idx = neighbors.row.astype(np.int64, copy=False)
        col_idx = neighbors.col.astype(np.int64, copy=False)
        counts = np.bincount(row_idx, minlength=n_samples)
        return self._labels_from_neighbor_edges(n_samples, counts, row_idx, col_idx)

    def _fit_numpy_dense_pdist(self, X_np):
        n_samples = X_np.shape[0]
        distances = pdist(X_np, metric="sqeuclidean")
        pair_rows, pair_cols = self._condensed_indices_to_pairs(distances <= float(self.eps) ** 2, n_samples)
        row_idx = np.concatenate([np.arange(n_samples, dtype=np.int64), pair_rows, pair_cols])
        col_idx = np.concatenate([np.arange(n_samples, dtype=np.int64), pair_cols, pair_rows])
        counts = np.bincount(row_idx, minlength=n_samples)
        return self._labels_from_neighbor_edges(n_samples, counts, row_idx, col_idx)

    def _condensed_indices_to_pairs(self, mask, n_samples):
        condensed = np.flatnonzero(mask)
        if not condensed.size:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty
        b = 1 - 2 * n_samples
        rows = np.floor((-b - np.sqrt(float(b * b) - 8.0 * condensed)) / 2.0).astype(np.int64)
        row_start = n_samples * rows - rows * (rows + 1) // 2
        cols = condensed - row_start + rows + 1
        return rows, cols.astype(np.int64, copy=False)

    def _fit_numpy_sparse(self, tree, X_np, n_samples, workers):
        counts = self._query_counts(tree, X_np, n_samples, workers)
        core_mask = counts >= int(self.min_samples)
        core_indices = np.flatnonzero(core_mask).astype(np.int64)
        if not core_indices.size:
            return np.full(n_samples, -1, dtype=np.int64), core_indices
        try:
            pairs = tree.query_pairs(float(self.eps), output_type="ndarray")
        except TypeError:
            pairs = np.asarray(list(tree.query_pairs(float(self.eps))), dtype=np.int64)
        pairs = np.asarray(pairs, dtype=np.int64)
        if pairs.size:
            row_idx = np.concatenate([pairs[:, 0], pairs[:, 1]])
            col_idx = np.concatenate([pairs[:, 1], pairs[:, 0]])
        else:
            row_idx = np.empty(0, dtype=np.int64)
            col_idx = np.empty(0, dtype=np.int64)
        return self._labels_from_neighbor_edges(n_samples, counts, row_idx, col_idx)

    def _labels_from_neighbor_edges(self, n_samples, counts, row_idx, col_idx):
        counts = np.asarray(counts, dtype=np.int64)
        core_mask = counts >= int(self.min_samples)
        core_indices = np.flatnonzero(core_mask).astype(np.int64)
        labels = np.full(n_samples, -1, dtype=np.int64)
        if not core_indices.size:
            return labels, core_indices

        core_position = np.full(n_samples, -1, dtype=np.int64)
        core_position[core_indices] = np.arange(core_indices.size, dtype=np.int64)
        core_edges = core_mask[row_idx] & core_mask[col_idx]
        graph = csr_matrix(
            (
                np.ones(int(np.sum(core_edges)), dtype=bool),
                (core_position[row_idx[core_edges]], core_position[col_idx[core_edges]]),
            ),
            shape=(core_indices.size, core_indices.size),
        )
        _, core_labels = connected_components(graph, directed=False, return_labels=True)
        labels[core_indices] = core_labels.astype(np.int64, copy=False)

        border_edges = (~core_mask[row_idx]) & core_mask[col_idx]
        if np.any(border_edges):
            border_rows = row_idx[border_edges]
            border_labels = labels[col_idx[border_edges]]
            order = np.argsort(border_rows, kind="mergesort")
            border_rows = border_rows[order]
            border_labels = border_labels[order]
            first = np.r_[True, border_rows[1:] != border_rows[:-1]]
            labels[border_rows[first]] = border_labels[first]
        return labels, core_indices

    def _neighbor_graph(self, backend, X):
        n_samples = X.shape[0]
        batch_size = n_samples if self.batch_size is None else min(int(self.batch_size), n_samples)
        x_norm = backend.sum(X * X, axis=1, keepdims=True)
        rows = []
        eps_sq = float(self.eps) ** 2
        for start in range(0, n_samples, batch_size):
            stop = min(start + batch_size, n_samples)
            X_chunk = X[start:stop]
            chunk_norm = x_norm[start:stop]
            distances = chunk_norm + backend.reshape(x_norm, (1, n_samples)) - 2.0 * backend.matmul(X_chunk, X.T)
            rows.append(backend.maximum(distances, 0.0) <= eps_sq)
        return backend.concatenate(rows, axis=0) if len(rows) > 1 else rows[0]

    def fit(self, X, y=None):
        reject_sparse(X, "DBSCAN")
        backend = self._get_backend()
        X_arr = backend.asarray(X, dtype=backend.float64)
        check_2d_array(X_arr)
        n_samples, n_features = X_arr.shape
        self._validate_params(n_samples)
        if backend.name == "numpy":
            return self._fit_numpy(X_arr)

        neighbors = self._neighbor_graph(backend, X_arr)
        counts = backend.sum(neighbors, axis=1)
        core_mask = counts >= int(self.min_samples)
        core_adj = neighbors & backend.expand_dims(core_mask, 0) & backend.expand_dims(core_mask, 1)

        large = int(n_samples)
        initial = backend.arange(n_samples, dtype=backend.int64)
        labels = backend.where(core_mask, initial, backend.full((n_samples,), large, dtype=backend.int64))
        for _ in range(n_samples):
            candidate_labels = backend.where(
                core_adj,
                backend.expand_dims(labels, 0),
                backend.full((n_samples, n_samples), large, dtype=backend.int64),
            )
            new_labels = backend.min(candidate_labels, axis=1)
            new_labels = backend.where(core_mask, new_labels, backend.full((n_samples,), large, dtype=backend.int64))
            changed = scalar_to_int(backend.sum(new_labels != labels))
            labels = new_labels
            if changed == 0:
                break

        labels_np = np.full(n_samples, -1, dtype=np.int64)
        core_np = backend.to_numpy(core_mask).astype(bool, copy=False)
        raw_core_labels = backend.to_numpy(labels).astype(np.int64, copy=False)
        unique_core = sorted(int(v) for v in np.unique(raw_core_labels[core_np]) if int(v) < large)
        label_map = {raw: i for i, raw in enumerate(unique_core)}
        for i in np.flatnonzero(core_np):
            labels_np[i] = label_map[int(raw_core_labels[i])]

        neighbors_np = backend.to_numpy(neighbors)
        core_indices = np.flatnonzero(core_np).astype(np.int64)
        for i in np.flatnonzero(~core_np):
            reachable_core = core_indices[neighbors_np[i, core_indices]]
            if reachable_core.size:
                labels_np[i] = labels_np[int(reachable_core[0])]

        core_backend = backend.asarray(core_indices, dtype=backend.int64)
        self.labels_ = backend.asarray(labels_np, dtype=backend.int64)
        self.core_sample_indices_ = core_backend
        self.components_ = X_arr[core_backend] if core_indices.size else X_arr[:0]
        self.n_features_in_ = int(n_features)
        self._backend_name = backend.name
        self._fitted = True
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X, y=y).labels_

    def predict(self, X):
        raise NotImplementedError("DBSCAN does not support predict for unseen samples")

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params.update(
            {
                "eps": self.eps,
                "min_samples": self.min_samples,
                "metric": self.metric,
                "batch_size": self.batch_size,
            }
        )
        return params
