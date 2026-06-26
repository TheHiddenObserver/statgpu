"""Density-based spatial clustering."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from statgpu._base import BaseEstimator
from statgpu._config import Device
from statgpu.unsupervised._utils import check_2d_array, reject_sparse, scalar_to_int

# Optional Cython fast path (compiled via setup.py)
try:
    from statgpu.unsupervised._dbscan_cy_fast import (
        dbscan_labels_from_csr,
        dbscan_labels_from_pairs,
    )

    _HAS_CY_FAST = True
except Exception:  # pragma: no cover
    _HAS_CY_FAST = False


class DBSCAN(BaseEstimator):
    """DBSCAN clustering for dense Euclidean data.

    CPU strategy:
      - p ≤ 12 (low-dim): cKDTree ``query_pairs`` → Cython Union-Find
      - p > 12 (high-dim): sklearn ``radius_neighbors_graph`` → Cython CSR

    GPU strategy:
      - Batched distance computation → sparse neighbor graph → connected components
    """

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
            raise NotImplementedError("DBSCAN only supports metric='euclidean'")
        if self.batch_size is not None:
            if not isinstance(self.batch_size, (int, np.integer)) or int(self.batch_size) < 1:
                raise ValueError("batch_size must be None or a positive integer")
        if n_samples < 1:
            raise ValueError("DBSCAN requires at least one sample")

    # ------------------------------------------------------------------ #
    #  CPU dispatch                                                       #
    # ------------------------------------------------------------------ #

    def _fit_numpy(self, X):
        X_np = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X_np.shape

        if n_features <= 12:
            labels, core_indices = self._fit_numpy_kdtree(X_np, n_samples)
        else:
            labels, core_indices = self._fit_numpy_bruteforce(X_np, n_samples)

        self.labels_ = labels
        self.core_sample_indices_ = core_indices
        self.components_ = X_np[core_indices] if core_indices.size else X_np[:0]
        self.n_features_in_ = int(n_features)
        self._backend_name = "numpy"
        self._fitted = True
        return self

    def _fit_numpy_kdtree(self, X_np, n_samples):
        """Low-dim path: cKDTree query_pairs (single tree traversal)."""
        eps = float(self.eps)
        min_samples = int(self.min_samples)
        tree = cKDTree(X_np)

        pairs = tree.query_pairs(r=eps, output_type="ndarray")
        if not pairs.size:
            return np.full(n_samples, -1, dtype=np.int64), np.empty(0, dtype=np.int64)

        pairs64 = np.ascontiguousarray(pairs, dtype=np.int64)

        if _HAS_CY_FAST:
            return dbscan_labels_from_pairs(n_samples, min_samples, pairs64)

        # Pure Python fallback
        row_idx = np.concatenate([pairs64[:, 0], pairs64[:, 1]])
        col_idx = np.concatenate([pairs64[:, 1], pairs64[:, 0]])
        counts = np.bincount(row_idx, minlength=n_samples)
        return self._labels_from_edges(n_samples, counts, row_idx, col_idx)

    def _fit_numpy_bruteforce(self, X_np, n_samples):
        """High-dim path: sklearn brute-force BLAS + Cython CSR."""
        from sklearn.neighbors import NearestNeighbors

        eps = float(self.eps)
        min_samples = int(self.min_samples)

        nn = NearestNeighbors(radius=eps, algorithm="brute", metric="euclidean")
        nn.fit(X_np)
        csr = nn.radius_neighbors_graph(X_np, mode="connectivity").tocsr()

        if _HAS_CY_FAST:
            return dbscan_labels_from_csr(
                n_samples, min_samples,
                csr.indptr.astype(np.int64),
                csr.indices.astype(np.int64),
            )

        # Pure Python fallback
        counts = np.asarray(csr.sum(axis=1)).flatten().astype(np.int64)
        core_mask = counts >= min_samples
        core_indices = np.flatnonzero(core_mask).astype(np.int64)
        if not core_indices.size:
            return np.full(n_samples, -1, dtype=np.int64), core_indices

        _, core_labels = connected_components(
            csr[core_indices][:, core_indices], directed=False, return_labels=True
        )
        labels = np.full(n_samples, -1, dtype=np.int64)
        labels[core_indices] = core_labels.astype(np.int64)

        # Border points
        border_mask = ~core_mask
        if np.any(border_mask):
            bg = csr[border_mask][:, core_indices].tocsr()
            for i, idx in enumerate(np.flatnonzero(border_mask)):
                s, e = bg.indptr[i], bg.indptr[i + 1]
                if s < e:
                    labels[idx] = core_labels[bg.indices[s]]

        return labels, core_indices

    # ------------------------------------------------------------------ #
    #  Pure Python fallback (used when Cython is not compiled)            #
    # ------------------------------------------------------------------ #

    def _labels_from_edges(self, n_samples, counts, row_idx, col_idx):
        """Build labels from edge lists — pure Python fallback."""
        counts = np.asarray(counts, dtype=np.int64)
        core_mask = counts >= int(self.min_samples)
        core_indices = np.flatnonzero(core_mask).astype(np.int64)
        labels = np.full(n_samples, -1, dtype=np.int64)
        if not core_indices.size:
            return labels, core_indices

        n_core = core_indices.size
        core_position = np.full(n_samples, -1, dtype=np.int64)
        core_position[core_indices] = np.arange(n_core, dtype=np.int64)

        # Filter core-core edges
        core_edges = core_mask[row_idx] & core_mask[col_idx]
        core_row = core_position[row_idx[core_edges]]
        core_col = core_position[col_idx[core_edges]]

        # Connected components
        if len(core_row) > 0:
            graph = csr_matrix(
                (np.ones(len(core_row), dtype=bool), (core_row, core_col)),
                shape=(n_core, n_core),
            )
            _, core_labels = connected_components(graph, directed=False, return_labels=True)
        else:
            core_labels = np.zeros(n_core, dtype=np.int32)

        labels[core_indices] = core_labels.astype(np.int64, copy=False)

        # Border points (both directions)
        d1 = (~core_mask[row_idx]) & core_mask[col_idx]
        d2 = core_mask[row_idx] & (~core_mask[col_idx])
        if np.any(d1) or np.any(d2):
            parts_r, parts_l = [], []
            if np.any(d1):
                parts_r.append(row_idx[d1])
                parts_l.append(labels[col_idx[d1]])
            if np.any(d2):
                parts_r.append(col_idx[d2])
                parts_l.append(labels[row_idx[d2]])
            br = np.concatenate(parts_r)
            bl = np.concatenate(parts_l)
            order = np.argsort(br, kind="mergesort")
            br, bl = br[order], bl[order]
            first = np.r_[True, br[1:] != br[:-1]]
            labels[br[first]] = bl[first]

        return labels, core_indices

    # ------------------------------------------------------------------ #
    #  GPU path                                                           #
    # ------------------------------------------------------------------ #

    def _neighbor_graph_sparse(self, backend, X):
        """Build sparse neighbor graph from GPU distance computation."""
        n_samples = X.shape[0]
        if self.batch_size is not None:
            batch_size = min(int(self.batch_size), n_samples)
        else:
            batch_size = min(n_samples, max(1000, 400_000_000 // (n_samples * 4)))

        X_f32 = backend.asarray(X, dtype=backend.float32) if hasattr(backend, "float32") else X
        x_norm = backend.sum(X_f32 * X_f32, axis=1, keepdims=True)
        eps_sq = float(self.eps) ** 2

        all_rows, all_cols = [], []
        for start in range(0, n_samples, batch_size):
            stop = min(start + batch_size, n_samples)
            distances = (
                x_norm[start:stop]
                + backend.reshape(x_norm, (1, n_samples))
                - 2.0 * backend.matmul(X_f32[start:stop], X_f32.T)
            )
            distances = backend.maximum(distances, 0.0)
            mask = distances <= eps_sq

            if hasattr(mask, "cpu"):  # torch
                import torch

                nz = torch.nonzero(mask, as_tuple=False)
                nz_np = nz.cpu().numpy()
            elif hasattr(mask, "get"):  # cupy
                import cupy as cp

                nz_np = cp.argwhere(mask).get()
            else:
                nz_np = np.argwhere(mask)

            if len(nz_np) > 0:
                all_rows.append(nz_np[:, 0] + start)
                all_cols.append(nz_np[:, 1])

        if all_rows:
            row_idx = np.concatenate(all_rows).astype(np.int64)
            col_idx = np.concatenate(all_cols).astype(np.int64)
        else:
            row_idx = np.empty(0, dtype=np.int64)
            col_idx = np.empty(0, dtype=np.int64)
        return row_idx, col_idx

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def fit(self, X, y=None):
        reject_sparse(X, "DBSCAN")
        backend = self._get_backend()
        X_arr = backend.asarray(X, dtype=backend.float64)
        check_2d_array(X_arr)
        n_samples, n_features = X_arr.shape
        self._validate_params(n_samples)

        if backend.name == "numpy":
            return self._fit_numpy(X_arr)

        # GPU path
        row_idx, col_idx = self._neighbor_graph_sparse(backend, X_arr)
        counts = np.bincount(row_idx, minlength=n_samples)
        core_mask_np = counts >= int(self.min_samples)
        core_indices = np.flatnonzero(core_mask_np).astype(np.int64)

        if not core_indices.size:
            self.labels_ = backend.asarray(np.full(n_samples, -1, dtype=np.int64), dtype=backend.int64)
            self.core_sample_indices_ = backend.asarray(core_indices, dtype=backend.int64)
            self.components_ = X_arr[:0]
            self.n_features_in_ = int(n_features)
            self._backend_name = backend.name
            self._fitted = True
            return self

        core_edges = core_mask_np[row_idx] & core_mask_np[col_idx]
        core_position = np.full(n_samples, -1, dtype=np.int64)
        core_position[core_indices] = np.arange(core_indices.size, dtype=np.int64)

        graph = csr_matrix(
            (np.ones(np.sum(core_edges), dtype=bool),
             (core_position[row_idx[core_edges]], core_position[col_idx[core_edges]])),
            shape=(core_indices.size, core_indices.size),
        )
        _, core_labels = connected_components(graph, directed=False, return_labels=True)

        labels_np = np.full(n_samples, -1, dtype=np.int64)
        labels_np[core_indices] = core_labels.astype(np.int64)

        border_edges = (~core_mask_np[row_idx]) & core_mask_np[col_idx]
        if np.any(border_edges):
            border_rows = row_idx[border_edges]
            border_labels = labels_np[col_idx[border_edges]]
            order = np.argsort(border_rows, kind="mergesort")
            border_rows, border_labels = border_rows[order], border_labels[order]
            first = np.r_[True, border_rows[1:] != border_rows[:-1]]
            labels_np[border_rows[first]] = border_labels[first]

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
