"""Density-based spatial clustering."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from statgpu._base import BaseEstimator
from statgpu._config import Device
from statgpu.backends._utils import _to_numpy
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

try:
    from statgpu.unsupervised._unionfind import connected_components_uf
    _HAS_UNIONFIND = True
except Exception:
    _HAS_UNIONFIND = False


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
        algorithm: str = "auto",
        batch_size: Optional[int] = None,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.algorithm = algorithm
        self.batch_size = batch_size

    def _validate_params(self, n_samples: int):
        if float(self.eps) <= 0.0:
            raise ValueError("eps must be positive")
        if not isinstance(self.min_samples, (int, np.integer)) or int(self.min_samples) < 1:
            raise ValueError("min_samples must be a positive integer")
        if self.metric != "euclidean":
            raise NotImplementedError("DBSCAN only supports metric='euclidean'")
        if self.algorithm not in ("auto", "brute", "ball_tree", "kd_tree"):
            raise ValueError("algorithm must be one of: 'auto', 'brute', 'ball_tree', 'kd_tree'")
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
        """High-dim path: sklearn neighbor search + Cython label assignment."""
        from sklearn.neighbors import NearestNeighbors

        eps = float(self.eps)
        min_samples = int(self.min_samples)
        algo = self.algorithm if self.algorithm != "auto" else "auto"

        nn = NearestNeighbors(radius=eps, algorithm=algo, metric="euclidean")
        nn.fit(X_np)

        # Use radius_neighbors (raw indices) instead of radius_neighbors_graph (sparse matrix)
        # This avoids the overhead of constructing the sparse matrix
        indices_list, distances_list = nn.radius_neighbors(X_np, return_distance=True)

        # Build CSR arrays directly from index lists
        indptr = np.zeros(n_samples + 1, dtype=np.int64)
        for i in range(n_samples):
            indptr[i + 1] = indptr[i] + len(indices_list[i])

        total_nnz = int(indptr[-1])
        indices = np.empty(total_nnz, dtype=np.int64)
        for i in range(n_samples):
            start = indptr[i]
            end = indptr[i + 1]
            indices[start:end] = indices_list[i]

        if _HAS_CY_FAST:
            return dbscan_labels_from_csr(n_samples, min_samples, indptr, indices)

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

    # ------------------------------------------------------------------ #
    #  GPU path                                                           #
    # ------------------------------------------------------------------ #

    def _neighbor_graph_sparse(self, backend, X):
        """Build sparse neighbor graph using GPU batched distance computation."""
        n_samples = X.shape[0]
        eps = float(self.eps)

        if self.batch_size is not None:
            batch_size = min(int(self.batch_size), n_samples)
        else:
            batch_size = min(n_samples, max(1000, 400_000_000 // (n_samples * 4)))

        X_f32 = backend.asarray(X, dtype=backend.float32) if hasattr(backend, "float32") else X
        x_norm = backend.sum(X_f32 * X_f32, axis=1, keepdims=True)
        eps_sq = eps ** 2

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

        return backend.asarray(row_idx, dtype=backend.int64), backend.asarray(col_idx, dtype=backend.int64)

    # ------------------------------------------------------------------ #
    #  GPU path (fully on-device for torch, no GPU→CPU transfer)         #
    # ------------------------------------------------------------------ #

    def _fit_gpu(self, backend, X_arr, n_samples, n_features):
        """Fully GPU-based DBSCAN: single-pass distance → graph → labels.

        Computes distances once, caches edges as GPU tensors, then processes
        entirely on GPU. Only final labels transferred to CPU.
        """
        import torch

        eps = float(self.eps)
        min_samples = int(self.min_samples)
        device = X_arr.device if hasattr(X_arr, "device") else None
        X_f32 = X_arr.float()
        x_norm = (X_f32 * X_f32).sum(dim=1, keepdim=True)
        eps_sq = eps ** 2

        if self.batch_size is not None:
            batch_size = min(int(self.batch_size), n_samples)
        else:
            batch_size = min(n_samples, max(2000, 2_000_000_000 // (n_samples * 4)))

        # --- Single pass: count neighbors + collect edges (GPU) ---
        counts = torch.zeros(n_samples, dtype=torch.int64, device=device)
        all_rows, all_cols = [], []

        for start in range(0, n_samples, batch_size):
            stop = min(start + batch_size, n_samples)
            dist = x_norm[start:stop] + x_norm.T - 2.0 * X_f32[start:stop] @ X_f32.T
            dist.clamp_(min=0.0)
            mask = dist <= eps_sq
            counts[start:stop] = mask.sum(dim=1)

            nz = torch.nonzero(mask, as_tuple=False)
            if nz.numel() > 0:
                all_rows.append(nz[:, 0] + start)
                all_cols.append(nz[:, 1])
            del dist, mask  # free memory immediately

        # --- Find core points ---
        core_mask = counts >= min_samples
        core_indices = torch.nonzero(core_mask, as_tuple=False).squeeze(1)
        n_core = core_indices.numel()

        if n_core == 0:
            labels = torch.full((n_samples,), -1, dtype=torch.int64, device=device)
            return labels.cpu().numpy(), core_indices.cpu().numpy()

        if not all_rows:
            labels = torch.full((n_samples,), -1, dtype=torch.int64, device=device)
            labels[core_indices] = torch.arange(n_core, device=device, dtype=torch.int64)
            return labels.cpu().numpy(), core_indices.cpu().numpy()

        row_idx = torch.cat(all_rows)
        col_idx = torch.cat(all_cols)

        # --- Core-core graph (filter on GPU) ---
        core_pair = core_mask[row_idx] & core_mask[col_idx]
        core_row = row_idx[core_pair]
        core_col = col_idx[core_pair]

        # Map to core-index space
        core_pos = torch.full((n_samples,), -1, dtype=torch.int64, device=device)
        core_pos[core_indices] = torch.arange(n_core, device=device, dtype=torch.int64)
        cr = core_pos[core_row]
        cc = core_pos[core_col]

        # --- Connected components via label propagation (GPU) ---
        n_edges = cr.numel()
        indices = torch.stack([cr, cc])
        adj = torch.sparse_coo_tensor(
            indices, torch.ones(n_edges, device=device), (n_core, n_core)
        ).coalesce()
        adj_indices = adj.indices()

        labels_core = torch.arange(n_core, device=device, dtype=torch.int64)
        for _ in range(50):
            src_labels = labels_core[adj_indices[0]]
            dst_labels = labels_core[adj_indices[1]]
            min_labels = torch.minimum(src_labels, dst_labels)
            new_labels = labels_core.clone()
            new_labels.scatter_reduce_(0, adj_indices[0], min_labels, reduce="amin")
            new_labels.scatter_reduce_(0, adj_indices[1], min_labels, reduce="amin")
            if torch.equal(new_labels, labels_core):
                break
            labels_core = new_labels

        # --- Assign labels ---
        _, compact = torch.unique(labels_core, return_inverse=True)
        labels = torch.full((n_samples,), -1, dtype=torch.int64, device=device)
        labels[core_indices] = compact

        # Border points (reuse cached edges)
        border_pair = (~core_mask[row_idx]) & core_mask[col_idx]
        if border_pair.any():
            border_pts = row_idx[border_pair]
            core_nbrs = col_idx[border_pair]
            unlabeled = labels[border_pts] == -1
            labels[border_pts[unlabeled]] = labels[core_nbrs[unlabeled]]

        return labels.cpu().numpy(), core_indices.cpu().numpy()

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

        # GPU path: fully on-device computation
        if hasattr(X_arr, "is_cuda") and X_arr.is_cuda:
            # PyTorch CUDA: use fully GPU-based approach
            labels_np, core_indices = self._fit_gpu(backend, X_arr, n_samples, n_features)
        else:
            # CuPy or other backend: fall back to CPU label assignment
            labels_np, core_indices = self._fit_gpu_fallback(
                backend, X_arr, n_samples
            )

        self.labels_ = backend.asarray(labels_np, dtype=backend.int64)
        self.core_sample_indices_ = backend.asarray(core_indices, dtype=backend.int64)
        self.components_ = X_arr[core_indices] if core_indices.size else X_arr[:0]
        self.n_features_in_ = int(n_features)
        self._backend_name = backend.name
        self._fitted = True
        return self

    def _fit_gpu_fallback(self, backend, X_arr, n_samples):
        """GPU distance computation + CPU label assignment (for CuPy etc.)."""
        row_idx, col_idx = self._neighbor_graph_sparse(backend, X_arr)
        row_np = _to_numpy(row_idx).astype(np.int64)
        col_np = _to_numpy(col_idx).astype(np.int64)

        if _HAS_CY_FAST:
            from scipy.sparse import coo_matrix
            graph_coo = coo_matrix(
                (np.ones(len(row_np), dtype=np.int8), (row_np, col_np)),
                shape=(n_samples, n_samples),
            )
            csr = graph_coo.tocsr()
            return dbscan_labels_from_csr(
                n_samples, int(self.min_samples),
                csr.indptr.astype(np.int64),
                csr.indices.astype(np.int64),
            )

        counts = np.bincount(row_np, minlength=n_samples)
        core_mask_np = counts >= int(self.min_samples)
        core_indices = np.flatnonzero(core_mask_np).astype(np.int64)
        if not core_indices.size:
            return np.full(n_samples, -1, dtype=np.int64), core_indices

        core_edges = core_mask_np[row_np] & core_mask_np[col_np]
        core_row, core_col = row_np[core_edges], col_np[core_edges]
        n_core = len(core_indices)
        core_position = np.full(n_samples, -1, dtype=np.int64)
        core_position[core_indices] = np.arange(n_core, dtype=np.int64)
        graph = csr_matrix(
            (np.ones(len(core_row), dtype=bool),
             (core_position[core_row], core_position[core_col])),
            shape=(n_core, n_core),
        )
        _, core_labels = connected_components(graph, directed=False, return_labels=True)
        labels_np = np.full(n_samples, -1, dtype=np.int64)
        labels_np[core_indices] = core_labels.astype(np.int64)

        border_edges = (~core_mask_np[row_np]) & core_mask_np[col_np]
        if np.any(border_edges):
            border_rows = row_np[border_edges]
            border_labels = labels_np[col_np[border_edges]]
            order = np.argsort(border_rows, kind="mergesort")
            border_rows, border_labels = border_rows[order], border_labels[order]
            first = np.r_[True, border_rows[1:] != border_rows[:-1]]
            labels_np[border_rows[first]] = border_labels[first]

        return labels_np, core_indices

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
