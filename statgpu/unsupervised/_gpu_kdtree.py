"""GPU KD-tree for fast range queries in DBSCAN.

Approach:
  - Build tree on CPU (sequential, fast)
  - Flatten tree into arrays (split_dim, split_val, left, right, point_idx)
  - Range query on GPU (one thread per query point, stack-based traversal)
"""
from __future__ import annotations

import numpy as np


class GPUTree:
    """KD-tree built on CPU, queried on GPU."""

    def __init__(self, X, leaf_size=32):
        self.n, self.d = X.shape
        self.leaf_size = leaf_size
        self.indices = np.arange(self.n)

        # Build tree recursively
        nodes = []
        self._build(X, 0, self.n, 0, nodes)

        # Flatten into arrays
        self.split_dim = np.array([n[0] for n in nodes], dtype=np.int32)
        self.split_val = np.array([n[1] for n in nodes], dtype=np.float64)
        self.left_child = np.array([n[2] for n in nodes], dtype=np.int32)
        self.right_child = np.array([n[3] for n in nodes], dtype=np.int32)
        self.node_start = np.array([n[4] for n in nodes], dtype=np.int32)
        self.node_end = np.array([n[5] for n in nodes], dtype=np.int32)
        self.n_nodes = len(nodes)

    def _build(self, X, start, end, depth, nodes):
        n_points = end - start
        if n_points <= 0:
            return -1

        node_idx = len(nodes)
        nodes.append([0, 0.0, -1, -1, 0, 0])  # placeholder

        if n_points <= self.leaf_size:
            nodes[node_idx] = [-1, 0.0, -1, -1, start, end]
            return node_idx

        # Find splitting dimension (max spread)
        split_dim = 0
        max_spread = 0.0
        for dim in range(self.d):
            vals = X[self.indices[start:end], dim]
            spread = float(vals.max() - vals.min())
            if spread > max_spread:
                max_spread = spread
                split_dim = dim

        # Find median
        mid = start + n_points // 2
        vals = X[self.indices[start:end], split_dim]
        order = np.argsort(vals)
        self.indices[start:end] = self.indices[start:end][order]
        split_val = float(X[self.indices[mid], split_dim])

        # Build children
        left = self._build(X, start, mid, depth + 1, nodes)
        right = self._build(X, mid + 1, end, depth + 1, nodes)

        nodes[node_idx] = [split_dim, split_val, left, right, start, end]
        return node_idx

    def query_radius(self, X_query, eps, backend):
        """Find all neighbors within eps for each query point.

        Uses GPU kernel for range query (one thread per query point).

        Parameters
        ----------
        X_query : (m, d) array
            Query points.
        eps : float
            Search radius.
        backend : str
            Backend to use ('cuda' or 'torch').

        Returns
        -------
        row_idx : int64 array
            Row indices of neighbor pairs.
        col_idx : int64 array
            Column indices of neighbor pairs.
        counts : int64 array
            Number of neighbors per query point.
        """
        m = X_query.shape[0]
        eps_sq = eps * eps

        # Transfer tree to GPU
        if backend == "torch":
            return self._query_torch(X_query, eps_sq, m)
        elif backend == "cupy":
            return self._query_cupy(X_query, eps_sq, m)
        else:
            return self._query_numpy(X_query, eps_sq, m)

    def _query_numpy(self, X_query, eps_sq, m):
        """NumPy range query: tree traversal + batch distance."""
        eps = np.sqrt(eps_sq)

        # Step 1: Find candidate leaf nodes for each query point
        candidates = [[] for _ in range(m)]
        for i in range(m):
            stack = [0]
            while stack:
                node_idx = stack.pop()
                if node_idx < 0:
                    continue
                split_dim = self.split_dim[node_idx]
                if split_dim == -1:
                    candidates[i].append(node_idx)
                else:
                    split_val = self.split_val[node_idx]
                    qval = X_query[i, split_dim]
                    if qval - eps <= split_val:
                        stack.append(self.left_child[node_idx])
                    if qval + eps > split_val:
                        stack.append(self.right_child[node_idx])

        # Step 2: Build batch of (query, candidate) pairs
        all_query_idx = []
        all_cand_idx = []
        for i, cands in enumerate(candidates):
            for node_idx in cands:
                for j in range(self.node_start[node_idx], self.node_end[node_idx]):
                    all_query_idx.append(i)
                    all_cand_idx.append(self.indices[j])

        if not all_query_idx:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.zeros(m, dtype=np.int64)

        query_idx = np.array(all_query_idx, dtype=np.int64)
        cand_idx = np.array(all_cand_idx, dtype=np.int64)

        # Step 3: Batch distance computation
        diff = X_query[query_idx] - X_query[cand_idx]
        dist_sq = np.sum(diff * diff, axis=1)
        mask = dist_sq <= eps_sq

        # Step 4: Extract valid pairs
        valid = np.nonzero(mask)[0]
        row_idx = query_idx[valid]
        col_idx = cand_idx[valid]
        counts = np.bincount(row_idx, minlength=m).astype(np.int64)

        return row_idx, col_idx, counts

    def _query_torch(self, X_query, eps_sq, m):
        """PyTorch GPU range query: CPU tree traversal + GPU leaf distance."""
        import torch

        eps = np.sqrt(eps_sq)

        # Step 1: CPU - find candidate leaf nodes for each query point
        candidates = [[] for _ in range(m)]
        for i in range(m):
            stack = [0]
            while stack:
                node_idx = stack.pop()
                if node_idx < 0:
                    continue
                split_dim = self.split_dim[node_idx]
                if split_dim == -1:
                    # Leaf node - add as candidate
                    candidates[i].append(node_idx)
                else:
                    split_val = self.split_val[node_idx]
                    qval = X_query[i, split_dim]
                    if qval - eps <= split_val:
                        stack.append(self.left_child[node_idx])
                    if qval + eps > split_val:
                        stack.append(self.right_child[node_idx])

        # Step 2: Build batch of (query, candidate) pairs
        all_query_idx = []
        all_cand_idx = []
        for i, cands in enumerate(candidates):
            for node_idx in cands:
                for j in range(self.node_start[node_idx], self.node_end[node_idx]):
                    all_query_idx.append(i)
                    all_cand_idx.append(self.indices[j])

        if not all_query_idx:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.zeros(m, dtype=np.int64)

        query_idx = torch.tensor(all_query_idx, device='cuda', dtype=torch.int64)
        cand_idx = torch.tensor(all_cand_idx, device='cuda', dtype=torch.int64)
        X_gpu = torch.tensor(X_query, device='cuda', dtype=torch.float64)

        # Step 3: GPU - compute distances for all candidate pairs
        diff = X_gpu[query_idx] - X_gpu[cand_idx]
        dist_sq = torch.sum(diff * diff, dim=1)
        mask = dist_sq <= eps_sq

        # Step 4: Extract valid pairs
        valid = torch.nonzero(mask, as_tuple=False).squeeze()
        if valid.numel() == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.zeros(m, dtype=np.int64)

        row_idx = query_idx[valid].cpu().numpy()
        col_idx = cand_idx[valid].cpu().numpy()
        counts = np.bincount(row_idx, minlength=m).astype(np.int64)

        return row_idx, col_idx, counts

    def _query_cupy(self, X_query, eps_sq, m):
        """CuPy GPU range query: CPU tree traversal + GPU leaf distance."""
        import cupy as cp

        eps = np.sqrt(eps_sq)

        # Step 1: CPU - find candidate leaf nodes for each query point
        candidates = [[] for _ in range(m)]
        for i in range(m):
            stack = [0]
            while stack:
                node_idx = stack.pop()
                if node_idx < 0:
                    continue
                split_dim = self.split_dim[node_idx]
                if split_dim == -1:
                    candidates[i].append(node_idx)
                else:
                    split_val = self.split_val[node_idx]
                    qval = X_query[i, split_dim]
                    if qval - eps <= split_val:
                        stack.append(self.left_child[node_idx])
                    if qval + eps > split_val:
                        stack.append(self.right_child[node_idx])

        # Step 2: Build batch of (query, candidate) pairs
        all_query_idx = []
        all_cand_idx = []
        for i, cands in enumerate(candidates):
            for node_idx in cands:
                for j in range(self.node_start[node_idx], self.node_end[node_idx]):
                    all_query_idx.append(i)
                    all_cand_idx.append(self.indices[j])

        if not all_query_idx:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.zeros(m, dtype=np.int64)

        query_idx = cp.asarray(all_query_idx, dtype=cp.int64)
        cand_idx = cp.asarray(all_cand_idx, dtype=cp.int64)
        X_gpu = cp.asarray(X_query, dtype=cp.float64)

        # Step 3: GPU - compute distances for all candidate pairs
        diff = X_gpu[query_idx] - X_gpu[cand_idx]
        dist_sq = cp.sum(diff * diff, axis=1)
        mask = dist_sq <= eps_sq

        # Step 4: Extract valid pairs
        valid = cp.argwhere(mask).squeeze()
        if valid.size == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.zeros(m, dtype=np.int64)

        row_idx = query_idx[valid].get()
        col_idx = cand_idx[valid].get()
        counts = np.bincount(row_idx, minlength=m).astype(np.int64)

        return row_idx, col_idx, counts


def dbscan_gpu_kdtree(X, eps, min_samples, backend="torch"):
    """DBSCAN using GPU KD-tree.

    Parameters
    ----------
    X : (n, d) array
        Input data.
    eps : float
        Neighborhood radius.
    min_samples : int
        Minimum samples for core point.
    backend : str
        Backend to use.

    Returns
    -------
    labels : (n,) int64 array
        Cluster labels (-1 for noise).
    core_indices : int64 array
        Indices of core points.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    n = X.shape[0]

    # Build KD-tree on CPU
    tree = GPUTree(X, leaf_size=32)

    # Query neighbors
    row_idx, col_idx, counts = tree.query_radius(X, eps, backend)

    # Find core points
    core_mask = counts >= min_samples
    core_indices = np.flatnonzero(core_mask).astype(np.int64)
    n_core = len(core_indices)

    if n_core == 0:
        return np.full(n, -1, dtype=np.int64), core_indices

    # Build core-core adjacency
    core_edges = core_mask[row_idx] & core_mask[col_idx]
    core_position = np.full(n, -1, dtype=np.int64)
    core_position[core_indices] = np.arange(n_core, dtype=np.int64)

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

    # Assign labels
    labels = np.full(n, -1, dtype=np.int64)
    labels[core_indices] = core_labels.astype(np.int64)

    # Border points
    border_mask = (~core_mask[row_idx]) & core_mask[col_idx]
    if np.any(border_mask):
        border_rows = row_idx[border_mask]
        border_labels = labels[col_idx[border_mask]]
        order = np.argsort(border_rows, kind="mergesort")
        border_rows = border_rows[order]
        border_labels = border_labels[order]
        first = np.r_[True, border_rows[1:] != border_rows[:-1]]
        labels[border_rows[first]] = border_labels[first]

    return labels, core_indices
