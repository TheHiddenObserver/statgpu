"""Cython KD-tree for fast range queries in DBSCAN.

Simplified implementation using numpy arrays.
"""
# cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport sqrt

np.import_array()


def dbscan_kdtree_query(X, double eps, int leaf_size=32):
    """Build KD-tree and query all neighbors within eps.

    Uses a balanced KD-tree with median-of-three splitting.
    Queries use stack-based iteration (no recursion).

    Parameters
    ----------
    X : (n, d) array
        Input data.
    eps : float
        Search radius.
    leaf_size : int
        Max points per leaf node.

    Returns
    -------
    counts : (n,) int64 array
        Number of neighbors per point.
    row_idx : int64 array
        Row indices of neighbor pairs.
    col_idx : int64 array
        Column indices of neighbor pairs.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    cdef int n = X.shape[0]
    cdef int d = X.shape[1]
    cdef double eps_sq = eps * eps

    # Build tree structure using numpy arrays
    # Each node: [split_dim, split_val, left_child, right_child, start, end]
    cdef list nodes = []
    cdef list indices = list(range(n))

    # Recursive build
    def build(int start, int end, int depth):
        cdef int node_idx = len(nodes)
        nodes.append([0, 0.0, -1, -1, start, end])  # placeholder

        cdef int n_points = end - start
        if n_points <= leaf_size:
            nodes[node_idx] = [-1, 0.0, -1, -1, start, end]
            return node_idx

        # Find splitting dimension (max spread)
        cdef int split_dim = 0
        cdef double max_spread = 0.0
        cdef double min_val, max_val, spread
        cdef int dim, i

        for dim in range(d):
            min_val = X[indices[start], dim]
            max_val = min_val
            for i in range(start + 1, end):
                if X[indices[i], dim] < min_val:
                    min_val = X[indices[i], dim]
                if X[indices[i], dim] > max_val:
                    max_val = X[indices[i], dim]
            spread = max_val - min_val
            if spread > max_spread:
                max_spread = spread
                split_dim = dim

        # Find median using numpy argsort
        cdef int mid = start + n_points // 2
        vals = np.array([X[indices[j], split_dim] for j in range(start, end)])
        order = np.argsort(vals)
        sorted_indices = [indices[start + j] for j in order]
        for j in range(n_points):
            indices[start + j] = sorted_indices[j]

        split_val = X[indices[mid], split_dim]

        # Build children
        left_idx = build(start, mid, depth + 1)
        right_idx = build(mid + 1, end, depth + 1)

        nodes[node_idx] = [split_dim, split_val, left_idx, right_idx, start, end]
        return node_idx

    root = build(0, n, 0)

    # Query all points
    counts = np.zeros(n, dtype=np.int64)
    cdef list all_rows = []
    cdef list all_cols = []

    cdef int i, node_idx, j, k
    cdef double dist_sq, diff
    cdef int split_dim
    cdef double split_val
    cdef int left, right, start, end

    for i in range(n):
        # Stack-based iterative query
        stack = [root]
        while stack:
            node_idx = stack.pop()
            node = nodes[node_idx]
            split_dim = node[0]
            split_val = node[1]
            left = node[2]
            right = node[3]
            start = node[4]
            end = node[5]

            if split_dim == -1:
                # Leaf: check all points
                for j in range(start, end):
                    dist_sq = 0.0
                    for k in range(d):
                        diff = X[i, k] - X[indices[j], k]
                        dist_sq += diff * diff
                    if dist_sq <= eps_sq:
                        all_rows.append(i)
                        all_cols.append(indices[j])
                        counts[i] += 1
            else:
                # Internal: check which children to visit
                diff = X[i, split_dim] - split_val
                if diff <= 0:
                    if right >= 0:
                        stack.append(right)
                    stack.append(left)
                else:
                    if left >= 0:
                        stack.append(left)
                    stack.append(right)

    if len(all_rows) > 0:
        row_idx = np.array(all_rows, dtype=np.int64)
        col_idx = np.array(all_cols, dtype=np.int64)
    else:
        row_idx = np.empty(0, dtype=np.int64)
        col_idx = np.empty(0, dtype=np.int64)

    return counts, row_idx, col_idx
