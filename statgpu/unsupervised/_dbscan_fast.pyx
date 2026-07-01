"""Complete Cython DBSCAN implementation.

Avoids all Python object overhead by using typed memoryviews and
C-level operations for the critical path.
"""
# cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from libc.string cimport memset

np.import_array()

ctypedef np.float64_t DTYPE_t
ctypedef np.int64_t ITYPE_t
ctypedef np.uint8_t BOOL_t


def dbscan_fast(np.ndarray[DTYPE_t, ndim=2] X, double eps, int min_samples):
    """Fast DBSCAN using scipy cKDTree + Cython graph operations.

    Parameters
    ----------
    X : (n, d) array
        Input data.
    eps : float
        Neighborhood radius.
    min_samples : int
        Minimum samples for core point.

    Returns
    -------
    labels : (n,) int64 array
        Cluster labels (-1 for noise).
    core_indices : int64 array
        Indices of core points.
    """
    from scipy.spatial import cKDTree

    cdef int n = X.shape[0]
    cdef int d = X.shape[1]
    cdef int workers = -1  # Use all cores

    # Step 1: Build tree and count neighbors (fast, no Python list creation)
    tree = cKDTree(X)
    raw_counts = tree.query_ball_point(X, r=eps, workers=workers, return_length=True)
    cdef np.ndarray[ITYPE_t, ndim=1] counts = np.asarray(raw_counts, dtype=np.int64)

    # Step 2: Find core points
    cdef np.ndarray[BOOL_t, ndim=1] core_mask = counts >= min_samples
    cdef np.ndarray[ITYPE_t, ndim=1] core_indices = np.flatnonzero(core_mask).astype(np.int64)
    cdef int n_core = len(core_indices)

    if n_core == 0:
        return np.full(n, -1, dtype=np.int64), core_indices

    # Step 3: Query neighbors for core points only (much fewer lists)
    X_core = X[core_indices]
    core_neighbors = [list(x) for x in tree.query_ball_point(X_core, r=eps, workers=workers)]

    # Step 4: Build sparse adjacency using Cython (avoid Python list overhead)
    cdef np.ndarray[ITYPE_t, ndim=1] row_idx
    cdef np.ndarray[ITYPE_t, ndim=1] col_idx
    row_idx, col_idx = _build_adjacency_cy(core_neighbors, core_indices, n_core)

    # Step 5: Connected components (scipy - already optimized)
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    if len(row_idx) > 0:
        core_pos = np.full(n, -1, dtype=np.int64)
        core_pos[core_indices] = np.arange(n_core, dtype=np.int64)
        core_row = core_pos[row_idx]
        core_col = core_pos[col_idx]
        valid = (core_row >= 0) & (core_col >= 0)
        graph = csr_matrix(
            (np.ones(np.sum(valid), dtype=bool), (core_row[valid], core_col[valid])),
            shape=(n_core, n_core),
        )
        _, core_labels = connected_components(graph, directed=False, return_labels=True)
    else:
        core_labels = np.zeros(n_core, dtype=np.int32)

    # Step 6: Assign labels using Cython
    cdef np.ndarray[ITYPE_t, ndim=1] labels = _assign_labels_cy(
        n, core_indices, core_labels, core_mask, row_idx, col_idx
    )

    return labels, core_indices


cdef _build_adjacency_cy(list core_neighbors, np.ndarray[ITYPE_t, ndim=1] core_indices, int n_core):
    """Build adjacency lists from core point neighbors (Cython-optimized)."""
    cdef int i, core_idx, j
    cdef list neighbors
    cdef int total = 0

    # Count total edges
    for i in range(n_core):
        total += len(core_neighbors[i])

    # Pre-allocate arrays
    cdef np.ndarray[ITYPE_t, ndim=1] row_idx = np.empty(total, dtype=np.int64)
    cdef np.ndarray[ITYPE_t, ndim=1] col_idx = np.empty(total, dtype=np.int64)
    cdef int pos = 0

    for i in range(n_core):
        core_idx = core_indices[i]
        neighbors = core_neighbors[i]
        for j in range(len(neighbors)):
            row_idx[pos] = core_idx
            col_idx[pos] = neighbors[j]
            pos += 1

    return row_idx, col_idx


cdef np.ndarray[ITYPE_t, ndim=1] _assign_labels_cy(
    int n,
    np.ndarray[ITYPE_t, ndim=1] core_indices,
    core_labels,
    np.ndarray[BOOL_t, ndim=1] core_mask,
    np.ndarray[ITYPE_t, ndim=1] row_idx,
    np.ndarray[ITYPE_t, ndim=1] col_idx,
):
    """Assign labels to all points (Cython-optimized)."""
    cdef int n_core = len(core_indices)
    cdef np.ndarray[ITYPE_t, ndim=1] labels = np.full(n, -1, dtype=np.int64)
    cdef int i, j, idx

    # Assign core labels
    for i in range(n_core):
        labels[core_indices[i]] = int(core_labels[i])

    # Assign border points
    cdef int n_edges = len(row_idx)
    for i in range(n_edges):
        if not core_mask[row_idx[i]] and core_mask[col_idx[i]]:
            # row is border, col is core
            if labels[row_idx[i]] == -1:
                labels[row_idx[i]] = labels[col_idx[i]]

    return labels


def dbscan_fast_bruteforce(np.ndarray[DTYPE_t, ndim=2] X, double eps, int min_samples):
    """Fast DBSCAN using brute force distance computation + Cython.

    For high-dimensional data where KD-tree is not efficient.
    """
    cdef int n = X.shape[0]
    cdef int d = X.shape[1]
    cdef double eps_sq = eps * eps

    # Step 1: Compute distance matrix in batches (BLAS-optimized)
    cdef int batch_size = min(n, max(1000, 200000000 // (n * 8)))
    cdef np.ndarray[DTYPE_t, ndim=1] x_norm = np.sum(X * X, axis=1)

    # Pre-allocate edge lists
    cdef int est_edges = n * 10
    cdef np.ndarray[ITYPE_t, ndim=1] row_buf = np.empty(est_edges, dtype=np.int64)
    cdef np.ndarray[ITYPE_t, ndim=1] col_buf = np.empty(est_edges, dtype=np.int64)
    cdef np.ndarray[ITYPE_t, ndim=1] counts = np.zeros(n, dtype=np.int64)
    cdef int edge_count = 0

    cdef int start, stop, i, j
    cdef double dist_sq

    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        # Distance computation using BLAS
        X_batch = X[start:stop]
        dist = x_norm[start:stop].reshape(-1, 1) + x_norm.reshape(1, -1) - 2.0 * X_batch.dot(X.T)
        dist = np.maximum(dist, 0.0)
        mask = dist <= eps_sq

        # Find neighbors
        rows, cols = np.nonzero(mask)

        # Grow buffer if needed
        new_count = edge_count + len(rows)
        if new_count > row_buf.shape[0]:
            new_size = max(new_count, row_buf.shape[0] * 2)
            row_buf = np.resize(row_buf, new_size)
            col_buf = np.resize(col_buf, new_size)

        row_buf[edge_count:new_count] = rows + start
        col_buf[edge_count:new_count] = cols
        edge_count = new_count

        # Count neighbors
        for i in range(start, stop):
            counts[i] = <ITYPE_t>np.sum(mask[i - start])

    # Trim buffers
    row_idx = row_buf[:edge_count].copy()
    col_idx = col_buf[:edge_count].copy()

    # Step 2: Find core points
    cdef np.ndarray[BOOL_t, ndim=1] core_mask = counts >= min_samples
    cdef np.ndarray[ITYPE_t, ndim=1] core_indices = np.flatnonzero(core_mask).astype(np.int64)
    cdef int n_core = len(core_indices)

    if n_core == 0:
        return np.full(n, -1, dtype=np.int64), core_indices

    # Step 3: Connected components
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    core_pos = np.full(n, -1, dtype=np.int64)
    core_pos[core_indices] = np.arange(n_core, dtype=np.int64)
    core_edges = core_mask[row_idx] & core_mask[col_idx]
    core_row = core_pos[row_idx[core_edges]]
    core_col = core_pos[col_idx[core_edges]]

    if len(core_row) > 0:
        graph = csr_matrix(
            (np.ones(len(core_row), dtype=bool), (core_row, core_col)),
            shape=(n_core, n_core),
        )
        _, core_labels = connected_components(graph, directed=False, return_labels=True)
    else:
        core_labels = np.zeros(n_core, dtype=np.int32)

    # Step 4: Assign labels
    cdef np.ndarray[ITYPE_t, ndim=1] labels = _assign_labels_cy(
        n, core_indices, core_labels, core_mask, row_idx, col_idx
    )

    return labels, core_indices
