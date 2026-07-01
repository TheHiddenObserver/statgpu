"""Cython-optimized DBSCAN core routines.

Provides fast neighbor counting and edge building without Python object overhead.
"""
# cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport sqrt

np.import_array()

ctypedef np.float64_t DTYPE_t
ctypedef np.int64_t ITYPE_t
ctypedef np.int32_t ITYPE32_t


def dbscan_neighbor_graph(
    np.ndarray[DTYPE_t, ndim=2] X,
    double eps,
):
    """Build sparse neighbor graph using brute force with BLAS.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data.
    eps : float
        Neighborhood radius.

    Returns
    -------
    row_idx : ndarray of int64
        Row indices of neighbor pairs.
    col_idx : ndarray of int64
        Column indices of neighbor pairs.
    counts : ndarray of int64
        Number of neighbors per point.
    """
    cdef Py_ssize_t n = X.shape[0]
    cdef Py_ssize_t d = X.shape[1]
    cdef double eps_sq = eps * eps

    # Compute squared norms
    cdef np.ndarray[DTYPE_t, ndim=1] x_norm = np.sum(X * X, axis=1)

    # Output arrays (pre-allocate with estimated size)
    cdef Py_ssize_t est_pairs = n * 10  # initial estimate
    cdef np.ndarray[ITYPE_t, ndim=1] row_buf = np.empty(est_pairs, dtype=np.int64)
    cdef np.ndarray[ITYPE_t, ndim=1] col_buf = np.empty(est_pairs, dtype=np.int64)
    cdef np.ndarray[ITYPE_t, ndim=1] counts = np.zeros(n, dtype=np.int64)
    cdef Py_ssize_t pair_count = 0

    cdef Py_ssize_t i, j, k
    cdef double dist_sq, diff

    # Brute force with numpy matmul for the heavy part
    cdef Py_ssize_t batch_size = min(n, max(1000, 200000000 // (n * 8)))
    cdef Py_ssize_t start, stop

    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        # Distance matrix for this batch: (batch_size, n)
        X_batch = X[start:stop]
        dist = x_norm[start:stop].reshape(-1, 1) + x_norm.reshape(1, -1) - 2.0 * X_batch.dot(X.T)
        dist = np.maximum(dist, 0.0)

        # Find neighbors
        mask = dist <= eps_sq
        rows, cols = np.nonzero(mask)

        # Grow buffer if needed
        new_count = pair_count + len(rows)
        if new_count > row_buf.shape[0]:
            new_size = max(new_count, row_buf.shape[0] * 2)
            row_buf = np.resize(row_buf, new_size)
            col_buf = np.resize(col_buf, new_size)

        # Copy to buffer
        row_buf[pair_count:new_count] = rows + start
        col_buf[pair_count:new_count] = cols
        pair_count = new_count

        # Count neighbors per row
        for i in range(start, stop):
            counts[i] = <ITYPE_t>np.sum(mask[i - start])

    # Trim to actual size
    row_idx = row_buf[:pair_count].copy()
    col_idx = col_buf[:pair_count].copy()

    return row_idx, col_idx, counts


def dbscan_count_neighbors_kdtree(
    object tree,
    np.ndarray[DTYPE_t, ndim=2] X,
    double eps,
    int workers,
):
    """Count neighbors using scipy cKDTree (fast counting mode).

    This avoids creating Python lists for each point.
    """
    cdef Py_ssize_t n = X.shape[0]
    cdef np.ndarray[ITYPE_t, ndim=1] counts = np.zeros(n, dtype=np.int64)

    # Use query_ball_point with return_length=True (fast, no list creation)
    raw_counts = tree.query_ball_point(X, r=eps, workers=workers, return_length=True)
    cdef Py_ssize_t i
    for i in range(n):
        counts[i] = <ITYPE_t>int(raw_counts[i])

    return counts


def dbscan_build_edges(
    object tree,
    np.ndarray[DTYPE_t, ndim=2] X_core,
    np.ndarray[ITYPE_t, ndim=1] core_indices,
    double eps,
    int workers,
):
    """Build edge lists from core point neighbors.

    Only queries neighbors for core points (much fewer than all points).
    """
    cdef Py_ssize_t n_core = X_core.shape[0]
    neighbor_lists = tree.query_ball_point(X_core, r=eps, workers=workers)

    # Build edge lists
    cdef list all_rows = []
    cdef list all_cols = []
    cdef Py_ssize_t i, core_idx
    cdef list neighbors

    for i in range(n_core):
        core_idx = core_indices[i]
        neighbors = neighbor_lists[i]
        if len(neighbors) > 0:
            all_rows.append(np.full(len(neighbors), core_idx, dtype=np.int64))
            all_cols.append(np.array(neighbors, dtype=np.int64))

    if len(all_rows) > 0:
        row_idx = np.concatenate(all_rows)
        col_idx = np.concatenate(all_cols)
    else:
        row_idx = np.empty(0, dtype=np.int64)
        col_idx = np.empty(0, dtype=np.int64)

    return row_idx, col_idx
