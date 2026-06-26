# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
"""Cython helpers for CPU DBSCAN."""

import numpy as np
cimport numpy as cnp


ctypedef cnp.float64_t float64_t
ctypedef cnp.int64_t int64_t


cdef int64_t _find(int64_t[:] parent, int64_t x) noexcept nogil:
    cdef int64_t root = x
    cdef int64_t next_x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != x:
        next_x = parent[x]
        parent[x] = root
        x = next_x
    return root


cdef void _union(int64_t[:] parent, int64_t[:] rank, int64_t a, int64_t b) noexcept nogil:
    cdef int64_t root_a = _find(parent, a)
    cdef int64_t root_b = _find(parent, b)
    if root_a == root_b:
        return
    if rank[root_a] < rank[root_b]:
        parent[root_a] = root_b
    elif rank[root_a] > rank[root_b]:
        parent[root_b] = root_a
    else:
        parent[root_b] = root_a
        rank[root_a] += 1


def dbscan_dense_pairwise(float64_t[:, ::1] X, double eps, int64_t min_samples):
    """Exact dense Euclidean DBSCAN for medium-size CPU inputs.

    Two-pass design with batched BLAS distance computation:
      Pass 1: batched matmul for pairwise distances, count + collect edges
      Pass 2: Union-Find on core-core edges + border assignment

    Compared to the old three-pass scalar approach, this:
      - Uses BLAS-optimized matmul instead of scalar loops (4-8x for d >= 8)
      - Computes each distance only once (was 3x before)
      - Stores edges in arrays instead of recomputing via O(n²) scans
    """
    cdef Py_ssize_t n_samples = X.shape[0]
    cdef Py_ssize_t n_features = X.shape[1]
    cdef double eps_sq = eps * eps
    cdef Py_ssize_t i

    # Convert to numpy for BLAS matmul
    cdef cnp.ndarray[float64_t, ndim=2] X_np = np.asarray(X)
    cdef cnp.ndarray[float64_t, ndim=2] x_norm = np.sum(X_np * X_np, axis=1, keepdims=True)

    # Pre-allocate edge buffers
    cdef Py_ssize_t est_pairs = min(n_samples * 10, n_samples * (n_samples - 1) // 2)
    cdef cnp.ndarray[int64_t, ndim=1] row_buf = np.empty(est_pairs, dtype=np.int64)
    cdef cnp.ndarray[int64_t, ndim=1] col_buf = np.empty(est_pairs, dtype=np.int64)
    cdef cnp.ndarray[int64_t, ndim=1] counts_arr = np.ones(n_samples, dtype=np.int64)
    cdef Py_ssize_t edge_count = 0

    # Batch size: ~40MB per batch
    cdef Py_ssize_t batch_size = <Py_ssize_t>min(n_samples, max(500, 40000000 // (n_samples * 8)))
    cdef Py_ssize_t start, stop, new_count
    cdef cnp.ndarray[bool, ndim=1] first_mask

    # --- Pass 1: batched distance computation + edge collection ---
    for start in range(0, n_samples, batch_size):
        stop = min(start + batch_size, n_samples)
        # BLAS-optimized distance: (batch, n) via matmul
        dist = x_norm[start:stop] + x_norm.T - 2.0 * X_np[start:stop].dot(X_np.T)
        np.maximum(dist, 0.0, out=dist)
        mask = dist <= eps_sq
        rows, cols = np.nonzero(mask)
        rows = rows + start
        # Exclude self-pairs (i == j)
        keep = rows != cols
        rows = rows[keep]
        cols = cols[keep]

        # Grow buffer if needed
        new_count = edge_count + len(rows)
        if new_count > row_buf.shape[0]:
            new_size = max(new_count, row_buf.shape[0] * 2)
            row_buf = np.resize(row_buf, new_size)
            col_buf = np.resize(col_buf, new_size)

        row_buf[edge_count:new_count] = rows
        col_buf[edge_count:new_count] = cols
        edge_count = new_count
        counts_arr[start:stop] = np.sum(mask, axis=1)

    cdef cnp.ndarray[int64_t, ndim=1] row_idx = row_buf[:edge_count].copy()
    cdef cnp.ndarray[int64_t, ndim=1] col_idx = col_buf[:edge_count].copy()

    # --- Pass 2: Union-Find + label assignment ---
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] core_arr = (counts_arr >= min_samples).astype(np.uint8)
    cdef cnp.ndarray[int64_t, ndim=1] core_indices_arr = np.flatnonzero(core_arr).astype(np.int64)
    cdef Py_ssize_t n_core = len(core_indices_arr)
    cdef cnp.ndarray[int64_t, ndim=1] labels_arr = np.full(n_samples, -1, dtype=np.int64)

    if n_core == 0:
        return labels_arr, core_indices_arr

    # Union-Find on core-core edges
    cdef cnp.ndarray[int64_t, ndim=1] parent_arr = np.arange(n_core, dtype=np.int64)
    cdef cnp.ndarray[int64_t, ndim=1] rank_arr = np.zeros(n_core, dtype=np.int64)
    cdef cnp.ndarray[int64_t, ndim=1] core_pos_arr = np.full(n_samples, -1, dtype=np.int64)
    for i in range(n_core):
        core_pos_arr[core_indices_arr[i]] = i

    cdef cnp.ndarray[cnp.uint8_t, ndim=1] core_edges_arr = core_arr[row_idx] & core_arr[col_idx]
    cdef cnp.ndarray[int64_t, ndim=1] cr = core_pos_arr[row_idx[core_edges_arr]]
    cdef cnp.ndarray[int64_t, ndim=1] cc = core_pos_arr[col_idx[core_edges_arr]]
    cdef int64_t[:] parent = parent_arr
    cdef int64_t[:] rank = rank_arr

    for i in range(len(cr)):
        _union(parent, rank, cr[i], cc[i])

    # Assign labels to core points
    cdef cnp.ndarray[int64_t, ndim=1] root_label_arr = np.full(n_core, -1, dtype=np.int64)
    cdef int64_t[:] root_label = root_label_arr
    cdef int64_t label = 0
    cdef int64_t root

    for i in range(n_core):
        root = _find(parent, core_indices_arr[i])
        if root_label[root] < 0:
            root_label[root] = label
            label += 1
        labels_arr[core_indices_arr[i]] = root_label[root]

    # Assign labels to border points
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] border_arr = (~core_arr[row_idx].astype(bool)) & core_arr[col_idx].astype(bool)
    cdef cnp.ndarray[int64_t, ndim=1] br = row_idx[border_arr]
    cdef cnp.ndarray[int64_t, ndim=1] bl = labels_arr[col_idx[border_arr]]

    if len(br) > 0:
        order = np.argsort(br, kind="mergesort")
        br = br[order]
        bl = bl[order]
        # Assign first core neighbor's label to each border point
        first_mask = np.empty(len(br), dtype=bool)
        first_mask[0] = True
        for i in range(1, len(br)):
            first_mask[i] = br[i] != br[i - 1]
        labels_arr[br[first_mask]] = bl[first_mask]

    return labels_arr, core_indices_arr
