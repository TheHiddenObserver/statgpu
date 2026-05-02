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
    """Exact dense Euclidean DBSCAN core for medium-size CPU inputs.

    This routine avoids materializing the pairwise distance matrix. It performs
    two pairwise scans: one to count eps-neighborhood sizes and another to union
    core-core edges and attach border points.
    """
    cdef Py_ssize_t n_samples = X.shape[0]
    cdef Py_ssize_t n_features = X.shape[1]
    cdef double eps_sq = eps * eps
    cdef Py_ssize_t i, j, f
    cdef double dist, diff
    cdef int64_t root
    cdef int64_t label

    cdef cnp.ndarray[int64_t, ndim=1] counts_arr = np.ones(n_samples, dtype=np.int64)
    cdef cnp.ndarray[int64_t, ndim=1] parent_arr = np.arange(n_samples, dtype=np.int64)
    cdef cnp.ndarray[int64_t, ndim=1] rank_arr = np.zeros(n_samples, dtype=np.int64)
    cdef cnp.ndarray[int64_t, ndim=1] labels_arr = np.full(n_samples, -1, dtype=np.int64)
    cdef cnp.ndarray[int64_t, ndim=1] root_label_arr = np.full(n_samples, -1, dtype=np.int64)
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] core_arr = np.zeros(n_samples, dtype=np.uint8)
    cdef cnp.ndarray[int64_t, ndim=1] core_indices_arr

    cdef int64_t[:] counts = counts_arr
    cdef int64_t[:] parent = parent_arr
    cdef int64_t[:] rank = rank_arr
    cdef int64_t[:] labels = labels_arr
    cdef int64_t[:] root_label = root_label_arr
    cdef cnp.uint8_t[:] core = core_arr

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            dist = 0.0
            for f in range(n_features):
                diff = X[i, f] - X[j, f]
                dist += diff * diff
                if dist > eps_sq:
                    break
            if dist <= eps_sq:
                counts[i] += 1
                counts[j] += 1

    for i in range(n_samples):
        if counts[i] >= min_samples:
            core[i] = 1

    for i in range(n_samples):
        if not core[i]:
            continue
        for j in range(i + 1, n_samples):
            if not core[j]:
                continue
            dist = 0.0
            for f in range(n_features):
                diff = X[i, f] - X[j, f]
                dist += diff * diff
                if dist > eps_sq:
                    break
            if dist <= eps_sq:
                _union(parent, rank, i, j)

    label = 0
    for i in range(n_samples):
        if core[i]:
            root = _find(parent, i)
            if root_label[root] < 0:
                root_label[root] = label
                label += 1
            labels[i] = root_label[root]

    for i in range(n_samples):
        if core[i]:
            continue
        for j in range(n_samples):
            if not core[j]:
                continue
            dist = 0.0
            for f in range(n_features):
                diff = X[i, f] - X[j, f]
                dist += diff * diff
                if dist > eps_sq:
                    break
            if dist <= eps_sq:
                labels[i] = labels[j]
                break

    core_indices_arr = np.flatnonzero(core_arr).astype(np.int64, copy=False)
    return labels_arr, core_indices_arr
