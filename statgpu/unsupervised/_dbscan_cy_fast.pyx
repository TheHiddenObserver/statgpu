# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Cython-accelerated DBSCAN graph processing.

Handles the critical path in pure C:
  - Edge filtering (core-core, core-border)
  - Union-Find connected components
  - Border label assignment
"""
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.string cimport memset

np.import_array()

ctypedef np.int64_t INT64
ctypedef np.intp_t INTP
ctypedef np.uint8_t UINT8


# ---------- Union-Find (C-level, no Python overhead) ----------

cdef struct UF:
    INTP* parent
    INTP* rank
    int n


cdef inline void uf_init(UF* uf, int n) noexcept nogil:
    cdef int i
    uf.n = n
    uf.parent = <INTP*>malloc(n * sizeof(INTP))
    uf.rank = <INTP*>malloc(n * sizeof(INTP))
    for i in range(n):
        uf.parent[i] = i
        uf.rank[i] = 0


cdef inline void uf_free(UF* uf) noexcept nogil:
    free(uf.parent)
    free(uf.rank)


cdef inline INTP uf_find(UF* uf, INTP x) noexcept nogil:
    cdef INTP root = x
    cdef INTP next_x
    while uf.parent[root] != root:
        root = uf.parent[root]
    while uf.parent[x] != x:
        next_x = uf.parent[x]
        uf.parent[x] = root
        x = next_x
    return root


cdef inline void uf_union(UF* uf, INTP a, INTP b) noexcept nogil:
    cdef INTP ra = uf_find(uf, a)
    cdef INTP rb = uf_find(uf, b)
    if ra == rb:
        return
    if uf.rank[ra] < uf.rank[rb]:
        uf.parent[ra] = rb
    elif uf.rank[ra] > uf.rank[rb]:
        uf.parent[rb] = ra
    else:
        uf.parent[rb] = ra
        uf.rank[ra] += 1


# ---------- Main entry point ----------

def dbscan_labels_from_edges(
    INT64 n_samples,
    INT64 min_samples,
    list core_neighbors,   # list of numpy arrays (neighbor indices per core point)
    INT64[:] core_indices,  # original indices of core points
):
    """Build graph from core neighbor lists, run Union-Find, assign labels.

    All critical loops run in C — no Python overhead for edge processing.
    Returns (labels, core_indices).
    """
    cdef INT64 n_core = len(core_indices)
    cdef INT64 i, j, ci, k, nb_val
    cdef INT64 total_edges = 0
    cdef INT64[:] counts_arr
    cdef INT64[:] core_pos_arr
    cdef np.ndarray[INT64, ndim=1] labels_arr
    cdef INTP[:] core_row, core_col
    cdef UF uf
    cdef INT64 label, root
    cdef INTP* root_label  # maps UF root → cluster label

    # Count total edges
    for i in range(n_core):
        total_edges += <INT64>len(core_neighbors[i])

    # Build edge arrays + count neighbors in one pass (C-level loop)
    cdef np.ndarray[INT64, ndim=1] row_arr = np.empty(total_edges, dtype=np.int64)
    cdef np.ndarray[INT64, ndim=1] col_arr = np.empty(total_edges, dtype=np.int64)
    cdef INT64[:] row_view = row_arr
    cdef INT64[:] col_view = col_arr
    cdef INT64 pos = 0

    # Also count neighbors per point (for border detection)
    cdef np.ndarray[INT64, ndim=1] counts_all = np.zeros(n_samples, dtype=np.int64)
    cdef INT64[:] counts_view = counts_all

    for i in range(n_core):
        ci = core_indices[i]
        nb_arr = np.asarray(core_neighbors[i], dtype=np.int64)
        k = <INT64>nb_arr.shape[0]
        for j in range(k):
            nb_val = nb_arr[j]
            row_view[pos] = ci
            col_view[pos] = nb_val
            counts_view[nb_val] += 1
            pos += 1
        counts_view[ci] += 1  # self-count (query_ball_point includes self)

    # Map core points to positions 0..n_core-1
    core_pos_arr_np = np.full(n_samples, -1, dtype=np.intp)
    cdef INTP[:] core_pos_view = core_pos_arr_np
    for i in range(n_core):
        core_pos_view[core_indices[i]] = i

    # Filter core-core edges + run Union-Find (all in C)
    uf_init(&uf, <int>n_core)

    cdef INT64 n_core_edges = 0
    for i in range(total_edges):
        ci = row_view[i]
        nb_val = col_view[i]
        # Both endpoints must be core
        if core_pos_view[ci] >= 0 and core_pos_view[nb_val] >= 0:
            uf_union(&uf, core_pos_view[ci], core_pos_view[nb_val])
            n_core_edges += 1

    # Assign labels to core points
    root_label = <INTP*>malloc(n_core * sizeof(INTP))
    for i in range(n_core):
        root_label[i] = -1

    labels_arr = np.full(n_samples, -1, dtype=np.int64)
    cdef INT64[:] labels_view = labels_arr
    label = 0

    for i in range(n_core):
        root = uf_find(&uf, i)
        if root_label[root] < 0:
            root_label[root] = label
            label += 1
        labels_view[core_indices[i]] = root_label[root]

    # Assign labels to border points (non-core with core neighbors)
    for i in range(total_edges):
        ci = row_view[i]       # core point (since edges are from core_neighbors)
        nb_val = col_view[i]   # neighbor (could be core or non-core)
        if core_pos_view[nb_val] < 0 and labels_view[nb_val] < 0:
            # nb_val is non-core (border) and not yet labeled
            labels_view[nb_val] = labels_view[ci]

    # Cleanup
    free(root_label)
    uf_free(&uf)

    return labels_arr, np.asarray(core_indices)


def dbscan_labels_from_pairs(
    INT64 n_samples,
    INT64 min_samples,
    np.ndarray[INT64, ndim=2] pairs,  # (n_pairs, 2) with pairs[i,0] < pairs[i,1]
):
    """Complete DBSCAN from query_pairs output — all processing in C.

    Takes raw (i,j) pairs where i<j, counts neighbors, finds core points,
    runs Union-Find, and assigns labels. No Python loops for edge processing.

    Returns (labels, core_indices).
    """
    cdef INT64 n_pairs = pairs.shape[0]
    cdef INT64 i, a, b
    cdef INT64[:] p0 = pairs[:, 0]
    cdef INT64[:] p1 = pairs[:, 1]

    # Count neighbors: each pair (i,j) contributes +1 to both i and j
    cdef np.ndarray[INT64, ndim=1] counts_arr = np.zeros(n_samples, dtype=np.int64)
    cdef INT64[:] counts = counts_arr
    for i in range(n_pairs):
        counts[p0[i]] += 1
        counts[p1[i]] += 1

    # Find core points
    cdef INT64 n_core = 0
    cdef np.ndarray[np.uint8_t, ndim=1] core_mask_arr = np.zeros(n_samples, dtype=np.uint8)
    cdef np.uint8_t[:] core_mask = core_mask_arr
    for i in range(n_samples):
        if counts[i] >= min_samples - 1:
            core_mask[i] = 1
            n_core += 1

    cdef np.ndarray[INT64, ndim=1] core_indices_arr = np.empty(n_core, dtype=np.int64)
    cdef INT64[:] core_indices = core_indices_arr
    cdef INT64 ci = 0
    for i in range(n_samples):
        if core_mask[i]:
            core_indices[ci] = i
            ci += 1

    cdef np.ndarray[INT64, ndim=1] labels_arr = np.full(n_samples, -1, dtype=np.int64)
    cdef INT64[:] labels = labels_arr

    if n_core == 0:
        return labels_arr, core_indices_arr

    # Map points to core positions
    cdef np.ndarray[INTP, ndim=1] core_pos_arr = np.full(n_samples, -1, dtype=np.intp)
    cdef INTP[:] core_pos = core_pos_arr
    for i in range(n_core):
        core_pos[core_indices[i]] = i

    # Union-Find on core-core pairs
    cdef UF uf
    uf_init(&uf, <int>n_core)

    for i in range(n_pairs):
        a = p0[i]
        b = p1[i]
        if core_mask[a] and core_mask[b]:
            uf_union(&uf, core_pos[a], core_pos[b])

    # Assign labels to core points
    cdef INTP* root_label = <INTP*>malloc(n_core * sizeof(INTP))
    for i in range(n_core):
        root_label[i] = -1

    cdef INT64 label = 0
    cdef INTP root

    for i in range(n_core):
        root = uf_find(&uf, i)
        if root_label[root] < 0:
            root_label[root] = label
            label += 1
        labels[core_indices[i]] = root_label[root]

    # Assign labels to border points
    for i in range(n_pairs):
        a = p0[i]
        b = p1[i]
        # Border: non-core point gets label of its core neighbor
        if core_mask[a] and not core_mask[b] and labels[b] < 0:
            labels[b] = labels[a]
        elif core_mask[b] and not core_mask[a] and labels[a] < 0:
            labels[a] = labels[b]

    free(root_label)
    uf_free(&uf)

    return labels_arr, core_indices_arr


def dbscan_labels_from_csr(
    INT64 n_samples,
    INT64 min_samples,
    np.ndarray[INT64, ndim=1] indptr,   # CSR row pointers
    np.ndarray[INT64, ndim=1] indices,  # CSR column indices
):
    """Complete DBSCAN from CSR sparse neighbor graph — all processing in C.

    Takes the raw CSR arrays from a sparse neighbor graph (e.g. from
    sklearn's radius_neighbors_graph). Runs counting, Union-Find, and
    label assignment entirely in C.

    Returns (labels, core_indices).
    """
    cdef INT64 i, j, a, b, start, end
    cdef INT64* indptr_p = <INT64*>np.PyArray_DATA(indptr)
    cdef INT64* indices_p = <INT64*>np.PyArray_DATA(indices)

    # Count neighbors from CSR structure
    # Note: radius_neighbors includes self (distance=0), so counts already include self
    cdef np.ndarray[INT64, ndim=1] counts_arr = np.zeros(n_samples, dtype=np.int64)
    cdef INT64[:] counts = counts_arr
    for i in range(n_samples):
        counts[i] = indptr_p[i + 1] - indptr_p[i]

    # Find core points (counts already include self, use min_samples directly)
    cdef INT64 n_core = 0
    cdef np.ndarray[np.uint8_t, ndim=1] core_mask_arr = np.zeros(n_samples, dtype=np.uint8)
    cdef np.uint8_t[:] core_mask = core_mask_arr
    for i in range(n_samples):
        if counts[i] >= min_samples:
            core_mask[i] = 1
            n_core += 1

    cdef np.ndarray[INT64, ndim=1] core_indices_arr = np.empty(n_core, dtype=np.int64)
    cdef INT64[:] core_indices = core_indices_arr
    cdef INT64 ci = 0
    for i in range(n_samples):
        if core_mask[i]:
            core_indices[ci] = i
            ci += 1

    cdef np.ndarray[INT64, ndim=1] labels_arr = np.full(n_samples, -1, dtype=np.int64)
    cdef INT64[:] labels = labels_arr

    if n_core == 0:
        return labels_arr, core_indices_arr

    # Map points to core positions
    cdef np.ndarray[INTP, ndim=1] core_pos_arr = np.full(n_samples, -1, dtype=np.intp)
    cdef INTP[:] core_pos = core_pos_arr
    for i in range(n_core):
        core_pos[core_indices[i]] = i

    # Union-Find on core-core edges from CSR
    cdef UF uf
    uf_init(&uf, <int>n_core)

    for i in range(n_samples):
        if not core_mask[i]:
            continue
        start = indptr_p[i]
        end = indptr_p[i + 1]
        for j in range(start, end):
            b = indices_p[j]
            if core_mask[b] and b > i:  # each edge once (upper triangle)
                uf_union(&uf, core_pos[i], core_pos[b])

    # Assign labels to core points
    cdef INTP* root_label = <INTP*>malloc(n_core * sizeof(INTP))
    for i in range(n_core):
        root_label[i] = -1

    cdef INT64 label = 0
    cdef INTP root
    for i in range(n_core):
        root = uf_find(&uf, i)
        if root_label[root] < 0:
            root_label[root] = label
            label += 1
        labels[core_indices[i]] = root_label[root]

    # Assign labels to border points
    for i in range(n_samples):
        if core_mask[i]:
            continue
        start = indptr_p[i]
        end = indptr_p[i + 1]
        for j in range(start, end):
            b = indices_p[j]
            if core_mask[b]:
                labels[i] = labels[b]
                break  # first core neighbor wins

    free(root_label)
    uf_free(&uf)

    return labels_arr, core_indices_arr
