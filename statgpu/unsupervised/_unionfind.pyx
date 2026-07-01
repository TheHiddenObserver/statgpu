"""Cython Union-Find for fast connected components in DBSCAN.

Union-Find with path compression and union by rank.
Nearly O(n) for n elements with α(n) ≈ 1.
"""
# cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

np.import_array()

ctypedef np.int64_t ITYPE_t


cdef class UnionFind:
    """Union-Find data structure with path compression and union by rank."""

    cdef ITYPE_t* parent
    cdef ITYPE_t* rank
    cdef int n

    def __cinit__(self, int n):
        self.n = n
        self.parent = <ITYPE_t*>malloc(n * sizeof(ITYPE_t))
        self.rank = <ITYPE_t*>malloc(n * sizeof(ITYPE_t))
        cdef int i
        for i in range(n):
            self.parent[i] = i
            self.rank[i] = 0

    def __dealloc__(self):
        if self.parent != NULL: free(self.parent)
        if self.rank != NULL: free(self.rank)

    cdef ITYPE_t find(self, ITYPE_t x):
        """Find root with path compression."""
        cdef ITYPE_t root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # Path compression
        cdef ITYPE_t curr = x
        cdef ITYPE_t next_node
        while self.parent[curr] != root:
            next_node = self.parent[curr]
            self.parent[curr] = root
            curr = next_node
        return root

    cdef void union(self, ITYPE_t x, ITYPE_t y):
        """Union by rank."""
        cdef ITYPE_t rx = self.find(x)
        cdef ITYPE_t ry = self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1

    def find_all(self):
        """Find roots for all elements."""
        cdef np.ndarray[ITYPE_t, ndim=1] roots = np.empty(self.n, dtype=np.int64)
        cdef int i
        for i in range(self.n):
            roots[i] = self.find(i)
        return roots


def connected_components_uf(np.ndarray[ITYPE_t, ndim=1] row_idx,
                            np.ndarray[ITYPE_t, ndim=1] col_idx,
                            int n_nodes):
    """Compute connected components using Union-Find.

    Much faster than scipy's connected_components for dense graphs.

    Parameters
    ----------
    row_idx : int64 array
        Row indices of edges.
    col_idx : int64 array
        Column indices of edges.
    n_nodes : int
        Number of nodes.

    Returns
    -------
    labels : (n_nodes,) int32 array
        Component labels (0, 1, 2, ...).
    n_components : int
        Number of connected components.
    """
    cdef UnionFind uf = UnionFind(n_nodes)
    cdef int n_edges = len(row_idx)
    cdef int i

    # Union all edges
    for i in range(n_edges):
        uf.union(row_idx[i], col_idx[i])

    # Find roots and relabel
    cdef np.ndarray roots = uf.find_all()

    # Compact labels
    cdef dict label_map = {}
    cdef int next_label = 0
    cdef np.ndarray[np.int32_t, ndim=1] labels = np.empty(n_nodes, dtype=np.int32)
    cdef int root

    for i in range(n_nodes):
        root = int(roots[i])
        if root not in label_map:
            label_map[root] = next_label
            next_label += 1
        labels[i] = label_map[root]

    return labels, next_label
