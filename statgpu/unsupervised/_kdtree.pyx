"""Complete Cython KD-tree for fast range queries.

Implements:
- Array-based tree storage (no Python objects in inner loops)
- Stack-based iterative range queries
- Median-of-three splitting for balanced trees
- Integration with DBSCAN
"""
# cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from libc.stdlib cimport malloc, free, realloc
from libc.string cimport memset

np.import_array()

ctypedef np.float64_t DTYPE_t
ctypedef np.int64_t ITYPE_t

# Node structure stored in flat arrays for cache efficiency
# split_dim[i], split_val[i], left[i], right[i], start[i], end[i]


cdef class KDTree:
    """Cython KD-tree with C-level range queries."""

    cdef double* data           # (n, d) data matrix
    cdef int* indices           # Point indices (permuted during build)
    cdef int* split_dim         # Splitting dimension per node
    cdef double* split_val      # Splitting value per node
    cdef int* left_child        # Left child index (-1 if leaf)
    cdef int* right_child       # Right child index (-1 if leaf)
    cdef int* node_start        # Start index in point array
    cdef int* node_end          # End index in point array
    cdef int n_nodes            # Number of nodes
    cdef int n                  # Number of points
    cdef int d                  # Dimensionality
    cdef int leaf_size          # Max points per leaf
    cdef int max_nodes          # Allocated node capacity

    def __cinit__(self, double[:, :] X, int leaf_size=32):
        cdef int i, j

        self.n = X.shape[0]
        self.d = X.shape[1]
        self.leaf_size = leaf_size
        self.max_nodes = 4 * (self.n // max(leaf_size, 1) + 10)
        self.n_nodes = 0

        # Allocate arrays
        self.data = <double*>malloc(self.n * self.d * sizeof(double))
        self.indices = <int*>malloc(self.n * sizeof(int))
        self.split_dim = <int*>malloc(self.max_nodes * sizeof(int))
        self.split_val = <double*>malloc(self.max_nodes * sizeof(double))
        self.left_child = <int*>malloc(self.max_nodes * sizeof(int))
        self.right_child = <int*>malloc(self.max_nodes * sizeof(int))
        self.node_start = <int*>malloc(self.max_nodes * sizeof(int))
        self.node_end = <int*>malloc(self.max_nodes * sizeof(int))

        # Copy data to C array (row-major)
        for i in range(self.n):
            for j in range(self.d):
                self.data[i * self.d + j] = X[i, j]
            self.indices[i] = i

        # Build tree
        self._build(0, self.n, 0)

    def __dealloc__(self):
        if self.data != NULL: free(self.data)
        if self.indices != NULL: free(self.indices)
        if self.split_dim != NULL: free(self.split_dim)
        if self.split_val != NULL: free(self.split_val)
        if self.left_child != NULL: free(self.left_child)
        if self.right_child != NULL: free(self.right_child)
        if self.node_start != NULL: free(self.node_start)
        if self.node_end != NULL: free(self.node_end)

    cdef int _build(self, int start, int end, int depth):
        """Build subtree for points in [start, end)."""
        cdef int n_points = end - start
        if n_points <= 0:
            return -1

        cdef int node_idx = self.n_nodes
        self.n_nodes += 1

        # Leaf node
        if n_points <= self.leaf_size:
            self.split_dim[node_idx] = -1
            self.split_val[node_idx] = 0.0
            self.left_child[node_idx] = -1
            self.right_child[node_idx] = -1
            self.node_start[node_idx] = start
            self.node_end[node_idx] = end
            return node_idx

        # Find splitting dimension (max spread)
        cdef int split_dim = 0
        cdef double max_spread = 0.0
        cdef double min_val, max_val, spread
        cdef int dim, i, idx

        for dim in range(self.d):
            min_val = self.data[self.indices[start] * self.d + dim]
            max_val = min_val
            for i in range(start + 1, end):
                idx = self.indices[i]
                if self.data[idx * self.d + dim] < min_val:
                    min_val = self.data[idx * self.d + dim]
                if self.data[idx * self.d + dim] > max_val:
                    max_val = self.data[idx * self.d + dim]
            spread = max_val - min_val
            if spread > max_spread:
                max_spread = spread
                split_dim = dim

        # Find median using quickselect
        cdef int mid = start + n_points // 2
        self._quickselect(start, end, mid, split_dim)
        cdef double split_val = self.data[self.indices[mid] * self.d + split_dim]

        # Build children
        cdef int left_idx = self._build(start, mid, depth + 1)
        cdef int right_idx = self._build(mid + 1, end, depth + 1)

        # Store node
        self.split_dim[node_idx] = split_dim
        self.split_val[node_idx] = split_val
        self.left_child[node_idx] = left_idx
        self.right_child[node_idx] = right_idx
        self.node_start[node_idx] = start
        self.node_end[node_idx] = end

        return node_idx

    cdef void _quickselect(self, int start, int end, int k, int dim):
        """Quickselect to find k-th element along dim."""
        cdef int left = start
        cdef int right = end - 1
        cdef int pivot_idx, store, j
        cdef double pivot_val
        cdef int temp

        while left < right:
            # Median-of-three pivot
            pivot_idx = self._median_of_three(left, right, dim)
            pivot_val = self.data[self.indices[pivot_idx] * self.d + dim]

            # Partition
            temp = self.indices[pivot_idx]
            self.indices[pivot_idx] = self.indices[right]
            self.indices[right] = temp

            store = left
            for j in range(left, right):
                if self.data[self.indices[j] * self.d + dim] < pivot_val:
                    temp = self.indices[j]
                    self.indices[j] = self.indices[store]
                    self.indices[store] = temp
                    store += 1

            temp = self.indices[store]
            self.indices[store] = self.indices[right]
            self.indices[right] = temp

            if store == k:
                return
            elif store < k:
                left = store + 1
            else:
                right = store - 1

    cdef int _median_of_three(self, int left, int right, int dim):
        """Find median of left, mid, right along dim."""
        cdef int mid = left + (right - left) // 2
        cdef double a = self.data[self.indices[left] * self.d + dim]
        cdef double b = self.data[self.indices[mid] * self.d + dim]
        cdef double c = self.data[self.indices[right] * self.d + dim]

        if a < b:
            if b < c:
                return mid
            elif a < c:
                return right
            else:
                return left
        else:
            if a < c:
                return left
            elif b < c:
                return right
            else:
                return mid

    def query_radius(self, double[:, :] query_points, double eps):
        """Find all neighbors within eps for each query point.

        Two-pass approach: first count, then fill exact-size arrays.
        Avoids Python list overhead entirely.
        """
        cdef int m = query_points.shape[0]
        cdef double eps_sq = eps * eps

        # Pass 1: Count neighbors per point
        cdef np.ndarray[ITYPE_t, ndim=1] counts = np.zeros(m, dtype=np.int64)
        cdef int* stack = <int*>malloc(self.max_nodes * sizeof(int))
        cdef int stack_size
        cdef int i, node_idx, j, k, idx
        cdef double dist_sq, diff
        cdef int split_dim
        cdef double split_val, qval

        for i in range(m):
            stack_size = 0
            stack[0] = 0
            stack_size = 1
            while stack_size > 0:
                stack_size -= 1
                node_idx = stack[stack_size]
                if node_idx < 0:
                    continue
                split_dim = self.split_dim[node_idx]
                if split_dim == -1:
                    for j in range(self.node_start[node_idx], self.node_end[node_idx]):
                        idx = self.indices[j]
                        dist_sq = 0.0
                        for k in range(self.d):
                            diff = query_points[i, k] - self.data[idx * self.d + k]
                            dist_sq += diff * diff
                        if dist_sq <= eps_sq:
                            counts[i] += 1
                else:
                    split_val = self.split_val[node_idx]
                    qval = query_points[i, split_dim]
                    if qval - eps <= split_val:
                        stack[stack_size] = self.left_child[node_idx]
                        stack_size += 1
                    if qval + eps > split_val:
                        stack[stack_size] = self.right_child[node_idx]
                        stack_size += 1

        # Compute total pairs and prefix sum for offsets
        cdef int total = 0
        for i in range(m):
            total += counts[i]

        cdef np.ndarray[ITYPE_t, ndim=1] out_rows = np.empty(total, dtype=np.int64)
        cdef np.ndarray[ITYPE_t, ndim=1] out_cols = np.empty(total, dtype=np.int64)

        # Pass 2: Fill arrays
        cdef int offset = 0
        for i in range(m):
            stack_size = 0
            stack[0] = 0
            stack_size = 1
            while stack_size > 0:
                stack_size -= 1
                node_idx = stack[stack_size]
                if node_idx < 0:
                    continue
                split_dim = self.split_dim[node_idx]
                if split_dim == -1:
                    for j in range(self.node_start[node_idx], self.node_end[node_idx]):
                        idx = self.indices[j]
                        dist_sq = 0.0
                        for k in range(self.d):
                            diff = query_points[i, k] - self.data[idx * self.d + k]
                            dist_sq += diff * diff
                        if dist_sq <= eps_sq:
                            out_rows[offset] = i
                            out_cols[offset] = idx
                            offset += 1
                else:
                    split_val = self.split_val[node_idx]
                    qval = query_points[i, split_dim]
                    if qval - eps <= split_val:
                        stack[stack_size] = self.left_child[node_idx]
                        stack_size += 1
                    if qval + eps > split_val:
                        stack[stack_size] = self.right_child[node_idx]
                        stack_size += 1

        free(stack)
        return counts, out_rows, out_cols


def dbscan_kdtree(np.ndarray[DTYPE_t, ndim=2] X, double eps, int min_samples, int leaf_size=32):
    """DBSCAN using Cython KD-tree.

    Parameters
    ----------
    X : (n, d) array
        Input data.
    eps : float
        Neighborhood radius.
    min_samples : int
        Minimum samples for core point.
    leaf_size : int
        Max points per leaf node.

    Returns
    -------
    labels : (n,) int64 array
        Cluster labels (-1 for noise).
    core_indices : int64 array
        Indices of core points.
    """
    cdef int n = X.shape[0]

    # Build tree
    tree = KDTree(X, leaf_size)

    # Query all neighbors
    cdef np.ndarray[ITYPE_t, ndim=1] counts
    cdef np.ndarray[ITYPE_t, ndim=1] row_idx
    cdef np.ndarray[ITYPE_t, ndim=1] col_idx
    counts, row_idx, col_idx = tree.query_radius(X, eps)

    # Find core points
    cdef np.ndarray[ITYPE_t, ndim=1] core_indices = np.flatnonzero(counts >= min_samples).astype(np.int64)
    cdef int n_core = len(core_indices)

    if n_core == 0:
        return np.full(n, -1, dtype=np.int64), core_indices

    # Build core-core adjacency
    core_mask_np = (counts >= min_samples).astype(np.uint8)

    core_edges = core_mask_np[row_idx].astype(bool) & core_mask_np[col_idx].astype(bool)
    core_pos = np.full(n, -1, dtype=np.int64)
    core_pos[core_indices] = np.arange(n_core, dtype=np.int64)

    core_row = core_pos[row_idx[core_edges]]
    core_col = core_pos[col_idx[core_edges]]

    if len(core_row) > 0:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        graph = csr_matrix(
            (np.ones(len(core_row), dtype=bool), (core_row, core_col)),
            shape=(n_core, n_core),
        )
        _, core_labels = connected_components(graph, directed=False, return_labels=True)
    else:
        core_labels = np.zeros(n_core, dtype=np.int32)

    # Assign labels
    cdef np.ndarray[ITYPE_t, ndim=1] labels = np.full(n, -1, dtype=np.int64)
    for i in range(n_core):
        labels[core_indices[i]] = int(core_labels[i])

    # Border points
    for i in range(len(row_idx)):
        if not core_mask_np[row_idx[i]] and core_mask_np[col_idx[i]]:
            if labels[row_idx[i]] == -1:
                labels[row_idx[i]] = labels[col_idx[i]]

    return labels, core_indices
