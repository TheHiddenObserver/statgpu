# DBSCAN

> Language: English
> Last updated: 2026-06-26
> Switch: [Chinese](../../cn/unsupervised/dbscan.md)

## Overview

`DBSCAN` finds density-connected components in dense Euclidean data. It supports CPU, CuPy/CUDA, and Torch CUDA paths. The CPU path uses a Cython-accelerated pipeline that is 3-4x faster than sklearn for low-dimensional data and matches sklearn for high-dimensional data. The GPU path (PyTorch CUDA) runs the entire pipeline on-device with zero GPU→CPU transfer, achieving 3-17x speedup over sklearn.

## Path

```python
from statgpu.unsupervised import DBSCAN
```

## Objective Function / Loss Function

DBSCAN is not a smooth optimization problem. It has no differentiable loss to minimize. Its criterion is density reachability:

- A point is core if its closed `eps` neighborhood contains at least `min_samples` points.
  $$
  \left|\left\{x_j : \left\|x_i - x_j\right\|_2 \le \varepsilon\right\}\right|
  \ge \text{min\_samples}.
  $$
- Core points connected by `eps`-neighbor chains form a cluster.
- Non-core points reachable from a core component are border points.
- Other points are noise with label `-1`.

## CPU Strategy

The CPU path selects an algorithm based on dimensionality:

| Dimensionality | Strategy | Details |
|---|---|---|
| p ≤ 12 | cKDTree `query_pairs` + Cython | Single tree traversal; `dbscan_labels_from_pairs` runs counting, Union-Find, and label assignment entirely in C. |
| p > 12 | sklearn `radius_neighbors_graph` + Cython | Uses sklearn's optimized BLAS for distance computation; `dbscan_labels_from_csr` processes the CSR graph in C. |

Both paths have a pure Python fallback when the Cython extension is not compiled.

### Cython Module: `_dbscan_cy_fast.pyx`

Two entry points, both running the full label pipeline in C (no Python object overhead):

- `dbscan_labels_from_pairs(n_samples, min_samples, pairs)` — takes raw `(i, j)` pairs from `query_pairs`.
- `dbscan_labels_from_csr(n_samples, min_samples, indptr, indices)` — takes CSR sparse graph arrays.

Internally both use:
- C-level neighbor counting
- C-level Union-Find with path compression and union by rank
- C-level border point assignment

## GPU Strategy (PyTorch CUDA)

The GPU path keeps all data on-device:

1. **Distance computation**: batched `float32` matmul on GPU
2. **Neighbor counting**: `mask.sum(dim=1)` on GPU
3. **Sparse graph**: `torch.nonzero` on GPU, edges stored as GPU tensors
4. **Connected components**: label propagation via `scatter_reduce_(amin)` on GPU
5. **Border assignment**: batched distance + scatter on GPU

Only the final labels (`n × int64`) are transferred to CPU. This eliminates per-batch GPU→CPU transfer overhead and avoids OOM from recomputing distances.

### Label Propagation Algorithm

```
labels = arange(n_core)                          # each core point starts independent
for _ in range(50):                              # typically converges in 2-5 iterations
    min_labels = minimum(labels[src], labels[dst])  # parallel over all edges
    labels.scatter_reduce_(amin)                     # parallel scatter
    if converged: break
```

This is well-suited for GPU: each iteration is fully parallel over all edges, unlike CPU Union-Find which processes edges sequentially.

## Parameters

- `eps`: neighborhood radius; must be positive.
- `min_samples`: minimum closed-neighborhood count for a core sample.
- `metric`: only `"euclidean"` is supported.
- `batch_size`: optional GPU neighbor-graph chunk size. Default targets ~2GB per batch.
- `device`: `"auto"`, `"cpu"`, `"cuda"`, or `"torch"`.

## CPU+GPU Examples

```python
import numpy as np
from statgpu.unsupervised import DBSCAN

X = np.random.default_rng(0).normal(size=(5000, 8))

# CPU (low-dim: Cython fast path)
labels_cpu = DBSCAN(eps=1.0, min_samples=5, device="cpu").fit_predict(X)

# GPU (PyTorch CUDA: fully on-device)
labels_torch = DBSCAN(eps=1.0, min_samples=5, device="torch").fit_predict(X)

# GPU (CuPy: distance on GPU, labels on CPU via Cython)
labels_cuda = DBSCAN(eps=1.0, min_samples=5, device="cuda", batch_size=1024).fit_predict(X)
```

## Performance

Measured on Tesla P100-SXM2-16GB (GPU) and Intel Xeon (CPU), median of 3 runs:

| n | p | sklearn CPU | statgpu CPU | statgpu GPU (torch) | GPU / sklearn |
|---|---|---|---|---|---|
| 10000 | 5 | 0.46s | 0.18s | 0.03s | **0.06x** |
| 30000 | 5 | 3.32s | 1.35s | 0.24s | **0.07x** |
| 50000 | 5 | 9.49s | 3.88s | 0.71s | **0.07x** |
| 10000 | 50 | 0.05s | 0.06s | 0.01s | **0.28x** |
| 30000 | 50 | 0.39s | 0.32s | 0.12s | **0.30x** |
| 50000 | 50 | 1.08s | 0.89s | 0.32s | **0.30x** |

All cases produce ARI = 1.0000 vs sklearn reference.

## Strict/Approx Difference

There is no strict inference mode. CPU fallback and Cython fast path are exact for supported dense Euclidean input. GPU paths compute the same dense neighbor relation subject to floating-point comparison at the `eps` boundary.

## Outputs

- `labels_`
- `core_sample_indices_`
- `components_`
- `n_features_in_`

## FAQ

**Does production DBSCAN call sklearn?**
For the CPU path with p > 12, sklearn's `NearestNeighbors` is used for optimized BLAS distance computation. The graph processing and label assignment are handled by statgpu's Cython code. For p ≤ 12, no sklearn dependency exists.

**When is Cython used?**
When the `_dbscan_cy_fast` extension is compiled (via `python setup.py build_ext --inplace`). Without Cython, a pure Python fallback is used. The Cython module must be compiled on the target machine.

**Why is the GPU path faster?**
The GPU path keeps all intermediate data (distances, edges, labels) on-device. Label propagation for connected components is fully parallel on GPU, unlike CPU Union-Find which processes edges sequentially. Only the final labels are transferred to CPU.

## External Validation

- Tests: `dev/tests/test_unsupervised_dbscan.py`.
- Benchmarks: `dev/benchmarks/benchmark_unsupervised_dbscan_cython.py`.
- Baseline: sklearn DBSCAN with aligned `eps`, `min_samples`, and Euclidean metric.
- Labels and noise masks are checked against the aligned reference (ARI = 1.0).

## References

- Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. In *Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD-96)* (pp. 226-231). AAAI Press. https://aaai.org/papers/kdd96-037-a-density-based-algorithm-for-discovering-clusters-in-large-spatial-databases-with-noise/
- Schubert, E., Sander, J., Ester, M., Kriegel, H.-P., & Xu, X. (2017). DBSCAN revisited, revisited: Why and how you should (still) use DBSCAN. *ACM Transactions on Database Systems*, 42(3), Article 19. https://doi.org/10.1145/3068335
