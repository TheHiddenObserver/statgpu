# AgglomerativeClustering

> Language: English
> Last updated: 2026-05-09
> Switch: [Chinese](../../unsupervised/agglomerative-clustering.md)

## Overview

`AgglomerativeClustering` builds an exact hierarchical clustering tree for dense Euclidean data. It supports `"single"`, `"complete"`, `"average"`, and `"ward"` linkage on CPU, CuPy/CUDA, and Torch CUDA paths. The GPU paths are dense exact v1 implementations intended for small to medium data; they do not silently fall back to CPU.

## Path

```python
from statgpu.unsupervised import AgglomerativeClustering
```

## Objective Function / Loss Function

Agglomerative clustering is a greedy hierarchical procedure, not a global smooth optimization. At each step it merges the pair of clusters with the smallest linkage criterion.

Single linkage:

$$
d(A, B)
=
\min_{x \in A,\; y \in B}
\left\|x - y\right\|_2 .
$$

Complete linkage:

$$
d(A, B)
=
\max_{x \in A,\; y \in B}
\left\|x - y\right\|_2 .
$$

Average linkage:

$$
d(A, B)
=
\frac{1}{|A||B|}
\sum_{x \in A}\sum_{y \in B}
\left\|x - y\right\|_2 .
$$

Ward linkage merges the pair that minimizes the increase in within-cluster squared error:

$$
\Delta(A, B)
=
\frac{|A||B|}{|A|+|B|}
\left\|\bar{x}_A-\bar{x}_B\right\|_2^2 .
$$

## Estimating Equation

- Start with every sample as its own cluster.
- Repeatedly merge the two clusters with the smallest selected linkage criterion.
- Store the merge tree as `children_` and merge distances as `distances_`.
- Cut the tree to produce `n_clusters` labels.

The CPU path delegates exact linkage computation to SciPy hierarchy routines. Explicit CuPy/Torch paths use statgpu-owned backend-resident dense distance matrices and Lance-Williams linkage updates.

## Parameters

- `n_clusters`: number of clusters after cutting the tree.
- `linkage`: `"single"`, `"complete"`, `"average"`, or `"ward"`.
- `metric`: only `"euclidean"` is supported.
- `device`: `"cpu"`, `"cuda"`, `"torch"`, or `"auto"`. `device="auto"` keeps the CPU default for this estimator; explicit GPU devices use dense exact backend execution.

## CPU+GPU Examples

```python
import numpy as np
from statgpu.unsupervised import AgglomerativeClustering

X = np.random.default_rng(0).normal(size=(300, 6))

model = AgglomerativeClustering(n_clusters=4, linkage="ward", device="cpu")
labels = model.fit_predict(X)

# With CuPy installed, keep data on the CUDA backend for the GPU path.
# import cupy as cp
# X_gpu = cp.asarray(X)
model_gpu = AgglomerativeClustering(n_clusters=4, linkage="ward", device="cuda")
labels_gpu = model_gpu.fit_predict(X_gpu)
```

## Strict/Approx Difference

There is no strict inference mode. Supported linkages are exact for dense Euclidean inputs on CPU, CuPy, and Torch. GPU execution allocates a dense distance matrix and raises a clear `MemoryError` if the configured v1 memory limit would be exceeded.

## Outputs

- `labels_`
- `children_`
- `distances_`
- `n_features_in_`

## FAQ

**When should I use the GPU path?**
Use explicit `device="cuda"` or `device="torch"` for small to medium dense datasets where backend-resident execution is useful. Hierarchical clustering is still sequential and dense-memory heavy, so large datasets may be better handled by the CPU path or a different clustering method.

**Can it predict labels for new samples?**
No. Agglomerative clustering does not support `predict` for unseen samples in this implementation.

## External Validation

- Tests: `dev/tests/test_unsupervised_agglomerative.py`.
- Benchmark: `dev/benchmarks/benchmark_unsupervised_phase3b.py`.
- Latest remote artifacts: `results/unsupervised_agglomerative_gpu_verify_20260509_agglo_gpu.json` and `results/unsupervised_agglomerative_gpu_verify_summary_20260509_agglo_gpu.md`.
- Baselines: sklearn `AgglomerativeClustering`, SciPy `linkage`, and R `cluster::agnes` where parameter alignment is available.
- Phase 3B validation target: label agreement up to permutation, ARI, and linkage distances for `"single"`, `"complete"`, `"average"`, and `"ward"` where comparable.

## References

- Sneath, P. H. A. (1957). The application of computers to taxonomy. *Journal of General Microbiology*, 17(1), 201-226. https://doi.org/10.1099/00221287-17-1-201
- Murtagh, F. (1983). A survey of recent advances in hierarchical clustering algorithms. *The Computer Journal*, 26(4), 354-359. https://doi.org/10.1093/comjnl/26.4.354
- Muellner, D. (2013). fastcluster: Fast hierarchical, agglomerative clustering routines for R and Python. *Journal of Statistical Software*, 53(9), 1-18. https://doi.org/10.18637/jss.v053.i09
- SciPy Developers. `scipy.cluster.hierarchy`: Hierarchical clustering. SciPy documentation. https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html
