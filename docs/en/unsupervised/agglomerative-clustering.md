# AgglomerativeClustering

> Language: English
> Last updated: 2026-10-05
> Switch: [Chinese](../../cn/unsupervised/agglomerative-clustering.md)

## Overview

`AgglomerativeClustering` builds an exact hierarchical clustering tree for dense Euclidean data. It supports `"single"`, `"complete"`, `"average"`, and `"ward"` linkage on CPU, CuPy/CUDA, and Torch CUDA paths. The GPU paths are dense exact implementations intended for small to medium data; they do not silently fall back to CPU.

## When to use it

Use hierarchical clustering when the nested merge structure matters. Single linkage can chain groups through a few bridge points; complete linkage emphasizes diameter, average linkage averages pair distances, and Ward favors compact groups. Distance scaling changes the hierarchy.

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

The CPU path delegates exact linkage computation to SciPy hierarchy routines. Explicit CuPy/Torch paths use statgpu-owned backend-resident dense distance matrices. Single linkage builds a minimum spanning tree and assembles the merge tree on CPU; complete, average, and Ward linkage use Lance–Williams updates.

## Parameters

- `n_clusters`: number of clusters after cutting the tree.
- `linkage`: `"single"`, `"complete"`, `"average"`, or `"ward"`.
- `metric`: only `"euclidean"` is supported.
- `device`: `"cpu"`, `"cuda"`, `"torch"`, or `"auto"`. `device="auto"` keeps the CPU default for this estimator; explicit GPU devices use dense exact backend execution.

## A small CPU example

<!-- learner-example: agglomerative-clustering -->
```python
import numpy as np
from statgpu.unsupervised import AgglomerativeClustering

rng = np.random.default_rng(0)
X = np.vstack([rng.normal(-2, 0.3, (15, 2)), rng.normal(2, 0.3, (15, 2))])
model = AgglomerativeClustering(n_clusters=2, linkage="ward", device="cpu")
labels = model.fit_predict(X)
print(labels.shape, model.children_.shape, model.distances_[-3:])
```

The tree has `(29, 2)` merge pairs for 30 observations. Inspect merge heights as well as the cut. The CPU maxclust cut may return fewer than the requested groups when heights tie. No prediction for new rows is exposed.

For a supported GPU installation, construct a new estimator with `device="cuda"` (CuPy) or `device="torch"` (Torch CUDA). Labels, merge pairs and merge heights are always returned as NumPy arrays, including after GPU fitting; see the [API reference](api-reference.md#agglomerativeclustering) for output ownership and host-side work. An unavailable explicit GPU raises an error.

## Approximation and interpretation

There is no strict inference mode. Supported linkages are exact for dense Euclidean inputs on CPU, CuPy, and Torch. GPU execution allocates a dense distance matrix and raises a clear `MemoryError` if the configured memory limit would be exceeded.

The GPU paths evaluate pairwise distances through expanded squared norms. Large common feature offsets can cause cancellation and change the hierarchy; subtract a training-derived offset in float64 before fitting. This preserves the intended Euclidean geometry. The SciPy CPU linkage path does not use that shared GPU distance formula.

Ward merge heights in `distances_` are $\sqrt{2\Delta(A,B)}$, not the raw increase $\Delta(A,B)$ itself. GPU distance-matrix estimates are checked against a configured 1 GiB default cap (`STATGPU_AGGLOMERATIVE_GPU_MAX_BYTES`, read when importing the module), not the available GPU memory.

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


## Complete API reference

Constructor defaults, all public methods, output shapes, and restrictions are listed in the [AgglomerativeClustering API reference](api-reference.md#agglomerativeclustering).

## References

- Sneath, P. H. A. (1957). The application of computers to taxonomy. *Journal of General Microbiology*, 17(1), 201-226. https://doi.org/10.1099/00221287-17-1-201
- Murtagh, F. (1983). A survey of recent advances in hierarchical clustering algorithms. *The Computer Journal*, 26(4), 354-359. https://doi.org/10.1093/comjnl/26.4.354
- Muellner, D. (2013). fastcluster: Fast hierarchical, agglomerative clustering routines for R and Python. *Journal of Statistical Software*, 53(9), 1-18. https://doi.org/10.18637/jss.v053.i09
- SciPy Developers. `scipy.cluster.hierarchy`: Hierarchical clustering. SciPy documentation. https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html
