# AgglomerativeClustering

> Language: English
> Last updated: 2026-05-07
> Switch: [Chinese](../../unsupervised/agglomerative-clustering.md)

## Overview

`AgglomerativeClustering` builds an exact hierarchical clustering tree for dense Euclidean data. It supports `"single"`, `"complete"`, `"average"`, and `"ward"` linkage on the CPU path. Explicit `device="cuda"` or `device="torch"` raises `NotImplementedError` rather than silently falling back.

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

The implementation delegates exact CPU linkage computation to SciPy hierarchy routines and does not expose a GPU path.

## Parameters

- `n_clusters`: number of clusters after cutting the tree.
- `linkage`: `"single"`, `"complete"`, `"average"`, or `"ward"`.
- `metric`: only `"euclidean"` is supported.
- `device`: CPU only; explicit `"cuda"` or `"torch"` raises `NotImplementedError`.

## CPU+GPU Examples

```python
import numpy as np
from statgpu.unsupervised import AgglomerativeClustering

X = np.random.default_rng(0).normal(size=(300, 6))

model = AgglomerativeClustering(n_clusters=4, linkage="ward", device="cpu")
labels = model.fit_predict(X)
```

## Strict/Approx Difference

There is no strict inference mode. Supported CPU linkages are exact for dense Euclidean inputs. GPU execution is intentionally unavailable rather than silently downgraded.

## Outputs

- `labels_`
- `children_`
- `distances_`
- `n_features_in_`

## FAQ

**Why is GPU unsupported?**
The current goal is an exact, clear CPU baseline. GPU hierarchical clustering needs a separate implementation plan because efficient linkage updates and memory behavior differ from the dense matrix paths used by other estimators.

**Can it predict labels for new samples?**
No. Agglomerative clustering does not support `predict` for unseen samples in this implementation.

## External Validation

- Tests: `dev/tests/test_unsupervised_agglomerative.py`.
- Benchmark: `dev/benchmarks/benchmark_unsupervised_phase3b.py`.
- Latest remote artifact: `results/unsupervised_phase3b_verify_20260507_003957.json`.
- Baselines: sklearn `AgglomerativeClustering`, SciPy `linkage`, and R `cluster::agnes` where parameter alignment is available.
- Phase 3B validation target: label agreement up to permutation, ARI, and linkage distances for `"single"`, `"complete"`, `"average"`, and `"ward"` where comparable.

## References

- Sneath, P. H. A. (1957). The application of computers to taxonomy. *Journal of General Microbiology*, 17(1), 201-226. https://doi.org/10.1099/00221287-17-1-201
- Murtagh, F. (1983). A survey of recent advances in hierarchical clustering algorithms. *The Computer Journal*, 26(4), 354-359. https://doi.org/10.1093/comjnl/26.4.354
- Muellner, D. (2013). fastcluster: Fast hierarchical, agglomerative clustering routines for R and Python. *Journal of Statistical Software*, 53(9), 1-18. https://doi.org/10.18637/jss.v053.i09
- SciPy Developers. `scipy.cluster.hierarchy`: Hierarchical clustering. SciPy documentation. https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html
