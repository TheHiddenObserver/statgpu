# AgglomerativeClustering

> Language: English
> Last updated: 2026-05-02
> Switch: [Chinese](../../unsupervised/agglomerative-clustering.md)

## Overview

`AgglomerativeClustering` builds an exact single-linkage hierarchy for dense Euclidean data. Phase 2 is CPU-only.

## Path

```python
from statgpu.unsupervised import AgglomerativeClustering
```

## Objective Function / Loss Function

Single-linkage agglomerative clustering is a greedy hierarchical procedure, not a global smooth optimization. At each step it merges the two clusters with minimum pairwise distance:

$$
d(A, B)
=
\min_{x \in A,\; y \in B}
\left\|x - y\right\|_2 .
$$

## Estimating Equation

- Start with every sample as its own cluster.
- Repeatedly merge the two clusters with smallest single-linkage distance.
- Store the merge tree as `children_` and merge distances as `distances_`.
- Cut the tree to produce `n_clusters` labels.

The current implementation delegates the exact CPU linkage computation to SciPy's hierarchy routines and does not expose a GPU path.

## Parameters

- `n_clusters`: number of clusters after cutting the tree.
- `linkage`: only `"single"` is supported.
- `metric`: only `"euclidean"` is supported.
- `device`: CPU only in Phase 2; explicit `"cuda"` or `"torch"` raises `NotImplementedError`.

## CPU+GPU Examples

```python
import numpy as np
from statgpu.unsupervised import AgglomerativeClustering

X = np.random.default_rng(0).normal(size=(300, 6))

model = AgglomerativeClustering(n_clusters=4, linkage="single", device="cpu")
labels = model.fit_predict(X)
```

## Strict/Approx Difference

There is no strict inference mode. Phase 2 uses exact CPU single linkage for supported dense Euclidean inputs. GPU execution is intentionally unavailable rather than silently downgraded.

## Outputs

- `labels_`
- `children_`
- `distances_`
- `n_features_in_`

## FAQ

**Why is GPU unsupported?**
The Phase 2 goal is an exact, clear CPU baseline. GPU single-linkage support needs a separate implementation plan.

**Can it predict labels for new samples?**
No. Agglomerative clustering does not support `predict` for unseen samples in this implementation.

## External Validation

- Tests: `dev/tests/test_unsupervised_agglomerative.py`.
- Benchmark: `dev/benchmarks/benchmark_unsupervised_phase2.py`.
- Baselines: sklearn `AgglomerativeClustering(linkage="single")`, SciPy `linkage(method="single")`, and R `cluster::agnes` when available.
- Latest remote matrix: sklearn and SciPy labels match statgpu CPU with ARI `1.0`.

## References

- Sneath, P. H. A. (1957). The application of computers to taxonomy. *Journal of General Microbiology*, 17(1), 201-226. https://doi.org/10.1099/00221287-17-1-201
- Murtagh, F. (1983). A survey of recent advances in hierarchical clustering algorithms. *The Computer Journal*, 26(4), 354-359. https://doi.org/10.1093/comjnl/26.4.354
- Müllner, D. (2013). fastcluster: Fast hierarchical, agglomerative clustering routines for R and Python. *Journal of Statistical Software*, 53(9), 1-18. https://doi.org/10.18637/jss.v053.i09
- SciPy Developers. `scipy.cluster.hierarchy`: Hierarchical clustering. SciPy documentation. https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html
