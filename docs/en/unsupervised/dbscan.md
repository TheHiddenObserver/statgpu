# DBSCAN

> Language: English
> Last updated: 2026-05-02
> Switch: [Chinese](../../unsupervised/dbscan.md)

## Overview

`DBSCAN` finds density-connected components in dense Euclidean data. It supports CPU, CuPy/CUDA, and Torch CUDA paths. The CPU path has an exact NumPy/SciPy fallback and an optional statgpu-owned Cython fast path for compact dense cases.

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

## Estimating Equation

- CPU first estimates neighborhood density using `cKDTree`.
- Compact dense CPU inputs can use `_dbscan_cpu.pyx`, which scans pairwise distances and unions core-core neighbor pairs.
- CPU fallback uses SciPy/NumPy exact strategies: condensed `pdist`, sparse distance matrix, or `query_pairs` depending on density and memory.
- CuPy/Torch paths build a dense boolean neighbor graph in batches, identify core samples, propagate connected component labels over the core graph, then assign border samples.

## Parameters

- `eps`: neighborhood radius; must be positive.
- `min_samples`: minimum closed-neighborhood count for a core sample.
- `metric`: only `"euclidean"` is supported.
- `batch_size`: optional GPU neighbor-graph chunk size.
- `device`: `"auto"`, `"cpu"`, `"cuda"`, or `"torch"`.

## CPU+GPU Examples

```python
import numpy as np
from statgpu.unsupervised import DBSCAN

X = np.random.default_rng(0).normal(size=(5000, 8))

labels_cpu = DBSCAN(eps=1.0, min_samples=5, device="cpu").fit_predict(X)
labels_cuda = DBSCAN(eps=1.0, min_samples=5, device="cuda", batch_size=1024).fit_predict(X)
```

## Strict/Approx Difference

There is no strict inference mode. CPU fallback and Cython fast path are exact for supported dense Euclidean input. GPU paths compute the same dense neighbor relation subject to floating-point comparison at the `eps` boundary.

## Outputs

- `labels_`
- `core_sample_indices_`
- `components_`
- `n_features_in_`

## FAQ

**Does production DBSCAN call sklearn?**
No. sklearn is used only for tests and benchmarks.

**When is Cython used?**
Only when the optional extension is built and the CPU selector identifies compact dense input. Variable-density, sparse/all-noise, and no-compiler environments use fallback.

**Why can Cython still be slower than sklearn?**
The Cython path is statgpu-owned, and performance depends on data density, selector choice, CPU library overhead, and hardware. Detailed timing conclusions live in the benchmark artifacts rather than in this model page.

## External Validation

- Tests: `dev/tests/test_unsupervised_dbscan.py`.
- Benchmarks: `dev/benchmarks/benchmark_unsupervised_phase2.py` and `dev/benchmarks/benchmark_unsupervised_dbscan_cython.py`.
- Baseline: sklearn DBSCAN with aligned `eps`, `min_samples`, and Euclidean metric.
- Latest artifacts cover compact, variable-density, and all-noise cases across statgpu CPU fallback, optional Cython CPU, CuPy, Torch, and sklearn CPU baselines. Labels and noise masks are checked against the aligned reference.

## References

- Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. In *Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD-96)* (pp. 226-231). AAAI Press. https://aaai.org/papers/kdd96-037-a-density-based-algorithm-for-discovering-clusters-in-large-spatial-databases-with-noise/
- Schubert, E., Sander, J., Ester, M., Kriegel, H.-P., & Xu, X. (2017). DBSCAN revisited, revisited: Why and how you should (still) use DBSCAN. *ACM Transactions on Database Systems*, 42(3), Article 19. https://doi.org/10.1145/3068335
