# DBSCAN

> Language: English
> Last updated: 2026-10-05
> Switch: [Chinese](../../cn/unsupervised/dbscan.md)

## Overview

`DBSCAN` finds density-connected components in dense Euclidean data. It supports CPU, CuPy/CUDA, and Torch CUDA paths. Execution and memory costs depend on data density, dimensionality and optional compiled extensions; GPU paths include host transfers.

## When to use it

Use DBSCAN when dense regions may have irregular shapes and some observations should remain unassigned. `eps` is measured in feature units; inspect several radii and `min_samples` values. Unequal cluster densities can make one global radius unsuitable.

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

## Execution and memory

CPU data with at most 12 features use SciPy tree search. Wider CPU data use scikit-learn `NearestNeighbors`; install scikit-learn for that path. Compiled statgpu Cython extensions can accelerate graph labeling; an uncompiled installation uses a Python implementation.

GPU distances use float32, while stored core observations use float64. Both GPU paths include host transfers, and the CuPy path also performs some graph processing in Python. Include those costs when timing the complete fit. `batch_size` limits distance batches, not the total neighbor graph: dense neighborhoods can still require quadratic memory. Check label/noise stability near the `eps` boundary, where float32 comparisons can differ.

Large common feature offsets can also erase separations during the GPU float32 conversion or expanded-distance calculation, even far from the `eps` boundary. Subtract one training-derived feature offset while the data are still float64, before fitting; translation preserves Euclidean distances and leaves `eps` unchanged. Keep that offset to interpret `components_` in original units. Finite labels alone do not establish that the neighbor graph is correct.

## Parameters

- `eps`: neighborhood radius; must be positive and finite. Non-finite values are not reliably rejected, so validate this control before fitting.
- `min_samples`: minimum closed-neighborhood count for a core sample.
- `metric`: only `"euclidean"` is supported.
- `batch_size`: optional GPU distance-batch size; does not bound all stored graph edges.
- `device`: `"auto"`, `"cpu"`, `"cuda"`, or `"torch"`.

## A small CPU example

<!-- learner-example: dbscan -->
```python
import numpy as np
from statgpu.unsupervised import DBSCAN

rng = np.random.default_rng(0)
X = np.vstack([rng.normal(-2, 0.1, (20, 2)), rng.normal(2, 0.1, (20, 2)), [[8., 8.]]])
model = DBSCAN(eps=0.5, min_samples=3, device="cpu")
labels = model.fit_predict(X)
print(labels.shape, np.unique(labels), model.core_sample_indices_.shape)
```

The last row is isolated and receives `-1`, the noise label. `components_` stores core observations, not cluster centers. There is no new-data prediction rule; refitting a combined dataset can change earlier assignments.

For a supported GPU installation, construct a new estimator with `device="cuda"` (CuPy) or `device="torch"` (Torch CUDA). Arrays generally stay on that backend; see the [API reference](api-reference.md#dbscan) for output ownership and host-side work. An unavailable explicit GPU raises an error.

## Approximation and interpretation

The intended algorithm uses Euclidean neighborhoods rather than statistical inference. A current limitation in the uncompiled CPU path can incorrectly merge disconnected core points when no core-core edges exist. Validate such sparse-core configurations against an independent implementation before interpreting the labels; dense-cluster examples do not exercise this case.

## Outputs

- `labels_`
- `core_sample_indices_`
- `components_`
- `n_features_in_`

## FAQ

**Can I predict a cluster for a new observation?**
No. Only training-set `fit_predict` is available; `predict` raises `NotImplementedError`.

**Will GPU fitting be faster?**
That depends on data size, density, transfers and available memory. Benchmark the complete workload; GPU support alone is not a speed guarantee.


## Complete API reference

Constructor defaults, all public methods, output shapes, and restrictions are listed in the [DBSCAN API reference](api-reference.md#dbscan).

## References

- Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. In *Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD-96)* (pp. 226-231). AAAI Press. https://aaai.org/papers/kdd96-037-a-density-based-algorithm-for-discovering-clusters-in-large-spatial-databases-with-noise/
- Schubert, E., Sander, J., Ester, M., Kriegel, H.-P., & Xu, X. (2017). DBSCAN revisited, revisited: Why and how you should (still) use DBSCAN. *ACM Transactions on Database Systems*, 42(3), Article 19. https://doi.org/10.1145/3068335
