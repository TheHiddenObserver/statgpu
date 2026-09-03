# DBSCAN

> Language: English
> Last updated: 2026-09-03
> Switch: [简体中文](../../cn/unsupervised/dbscan.md)

## What problem does it solve?

`DBSCAN` finds dense groups of nearby observations and marks isolated observations as noise. Unlike K-Means, it does not require the number of clusters in advance and can recover non-spherical cluster shapes.

statgpu's implementation targets dense Euclidean data and provides CPU, CuPy/CUDA, and Torch CUDA paths.

## When to use it

DBSCAN is a good starting point when:

- clusters are defined by local density rather than by distance to a centroid;
- clusters may have curved or irregular shapes;
- noise detection is part of the task;
- you do not know the number of clusters beforehand;
- one meaningful neighborhood scale can describe most clusters.

Choose another method when cluster densities differ strongly, distances become uninformative in very high dimensions, every new observation must receive a prediction, or you need soft membership probabilities. Reduce dimension or choose a meaningful metric before DBSCAN when raw features do not define useful Euclidean distances.

## Intuition

DBSCAN classifies training points into three roles:

- **core point:** at least `min_samples` observations, including itself, lie within radius `eps`;
- **border point:** not a core point, but lies within `eps` of a core point;
- **noise point:** not density-reachable from any core component and receives label `-1`.

Connected core points form a cluster, and reachable border points attach to it. DBSCAN therefore discovers components of sufficiently dense regions instead of drawing a fixed number of Voronoi cells.

## Density criterion

Point $x_i$ is a core point when

$$
\left|
\left\{
x_j:\lVert x_i-x_j\rVert_2\leq\varepsilon
\right\}
\right|
\geq \text{min\_samples}.
$$

Here, $\varepsilon$ is `eps`. The neighborhood is closed and includes $x_i$ itself.

DBSCAN has no differentiable loss function. Its result is determined by the neighborhood graph and density-reachability rules.

## Prepare the data first

`eps` uses the same units as the features. A variable measured in thousands can dominate a variable measured between 0 and 1. Standardize continuous features or apply a scientifically meaningful transformation before fitting.

Also inspect duplicates, missing values, and high-dimensional sparsity. This implementation accepts dense arrays and currently supports only `metric="euclidean"`.

## Minimal runnable example

The example creates three compact groups plus uniformly distributed noise.

```python
import numpy as np
from statgpu.unsupervised import DBSCAN

rng = np.random.default_rng(3)
cluster_a = rng.normal(loc=(-2.0, -1.0), scale=0.22, size=(120, 2))
cluster_b = rng.normal(loc=(0.2, 1.8), scale=0.25, size=(140, 2))
cluster_c = rng.normal(loc=(2.2, -0.5), scale=0.20, size=(110, 2))
noise = rng.uniform(low=-4.0, high=4.0, size=(30, 2))
X = np.vstack([cluster_a, cluster_b, cluster_c, noise])

model = DBSCAN(
    eps=0.45,
    min_samples=6,
    device="cpu",
).fit(X)

labels = model.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = int(np.sum(labels == -1))

print("clusters:", n_clusters)
print("noise points:", n_noise)
print("first labels:", labels[:10])
```

This dataset should produce about three clusters. Some uniformly sampled points may attach to a nearby cluster, so the number labeled as noise need not equal the 30 generated noise candidates.

## How to read the result

- `labels_[i]` is the cluster assigned to training row `i`.
- Labels `0, 1, ...` are identifiers only; their numeric order has no ranking meaning.
- Label `-1` denotes noise.
- `core_sample_indices_` contains indices of core points.
- `components_` contains the fitted core samples.
- The number of clusters is the number of distinct nonnegative labels.

DBSCAN does not implement `predict` for unseen samples. To handle new observations, refit the clustering or define and validate a separate assignment rule appropriate to the application.

## Key parameters and how to choose them

| Parameter | Default | Guidance |
|---|---:|---|
| `eps` | `0.5` | Neighborhood radius in feature units; start from domain knowledge or inspect sorted $k$-nearest-neighbor distances |
| `min_samples` | `5` | Larger values demand denser evidence and label more points as noise; smaller values admit smaller, noisier groups |
| `metric` | `"euclidean"` | The only currently supported metric |
| `algorithm` | `"auto"` | CPU neighbor-search hint; usually leave automatic |
| `batch_size` | `None` | GPU distance batch size; reduce it if temporary distance blocks exceed memory |
| `device` | `"auto"` | Choose `cpu`, `cuda`, or `torch` according to data scale and installed backend |

A practical tuning sequence is:

1. scale or transform features;
2. choose `min_samples` from the minimum local support you consider credible;
3. inspect the distance to each point's `min_samples`-th neighbor;
4. try `eps` values around a visible bend in that distance curve;
5. compare cluster stability and domain usefulness, not just the number of clusters.

There is no universal best heuristic, especially when densities vary.

## Compare with alternatives

| Method | Prefer it when |
|---|---|
| K-Means | clusters are compact, roughly spherical, and the number of clusters is known |
| Gaussian mixture | you need soft membership probabilities and elliptical components |
| Agglomerative clustering | a hierarchy or dendrogram is useful |
| HDBSCAN (not currently in this API) | cluster densities vary and hierarchical density stability is important |

## CPU and GPU behavior

All paths use the same dense Euclidean neighborhood criterion:

- low-dimensional CPU data use `cKDTree` and a compiled Cython label pipeline when available;
- higher-dimensional CPU data use optimized nearest-neighbor search and the same labeling contract;
- CuPy and Torch paths batch distance work on GPU and build connected components with backend-specific routines.

The [CPU/GPU acceleration internals](../guides/acceleration-internals.md) explain why `batch_size`, device transfer, synchronization, and label-invariant validation matter for this path.

```python
# CuPy CUDA path
gpu_model = DBSCAN(
    eps=0.45,
    min_samples=6,
    batch_size=2048,
    device="cuda",
).fit(X)

# Torch CUDA path
torch_model = DBSCAN(
    eps=0.45,
    min_samples=6,
    device="torch",
).fit(X)
```

CPU and GPU labels can use different numeric identifiers while describing the same partition. Compare partitions with a label-invariant metric such as adjusted Rand index, not by requiring label numbers to match. Points exactly near the `eps` boundary may differ because of floating-point comparisons.

Current performance evidence belongs in the versioned [benchmark dashboard](/dashboard/), where hardware and source metadata are visible.

## Common pitfalls

- Fitting unscaled features is the most common source of misleading neighborhoods.
- Increasing `eps` can merge separate groups; decreasing it can fragment groups and create more noise.
- Increasing `min_samples` does not simply create “better” clusters—it changes the density definition.
- In high dimensions, most distances can become similar. Dimension reduction may be necessary.
- Cluster labels are exploratory structure, not ground-truth classes.
- DBSCAN on the full dataset has no built-in out-of-sample prediction step.

## API and validation

Import path: `statgpu.unsupervised.DBSCAN`

Methods: `fit` and `fit_predict`. Calling `predict` raises `NotImplementedError` by design.

Outputs: `labels_`, `core_sample_indices_`, `components_`, and `n_features_in_`.

Validation against scikit-learn's aligned Euclidean configuration is in `dev/tests/test_unsupervised_dbscan.py`. The benchmark generator records accuracy and runtime provenance separately from this conceptual guide.

## References

- Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. In *KDD-96* (pp. 226-231).
- Schubert, E., Sander, J., Ester, M., Kriegel, H.-P., & Xu, X. (2017). DBSCAN revisited, revisited: Why and how you should (still) use DBSCAN. *ACM Transactions on Database Systems*, 42(3), Article 19.
