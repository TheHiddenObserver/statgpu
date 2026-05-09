# MiniBatchKMeans

> Language: English
> Last updated: 2026-05-09
> Path: `statgpu.unsupervised.MiniBatchKMeans`

## Overview

`MiniBatchKMeans` fits K-Means centers with small batches instead of full Lloyd passes over the whole dataset.

## Path

Import from `statgpu.unsupervised`:

```python
from statgpu.unsupervised import MiniBatchKMeans
```

## Objective Function / Loss Function

The target is the same inertia as KMeans:

$$
\sum_i \min_j \|x_i - c_j\|_2^2.
$$

Mini-batch updates optimize this objective approximately.

## Estimating Equation

For a batch-assigned cluster `j`, statgpu updates its center with cumulative counts:

$$
c_j \leftarrow c_j + \eta_j(\bar{x}_{B_j} - c_j),
\qquad
\eta_j = \frac{|B_j|}{n_j + |B_j|}.
$$

After mini-batch updates, `fit` runs a small exact Lloyd polishing pass on the full dense dataset. This keeps the estimator mini-batch driven while reducing the final inertia gap against full-data label assignments.

## Parameters

`n_clusters`, `init`, `n_init`, `batch_size`, `max_iter`, `max_no_improvement`, `tol`, `random_state`, and `device`.

## CPU+GPU Examples

```python
from statgpu.unsupervised import MiniBatchKMeans

km = MiniBatchKMeans(n_clusters=20, batch_size=4096, device="cpu")
labels = km.fit_predict(X)

km_gpu = MiniBatchKMeans(n_clusters=20, batch_size=4096, device="torch")
labels_gpu = km_gpu.fit_predict(X_torch)
```

## Strict/Approx Difference

The method is stochastic and approximate. Fair comparisons should use the same initial centers, batch order, tolerance, and iteration budget.

## Outputs

`cluster_centers_`, `labels_`, `inertia_`, `n_iter_`, `n_steps_`, `counts_`, and `n_features_in_`.

## FAQ

Phase 3A supports dense Euclidean data only. Sparse input, sample weights, and callable initialization are not supported.

## External Validation

Tests: `dev/tests/test_unsupervised_minibatch_kmeans.py`.
Benchmark: `dev/benchmarks/benchmark_unsupervised_phase3.py`.
Baseline: sklearn `MiniBatchKMeans`.

## References

- Sculley, D. (2010). Web-scale k-means clustering. *Proceedings of the 19th International Conference on World Wide Web*, 1177-1178.
- scikit-learn developers. `sklearn.cluster.MiniBatchKMeans` API documentation.
