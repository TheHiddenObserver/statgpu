# MiniBatchKMeans

> Language: English
> Last updated: 2026-10-05
> Path: `statgpu.unsupervised.MiniBatchKMeans`

## Overview

`MiniBatchKMeans` fits K-Means centers with small batches instead of full Lloyd passes over the whole dataset.

## When to use it

Use MiniBatchKMeans when repeated smaller center updates are useful. `fit` still takes the whole dense dataset and performs final full-data polishing; use `partial_fit` for an external stream. Fix feature preprocessing across batches and supply enough rows to initialize all clusters.

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

## A small CPU example

<!-- learner-example: minibatch-kmeans -->
```python
import numpy as np
from statgpu.unsupervised import MiniBatchKMeans

rng = np.random.default_rng(0)
X = np.vstack([rng.normal(-2, 0.3, (30, 2)), rng.normal(2, 0.3, (30, 2))])
rng.shuffle(X)
model = MiniBatchKMeans(n_clusters=2, random_state=0, device="cpu")
for start in range(0, len(X), 15):
    model.partial_fit(X[start:start + 15])
labels = model.predict(X)
print(labels.shape, model.labels_.shape, model.n_steps_)
```

The requested all-data labels have shape `(60,)`, while `labels_` only covers the last batch `(15,)`. That stored batch labeling predates its center update; `predict` uses current centers. `counts_` accumulates assignments after `partial_fit`.

For a supported GPU installation, construct a new estimator with `device="cuda"` (CuPy) or `device="torch"` (Torch CUDA). Arrays generally stay on that backend; see the [API reference](api-reference.md#minibatchkmeans) for output ownership and host-side work. An unavailable explicit GPU raises an error.

## Approximation and interpretation

The method is stochastic and approximate. Fair comparisons should use the same initial centers, batch order, tolerance, and iteration budget.

## Outputs

`cluster_centers_`, `labels_`, `inertia_`, `n_iter_`, `n_steps_`, `counts_`, and `n_features_in_`.

## FAQ

The current implementation supports dense Euclidean data only. Sparse input, sample weights, and callable initialization are not supported.


## Numerical and lifecycle cautions

Squared distances are evaluated through expanded norms. A large shared offset can cause cancellation, changing distances, inertia and potentially labels. Subtract a training-derived feature offset before fitting and use that same offset for prediction; this preserves Euclidean geometry. Centers can be reported in original units by adding the offset back.

## Complete API reference

Constructor defaults, all public methods, output shapes, and restrictions are listed in the [MiniBatchKMeans API reference](api-reference.md#minibatchkmeans).

## References

- Sculley, D. (2010). Web-scale k-means clustering. *Proceedings of the 19th International Conference on World Wide Web*, 1177-1178.
- scikit-learn developers. `sklearn.cluster.MiniBatchKMeans` API documentation.
