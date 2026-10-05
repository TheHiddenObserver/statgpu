# UMAP

> Language: English
> Last updated: 2026-10-05
> Path: `statgpu.unsupervised.UMAP`

## Overview

`UMAP` builds a fuzzy neighbor graph in the input space and optimizes a low-dimensional embedding. It supports dense exact Euclidean neighbors and an internal NNDescent neighbor-search option.

## When to use it

Use UMAP to explore local neighborhood structure visually. Compare several seeds, neighborhood sizes and `min_dist` settings. Distances between separate islands and apparent cluster sizes are not direct estimates of population separation or prevalence.

## Path

Import from `statgpu.unsupervised`:

```python
from statgpu.unsupervised import UMAP
```

## Objective Function / Loss Function

The standard UMAP reference objective is fuzzy-set cross-entropy between high-dimensional graph weights `w_ij` and low-dimensional affinities `q_ij`:

$$
\sum_{i,j} w_{ij}\log\frac{w_{ij}}{q_{ij}}
+ (1-w_{ij})\log\frac{1-w_{ij}}{1-q_{ij}}.
$$

## Estimating Equation

The implementation selects `n_neighbors` with dense exact search by default (`nn_method='auto'` resolves to `exact`) or internal NNDescent when requested, symmetrizes fuzzy memberships, then applies attractive and sampled repulsive force updates. These updates are not the exact gradient of the standard cross-entropy above, so do not interpret this implementation as numerically equivalent to umap-learn.

## Parameters

`n_neighbors`, `n_components`, `metric`, `min_dist`, `spread`, `n_epochs`, `learning_rate`, `init`, `negative_sample_rate`, `repulsion_strength`, `random_state`, and `device`.

## A small CPU example

<!-- learner-example: umap -->
```python
import numpy as np
from statgpu.unsupervised import UMAP

rng = np.random.default_rng(0)
X = np.vstack([rng.normal(-2, 0.4, (20, 3)), rng.normal(2, 0.4, (20, 3))])
model = UMAP(n_neighbors=5, n_epochs=20, init="random", nn_method="exact", random_state=0, device="cpu")
embedding = model.fit_transform(X)
print(embedding.shape, model.n_epochs_)
```

The embedding has shape `(40, 2)` and covers only the fitted rows. The short run illustrates the API, not optimized visualization quality. There is no new-data `transform`; retain row identifiers when matching points back to observations.

For a supported GPU installation, construct a new estimator with `device="cuda"` (CuPy) or `device="torch"` (Torch CUDA). Arrays generally stay on that backend; see the [API reference](api-reference.md#umap) for output ownership and host-side work. An unavailable explicit GPU raises an error.

## Backend and Host Boundary

Distance evaluation, graph weights and embedding arrays use the selected NumPy, CuPy or Torch backend. On NumPy 2, an array-dispatch limitation can additionally route CPU negative sampling through Torch CPU when Torch is installed. The current fuzzy-union graph assembly is intentionally a documented host boundary: its O(n*k) edge indices and weights are copied to host memory, assembled with SciPy sparse COO/CSR operations, and copied back to the selected backend. This is not a silent CPU fallback for optimization, but it is not yet a device-native sparse-graph path. Exact neighbors also require O(n^2) dense distance memory; `nn_method='nndescent'` avoids that distance matrix where its approximate-neighbor path works, but currently fails on CPU with NumPy 2.

## Approximation and interpretation

`nn_method='exact'` exhaustively searches dense Euclidean neighbors using float32 distance arithmetic; rounding can affect near-ties. `nn_method='nndescent'` is approximate and backend-aware. Both modes use the SciPy host-side fuzzy-union boundary described above; graph assembly therefore needs host memory.

## Outputs

`embedding_`, `graph_`, `n_epochs_`, and `n_features_in_`.

## FAQ

Sparse input, non-Euclidean metrics, and new-data `transform` are not supported. Approximate neighbors are available through `nn_method='nndescent'`; graph assembly still requires SciPy and host memory.


## Current restrictions

- `nn_method="auto"` always selects exact search. CPU `nn_method="nndescent"` currently fails with NumPy 2; use exact search there.
- CPU `n_components=1` currently fails during force accumulation; use at least two dimensions.
- Each epoch draws `n_samples * negative_sample_rate` uniform source/target pairs for repulsion, rather than sampling separately for every attractive edge.
- The sparse `init="spectral"` path can retain a constant graph eigenvector instead of one of the requested nontrivial directions. The eigensolver starting vector is not controlled by `random_state`; use `init="random"` when seeded initialization matters.
- Large common feature offsets can destroy small separations during float32 conversion and expanded-distance evaluation. Center features in float64 before fitting; finite output alone does not establish that the neighbor graph is reliable.
- Supply finite numeric controls; non-finite learning rates are not reliably rejected and can produce invalid embeddings.

The affinity curve is $q_{ij}=(1+a\,r_{ij}^{b})^{-1}$ with $r_{ij}=\|y_i-y_j\|^2$. The current attractive contribution is proportional to $w_{ij}q_{ij}(y_i-y_j)$ and the sampled repulsive contribution to $q_{ij}^2(y_i-y_j)$. The standard cross-entropy gradient has additional distance-dependent factors. Treat the output as an approximate neighborhood layout and check its usefulness directly.

## Complete API reference

Constructor defaults, all public methods, output shapes, and restrictions are listed in the [UMAP API reference](api-reference.md#umap).

## References

- McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. *arXiv:1802.03426*.
- umap-learn developers. UMAP API documentation.
