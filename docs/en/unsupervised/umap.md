# UMAP

> Language: English
> Last updated: 2026-10-06
> Switch: [Chinese](../../cn/unsupervised/umap.md)
> Path: `statgpu.unsupervised.UMAP`

## Overview

`UMAP` builds a fuzzy neighbor graph in the input space and optimizes a low-dimensional embedding. It supports dense exact Euclidean neighbors and an internal NNDescent neighbor-search option.

## When to use it

Use UMAP to explore local neighborhood structure visually. Compare several seeds, neighborhood sizes and `min_dist` settings. Distances between separate islands and apparent cluster sizes are not direct estimates of population separation or prevalence.

Attraction brings connected neighbors together; repulsion spreads points apart. The graph weighting and force updates differ from umap-learn, so matching parameter values does not imply equivalent embeddings. Treat the result as an approximate neighborhood layout; the [advanced equations](#advanced-graph-and-layout-equations) describe the differences.

## Path

Import from `statgpu.unsupervised`:

```python
from statgpu.unsupervised import UMAP
```

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

## Parameters

- `n_neighbors` controls neighborhood size; compare local detail with broader structure as you vary it.
- `min_dist` and `spread` control the low-dimensional attraction curve. They do not repair an unreliable input neighbor graph.
- `n_components` controls output dimensions; use at least two on CPU. `metric` currently supports Euclidean distance only.
- `init="random"` with `random_state` provides seeded initialization; the spectral path has the limitation below.
- `n_epochs` and `learning_rate` control optimization; `negative_sample_rate` and `repulsion_strength` control sampled repulsion. Supply finite numeric controls.
- `nn_method="auto"` selects exact search. `nn_method="nndescent"` requests approximate neighbors but currently fails on CPU with NumPy 2. `device` selects the numerical backend; GPU fits still need host memory.

All constructor defaults and approximate-search controls are in the [API reference](api-reference.md#umap).

## Backend and Host Boundary

Distance evaluation, graph weights and embedding arrays use the selected NumPy, CuPy or Torch backend. On NumPy 2, an array-dispatch limitation can additionally route CPU negative sampling through Torch CPU when Torch is installed. Fuzzy-union graph assembly copies O(n*k) edge indices and weights to host memory, assembles them with SciPy sparse COO/CSR operations, and copies the result back to the selected backend. Spectral initialization and attraction-curve fitting also use host SciPy. GPU fitting therefore still requires CPU computation and host memory. Exact neighbors also require O(n^2) dense distance memory; `nn_method='nndescent'` avoids that distance matrix where its approximate-neighbor path works, but currently fails on CPU with NumPy 2.

## Approximation and interpretation

`nn_method='exact'` exhaustively searches dense Euclidean neighbors using float32 distance arithmetic; rounding can affect near-ties. `nn_method='nndescent'` is approximate and backend-aware. Both modes use the SciPy host-side fuzzy-union boundary described above; graph assembly therefore needs host memory.

## Outputs

`embedding_`, `graph_`, `n_epochs_`, and `n_features_in_`.

## FAQ

Sparse input, non-Euclidean metrics, and new-data `transform` are not supported. Approximate neighbors are available through `nn_method='nndescent'`; graph assembly still requires SciPy and host memory.


## Current restrictions

- `nn_method="auto"` always selects exact search. CPU `nn_method="nndescent"` currently fails with NumPy 2; use exact search there.
- CPU `n_components=1` currently fails during force accumulation; use at least two dimensions.
- The sparse `init="spectral"` path can retain a constant graph eigenvector instead of one of the requested nontrivial directions. The eigensolver starting vector is not controlled by `random_state`; use `init="random"` when seeded initialization matters.
- Large common feature offsets can destroy small separations during float32 conversion and expanded-distance evaluation. Center features in float64 before fitting; finite output alone does not establish that the neighbor graph is reliable.
- Very large pairwise distances can cause exact search to select an observation as its own neighbor, leaving too few distinct neighbors after self-edges are removed. Center in float64, then divide every feature by one common positive scale computed from the training data so coordinates have moderate magnitudes. Reuse the offset and scale for comparable data; a common scale preserves Euclidean neighbor ordering, whereas separate per-feature scales change the metric. Centering alone does not address this restriction.
- Supply finite numeric controls; non-finite learning rates are not reliably rejected and can produce invalid embeddings.

## Advanced: graph and layout equations

### Objective Function / Loss Function

The standard UMAP reference objective is fuzzy-set cross-entropy between high-dimensional graph weights `w_ij` and low-dimensional affinities `q_ij`:

$$
\sum_{i<j}\left[
w_{ij}\log\frac{w_{ij}}{q_{ij}}
+ (1-w_{ij})\log\frac{1-w_{ij}}{1-q_{ij}}
\right].
$$

The sum is over distinct unordered observation pairs; self-pairs are excluded. Terms with zero numerator weight contribute zero by continuity. Here $q_{ij}=(1+a\|y_i-y_j\|^{2b})^{-1}$ with positive curve parameters $a,b$ determined from `min_dist` and `spread`. This is a reference objective, not a claim that the current force updates minimize it exactly.

### Estimating Equation

The implementation selects `n_neighbors` with dense exact search by default (`nn_method='auto'` resolves to `exact`) or internal NNDescent when requested, symmetrizes fuzzy memberships, then applies attractive and sampled repulsive force updates. These updates are not the exact gradient of the standard cross-entropy above, so do not interpret this implementation as numerically equivalent to umap-learn.

#### Graph weights

For the sorted distances $d_{i1},\ldots,d_{ik}$ to the selected neighbors, this implementation uses

$$
\rho_i=d_{i1},\qquad
\sigma_i=\max\left(\frac{1}{k}\sum_{j=1}^{k}\max(d_{ij}-\rho_i,0),10^{-12}\right),
\qquad
v_{ij}=\exp\left(-\frac{\max(d_{ij}-\rho_i,0)}{\sigma_i}\right).
$$

Unselected directed edges have weight zero. The symmetric graph uses the fuzzy union $w_{ij}=v_{ij}+v_{ji}-v_{ij}v_{ji}$ and removes self-edges. A duplicate neighbor can make $\rho_i=0$.

This is a mean-excess-distance bandwidth rule. It does not solve the local membership-sum calibration used by umap-learn's `smooth_knn_dist`, so graph weights can differ even with the same neighbors and before optimization begins. `min_dist` and `spread` control the low-dimensional attraction curve; they do not change this high-dimensional graph calculation. Changing them is not a way to repair a poor neighbor graph.

### Layout force updates

Each epoch draws `n_samples * negative_sample_rate` uniform source/target pairs for repulsion, rather than sampling separately for every attractive edge.

The affinity curve is $q_{ij}=(1+a\,r_{ij}^{b})^{-1}$ with $r_{ij}=\|y_i-y_j\|^2$. The current attractive contribution is proportional to $w_{ij}q_{ij}(y_i-y_j)$ and the sampled repulsive contribution to $q_{ij}^2(y_i-y_j)$. The standard cross-entropy gradient has additional distance-dependent factors. Treat the output as an approximate neighborhood layout and check its usefulness directly.

## Complete API reference

Constructor defaults, all public methods, output shapes, and restrictions are listed in the [UMAP API reference](api-reference.md#umap).

## References

- McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. *arXiv:1802.03426*.
- umap-learn developers. UMAP API documentation.

- umap-learn developers. [`smooth_knn_dist` source reference](https://umap-learn.readthedocs.io/en/latest/_modules/umap/umap_.html#smooth_knn_dist).
