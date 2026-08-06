# UMAP

> Language: English
> Last updated: 2026-07-23
> Path: `statgpu.unsupervised.UMAP`

## Overview

`UMAP` builds a fuzzy neighbor graph in the input space and optimizes a low-dimensional embedding. It supports dense exact Euclidean neighbors and an internal NNDescent neighbor-search option.

## Backend and Host Boundary

Distance evaluation, neighbor search, membership weights, embedding optimization, and negative sampling use the selected NumPy, CuPy, or Torch backend. The current fuzzy-union graph assembly is intentionally a documented host boundary: its O(n*k) edge indices and weights are copied to host memory, assembled with SciPy sparse COO/CSR operations, and copied back to the selected backend. This is not a silent CPU fallback for optimization, but it is not yet a device-native sparse-graph path. Exact neighbors also require O(n^2) dense distance memory; use `nn_method='nndescent'` to avoid that distance matrix when its approximate-neighbor trade-off is acceptable.

## Path

Import from `statgpu.unsupervised`:

```python
from statgpu.unsupervised import UMAP
```

## Objective Function / Loss Function

UMAP optimizes a fuzzy-set cross-entropy between high-dimensional graph weights `w_ij` and low-dimensional affinities `q_ij`:

$$
\sum_{i,j} w_{ij}\log\frac{w_{ij}}{q_{ij}}
+ (1-w_{ij})\log\frac{1-w_{ij}}{1-q_{ij}}.
$$

## Estimating Equation

The implementation selects `n_neighbors` with dense exact search by default (`nn_method='auto'` resolves to `exact`) or internal NNDescent when requested, symmetrizes fuzzy memberships, then performs gradient steps on the embedding.

## Parameters

`n_neighbors`, `n_components`, `metric`, `min_dist`, `spread`, `n_epochs`, `learning_rate`, `init`, `negative_sample_rate`, `repulsion_strength`, `random_state`, and `device`.

## CPU+GPU Examples

```python
from statgpu.unsupervised import UMAP

embedding = UMAP(n_neighbors=15, device="cpu").fit_transform(X)
embedding_gpu = UMAP(n_neighbors=15, device="cuda").fit_transform(X_gpu)
```

## Strict/Approx Difference

`nn_method='exact'` is exact for dense Euclidean neighbor search. `nn_method='nndescent'` is approximate and backend-aware. Both modes use the SciPy host-side fuzzy-union boundary described above; a fully device-native sparse graph pipeline is planned but not yet implemented.

## Outputs

`embedding_`, `graph_`, `n_epochs_`, and `n_features_in_`.

## FAQ

Sparse input, non-Euclidean metrics, and new-data `transform` are not supported. Approximate neighbors are available through `nn_method='nndescent'`; graph assembly still requires SciPy and host memory.

## External Validation

Tests: `dev/tests/test_unsupervised_umap.py`.
Benchmark: `dev/benchmarks/benchmark_unsupervised_phase3.py`.
Baseline: `umap-learn`, plus cuML UMAP if available remotely.

## References

- McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. *arXiv:1802.03426*.
- umap-learn developers. UMAP API documentation.
