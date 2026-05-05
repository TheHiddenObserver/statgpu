# UMAP

> Language: English
> Last updated: 2026-05-04
> Path: `statgpu.unsupervised.UMAP`

## Overview

`UMAP` builds a fuzzy neighbor graph in the input space and optimizes a low-dimensional embedding. Phase 3A implements a dense exact Euclidean path.

## Objective Function / Loss Function

UMAP optimizes a fuzzy-set cross-entropy between high-dimensional graph weights `w_ij` and low-dimensional affinities `q_ij`:

$$
\sum_{i,j} w_{ij}\log\frac{w_{ij}}{q_{ij}}
+ (1-w_{ij})\log\frac{1-w_{ij}}{1-q_{ij}}.
$$

## Estimating Equation

The implementation computes exact pairwise distances, selects the `n_neighbors` nearest neighbors, symmetrizes fuzzy memberships, then performs gradient steps on the embedding.

## Parameters

`n_neighbors`, `n_components`, `metric`, `min_dist`, `spread`, `n_epochs`, `learning_rate`, `init`, `negative_sample_rate`, `repulsion_strength`, `random_state`, and `device`.

## CPU+GPU Examples

```python
from statgpu.unsupervised import UMAP

embedding = UMAP(n_neighbors=15, device="cpu").fit_transform(X)
embedding_gpu = UMAP(n_neighbors=15, device="cuda").fit_transform(X_gpu)
```

## Strict/Approx Difference

This v1 path is exact for dense Euclidean neighbor search but simplified relative to `umap-learn`: it does not implement NNDescent or the full sparse graph pipeline.

## Outputs

`embedding_`, `graph_`, `n_epochs_`, and `n_features_in_`.

## FAQ

Sparse input, non-Euclidean metrics, approximate neighbors, and new-data `transform` are not supported in Phase 3A.

## External Validation

Tests: `dev/tests/test_unsupervised_umap.py`.
Benchmark: `dev/benchmarks/benchmark_unsupervised_phase3.py`.
Baseline: `umap-learn`, plus cuML UMAP if available remotely.

## References

- McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. *arXiv:1802.03426*.
- umap-learn developers. UMAP API documentation.
