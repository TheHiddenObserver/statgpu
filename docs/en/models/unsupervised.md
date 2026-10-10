# Unsupervised Learning

> Language: English
> Last updated: 2026-10-05
> This page: unsupervised model overview
> Switch: [Chinese](../../cn/models/unsupervised.md)

## Overview

`statgpu.unsupervised` contains estimators for dimensionality reduction, clustering, density-based grouping, mixture modeling, non-negative matrix factorization, and manifold embedding. UMAP can use internal approximate nearest-neighbor search; that search is not a separately exported estimator. The API follows the familiar `fit`, `transform`, `predict`, `fit_predict`, and `score` style where those operations make sense for the model.

## Model Summary

| Estimator | Main Use | Core Criterion |
|---|---|---|
| [PCA](../unsupervised/pca.md) | Linear dimensionality reduction | Maximize projected variance / minimize rank-k reconstruction error |
| [KMeans](../unsupervised/kmeans.md) | Prototype-based clustering | Minimize squared Euclidean inertia |
| [DBSCAN](../unsupervised/dbscan.md) | Density-based clustering with noise | Density reachability and connected components |
| [GaussianMixture](../unsupervised/gaussian-mixture.md) | Probabilistic soft clustering | Maximize Gaussian mixture log likelihood with EM |
| [NMF](../unsupervised/nmf.md) | Parts-based non-negative factorization | Minimize Frobenius reconstruction error under non-negativity |
| [AgglomerativeClustering](../unsupervised/agglomerative-clustering.md) | Hierarchical clustering | Greedy linkage merges |
| [TruncatedSVD](../unsupervised/truncated-svd.md) | Uncentered low-rank projection | Minimize rank-k dense reconstruction error |
| [MiniBatchKMeans](../unsupervised/minibatch-kmeans.md) | Larger-scale prototype clustering | Approximate inertia minimization with mini-batch updates |
| [IncrementalPCA](../unsupervised/incremental-pca.md) | Batch-wise linear dimensionality reduction | Approximate centered rank-k reconstruction |
| [MiniBatchNMF](../unsupervised/minibatch-nmf.md) | Larger-scale non-negative factorization | Mini-batch Frobenius reconstruction loss |
| [UMAP](../unsupervised/umap.md) | Manifold embedding | Approximate neighborhood-layout forces |
| [TSNE](../unsupervised/tsne.md) | Manifold visualization | KL divergence between affinity distributions |

## Device Behavior

Most unsupervised estimators expose `device="auto"`, `"cpu"`, `"cuda"`, and `"torch"`. If an explicitly requested GPU backend is unavailable, fitting raises an error. Some algorithms still perform CPU work or return NumPy arrays; check the per-model page for supported operations, output placement, and limitations before relying on a GPU path.

## Input validation

Observation matrices are checked for NaN/Inf before SVD, eigendecomposition, distance computation, or iterative updates and raise a public validation error. This does not cover every constructor setting: numeric controls and explicit MiniBatchKMeans initial centers also need to be finite, but are not all checked reliably. See the [shared input rules](../unsupervised/api-reference.md#shared-inputs-devices-and-state) and [initial-center precautions](../unsupervised/minibatch-kmeans.md#numerical-and-lifecycle-cautions).

## Notes

Unsupervised estimators do not expose statistical inference fields such as standard errors, p-values, confidence intervals, AIC, or BIC unless the model naturally defines them.

For detailed API behavior and model-specific caveats, continue to the per-model pages linked above.

The twelve public estimator classes and complete constructor/method contracts are listed in the [unsupervised index](../unsupervised/README.md) and [API reference](../unsupervised/api-reference.md). NNDescent is an internal neighbor-search implementation used by UMAP, not an additional export from `statgpu.unsupervised`.
