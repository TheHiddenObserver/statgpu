# Unsupervised Learning

> Language: English
> Last updated: 2026-05-02
> This page: unsupervised model overview
> Switch: [Chinese](../../models/unsupervised.md)

## Overview

`statgpu.unsupervised` contains estimators for dimensionality reduction, clustering, density-based grouping, mixture modeling, and non-negative matrix factorization. The API follows the familiar `fit`, `transform`, `predict`, `fit_predict`, and `score` style where those operations make sense for the model.

The detailed pages under [docs/en/unsupervised/](../unsupervised/README.md) describe each estimator's objective function, estimating procedure, device behavior, outputs, limitations, and validation approach.

## Model Summary

| Estimator | Main Use | Core Criterion |
|---|---|---|
| [PCA](../unsupervised/pca.md) | Linear dimensionality reduction | Maximize projected variance / minimize rank-k reconstruction error |
| [KMeans](../unsupervised/kmeans.md) | Prototype-based clustering | Minimize squared Euclidean inertia |
| [DBSCAN](../unsupervised/dbscan.md) | Density-based clustering with noise | Density reachability and connected components |
| [GaussianMixture](../unsupervised/gaussian-mixture.md) | Probabilistic soft clustering | Maximize diagonal Gaussian mixture log likelihood with EM |
| [NMF](../unsupervised/nmf.md) | Parts-based non-negative factorization | Minimize Frobenius reconstruction error under non-negativity |
| [AgglomerativeClustering](../unsupervised/agglomerative-clustering.md) | Hierarchical clustering | Greedy single-linkage merges |

## Device Behavior

Most unsupervised estimators expose `device="auto"`, `"cpu"`, `"cuda"`, and `"torch"` following the project-wide device rules. Explicit GPU devices must either run on that backend or raise a clear error; they should not silently fall back to CPU. Some algorithms have narrower support, so check the per-model page before relying on a GPU path.

## Notes

Unsupervised estimators do not expose statistical inference fields such as standard errors, p-values, confidence intervals, AIC, or BIC unless the model naturally defines them. For these models, documentation focuses on algorithmic objective, exact versus iterative behavior, device support, and output semantics.

For detailed API behavior and model-specific caveats, continue to the per-model pages linked above.
