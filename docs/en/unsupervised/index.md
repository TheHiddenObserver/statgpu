# Unsupervised learning

> Switch: [简体中文](../../cn/unsupervised/)

GPU-aware decomposition, clustering, manifold, and matrix factorization APIs.

## Solver lookup

| Model | Public control | Supported values |
|---|---|---|
| [PCA](pca.md#solver-support) | `svd_solver` | `auto`, `full`, `covariance`, `randomized` |
| [NMF](nmf.md#solver-support) | `solver` | `mu` only |

Other pages describe their fixed estimation algorithm even when no public `solver` parameter is exposed.

## Decomposition

- [PCA](pca.md)
- [Incremental PCA](incremental-pca.md)
- [Truncated SVD](truncated-svd.md)

## Matrix factorization

- [Non-negative matrix factorization](nmf.md)
- [Mini-batch NMF](minibatch-nmf.md)

## Clustering

- [K-means](kmeans.md)
- [Mini-batch K-means](minibatch-kmeans.md)
- [Agglomerative clustering](agglomerative-clustering.md)
- [DBSCAN](dbscan.md)
- [Gaussian mixtures](gaussian-mixture.md)

## Manifold learning

- [t-SNE](tsne.md)
- [UMAP](umap.md)

## Overview

- [Full unsupervised inventory](README.md)
