# Models Overview

> Language: English  
> Last updated: 2026-05-08
> This page: Model index  
> Switch: [Chinese](../../models/README.md)

Language switch: [Chinese](../../models/README.md)

## Linear and GLM Models

- [LinearRegression](linear-regression.md)
- [GeneralizedLinearModel and Penalized GLM](generalized-linear-model.md)
- [PoissonRegression](poisson-regression.md)
- [Ridge](ridge.md)
- [Lasso](lasso.md)
- [ElasticNet](elastic-net.md)
- [LogisticRegression](logistic-regression.md)
- [Ordered Generalized Linear Models (Logit/Probit)](ordered.md)

## Survival

- [CoxPH](coxph.md)

## Feature Selection

- [Knockoff](knockoff.md)

## Nonparametric

- [Nonparametric](nonparametric.md)

## Unsupervised Learning

- [Unsupervised learning compatibility entry](unsupervised.md)
- [Detailed unsupervised docs](../unsupervised/README.md)
- Dimensionality and factorization: [PCA](../unsupervised/pca.md), [TruncatedSVD](../unsupervised/truncated-svd.md), [IncrementalPCA](../unsupervised/incremental-pca.md), [NMF](../unsupervised/nmf.md), [MiniBatchNMF](../unsupervised/minibatch-nmf.md)
- Clustering and mixtures: [KMeans](../unsupervised/kmeans.md), [MiniBatchKMeans](../unsupervised/minibatch-kmeans.md), [DBSCAN](../unsupervised/dbscan.md), [GaussianMixture](../unsupervised/gaussian-mixture.md), [AgglomerativeClustering](../unsupervised/agglomerative-clustering.md)
- Manifold embeddings: [UMAP](../unsupervised/umap.md), [TSNE](../unsupervised/tsne.md)

## Current Coverage Notes

- Device support is model-specific. Explicit `device="cuda"` and `device="torch"` must raise when unavailable or unsupported; they must not silently fall back to CPU.
- GPU memory cleanup support is model-specific and documented on pages where it affects behavior.
- `GeneralizedLinearModel` and typed penalized GLMs are documented in [GeneralizedLinearModel and Penalized GLM](generalized-linear-model.md).
- `PoissonRegression` is documented separately as the ordinary Poisson GLM estimator.
- `statgpu.unsupervised` includes dimensionality reduction, clustering, mixture, factorization, and manifold estimators. Most expose CPU/CuPy/Torch paths; `AgglomerativeClustering(single/complete/average/ward)` is CPU-only. See [Detailed unsupervised docs](../unsupervised/README.md) for per-model objectives, estimating procedures, device notes, examples, and external validation artifacts.
- Inference-rich models:
  - `LinearRegression`: classical + `HC0/HC1/HC2/HC3/HAC`
  - `Ridge`: classical + `HC0/HC1/HC2/HC3/HAC`
  - `Lasso`: CPU/GPU OLS-style inference + bootstrap
  - `LogisticRegression`: classical + `HC0/HC1/HC2/HC3/HAC`
- `CoxPH` supports Breslow/Efron ties and CPU/GPU fitting paths.
- `OrderedLogitRegression` / `OrderedProbitRegression` support CPU/CuPy/Torch backends with cross-backend precision fix (coef diff < 1e-2).
- `CoxPH` delayed entry (`entry`) support:
  - `entry + breslow`: CPU/CUDA/Torch
  - `entry + efron`: CPU/CUDA/Torch
- Feature selection:
  - `Knockoff`: fixed-X/model-X unified API + selector wrappers
- `LassoCV` is implemented and trainable.
- Exported CV classes status:
  - `RidgeCV`, `LogisticRegressionCV`, and `CoxPHCV` are implemented and trainable.
  - Current `CoxPHCV` boundary: on GPU paths, `entry` currently supports only `ties='breslow'`; `cluster`-robust CV is not yet supported and raises `NotImplementedError`.
