# Unsupervised Learning

> Language: English
> Last updated: 2026-10-04
> This page: Unsupervised learning index
> Switch: [Chinese](../../cn/unsupervised/README.md)

## Overview

`statgpu.unsupervised` contains sklearn-style unsupervised estimators with explicit CPU, CuPy/CUDA, and Torch CUDA device behavior. This directory documents each estimator separately so the loss function, estimating algorithm, backend behavior, and validation evidence are visible without compressing all models into one page.

## Choose a starting point

| Your question | Start with | What to inspect |
|---|---|---|
| Can I summarize variation with fewer numeric features? | [PCA](pca.md); [TruncatedSVD](truncated-svd.md) when centering is not wanted | Component loadings, explained variation and reconstruction, not just the first two coordinates |
| Can I describe non-negative data as additive parts? | [NMF](nmf.md) | Non-negative factors and reconstruction error; mean-centering can violate the required non-negativity |
| Can I group observations into a chosen number of compact groups? | [KMeans](kmeans.md) | Centers, labels and within-cluster sum of squares (inertia); labels are arbitrary identifiers, not ordered outcomes |
| Do I need probabilistic membership in overlapping groups? | [GaussianMixture](gaussian-mixture.md) | Membership probabilities and the fitted covariance structure; compare component counts under the same data and scoring setup |
| Do I want density-connected groups and possible noise points? | [DBSCAN](dbscan.md) | Neighborhood scale, minimum density and noise assignments; feature units affect Euclidean distances |
| Do I need a hierarchy rather than one fixed partition? | [AgglomerativeClustering](agglomerative-clustering.md) | Merge structure and the selected linkage; inspect how the cut changes the groups |
| Do I want a low-dimensional visualization of neighborhoods? | [UMAP](umap.md) or [TSNE](tsne.md) | Stability across seeds and settings; visual separation alone is not evidence of distinct populations |
| Must I process data in batches? | [IncrementalPCA](incremental-pca.md), [MiniBatchKMeans](minibatch-kmeans.md), or [MiniBatchNMF](minibatch-nmf.md) | The corresponding model's batch-size, initialization and `partial_fit` requirements |

These implementations target dense inputs. In particular, the current TruncatedSVD is not a sparse-text pipeline. UMAP provides `nn_method="exact"` and `"nndescent"` (approximate) neighbor search, with `"auto"` choosing a path; graph assembly still uses host-side SciPy. TSNE uses exact dense distances. Neither visualization estimator supports new-data `transform`. Read the selected model's limits before choosing it for a large dataset.

## A practical first pass

1. Decide what each row represents and which numeric features and units matter. Handle missing/non-finite values before fitting. Distance-based results can change substantially with feature scaling; choose scaling for the question rather than treating it as a universal default.
2. Prepare a small representative numeric sample and follow the linked model page, selecting `device="cpu"` for the first fit. Check its input shape, fitted attributes and supported prediction/transform methods: the sklearn-style naming does not mean every estimator supports every method.
3. Inspect the result in the third column above and compare reasonable parameter settings. When evaluating on held-out observations, fit any preprocessing and model selection on the training portion only. Do not compare unrelated models by assuming their `score()` values mean the same thing.
4. Move to batches or a supported GPU backend only after the small workflow is understood. Check the model's memory limits and measure the complete workload, including required transfers; a GPU does not guarantee a faster small fit.

The pages below contain model-specific examples, parameters, output contracts and failure modes. This index helps choose the next page rather than replacing those guides.

## Try a small CPU workflow

This PCA example compresses three numeric features into two. The third feature is almost a copy of the first; the units are deliberately comparable, so no additional scaling is used. Fit on the training rows, then transform new rows with that same fitted model.

<!-- learner-example: unsupervised-pca -->
```python
import numpy as np
from statgpu.unsupervised import PCA

rng = np.random.default_rng(12)
X = rng.normal(size=(120, 3))
X[:, 2] = X[:, 0] + 0.05 * rng.normal(size=120)
X_train, X_test = X[:90], X[90:]
pca = PCA(n_components=2, svd_solver="full", device="cpu")
Z_train = pca.fit_transform(X_train)
Z_test = pca.transform(X_test)
X_reconstructed = pca.inverse_transform(Z_test)
print("reduced shapes:", Z_train.shape, Z_test.shape)
print("explained fraction:", round(float(pca.explained_variance_ratio_.sum()), 4))
print("held-out reconstruction MSE:", round(float(np.mean((X_test - X_reconstructed) ** 2)), 4))
```

The shapes are `(90, 2)` and `(30, 2)`. The retained variance fraction is close to 1 and the held-out reconstruction error is small because these simulated data are nearly two-dimensional. This is compression, not classification or a test of scientific importance. See [PCA](pca.md) for component interpretation, sign ambiguity, scaling and solver choices.

## Estimators

- [PCA](pca.md): exact or randomized principal component analysis.
- [KMeans](kmeans.md): Lloyd clustering with random or greedy k-means++ initialization.
- [DBSCAN](dbscan.md): dense Euclidean density clustering with optional statgpu-owned Cython CPU acceleration.
- [GaussianMixture](gaussian-mixture.md): Gaussian mixture fitted by EM with diagonal, spherical, tied, or full covariance.
- [NMF](nmf.md): non-negative matrix factorization with multiplicative updates and Frobenius loss.
- [AgglomerativeClustering](agglomerative-clustering.md): exact dense single, complete, average, or ward linkage clustering.
- [TruncatedSVD](truncated-svd.md): dense uncentered truncated SVD for low-rank projection.
- [MiniBatchKMeans](minibatch-kmeans.md): mini-batch Euclidean K-Means for larger dense datasets.
- [IncrementalPCA](incremental-pca.md): dense batch-wise principal component analysis.
- [MiniBatchNMF](minibatch-nmf.md): dense mini-batch non-negative matrix factorization.
- [UMAP](umap.md): dense Euclidean UMAP with exact or approximate neighbor search and host-side SciPy graph assembly.
- [TSNE](tsne.md): dense exact Euclidean t-SNE v1.

## Support Matrix

| Estimator | CPU | CuPy/CUDA | Torch CUDA | Main objective or criterion |
|---|---|---|---|---|
| `PCA` | yes | yes | yes | Maximum variance / rank-k reconstruction loss |
| `KMeans` | yes | yes | yes | Within-cluster sum of squared Euclidean distances |
| `DBSCAN` | yes | yes | yes | Density reachability and connected components |
| `GaussianMixture` | yes | yes | yes | Gaussian mixture log likelihood |
| `NMF` | yes | yes | yes | Frobenius reconstruction loss under non-negativity |
| `AgglomerativeClustering` | yes | yes | yes | Hierarchical linkage merge criterion |
| `TruncatedSVD` | yes | yes | yes | Uncentered low-rank reconstruction |
| `MiniBatchKMeans` | yes | yes | yes | Within-cluster sum of squares for mini-batch clustering |
| `IncrementalPCA` | yes | yes | yes | Batch-wise centered low-rank reconstruction |
| `MiniBatchNMF` | yes | yes | yes | Mini-batch Frobenius reconstruction loss |
| `UMAP` | yes | yes, host SciPy graph assembly | yes, host SciPy graph assembly | Fuzzy graph cross-entropy |
| `TSNE` | yes | yes | yes | KL divergence between high- and low-dimensional affinities |

Explicit `device="cuda"` and `device="torch"` do not silently fall back to CPU. Unsupported GPU paths raise clear errors.

## Validation and interpretation limits

These estimators have unit tests and, where applicable, numerical comparisons with scikit-learn, cuML, umap-learn or openTSNE. GPU checks also cover device behavior and CPU/GPU consistency. Such comparisons apply to the recorded settings; they do not guarantee the same accuracy or speed for every dataset, version or device. Use the model pages for relevant assumptions and validation context, and benchmark your own workload when performance matters.

External packages serve as validation or benchmark references; production estimator code remains statgpu-owned.
