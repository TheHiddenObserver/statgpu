# Unsupervised estimator API reference

> Language: English
> Last updated: 2026-10-05
> This page: Complete API reference
> Switch: [Chinese](../../cn/unsupervised/api-reference.md)

This reference lists the constructor defaults, public model methods, return values, fitted outputs, and important restrictions of the twelve estimators exported by `statgpu.unsupervised`. Use the [index](README.md) to choose a model and its individual guide for intuition, objective equations and statistical references.

## Shared inputs, devices, and state

Import any class with `from statgpu.unsupervised import ClassName`. Here `n` is the number of training rows, `p` the number of input features, `m` the number of rows in a later call, and `k` the requested/selected number of components or clusters. `X` is a nonempty finite dense numeric matrix `(n,p)` or `(m,p)`. These estimators do not provide a formula/dataframe-design interface; encode categories and handle missing values before calling them. Subsequent inputs must preserve the feature order and width. `y=None` is an unused sklearn-compatible argument, not a supervised target.

- `device="cpu"` uses NumPy; `"cuda"` requests CuPy CUDA; `"torch"` requests Torch CUDA. An explicit unavailable GPU raises instead of silently choosing CPU. `"auto"` follows global device configuration and available backends for most classes; AgglomerativeClustering instead keeps its CPU path even when global configuration selects a GPU. Use `"cpu"` for reproducible CPU examples. See [device and memory](../guides/device-and-memory.md).
- Numeric calculations generally use float64; UMAP neighbor search and DBSCAN GPU distance calculations use float32 internally. Fitted numeric arrays and method arrays normally remain on the selected backend. AgglomerativeClustering publishes NumPy labels/tree arrays even after GPU fitting; UMAP publishes a graph tuple. Integer labels are identifiers, not continuous predictions.
- Use `np.asarray(a)` for NumPy arrays, `cupy.asnumpy(a)` for CuPy, or `a.detach().cpu().numpy()` for Torch to obtain a CPU reporting array. These conversions do not always make an independent copy: NumPy arrays and CPU Torch tensors can share storage with the result. Add `.copy()` to the resulting NumPy array before editing it independently. GPU conversion can transfer/synchronize data. Scores and scalar fit diagnostics are host numbers. Numeric hyperparameters must also be finite; range checks do not reliably reject every NaN/Inf setting.
- `fit(...)` and supported `partial_fit(...)` return the estimator itself. Post-fit transforms, predictions and scores require a successful fit. Ordinary `fit` starts a new model; only the three documented `partial_fit` APIs accumulate batches. After changing settings with `set_params(...)`, fit again before using results. A failed refit is not a successful replacement: old or partially updated attributes may remain on the instance. Use a fresh estimator and validate the new result after an error.
- Shared `get_params(deep=True)` returns a configuration dictionary; `set_params(**params)` returns `self`, validates parameter names, and resets fitted state for these classes. The complete inherited signatures, inference-helper restrictions and examples are in [parameter management](../reference/estimator-api.md#parameter-management) and [generic inference helpers](../reference/estimator-api.md#inference-helpers). The inherited `adjust_pvalues`, `combine_pvalues`, `bootstrap_statistic`, and `permutation_test` helpers do not by themselves supply valid cluster/component inference. None of these twelve classes provides a model-specific `summary()` or coefficient standard errors.
- `n_jobs=None` appears in every constructor. It is retained as common estimator configuration; the unsupervised implementations currently do not use it to set kernel/thread parallelism. Do not interpret `n_jobs=-1` as a measured speedup or enforced thread count here.

Array-returning methods do not universally make defensive copies. `KMeans`, `MiniBatchKMeans`, `DBSCAN`, and `AgglomerativeClustering` return their `labels_` object from `fit_predict`; UMAP and TSNE return their `embedding_` object from `fit_transform`. NMF returns its stored joint-fit factors from `fit_transform`. Treat these arrays and fitted attributes as read-only, or copy them before modifying them (`a.copy()` for NumPy/CuPy, `a.clone()` for Torch).

## Method availability

A method not listed in a model's table is not an additional model capability. In particular, the `predict`/`transform` stubs for clustering or visualization models below are listed explicitly as unsupported. No class in this module exposes a dedicated CV estimator. Select preprocessing, rank, cluster count and other settings using criteria suitable for the scientific question; arbitrary `score` values are not comparable across models.

## PCA

[Model guide](pca.md)

```text
PCA(n_components=None, svd_solver='auto', whiten=False, copy=True, random_state=None, n_oversamples=10, iterated_power=2, device='auto', n_jobs=None)
```

| Parameter | Default | Meaning / accepted values |
|---|---|---|
| `n_components` | `None` | Integer in `[1, min(n,p)]`; `None` keeps `min(n,p)` components. |
| `svd_solver` | `'auto'` | `"auto"`, `"full"`, `"covariance"`, or `"randomized"`; `auto` uses covariance/eigh for `n >= p`, otherwise full SVD. |
| `whiten` | `False` | Whether transformed PCA coordinates are rescaled by the fitted component standard deviations. |
| `copy` | `True` | Compatibility option; input data are not modified, including when `False`. |
| `random_state` | `None` | Integer seed or `None`; controls randomized initialization/approximation. A fixed seed does not promise identical results across backends or library versions. |
| `n_oversamples` | `10` | Nonnegative integer giving extra random projection directions. |
| `iterated_power` | `2` | Nonnegative number of randomized power iterations. |
| `device` | `'auto'` | `"auto"`, `"cpu"`, `"cuda"` (CuPy), or `"torch"` (Torch CUDA); see the shared device rules. |
| `n_jobs` | `None` | Accepted common CPU-job configuration; these implementations do not use it to control their numerical kernels or guarantee a thread count. |

| Method signature | Return and restrictions |
|---|---|
| `fit(X, y=None)` | Fit centered components; requires at least two rows. Returns `self`. |
| `transform(X)` | Return centered component coordinates, shape `(m,k)`; whiten when requested. |
| `fit_transform(X, y=None)` | Fit and return training coordinates `(n,k)`. |
| `inverse_transform(X)` | Input coordinates `(m,k)`; undo whitening, project back and add the training mean. Returns `(m,p)`. |
| `predict(X)` | Alias for `transform(X)`, not a class prediction. |

| Fitted output | Meaning and shape |
|---|---|
| `components_` | Orthonormal row directions `(k,p)`; signs are arbitrary. |
| `mean_` | Training mean `(p,)`. |
| `explained_variance_`, `explained_variance_ratio_`, `singular_values_` | Three vectors `(k,)`: sample variances (divisor `n-1`), fractions of total variance, singular values. |
| `n_components_`, `n_features_in_` | Selected rank `k` and input width `p`. |

Write $W=\mathrm{components\_}$, $Z$ for reduced coordinates and $\bar X=\mathrm{mean\_}$. Without whitening, transform is $(X-\bar X)W^\top$ and inverse transform is $ZW+\bar X$. With whitening, the coordinates are divided by $\sqrt{\mathrm{explained\_variance\_}}$ and that scaling is reversed on reconstruction. Zero-variance retained components make ordinary PCA whitening undefined; avoid retaining them. There is no `partial_fit`, `score`, or probabilistic inference method.

## KMeans

[Model guide](kmeans.md)

```text
KMeans(n_clusters=8, init='k-means++', n_init='auto', max_iter=300, tol=0.0001, random_state=None, device='auto', n_jobs=None)
```

| Parameter | Default | Meaning / accepted values |
|---|---|---|
| `n_clusters` | `8` | Positive integer at most `n`. |
| `init` | `'k-means++'` | `"k-means++"` or `"random"`; explicit center arrays and callable initializers are not supported. |
| `n_init` | `'auto'` | Positive integer or `"auto"`: one run for k-means++, ten for random initialization. |
| `max_iter` | `300` | Positive integer iteration budget; see each model for iterations versus epochs. |
| `tol` | `0.0001` | Nonnegative convergence threshold; the model-specific criterion is described below. |
| `random_state` | `None` | Integer seed or `None`; controls randomized initialization/approximation. A fixed seed does not promise identical results across backends or library versions. |
| `device` | `'auto'` | `"auto"`, `"cpu"`, `"cuda"` (CuPy), or `"torch"` (Torch CUDA); see the shared device rules. |
| `n_jobs` | `None` | Accepted common CPU-job configuration; these implementations do not use it to control their numerical kernels or guarantee a thread count. |

| Method signature | Return and restrictions |
|---|---|
| `fit(X, y=None, sample_weight=None)` | Fit centers, returning `self`; `sample_weight` must be `None`. |
| `fit_predict(X, y=None)` | Fit and return training labels `(n,)`. |
| `transform(X)` | Return Euclidean distances to every center `(m,k)`, not squared distances. |
| `predict(X)` | Return nearest-center labels `(m,)`. |
| `score(X, y=None)` | Return a Python float: negative sum of squared distances to nearest centers; higher is better. |

| Fitted output | Meaning and shape |
|---|---|
| `cluster_centers_` | Centers `(k,p)`. |
| `labels_` | Training integer labels `(n,)`, from `0` to `k-1`. |
| `inertia_` | Training within-cluster squared-distance sum; a float. |
| `n_iter_`, `n_features_in_` | Lloyd iterations in the selected run and input width. |

`tol` bounds the sum of squared center movement, not a relative objective tolerance. `max_iter` applies to each restart. $\mathrm{score}(X)=-\sum_i\min_j\|x_i-c_j\|^2$, where $x_i$ is an observation and $c_j$ a fitted center. Compare scores on the same observations and feature scaling. No `partial_fit` is provided; use MiniBatchKMeans for incremental updates.

## DBSCAN

[Model guide](dbscan.md)

```text
DBSCAN(eps=0.5, min_samples=5, metric='euclidean', algorithm='auto', batch_size=None, device='auto', n_jobs=None)
```

| Parameter | Default | Meaning / accepted values |
|---|---|---|
| `eps` | `0.5` | Positive finite Euclidean neighborhood radius; non-finite values are not reliably rejected. |
| `min_samples` | `5` | Positive integer neighbor count including the observation itself. |
| `metric` | `'euclidean'` | Only `"euclidean"` is supported; other metrics, including precomputed distances, are unsupported. |
| `algorithm` | `'auto'` | `"auto"`, `"brute"`, `"ball_tree"`, or `"kd_tree"`; used by high-dimensional CPU scikit-learn neighbor search, not every backend. |
| `batch_size` | `None` | Positive integer or `None`; bounds GPU distance batches where that path uses batching. `None` chooses an automatic batch size. |
| `device` | `'auto'` | `"auto"`, `"cpu"`, `"cuda"` (CuPy), or `"torch"` (Torch CUDA); see the shared device rules. |
| `n_jobs` | `None` | Accepted common CPU-job configuration; these implementations do not use it to control their numerical kernels or guarantee a thread count. |

| Method signature | Return and restrictions |
|---|---|
| `fit(X, y=None)` | Find density-connected training groups; return `self`. |
| `fit_predict(X, y=None)` | Fit and return training labels `(n,)`. |
| `predict(X)` | Always raises `NotImplementedError`; no new-observation assignment rule is exposed. |

| Fitted output | Meaning and shape |
|---|---|
| `labels_` | Integer training labels `(n,)`; `-1` denotes noise. |
| `core_sample_indices_` | Indices of core training observations `(n_core,)`. |
| `components_` | Core observations `(n_core,p)`, not cluster centers. |
| `n_features_in_` | Training input width `p`. |

CPU input with more than 12 features uses scikit-learn `NearestNeighbors`; lower-dimensional CPU data use SciPy tree search. GPU distances use float32; core observations use float64. Both GPU paths include host transfers/bookkeeping. Large common offsets can collapse GPU float32 distances; subtract a training-derived offset in float64 before fitting, retaining the same `eps`. The uncompiled CPU path can currently merge disconnected isolated cores incorrectly; see the model guide before using sparse-core configurations. No `transform`, `score`, or `partial_fit` is provided.

## GaussianMixture

[Model guide](gaussian-mixture.md)

```text
GaussianMixture(n_components=1, covariance_type='diag', tol=0.001, reg_covar=1e-06, max_iter=100, n_init=1, init_params='kmeans', random_state=None, device='auto', n_jobs=None)
```

| Parameter | Default | Meaning / accepted values |
|---|---|---|
| `n_components` | `1` | Positive integer mixture-component count at most `n`. |
| `covariance_type` | `'diag'` | `"diag"`, `"spherical"`, `"tied"`, or `"full"`; determines covariance shape below. |
| `tol` | `0.001` | Nonnegative convergence threshold; the model-specific criterion is described below. |
| `reg_covar` | `1e-06` | Nonnegative covariance regularization; see the model guide for diagonal floors versus full-matrix ridge updates. |
| `max_iter` | `100` | Positive integer iteration budget; see each model for iterations versus epochs. |
| `n_init` | `1` | Positive number of EM restarts; retain the best fitted lower bound. |
| `init_params` | `'kmeans'` | `"kmeans"` or `"random"` mean initialization. |
| `random_state` | `None` | Integer seed or `None`; controls randomized initialization/approximation. A fixed seed does not promise identical results across backends or library versions. |
| `device` | `'auto'` | `"auto"`, `"cpu"`, `"cuda"` (CuPy), or `"torch"` (Torch CUDA); see the shared device rules. |
| `n_jobs` | `None` | Accepted common CPU-job configuration; these implementations do not use it to control their numerical kernels or guarantee a thread count. |

| Method signature | Return and restrictions |
|---|---|
| `fit(X, y=None)` | Fit the mixture by EM; return `self`. |
| `fit_predict(X, y=None)` | Fit and return training maximum-responsibility labels `(n,)`. |
| `predict_proba(X)` | Return responsibilities `(m,k)`; each row sums to one. |
| `predict(X)` | Return maximum-responsibility component labels `(m,)`. |
| `score_samples(X)` | Return per-observation log mixture densities `(m,)`. |
| `score(X, y=None)` | Return mean log density as a Python float; higher is better. |
| `aic(X)` | Return AIC for the supplied observations; pass an array with `.shape`, lower is better. |
| `bic(X)` | Return BIC for the supplied observations; pass an array with `.shape`, lower is better. |

| Fitted output | Meaning and shape |
|---|---|
| `weights_`, `means_` | Mixing weights `(k,)` and means `(k,p)`. |
| `covariances_`, `precisions_cholesky_` | Covariances and precision factors: `(k,p)` for diag, `(k,)` for spherical, `(p,p)` for tied, `(k,p,p)` for full. Diag/spherical factors are inverse standard deviations; full/tied factors are lower triangular `L` with `L @ L.T = inv(covariance)`. |
| `converged_`, `n_iter_`, `lower_bound_`, `n_features_in_` | Convergence flag, EM iterations, last monitored mean log likelihood before the final M-step, and feature count. Check convergence before interpreting scores. |

`tol` controls absolute change in mean log likelihood, and `max_iter` limits each restart. For total log likelihood $L=m\,\mathrm{score}(X)$ and free-parameter count $d$, $\mathrm{AIC}=2d-2L$ and $\mathrm{BIC}=d\log m-2L$. Here $d=kp+(k-1)+d_{\mathrm{cov}}$, where the covariance parameter count is $kp$ (diag), $k$ (spherical), $p(p+1)/2$ (tied), or $kp(p+1)/2$ (full). Compare candidates on the same data. No `transform`, `partial_fit`, or coefficient-inference API is exposed.

## NMF

[Model guide](nmf.md)

```text
NMF(n_components=None, init='random', solver='mu', beta_loss='frobenius', max_iter=200, tol=0.0001, random_state=None, device='auto', n_jobs=None)
```

| Parameter | Default | Meaning / accepted values |
|---|---|---|
| `n_components` | `None` | Positive integer rank; `None` uses `min(n,p)`. |
| `init` | `'random'` | Only `"random"` is supported. |
| `solver` | `'mu'` | Only multiplicative updates, `"mu"`, are supported. |
| `beta_loss` | `'frobenius'` | Only `"frobenius"` is supported. |
| `max_iter` | `200` | Positive integer iteration budget; see each model for iterations versus epochs. |
| `tol` | `0.0001` | Nonnegative convergence threshold; the model-specific criterion is described below. |
| `random_state` | `None` | Integer seed or `None`; controls randomized initialization/approximation. A fixed seed does not promise identical results across backends or library versions. |
| `device` | `'auto'` | `"auto"`, `"cpu"`, `"cuda"` (CuPy), or `"torch"` (Torch CUDA); see the shared device rules. |
| `n_jobs` | `None` | Accepted common CPU-job configuration; these implementations do not use it to control their numerical kernels or guarantee a thread count. |

| Method signature | Return and restrictions |
|---|---|
| `fit(X, y=None)` | Fit nonnegative factors; return `self`. |
| `fit_transform(X, y=None)` | Return fitted training factors `W`, shape `(n,k)`. |
| `transform(X)` | With the learned components fixed, solve nonnegative factors for new data; return `(m,k)`. |
| `inverse_transform(X)` | Multiply input factors `(m,k)` by components `(k,p)`; return reconstruction `(m,p)`. |
| `predict(X)` | Alias for `transform(X)`. |

| Fitted output | Meaning and shape |
|---|---|
| `components_` | Nonnegative dictionary `H`, shape `(k,p)`. |
| `reconstruction_err_` | Frobenius norm of the training residual `X-WH`, a float (not squared loss). |
| `n_iter_`, `n_components_`, `n_features_in_` | Fitting iterations, rank, and input width. |

All fitted/transformed data must be nonnegative. `tol` tests relative change in reconstruction error at periodic checks. `fit_transform` returns the joint-fit `W`; a later `transform(X)` resolves `W` with `H` fixed and need not be identical. No `score` or `partial_fit` is provided.

## AgglomerativeClustering

[Model guide](agglomerative-clustering.md)

```text
AgglomerativeClustering(n_clusters=2, linkage='single', metric='euclidean', device='auto', n_jobs=None)
```

| Parameter | Default | Meaning / accepted values |
|---|---|---|
| `n_clusters` | `2` | Positive requested cluster count at most `n`; tied merge heights on the CPU maxclust cut can yield fewer groups. |
| `linkage` | `'single'` | `"single"`, `"complete"`, `"average"`, or `"ward"`. |
| `metric` | `'euclidean'` | Only `"euclidean"` is supported; other metrics, including precomputed distances, are unsupported. |
| `device` | `'auto'` | `"auto"`, `"cpu"`, `"cuda"` (CuPy), or `"torch"` (Torch CUDA); see the shared device rules. |
| `n_jobs` | `None` | Accepted common CPU-job configuration; these implementations do not use it to control their numerical kernels or guarantee a thread count. |

| Method signature | Return and restrictions |
|---|---|
| `fit(X, y=None)` | Build the dense hierarchy and selected cut; return `self`. |
| `fit_predict(X, y=None)` | Fit and return training labels `(n,)`. |
| `predict(X)` | Always raises `NotImplementedError` for unseen-sample prediction. |

| Fitted output | Meaning and shape |
|---|---|
| `labels_` | Training integer labels `(n,)`, returned as a NumPy array even for GPU fitting. |
| `children_`, `distances_` | NumPy merge pairs `(n-1,2)` and merge distances `(n-1,)`; leaf ids are `0..n-1`, merge row `i` has id `n+i`. |
| `n_features_in_` | Training input width. |

One observation with `n_clusters=1` has an empty merge tree. GPU paths use dense pairwise distances and reject estimates exceeding `STATGPU_AGGLOMERATIVE_GPU_MAX_BYTES` (default 1 GiB); this is a configured cap, not a measurement of available memory. The GPU expanded-distance formula can lose small separations when inputs have large common offsets; center in float64 before fitting. No sparse connectivity constraint, `transform`, `score`, or `partial_fit` is exposed.

## TruncatedSVD

[Model guide](truncated-svd.md)

```text
TruncatedSVD(n_components=2, algorithm='randomized', n_iter=5, n_oversamples=10, random_state=None, device='auto', n_jobs=None)
```

| Parameter | Default | Meaning / accepted values |
|---|---|---|
| `n_components` | `2` | Integer rank in `[1,min(n,p)]`. |
| `algorithm` | `'randomized'` | `"randomized"` or `"full"`. |
| `n_iter` | `5` | Nonnegative power-iteration count for the randomized method. |
| `n_oversamples` | `10` | Nonnegative integer giving extra random projection directions. |
| `random_state` | `None` | Integer seed or `None`; controls randomized initialization/approximation. A fixed seed does not promise identical results across backends or library versions. |
| `device` | `'auto'` | `"auto"`, `"cpu"`, `"cuda"` (CuPy), or `"torch"` (Torch CUDA); see the shared device rules. |
| `n_jobs` | `None` | Accepted common CPU-job configuration; these implementations do not use it to control their numerical kernels or guarantee a thread count. |

| Method signature | Return and restrictions |
|---|---|
| `fit(X, y=None)` | Fit uncentered low-rank components; return `self`. |
| `fit_transform(X, y=None)` | Fit and return training coordinates `(n,k)`. |
| `transform(X)` | Return uncentered projection `X @ components_.T`, shape `(m,k)`. |
| `inverse_transform(X)` | Return `X @ components_` from coordinates `(m,k)`, shape `(m,p)`; no mean is added. |
| `predict(X)` | Alias for `transform(X)`. |

| Fitted output | Meaning and shape |
|---|---|
| `components_`, `singular_values_` | Row components `(k,p)` and singular values `(k,)`. |
| `explained_variance_`, `explained_variance_ratio_` | Population variance of projected training columns (divisor `n`) and fraction of summed original feature variances; both `(k,)`. |
| `n_components_`, `n_features_in_` | Selected rank and training width. |

Dense input only: this is not a sparse-matrix replacement for scikit-learn TruncatedSVD. Projection is uncentered even though the reported variances subtract each column mean. No whitening, `score`, or `partial_fit` is provided.

## MiniBatchKMeans

[Model guide](minibatch-kmeans.md)

```text
MiniBatchKMeans(n_clusters=8, init='k-means++', n_init='auto', batch_size=1024, max_iter=100, max_no_improvement=10, tol=0.0, random_state=None, device='auto', n_jobs=None)
```

| Parameter | Default | Meaning / accepted values |
|---|---|---|
| `n_clusters` | `8` | Positive integer; ordinary `fit` requires at least that many rows. |
| `init` | `'k-means++'` | `"k-means++"`, `"random"`, or an explicit center array `(k,p)`; callable initialization is unsupported. |
| `n_init` | `'auto'` | Positive integer or `"auto"`: one run for k-means++/explicit centers, three for random. Used by `fit`, not repeated `partial_fit`. |
| `batch_size` | `1024` | Positive maximum batch size within `fit`; each `partial_fit` consumes its entire supplied batch. |
| `max_iter` | `100` | Positive integer iteration budget; see each model for iterations versus epochs. |
| `max_no_improvement` | `10` | Nonnegative integer budget of batches without a new best batch inertia, or `None` to disable this stop rule. |
| `tol` | `0.0` | Nonnegative convergence threshold; the model-specific criterion is described below. |
| `random_state` | `None` | Integer seed or `None`; controls randomized initialization/approximation. A fixed seed does not promise identical results across backends or library versions. |
| `device` | `'auto'` | `"auto"`, `"cpu"`, `"cuda"` (CuPy), or `"torch"` (Torch CUDA); see the shared device rules. |
| `n_jobs` | `None` | Accepted common CPU-job configuration; these implementations do not use it to control their numerical kernels or guarantee a thread count. |

| Method signature | Return and restrictions |
|---|---|
| `fit(X, y=None, sample_weight=None)` | Start a new fit over the whole array, then polish centers with full-data Lloyd steps; return `self`. `sample_weight` must be `None`. |
| `partial_fit(X, y=None, sample_weight=None)` | Update centers/counts once using this batch; return `self`. `sample_weight` must be `None`. |
| `fit_predict(X, y=None)` | Fit and return whole-training labels `(n,)`. |
| `transform(X)` | Return distances to all centers `(m,k)`. |
| `predict(X)` | Return nearest-center labels `(m,)`. |
| `score(X, y=None)` | Return negative nearest-center squared-distance sum as a Python float. |

| Fitted output | Meaning and shape |
|---|---|
| `cluster_centers_`, `counts_` | Centers `(k,p)` and per-center counts `(k,)`; `fit` publishes final full-data assignment counts, while `partial_fit` accumulates batch assignments. |
| `labels_`, `inertia_` | For `fit`, whole-training labels/inertia; after `partial_fit`, only the latest batch. Labels have the latest input row count. |
| `n_iter_`, `n_steps_`, `n_features_in_` | For `fit`: epochs entered (the last can stop partway through), batch updates, and feature count. Each `partial_fit` increments both counters once. |

With a string initializer, the first `partial_fit` batch needs at least `n_clusters` rows. Explicit centers allow a smaller first batch. Later batches may have fewer rows but must retain feature width and order. `max_iter`, `tol` (squared center movement), and `max_no_improvement` govern `fit`; they do not turn one `partial_fit` call into an epoch loop. Unlike `fit`, `partial_fit` does not run a full-data polishing pass. Its `labels_` and `inertia_` record the batch assignment used before moving the centers; use `predict(batch)` and `-score(batch)` to evaluate the updated centers.

## IncrementalPCA

[Model guide](incremental-pca.md)

```text
IncrementalPCA(n_components=None, batch_size=None, whiten=False, copy=True, device='auto', n_jobs=None)
```

| Parameter | Default | Meaning / accepted values |
|---|---|---|
| `n_components` | `None` | Positive rank no greater than `p`; `None` means `min(n,p)` in `fit`, or `min(first_batch_rows,p)` fixed by the first `partial_fit`. |
| `batch_size` | `None` | Positive integer or `None`; `fit` uses all rows at once by default and expands a too-small first batch to fit the resolved rank. Does not split a `partial_fit` argument. |
| `whiten` | `False` | Whether transformed PCA coordinates are rescaled by the fitted component standard deviations. |
| `copy` | `True` | Compatibility option; input data are not modified, including when `False`. |
| `device` | `'auto'` | `"auto"`, `"cpu"`, `"cuda"` (CuPy), or `"torch"` (Torch CUDA); see the shared device rules. |
| `n_jobs` | `None` | Accepted common CPU-job configuration; these implementations do not use it to control their numerical kernels or guarantee a thread count. |

| Method signature | Return and restrictions |
|---|---|
| `fit(X, y=None)` | Reset incremental state and process this dataset in batches; return `self`. |
| `partial_fit(X, y=None)` | Extend the decomposition using all supplied rows; return `self`. |
| `fit_transform(X, y=None)` | Fit, then transform all training rows with the final basis; return `(n,k)`. |
| `transform(X)` | Center by the accumulated mean, project, optionally whiten; return `(m,k)`. |
| `inverse_transform(X)` | Undo whitening, reconstruct coordinates `(m,k)`, add accumulated mean; return `(m,p)`. |
| `predict(X)` | Alias for `transform(X)`. |

| Fitted output | Meaning and shape |
|---|---|
| `components_`, `singular_values_` | Retained row directions `(k,p)` and singular values `(k,)`. |
| `mean_`, `var_` | Accumulated feature means and population variances `(p,)`. |
| `explained_variance_`, `explained_variance_ratio_` | Retained variances (divisor `n_samples_seen_-1` when greater than zero) and fractions, `(k,)`. |
| `n_samples_seen_`, `n_components_`, `n_features_in_` | Total processed rows, retained rank, and feature width. |

The first `partial_fit` requires at least an explicitly requested `n_components` rows; later batches may be smaller. Feature width must remain fixed. With `n_components=None`, choosing a small first batch also limits the retained rank; it does not grow automatically. Truncation and batch order can affect results. Whitening uses a small variance floor for numerical stability. No `score` is exposed.

## MiniBatchNMF

[Model guide](minibatch-nmf.md)

```text
MiniBatchNMF(n_components=None, init='random', batch_size=None, max_iter=200, tol=0.0001, random_state=None, device='auto', n_jobs=None)
```

| Parameter | Default | Meaning / accepted values |
|---|---|---|
| `n_components` | `None` | Positive rank; `None` uses `min(n,p)` for `fit` or fixes the rank from the first `partial_fit` batch. |
| `init` | `'random'` | Only `"random"` is supported. |
| `batch_size` | `None` | Positive integer or `None`; controls batches inside `fit`. `None` chooses a data-size-dependent batch, possibly the full dataset. `partial_fit` consumes the supplied batch. |
| `max_iter` | `200` | Positive integer iteration budget; see each model for iterations versus epochs. |
| `tol` | `0.0001` | Nonnegative convergence threshold; the model-specific criterion is described below. |
| `random_state` | `None` | Integer seed or `None`; controls randomized initialization/approximation. A fixed seed does not promise identical results across backends or library versions. |
| `device` | `'auto'` | `"auto"`, `"cpu"`, `"cuda"` (CuPy), or `"torch"` (Torch CUDA); see the shared device rules. |
| `n_jobs` | `None` | Accepted common CPU-job configuration; these implementations do not use it to control their numerical kernels or guarantee a thread count. |

| Method signature | Return and restrictions |
|---|---|
| `fit(X, y=None)` | Start a new nonnegative factor fit; return `self`. |
| `partial_fit(X, y=None)` | Accumulate batch factor statistics and update components; return `self`. |
| `fit_transform(X, y=None)` | Fit components, then solve training factors `(n,k)` with those components fixed. |
| `transform(X)` | Return nonnegative factors `(m,k)` with learned components fixed. |
| `inverse_transform(X)` | Multiply factors `(m,k)` by components; return `(m,p)`. |
| `predict(X)` | Alias for `transform(X)`. |

| Fitted output | Meaning and shape |
|---|---|
| `components_` | Nonnegative dictionary `(k,p)`. |
| `reconstruction_err_` | Frobenius residual norm from the fitting factors; whole-data after `fit`, latest-batch after `partial_fit`. A later `transform` may refine factors and give a different error. |
| `n_iter_`, `n_components_`, `n_features_in_` | Fit epochs (or number of partial updates), chosen rank, and fixed input width. |

Use nonnegative dense data and keep feature width/order fixed. An explicitly positive rank need not be smaller than the first batch; with `None`, the first batch determines it. `max_iter` limits `fit` epochs and influences the fixed-component transform solve; `tol` checks relative component change during fitting, not relative reconstruction-error change. Neither controls a convergence loop in `partial_fit`. A feature that is zero throughout the first batch can become permanently zero in the dictionary, even if later batches contain positive values. Buffer representative initialization rows or restart with representative retained data when later data contain positive values for a zero dictionary column; see the [initial-batch precautions](minibatch-nmf.md#initial-batches-with-zero-features). No `score` or sample-weight argument is exposed.

## UMAP

[Model guide](umap.md)

```text
UMAP(n_neighbors=15, n_components=2, metric='euclidean', min_dist=0.1, spread=1.0, n_epochs=None, learning_rate=1.0, init='spectral', negative_sample_rate=5, repulsion_strength=1.0, random_state=None, nn_method='auto', device='auto', n_jobs=None)
```

| Parameter | Default | Meaning / accepted values |
|---|---|---|
| `n_neighbors` | `15` | Integer in `[2,n-1]`; size of local neighborhoods. |
| `n_components` | `2` | Positive integer embedding width less than `n`; the CPU one-dimensional path currently fails, so use at least two components on CPU. |
| `metric` | `'euclidean'` | Only `"euclidean"` is supported; other metrics, including precomputed distances, are unsupported. |
| `min_dist` | `0.1` | Nonnegative low-dimensional compactness setting; interpret jointly with `spread`. |
| `spread` | `1.0` | Positive scale of the low-dimensional attraction curve. |
| `n_epochs` | `None` | Positive integer or `None`; current automatic schedule is 500 for `n<=2000`, 200 for `n<=10000`, otherwise 100. Actual value is `n_epochs_`. |
| `learning_rate` | `1.0` | Positive initial optimization step size. |
| `init` | `'spectral'` | `"spectral"` (host SciPy eigensolver) or `"random"`; use random initialization to avoid the current sparse spectral-initialization limitation described below. |
| `negative_sample_rate` | `5` | Positive integer; each epoch samples `n * negative_sample_rate` independent source/target pairs, not that many pairs per attractive edge. |
| `repulsion_strength` | `1.0` | Positive repulsive-force multiplier. |
| `random_state` | `None` | Integer seed or `None`; controls randomized initialization/approximation. A fixed seed does not promise identical results across backends or library versions. |
| `nn_method` | `'auto'` | `"auto"`, `"exact"`, or `"nndescent"`; exact search uses dense distances, NNDescent is approximate, and auto currently always chooses exact search. |
| `device` | `'auto'` | `"auto"`, `"cpu"`, `"cuda"` (CuPy), or `"torch"` (Torch CUDA); see the shared device rules. |
| `n_jobs` | `None` | Accepted common CPU-job configuration; these implementations do not use it to control their numerical kernels or guarantee a thread count. |

| Method signature | Return and restrictions |
|---|---|
| `fit(X, y=None)` | Fit a training embedding; return `self`. |
| `fit_transform(X, y=None)` | Fit and return `embedding_`, shape `(n,k)`. |
| `transform(X)` | Always raises `NotImplementedError`, including for the original data; use `embedding_` for those rows. |
| `predict(X)` | Always raises `NotImplementedError`. |

| Fitted output | Meaning and shape |
|---|---|
| `embedding_` | Training coordinates `(n,k)` on the selected backend. |
| `graph_` | A tuple `(source_rows, target_rows, edge_weights, n_samples)`, not a SciPy adjacency matrix. The first three entries are backend arrays of equal edge count. |
| `n_epochs_`, `n_features_in_` | Executed epoch count and original feature count. |

Neighbor distances use float32 internally; embedding optimization uses float64. Large common feature offsets can collapse distinct observations during float32 conversion or corrupt expanded squared distances; subtract a training-derived offset before fitting, while the data are still float64. Very large pairwise distances can make exact search select self-neighbors that are later removed, even for centered input. After centering, divide all features by one common positive training-derived scale to obtain moderate coordinates; reuse this preparation for comparable data. This preserves Euclidean neighbor ordering and differs from per-feature rescaling. Graph assembly uses host SciPy even on GPU, and spectral initialization also uses host SciPy. Seeded random initialization is useful when testing shape/API behavior; visualization quality needs separate checks. The current force updates approximate a neighborhood layout but are not the exact gradient of standard UMAP cross-entropy. With NumPy 2, CPU `nn_method="nndescent"` currently fails during backend dispatch; use `"exact"` or `"auto"`. The sparse spectral initializer can retain the constant graph eigenvector in place of a nontrivial direction; its eigensolver start is also not controlled by `random_state`. Use `init="random"` for seeded initialization. No inverse transform, score, or incremental fit is provided.

## TSNE

[Model guide](tsne.md)

```text
TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate='auto', max_iter=1000, init='pca', random_state=None, metric='euclidean', device='auto', n_jobs=None)
```

| Parameter | Default | Meaning / accepted values |
|---|---|---|
| `n_components` | `2` | Positive embedding width less than `n`; `init="pca"` additionally requires at most `min(n,p)` components. |
| `perplexity` | `30.0` | Positive neighborhood-perplexity target strictly smaller than `n`. |
| `early_exaggeration` | `12.0` | Positive multiplier of high-dimensional affinities during early optimization. |
| `learning_rate` | `'auto'` | Positive number or `"auto"`; auto is `max(n / early_exaggeration / 4, 10)`, which need not match another library. |
| `max_iter` | `1000` | Integer at least 250; total optimization iterations. |
| `init` | `'pca'` | `"pca"` or `"random"`; no user-supplied embedding option. |
| `random_state` | `None` | Integer seed or `None`; controls randomized initialization/approximation. A fixed seed does not promise identical results across backends or library versions. |
| `metric` | `'euclidean'` | Only `"euclidean"` is supported; other metrics, including precomputed distances, are unsupported. |
| `device` | `'auto'` | `"auto"`, `"cpu"`, `"cuda"` (CuPy), or `"torch"` (Torch CUDA); see the shared device rules. |
| `n_jobs` | `None` | Accepted common CPU-job configuration; these implementations do not use it to control their numerical kernels or guarantee a thread count. |

| Method signature | Return and restrictions |
|---|---|
| `fit(X, y=None)` | Fit exact dense Euclidean t-SNE; return `self`. |
| `fit_transform(X, y=None)` | Fit and return training embedding `(n,k)`. |
| `transform(X)` | Always raises `NotImplementedError`; retain `embedding_` for the training rows. |
| `predict(X)` | Always raises `NotImplementedError`. |

| Fitted output | Meaning and shape |
|---|---|
| `embedding_` | Training coordinates `(n,k)` on the selected backend. |
| `kl_divergence_` | Final high-/low-dimensional affinity KL objective, a Python float; not a general held-out score. |
| `n_iter_`, `n_features_in_` | Executed iterations (the configured budget) and original input width. |

This implementation allocates dense pairwise arrays, so memory grows quadratically with sample count. It has no Barnes–Hut/FFT method selector, sparse/precomputed-distance support, inverse transform, score, or incremental fit. Do not transfer learning-rate or perplexity defaults from another library without checking its convention. The current affinity bandwidth search can fail for extremely large or tiny feature scales, and can return an invalid negative KL value. Subtract a training-derived offset before rescaling to moderate magnitudes: the expanded distance calculation can corrupt affinities at large offsets even when KL remains finite and nonnegative. These checks are necessary but not sufficient for valid affinities; see the [TSNE numerical cautions](tsne.md#numerical-and-lifecycle-cautions).

## Incremental example

Each first batch below is large enough for the requested rank/cluster count. All subsequent batches retain the same feature order.

```python
# Example: incremental_unsupervised_api
import numpy as np
from statgpu.unsupervised import IncrementalPCA, MiniBatchKMeans, MiniBatchNMF

rng = np.random.default_rng(24)
X = rng.normal(size=(36, 4))
positive = np.abs(X) + 0.1
ipca = IncrementalPCA(n_components=2, device="cpu")
km = MiniBatchKMeans(n_clusters=3, random_state=24, device="cpu")
nmf = MiniBatchNMF(n_components=2, random_state=24, device="cpu")
for start in range(0, len(X), 12):
    ipca.partial_fit(X[start:start + 12])
    km.partial_fit(X[start:start + 12])
    nmf.partial_fit(positive[start:start + 12])
Z = ipca.transform(X[:5])
reconstructed = ipca.inverse_transform(Z)
labels = km.predict(X[:5])
factors = nmf.transform(positive[:5])
print("PCA coordinates/reconstruction:", Z.shape, reconstructed.shape)
print("Total rows / latest cluster labels:", ipca.n_samples_seen_, km.labels_.shape)
print("New labels / NMF factors:", labels.shape, factors.shape)
```

Expected shapes are `(5,2)` and `(5,4)`; total PCA rows are `36`, while `km.labels_` has shape `(12,)` for only the latest batch. New labels have shape `(5,)` and NMF factors `(5,2)`. Use `km.predict(all_rows)` when you need labels for all processed observations. These shape checks illustrate API behavior, not model quality or GPU performance.
