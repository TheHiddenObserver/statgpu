# TSNE

> Language: English
> Last updated: 2026-10-14
> Switch: [Chinese](../../cn/unsupervised/tsne.md)
> Path: `statgpu.unsupervised.TSNE`

## Overview

t-SNE places numeric observations in a low-dimensional map, usually with two or three coordinates per row, to explore local similarities. It aims to keep important neighbors close rather than preserve all pairwise distances or reconstruct the original measurements.

For example, suppose each image in a collection is described by hundreds of numeric features, and you want to inspect which images have similar feature profiles without predefined categories. A scatter plot of two columns discards the other features. t-SNE aims to place the images on a two-dimensional map so that observations close in the original feature space remain neighbors on the map.

To do this, t-SNE converts distances into neighbor affinities: weights that express how close observations are, with larger weights for nearer neighbors. Each observation gets its own neighborhood scale. The map then moves points to reproduce those affinities, giving priority to keeping strongly related observations close rather than preserving every distance. The resulting coordinates are an embedding, a low-dimensional representation with one point per observation. statgpu implements exact dense Euclidean t-SNE; the features and their scaling must therefore make Euclidean proximity meaningful for your question.

## When to use it

Use t-SNE when inspecting local similarities in a fixed set of observations is the goal, for example to review representative images from nearby points or investigate why an observation sits apart.

- Start with a raw-feature plot or [PCA](pca.md) when a few measured variables or a linear summary already answer the question. PCA summarizes large directions of variation with a linear projection, which can overlap neighborhoods on a curved structure; t-SNE instead focuses on local affinities. Prefer PCA when you need explained variance, reconstruction, or a fitted map that can transform later observations.
- Compare [UMAP](umap.md) when you want to explore a selected-neighbor graph and vary its neighborhood size separately from its layout controls. t-SNE instead matches pairwise affinities, with `perplexity` setting an effective neighborhood scale. Neither statgpu implementation supports new-data `transform`.
- If you need cluster assignments, specify and validate a clustering method separately. A t-SNE island does not establish a distinct population or choose the number of clusters for you.

## Path

Import from `statgpu.unsupervised`:

```python
from statgpu.unsupervised import TSNE
```

## Objective Function / Loss Function

Write $p_{ij}$ for the input-space affinity of a pair and $q_{ij}$ for its map affinity. A pair with large $p_{ij}$ contributes strongly to the loss if the map gives it very small $q_{ij}$: breaking an important neighborhood is costly. t-SNE minimizes KL divergence:

$$
\operatorname{KL}(P \| Q)
= \sum_{i \ne j} p_{ij}\log\frac{p_{ij}}{q_{ij}}.
$$

## Estimating Equation

For each observation, Gaussian conditional probabilities assign more weight to nearby observations. For $j\ne i$,

$$
p_{j\mid i}=\frac{\exp(-\beta_i\|x_i-x_j\|_2^2)}
{\sum_{\ell\ne i}\exp(-\beta_i\|x_i-x_\ell\|_2^2)},
\qquad p_{i\mid i}=0,
\qquad \beta_i=\frac{1}{2\sigma_i^2}.
$$

$$
\operatorname{Perp}_i=\exp\left(-\sum_{j\ne i}p_{j\mid i}\log p_{j\mid i}\right),
\qquad p_{ij}=\frac{p_{j\mid i}+p_{i\mid j}}{2n},
\qquad p_{ii}=0.
$$

Here $n$ is the number of observations and $\sigma_i$ is a row-specific bandwidth. Binary search adjusts $\beta_i$ to seek the requested `perplexity`; symmetrization produces the joint affinities $P$ used in the objective. The low-dimensional affinities for distinct pairs use:

$$
q_{ij} =
\frac{(1+\|y_i-y_j\|_2^2)^{-1}}
{\sum_{a \ne b}(1+\|y_a-y_b\|_2^2)^{-1}}.
$$

Set $q_{ii}=0$. Both joint affinity matrices sum to one across distinct ordered pairs when the bandwidth search succeeds. The map affinities use a Student-t distribution with one degree of freedom. Its heavier tails, compared with the input-space Gaussian affinities, let moderately separated map points retain more affinity. This helps reduce crowding, the difficulty of fitting many high-dimensional neighbors into a small low-dimensional region, as described in the [original t-SNE method](https://jmlr.org/papers/v9/vandermaaten08a.html). The embedding is optimized with early exaggeration, momentum, and adaptive per-coordinate gains.

Perplexity is an effective neighborhood size, not a literal number of selected neighbors. A distribution over `n-1` other observations has perplexity between 1 and `n-1`; tied nearest neighbors can impose a larger lower limit. The API range check only requires `0 < perplexity < n`, so an accepted setting need not be attainable. Prefer a value comfortably inside the feasible range and below the sample count; duplicate observations and failed numerical calibration can prevent the target from being reached. Compare settings within the same dataset rather than comparing KL values computed from different target affinities.

## Parameters

`n_components`, `perplexity`, `early_exaggeration`, `learning_rate`, `max_iter`, `init`, `random_state`, `metric`, and `device`.

## A small CPU example

<!-- learner-example: tsne -->
```python
import numpy as np
from statgpu.unsupervised import TSNE

rng = np.random.default_rng(0)
X = np.vstack([rng.normal(-2, 0.4, (20, 3)), rng.normal(2, 0.4, (20, 3))])
model = TSNE(perplexity=5, max_iter=300, init="random", random_state=0, device="cpu")
embedding = model.fit_transform(X)
print(embedding.shape, model.n_iter_, model.kl_divergence_)
```

The embedding has shape `(40, 2)`, with one row for each input observation; retain row identifiers to look up the original features or images. KL divergence describes this training affinity fit, not a held-out prediction score. Even exact pairwise affinities lead to a non-convex optimization and do not promise a globally optimal embedding. The two groups in this synthetic example were put into the input deliberately; a separated-looking result on unknown data would not supply the same evidence.

For a supported GPU installation, construct a new estimator with `device="cuda"` (CuPy) or `device="torch"` (Torch CUDA). Arrays generally stay on that backend; see the [API reference](api-reference.md#tsne) for output ownership and host-side work. An unavailable explicit GPU raises an error.

## Approximation and interpretation

This is exact dense t-SNE and requires pairwise arrays whose memory grows quadratically with the number of observations. Barnes-Hut and FFT/FIt-SNE acceleration are not supported.

Read the map as a source of questions to check in the original features. Compare several feasible perplexities and random seeds, then inspect whether apparent neighbors really are similar under your chosen preprocessing. Plot axes have no feature units; gaps between islands and island areas do not measure population separation or density. Even unclustered data can produce apparent clumps, as demonstrated in [How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/). Stability across settings is a useful check, but is not proof of a genuine cluster or a statistical significance test.

## Outputs

`embedding_`, `kl_divergence_`, `n_iter_`, and `n_features_in_`.

## FAQ

Sparse input, non-Euclidean metrics, Barnes-Hut, FFT/FIt-SNE, and new-data `transform` are not supported.


## Numerical and lifecycle cautions

All numeric controls must be finite; non-finite values such as `learning_rate=np.nan` are not reliably rejected and can produce non-finite embeddings. Very large or tiny feature scales can also defeat the current affinity bandwidth search. First subtract a training-derived feature offset in float64, then rescale to moderate magnitudes before fitting. Large common offsets can corrupt the expanded distance formula even when the affinity matrix remains normalized and KL is nonnegative. Reject non-finite embeddings or negative `kl_divergence_`; a negative KL is invalid, not an unusually good fit, and nonnegative KL alone is not a correctness check.

## Complete API reference

Constructor defaults, all public methods, output shapes, and restrictions are listed in the [TSNE API reference](api-reference.md#tsne).

## References

- van der Maaten, L., & Hinton, G. (2008). [Visualizing data using t-SNE](https://jmlr.org/papers/v9/vandermaaten08a.html). *Journal of Machine Learning Research*, 9, 2579-2605.
- Wattenberg, M., Viégas, F., & Johnson, I. (2016). [How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/). *Distill*.
- Linderman, G. C., Rachh, M., Hoskins, J. G., Steinerberger, S., & Kluger, Y. (2019). Fast interpolation-based t-SNE for improved visualization of single-cell RNA-seq data. *Nature Methods*, 16, 243-245.
