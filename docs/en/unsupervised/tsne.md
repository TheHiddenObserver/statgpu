# TSNE

> Language: English
> Last updated: 2026-10-06
> Switch: [Chinese](../../cn/unsupervised/tsne.md)
> Path: `statgpu.unsupervised.TSNE`

## Overview

`TSNE` embeds dense data by matching high-dimensional Gaussian affinities with low-dimensional Student-t affinities. The current implementation implements exact dense Euclidean t-SNE.

## When to use it

Use t-SNE for a visualization of local similarities rather than a predictive feature map. Compare perplexity and random seeds before interpreting islands. Global distances, plot axes and island areas are not quantitative measures of population differences.

## Path

Import from `statgpu.unsupervised`:

```python
from statgpu.unsupervised import TSNE
```

## Objective Function / Loss Function

t-SNE minimizes KL divergence:

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

Set $q_{ii}=0$. Both joint affinity matrices sum to one across distinct ordered pairs when the bandwidth search succeeds. The embedding is optimized with early exaggeration, momentum, and adaptive per-coordinate gains.

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

The embedding has shape `(40, 2)`. KL divergence describes this training affinity fit, not a held-out prediction score. Even exact pairwise affinities lead to a non-convex optimization and do not promise a globally optimal embedding.

For a supported GPU installation, construct a new estimator with `device="cuda"` (CuPy) or `device="torch"` (Torch CUDA). Arrays generally stay on that backend; see the [API reference](api-reference.md#tsne) for output ownership and host-side work. An unavailable explicit GPU raises an error.

## Approximation and interpretation

This is exact dense t-SNE. Barnes-Hut, FFT/FIt-SNE, and openTSNE acceleration are external baselines only.

## Outputs

`embedding_`, `kl_divergence_`, `n_iter_`, and `n_features_in_`.

## FAQ

Sparse input, non-Euclidean metrics, Barnes-Hut, FFT/FIt-SNE, and new-data `transform` are not supported.


## Numerical and lifecycle cautions

All numeric controls must be finite; non-finite values such as `learning_rate=np.nan` are not reliably rejected and can produce non-finite embeddings. Very large or tiny feature scales can also defeat the current affinity bandwidth search. First subtract a training-derived feature offset in float64, then rescale to moderate magnitudes before fitting. Large common offsets can corrupt the expanded distance formula even when the affinity matrix remains normalized and KL is nonnegative. Reject non-finite embeddings or negative `kl_divergence_`; a negative KL is invalid, not an unusually good fit, and nonnegative KL alone is not a correctness check.

## Complete API reference

Constructor defaults, all public methods, output shapes, and restrictions are listed in the [TSNE API reference](api-reference.md#tsne).

## References

- van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. *Journal of Machine Learning Research*, 9, 2579-2605.
- Linderman, G. C., Rachh, M., Hoskins, J. G., Steinerberger, S., & Kluger, Y. (2019). Fast interpolation-based t-SNE for improved visualization of single-cell RNA-seq data. *Nature Methods*, 16, 243-245.
