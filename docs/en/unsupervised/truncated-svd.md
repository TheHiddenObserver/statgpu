# TruncatedSVD

> Language: English
> Last updated: 2026-10-14
> Switch: [Chinese](../../cn/unsupervised/truncated-svd.md)
> Path: `statgpu.unsupervised.TruncatedSVD`

## Overview

In a document-term matrix, rows describe documents and columns record term weights. You want fewer coordinates that summarize patterns of terms occurring together, while still approximating the original matrix. Selecting only a few common terms discards their joint structure; taking a row total reduces every document to one magnitude. Subtracting each term's mean would answer another question: how each document differs from an average document.

TruncatedSVD learns a few shared patterns, each assigning a weight to every term. A document is approximated by scaling these patterns by its own coordinates and adding them together, so each coordinate summarizes a combination of terms rather than a single word. These patterns are the matrix's retained singular directions; they provide a low-rank approximation without subtracting feature means. Applied to a document-term matrix, this is the basic idea of latent semantic analysis (LSA). The same uncentered approximation applies to other numeric matrices; it does not require text or nonnegative values.

The choice to retain the origin is substantive. A large common level may dominate the first direction even if it explains little of the differences around the mean. A singular direction is also not automatically a named topic or an independently meaningful source.

## When to use it

Use TruncatedSVD when approximating the uncentered matrix is the intended goal. Choose [PCA](pca.md) when variation around feature means is what matters; applying SVD after centering leads to the PCA problem. Choose [NMF](nmf.md) when both the patterns and their contributions must be nonnegative: TruncatedSVD allows positive and negative coefficients.

The statgpu implementation accepts dense input only. The LSA motivation does not imply support for sparse document-term matrices; use a sparse-capable implementation if the dense matrix would be impractical. Choose rank from reconstruction and the intended use of the reduced coordinates, rather than assigning a topic interpretation from the component order alone.

## Path

Import from `statgpu.unsupervised`:

```python
from statgpu.unsupervised import TruncatedSVD
```

This estimator supports dense input on CPU, CuPy/CUDA, and Torch CUDA.

## Objective Function / Loss Function

For a rank `k` approximation, Truncated SVD solves:

$$
\min_{\operatorname{rank}(Z) \le k} \|X - Z\|_F^2.
$$

The returned components are the leading right singular vectors of `X`.

## Estimating Equation

The exact path computes:

$$
X = U \Sigma V^\top.
$$

The randomized path projects `X` to a lower-dimensional random subspace, re-orthogonalizes each power iteration, computes a small SVD of the projected matrix, then applies a deterministic component sign convention.

## Parameters

`n_components`, `algorithm`, `n_iter`, `n_oversamples`, `random_state`, and `device` control the decomposition and backend.

## A small CPU example

<!-- learner-example: truncated-svd -->
```python
import numpy as np
from statgpu.unsupervised import TruncatedSVD

rng = np.random.default_rng(0)
X = rng.normal(size=(60, 5)) + 2.0
model = TruncatedSVD(n_components=2, algorithm="full", device="cpu")
Z = model.fit_transform(X)
X_hat = model.inverse_transform(Z)
print(Z.shape, X_hat.shape, model.explained_variance_ratio_.sum())
```

Coordinates have shape `(60, 2)`. The reconstruction does not add a mean. Explained variance is the centered variance of projected coordinates (divisor `n`), although fitting itself is uncentered; it is not simply the retained squared singular values divided by total squared norm.

For a supported GPU installation, construct a new estimator with `device="cuda"` (CuPy) or `device="torch"` (Torch CUDA). Arrays generally stay on that backend; see the [API reference](api-reference.md#truncatedsvd) for output ownership and host-side work. An unavailable explicit GPU raises an error.

## Approximation and interpretation

`algorithm="full"` is exact dense SVD. `algorithm="randomized"` is approximate and should be compared with sign- or subspace-invariant metrics.

## Outputs

`components_`, `explained_variance_`, `explained_variance_ratio_`, `singular_values_`, `n_components_`, and `n_features_in_`.

## FAQ

Sparse input and ARPACK are not supported.


## Complete API reference

Constructor defaults, all public methods, output shapes, and restrictions are listed in the [TruncatedSVD API reference](api-reference.md#truncatedsvd).

## References

- Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. *SIAM Review*, 53(2), 217-288. https://doi.org/10.1137/090771806
- scikit-learn Developers. [TruncatedSVD: uncentered projection and LSA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html).
