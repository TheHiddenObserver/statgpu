# TruncatedSVD

> Language: English
> Last updated: 2026-10-14
> Switch: [Chinese](../../cn/unsupervised/truncated-svd.md)
> Path: `statgpu.unsupervised.TruncatedSVD`

## Overview

TruncatedSVD approximates a numeric matrix using a chosen number of shared patterns, without first subtracting column means. Each pattern assigns a weight to every input feature. Each row gets coordinates that scale these patterns; adding the scaled patterns reconstructs an approximation of that row. The patterns and coordinates can contain positive or negative values.

The method keeps the strongest singular directions: directions that capture the most squared magnitude in the original matrix. Keeping a few gives a low-rank approximation and a shorter coordinate vector for each row. Because fitting is uncentered, a large pattern shared by most rows can dominate the result even if it explains little of the differences between rows. A component need not represent an independently meaningful source.

For example, in a document-term matrix, rows describe documents and columns record term weights. Selecting only a few common terms discards their joint structure; taking a row total reduces every document to one magnitude. TruncatedSVD instead summarizes combinations of terms occurring together. It does not first subtract each term's average weight, so the first component may mostly reflect shared word use. A component need not correspond to a topic you can name. This text application is called latent semantic analysis (LSA); text and nonnegative values are not requirements of the method.

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

Each of the 60 input rows has five measurements; each output row has two numbers giving its coordinates on the two learned patterns, so `Z` has shape `(60, 2)`. For document-term data, that would mean two coordinates per document. Here `Z = X @ model.components_.T` and `X_hat = Z @ model.components_`: neither step subtracts or adds a mean. Explained variance is the centered variance of projected coordinates (divisor `n`), although fitting itself is uncentered; it is not simply the retained squared singular values divided by total squared norm.

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
