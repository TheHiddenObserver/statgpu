# TruncatedSVD

> Language: English
> Last updated: 2026-10-05
> Switch: [Chinese](../../cn/unsupervised/truncated-svd.md)
> Path: `statgpu.unsupervised.TruncatedSVD`

## Overview

`TruncatedSVD` computes a low-rank projection without centering the input matrix. This makes it different from `PCA` and suitable for dense LSA-style workflows.

## When to use it

Use TruncatedSVD when the origin has meaning and centering would change the question. Unlike PCA, a large feature mean can dominate its leading direction. This implementation requires dense input; it is not a sparse text-matrix solver.

## Path

Import from `statgpu.unsupervised`:

```python
from statgpu.unsupervised import TruncatedSVD
```

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

- Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. *SIAM Review*, 53(2), 217-288.
- scikit-learn developers. `sklearn.decomposition.TruncatedSVD` API documentation.
