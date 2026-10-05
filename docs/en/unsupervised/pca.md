# PCA

> Language: English
> Last updated: 2026-10-05
> Switch: [Chinese](../../cn/unsupervised/pca.md)

## Overview

`PCA` estimates an orthonormal low-dimensional basis that captures the largest variance directions of centered dense data. It supports CPU, CuPy/CUDA, and Torch CUDA backends.

## When to use it

Use PCA when correlated numeric features can be summarized by a smaller set of linear directions. Centering is automatic, but feature scaling is not: choose units before fitting, and fit preprocessing only on training data. Choose rank from retained variance and held-out reconstruction, rather than treating high variance as scientific importance.

## Path

```python
from statgpu.unsupervised import PCA
```

## Objective Function / Loss Function

For centered data `X_c = X - mean(X)`, PCA solves:

$$
\begin{aligned}
\max_{W \in \mathbb{R}^{p \times k}} \quad
& \operatorname{tr}\left(W^\top X_c^\top X_c W\right) \\
\text{s.t.} \quad
& W^\top W = I_k .
\end{aligned}
$$

Keeping `k` components is also the best rank-`k` squared-error reconstruction among orthonormal projections:

$$
\begin{aligned}
\min_{W \in \mathbb{R}^{p \times k}} \quad
& \left\|X_c - X_c W W^\top\right\|_F^2 \\
\text{s.t.} \quad
& W^\top W = I_k .
\end{aligned}
$$

The two objectives are equivalent because total variance is fixed after centering.

## Estimating Equation

- `svd_solver="covariance"` computes
  $$
  \Sigma = \frac{X_c^\top X_c}{n - 1}
  $$
  and solves
  $$
  \Sigma v_j = \lambda_j v_j
  $$
  with `eigh`.
- `svd_solver="full"` computes
  $$
  X_c = U S V^\top
  $$
  and uses rows of `V.T` as components.
- `svd_solver="auto"` uses covariance/eigh when `n_samples >= n_features`, otherwise full SVD.
- `svd_solver="randomized"` draws a random projection, performs power iterations, factorizes the smaller projected matrix, and keeps the leading right singular vectors.
- Explained variance is computed as
  $$
  \operatorname{explained\_variance}_j = \frac{s_j^2}{n - 1}.
  $$
- `explained_variance_ratio_` divides each retained variance by total centered variance.

## Parameters

- `n_components`: number of principal components to keep; `None` keeps all feasible components.
- `svd_solver`: `"auto"`, `"full"`, `"covariance"`, or `"randomized"`.
- `whiten`: if `True`, transformed coordinates are divided by `sqrt(explained_variance_)`.
- `random_state`, `n_oversamples`, `iterated_power`: randomized solver controls.
- `device`: `"auto"`, `"cpu"`, `"cuda"`, or `"torch"`.

## A small CPU example

<!-- learner-example: pca -->
```python
import numpy as np
from statgpu.unsupervised import PCA

rng = np.random.default_rng(0)
X = rng.normal(size=(80, 4))
X[:, 3] = X[:, 0] + 0.05 * rng.normal(size=80)
model = PCA(n_components=3, svd_solver="full", device="cpu")
Z = model.fit_transform(X)
X_hat = model.inverse_transform(Z)
print(Z.shape, np.mean((X - X_hat) ** 2))
```

The coordinate array has shape `(80, 3)`; reconstruction is `(80, 4)`. Rows of `components_` give feature directions, not labels. A component and its negative describe the same direction.

For a supported GPU installation, construct a new estimator with `device="cuda"` (CuPy) or `device="torch"` (Torch CUDA). Arrays generally stay on that backend; see the [API reference](api-reference.md#pca) for output ownership and host-side work. An unavailable explicit GPU raises an error.

## Approximation and interpretation

PCA has no statistical strict inference mode. Exactness refers to the decomposition:

- `full` and `covariance` are exact dense solvers up to floating-point error.
- `randomized` is approximate and controlled by `random_state`, `n_oversamples`, and `iterated_power`.
- Component signs are not identifiable; `v` and `-v` describe the same component.

## Outputs

- `components_`
- `mean_`
- `explained_variance_`
- `explained_variance_ratio_`
- `singular_values_`
- `n_components_`
- `n_features_in_`

## FAQ

**Why do components differ by sign from sklearn?**
Eigenvectors and singular vectors are sign-indeterminate. Validation must compare subspaces or use sign-aware comparisons.

**What does whitening do?**
It scales transformed scores by `1 / sqrt(explained_variance_)`, producing unit-variance component scores under the fitted model.


## Numerical and lifecycle cautions

The covariance solver forms uncentered second moments and subtracts the squared mean. Large common offsets relative to variation can cause severe cancellation, including incorrect zero variance ratios. Use `svd_solver="full"` for such data, or subtract a training-derived offset before fitting and apply it to later rows. Whitening zero-variance components can return non-finite coordinates; reduce the rank or disable whitening. `inverse_transform` does not validate finiteness, so validate supplied coordinates yourself.

## Complete API reference

Constructor defaults, all public methods, output shapes, and restrictions are listed in the [PCA API reference](api-reference.md#pca).

## References

- Pearson, K. (1901). On lines and planes of closest fit to systems of points in space. *The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science*, Series 6, 2(11), 559-572. https://doi.org/10.1080/14786440109462720
- Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer Series in Statistics. Springer. https://doi.org/10.1007/b98835
- Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. *SIAM Review*, 53(2), 217-288. https://doi.org/10.1137/090771806
