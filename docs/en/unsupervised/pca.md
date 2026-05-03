# PCA

> Language: English
> Last updated: 2026-05-02
> Switch: [Chinese](../../unsupervised/pca.md)

## Overview

`PCA` estimates an orthonormal low-dimensional basis that captures the largest variance directions of centered dense data. It supports CPU, CuPy/CUDA, and Torch CUDA backends.

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

## CPU+GPU Examples

```python
import numpy as np
from statgpu.unsupervised import PCA

X = np.random.default_rng(0).normal(size=(2000, 50))

pca_cpu = PCA(n_components=10, svd_solver="covariance", device="cpu")
Z_cpu = pca_cpu.fit_transform(X)

pca_gpu = PCA(n_components=10, svd_solver="covariance", device="cuda")
Z_gpu = pca_gpu.fit_transform(X)
```

## Strict/Approx Difference

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

## External Validation

- Tests: `dev/tests/test_unsupervised_pca.py`.
- Benchmark: `dev/benchmarks/benchmark_unsupervised.py`.
- Baselines: sklearn PCA, statsmodels/R PCA comparisons from the earlier unsupervised matrix where available.
- Latest Phase 2 artifact summary: `results/unsupervised_phase2_verify_summary_20260502_210000.md`.

## References

- Pearson, K. (1901). On lines and planes of closest fit to systems of points in space. *The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science*, Series 6, 2(11), 559-572. https://doi.org/10.1080/14786440109462720
- Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer Series in Statistics. Springer. https://doi.org/10.1007/b98835
- Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. *SIAM Review*, 53(2), 217-288. https://doi.org/10.1137/090771806
