# Kernel Methods

> Language: English  
> Last updated: 2026-07-24  
> Switch: [Chinese](../../cn/models/kernel-methods.md)

## Overview

The kernel-methods module provides:

- `KernelRidge`
- `KernelRidgeCV`
- `KernelPCA`
- `Nystroem`
- `pairwise_kernels`
- RBF, polynomial, linear, Laplacian, sigmoid, cosine, and chi-squared kernels

The public implementations expose NumPy, CuPy, and Torch execution paths where
supported by the selected estimator and kernel.

## Paths

```text
statgpu.nonparametric.kernel_methods.KernelRidge
statgpu.nonparametric.kernel_methods.KernelRidgeCV
statgpu.nonparametric.kernel_methods.KernelPCA
statgpu.nonparametric.kernel_methods.Nystroem
statgpu.nonparametric.kernel_methods.pairwise_kernels
```

Individual kernel functions are also importable from
`statgpu.nonparametric.kernel_methods`.

## Kernel Ridge Regression

Given a training kernel matrix $K$, kernel ridge regression solves the dual
system

$$
(K+\alpha I)c=y.
$$

Equivalently, the dual objective is

$$
\min_c \lVert y-Kc\rVert_2^2+\alpha\lVert c\rVert_2^2.
$$

Predictions for test observations are

$$
\hat y_{test}=K(X_{test},X_{train})c.
$$

`KernelRidge` solves the regularized linear system directly. Multi-output
responses are supported when the fitted implementation receives a compatible
response matrix.

## Kernel Ridge Cross-Validation

`KernelRidgeCV` evaluates a grid of regularization parameters across CV folds and
refits the selected value on the complete dataset. Backend-specific
implementations may reuse a kernel eigendecomposition or vectorize the alpha
sweep rather than solving every system independently.

The selected alpha is exposed as `alpha_`. CV diagnostics are stored in
`cv_results_`; consult the fitted object for the exact fields produced by the
selected path.

## Kernel PCA

For the centered kernel matrix $\widetilde K$, Kernel PCA eigendecomposes

$$
\widetilde K = V\Lambda V^\top.
$$

The leading eigenvectors define nonlinear components. Transforming new data
requires computing the test-to-training kernel, applying the training centering
quantities, and projecting onto the retained components.

## Nystroem Approximation

Nystroem selects $m$ landmark observations and forms an explicit approximate
feature map. If the landmark kernel is

$$
K_{mm}=V\Lambda V^\top,
$$

then the transformed features have the form

$$
Z=K_{nm}V\Lambda^{-1/2}.
$$

This replaces a full $n\times n$ kernel representation with an $n\times m$
feature matrix when $m\ll n$.

## Built-In Kernels

| Kernel | Definition |
|---|---|
| RBF | $\exp(-\gamma\lVert x-y\rVert_2^2)$ |
| Polynomial | $(\gamma x^\top y+c_0)^d$ |
| Linear | $x^\top y$ |
| Laplacian | $\exp(-\gamma\lVert x-y\rVert_1)$ |
| Sigmoid | $\tanh(\gamma x^\top y+c_0)$ |
| Cosine | $x^\top y/(\lVert x\rVert\lVert y\rVert)$ |
| Chi-squared | $\exp\{-\gamma\sum_j (x_j-y_j)^2/(x_j+y_j)\}$ |

The chi-squared kernel requires non-negative input features. A callable kernel
may be supplied where accepted by the estimator; it must return an array on the
requested backend and obey the expected pairwise-kernel shape.

## Parameters

### KernelRidge

| Parameter | Default | Description |
|---|---:|---|
| `alpha` | `1.0` | Ridge regularization strength |
| `kernel` | `"rbf"` | Built-in kernel name or callable |
| `gamma` | `None` | Kernel coefficient where applicable |
| `degree` | `3` | Polynomial degree |
| `coef0` | `1` | Polynomial or sigmoid intercept term |
| `kernel_params` | `None` | Additional callable-kernel parameters |
| `device` | `"auto"` | `"cpu"`, `"cuda"` (CuPy), `"torch"`, or `"auto"` |
| `n_jobs` | `None` | Reserved where parallel execution is not implemented |

### KernelRidgeCV

In addition to the kernel parameters:

| Parameter | Default | Description |
|---|---:|---|
| `alphas` | `None` | Candidate regularization strengths |
| `cv` | `5` | Number of CV folds |
| `random_state` | `None` | Fold random state where used |

### KernelPCA

Common parameters include `n_components`, `kernel`, `gamma`, `degree`, `coef0`,
`alpha`, `eigen_solver`, and `device`.

### Nystroem

Common parameters include `kernel`, `n_components`, `gamma`, `degree`, `coef0`,
`random_state`, and `device`.

Consult class docstrings for exact aliases, accepted callables, and parameter
validation.

## Fitted Attributes and Outputs

### KernelRidge

Typical fitted state includes the training observations, dual coefficients,
resolved kernel parameters, and fitted feature counts. `predict(X)` returns the
same backend family as the maintained estimator path. `score(X, y)` reports the
coefficient of determination.

### KernelRidgeCV

In addition to the final refitted state, the model exposes `alpha_`,
`best_score_` where implemented, and `cv_results_`.

### KernelPCA

Fitted outputs include retained eigenvalues/eigenvectors or normalized dual
components, training-kernel centering quantities, and transformed components
from `fit_transform` or `transform`.

### Nystroem

Fitted state includes landmark indices or components and the normalization matrix.
`transform(X)` returns the explicit approximate feature map.

## CPU and GPU Examples

### NumPy

```python
import numpy as np
from statgpu.nonparametric.kernel_methods import (
    KernelRidge,
    KernelRidgeCV,
    KernelPCA,
    Nystroem,
)

rng = np.random.default_rng(42)
X = rng.normal(size=(500, 10))
y = X[:, 0] - 0.5 * X[:, 1] + rng.normal(scale=0.1, size=500)

kr = KernelRidge(alpha=1.0, kernel="rbf", device="cpu").fit(X, y)
print(kr.score(X, y))

kr_cv = KernelRidgeCV(kernel="rbf", cv=5, device="cpu").fit(X, y)
print(kr_cv.alpha_)

kpca = KernelPCA(n_components=3, kernel="rbf", device="cpu")
X_kpca = kpca.fit_transform(X)

nystroem = Nystroem(kernel="rbf", n_components=50, random_state=42)
X_features = nystroem.fit_transform(X)
```

### CuPy

```python
import cupy as cp
from statgpu.nonparametric.kernel_methods import KernelRidgeCV

X = cp.random.randn(500, 10, dtype=cp.float64)
y = X[:, 0] - 0.5 * X[:, 1]
model = KernelRidgeCV(kernel="rbf", cv=5, device="cuda").fit(X, y)
```

### Torch CUDA

```python
import torch
from statgpu.nonparametric.kernel_methods import KernelRidgeCV

X = torch.randn(500, 10, device="cuda", dtype=torch.float64)
y = X[:, 0] - 0.5 * X[:, 1]
model = KernelRidgeCV(kernel="rbf", cv=5, device="torch").fit(X, y)
```

`device="cuda"` selects CuPy; `device="torch"` selects Torch.

## Backend and Execution Boundaries

Pairwise kernel construction, regularized solves, eigendecompositions, projections,
and transformed feature arrays remain on the selected backend where supported.
Small random-index metadata, CV fold indices, parameter bookkeeping, and scalar
scores may be represented on CPU. Explicit device requests do not silently select
another backend.

`KernelPCA` and `Nystroem` reject NaN/Inf during fitting and transformation on
maintained validation paths. Kernel-specific domain checks, such as non-negative
inputs for the chi-squared kernel, fail explicitly.

## Inference Semantics

Kernel methods do not currently expose coefficient-level standard errors,
hypothesis tests, or confidence intervals. Model quality is evaluated through
prediction scores, cross-validation loss, embedding properties, reconstruction
or approximation diagnostics, and application-specific validation.

There is no strict/approximate inference mode distinction in this module.
Nystroem is an explicit low-rank kernel approximation, not a silent fallback for
an exact kernel estimator.

## Complexity and Performance Notes

- Exact kernel methods materialize an $n\times n$ training kernel and therefore
  have quadratic memory cost.
- Direct kernel-ridge solves and dense eigendecompositions have cubic worst-case
  arithmetic cost in the number of training observations.
- `KernelRidgeCV` may reuse decompositions across alpha values, but CV still
  multiplies work across folds.
- `Nystroem` reduces kernel storage to $O(nm)$ plus landmark linear algebra.
- GPU speed depends on sample size, dtype, kernel, synchronization, and available
  memory; small problems may be faster on CPU.

## Limitations and Failure Modes

- Kernel matrices can become poorly conditioned; increase `alpha` or adjust the
  kernel scale when necessary.
- RBF-like kernels are sensitive to `gamma`.
- Chi-squared kernels require non-negative inputs.
- Dense exact kernel methods may exhaust device memory for large $n$.
- User-supplied kernels are responsible for backend, dtype, shape, and symmetry
  contracts required by the selected estimator.
- `KernelRidgeCV` can be expensive for large fold and alpha grids.

## External Validation

Maintained tests cover NumPy/Torch parity, finite-input validation, rank-deficient
kernel safeguards, CV alpha selection and refit behavior, kernel-domain errors,
and output-backend preservation. Accuracy and performance claims remain scoped to
the exact estimator, backend, hardware, and commit recorded by the corresponding
test or benchmark artifact.

## FAQ

### Should I use KernelRidge or Nystroem plus a linear model?

Use exact Kernel Ridge when the training kernel fits comfortably in memory and the
exact kernel representation is important. Use Nystroem when a controlled low-rank
feature approximation is preferable for scale or downstream reuse.

### Why can a GPU kernel method be slower than CPU?

Kernel construction and linear algebra must be large enough to amortize device
launch, synchronization, and memory-transfer overhead.

### Does `device="auto"` silently change an explicit request?

No. `"auto"` is itself an automatic selection request. An explicit `"cuda"` or
`"torch"` request fails if that backend is unavailable.

### Are KernelPCA components directly comparable across separate fits?

Eigenvector signs and bases within repeated or nearly repeated eigenspaces are not
uniquely identified. Compare represented subspaces or downstream quantities when
that ambiguity matters.

## References

- Schölkopf, B., Smola, A., & Müller, K.-R. (1998). Nonlinear component analysis
  as a kernel eigenvalue problem.
- Williams, C. K. I., & Seeger, M. (2001). Using the Nystroem method to speed up
  kernel machines.
- Shawe-Taylor, J., & Cristianini, N. (2004). *Kernel Methods for Pattern Analysis*.
