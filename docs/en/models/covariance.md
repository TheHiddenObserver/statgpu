# Covariance Estimation

> Language: English  
> Last updated: 2026-07-24  
> Switch: [Chinese](../../cn/models/covariance.md)

## Overview

The `statgpu.covariance` module provides seven covariance and precision-matrix
estimators:

- `EmpiricalCovariance`
- `LedoitWolf`
- `OAS`
- `ShrunkCovariance`
- `MinCovDet`
- `GraphicalLasso`
- `GraphicalLassoCV`

The public estimators expose NumPy, CuPy, and Torch execution paths. Backend
availability means that the public path exists; numerical and performance claims
remain scoped to the exact estimator, backend, hardware, and commit tested.

## Paths

```python
from statgpu.covariance import (
    EmpiricalCovariance,
    LedoitWolf,
    OAS,
    ShrunkCovariance,
    MinCovDet,
    GraphicalLasso,
    GraphicalLassoCV,
)
```

## Objectives

### Empirical covariance

For centered observations $X\in\mathbb R^{n\times p}$,

$$
\hat S = \frac{1}{n}X^\top X.
$$

Unless `assume_centered=True`, the column mean is estimated and removed before
forming the covariance matrix.

### Shrinkage estimators

`LedoitWolf`, `OAS`, and `ShrunkCovariance` use

$$
\hat\Sigma=(1-\alpha)\hat S+\alpha\mu I,
\qquad
\mu=\frac{\operatorname{tr}(\hat S)}{p}.
$$

`LedoitWolf` and `OAS` estimate $\alpha$ analytically. `ShrunkCovariance` uses the
user-supplied `shrinkage` value.

### Minimum covariance determinant

`MinCovDet` searches for a concentrated subset with a small covariance
determinant, applies FAST-MCD concentration steps, and then reweights observations
using robust Mahalanobis distances. It is intended for covariance estimation in
the presence of multivariate outliers.

### Graphical Lasso

`GraphicalLasso` estimates a sparse precision matrix $\Theta$ by solving

$$
\max_{\Theta\succ0}
\left\{
\log\det(\Theta)-\operatorname{tr}(S\Theta)
-\alpha\lVert\Theta\rVert_{1,\mathrm{off}}
\right\}.
$$

The precision diagonal is not L1-penalized. `GraphicalLassoCV` evaluates an alpha
grid by cross-validation and refits the selected model on the complete dataset.

## Estimation Algorithms

- `EmpiricalCovariance` computes the sample covariance directly and obtains a
  precision matrix by inversion, using stabilization only when the exact inverse
  fails or is non-finite.
- `LedoitWolf` and `OAS` evaluate closed-form shrinkage intensities and then invert
  the shrunk covariance.
- `ShrunkCovariance` follows the same direct path with a fixed intensity.
- `MinCovDet` uses repeated initial subsets, concentration steps, consistency
  correction, and reweighting.
- `GraphicalLasso` uses block coordinate updates with soft-thresholded inner
  regressions.
- `GraphicalLassoCV` fits the Graphical Lasso across folds and candidate alpha
  values before the final refit.

## Common Parameters

| Parameter | Default | Description |
|---|---:|---|
| `assume_centered` | `False` | Treat input as already centered |
| `device` | `"auto"` | `"cpu"`, `"cuda"` (CuPy), `"torch"`, or `"auto"` |
| `n_jobs` | `None` | Reserved for API compatibility where not implemented |

Estimator-specific parameters include:

| Estimator | Parameters |
|---|---|
| `ShrunkCovariance` | `shrinkage` |
| `MinCovDet` | `support_fraction`, `random_state` |
| `GraphicalLasso` | `alpha`, `max_iter`, `tol` |
| `GraphicalLassoCV` | `alphas`, `cv`, `max_iter`, `tol` |

Consult class docstrings for the exact accepted type and range of each parameter.

## Fitted Attributes and Outputs

Common fitted attributes include:

| Attribute | Description |
|---|---|
| `covariance_` | Estimated covariance matrix |
| `precision_` | Estimated inverse covariance or sparse precision matrix |
| `location_` | Estimated mean vector; zero when centered input is assumed |
| `n_samples_` | Number of fitted observations |
| `n_features_` | Number of fitted features |

Additional attributes include:

- `shrinkage_` for shrinkage estimators;
- `support_`, `raw_location_`, `raw_covariance_`, and robust distances for
  `MinCovDet`;
- `n_iter_` for iterative sparse estimators;
- `alpha_`, CV scores, and the refitted model state for `GraphicalLassoCV`.

Where exposed, `score(X)` evaluates the fitted Gaussian covariance model and
`mahalanobis(X)` returns squared Mahalanobis distances under the fitted location
and precision.

## CPU and GPU Examples

### NumPy

```python
import numpy as np
from statgpu.covariance import LedoitWolf, MinCovDet, GraphicalLassoCV

rng = np.random.default_rng(42)
X = rng.normal(size=(500, 10))

lw = LedoitWolf(device="cpu").fit(X)
print(lw.covariance_.shape, lw.shrinkage_)
print(lw.score(X))

mcd = MinCovDet(random_state=42, device="cpu").fit(X)
print(mcd.support_.sum())

glcv = GraphicalLassoCV(alphas=4, cv=5, device="cpu").fit(X)
print(glcv.alpha_)
```

### CuPy

```python
import cupy as cp
from statgpu.covariance import LedoitWolf

X_cupy = cp.random.randn(500, 10, dtype=cp.float64)
model_cupy = LedoitWolf(device="cuda").fit(X_cupy)
print(model_cupy.covariance_.shape)
```

### Torch CUDA

```python
import torch
from statgpu.covariance import LedoitWolf

X_torch = torch.randn(500, 10, device="cuda", dtype=torch.float64)
model_torch = LedoitWolf(device="torch").fit(X_torch)
print(model_torch.covariance_.shape)
```

`device="cuda"` selects CuPy. Use `device="torch"` for Torch tensors; the two
explicit GPU device values are not interchangeable.

## Covariance, Precision, and Inference Semantics

These classes estimate covariance or precision matrices; they do not generally
expose coefficient-level standard errors or regression p-values. Numerical
uncertainty should be assessed with a method appropriate to the covariance
estimator and application, such as resampling or a downstream model with a
specified inference contract.

A singular or nearly singular empirical covariance may require stabilization for
precision computation. Stabilization is a numerical safeguard and does not turn
a rank-deficient covariance into fully identified information in every direction.

## Backend and Execution Boundaries

Centering, covariance updates, matrix products, linear algebra, FAST-MCD
concentration steps, and Graphical Lasso coordinate updates remain on the
selected numerical backend where implemented. Small integer index metadata,
random-subset bookkeeping, convergence scalars, and scalar chi-squared
distribution evaluations may cross to CPU.

Input validation for empty feature dimensions and NaN/Inf values occurs before
centering or inversion so invalid data is not misreported as a singular covariance
problem.

## Strict and Approximate Behavior

There is no global strict/approximate switch shared by all covariance estimators.
Each estimator uses its documented algorithm. Numerical inversion stabilization,
robust subset search, and CV selection are explicit parts of the corresponding
algorithm rather than silent backend fallbacks.

## Limitations and Failure Modes

- `EmpiricalCovariance` can be poorly conditioned when $p$ is large relative to
  $n$; shrinkage may be preferable.
- `LedoitWolf` and `OAS` shrink toward a scaled identity and may be inappropriate
  when a different structural target is required.
- `MinCovDet` is more expensive than direct covariance estimators and requires
  enough observations for a meaningful support subset.
- `GraphicalLasso` assumes a sparse precision representation and may fail to
  converge for unsuitable alpha or tolerance settings.
- `GraphicalLassoCV` multiplies the fitting cost by the number of folds and alpha
  candidates.
- Explicit GPU requests fail when the requested runtime is unavailable; they do
  not silently execute on CPU.

## External Validation

Maintained tests cover finite-input validation, backend-preserving fitted arrays,
reference comparisons with scientific Python covariance estimators, robust support
semantics, sparse-precision convergence, and CV refit behavior. Hardware-specific
accuracy and performance evidence belongs to the corresponding maintained test or
benchmark artifact.

## FAQ

### Which estimator should I use when $p$ is close to $n$?

A shrinkage estimator such as `LedoitWolf` or `OAS` is usually more stable than the
unregularized empirical covariance.

### Does `MinCovDet` remove observations?

It identifies a robust support and returns robust covariance estimates. Inspect
`support_` and robust distances rather than assuming every observation contributes
equally to the final estimate.

### Why is the Graphical Lasso precision sparse but the covariance dense?

The L1 penalty is applied to off-diagonal precision entries. The inverse of a
sparse precision matrix need not be sparse.

### Can I pass a Torch CUDA tensor with `device="cuda"`?

No. `device="cuda"` denotes the CuPy backend. Use `device="torch"` for a Torch
execution request.

## References

- Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for
  large-dimensional covariance matrices.
- Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O. (2010). Shrinkage algorithms
  for MMSE covariance estimation.
- Rousseeuw, P. J., & Van Driessen, K. (1999). A fast algorithm for the minimum
  covariance determinant estimator.
- Friedman, J., Hastie, T., & Tibshirani, R. (2008). Sparse inverse covariance
  estimation with the graphical lasso.
