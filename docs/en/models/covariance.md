# Covariance Estimation

> Language: English  
> Last updated: 2026-07-24  
> Switch: [Chinese](../../cn/models/covariance.md)

## Overview

The `statgpu.covariance` module provides:

- `EmpiricalCovariance`
- `LedoitWolf`
- `OAS`
- `ShrunkCovariance`
- `MinCovDet`
- `GraphicalLasso`
- `GraphicalLassoCV`

The estimators expose NumPy, CuPy, and Torch execution paths. Backend availability
means that the public path exists; numerical and performance validation remains scoped
to the exact estimator, backend, hardware, and commit tested.

## Core Definitions

The empirical covariance of centered observations is

$$
\hat S = \frac{1}{n}X^\top X.
$$

Shrinkage estimators use

$$
\hat\Sigma = (1-\alpha)\hat S + \alpha\mu I,
\qquad
\mu = \frac{\operatorname{tr}(\hat S)}{p}.
$$

`LedoitWolf` and `OAS` estimate the shrinkage intensity analytically;
`ShrunkCovariance` uses the user-supplied `shrinkage` value.

`GraphicalLasso` estimates a sparse precision matrix by solving

$$
\max_{\Theta\succ 0}
\left\{
\log\det(\Theta)-\operatorname{tr}(S\Theta)
-\alpha\lVert\Theta\rVert_{1,\mathrm{off}}
\right\}.
$$

`MinCovDet` uses FAST-MCD concentration steps followed by reweighting.

## Common Parameters

| Parameter | Default | Description |
|---|---:|---|
| `assume_centered` | `False` | Skip mean estimation when the data is already centered |
| `device` | `"auto"` | `"cpu"`, `"cuda"` (CuPy), `"torch"`, or `"auto"` |
| `n_jobs` | `None` | Reserved for API compatibility where not implemented |

Estimator-specific parameters include `shrinkage`, `support_fraction`,
`random_state`, `alpha`, `alphas`, `cv`, `max_iter`, and `tol`.

## Fitted Attributes

Common outputs include:

- `covariance_`
- `precision_`
- `location_`
- `n_samples_`
- `n_features_`

Shrinkage estimators expose `shrinkage_`; robust and sparse estimators expose
additional support or convergence attributes documented by their class API.

## Examples

### NumPy

```python
import numpy as np
from statgpu.covariance import LedoitWolf

X = np.random.randn(500, 10)
model = LedoitWolf(device="cpu").fit(X)
print(model.covariance_.shape)
print(model.score(X))
```

### CuPy

```python
import cupy as cp
from statgpu.covariance import LedoitWolf

X_cupy = cp.random.randn(500, 10, dtype=cp.float64)
model_cupy = LedoitWolf(device="cuda").fit(X_cupy)
```

### Torch CUDA

```python
import torch
from statgpu.covariance import LedoitWolf

X_torch = torch.randn(500, 10, device="cuda", dtype=torch.float64)
model_torch = LedoitWolf(device="torch").fit(X_torch)
```

`device="cuda"` selects the CuPy backend. Use `device="torch"` for Torch tensors;
the two explicit GPU device values are not interchangeable.

## Execution Boundaries

Centering, covariance updates, linear algebra, FAST-MCD concentration steps, and
Graphical Lasso coordinate updates remain on the selected numerical backend where
implemented. Small integer index metadata, convergence scalars, and scalar
chi-squared distribution evaluations may cross to CPU when the backend does not
provide an equivalent operation.

Input validation for empty feature dimensions and NaN/Inf values occurs before
centering or inversion so invalid data is not misreported as a singular covariance
problem.

## Validation

This page does not maintain a global `pending` or `complete` GPU status. Physical-GPU
results and benchmark evidence belong to the corresponding maintained tests, release
records, and hardware-specific artifacts.

## References

- Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices.
- Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O. (2010). Shrinkage algorithms for MMSE covariance estimation.
- Rousseeuw, P. J., & Van Driessen, K. (1999). A fast algorithm for the minimum covariance determinant estimator.
- Friedman, J., Hastie, T., & Tibshirani, R. (2008). Sparse inverse covariance estimation with the graphical lasso.
