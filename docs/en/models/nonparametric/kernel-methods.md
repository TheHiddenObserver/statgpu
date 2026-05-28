# Kernel Methods

> Language: English
> Last updated: 2026-05-28
> This page: Model documentation
> Switch: [Chinese](../../../models/nonparametric/kernel-methods.md)

Language switch: [Chinese](../../../models/nonparametric/kernel-methods.md)

## Overview

The kernel methods module provides kernel ridge regression (`KernelRidge`), cross-validated kernel ridge regression (`KernelRidgeCV`), and six kernel functions (RBF, polynomial, linear, Laplacian, sigmoid, cosine). Both estimators accept a `kernel` parameter that selects one of the built-in kernels or a user-supplied callable. All computation dispatches through a backend-agnostic array interface and supports CPU (NumPy), CuPy, and PyTorch backends, including automatic CUDA acceleration for `KernelRidgeCV`.

## Path

```
statgpu.nonparametric.kernel_methods.KernelRidge
statgpu.nonparametric.kernel_methods.KernelRidgeCV
statgpu.nonparametric.kernel_methods.pairwise_kernels
```

Individual kernel functions are also importable:

```
statgpu.nonparametric.kernel_methods.rbf_kernel
statgpu.nonparametric.kernel_methods.polynomial_kernel
statgpu.nonparametric.kernel_methods.linear_kernel
statgpu.nonparametric.kernel_methods.laplacian_kernel
statgpu.nonparametric.kernel_methods.sigmoid_kernel
statgpu.nonparametric.kernel_methods.cosine_kernel
```

## Objective Function

**KernelRidge** solves the kernel ridge regression dual problem. Given an \(n \times n\) kernel matrix \(K\) computed from the training data, the objective in the dual space is:

\[
\min_{\boldsymbol{\alpha}} \| \mathbf{y} - K \boldsymbol{\alpha} \|_2^2 + \lambda \| \boldsymbol{\alpha} \|_2^2
\]

where \(\lambda\) is the regularization strength (`alpha`). The closed-form solution is:

\[
\boldsymbol{\alpha} = (K + \lambda I)^{-1} \mathbf{y}
\]

**KernelRidgeCV** uses eigendecomposition \(K = Q \Lambda Q^\top\) of the kernel matrix to efficiently evaluate the solution across a grid of regularization parameters without re-solving the linear system for each value. The solution for any \(\lambda\) is:

\[
\boldsymbol{\alpha}(\lambda) = Q \, \text{diag}\!\left(\frac{1}{\lambda_i + \lambda}\right) Q^\top \mathbf{y}
\]

where \(\lambda_i\) are the eigenvalues of \(K\). Cross-validation MSE is computed for each \(\lambda\) in the grid and the value that minimizes the mean CV MSE is selected.

## Estimating Equation

**KernelRidge**: The first-order condition of the dual problem yields the linear system

\[
(K + \lambda I) \boldsymbol{\alpha} = \mathbf{y}
\]

which is solved directly via `xp.linalg.solve`. Predictions for new data \(X_{\text{test}}\) are computed as:

\[
\hat{\mathbf{y}} = K_{\text{test}} \boldsymbol{\alpha}
\]

where \(K_{\text{test}}\) is the kernel matrix between the test and training data.

**KernelRidgeCV**: For each cross-validation fold, the training kernel matrix is eigendecomposed once. The dual coefficients for all alpha values are then computed in a single vectorized operation:

\[
\boldsymbol{\alpha}(\lambda) = Q \, \text{diag}\!\left(\frac{1}{\lambda_i + \lambda}\right) Q^\top \mathbf{y}_{\text{train}}
\]

On the torch CUDA backend, this alpha sweep is fully vectorized across all alpha values using batched matrix operations, avoiding any Python-level loop over the alpha grid.

## Covariance/Inference

Kernel methods are non-parametric and do not produce coefficient-level inference (no standard errors, t-values, or p-values). Model quality is assessed through:

- **R-squared**: the coefficient of determination \(R^2 = 1 - \text{SS}_{\text{res}} / \text{SS}_{\text{tot}}\) computed by the `score` method.
- **Cross-validation MSE** (KernelRidgeCV): the mean squared error averaged over K folds, stored in `cv_results_["mean_mse"]`.

## Parameters

**KernelRidge**:

| Parameter | Default | Description |
|---|---:|---|
| `alpha` | `1.0` | Regularization strength (\(\lambda\)) |
| `kernel` | `"rbf"` | Kernel function: `rbf`, `gaussian`, `linear`, `polynomial`, `poly`, `laplacian`, `sigmoid`, `cosine`, or a callable |
| `gamma` | `None` | Kernel coefficient for rbf/polynomial/laplacian/sigmoid. Defaults to `1 / n_features` |
| `degree` | `3` | Degree for the polynomial kernel |
| `coef0` | `1` | Independent term for polynomial and sigmoid kernels |
| `kernel_params` | `None` | Additional keyword arguments passed to the kernel function |
| `device` | `"auto"` | `cpu` / `cuda` / `auto` |
| `n_jobs` | `None` | Not used; kept for API compatibility |

**KernelRidgeCV** (inherits all kernel-related parameters above, plus):

| Parameter | Default | Description |
|---|---:|---|
| `alphas` | `None` | Array of regularization strengths to evaluate. Auto-generated as a 100-point log-spaced grid if `None` |
| `cv` | `5` | Number of cross-validation folds |
| `random_state` | `None` | Random state for fold shuffling |

**Kernel functions**:

| Function | Formula |
|---|---|
| `rbf_kernel` | \(K(x, y) = \exp(-\gamma \|x - y\|^2)\) |
| `polynomial_kernel` | \(K(x, y) = (\gamma \, x^\top y + c_0)^d\) |
| `linear_kernel` | \(K(x, y) = x^\top y\) |
| `laplacian_kernel` | \(K(x, y) = \exp(-\gamma \|x - y\|_1)\) |
| `sigmoid_kernel` | \(K(x, y) = \tanh(\gamma \, x^\top y + c_0)\) |
| `cosine_kernel` | \(K(x, y) = \frac{x^\top y}{\|x\| \, \|y\|}\) |

All kernel functions accept an optional `xp` argument for backend dispatch (numpy/cupy/torch). When `xp` is `None`, they default to NumPy.

## CPU+GPU Examples

```python
from statgpu.nonparametric.kernel_methods import KernelRidge, KernelRidgeCV
import numpy as np

X = np.random.randn(500, 10)
y = X @ np.random.randn(10) + 0.1 * np.random.randn(500)

# CPU
kr = KernelRidge(alpha=1.0, kernel="rbf", device="cpu")
kr.fit(X, y)
print(f"R^2: {kr.score(X, y):.4f}")

# GPU with CV
kr_cv = KernelRidgeCV(cv=5, kernel="rbf", device="cuda")
kr_cv.fit(X, y)
print(f"Best alpha: {kr_cv.alpha_:.6f}, R^2: {kr_cv.best_score_:.4f}")

# Predict
y_pred = kr_cv.predict(X)

# Inspect CV results
print(f"Alpha grid size: {len(kr_cv.cv_results_['alphas'])}")
print(f"CV results shape: {kr_cv.cv_results_['mean_mse'].shape}")
```

Using a custom kernel:

```python
from statgpu.nonparametric.kernel_methods import KernelRidge

# Polynomial kernel with custom parameters
kr_poly = KernelRidge(alpha=0.1, kernel="polynomial", degree=4, coef0=2.0, device="auto")
kr_poly.fit(X, y)

# Laplacian kernel
kr_lap = KernelRidge(alpha=1.0, kernel="laplacian", gamma=0.5, device="cpu")
kr_lap.fit(X, y)

# User-defined kernel function
def my_kernel(X, Y=None, xp=None):
    if xp is None:
        xp = np
    if Y is None:
        Y = X
    return xp.tanh(X @ Y.T + 1.0)

kr_custom = KernelRidge(alpha=1.0, kernel=my_kernel, device="cpu")
kr_custom.fit(X, y)
```

## strict/approx difference

There is no strict/approx mode distinction in the kernel methods module. The closed-form dual solution is computed directly with no iterative approximation.

`KernelRidgeCV` auto-generates a log-spaced alpha grid of 100 points when `alphas` is not provided. The grid spans from `max(lambda_min * 1e-3, 1e-8)` to `max(lambda_max * 10, 1)`, where `lambda_min` and `lambda_max` are the extreme eigenvalues of the training kernel matrix. Users can supply a custom `alphas` array to override this behavior.

## Outputs

**KernelRidge fitted attributes**:

| Attribute | Shape | Description |
|---|---|---|
| `dual_coef_` | `(n_samples,)` or `(n_samples, n_targets)` | Dual coefficients in kernel space |
| `X_fit_` | `(n_samples, n_features)` | Training data stored for prediction |

**KernelRidgeCV fitted attributes**:

| Attribute | Shape | Description |
|---|---|---|
| `alpha_` | scalar | Best regularization parameter selected by CV |
| `best_score_` | scalar | R-squared score corresponding to the best alpha |
| `cv_results_` | dict | Dictionary with keys: `alphas`, `mean_mse`, `mse_table`, `best_alpha`, `best_score` |
| `estimator_` | `KernelRidge` | Fitted `KernelRidge` instance using the best alpha |
| `dual_coef_` | `(n_samples,)` or `(n_samples, n_targets)` | Shortcut to `estimator_.dual_coef_` |
| `X_fit_` | `(n_samples, n_features)` | Shortcut to `estimator_.X_fit_` |

**Methods** (both classes):

| Method | Description |
|---|---|
| `fit(X, y)` | Fit the model. Returns `self`. |
| `predict(X)` | Predict targets for new data. |
| `score(X, y)` | Return the coefficient of determination R-squared. |

## FAQ

**Q: How is the alpha grid generated when I do not provide one?**
A: `KernelRidgeCV` computes the eigenvalues of the training kernel matrix and creates a 100-point log-spaced grid from `max(lambda_min * 1e-3, 1e-8)` to `max(lambda_max * 10, 1)`. If the range is degenerate (alpha_min >= alpha_max), the grid is set to span `alpha_max * 1e-4` to `alpha_max`.

**Q: What is the GPU advantage for KernelRidgeCV?**
A: When using the torch CUDA backend, the eigendecomposition of each fold's training kernel matrix and the entire alpha sweep (computing dual coefficients and predictions for all 100 alpha values) run entirely on the GPU using batched matrix operations. The numpy/CuPy path falls back to a Python-level loop over alpha values.

**Q: Can I use a custom kernel function?**
A: Yes. Pass any callable as the `kernel` parameter. The callable should accept `(X, Y, xp=None, **kwargs)` and return a kernel matrix of shape `(n_samples_X, n_samples_Y)`. The `xp` argument provides the array module for backend dispatch.

**Q: How does `gamma` default when I do not specify it?**
A: When `gamma` is `None`, the kernel functions default to `1 / n_features`, following the convention used by scikit-learn.

**Q: Does KernelRidge support multi-output targets?**
A: Yes. If `y` has shape `(n_samples, n_targets)`, both `KernelRidge` and `KernelRidgeCV` fit all targets simultaneously. The dual coefficients will have shape `(n_samples, n_targets)`.

## External Validation

KernelRidge results are validated against `sklearn.kernel_ridge.KernelRidge` with relative error below \(10^{-10}\) for all supported kernel types. Consistency checks are maintained in the test suite covering RBF, polynomial, linear, Laplacian, sigmoid, and cosine kernels across both CPU and GPU backends.

## References

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. Chapter 6. [https://hastie.su.domains/ElemStatLearn/](https://hastie.su.domains/ElemStatLearn/)
- Saunders, C., Gammerman, A., & Vovk, V. (1998). Ridge regression learning algorithm in dual variables. *Proceedings of the 15th International Conference on Machine Learning (ICML)*, 515-521.
