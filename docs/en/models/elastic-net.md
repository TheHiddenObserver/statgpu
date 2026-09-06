# Elastic Net

> Language: English  
> Last updated: 2026-09-06<br>
> This page: Model documentation  
> Language switch: [Chinese](../../cn/models/elastic-net.md)

## Overview

`ElasticNet` combines L1 and L2 regularization for linear regression, balancing sparse feature selection (Lasso) and coefficient shrinkage (Ridge). It supports CPU, CuPy GPU, and PyTorch GPU execution. Direct fitting uses one backend-neutral `solver` interface; `device` controls where the computation runs.

## Path

`statgpu.linear_model.ElasticNet`

## Objective Function

The Elastic Net optimization problem is:

$$
\min_{\beta} \frac{1}{2n}\|y - X\beta\|_2^2 + \alpha \cdot \lambda \cdot \|\beta\|_1 + \frac{\alpha}{2} \cdot (1 - \lambda) \cdot \|\beta\|_2^2
$$

where:
- `alpha` (α) controls overall regularization strength
- `l1_ratio` (λ) mixes L1 vs L2: λ=1 gives Lasso, λ=0 gives Ridge
- loss scaling by `1/(2n)` makes the public `alpha` use an average-loss convention

**Note on regularization scaling**: `ElasticNet` and `Ridge` use the same average-loss convention. Therefore, with `l1_ratio=0`, `ElasticNet(alpha)` is equivalent to `Ridge(alpha)` at the same public `alpha`.

## Estimating Equation

The Elastic Net estimator solves the first-order optimality condition:

$$
\frac{1}{n} X^\top (X\hat{\beta} - y) + \alpha(1-\lambda)\hat{\beta} + \alpha\lambda \cdot \partial\|\hat{\beta}\|_1 = 0.
$$

`solver` is the authoritative direct-fit algorithm selector on every backend. Use `device` separately to select CPU/CuPy/Torch execution. The historical `cpu_solver` argument is deprecated and no longer selects a second CPU-specific direct-fit algorithm in the unified engine. See the [penalized solver API migration guide](../guides/penalized-solver-api-migration.md).

## Estimation Algorithm

The normal default is **FISTA** (Fast Iterative Shrinkage-Thresholding Algorithm), a proximal-gradient method with Nesterov acceleration. Other solver values may be available according to the public solver compatibility contract.

### Key Optimization Insight

The L1 and L2 parts are handled by the Elastic Net proximal operator:

```python
# Gradient of the average squared-error term
grad = (X.T @ X @ w - X.T @ y) / n

# Elastic Net proximal step
w = soft_threshold(w_tilde, alpha * l1_ratio * step) / (
    1 + alpha * (1 - l1_ratio) * step
)
```

### Convergence Criteria

Two stopping modes are available through `stopping`:

| Mode | Description |
|------|-------------|
| `coef_delta` | Stop when coefficient movement is below `tol` |
| `kkt` | Stop when the KKT subgradient violation is below the configured tolerance |

Numerical convergence only establishes that the declared optimization problem has been solved to the requested criterion; it is not a separate statistical approximation.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | `1.0` | Overall regularization strength |
| `l1_ratio` | `0.5` | L1 mixing proportion: 0=Ridge, 1=Lasso |
| `fit_intercept` | `True` | Fit an unpenalized intercept |
| `max_iter` | `1000` | Maximum solver iterations |
| `tol` | `1e-4` | Convergence tolerance |
| `stopping` | `"coef_delta"` | `"coef_delta"` or `"kkt"` stopping rule |
| `device` | `"auto"` | `"auto"`, `"cpu"`, `"cuda"` (CuPy), or `"torch"` |
| `n_jobs` | `None` | CPU parallelism where supported |
| `solver` | `"fista"` | Backend-neutral direct-fit optimization method |
| `cpu_solver` | `"fista"` | **Deprecated compatibility parameter**; use `solver` instead |
| `lipschitz_L` | `None` | Optional user-supplied Lipschitz constant |
| `gpu_memory_cleanup` | `False` | Release backend memory pools after fit where supported |
| `compute_inference` | `False` | Compute post-fit coefficient inference |
| `inference_method` | `"debiased"` | `"debiased"`, `"cpu_ols"`, or `"bootstrap"` |
| `cov_type` | `"nonrobust"` | Covariance convention where applicable |
| `hac_maxlags` | `None` | HAC lag count where supported |

The public wrapper does not accept separate `backend`, `warm_start`, or `random_state` constructor parameters. Backend selection is controlled by `device`; a one-fit warm start can be supplied through `fit(initial_coef=...)`.

## CPU/GPU Examples

```python
from statgpu.linear_model import ElasticNet

# CPU: solver selects the algorithm; device selects the backend.
model_cpu = ElasticNet(
    alpha=0.1,
    l1_ratio=0.5,
    device="cpu",
    solver="fista",
)
model_cpu.fit(X, y)
print(f"R²: {model_cpu.score(X, y):.4f}")

# GPU with the same solver interface
model_gpu_cupy = ElasticNet(
    alpha=0.1,
    l1_ratio=0.5,
    device="cuda",
    solver="fista",
    gpu_memory_cleanup=True,
)
model_gpu_cupy.fit(X, y)

model_gpu_torch = ElasticNet(
    alpha=0.1,
    l1_ratio=0.5,
    device="torch",
    solver="fista",
)
model_gpu_torch.fit(X, y)
```

Backend performance depends on sample size, feature dimension, dtype, hardware, data residency, and transfer costs. Benchmark the target workload before selecting a backend solely for speed.

## Covariance/Inference

`ElasticNet` is estimation-only by default. Set `compute_inference=True` to run post-fit inference through the shared penalized-linear inference engine. The default `inference_method="debiased"` uses nodewise Lasso to construct a bias-corrected estimator, standard errors, z statistics, p-values, and confidence intervals. `summary()` is available after inference succeeds.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `compute_inference` | `False` | Enable post-fit coefficient inference |
| `inference_method` | `"debiased"` | `"debiased"`, `"cpu_ols"`, or `"bootstrap"` |
| `cov_type` | `"nonrobust"` | Covariance convention where applicable |
| `hac_maxlags` | `None` | HAC lag count where the selected inference method supports HAC |

Post-selection OLS is heuristic and does not provide general selective-inference coverage. Inference is conditional on selected regularization parameters and does not alter the fitted penalized coefficients.

For `ElasticNetCV`, `compute_inference=True` applies inference only to the final full-data refit after alpha and `l1_ratio` have been selected. Fold models remain estimation-only.

## Solver and Inference Semantics

For a direct `ElasticNet.fit`, **use `solver` on both CPU and GPU**. `device` chooses the execution backend; `solver` chooses the optimization algorithm. `cpu_solver` is a deprecated compatibility argument from the earlier hardware-split API and should not be used for new code.

`compute_inference=False` returns the penalized estimate only. With `compute_inference=True`, the same fitted coefficients are retained and the selected post-fit inference method runs afterward.

## Outputs

After fitting, the following attributes are available:

| Attribute | Description |
|-----------|-------------|
| `coef_` | Estimated coefficients |
| `intercept_` | Fitted intercept |
| `n_iter_` | Number of iterations until convergence |
| `aic` | Akaike Information Criterion when available |
| `bic` | Bayesian Information Criterion when available |

Methods: `fit(X, y)`, `predict(X)`, `score(X, y)`, `summary()`

## Numerical Validation

The maintained regression suite checks agreement across supported backends and reference implementations at tolerances appropriate to each dtype and solver path. Solver API migration behavior is covered by `dev/tests/test_penalized_solver_api_cleanup.py`.

## References

- Zou, H., & Hastie, T. (2005). Regularization and variable selection via the Elastic Net. *Journal of the Royal Statistical Society: Series B*, 67(2), 301-320.
- Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm for linear inverse problems. *SIAM Journal on Imaging Sciences*, 2(1), 183-202.
