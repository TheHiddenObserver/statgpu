# Elastic Net

> Language: English  
> Last updated: 2026-08-05<br>
> This page: Model documentation  
> Language switch: [Chinese](../../cn/models/elastic-net.md)

## Overview

`ElasticNet` combines L1 and L2 regularization for linear regression, enabling a balance between sparse feature selection (Lasso) and coefficient shrinkage (Ridge). It supports CPU, CuPy GPU, and PyTorch GPU backends with configurable device selection.

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
- Loss scaling by `1/(2n)` makes `alpha` interpretation scale-invariant to sample size

**Note on regularization scaling**: `ElasticNet` and `Ridge` both use the same average-loss convention. Therefore, with `l1_ratio=0`, `ElasticNet(alpha)` is equivalent to `Ridge(alpha)`; no sample-size rescaling of the public `alpha` is required.

## Estimating Equation

The Elastic Net estimator solves the following first-order optimality (KKT) condition:

$$
\frac{1}{n} X^\top (X\hat{\beta} - y) + \alpha(1-\lambda)\hat{\beta} + \alpha\lambda \cdot \partial\|\hat{\beta}\|_1 = 0
$$

where $\partial\|\hat{\beta}\|_1$ is the subdifferential of the L1 norm:
- For $\hat{\beta}_j \neq 0$: $\text{sign}(\hat{\beta}_j)$
- For $\hat{\beta}_j = 0$: any value in $[-1, 1]$

At convergence, the KKT residual (subgradient violation) satisfies:
$$
\left| \frac{1}{n} X_j^\top(y - X\hat{\beta}) - \alpha(1-\lambda)\hat{\beta}_j \right| \leq \alpha\lambda \quad \forall j
$$

## Estimation Algorithm

Elastic Net is solved via **FISTA** (Fast Iterative Shrinkage-Thresholding Algorithm), a proximal gradient method with Nesterov momentum acceleration.

### Key Optimization Insight

The L2 regularization term is handled **only in the proximal step**, not in gradient computation:

```python
# Gradient of RSS only (L2 handled separately)
grad = (X.T @ X @ w - X.T @ y) / n

# Proximal step with soft thresholding and L2 scaling
w = soft_threshold(w_tilde, alpha * l1_ratio * step) / (1 + alpha * (1 - l1_ratio) * step)
```

This avoids redundant computation and improves numerical stability.

### Convergence Criteria

Two stopping modes available via `stopping` parameter:

| Mode | Description |
|------|-------------|
| `coef_delta` | Stop when `||w_new - w_old||_∞ < tol` |
| `kkt` | Stop when KKT subgradient violation < tol |

For `kkt` mode, the optimality condition is:
- For non-zero coefficients: `|∇f + α(1-λ)w + αλ·sign(w)| < tol`
- For zero coefficients: `|∇f + α(1-λ)w| ≤ αλ`

**Note**: KKT violation ~1e-2 is acceptable for numerical solutions; exact zero is not required.

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
| `solver` | `"fista"` | Backend-aware optimization method |
| `cpu_solver` | `"fista"` | CPU solver override |
| `lipschitz_L` | `None` | Optional user-supplied Lipschitz constant |
| `gpu_memory_cleanup` | `False` | Release backend memory pools after fit where supported |
| `compute_inference` | `False` | Compute post-fit coefficient inference |
| `inference_method` | `"debiased"` | `"debiased"`, `"cpu_ols"`, or `"bootstrap"` |
| `cov_type` | `"nonrobust"` | Covariance convention where applicable |
| `hac_maxlags` | `None` | HAC lag count where supported |

The public wrapper does not accept separate `backend`, `warm_start`, or
`random_state` constructor parameters. Backend selection is controlled by
`device`; a one-fit warm start can be supplied through `fit(initial_coef=...)`.

## CPU/GPU Examples

```python
from statgpu.linear_model import ElasticNet

# CPU with NumPy
model_cpu = ElasticNet(alpha=0.1, l1_ratio=0.5, device="cpu")
model_cpu.fit(X, y)
print(f"R²: {model_cpu.score(X, y):.4f}")

# GPU with CuPy
model_gpu_cupy = ElasticNet(
    alpha=0.1, l1_ratio=0.5, device="cuda",
    gpu_memory_cleanup=True
)
model_gpu_cupy.fit(X, y)

# GPU with PyTorch
model_gpu_torch = ElasticNet(
    alpha=0.1, l1_ratio=0.5, device="torch"
)
model_gpu_torch.fit(X, y)
```

Backend performance depends on sample size, feature dimension, dtype, hardware,
data residency, and transfer costs. Benchmark the actual target workload before
selecting a backend solely for speed.

## Covariance/Inference

`ElasticNet` is estimation-only by default. Set `compute_inference=True` to run
post-fit inference through the shared penalized-linear inference engine. The
default `inference_method="debiased"` uses nodewise Lasso to construct a
bias-corrected estimator, standard errors, z statistics, p-values, and 95%
confidence intervals. `summary()` is available after inference succeeds.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `compute_inference` | `False` | Enable post-fit coefficient inference |
| `inference_method` | `"debiased"` | `"debiased"`, `"cpu_ols"`, or `"bootstrap"` |
| `cov_type` | `"nonrobust"` | Covariance convention where applicable |
| `hac_maxlags` | `None` | HAC lag count where the selected inference method supports HAC |

Debiased inference is implemented for NumPy, CuPy, and Torch fitting paths. CPU
validation is part of the hosted test suite; physical CUDA validation for CuPy
and Torch remains a required remote gate for each exact release candidate.
Post-selection OLS is a heuristic and does not provide valid selective-inference
coverage. Inference is conditional on the selected regularization parameters
and does not alter the fitted penalized coefficients.

For `ElasticNetCV`, `compute_inference=True` applies inference only to the final
full-data refit after alpha and `l1_ratio` have been selected. Fold models remain
estimation-only.

## Solver and Inference Semantics

The default estimator uses FISTA for the declared Elastic Net objective. The
`stopping` option changes only the convergence diagnostic (`coef_delta` versus
KKT violation); it does not define a separate statistical approximation mode.

`compute_inference=False` returns the penalized estimate only. With
`compute_inference=True`, the same fitted coefficients are retained and the
selected post-fit inference method is run afterward. The standalone
`ElasticNet` wrapper and the final full-data refit of `ElasticNetCV` both support
this contract directly; users do not need to switch estimator classes merely
to request debiased inference.

## Outputs

After fitting, the following attributes are available:

| Attribute | Description |
|-----------|-------------|
| `coef_` | Estimated coefficients (shape: n_features) |
| `intercept_` | Fitted intercept |
| `n_iter_` | Number of iterations until convergence |
| `aic` | Akaike Information Criterion (when inference provides it) |
| `bic` | Bayesian Information Criterion (when inference provides it) |

Methods: `fit(X, y)`, `predict(X)`, `score(X, y)`, `summary()`

## Numerical Validation

The maintained regression suite checks agreement across supported backends and
against reference implementations at tolerances chosen for each dtype and
solver path. Physical CUDA validation remains part of the exact-head handoff;
no universal coefficient tolerance or speedup applies to every workload.

## References

- Zou, H., & Hastie, T. (2005). Regularization and variable selection via the Elastic Net. *Journal of the Royal Statistical Society: Series B*, 67(2), 301-320.
- Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm for linear inverse problems. *SIAM Journal on Imaging Sciences*, 2(1), 183-202.
