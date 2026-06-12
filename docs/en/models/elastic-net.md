# Elastic Net

> Language: English  
> Last updated: 2026-04-18  
> This page: Model documentation  
> Language switch: [Chinese](../../models/elastic-net.md)

## Overview

`ElasticNet` combines L1 and L2 regularization for linear regression, enabling a balance between sparse feature selection (Lasso) and coefficient shrinkage (Ridge). It supports CPU, CuPy GPU, and PyTorch GPU backends with configurable device selection.

## Path

`statgpu.linear_model.ElasticNet`

## Objective Function

The Elastic Net optimization problem is:

\[
\min_{\beta} \frac{1}{2n}\|y - X\beta\|_2^2 + \alpha \cdot \lambda \cdot \|\beta\|_1 + \frac{\alpha}{2} \cdot (1 - \lambda) \cdot \|\beta\|_2^2
\]

where:
- `alpha` (α) controls overall regularization strength
- `l1_ratio` (λ) mixes L1 vs L2: λ=1 gives Lasso, λ=0 gives Ridge
- Loss scaling by `1/(2n)` makes `alpha` interpretation scale-invariant to sample size

**Note on regularization scaling**: With `l1_ratio=0`, `ElasticNet(alpha)` is equivalent to `Ridge(n_samples * alpha)` due to the loss scaling convention.

## Estimating Equation

The Elastic Net estimator solves the following first-order optimality (KKT) condition:

\[
\frac{1}{n} X^\top (X\hat{\beta} - y) + \alpha(1-\lambda)\hat{\beta} + \alpha\lambda \cdot \partial\|\hat{\beta}\|_1 = 0
\]

where $\partial\|\hat{\beta}\|_1$ is the subdifferential of the L1 norm:
- For $\hat{\beta}_j \neq 0$: $\text{sign}(\hat{\beta}_j)$
- For $\hat{\beta}_j = 0$: any value in $[-1, 1]$

At convergence, the KKT residual (subgradient violation) satisfies:
\[
\left| \frac{1}{n} X_j^\top(y - X\hat{\beta}) - \alpha(1-\lambda)\hat{\beta}_j \right| \leq \alpha\lambda \quad \forall j
\]

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
|-----------|--------:|-------------|
| `alpha` | `1.0` | Regularization strength (α) |
| `l1_ratio` | `0.5` | L1 mixing parameter (λ): 0=Ridge, 1=Lasso |
| `device` | `"cpu"` | Device: `cpu` / `cuda` |
| `backend` | `None` | Backend: `numpy` / `cupy` / `torch` (auto-detected if None) |
| `max_iter` | `5000` | Maximum iterations |
| `tol` | `1e-6` | Convergence tolerance |
| `fit_intercept` | `True` | Whether to fit intercept |
| `stopping` | `"coef_delta"` | Stopping rule: `coef_delta` / `kkt` |
| `warm_start` | `False` | Reuse previous fit as initialization |
| `random_state` | `None` | Random seed for reproducibility |
| `gpu_memory_cleanup` | `False` | Clean GPU memory after fit (CuPy only) |

## CPU/GPU Examples

```python
from statgpu.linear_model import ElasticNet

# CPU with NumPy
model_cpu = ElasticNet(alpha=0.1, l1_ratio=0.5, device="cpu")
model_cpu.fit(X, y)
print(f"R²: {model_cpu.score(X, y):.4f}")

# GPU with CuPy
model_gpu_cupy = ElasticNet(
    alpha=0.1, l1_ratio=0.5, device="cuda", backend="cupy",
    gpu_memory_cleanup=True
)
model_gpu_cupy.fit(X, y)

# GPU with PyTorch (recommended for n >= 10,000)
model_gpu_torch = ElasticNet(
    alpha=0.1, l1_ratio=0.5, device="cuda", backend="torch"
)
model_gpu_torch.fit(X, y)
```

### Solver Selection by Data Scale

| Data Scale | Recommended Backend | Expected Speedup vs sklearn |
|------------|---------------------|----------------------------|
| n < 1,000 | CPU (NumPy) | 0.7x - 1.0x |
| 1,000 ≤ n < 10,000 | CPU (NumPy) | 1.5x - 4x |
| 10,000 ≤ n < 50,000 | GPU (Torch) | 2x - 3x |
| n ≥ 50,000 | GPU (Torch) | 3x - 4.4x |

## Covariance/Inference

ElasticNet does not provide built-in inference (standard errors, p-values, confidence intervals) because the L1 penalty introduces bias in the coefficient estimates, making standard OLS-based inference invalid.

**Planned inference support**:

| Method | Description | Status |
|--------|-------------|--------|
| Debiased Lasso | Bias-corrected inference via nodewise regression | 待实现 — `PenalizedGeneralizedLinearModel` with `compute_inference=True` |
| Bootstrap | Empirical confidence intervals via resampling | 待实现 |
| Selection inference | Post-selection conditional inference | 待实现 |

For debiased inference with ElasticNet penalties, use `PenalizedGeneralizedLinearModel(loss='squared_error', penalty='elasticnet')` which will support the debiased Lasso path once implemented.

## strict/approx difference

ElasticNet uses the **approximate** (default) solver path:
- **approx**: FISTA with fixed Lipschitz constant, convergence checked via coefficient delta. Fast but no inference guarantees.
- **strict**: Not applicable for standalone ElasticNet. For debiased inference, use `PenalizedGeneralizedLinearModel` with `compute_inference=True`, which runs nodewise Lasso to construct the debiasing matrix M.

## Outputs

After fitting, the following attributes are available:

| Attribute | Description |
|-----------|-------------|
| `coef_` | Estimated coefficients (shape: n_features) |
| `intercept_` | Fitted intercept |
| `n_iter_` | Number of iterations until convergence |
| `aic` | Akaike Information Criterion (if available) |
| `bic` | Bayesian Information Criterion (if available) |

Methods: `fit(X, y)`, `predict(X)`, `score(X, y)`, `summary()`

## Numerical Consistency

All statgpu backends (CPU, CuPy, Torch) produce numerically consistent results:

| Backend Pair | Max Coefficient Difference |
|--------------|----------------------------|
| CPU vs CuPy | < 3e-8 |
| CPU vs Torch | < 3e-8 |
| All vs sklearn | < 3e-8 |

## Performance Benchmarks

### vs sklearn (Python)

| Dataset | n | p | sklearn (ms) | statgpu CPU (ms) | Speedup |
|---------|---|---|--------------|------------------|---------|
| small | 200 | 20 | 0.77 | 1.10 | 0.70x |
| medium | 1,000 | 50 | 10.42 | 2.37 | **4.40x** |
| large | 5,000 | 100 | 6.01 | 4.13 | **1.45x** |

### vs glmnet (R)

| Dataset | n | p | R glmnet (ms) | statgpu CPU (ms) | Winner |
|---------|---|---|---------------|------------------|--------|
| small | 200 | 20 | 8.51 | **1.10** | statgpu |
| medium | 1,000 | 50 | 6.27 | **2.06** | statgpu |
| large | 5,000 | 100 | 10.70 | **6.14** | statgpu |

statgpu CPU wins 4/6 comparisons against R glmnet.

### Large-Scale Performance (n ≥ 10,000)

| Dataset | n | p | sklearn | statgpu CPU | statgpu Torch | Torch Speedup |
|---------|---|---|---------|-------------|---------------|---------------|
| n_10k_p100 | 10,000 | 100 | 11.39 | 11.24 | 12.02 | 0.95x |
| n_10k_p500 | 10,000 | 500 | 82.03 | 100.51 | **30.52** | **2.69x** |
| n_50k_p100 | 50,000 | 100 | 69.74 | 52.01 | **21.74** | **3.21x** |
| n_50k_p500 | 50,000 | 500 | 310.34 | 145.31 | **79.77** | **3.89x** |
| n_100k_p100 | 100,000 | 100 | 118.94 | 60.85 | **33.23** | **3.58x** |
| n_100k_p500 | 100,000 | 500 | 615.59 | 269.45 | **141.05** | **4.36x** |

**Key findings**:
- statgpu Torch is fastest in 5/6 large-scale tests (83%)
- Maximum speedup: **4.36x** vs sklearn on n=100k, p=500
- GPU acceleration becomes advantageous at n ≥ 10,000

## FAQ

**Q: How do I choose l1_ratio?**
- `l1_ratio=1.0`: Pure Lasso (sparse solutions)
- `l1_ratio=0.0`: Pure Ridge (dense shrinkage)
- `l1_ratio=0.5`: Balanced (default)
- Tune via cross-validation for best predictive performance

**Q: Why are CPU and GPU iteration counts different?**
Different numerical paths and floating-point arithmetic can lead to slightly different convergence trajectories. Compare final coefficients and R² rather than iteration counts.

**Q: When should I use GPU vs CPU?**
- n < 10,000: CPU is faster (no data transfer overhead)
- n ≥ 10,000: GPU (Torch backend) shows 2-4x speedup
- n ≥ 50,000: Both CuPy and Torch show significant advantages

**Q: Why do coefficients differ from sklearn by ~1e-8?**
This is within numerical precision for floating-point arithmetic. All backends solve the same optimization problem to tolerance 1e-6.

**Q: How does alpha relate to Ridge/Lasso?**
- `ElasticNet(l1_ratio=1.0, alpha=X)` ≈ `Lasso(alpha=X)`
- `ElasticNet(l1_ratio=0.0, alpha=X)` ≈ `Ridge(alpha=n_samples * X)`

## External Validation

Benchmark scripts:
- `dev/benchmarks/benchmark_elasticnet_sklearn.py` - sklearn comparison
- `dev/benchmarks/benchmark_glmnet_full.R` - R glmnet comparison
- `dev/benchmarks/benchmark_large_scale.py` - large-scale performance
- `dev/benchmarks/run_full_benchmark.py` - unified benchmark runner

Test scripts:
- `dev/scripts/remote_elasticnet_smoke.py` - basic validation
- `dev/scripts/remote_stability_en.py` - numerical stability tests

## References

- Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. *Journal of the Royal Statistical Society: Series B*, 67(2), 301-320. [https://doi.org/10.1111/j.1467-9868.2005.00503.x](https://doi.org/10.1111/j.1467-9868.2005.00503.x)
- Nesterov, Y. (2005). Smooth minimization of non-smooth functions. *Mathematical Programming*, 103(1), 127-152. [https://doi.org/10.1007/s10107-004-0552-5](https://doi.org/10.1007/s10107-004-0552-5)
- Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm for linear inverse problems. *SIAM Journal on Imaging Sciences*, 2(1), 183-202. [https://doi.org/10.1137/080716542](https://doi.org/10.1137/080716542)
- Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. *Journal of Statistical Software*, 33(1), 1-22. [https://www.jstatsoft.org/v33/i01/](https://www.jstatsoft.org/v33/i01/)
