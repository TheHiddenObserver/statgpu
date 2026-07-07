# Quantile Regression

> Language: English  
> Last updated: 2026-07-01  
> This page: Model documentation  
> Switch: [Chinese](../../cn/models/quantile.md)

## Overview

`QuantileLoss` implements pinball (check) loss for quantile regression. `PenalizedQuantileRegression` wraps it with up to 10 penalty types and 8 solvers, including the specialized Proximal IRLS-CD solver for SCAD/MCP.

| Component | Path |
|-----------|------|
| Loss | `statgpu.losses.QuantileLoss` |
| Penalized Model | `statgpu.linear_model.penalized.PenalizedQuantileRegression` |
| Specialized Solver | `statgpu.solvers._proximal_irls_quantile.proximal_irls_quantile_solver` |
| R Equivalent | `quantreg::rq()` |

## Objective Function

Pinball loss at quantile τ ∈ (0, 1):

$$
\ell(\eta, y) = \rho_\tau(y - \eta), \quad \rho_\tau(u) = u \cdot (\tau - \mathbf{1}\{u < 0\})
$$

Per-sample gradient (subgradient at u=0):

$$
\frac{\partial \ell}{\partial \eta} = -\tau + \mathbf{1}\{y - \eta < 0\}
$$

Key property: the gradient is a step function — it does not vary with residual magnitude. This makes `has_hessian = False` and `smooth_gradient = False`.

## Parameters

| Parameter | Default | Description |
|---|---:|---|
| `quantile` | `0.5` | Target quantile in (0, 1). τ=0.5 = median regression. |

No scale parameter; quantile regression is scale-free.

## Solver Compatibility

| Solver | Support | Notes |
|--------|:---:|-------|
| Proximal IRLS-CD | ✅ | Specialized: IRLS majorization + LLA for SCAD/MCP. ~49x GPU speedup at large scale. |
| FISTA | ✅ | For non-smooth penalties (L1, SCAD, MCP) and non-convex group penalties. |
| IRLS | ✅ | For smooth penalties (L2, none). Uses Frisch-Newton algorithm (matches statsmodels QuantReg). |
| L-BFGS | ✅ | For smooth penalties, moderate dimensions. |
| ADMM | ✅ | Alternative for all penalties. |
| Newton | ❌ | Quantile has no Hessian. |
| Proximal Newton | ❌ | Quantile has no Hessian. |

## Penaltiy Compatibility

| Penalty | Solver (auto) | Notes |
|---------|---------------|-------|
| l2 / none | IRLS | Converges in 5-15 iterations. |
| l1 / elasticnet | FISTA | Subgradient-based. |
| SCAD / MCP | Proximal IRLS-CD | Fastest: ~3x CPU / ~49x GPU over FISTA-LLA. |
| adaptive_l1 | FISTA-LLA | Weighted L1 proximal. |
| group_* | FISTA-LLA | Group proximal operators. |

## Examples

### CPU

```python
from statgpu.linear_model.penalized import PenalizedQuantileRegression

# Median regression (τ=0.5)
model = PenalizedQuantileRegression(quantile=0.5, penalty='scad', alpha=0.1)
model.fit(X, y)
print(model.coef_)

# Upper quartile with L2 penalty
model = PenalizedQuantileRegression(quantile=0.75, penalty='l2', alpha=0.01)
model.fit(X, y)

# Lower quartile with MCP
model = PenalizedQuantileRegression(quantile=0.25, penalty='mcp', alpha=0.1)
model.fit(X, y)
```

### GPU (torch-CUDA)

```python
import torch
X_t = torch.tensor(X, dtype=torch.float64).cuda()
y_t = torch.tensor(y, dtype=torch.float64).cuda()

model = PenalizedQuantileRegression(quantile=0.5, penalty='scad', alpha=0.1)
model.fit(X_t, y_t)
```

### GPU (cupy-CUDA)

```python
import cupy as cp
X_cp = cp.asarray(X)
y_cp = cp.asarray(y)

model = PenalizedQuantileRegression(quantile=0.5, penalty='scad', alpha=0.1)
model.fit(X_cp, y_cp)
```

### Weighted Quantile

```python
sample_weight = np.ones(n)
sample_weight[:50] = 5.0  # upweight first 50 observations

model = PenalizedQuantileRegression(quantile=0.5, penalty='l2', alpha=0.01)
model.fit(X, y, sample_weight=sample_weight)
```

## Algorithm Details

### Proximal IRLS-CD (SCAD/MCP)

For quantile + nonconvex penalties, the specialized solver uses:

1. **IRLS quadratic majorization**: At each iteration, compute weights w_i = τ_i / max(|r_i|, ε). This forms a quadratic upper bound of the non-smooth pinball loss: Q(β) = ½ Σ w_i(y_i − X_iβ)².

2. **LLA (Local Linear Approximation)**: Non-convex SCAD/MCP is converted to weighted L1 via P'(|β_j|) weights.

3. **Parallel diagonal majorization**: A Jacobi-style update uses matrix operations (O(np) per sweep) — GPU-friendly.

4. **GPU optimization**: Convergence check compares on-device, only syncs a bool to CPU. Throttled to every 5 iterations.

### IRLS (L2/none)

Uses the Frisch-Newton algorithm (matching statsmodels `QuantReg`):
1. IRLS weights: w_i = (τ + (1−2τ)·1_{r_i<0}) / max(|r_i|, ε)
2. Solve weighted least squares: (X'WX + n·α·I) β = X'Wy
3. Repeat until convergence (~5-15 iterations)

## Outputs

| Attribute | Type | Description |
|-----------|------|-------------|
| `coef_` | (p,) float | Estimated coefficients |
| `intercept_` | float | Estimated intercept |
| `n_iter_` | int | Number of iterations |
| `quantile` | float | Target quantile |

## External Validation

- **R `quantreg::rq()`**: IRLS path matches Frisch-Newton IRLS coefficient to 1e-6.
- **sklearn `QuantileRegressor`**: HiGHS LP solver generates same active set and coefficients (tol=1e-8).
- **FISTA-LLA parity**: Proximal IRLS-CD produces same active set as FISTA-LLA within rtol=0.15.

## Notes

- Score uses weighted pinball loss: `score()` returns negative mean pinball loss for sklearn compatibility.
- `sample_weight` fully supported across all solvers.
- GPU devices (`cuda`/`torch`) do not silently fall back to CPU.
- For large problems (n=10K, p=500), GPU is ~49x faster than CPU.

## References

- Koenker, R. & Bassett, G. (1978). Regression Quantiles. *Econometrica*, 46(1), 33-50.
- Koenker, R. (2005). *Quantile Regression*. Cambridge University Press.
- Wu, Y. & Liu, Y. (2009). Variable Selection in Quantile Regression. *Statistica Sinica*, 19, 801-817.
- Hunter, D. R. & Li, R. (2005). Variable Selection using MM Algorithms. *Annals of Statistics*, 33(4), 1617-1642.
