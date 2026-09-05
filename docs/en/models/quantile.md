# Quantile Regression

> Language: English  
> Last updated: 2026-07-01  
> This page: Model documentation  
> Switch: [Chinese](../../cn/models/quantile.md)

## Overview

`QuantileLoss` implements pinball (check) loss for quantile regression. `QuantileRegression` uses a fixed FISTA fit, while `PenalizedQuantileRegression` adds penalty-aware routing, including a specialized proximal-IRLS path for SCAD/MCP.

| Component | Path |
|-----------|------|
| Loss | `statgpu.losses.QuantileLoss` |
| Standalone Model | `statgpu.linear_model.QuantileRegression` |
| Penalized Model | `statgpu.linear_model.penalized.PenalizedQuantileRegression` |
| Specialized Solver | `statgpu.solvers._proximal_irls_quantile.proximal_irls_quantile_solver` |
| Low-level Coordinate Solver | `statgpu.solvers._quantile_cd.quantile_cd_solver` |
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

## Solver support

The two estimator surfaces intentionally differ:

| Estimator / `solver` value | Support | Actual path |
|---|:---:|---|
| `QuantileRegression` | fixed | FISTA; bootstrap inference uses batched pinball FISTA |
| `PenalizedQuantileRegression(solver="auto")` | yes | Penalty-aware paths in the table below |
| `solver="fista"` | yes | FISTA, except smooth L2/none is internally stabilized with quantile IRLS |
| `solver="fista_bb"` | yes | BB-step FISTA for ordinary compatible penalties |
| `solver="irls"` | L2/none only | Quantile-specific IRLS |
| `solver="admm"` | compatible penalties | Alternative split path; only uniform sample weights |
| `newton`, `lbfgs`, `exact` | no | Rejected because pinball loss has no Hessian, and exact is Ridge-only |

`proximal_irls_quantile_solver` and `quantile_cd_solver` are internal or low-level function names, not values to pass as `solver=`.

## Penalty Compatibility

| Penalty | Solver (auto) | Notes |
|---------|---------------|-------|
| l2 / none | FISTA dispatch, internally quantile IRLS | Smooth penalty; the internal IRLS tolerance is tightened. |
| l1 / elasticnet | FISTA | Proximal/subgradient path. |
| SCAD / MCP | Proximal IRLS + LLA | Model-specific continuation subpath. |
| adaptive_l1 | Weighted-L1 FISTA | Data-dependent proximal weights. |
| group penalties | Group proximal path | Exact route depends on the group penalty. |

## Examples

### Standalone Model (with inference)

```python
from statgpu.linear_model import QuantileRegression

# Median regression with kernel-based standard errors
model = QuantileRegression(
    quantile=0.5,
    compute_inference=True,
    inference_method="kernel",   # Powell (1991) sandwich
    kernel="epa",                # Epanechnikov kernel
    bandwidth="hsheather",       # Hall-Sheather bandwidth
)
model.fit(X, y)
print(model.coef_)        # coefficients
print(model._bse)         # standard errors
print(model._pvalues)     # p-values
print(model._conf_int)    # 95% confidence intervals

# Bootstrap inference with batched FISTA (GPU-accelerated)
model = QuantileRegression(
    quantile=0.5,
    compute_inference=True,
    inference_method="bootstrap",
    n_bootstrap=200,
    device="cuda",         # or "torch" / "cpu"
)
model.fit(X, y)
```

### Penalized Quantile (with penalty selection)

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

### Quantile proximal IRLS (SCAD/MCP)

**Source:** `statgpu/solvers/_proximal_irls_quantile.py`

This is the model-specific continuation path used for quantile loss with SCAD or MCP. For residual $r_i=y_i-x_i^\top\beta$ and target quantile $\tau$,

$$
\rho_\tau(r)=r\left(\tau-\mathbf 1\{r<0\}\right).
$$

Inside each continuation and local-linear-approximation (LLA) step, statgpu forms

$$
a_i
=
\frac{
\tau\mathbf 1\{r_i\ge0\}
+
(1-\tau)\mathbf 1\{r_i<0\}
}{
\max(|r_i|,\varepsilon)
},
\qquad
\omega_j=P_\lambda'(|\beta_j|).
$$

With $A=\operatorname{diag}(a_i)$,

$$
g=\tilde X^\top A(y-\tilde X\beta),
\qquad
h=\operatorname{diag}(\tilde X^\top A\tilde X),
$$

and all coordinates are updated in parallel:

$$
\beta_j^{\mathrm{new}}
=
\frac{
\mathcal S_{n\omega_j}(g_j+h_j\beta_j)
}{h_j}.
$$

This is a GPU-friendly Jacobi-style diagonal-majorization step, not cyclic coordinate descent. Analytic sample weights multiply $a_i$ after normalization to sum to $n$. Convergence checks stay on the selected device and synchronize only the Boolean result at a throttled interval.

**Primary references:**

- Koenker, R., & Bassett, G. (1978). [Regression quantiles](https://doi.org/10.2307/1913643). *Econometrica*, 46(1), 33–50.
- Wu, Y., & Liu, Y. (2009). [Variable selection in quantile regression](https://www3.stat.sinica.edu.tw/sstest/j19n2/j19n222/j19n222.html). *Statistica Sinica*, 19(2), 801–817.
- Zou, H., & Li, R. (2008). [One-step sparse estimates in nonconcave penalized likelihood models](https://doi.org/10.1214/07-AOS520). *Annals of Statistics*, 36(4), 1509–1533.

### Quantile coordinate descent

**Source:** `statgpu/solvers/_quantile_cd.py`

This NumPy-only low-level solver alternates LLA weights with cyclic coordinate updates for

$$
\min_\beta
\sum_{i=1}^n\rho_\tau(y_i-x_i^\top\beta)
+
\sum_{j=1}^p\omega_j|\beta_j|.
$$

Using $\psi_\tau(r)=\tau$ for $r\ge0$ and $\psi_\tau(r)=-(1-\tau)$ otherwise, the implemented coordinate step is

$$
\beta_j
\leftarrow
\frac{
\mathcal S_{\omega_j}
\left(
\sum_i x_{ij}\psi_\tau(r_i^{(-j)})
\right)
}{
\sum_i x_{ij}^2
}.
$$

It is exported for direct advanced use but is not selected by automatic estimator dispatch. Its `sample_weight` argument is present in the signature but is not consumed by the current implementation; use the routed FISTA or quantile-proximal-IRLS paths when observation weights matter.

**Primary reference:** Wu, Y., & Liu, Y. (2009). [Variable selection in quantile regression](https://www3.stat.sinica.edu.tw/sstest/j19n2/j19n222/j19n222.html). *Statistica Sinica*, 19(2), 801–817.

### IRLS (L2/none)

**Primary reference:** Koenker, R. (2005). *Quantile Regression*. Cambridge University Press.

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

- Koenker, R. (2005). *Quantile Regression*. Cambridge University Press.
- Hunter, D. R. & Li, R. (2005). Variable Selection using MM Algorithms. *Annals of Statistics*, 33(4), 1617-1642.
