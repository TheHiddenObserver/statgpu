# Loss Functions (LossBase)

> Language: English
>
> Last updated: 2026-07-12
>
> This page: Model documentation
>
> Switch: [Chinese](../../cn/models/losses.md)

## Overview

`LossBase` is the generic base class for all loss functions in statgpu. It provides a unified interface for optimization solvers and penalty functions.

> For solver algorithm details, see: [Solver Algorithms](../guides/solver-algorithms.md)
>
> For detailed per-loss documentation, see:
> - [Quantile Regression](quantile.md) — pinball loss, PenalizedQuantileRegression, Proximal IRLS-CD
> - [Robust Regression](robust.md) — Huber, Bisquare, Fair losses, PenalizedRobustRegression
> - [CoxPH](coxph.md) — Cox partial likelihood, three tie methods, counting-process data, and inference

Five new loss types extend `LossBase` beyond the existing GLM family:

| Loss | Class | R Equivalent | Use Case |
|------|-------|-------------|----------|
| Quantile | `QuantileLoss` | `quantreg::rq()` | Conditional quantiles, median regression |
| Huber | `HuberLoss` | `MASS::rlm()` | Robust regression (M-estimator) |
| Bisquare | `BisquareLoss` | `MASS::rlm(psi="bisquare")` | Redescending M-estimator |
| Fair | `FairLoss` | `MASS::rlm(psi="fair")` | Fair's M-estimator |
| Cox PH | `CoxPartialLikelihoodLoss` | `survival::coxph()` | Survival analysis |

The framework exposes common penalty and solver interfaces, but supported
combinations remain estimator-specific. Penalized wrappers are
`PenalizedQuantileRegression`, `PenalizedRobustRegression`, and
`PenalizedCoxPHModel`. The Cox wrapper supports L1, L2, ElasticNet, SCAD, and
MCP; it is estimation-only and never fits an intercept.

## Path

```
statgpu.losses.LossBase
statgpu.losses.QuantileLoss
statgpu.losses.HuberLoss
statgpu.losses.CoxPartialLikelihoodLoss
statgpu.linear_model.PenalizedCoxPHModel
```

## Architecture

```
LossBase (statgpu/losses/_base.py)
├── GLMLoss (statgpu/glm_core/_base.py) — adds _mu_from_eta, IRLS hints
│   ├── SquaredErrorLoss, LogisticLoss, PoissonLoss, ...
├── QuantileLoss — pinball loss, non-smooth
├── HuberLoss — robust, smooth
└── CoxPartialLikelihoodLoss — survival, has Hessian
```

## Objective Function

All losses minimize:
$$
\min_{\beta} \frac{1}{n} \sum_{i=1}^n \ell(X_i \beta, y_i) + \text{penalty}(\beta)
$$

### Quantile Loss (Pinball)

$$
\ell(\eta, y) = \rho_\tau(y - \eta), \quad \rho_\tau(u) = u \cdot (\tau - \mathbf{1}\{u < 0\})
$$

For $\tau = 0.5$, this is the absolute loss (median regression).

### Huber Loss

$$
\ell(\eta, y) = \begin{cases}
\frac{1}{2}(y - \eta)^2 & \text{if } |y - \eta| \le \delta \\
\delta(|y - \eta| - \frac{1}{2}\delta) & \text{otherwise}
\end{cases}
$$

### Cox Partial Likelihood (Negative)

$$
\ell(\beta) = -\frac{1}{n} \log L(\beta)
$$

where $L(\beta)$ is the Breslow or Efron partial likelihood. This low-level
loss class accepts a two-column response with `[time, event]`. The high-level
`statgpu.survival.CoxPH` estimator additionally implements Exact ties,
delayed-entry/start-stop data, and strata.

## Solver Compatibility

| Solver | Quantile | Huber | Bisquare | Fair | Cox PH |
|--------|----------|-------|----------|------|--------|
| FISTA | ✅ | ✅ | ✅ | ✅ | ✅ (L1/ElasticNet path) |
| FISTA-BB | ✅ | ✅ | ✅ | ✅ | ✅ (sparse convex path) |
| FISTA-LLA | ✅ (SCAD/MCP) | ✅ | ✅ | ✅ | ✅ (SCAD/MCP) |
| Proximal IRLS-CD | ✅ (SCAD/MCP) | ❌ | ❌ | ❌ | ❌ |
| Proximal Newton | ❌ (no Hessian) | ✅ (5-10 iter) | ✅ (5-10 iter) | ✅ | ❌ |
| Newton | ❌ (no Hessian) | ✅ | ✅ | ✅ | ✅ (unpenalized/L2) |
| L-BFGS | ✅ | ✅ | ✅ | ✅ | ✅ |
| ADMM | ✅ | ✅ | ✅ | ✅ | ✅ |
| IRLS | ✅ (L2 only) | ❌ | ❌ | ❌ | ❌ |

## Parameters

### QuantileLoss

| Parameter | Default | Description |
|---|---:|---|
| `quantile` | `0.5` | Target quantile in (0, 1) |

### HuberLoss

| Parameter | Default | Description |
|---|---:|---|
| `delta` | `1.0` | Threshold: quadratic for \|u\| ≤ delta, linear otherwise |

### CoxPartialLikelihoodLoss

| Parameter | Default | Description |
|---|---:|---|
| `ties` | `"breslow"` | Tie handling: `"breslow"` or `"efron"`; use `CoxPH` for Exact ties |

## Examples

### CPU

```python
from statgpu.losses import QuantileLoss, HuberLoss
from statgpu.solvers import lbfgs_solver

# Quantile regression
loss = QuantileLoss(quantile=0.5)
coef, n_iter = lbfgs_solver(loss, None, X, y)

# Robust regression
loss = HuberLoss(epsilon=1.345)
coef, n_iter = lbfgs_solver(loss, None, X, y)
```

### GPU (torch-CUDA)

```python
import torch
X_t = torch.tensor(X, dtype=torch.float64).cuda()
y_t = torch.tensor(y, dtype=torch.float64).cuda()

from statgpu.losses import HuberLoss
from statgpu.penalties import SCADPenalty
from statgpu.solvers import fista_solver

loss = HuberLoss(epsilon=1.345)
coef, n_iter = fista_solver(loss, SCADPenalty(alpha=0.1), X_t, y_t)
```

### Penalized Quantile with SCAD (CPU/GPU)

```python
from statgpu.linear_model.penalized import PenalizedQuantileRegression

# CPU
model = PenalizedQuantileRegression(quantile=0.5, penalty='scad', alpha=0.1)
model.fit(X, y)

# GPU (torch-CUDA)
model = PenalizedQuantileRegression(quantile=0.5, penalty='scad', alpha=0.1)
model.fit(X_t, y_t)
```

### Cox Partial Likelihood

```python
import numpy as np
from statgpu.losses import CoxPartialLikelihoodLoss

y_surv = np.column_stack([time, event])
loss = CoxPartialLikelihoodLoss(ties="efron")
coef = np.zeros(X.shape[1])
value = loss.value(X, y_surv, coef)
gradient = loss.gradient(X, y_surv, coef)
hessian = loss.hessian(X, y_surv, coef)
```

The same calls accept NumPy arrays, CuPy arrays, or Torch tensors and remain on
the selected backend.

### Regularized Survival

```python
from statgpu.linear_model import PenalizedCoxPHModel

model = PenalizedCoxPHModel(
    penalty="scad",
    alpha=0.1,
    ties="efron",
    device="cuda",
    fit_intercept=False,
    compute_inference=False,
)
model.fit(X, y_surv)
```

The supported penalties are `l1`, `l2`, `elasticnet`, `scad`, and `mcp`.
SCAD and MCP use FISTA-LLA continuation. `compute_inference=True` raises
`NotImplementedError`; use `statgpu.survival.CoxPH` for unpenalized inference.

## External Validation

- **QuantileLoss**: validated against R `quantreg::rq()` (Frisch-Newton IRLS) and sklearn `QuantileRegressor` (HiGHS LP solver). Coefficient parity to 1e-6.
- **HuberLoss**: validated against R `MASS::rlm()` with Huber psi function.
- **BisquareLoss**: validated against R `MASS::rlm(psi="bisquare")`. Supports SCAD/MCP via proximal Newton (5-10 iter convergence).
- **CoxPartialLikelihoodLoss**: Breslow/Efron value, gradient, and Hessian are
  checked across NumPy, CuPy, and Torch and against aligned
  `statsmodels.duration.PHReg` references. Exact ties are validated through the
  high-level `CoxPH` risk-set engine against brute-force references.

## Notes

- `CoxPartialLikelihoodLoss` uses native NumPy, CuPy, and PyTorch operations for
  Breslow and Efron. Explicit GPU inputs do not route through another backend
  or silently fall back to NumPy.
- `QuantileLoss` has `smooth_gradient=False` and `has_hessian=False`; use FISTA or proximal IRLS-CD (for SCAD/MCP).
- `HuberLoss` and `BisquareLoss` have `has_hessian=True`; proximal Newton converges in 5-10 iterations for SCAD/MCP.
- All losses accept `sample_weight` except `CoxPartialLikelihoodLoss`, which
  raises `NotImplementedError`.
- See [Loss × Penalty × Solver Framework](../guides/loss-penalty-solver-framework.md) for complete dispatch logic and coverage matrix.

## References

- Koenker, R. & Bassett, G. (1978). Regression Quantiles. *Econometrica*, 46(1), 33-50.
- Huber, P. J. (1964). Robust Estimation of a Location Parameter. *Annals of Mathematical Statistics*, 35(1), 73-101.
- Beaton, A. E. & Tukey, J. W. (1974). The Fitting of Power Series. *Technometrics*, 16(2), 147-185. (Bisquare)
- Cox, D. R. (1972). Regression Models and Life-Tables. *Journal of the Royal Statistical Society*, B34, 187-220.
- Wu, Y. & Liu, Y. (2009). Variable Selection in Quantile Regression. *Statistica Sinica*, 19, 801-817.
- Fan, J. & Li, R. (2001). Variable Selection via Nonconcave Penalized Likelihood. *JASA*, 96, 1348-1360. (SCAD)
