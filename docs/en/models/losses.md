# Loss Functions (LossBase)

> Language: English  
> Last updated: 2026-07-01  
> This page: Model documentation  
> Switch: [Chinese](../../cn/models/losses.md)

## Overview

`LossBase` is the generic base class for all loss functions in statgpu. It provides a unified interface for optimization solvers (FISTA, Newton, L-BFGS, ADMM) and penalty functions (L1, L2, ElasticNet, SCAD, MCP, etc.).

Three new loss types extend `LossBase` beyond the existing GLM family:

| Loss | Class | R Equivalent | Use Case |
|------|-------|-------------|----------|
| Quantile | `QuantileLoss` | `quantreg::rq()` | Conditional quantiles, median regression |
| Huber | `HuberLoss` | `MASS::rlm()` | Robust regression (M-estimator) |
| Cox PH | `CoxPartialLikelihoodLoss` | `survival::coxph()` | Survival analysis |

All losses automatically inherit support for 10 penalty types and 6 solver types.

## Path

```
statgpu.losses.LossBase
statgpu.losses.QuantileLoss
statgpu.losses.HuberLoss
statgpu.losses.CoxPartialLikelihoodLoss
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

where $L(\beta)$ is the Breslow or Efron partial likelihood.

## Solver Compatibility

| Solver | Quantile | Huber | Cox PH |
|--------|----------|-------|--------|
| FISTA | ✅ | ✅ | ✅ |
| FISTA-BB | ✅ | ✅ | ✅ |
| Newton | ❌ (no Hessian) | ✅ | ✅ |
| L-BFGS | ✅ | ✅ | ✅ |
| ADMM | ✅ | ✅ | ✅ |
| IRLS | ❌ (GLM only) | ❌ (GLM only) | ❌ (GLM only) |

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
| `ties` | `"breslow"` | Tie handling: `"breslow"` or `"efron"` |

## Examples

### Quantile Regression

```python
from statgpu.losses import QuantileLoss
from statgpu.solvers import lbfgs_solver

loss = QuantileLoss(quantile=0.5)  # median regression
coef, n_iter = lbfgs_solver(loss, None, X, y)
```

### Huber Robust Regression

```python
from statgpu.losses import HuberLoss
from statgpu.solvers import lbfgs_solver

loss = HuberLoss(delta=1.345)  # 95% efficiency at Gaussian
coef, n_iter = lbfgs_solver(loss, None, X, y)
```

### Cox PH with Newton Solver

```python
from statgpu.losses import CoxPartialLikelihoodLoss
from statgpu.solvers import newton_solver
from statgpu.penalties import L2Penalty

loss = CoxPartialLikelihoodLoss(ties='breslow')
y = {'time': time, 'event': event}
coef, n_iter = newton_solver(loss, L2Penalty(0.0), X, y)
```

### With Penalties (Regularized Survival)

```python
from statgpu.losses import CoxPartialLikelihoodLoss
from statgpu.solvers import lbfgs_solver
from statgpu.penalties import L1Penalty

loss = CoxPartialLikelihoodLoss(ties='efron')
coef, _ = lbfgs_solver(loss, L1Penalty(0.1), X, y)  # Lasso-Cox
```

## Notes

- `CoxPartialLikelihoodLoss` supports GPU via CuPy CUDA / PyTorch-CUDA kernels (both Breslow and Efron). Explicit GPU inputs raise `RuntimeError` if GPU path is unavailable; CPU inputs use numpy implementation.
- `QuantileLoss` has `smooth_gradient=False`; FISTA handles the non-smoothness via proximal operators.
- `HuberLoss` has `smooth_gradient=True` and `has_hessian=True`; proximal Newton is the preferred solver.
- `BisquareLoss` is also available via `statgpu.losses.BisquareLoss` for redescending M-estimation.
- All losses accept `sample_weight` in their API, except `CoxPartialLikelihoodLoss` which raises `NotImplementedError`.

## References

- Koenker, R. & Bassett, G. (1978). Regression Quantiles. *Econometrica*, 46(1), 33-50.
- Huber, P. J. (1964). Robust Estimation of a Location Parameter. *Annals of Mathematical Statistics*, 35(1), 73-101.
- Cox, D. R. (1972). Regression Models and Life-Tables. *Journal of the Royal Statistical Society*, B34, 187-220.
