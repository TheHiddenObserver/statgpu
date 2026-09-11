# Loss Functions (LossBase)

> Language: English
>
> Last updated: 2026-09-12
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

Five loss types extend `LossBase` beyond the GLM family:

| Loss | Class | R Equivalent | Use Case |
|------|-------|--------------|----------|
| Quantile | `QuantileLoss` | `quantreg::rq()` | Conditional quantiles, median regression |
| Huber | `HuberLoss` | `MASS::rlm()` | Robust regression (M-estimator) |
| Bisquare | `BisquareLoss` | `MASS::rlm(psi="bisquare")` | Redescending M-estimator |
| Fair | `FairLoss` | `MASS::rlm(psi="fair")` | Fair's M-estimator |
| Cox PH | `CoxPartialLikelihoodLoss` | `survival::coxph()` | Survival analysis |

The framework exposes common penalty and solver interfaces, but supported combinations remain estimator- and loss-specific. A solver accepting one loss does not imply that every optional control—especially `sample_weight`—has the same meaning for every other loss.

Penalized wrappers are `PenalizedQuantileRegression`, `PenalizedRobustRegression`, and `PenalizedCoxPHModel`. The Cox wrapper supports L1, L2, ElasticNet, SCAD, and MCP; it is estimation-only and never fits an intercept.

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

Unweighted loss objects minimize an average loss plus the declared penalty:

$$
\min_{\beta} \frac{1}{n} \sum_{i=1}^n \ell(X_i \beta, y_i) + \text{penalty}(\beta).
$$

Where a loss explicitly supports analytic objective weights, the maintained convention is a normalized weighted average,

$$
\frac{\sum_i w_i\ell_i}{\sum_i w_i},
$$

but **weight support belongs to the loss/solver/estimator combination**, not to `LossBase` merely because a method accepts a `sample_weight` keyword.

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

The table above describes the ordinary unweighted solver compatibility. In particular, it does **not** mean that all of these losses accept genuine non-uniform weights through every listed solver.

### Non-uniform weights and direct L-BFGS

`lbfgs_solver` keeps genuine non-uniform weighted execution opt-in at the loss-contract layer. Maintained `GLMLoss` subclasses opt in because their fused value/gradient contract defines one normalized analytic-weight objective. Generic non-GLM losses on this page—Quantile, Huber/Bisquare/Fair, and Cox—remain fail-closed for genuine non-uniform `sample_weight` through direct L-BFGS unless that specific loss later defines and validates such a contract.

Uniform weights remain compatible with the historical unweighted L-BFGS route. This boundary prevents a shared solver enhancement for GLMs from silently changing the statistical meaning of robust, quantile, or survival objectives.

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

# These examples are unweighted direct-solver calls.
loss = QuantileLoss(quantile=0.5)
coef, n_iter = lbfgs_solver(loss, None, X, y)

loss = HuberLoss(epsilon=1.345)
coef, n_iter = lbfgs_solver(loss, None, X, y)
```

Do not infer non-uniform weighted L-BFGS support for these losses from the unweighted examples. Use the model-specific documentation for supported weighted robust/quantile procedures.

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

model = PenalizedQuantileRegression(quantile=0.5, penalty='scad', alpha=0.1)
model.fit(X, y)

model_gpu = PenalizedQuantileRegression(quantile=0.5, penalty='scad', alpha=0.1)
model_gpu.fit(X_t, y_t)
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

The iterative numerical arrays in these calls remain on the selected NumPy,
CuPy, or Torch backend. Cox loss preprocessing makes a one-time host copy of
the sorted `time` and `event` vectors to construct deterministic failure-group
metadata; the resulting indices are cached on the selected device and the
design matrix, predictor, objective, gradient, and Hessian are not moved to CPU
during solver iterations.

The SCAD/MCP trusted-gradient path skips duplicate finite-state checks but keeps
adaptive predictor-range segmentation on every evaluation. Stable risk-set
scaling is therefore never disabled by the solver fast path.

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

Loss-specific numerical validation uses the corresponding maintained model and solver contracts. Cross-backend support is checked separately from statistical weight semantics: agreement across NumPy/CuPy/Torch is not sufficient evidence that a new weighting interpretation is valid for a loss that has not declared one.

## Notes

- `QuantileLoss` is non-smooth and has no Hessian; model-level SCAD/MCP paths use FISTA or proximal IRLS-CD.
- Robust losses expose their own estimator-level weight semantics; those do not automatically imply direct non-uniform weighted L-BFGS support.
- `CoxPartialLikelihoodLoss` keeps its current sample-weight restrictions; use Cox-specific model documentation for supported data structures and inference.
- See [Loss × Penalty × Solver Framework](../guides/loss-penalty-solver-framework.md) for broader compatibility details.

## References

- Koenker, R. & Bassett, G. (1978). Regression Quantiles. *Econometrica*, 46(1), 33-50.
- Huber, P. J. (1964). Robust Estimation of a Location Parameter. *Annals of Mathematical Statistics*, 35(1), 73-101.
- Beaton, A. E. & Tukey, J. W. (1974). The Fitting of Power Series. *Technometrics*, 16(2), 147-185.
- Cox, D. R. (1972). Regression Models and Life-Tables. *Journal of the Royal Statistical Society*, B34, 187-220.
- Wu, Y. & Liu, Y. (2009). Variable Selection in Quantile Regression. *Statistica Sinica*, 19, 801-817.
- Fan, J. & Li, R. (2001). Variable Selection via Nonconcave Penalized Likelihood. *JASA*, 96, 1348-1360.
