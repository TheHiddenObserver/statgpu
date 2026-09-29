# Loss Functions (LossBase)

> Language: English  
> Last updated: 2026-09-17  
> This page: Low-level loss reference  
> Switch: [Chinese](../../cn/models/losses.md)

## Overview

`LossBase` is the low-level interface used by statgpu loss functions to expose the data-fit term of an optimization problem. Most users should start from a model class; this page is for readers who need the loss definition itself, its numerical primitives, parameters, or direct loss-level API.

This page deliberately stops at the **loss layer**. Estimator solver selection, penalty-specific routing, CV behavior, inference, and model-level `sample_weight` support belong in the corresponding model and solver documentation.

Related documentation:

- [Quantile Regression](quantile.md) — quantile-regression estimators, solver choices, weighting, and inference
- [Robust Regression](robust.md) — Huber, Bisquare, and Fair estimators
- [CoxPH](coxph.md) — Cox models, tie handling, counting-process data, and inference
- [GeneralizedLinearModel](generalized-linear-model.md) — GLM objectives and analytic-weight semantics
- [Solver × Penalty Compatibility Matrix](../guides/solver-penalty-matrix.md) — supported solver combinations
- [Solver Algorithms](../guides/solver-algorithms.md) — numerical algorithms and their assumptions

## Public paths

```text
statgpu.losses.LossBase
statgpu.losses.QuantileLoss
statgpu.losses.HuberLoss
statgpu.losses.BisquareLoss
statgpu.losses.FairLoss
statgpu.losses.CoxPartialLikelihoodLoss
```

## Loss hierarchy

```text
LossBase (statgpu/losses/_base.py)
├── GLMLoss (statgpu/glm_core/_base.py)
│   ├── SquaredErrorLoss, LogisticLoss, PoissonLoss, ...
├── QuantileLoss
├── HuberLoss
├── BisquareLoss
├── FairLoss
└── CoxPartialLikelihoodLoss
```

The hierarchy provides a common numerical interface; it does not make these losses members of one statistical model family.

## Shared objective and weight semantics

For a per-observation loss \(\ell_i(\beta)\), the unweighted data-fit term is

$$
L(\beta)=\frac{1}{n}\sum_{i=1}^n \ell_i(\beta).
$$

When a `LossBase` implementation accepts analytic `sample_weight` through the shared first-order primitives, the normalized data-fit term is

$$
L_w(\beta)
=\frac{\sum_i w_i\ell_i(\beta)}{\sum_i w_i}.
$$

`value()`, `gradient()`, and `fused_value_and_gradient()` expose these loss-level primitives. Some subclasses additionally provide Hessian or specialized primitives; others may reject a requested weighted operation. A solver or estimator is supported only when all numerical quantities required by that route are defined consistently. See the [Solver × Penalty Compatibility Matrix](../guides/solver-penalty-matrix.md) for route-level support.

## Implemented non-GLM losses

| Loss | Class | Response | Smooth gradient | Hessian | Typical use |
|---|---|---|:---:|:---:|---|
| Quantile | `QuantileLoss` | continuous | ❌ | ❌ | Conditional quantiles, median regression |
| Huber | `HuberLoss` | continuous | ✅ | ✅ | Robust M-estimation |
| Bisquare | `BisquareLoss` | continuous | ✅ | ✅ | Redescending robust M-estimation |
| Fair | `FairLoss` | continuous | ✅ | ✅ | Smooth robust M-estimation |
| Cox PH | `CoxPartialLikelihoodLoss` | survival | ✅ | ✅ | Cox partial-likelihood optimization |

These properties describe the loss object itself. They are inputs to solver selection, not a substitute for the solver-support matrix.

### Quantile loss (check / pinball)

For residual \(u=y-\eta\) and quantile \(\tau\in(0,1)\),

$$
\rho_\tau(u)
=u\left(\tau-\mathbf 1\{u<0\}\right).
$$

At \(\tau=0.5\), \(\rho_{0.5}(u)=\tfrac12|u|\). The loss is piecewise linear, so `QuantileLoss` has a non-smooth step-function subgradient and no Hessian. For quantile-regression solver selection, penalties, weighting, CV, and inference, see [Quantile Regression](quantile.md).

### Huber loss

For residual \(u=y-\eta\),

$$
\rho_\delta(u)=
\begin{cases}
\frac12u^2, & |u|\le\delta,\\
\delta\left(|u|-\frac12\delta\right), & |u|>\delta.
\end{cases}
$$

### Bisquare loss (Tukey biweight)

For residual \(u=y-\eta\),

$$
\rho_c(u)=
\begin{cases}
\frac{c^2}{6}\left[1-\left(1-(u/c)^2\right)^3\right], & |u|\le c,\\
\frac{c^2}{6}, & |u|>c.
\end{cases}
$$

### Fair loss

For residual \(u=y-\eta\),

$$
\rho_c(u)
=c^2\left(\frac{|u|}{c}-\log\left(1+\frac{|u|}{c}\right)\right).
$$

Its score changes smoothly with the residual and gradually down-weights large residuals.

### Cox partial likelihood

`CoxPartialLikelihoodLoss` represents the negative log partial likelihood for ordinary right-censored Cox data. It accepts either a `{"time": ..., "event": ...}` dictionary or an `(n, 2)` `[time, event]` response and supports Breslow or Efron ties at this low level.

The partial likelihood is invariant to adding a common constant to all linear predictors, so an intercept is not identifiable from this loss. Higher-level Cox data structures, Exact ties, delayed entry, strata, and inference are documented on the [CoxPH](coxph.md) page.

## Parameters

### `QuantileLoss`

| Parameter | Default | Description |
|---|---:|---|
| `quantile` | `0.5` | Target quantile in `(0, 1)` |

### `HuberLoss`

| Parameter | Default | Description |
|---|---:|---|
| `delta` | `None` | Optional fixed threshold; when supplied, `epsilon` and `method` are ignored |
| `epsilon` | `1.35` | Robustness tuning constant used with an estimated scale |
| `method` | `"MAD"` | Scale handling: `"MAD"`, `"huber_prop2"`, or `"joint"` |

### `BisquareLoss`

| Parameter | Default | Description |
|---|---:|---|
| `epsilon` | `4.685` | Robustness tuning constant |
| `method` | `"MAD"` | Scale-estimation method |

### `FairLoss`

| Parameter | Default | Description |
|---|---:|---|
| `delta` | `None` | Optional fixed threshold; when supplied, `epsilon` and `method` are ignored |
| `epsilon` | `1.35` | Robustness tuning constant used with an estimated scale |
| `method` | `"MAD"` | Scale handling: `"MAD"` or `"huber_prop2"` |

### `CoxPartialLikelihoodLoss`

| Parameter | Default | Description |
|---|---:|---|
| `ties` | `"breslow"` | `"breslow"` or `"efron"`; use `CoxPH` for Exact ties |

## Direct loss-level examples

### Value and gradient

```python
import numpy as np
from statgpu.losses import HuberLoss

loss = HuberLoss(delta=1.5)
coef = np.zeros(X.shape[1])

value = loss.value(X, y, coef)
gradient = loss.gradient(X, y, coef)
```

### Weighted first-order evaluation

```python
sample_weight = np.ones(X.shape[0])
sample_weight[:50] = 5.0

value = loss.value(X, y, coef, sample_weight=sample_weight)
gradient = loss.gradient(X, y, coef, sample_weight=sample_weight)
```

This demonstrates the loss-layer weighted primitives only. Whether a complete estimator/solver route supports the same weights is documented separately.

### Cox partial likelihood

```python
from statgpu.losses import CoxPartialLikelihoodLoss

loss = CoxPartialLikelihoodLoss(ties="efron")
y_surv = np.column_stack([time, event])
coef = np.zeros(X.shape[1])

value = loss.value(X, y_surv, coef)
gradient = loss.gradient(X, y_surv, coef)
hessian = loss.hessian(X, y_surv, coef)
```

## Backend behavior

Loss evaluation follows the selected NumPy, CuPy, or Torch backend when the concrete loss supports that operation. Backend parity describes numerical execution; it does not by itself establish model-level solver, weighting, CV, or inference support.

Cox preprocessing makes a one-time host copy of sorted `time` and `event` values to construct deterministic failure-group metadata. The resulting indices are cached on the selected device, while the design matrix, linear predictor, objective, gradient, and Hessian remain on the numerical backend during iterative computation.

## Where to go next

- Use the model pages for estimator APIs, defaults, weighting, CV, and inference.
- Use the [Solver × Penalty Compatibility Matrix](../guides/solver-penalty-matrix.md) for supported combinations.
- Use [Solver Algorithms](../guides/solver-algorithms.md) for update equations and algorithmic assumptions.
- Use [Loss × Penalty × Solver Framework](../guides/loss-penalty-solver-framework.md) for the full computational architecture.

## References

- Koenker, R. & Bassett, G. (1978). Regression Quantiles. *Econometrica*, 46(1), 33-50.
- Huber, P. J. (1964). Robust Estimation of a Location Parameter. *Annals of Mathematical Statistics*, 35(1), 73-101.
- Beaton, A. E. & Tukey, J. W. (1974). The Fitting of Power Series. *Technometrics*, 16(2), 147-185.
- Cox, D. R. (1972). Regression Models and Life-Tables. *Journal of the Royal Statistical Society*, B34, 187-220.
