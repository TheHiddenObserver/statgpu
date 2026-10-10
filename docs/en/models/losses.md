# Loss Functions (LossBase)

> Language: English  
> Last updated: 2026-10-09\
> This page: Low-level loss reference  
> Switch: [Chinese](../../cn/models/losses.md)

## Overview

`LossBase` is the low-level interface used by statgpu loss functions to expose the data-fit term of an optimization problem. Most users should start from a model class; this page is for readers who need the loss definition itself, its numerical primitives, parameters, or direct loss-level API.

For estimator solver selection, penalty compatibility, CV, inference, and model-level `sample_weight` support, use the model and solver guides linked below.

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

These examples evaluate a loss at supplied coefficients; they do not fit a
model. Run the blocks within each example in order.

<a id="losses-huber"></a>

### Value and gradient

First import the fixed-threshold Huber loss and NumPy.

<!-- example: losses-huber -->
```python
import numpy as np
from statgpu.losses import HuberLoss
```

`X` has shape `(80, 3)`, with rows as observations and columns as predictors.
`y` has shape `(80,)`, one continuous response per row. This low-level API uses
`X @ coef` directly; no intercept column is added automatically.

```python
rng = np.random.default_rng(42)
X = rng.normal(size=(80, 3))
y = X @ np.array([2.0, -1.0, 0.5]) + rng.normal(scale=0.3, size=80)
```

Evaluate at the zero coefficient vector with Huber threshold `delta=1.5`.
The coefficient vector has one entry per column of `X`.

```python
loss = HuberLoss(delta=1.5)
coef = np.zeros(X.shape[1])
value = loss.value(X, y, coef)
gradient = loss.gradient(X, y, coef)
```

Inspect the scalar objective and one derivative per coefficient.

```python
print(round(float(value), 3))
print(np.round(gradient, 3))
```
<!-- example-end: losses-huber -->

The value is about `1.483`, and the gradient is approximately
`[-0.887, 0.352, -0.093]`. These derivatives describe local changes in the
objective at zero coefficients; they are neither fitted slopes nor standard
errors. A solver would use such evaluations to update the coefficients.

### Weighted first-order evaluation

Reuse `X`, `y`, `coef` and `loss` from [the Huber example](#losses-huber).
Here the first 50 observations receive five times the weight of each remaining
row. Weights have shape `(80,)` and align with rows.

<!-- example-requires: losses-huber -->
<!-- example: losses-weighted -->
```python
sample_weight = np.ones(X.shape[0])
sample_weight[:50] = 5.0
weighted_value = loss.value(X, y, coef, sample_weight=sample_weight)
weighted_gradient = loss.gradient(X, y, coef, sample_weight=sample_weight)
print(float(weighted_value), weighted_gradient.shape)
```
<!-- example-end: losses-weighted -->

The weighted objective divides by the sum of weights, as defined above;
changing all weights by the same positive factor does not change it. This
demonstrates loss-layer first-order primitives only. Whether a complete
estimator/solver supports the same weights is documented separately.

### Cox partial likelihood

This separate example supplies its own small survival dataset. Import the Cox
loss and NumPy; none of the preceding Huber variables are required.

<!-- example: losses-cox -->
```python
import numpy as np
from statgpu.losses import CoxPartialLikelihoodLoss
```

`X_surv` has shape `(6, 2)` with no intercept column. `time` is the observed
follow-up time and `event` is 1 for an event or 0 for right censoring. Stack
them in that order to form the `(6, 2)` response required by this interface.

```python
X_surv = np.array([[0., 1.], [1., 0.], [0.5, 1.],
                   [-0.5, 0.], [1.5, 1.], [-1., 0.]])
time = np.array([2., 3., 3., 5., 6., 8.])
event = np.array([1, 1, 0, 1, 0, 1])
y_surv = np.column_stack([time, event])
```

Evaluate the Efron loss, gradient and Hessian at zero slopes. This illustrates
the numerical interface only, not a fitted survival analysis.

```python
cox_loss = CoxPartialLikelihoodLoss(ties="efron")
coef_surv = np.zeros(X_surv.shape[1])
cox_value = cox_loss.value(X_surv, y_surv, coef_surv)
cox_gradient = cox_loss.gradient(X_surv, y_surv, coef_surv)
cox_hessian = cox_loss.hessian(X_surv, y_surv, coef_surv)
```

Check the result dimensions.

```python
print(round(float(cox_value), 3))
print(cox_gradient.shape, cox_hessian.shape)
```
<!-- example-end: losses-cox -->

The scalar value is about `0.750`; the gradient and Hessian have shapes
`(2,)` and `(2, 2)`. The Hessian describes objective curvature at the supplied
coefficients, not a ready-made covariance estimate. Use [CoxPH](coxph.md) for
fitting, survival prediction and the model-specific inference workflow.

## Backend behavior

Loss evaluation follows the selected NumPy, CuPy, or Torch backend when the concrete loss supports that operation. Backend parity describes numerical execution; it does not by itself establish model-level solver, weighting, CV, or inference support.

Cox preprocessing copies sorted `time` and `event` values to CPU. The design matrix, linear predictor, objective, gradient, and Hessian use the selected numerical backend during iteration.

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
