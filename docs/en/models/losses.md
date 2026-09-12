# Loss Functions (LossBase)

> Language: English  
> Last updated: 2026-09-12  
> This page: Low-level loss reference  
> Switch: [Chinese](../../cn/models/losses.md)

## Overview

`LossBase` is the common low-level interface between statgpu loss functions, penalties, and optimization solvers. Most users should start from a model class rather than instantiate a loss directly; this page is mainly useful when you need to understand solver compatibility or call a solver at the loss level.

Related model documentation:

- [Quantile Regression](quantile.md) — pinball loss and `PenalizedQuantileRegression`
- [Robust Regression](robust.md) — Huber, Bisquare, Fair losses and `PenalizedRobustRegression`
- [CoxPH](coxph.md) — Cox partial likelihood, tie handling, counting-process data, and inference
- [GeneralizedLinearModel](generalized-linear-model.md) — GLM objective and analytic-weight semantics

Five non-GLM loss types extend the shared loss interface:

| Loss | Class | R equivalent | Typical use |
|------|-------|--------------|-------------|
| Quantile | `QuantileLoss` | `quantreg::rq()` | Conditional quantiles, median regression |
| Huber | `HuberLoss` | `MASS::rlm()` | Robust M-estimation |
| Bisquare | `BisquareLoss` | `MASS::rlm(psi="bisquare")` | Redescending M-estimation |
| Fair | `FairLoss` | `MASS::rlm(psi="fair")` | Robust Fair loss |
| Cox PH | `CoxPartialLikelihoodLoss` | `survival::coxph()` | Survival analysis |

A shared interface does **not** imply a shared statistical contract. In particular, solver support, penalty support, and `sample_weight` semantics remain loss- and estimator-specific.

Penalized wrappers include `PenalizedQuantileRegression`, `PenalizedRobustRegression`, and `PenalizedCoxPHModel`. The Cox wrapper supports L1, L2, ElasticNet, SCAD, and MCP; it is estimation-only and never fits an intercept.

## Public paths

```
statgpu.losses.LossBase
statgpu.losses.QuantileLoss
statgpu.losses.HuberLoss
statgpu.losses.BisquareLoss
statgpu.losses.FairLoss
statgpu.losses.CoxPartialLikelihoodLoss
statgpu.linear_model.PenalizedCoxPHModel
```

## Architecture

```
LossBase (statgpu/losses/_base.py)
├── GLMLoss (statgpu/glm_core/_base.py) — GLM-specific value/gradient/Hessian contract
│   ├── SquaredErrorLoss, LogisticLoss, PoissonLoss, ...
├── QuantileLoss — pinball loss, non-smooth
├── HuberLoss — robust, smooth
├── BisquareLoss — redescending robust loss
├── FairLoss — robust Fair loss
└── CoxPartialLikelihoodLoss — survival loss with Hessian support
```

## Objective and weight semantics

An unweighted loss/penalty problem has the form

$$
\min_{\beta} \frac{1}{n}\sum_{i=1}^n \ell_i(\beta) + P(\beta).
$$

When a particular loss/solver/estimator combination explicitly supports analytic objective weights, statgpu uses the normalized weighted data-fit term

$$
\frac{\sum_i w_i\ell_i(\beta)}{\sum_i w_i}.
$$

The important point is that **weight support belongs to the complete loss × solver × estimator route**. A low-level method accepting a `sample_weight` argument does not by itself mean that every solver is statistically defined for arbitrary non-uniform weights on that loss.

### Quantile loss (pinball)

$$
\ell(\eta, y) = \rho_\tau(y - \eta), \quad
\rho_\tau(u) = u\left(\tau - \mathbf{1}\{u < 0\}\right).
$$

For $\tau = 0.5$, this is absolute loss (median regression).

### Huber loss

$$
\ell(\eta, y) = \begin{cases}
\frac{1}{2}(y - \eta)^2 & \text{if } |y - \eta| \le \delta, \\
\delta\left(|y - \eta| - \frac{1}{2}\delta\right) & \text{otherwise}.
\end{cases}
$$

### Cox partial likelihood (negative log scale)

$$
\ell(\beta) = -\frac{1}{n}\log L(\beta).
$$

`CoxPartialLikelihoodLoss` accepts a two-column response `[time, event]` and uses Breslow or Efron partial likelihood. The high-level `statgpu.survival.CoxPH` estimator additionally supports Exact ties, delayed-entry/start-stop data, and strata.

## Solver compatibility

The table below describes the maintained **unweighted** low-level compatibility. It should not be read as a weighted-support matrix.

| Solver | Quantile | Huber | Bisquare | Fair | Cox PH |
|--------|----------|-------|----------|------|--------|
| FISTA | ✅ | ✅ | ✅ | ✅ | ✅ (L1/ElasticNet path) |
| FISTA-BB | ✅ | ✅ | ✅ | ✅ | ✅ (sparse convex path) |
| FISTA-LLA | ✅ (SCAD/MCP) | ✅ | ✅ | ✅ | ✅ (SCAD/MCP) |
| Proximal IRLS-CD | ✅ (SCAD/MCP) | ❌ | ❌ | ❌ | ❌ |
| Proximal Newton | ❌ (no Hessian) | ✅ | ✅ | ✅ | ❌ |
| Newton | ❌ (no Hessian) | ✅ | ✅ | ✅ | ✅ (unpenalized/L2) |
| L-BFGS | ✅ | ✅ | ✅ | ✅ | ✅ |
| ADMM | ✅ | ✅ | ✅ | ✅ | ✅ |
| IRLS | ✅ (L2 only) | ❌ | ❌ | ❌ | ❌ |

### Non-uniform weights and direct L-BFGS

Direct L-BFGS is deliberately conservative about non-uniform weights:

| Direct `lbfgs_solver` route | Genuine non-uniform `sample_weight` |
|---|---|
| Maintained `GLMLoss` implementations | ✅ Supported by the GLM weighted-objective contract |
| Quantile / Huber / Bisquare / Fair | ❌ Not implied by unweighted L-BFGS support |
| Cox partial likelihood | ❌ Uses its own Cox-specific weight boundary |

`GLMLoss` can opt in because its fused value/gradient contract defines one normalized analytic-weight objective. Generic non-GLM losses remain closed to genuine non-uniform direct L-BFGS weights unless that specific loss later defines and validates an equivalent contract.

Uniform weights retain historical unweighted L-BFGS behavior. This distinction prevents a shared solver enhancement for GLMs from silently changing the statistical meaning of robust, quantile, or survival objectives.

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

### `CoxPartialLikelihoodLoss`

| Parameter | Default | Description |
|---|---:|---|
| `ties` | `"breslow"` | `"breslow"` or `"efron"`; use `CoxPH` for Exact ties |

## Examples

### Direct CPU solver calls

```python
from statgpu.losses import QuantileLoss, HuberLoss
from statgpu.solvers import lbfgs_solver

# These are intentionally unweighted low-level examples.
quantile_loss = QuantileLoss(quantile=0.5)
coef_q, n_iter_q = lbfgs_solver(quantile_loss, None, X, y)

huber_loss = HuberLoss(epsilon=1.345)
coef_h, n_iter_h = lbfgs_solver(huber_loss, None, X, y)
```

These examples demonstrate unweighted direct L-BFGS only. They do not establish non-uniform weighted L-BFGS support for Quantile or Huber loss. For weighted robust/quantile procedures, follow the corresponding model documentation.

### GPU (Torch CUDA)

```python
import torch
from statgpu.losses import HuberLoss
from statgpu.penalties import SCADPenalty
from statgpu.solvers import fista_solver

X_t = torch.tensor(X, dtype=torch.float64).cuda()
y_t = torch.tensor(y, dtype=torch.float64).cuda()

loss = HuberLoss(epsilon=1.345)
coef, n_iter = fista_solver(loss, SCADPenalty(alpha=0.1), X_t, y_t)
```

### Penalized quantile + SCAD (CPU/GPU)

```python
from statgpu.linear_model.penalized import PenalizedQuantileRegression

model = PenalizedQuantileRegression(quantile=0.5, penalty="scad", alpha=0.1)
model.fit(X, y)

model_gpu = PenalizedQuantileRegression(quantile=0.5, penalty="scad", alpha=0.1)
model_gpu.fit(X_t, y_t)
```

### Cox partial likelihood

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

Iterative numerical arrays remain on the selected NumPy, CuPy, or Torch backend. Cox preprocessing makes a one-time host copy of sorted `time` and `event` values to build deterministic failure-group metadata; the resulting indices are cached on the selected device, while the design matrix, predictor, objective, gradient, and Hessian remain on the numerical backend during solver iterations.

### Regularized survival

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

Supported penalties are `l1`, `l2`, `elasticnet`, `scad`, and `mcp`. SCAD and MCP use FISTA-LLA continuation. `compute_inference=True` raises `NotImplementedError`; use `statgpu.survival.CoxPH` for unpenalized inference.

## Validation and notes

Loss-specific numerical validation follows the corresponding model and solver contracts. Cross-backend parity and statistical weight semantics are separate questions: agreement across NumPy/CuPy/Torch is not, by itself, evidence that a new weighting interpretation is valid for a loss that has not declared one.

- `QuantileLoss` is non-smooth and has no Hessian; model-level SCAD/MCP paths use FISTA or proximal IRLS-CD.
- Robust losses expose estimator-level weight semantics; those semantics do not automatically extend to direct non-uniform weighted L-BFGS.
- `CoxPartialLikelihoodLoss` keeps its Cox-specific sample-weight restrictions; use the Cox model documentation for supported data structures and inference.
- See [Loss × Penalty × Solver Framework](../guides/loss-penalty-solver-framework.md) for broader compatibility details.

## References

- Koenker, R. & Bassett, G. (1978). Regression Quantiles. *Econometrica*, 46(1), 33-50.
- Huber, P. J. (1964). Robust Estimation of a Location Parameter. *Annals of Mathematical Statistics*, 35(1), 73-101.
- Beaton, A. E. & Tukey, J. W. (1974). The Fitting of Power Series. *Technometrics*, 16(2), 147-185.
- Cox, D. R. (1972). Regression Models and Life-Tables. *Journal of the Royal Statistical Society*, B34, 187-220.
- Wu, Y. & Liu, Y. (2009). Variable Selection in Quantile Regression. *Statistica Sinica*, 19, 801-817.
- Fan, J. & Li, R. (2001). Variable Selection via Nonconcave Penalized Likelihood. *JASA*, 96, 1348-1360.
