# Loss Functions (LossBase)

> Language: English  
> Last updated: 2026-09-16  
> This page: Low-level loss reference  
> Switch: [Chinese](../../cn/models/losses.md)

## Overview

`LossBase` is the common low-level interface between statgpu loss functions, penalties, and optimization solvers. Most users should start from a model class rather than instantiate a loss directly; this page is mainly useful when you need to understand solver compatibility or call a solver at the loss level.

Related model documentation:

- [Quantile Regression](quantile.md) — check loss (also called pinball loss) and quantile-regression estimators
- [Robust Regression](robust.md) — Huber, Bisquare, Fair losses and robust estimators
- [CoxPH](coxph.md) — Cox partial likelihood, tie handling, counting-process data, and inference
- [GeneralizedLinearModel](generalized-linear-model.md) — GLM objective and analytic-weight semantics

Five non-GLM loss types extend the shared loss interface:

| Loss | Class | Related R function | Typical use |
|------|-------|--------------------|-------------|
| Quantile | `QuantileLoss` | `quantreg::rq()` | Conditional quantiles, median regression |
| Huber | `HuberLoss` | `MASS::rlm()` | Robust M-estimation |
| Bisquare | `BisquareLoss` | `MASS::rlm(psi="bisquare")` | Redescending M-estimation |
| Fair | `FairLoss` | custom psi/reference required; MASS has no built-in Fair psi | Robust Fair loss |
| Cox PH | `CoxPartialLikelihoodLoss` | `survival::coxph()` | Survival analysis |

A shared interface does **not** imply a shared statistical contract. A low-level method accepting `sample_weight` does not mean that the loss supports arbitrary non-uniform weights under every solver or model.

This page lists the **loss layer only**. `QuantileRegression`, `PenalizedQuantileRegression`, `PenalizedRobustRegression`, `CoxPH`, and `PenalizedCoxPHModel` are estimators and belong in their model documentation rather than in the `statgpu.losses` public-path list.

Panel estimators are also outside the `LossBase` hierarchy. They use the separate `BasePanelModel` estimator architecture for panel-data preparation, transformed OLS, covariance/inference, and fit lifecycle, so panel models should not be added to the loss hierarchy on this page.

## Public paths

```text
statgpu.losses.LossBase
statgpu.losses.QuantileLoss
statgpu.losses.HuberLoss
statgpu.losses.BisquareLoss
statgpu.losses.FairLoss
statgpu.losses.CoxPartialLikelihoodLoss
```

## Architecture

```text
LossBase (statgpu/losses/_base.py)
├── GLMLoss (statgpu/glm_core/_base.py) — GLM-specific value/gradient/Hessian contract
│   ├── SquaredErrorLoss, LogisticLoss, PoissonLoss, ...
├── QuantileLoss — check / pinball loss, non-smooth
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

The shared `LossBase.value()`, `gradient()`, and `fused_value_and_gradient()` primitives already accept `sample_weight`. On those shared first-order primitives, analytic weights use the normalized data-fit term

$$
\frac{\sum_i w_i\ell_i(\beta)}{\sum_i w_i}.
$$

This standardizes the **loss-layer first-order weight semantics only**. A complete weighted fitting route still depends on the concrete loss, solver, and estimator; algorithms that require Hessians, Fisher information, Lipschitz constants, smooth-gradient differences, or quasi-Newton curvature must define those quantities consistently with the same objective. For maintained combinations, see the [Solver × Penalty Compatibility Matrix](../guides/solver-penalty-matrix.md) and [Solver Algorithms](../guides/solver-algorithms.md).

### Quantile loss (check / pinball loss)

**Check loss** and **pinball loss** are two names for the same quantile-regression loss; they are not two different objectives.

$$
\ell(\eta, y) = \rho_\tau(y - \eta), \quad
\rho_\tau(u) = u\left(\tau - \mathbf{1}\{u < 0\}\right).
$$

Equivalently,

$$
\rho_\tau(u)=
\begin{cases}
\tau u, & u\ge 0,\\
(\tau-1)u, & u<0.
\end{cases}
$$

The asymmetric linear slopes select the requested conditional quantile and give the objective its familiar pinball/check shape. At $\tau=0.5$,

$$
\rho_{0.5}(u)=\frac12|u|,
$$

so median regression differs from absolute loss only by a constant scale factor. Its subgradient is a step function, so low-level algorithms that require smooth-gradient curvature are not maintained Quantile routes.

### Huber loss

$$
\ell(\eta, y) = \begin{cases}
\frac{1}{2}(y - \eta)^2 & \text{if } |y - \eta| \le \delta, \\
\delta\left(|y - \eta| - \frac{1}{2}\delta\right) & \text{otherwise}.
\end{cases}
$$

### Bisquare loss (Tukey biweight)

Let $u=y-\eta$. Then

$$
\ell(\eta,y)=\rho_c(u),
$$

with

$$
\rho_c(u)=
\begin{cases}
\frac{c^2}{6}\left[1-\left(1-(u/c)^2\right)^3\right], & |u|\le c,\\
\frac{c^2}{6}, & |u|>c.
\end{cases}
$$

### Cox partial likelihood (negative log scale)

$$
\ell(\beta) = -\frac{1}{n}\log L(\beta).
$$

The low-level `CoxPartialLikelihoodLoss` targets ordinary right-censored data and accepts either a `{"time": ..., "event": ...}` dictionary or an `(n, 2)` `[time, event]` response. It implements Breslow or Efron partial likelihood. The high-level `statgpu.survival.CoxPH` estimator additionally supports Exact ties, delayed-entry/start-stop data, and strata.

The Cox partial likelihood is invariant to an additive constant in the linear predictor. If

$$
\eta_i=x_i^\top\beta+c,
$$

every risk-set numerator and denominator receives the same factor $e^c$, which cancels. Equivalently, the constant can be absorbed into the unknown baseline hazard. The intercept is therefore not identifiable from the Cox partial likelihood; `PenalizedCoxPHModel(fit_intercept=False)` reflects the model definition rather than a missing implementation feature.

## Solver compatibility

The table below describes the maintained **unweighted** low-level compatibility. It should not be read as a weighted-support matrix.

| Solver | Quantile | Huber | Bisquare | Fair | Cox PH |
|--------|----------|-------|----------|------|--------|
| FISTA | ✅ | ✅ | ✅ | ✅ | ✅ |
| FISTA-BB | ❌ (non-smooth gradient) | ✅ | ✅ | ✅ | ✅ |
| FISTA-LLA | ✅ (SCAD/MCP) | ✅ | ✅ | ✅ | ✅ (SCAD/MCP) |
| Proximal IRLS-CD | ✅ (SCAD/MCP) | ❌ | ❌ | ❌ | ❌ |
| Proximal Newton | ❌ (no Hessian) | ✅ (L2/no penalty) | ✅ (L2/no penalty) | ✅ (L2/no penalty) | ❌ |
| Newton | ❌ (no Hessian) | ✅ | ✅ | ✅ | ✅ |
| L-BFGS | ❌ (non-smooth objective) | ✅ | ✅ | ✅ | ✅ |
| ADMM | ❌ (shared w-update requires smooth gradient) | ✅ | ✅ | ✅ | ✅ |
| IRLS | ✅ (L2/no penalty) | ❌ (currently unavailable) | ✅ (L2/no penalty) | ✅ (L2/no penalty) | ❌ |

For Quantile loss, the public low-level `fista_bb_solver`, `lbfgs_solver`, and `admm_solver` exports fail closed before numerical iteration. Ordinary FISTA remains the maintained sparse convex route; L2/no-penalty uses Quantile IRLS, and SCAD/MCP use Proximal IRLS-CD. Huber IRLS is not currently exposed as a maintained public solver route; the ❌ entry there means that public dispatch does not select that route today.

### Non-uniform weights and direct L-BFGS

Direct L-BFGS is deliberately conservative about non-uniform weights:

| Direct `lbfgs_solver` route | Genuine non-uniform `sample_weight` |
|---|---|
| Maintained `GLMLoss` implementations | ✅ Supported by the GLM weighted-objective contract |
| Quantile | ❌ L-BFGS is not a maintained Quantile route even without weights |
| Huber / Bisquare / Fair | ❌ Not implied by unweighted L-BFGS support |
| Cox partial likelihood | ❌ `sample_weight` is currently unsupported |

`GLMLoss` can opt in because its fused value/gradient contract defines one normalized analytic-weight objective. Generic non-GLM losses remain closed to genuine non-uniform direct L-BFGS weights unless that specific loss later defines and validates an equivalent contract. For Quantile, the stronger non-smooth-objective exclusion applies regardless of weights.

Uniform weights retain historical unweighted L-BFGS behavior only for losses whose unweighted L-BFGS route remains maintained.

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
| `epsilon` | `4.685` | Robustness tuning constant (commonly used for about 95% Gaussian efficiency) |
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

## Examples

### Direct CPU solver calls

```python
from statgpu.losses import HuberLoss
from statgpu.solvers import lbfgs_solver

# An intentionally unweighted low-level smooth-loss example.
huber_loss = HuberLoss()
coef_h, n_iter_h = lbfgs_solver(huber_loss, None, X, y)
```

Quantile is intentionally absent from the direct L-BFGS example: its step-function subgradient does not satisfy the maintained smooth-objective L-BFGS contract. Use the Quantile model routes documented on [Quantile Regression](quantile.md).

### GPU (Torch CUDA)

```python
import torch
from statgpu.losses import HuberLoss
from statgpu.penalties import SCADPenalty
from statgpu.solvers import fista_solver

X_t = torch.tensor(X, dtype=torch.float64).cuda()
y_t = torch.tensor(y, dtype=torch.float64).cuda()

loss = HuberLoss()
coef, n_iter = fista_solver(loss, SCADPenalty(alpha=0.1), X_t, y_t)
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

For estimator APIs, data scope, and inference behavior of `CoxPH`, `CoxPHCV`, and `PenalizedCoxPHModel`, see the [CoxPH model documentation](coxph.md).

## Validation and notes

Loss-specific numerical validation follows the corresponding model and solver contracts. Cross-backend parity and statistical weight semantics are separate questions: agreement across NumPy/CuPy/Torch is not, by itself, evidence that a new weighting interpretation is valid for a loss that has not declared one.

- `QuantileLoss` is non-smooth and has no Hessian; maintained routes are ordinary FISTA for supported sparse convex objectives, IRLS for L2/no penalty, and Proximal IRLS-CD/FISTA-LLA where explicitly documented. FISTA-BB, L-BFGS, and shared ADMM fail closed for Quantile.
- Robust losses expose estimator-level weight semantics; those semantics do not automatically extend to direct non-uniform weighted L-BFGS.
- `CoxPartialLikelihoodLoss` currently rejects `sample_weight`; any future Cox case/frequency/sampling-weight support needs a separately defined statistical contract.
- Panel estimators use the separate `BasePanelModel` architecture rather than inheriting from `LossBase`.
- See [Loss × Penalty × Solver Framework](../guides/loss-penalty-solver-framework.md) for broader compatibility details.

## References

- Koenker, R. & Bassett, G. (1978). Regression Quantiles. *Econometrica*, 46(1), 33-50.
- Huber, P. J. (1964). Robust Estimation of a Location Parameter. *Annals of Mathematical Statistics*, 35(1), 73-101.
- Beaton, A. E. & Tukey, J. W. (1974). The Fitting of Power Series. *Technometrics*, 16(2), 147-185.
- Cox, D. R. (1972). Regression Models and Life-Tables. *Journal of the Royal Statistical Society*, B34, 187-220.
- Wu, Y. & Liu, Y. (2009). Variable Selection in Quantile Regression. *Statistica Sinica*, 19, 801-817.
- Fan, J. & Li, R. (2001). Variable Selection via Nonconcave Penalized Likelihood. *JASA*, 96, 1348-1360.
