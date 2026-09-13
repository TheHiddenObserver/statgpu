# Robust Regression

> Language: English  
> Last updated: 2026-09-13  
> This page: Model documentation  
> Switch: [Chinese](../../cn/models/robust.md)

## Overview

statgpu provides robust regression through M-estimation with robust scale handling. `PenalizedRobustRegression` combines Huber, Bisquare, and Fair losses with multiple penalty families; smooth objectives use Newton by default, while sparse and non-convex penalties use FISTA / LLA routes.

| Component | Path |
|-----------|------|
| Huber Loss | `statgpu.losses.HuberLoss` |
| Bisquare Loss | `statgpu.losses.BisquareLoss` |
| Fair Loss | `statgpu.losses.FairLoss` |
| Penalized Model | `statgpu.linear_model.penalized.PenalizedRobustRegression` |
| R Equivalent | `MASS::rlm()` |

## Loss Functions

### Huber Loss

$$
\ell(\eta, y) = \begin{cases}
\frac{1}{2}(y - \eta)^2 & |y - \eta| \le \delta \\
\delta|y - \eta| - \frac{1}{2}\delta^2 & \text{otherwise}
\end{cases}
$$

- `smooth_gradient=True`, `has_hessian=True`
- approaches OLS as $\delta\to\infty$; smaller thresholds increasingly limit the influence of large residuals
- default `epsilon=1.35`; in automatic-scale mode the effective threshold is `epsilon × scale`

### Bisquare (Tukey biweight) Loss

$$
\ell(\eta, y) = \rho_c(y - \eta),\quad
\rho_c(u) = \begin{cases}
\frac{c^2}{6}\bigl[1 - (1 - (u/c)^2)^3\bigr] & |u| \le c \\
c^2/6 & |u| > c
\end{cases}
$$

- `smooth_gradient=True`, `has_hessian=True`
- the loss is constant and the gradient is zero for $|u|>c$
- default `epsilon=4.685`, commonly used for about 95% Gaussian efficiency

### Fair Loss

$$
\ell(\eta, y) = c^2\left[\frac{|y-\eta|}{c} - \log\left(1 + \frac{|y-\eta|}{c}\right)\right]
$$

- `smooth_gradient=True`, `has_hessian=True`
- downweights residuals more gradually than hard-redescending losses

## Parameters

### `HuberLoss`

| Parameter | Default | Description |
|---|---:|---|
| `delta` | `None` | Optional fixed threshold; when supplied, fixed-threshold mode is used |
| `epsilon` | `1.35` | Multiplied by the estimated scale to obtain the effective Huber threshold |
| `method` | `"MAD"` | `"MAD"`, `"huber_prop2"`, or `"joint"` |

`method="joint"` jointly optimizes coefficients and `log_sigma`; this is a different problem from fixed-scale coefficient optimization.

### `BisquareLoss`

| Parameter | Default | Description |
|---|---:|---|
| `delta` | `None` | Optional fixed threshold |
| `epsilon` | `4.685` | Multiplied by the estimated scale to obtain the effective threshold |
| `method` | `"MAD"` | `"MAD"` or `"huber_prop2"` |

### `FairLoss`

| Parameter | Default | Description |
|---|---:|---|
| `c` | `1.4` | Fair-loss tuning constant |

## Scale Estimation

`RobustLossBase` provides MAD and Huber Proposal-2 scale estimation. The current `PenalizedRobustRegression` fit path calls `precompute_scale(...)` before entering the numerical solver, so ordinary `MAD` / `huber_prop2` coefficient optimization uses an already determined effective threshold during solver iterations.

- **MAD**: $\hat\sigma=\operatorname{median}(|r_i|)/0.6745$
- **Huber Proposal 2**: scale is estimated by a fixed-point iteration
- Huber uses $\delta=\epsilon\hat\sigma$
- Bisquare uses $c=\epsilon\hat\sigma$

Supplying `delta` selects fixed-threshold mode directly. `method="joint"` instead defines a separate joint coefficient-scale optimization problem.

## Solver Compatibility

The table below describes the current public model / low-level solver routes. `sample_weight` support still depends on the complete loss × solver × model path.

| Solver | Huber | Bisquare | Fair | Notes |
|--------|:---:|:---:|:---:|-------|
| Proximal Newton | ✅ (smooth objectives) | ✅ (smooth objectives) | ✅ (smooth objectives) | Current generic implementation executes Newton for L2/no penalty; non-smooth requests delegate to FISTA |
| FISTA | ✅ | ✅ | ✅ | Sparse / proximal routes |
| FISTA-BB | ✅ (supported combinations) | ✅ (supported combinations) | ✅ (supported combinations) | Adaptive step size |
| FISTA-LLA | ✅ | ✅ | ✅ | LLA route for SCAD/MCP and related non-convex penalties |
| IRLS | ❌ (currently unavailable) | ✅ (L2/no penalty) | ✅ (L2/no penalty) | Restoring and validating Huber IRLS is tracked in Issue #156 |
| Newton | ✅ | ✅ | ✅ | Main `solver="auto"` route for smooth L2/no-penalty objectives |
| L-BFGS | ✅ (smooth, unweighted/uniform weights) | ✅ (smooth, unweighted/uniform weights) | ✅ (smooth, unweighted/uniform weights) | Generic non-GLM `LossBase` does not currently declare direct non-uniform weighted L-BFGS |
| ADMM | ✅ (supported forms) | ✅ (supported forms) | ✅ (supported forms) | Shared entry currently accepts omitted or uniform `sample_weight` only |

### Main `solver="auto"` dispatch

| Penalty | Main route | Notes |
|---------|------------|-------|
| L2 / none | Newton | Robust losses provide gradient and Hessian primitives |
| L1 / ElasticNet | FISTA | Proximal sparse route |
| SCAD / MCP | FISTA + LLA | Local linear approximation forms a weighted convex surrogate |
| Adaptive L1 | FISTA / LLA route | Adaptive weighted proximal form |
| Group penalties | Group FISTA / Group FISTA-LLA | Corresponding group proximal operators |

## Examples

### CPU

```python
from statgpu.linear_model.penalized import PenalizedRobustRegression

# Huber + SCAD
model = PenalizedRobustRegression(loss="huber", penalty="scad", alpha=0.1)
model.fit(X, y)

# Bisquare + MCP
model = PenalizedRobustRegression(loss="bisquare", penalty="mcp", alpha=0.1)
model.fit(X, y)

# Fair + L2: solver="auto" uses the smooth Newton route
model = PenalizedRobustRegression(loss="fair", penalty="l2", alpha=0.01)
model.fit(X, y)
```

### GPU (Torch CUDA)

```python
import torch

X_t = torch.tensor(X, dtype=torch.float64).cuda()
y_t = torch.tensor(y, dtype=torch.float64).cuda()

model = PenalizedRobustRegression(loss="huber", penalty="scad", alpha=0.1)
model.fit(X_t, y_t)
```

### Direct Solver API

```python
from statgpu.losses import HuberLoss
from statgpu.penalties import SCADPenalty
from statgpu.solvers import fista_solver

loss = HuberLoss(epsilon=1.35)
coef, n_iter = fista_solver(loss, SCADPenalty(alpha=0.1), X, y)
```

## Algorithm Notes

### Smooth Huber / Bisquare / Fair

For L2/no-penalty objectives, the current automatic dispatch uses Newton. The loss supplies the actual gradient and Hessian and the solver combines a linear-system step with Armijo backtracking.

### SCAD / MCP

Non-convex penalties are handled through LLA, producing a locally weighted convex problem solved by FISTA-family inner iterations. The current generic Proximal Newton implementation does not treat an ordinary Euclidean prox as a Hessian-metric proximal subproblem, so non-smooth requests do not silently take the historical Proximal-Newton shortcut.

### Current Huber IRLS status

`HuberLoss.irls()` is currently an explicit rejection stub and `_supports_irls=False`, so `PenalizedRobustRegression(..., solver="irls")` does not enter a Huber IRLS path. For fixed-threshold Huber, the standard IRLS weight

$$
w_i=\frac{\psi_\delta(r_i)}{r_i}
=\min\left(1,\frac{\delta}{|r_i|}\right)
$$

is directly related to the Huber first-order condition. Restoring this as a maintained public solver route is being re-evaluated and validated in Issue #156. PR #151 documents the current implementation state only and does not change numerical source.

## Outputs

| Attribute | Type | Description |
|-----------|------|-------------|
| `coef_` | `(p,)` float | Estimated coefficients |
| `intercept_` | float | Estimated intercept |
| `n_iter_` | int | Number of iterations |
| `loss` | str | Loss name (`"huber"`, `"bisquare"`, `"fair"`) |

## External Validation

- **Huber**: historical validation aligned coefficients with R `MASS::rlm(psi=psi.huber)`; a restored explicit IRLS route should be revalidated under the same scale convention.
- **Bisquare**: aligned with R `MASS::rlm(psi=psi.bisquare)`; current non-convex penalty routes use LLA/FISTA.
- **Fair**: aligned with R `MASS::rlm(psi=psi.fair)`.

## Notes

- Scale computation currently uses NumPy host arrays; after scale precomputation, maintained numerical optimization continues on the selected NumPy/CuPy/Torch backend.
- `sample_weight` support depends on the loss, solver, and model route rather than being an automatic property of every robust solver.
- All three losses provide Hessian primitives. That supports smooth Newton routes, but does not imply that arbitrary non-smooth penalties support Proximal Newton.
- Huber IRLS is currently not exposed as a supported public solver route; Issue #156 tracks its mathematical and implementation validation.

## References

- Huber, P. J. (1964). Robust Estimation of a Location Parameter. *Annals of Mathematical Statistics*, 35(1), 73-101.
- Beaton, A. E. & Tukey, J. W. (1974). The Fitting of Power Series. *Technometrics*, 16(2), 147-185.
- Holland, P. W. & Welsch, R. E. (1977). Robust Regression using Iteratively Reweighted Least-Squares. *Communications in Statistics*, A6(9), 813-827.
- Fan, J. & Li, R. (2001). Variable Selection via Nonconcave Penalized Likelihood. *JASA*, 96, 1348-1360.