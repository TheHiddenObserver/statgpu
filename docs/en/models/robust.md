# Robust Regression

> Language: English  
> Last updated: 2026-07-01  
> This page: Model documentation  
> Switch: [Chinese](../../cn/models/robust.md)

## Overview

Robust regression via M-estimation with automatic scale estimation. `PenalizedRobustRegression` combines Huber, Bisquare, and Fair losses with penalty-aware solver routing. SCAD/MCP use the model's FISTA-LLA continuation path.

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
- Reduces to OLS when δ → ∞; to LAD when δ → 0
- Default ε=1.345 gives 95% efficiency at Gaussian

### Bisquare (Tukey biweight) Loss

$$
\ell(\eta, y) = \rho_c(y - \eta),\quad
\rho_c(u) = \begin{cases}
\frac{c^2}{6}\bigl[1 - (1 - (u/c)^2)^3\bigr] & |u| \le c \\
c^2/6 & |u| > c
\end{cases}
$$

- `smooth_gradient=True`, `has_hessian=True`
- Completely ignores residuals beyond threshold (gradient=0 for |u|>c)
- Higher breakdown point than Huber
- Default ε=4.685 gives 95% efficiency at Gaussian

### Fair Loss

$$
\ell(\eta, y) = c^2\left[\frac{|y-\eta|}{c} - \log(1 + \frac{|y-\eta|}{c})\right]
$$

- `smooth_gradient=True`, `has_hessian=True`
- Gentler than Huber, closer to OLS for small residuals

## Parameters

### HuberLoss

| Parameter | Default | Description |
|---|---:|---|
| `delta` | `1.0` | Threshold (fixed mode) |
| `epsilon` | `1.345` | Robustness tuning (auto-scale mode) |
| `method` | `"MAD"` | Scale estimation: `"MAD"` or `"huber_prop2"` |

### BisquareLoss

| Parameter | Default | Description |
|---|---:|---|
| `epsilon` | `4.685` | Robustness tuning |
| `method` | `"MAD"` | Scale estimation method |

## Scale Estimation

When `epsilon` is provided (auto-scale mode), scale σ is estimated before fitting:

- **MAD**: σ̂ = median(|r_i|) / 0.6745
- **Huber Proposal 2**: iteratively re-estimated

Then δ = ε · σ̂ (Huber) or c = ε · σ̂ (Bisquare).

Use `delta` for a fixed threshold (bypasses estimation).

## Solver support

| `solver` value | Huber | Bisquare | Fair | Constraint |
|---|:---:|:---:|:---:|---|
| `auto` | yes | yes | yes | Newton for L2/none; FISTA for sparse penalties |
| `fista` / `fista_bb` | yes | yes | yes | General proximal paths |
| `newton` / `lbfgs` | L2/none | L2/none | L2/none | Smooth penalties only |
| `admm` | compatible penalties | compatible penalties | compatible penalties | Only uniform sample weights |
| `irls` | no | not a documented estimator path | not a documented estimator path | Huber rejects the IRLS contract; Bisquare/Fair are not exposed here until their estimator-call signature is aligned |
| `exact` | no | no | no | Ridge-only |

Proximal Newton is not a `PenalizedRobustRegression(solver=...)` value. It is a low-level facade in the solver library; the current robust SCAD/MCP estimator route is FISTA-LLA.

## Penalty Compatibility

| Penalty | Solver (auto) | Notes |
|---------|---------------|-------|
| l2 / none | Newton | Current automatic smooth path. |
| l1 / elasticnet | FISTA | Proximal sparse path. |
| SCAD / MCP | FISTA-LLA | Continuation plus local linear approximation. |
| adaptive_l1 | Weighted-L1 FISTA | Data-dependent proximal weights. |
| group penalties | Group proximal path | Exact route depends on the penalty. |

## Examples

### CPU

```python
from statgpu.linear_model.penalized import PenalizedRobustRegression

# Huber with SCAD
model = PenalizedRobustRegression(loss='huber', penalty='scad', alpha=0.1)
model.fit(X, y)

# Bisquare with MCP
model = PenalizedRobustRegression(loss='bisquare', penalty='mcp', alpha=0.1)
model.fit(X, y)

# Fair with L2
model = PenalizedRobustRegression(loss='fair', penalty='l2', alpha=0.01)
model.fit(X, y)
```

### GPU (torch-CUDA)

```python
import torch
X_t = torch.tensor(X, dtype=torch.float64).cuda()
y_t = torch.tensor(y, dtype=torch.float64).cuda()

model = PenalizedRobustRegression(loss='huber', penalty='scad', alpha=0.1)
model.fit(X_t, y_t)
```

### Direct Solver API

```python
from statgpu.losses import HuberLoss, BisquareLoss
from statgpu.penalties import SCADPenalty
from statgpu.solvers import fista_solver

loss = HuberLoss(epsilon=1.345)
coef, n_iter = fista_solver(loss, SCADPenalty(alpha=0.1), X, y)
```

## Algorithm Details

### FISTA-LLA (SCAD/MCP)

1. Build a decreasing continuation path from a data-dependent starting penalty to the requested `alpha`.
2. Linearize the non-convex penalty at the current coefficient vector.
3. Solve the resulting weighted-L1 subproblem with FISTA.
4. Warm-start the next LLA/continuation step and stop from coefficient/LLA tolerances.

For L2/none, `auto` uses Newton with the robust loss gradient and Hessian. The standalone low-level loss classes may contain additional experimental methods; they are not automatically estimator-level `solver=` values.

## Outputs

| Attribute | Type | Description |
|-----------|------|-------------|
| `coef_` | (p,) float | Estimated coefficients |
| `intercept_` | float | Estimated intercept |
| `n_iter_` | int | Number of iterations |
| `loss` | str | Loss name ("huber", "bisquare", "fair") |

## External Validation

- **Huber**: Validated against R `MASS::rlm(psi=psi.huber)` with coefficient parity.
- **Bisquare**: Validated against R `MASS::rlm(psi=psi.bisquare)`; SCAD/MCP active set matches FISTA-LLA.
- **Fair**: Validated against R `MASS::rlm(psi=psi.fair)`.

## Notes

- `BisquareLoss` + SCAD/MCP: warm-start at LAST continuation step (target α). Starting from λ_max shrunk everything to zero in earlier versions (fixed in v0.2.1).
- Scale estimation uses CPU numpy (MAD / Proposal 2); GPU data is auto-converted.
- All losses accept `sample_weight`.
- `has_hessian=True` enables Newton/L-BFGS on smooth L2/no-penalty objectives; SCAD/MCP still route through FISTA-LLA.

## References

- Huber, P. J. (1964). Robust Estimation of a Location Parameter. *Annals of Mathematical Statistics*, 35(1), 73-101.
- Beaton, A. E. & Tukey, J. W. (1974). The Fitting of Power Series. *Technometrics*, 16(2), 147-185.
- Holland, P. W. & Welsch, R. E. (1977). Robust Regression using Iteratively Reweighted Least-Squares. *Communications in Statistics*, A6(9), 813-827.
- Fan, J. & Li, R. (2001). Variable Selection via Nonconcave Penalized Likelihood. *JASA*, 96, 1348-1360.
