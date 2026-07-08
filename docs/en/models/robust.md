# Robust Regression

> Language: English  
> Last updated: 2026-07-01  
> This page: Model documentation  
> Switch: [Chinese](../../cn/models/robust.md)

## Overview

Robust regression via M-estimation with automatic scale estimation. `PenalizedRobustRegression` wraps Huber, Bisquare, and Fair losses with up to 10 penalty types and 8 solvers, including the specialized Proximal Newton solver for SCAD/MCP.

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

## Solver Compatibility

| Solver | Huber | Bisquare | Fair | Notes |
|--------|:---:|:---:|:---:|-------|
| Proximal Newton | ✅ | ✅ | ✅ | Fastest for SCAD/MCP: 5-10 iterations |
| FISTA | ✅ | ✅ | ✅ | Any penalty |
| FISTA-BB | ✅ | ✅ | ✅ | Adaptive step size |
| FISTA-LLA | ✅ | ✅ | ✅ | LLA outer loop |
| IRLS | ✅ (L2) | ✅ (L2) | ✅ (L2) | Smooth penalties only |
| Newton | ✅ | ✅ | ✅ | L2 penalty |
| L-BFGS | ✅ | ✅ | ✅ | Moderate dimensions |
| ADMM | ✅ | ✅ | ✅ | Augmented Lagrangian |

## Penalty Compatibility

| Penalty | Solver (auto) | Notes |
|---------|---------------|-------|
| l2 / none | IRLS or Newton | Fast convergence. |
| SCAD / MCP | Proximal Newton | 5-10 iterations. Warm-start at target α. |
| adaptive_l1 | FISTA-LLA | Weighted L1 proximal. |
| group_* | FISTA-LLA | Group proximal operators. |

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

### Proximal Newton (SCAD/MCP)

1. Compute Hessian H = X'WX (W = diagonal Hessian weights) and gradient g
2. Newton direction: d = -H⁻¹·g
3. Armijo line search (max 25 retries) with proximal step
4. Update: β_new = proximal(β − step·d, step)
5. Typically 5-10 iterations per LLA step

### IRLS (L2/none)

Huber/Bisquare/Fair all have `irls()` methods:
1. IRLS weights from ψ'(r_i) / r_i
2. Solve weighted least squares with L2 penalty
3. Repeat until convergence

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
- `has_hessian=True` for all three losses enables proximal Newton for SCAD/MCP.

## References

- Huber, P. J. (1964). Robust Estimation of a Location Parameter. *Annals of Mathematical Statistics*, 35(1), 73-101.
- Beaton, A. E. & Tukey, J. W. (1974). The Fitting of Power Series. *Technometrics*, 16(2), 147-185.
- Holland, P. W. & Welsch, R. E. (1977). Robust Regression using Iteratively Reweighted Least-Squares. *Communications in Statistics*, A6(9), 813-827.
- Fan, J. & Li, R. (2001). Variable Selection via Nonconcave Penalized Likelihood. *JASA*, 96, 1348-1360.
