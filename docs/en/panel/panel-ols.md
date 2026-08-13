# PanelOLS

> Language: English  
> Last updated: 2026-08-13  
> Switch: [Chinese](../../cn/panel/panel-ols.md)

## Model

For entity and time effects,

$$
y_{it}=x_{it}^{\top}\beta+\alpha_i+\gamma_t+\varepsilon_{it}.
$$

Let $F$ denote the included fixed-effect design and

$$
M_F=I-F(F^\top F)^+F^\top.
$$

Then

$$
\widehat\beta_{\mathrm{FE}}
=(X^\top M_FX)^+X^\top M_Fy.
$$

One-way entity effects reduce to within demeaning. Two-way effects use backend-native alternating projection and fail closed if `demean_tol` is not reached within `demean_max_iter`.

## Covariance

The fit space is

$$
Z=M_FX,
\qquad
e=M_F(y-X\widehat\beta_{\mathrm{FE}}).
$$

See [Panel covariance](covariance.md). For Driscoll-Kraay, the fixed-effect nuisance rank is

$$
r_F=
\begin{cases}
N, & \text{entity only},\\
T, & \text{time only},\\
N+T-C, & \text{two way},
\end{cases}
$$

where $C$ is the number of connected components of the observed entity-time incidence graph.

## API

```python
from statgpu.panel import PanelOLS

model.fit(X, y, entity_ids=None, time_ids=None, cluster=None)
```

Main options are `entity_effects`, `time_effects`, `cov_type`, `bandwidth`, `kernel`, `group_debias`, `demean_max_iter`, `demean_tol`, `alpha`, and `device`.

Formula input supports either additive `EntityEffects` / `TimeEffects` tokens or pipe fixed-effect syntax; the two syntaxes cannot be mixed. Prediction using stored two-way effects requires identified entity-time labels in the same fitted incidence component.

Pooling F and Hausman tests are documented in [Panel diagnostics](diagnostics.md).
