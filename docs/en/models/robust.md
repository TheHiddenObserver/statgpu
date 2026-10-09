# Robust Regression

> Language: English  
> Last updated: 2026-10-09<br>
> This page: Model documentation  
> Switch: [Chinese](../../cn/models/robust.md)

## When to use robust regression

Consider robust regression when a few large response residuals could dominate a least-squares fit to a continuous outcome. Huber uses squared loss for small residuals and linear loss for large ones; Bisquare and Fair downweight differently. The target is the conditional location defined by the chosen loss, which need not equal the conditional mean.

`PenalizedRobustRegression` combines these losses with regularization: the loss limits the influence of unusual response residuals, while the penalty shrinks coefficients. These choices address different problems. Robust loss does not automatically repair erroneous data, high-leverage predictor outliers, or omitted variables. For a particular conditional percentile, consider [quantile regression](quantile.md).

<a id="cpu-example"></a>

## A complete CPU example

Run these steps in order. Start with Huber and a small L2 penalty; alternative losses, nonconvex penalties, and GPU use follow later.

### 1. Import

<!-- learner-example: robust-basic -->
```python
import numpy as np
from statgpu.linear_model.penalized import PenalizedRobustRegression
```

### 2. Prepare data with response outliers

`X` is a `(320, 2)` numeric matrix, one observation per row; `y` is a length-320 continuous response. Train on 240 rows and hold out 80. This simulation adds 12 unusually large responses only to the training rows, leaving the test set free of that added contamination. For real data, investigate outliers, handle missing values, and preserve feature order for prediction.

```python
rng = np.random.default_rng(27)
X = rng.normal(size=(320, 2))
y = 1.0 + X @ np.array([1.5, -0.7]) + rng.normal(scale=0.5, size=320)
y[:12] += 12.0
```

### 3. Fit a Huber model

`alpha=0.01` is an illustrative L2 strength. `solver="auto"` selects Newton for this smooth-penalty objective. By default, a residual scale is estimated before optimization and `epsilon × scale` sets the Huber threshold, as explained below.

```python
model = PenalizedRobustRegression(
    loss="huber", penalty="l2", alpha=0.01, device="cpu",
).fit(X[:240], y[:240])
print("Slopes:", np.round(model.coef_, 3))
```

The slopes round to `[1.522, -0.723]`, near the simulated `[1.5, -0.7]`; this is not an accuracy guarantee under other contamination patterns. Each slope describes a unit change in fitted location while holding the other feature fixed.

### 4. Predict and evaluate held-out rows

```python
prediction = model.predict(X[240:])
mae = np.mean(np.abs(y[240:] - prediction))
print("Predictions:", np.round(prediction[:3], 3))
print("Held-out MAE:", round(float(mae), 3))
```

The prediction array has shape `(80,)`; its first three entries round to `[2.590, 1.341, 0.479]`. Mean absolute error (MAE) is about `0.437`, in response units, and lower is better on the same evaluation set. This example computes MAE directly; the class's `score` returns response-scale R², not Huber loss.
<!-- example-end: robust-basic -->

## Choosing settings and checking results

- `epsilon` or a fixed threshold controls residual downweighting; `alpha` controls the penalty. Choose a held-out metric suited to the application.
- Feature units affect the penalty. Learn scaling and tuning inside training folds, then apply them to held-out data. Smaller training loss does not establish better generalization.
- Bisquare and SCAD/MCP involve nonconvex objectives; inspect initialization, numerical warnings, and stability.
- A robust fit does not automatically supply robust standard errors or selection-adjusted p-values. Shared inference controls remain subject to the [supported inference-method conditions](../guides/penalized-glm-inference.md).

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

## Loss parameters and estimator controls

These tables describe loss objects, not the complete estimator constructor. `PenalizedRobustRegression` exposes `loss`, `penalty`, `alpha`, `epsilon`, `method`, solver/device controls, and shared penalized-model options. For all parameters and methods, see its [public implementation](../../../statgpu/linear_model/penalized/_penalized_robust.py), the [shared penalized API](../reference/linear-model-api.md#penalizedgeneralizedlinearmodel), or installed `help(PenalizedRobustRegression)`.

The loss objects are imported from `statgpu.losses`: `HuberLoss`, `BisquareLoss`, and `FairLoss`.

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
| `delta` | `None` | Optional fixed threshold; when supplied, fixed-threshold mode is used |
| `epsilon` | `1.35` | Multiplied by the estimated scale to obtain the effective Fair threshold |
| `method` | `"MAD"` | `"MAD"` or `"huber_prop2"` |

## Scale Estimation

Automatic scale handling supports MAD and Huber Proposal-2. `PenalizedRobustRegression` estimates the scale before coefficient optimization, so ordinary `MAD` / `huber_prop2` fits use an already determined effective threshold during solver iterations.

- **MAD**: $\hat\sigma=\operatorname{median}(|r_i|)/0.6745$
- **Huber Proposal 2**: scale is estimated by a fixed-point iteration
- Huber uses $\delta=\epsilon\hat\sigma$
- Bisquare and Fair use $c=\epsilon\hat\sigma$ internally through their effective `delta`

Supplying `delta` selects fixed-threshold mode directly. `method="joint"` is Huber-only and defines a separate joint coefficient-scale optimization problem.

## Change the loss, penalty, or device

### Huber with SCAD

Each CPU subsection below reuses the imports and `X`, `y` from the completed [CPU example](#cpu-example), training only on the first 240 rows. SCAD is a nonconvex penalty; automatic dispatch uses LLA and FISTA. The alpha value below is illustrative.

<!-- example-requires: robust-basic -->
<!-- learner-example: robust-scad -->
```python
scad_model = PenalizedRobustRegression(
    loss="huber", penalty="scad", alpha=0.1, device="cpu",
).fit(X[:240], y[:240])
```
<!-- example-end: robust-scad -->

### Bisquare with MCP

Bisquare gives residuals beyond its threshold zero gradient, while MCP changes the coefficient penalty. Both differ from the preceding configuration. Reuse the imports and training data from the completed [CPU example](#cpu-example) and evaluate choices on the same held-out rows.

<!-- example-requires: robust-basic -->
<!-- learner-example: robust-bisquare -->
```python
bisquare_model = PenalizedRobustRegression(
    loss="bisquare", penalty="mcp", alpha=0.1, device="cpu",
).fit(X[:240], y[:240])
```
<!-- example-end: robust-bisquare -->

### Fair with L2

Reuse the imports and training data from the completed [CPU example](#cpu-example). Fair downweights residuals gradually; this fit retains L2 regularization and `auto` uses Newton.

<!-- example-requires: robust-basic -->
<!-- learner-example: robust-fair -->
```python
fair_model = PenalizedRobustRegression(
    loss="fair", penalty="l2", alpha=0.01, device="cpu",
).fit(X[:240], y[:240])
```
<!-- example-end: robust-fair -->

### GPU (Torch CUDA)

Reuse the imports and `X`, `y` from the completed [CPU example](#cpu-example), requesting Torch CUDA explicitly. This requires an installed Torch package and usable CUDA device; an unavailable explicit device raises. See the notes below for the CPU boundary in scale estimation.

```python
gpu_model = PenalizedRobustRegression(
    loss="huber", penalty="scad", alpha=0.1, device="torch",
).fit(X[:240], y[:240])
```

### Direct solver API

Reuse the training arrays from the completed [CPU example](#cpu-example). A low-level call requires a loss and penalty and does not add an intercept automatically. This deliberately fits a no-intercept objective with fixed Huber threshold 1, which differs from the automatic-scale estimator above. Prefer the estimator interface for ordinary modeling.

<!-- example-requires: robust-basic -->
<!-- learner-example: robust-direct-solver -->
```python
from statgpu.losses import HuberLoss
from statgpu.penalties import SCADPenalty
from statgpu.solvers import fista_solver

loss = HuberLoss(delta=1.0)
coef, n_iter = fista_solver(
    loss, SCADPenalty(alpha=0.1), X[:240], y[:240],
)
```
<!-- example-end: robust-direct-solver -->

## Solver Compatibility

The table below describes the current public model / low-level solver routes. `sample_weight` support still depends on the complete loss × solver × model path.

| Solver | Huber | Bisquare | Fair | Notes |
|--------|:---:|:---:|:---:|-------|
| Proximal Newton | ✅ (smooth objectives) | ✅ (smooth objectives) | ✅ (smooth objectives) | Current generic implementation executes Newton for L2/no penalty; non-smooth requests delegate to FISTA |
| FISTA | ✅ | ✅ | ✅ | Sparse / proximal routes |
| FISTA-BB | ✅ (supported combinations) | ✅ (supported combinations) | ✅ (supported combinations) | Adaptive step size |
| FISTA-LLA | ✅ | ✅ | ✅ | LLA route for SCAD/MCP and related non-convex penalties |
| IRLS | ❌ (currently unavailable) | ✅ (L2/no penalty) | ✅ (L2/no penalty) | Public dispatch does not currently select IRLS for Huber |
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

is directly related to the Huber first-order condition. This shows that Huber admits a natural IRLS construction, but the public API does not currently declare that route as supported; an explicit request for that combination should fail rather than silently switch algorithms.

## Outputs

| Attribute | Type | Description |
|-----------|------|-------------|
| `coef_` | `(p,)` float | Estimated coefficients |
| `intercept_` | float | Estimated intercept |
| `n_iter_` | int | Number of iterations |
| `loss` | str | Loss name (`"huber"`, `"bisquare"`, `"fair"`) |

## External Validation

- **Huber**: uses the classical Huber loss form. Numerical comparison with `MASS::rlm(psi=psi.huber)` requires the same tuning constant and scale-estimation convention; Huber IRLS is not currently part of the public support matrix.
- **Bisquare**: uses the Tukey biweight loss form. External comparison requires aligned tuning and scale conventions; current non-convex penalty routes use LLA/FISTA.
- **Fair**: uses the Fair loss form. External comparison requires aligned tuning and scale conventions.

## Notes

- Scale computation currently uses NumPy host arrays; after scale precomputation, numerical optimization continues on the selected NumPy/CuPy/Torch backend.
- `sample_weight` support depends on the loss, solver, and model route rather than being an automatic property of every robust solver.
- All three losses provide Hessian primitives. That supports smooth Newton routes, but does not imply that arbitrary non-smooth penalties support Proximal Newton.
- Huber IRLS is currently not exposed as a supported public solver route; explicitly selecting that combination raises an error.

## References

- Huber, P. J. (1964). Robust Estimation of a Location Parameter. *Annals of Mathematical Statistics*, 35(1), 73-101.
- Beaton, A. E. & Tukey, J. W. (1974). The Fitting of Power Series. *Technometrics*, 16(2), 147-185.
- Holland, P. W. & Welsch, R. E. (1977). Robust Regression using Iteratively Reweighted Least-Squares. *Communications in Statistics*, A6(9), 813-827.
- Fan, J. & Li, R. (2001). Variable Selection via Nonconcave Penalized Likelihood. *JASA*, 96, 1348-1360.
