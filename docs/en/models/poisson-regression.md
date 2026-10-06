# PoissonRegression

> Language: English  
> Last updated: 2026-10-06  
> This page: Model documentation  
> Switch: [Chinese](../../cn/models/poisson-regression.md)

Language switch: [Chinese](../../cn/models/poisson-regression.md)

## Overview

`PoissonRegression` implements Poisson GLM estimation for count data through the shared `GeneralizedLinearModel` stack. It is the ordinary Poisson entry point, but its default `C=1` and `solver="auto"` apply ridge regularization. Use `C=0` for an explicitly unpenalized IRLS fit. For the general alpha-based penalty interface, use `PenalizedPoissonRegression`.

Supports M-estimation sandwich inference: standard errors, z-statistics, p-values, and 95% confidence intervals via ``compute_inference=True``.  Uses expected Fisher information for model-based covariance (``cov_type='nonrobust'``; compare with ``statsmodels.GLM`` only after aligning the objective) and observed Hessian sandwich for robust covariance (``cov_type='hc0'``, ``'hc1'``).  Supports all three backends (NumPy, CuPy, Torch) via the backend-agnostic sandwich engine.

## Path

`statgpu.linear_model.PoissonRegression`

Top-level import is also available:

```python
from statgpu import PoissonRegression
```

## Objective Function

With log link, the model assumes:

$$
\mu_i = \exp(x_i^\top\beta)
$$

and minimizes the average Poisson negative log-likelihood, up to constants independent of the parameters:

$$
\min_\beta \frac{1}{n}\sum_i \left[\mu_i - y_i \log(\mu_i)\right]
$$

With an intercept, replace $x_i^\top\beta$ by $b+x_i^\top\beta$.
With analytic `sample_weight=w`, replace the average by a weighted sum divided
by `sum(w)`. For positive C, ordinary auto/IRLS adds
$\|\beta\|_2^2/(4C)$, excluding the intercept. `C=0` removes that term;
ordinary explicit `newton`, `lbfgs`, and `fista` ignore C and optimize the
unpenalized loss. Changing solvers with positive C can therefore change the
statistical model. This differs from standalone LogisticRegression's C scale.
See the [ordinary GLM objective](generalized-linear-model.md).

## Estimating Equation

The score equation for the unpenalized Poisson GLM is:

$$
\sum_i x_i(y_i - \mu_i)=0
$$

For positive C on auto/IRLS, the average-loss slope equation additionally contains the ridge gradient `beta/(2*C)`.

`PoissonRegression` defaults to `solver="auto"`, which currently dispatches to IRLS. Explicit `solver="newton"` and `solver="lbfgs"` are also available for smooth Poisson GLM objectives and run on the selected backend. The model inherits the GLM formula interface, so formula intercept semantics follow patsy/R conventions.

## Covariance/Inference

Set `compute_inference=True` to obtain post-fit inference. This complete CPU
example explicitly fits an unpenalized model:

<!-- learner-example: poisson-unpenalized -->
```python
import numpy as np
from statgpu import PoissonRegression

rng = np.random.default_rng(8)
X = rng.normal(size=(200, 2))
y = rng.poisson(np.exp(0.3 + X @ np.array([0.4, -0.2])))
model = PoissonRegression(
    C=0, solver="newton", device="cpu", compute_inference=True,
    cov_type="nonrobust", max_iter=200, tol=1e-9,
).fit(X, y)
print(np.round(model.coef_, 3))
print(model._conf_int.shape)
```

The slopes are approximately `[0.377, -0.154]` and interval shape is `(3, 2)`.
The first interval row is the intercept, followed by the two input columns.
Use `cov_type="hc0"` or `"hc1"` when those score-robust covariance assumptions
fit the application; they do not change the coefficient estimates.

**Covariance types**:
- ``'nonrobust'`` (default): model-based, φ·I(β)⁻¹ using expected Fisher information in the unpenalized case; positive-C IRLS adds penalty curvature.
- ``'hc0'``: sandwich H⁻¹·J·H⁻¹ with observed Hessian.
- ``'hc1'``: HC0 × n/(n−k) degrees-of-freedom correction.
- ``'hc2'``, ``'hc3'``, ``'hac'``: not yet implemented for Poisson (raises ``NotImplementedError``).

**Distribution**: z-statistics (asymptotic normal).  P-values are two-sided.
**Dispersion**: φ = 1.0 (Poisson variance = mean).  Pearson dispersion available via metadata.

These are marginal, asymptotic normal-reference intervals. With positive C on
IRLS, the covariance includes penalty curvature and describes the penalized fit;
it does not remove shrinkage bias or account for choosing C. For comparisons
with an unpenalized `statsmodels.GLM`, align C/solver, design, weights, covariance
and convergence settings. A comparison on one dataset is not a universal
precision guarantee.
**GPU**: fully supported (CuPy, Torch) — backend-agnostic sandwich engine runs on the same device as fitting.  No silent CPU fallback.

## Parameters

| Parameter | Default | Description |
|---|---:|---|
| `fit_intercept` | `True` | Whether to fit an intercept |
| `max_iter` | `100` | Maximum IRLS iterations |
| `tol` | `1e-4` | Convergence tolerance |
| `C` | `1.0` | Positive C: auto/IRLS adds slope penalty `sum(beta**2)/(4*C)`; zero removes it. Explicit Newton/L-BFGS/FISTA ignore C. |
| `device` | `"auto"` | `cpu` / `cuda` / `torch` / `auto` |
| `solver` | `"auto"` | `auto` / `irls` / `fista` / `newton` / `lbfgs` |
| `n_jobs` | `None` | Number of parallel jobs |
| `gpu_memory_cleanup` | `False` | Best-effort CuPy memory pool cleanup after fit |
| `compute_inference` | `False` | Compute supported coefficient uncertainty after fitting |
| `cov_type` | `"nonrobust"` | `nonrobust`, `hc0`, or `hc1` |

`formula` and `data` are arguments to
`fit(X=None, y=None, sample_weight=None, formula=None, data=None)`, not constructor
parameters. This class fixes the Poisson family and otherwise inherits the
[ordinary GLM methods and fitted fields](../reference/linear-model-api.md#generalizedlinearmodel),
including `predict`, `summary`, likelihood diagnostics and shared inference
helpers. It has no `score` or `predict_proba` method. `summary()` returns a string;
use `print(model.summary())`. On current auto/IRLS/FISTA paths, create a fresh
estimator after a failed refit, as explained in the [failed-refit warning](../reference/linear-model-api.md#failed-ordinary-glm-refits).

## CPU+GPU Examples

```python
from statgpu.linear_model import PoissonRegression

# CPU count model
m_cpu = PoissonRegression(C=0, device="cpu", max_iter=100, tol=1e-6)
m_cpu.fit(X, y_count)
mu_cpu = m_cpu.predict(X)

# GPU count model when CUDA backend is available
m_gpu = PoissonRegression(C=0, device="cuda", max_iter=100, tol=1e-6)
m_gpu.fit(X_gpu, y_count_gpu)
mu_gpu = m_gpu.predict(X_gpu)
```

Formula usage:

```python
from statgpu.linear_model import PoissonRegression

model = PoissonRegression(C=0, device="cpu")
model.fit(formula="count ~ exposure + x1 + C(group)", data=df)
pred = model.predict(df_new)
```

The `exposure` term above estimates an ordinary regression coefficient; it is
not an offset with its coefficient fixed at one. This API has no offset/exposure
argument. For large GPU workloads, prefer explicit `X, y` arrays because formula
parsing is CPU-side convenience.

## strict/approx difference

There is no public strict/approx inference switch for `PoissonRegression`. Supported inference follows the chosen covariance convention; changing a solver can change the penalty as described above.

## Outputs

- Coefficients: `intercept_`, `coef_`
- Iterations: `n_iter_`
- Methods: `fit`, `predict`
- Formula metadata is stored internally when fitting with `formula` and `data`

`predict` returns the inverse-link mean response, so for Poisson it returns estimated counts/rates \(\hat\mu\), not the linear predictor.

## FAQ

- When should I use `PoissonRegression` instead of `GeneralizedLinearModel(family="poisson")`? Use `PoissonRegression` when you want the explicit model class and clearer public API. Both share the GLM implementation.
- When should I use `PenalizedPoissonRegression`? Use it when you need L1, L2, ElasticNet, group, or adaptive penalty support.
- Does `PoissonRegression` provide standard errors and p-values? Yes — set ``compute_inference=True``.  Supports model-based (``cov_type='nonrobust'``) and sandwich (``hc0``, ``hc1``) covariance.  For unpenalized comparisons, explicitly align the objective and covariance settings.
- Does `device="cuda"` always guarantee GPU execution? For supported Poisson GLM solver paths, yes: core computation stays on CuPy or raises a clear error. `device="torch"` similarly requires Torch CUDA.

## External Validation

Poisson GLM validation should include:

- CPU/GPU coefficient and prediction consistency.
- Comparison against sklearn `PoissonRegressor` for aligned L2 settings.
- Comparison against statsmodels GLM Poisson for ordinary GLM estimation.
- Runtime benchmarks with warm-up and GPU synchronization on remote CUDA hardware.

See the [GLM comparison guidance](generalized-linear-model.md) for aligned objective and covariance settings.

## References

- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman & Hall/CRC.
- Cameron, A. C., & Trivedi, P. K. (2013). *Regression Analysis of Count Data* (2nd ed.). Cambridge University Press.
- scikit-learn PoissonRegressor documentation: [https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html)
- statsmodels GLM documentation: [https://www.statsmodels.org/stable/glm.html](https://www.statsmodels.org/stable/glm.html)
