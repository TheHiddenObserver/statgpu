# PoissonRegression

> Language: English  
> Last updated: 2026-05-20  
> This page: Model documentation  
> Switch: [Chinese](../../models/poisson-regression.md)

Language switch: [Chinese](../../models/poisson-regression.md)

## Overview

`PoissonRegression` implements Poisson GLM estimation for count data through the shared `GeneralizedLinearModel` stack. It is the ordinary, non-penalized Poisson entry point. For penalized Poisson models, use `PenalizedPoissonRegression`.

Supports M-estimation sandwich inference: standard errors, z-statistics, p-values, and 95% confidence intervals via ``compute_inference=True``.  Uses expected Fisher information for model-based covariance (``cov_type='nonrobust'``, matching ``statsmodels.GLM``) and observed Hessian sandwich for robust covariance (``cov_type='hc0'``, ``'hc1'``).  CPU only in this release; GPU inference raises ``NotImplementedError`` explicitly.

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

When `C` is finite, the shared IRLS path can include an L2-style ridge term controlled through the inherited GLM machinery.

## Estimating Equation

The score equation for the unpenalized Poisson GLM is:

$$
\sum_i x_i(y_i - \mu_i)=0
$$

`PoissonRegression` defaults to `solver="auto"`, which currently dispatches to IRLS. Explicit `solver="newton"` and `solver="lbfgs"` are also available for smooth Poisson GLM objectives and run on the selected backend. As of v23c, `solver="lbfgs"` correctly handles L2 penalties. The model inherits the GLM formula interface, so formula intercept semantics follow patsy/R conventions.

## Covariance/Inference

Set ``compute_inference=True`` to obtain post-fit inference:

```python
from statgpu import PoissonRegression
import numpy as np

X = np.random.randn(200, 5)
y = np.random.poisson(np.exp(0.3 + X @ [0.5, -0.3, 0.0, 0.8, 0.0]))

# Model-based (Fisher) SEs — matches statsmodels summary.glm()
m = PoissonRegression(solver='newton', compute_inference=True, cov_type='nonrobust')
m.fit(X, y)
print(m._bse)       # standard errors
print(m._pvalues)   # two-sided p-values (normal)
print(m._conf_int)  # 95% CI

# Robust sandwich SEs — HC0/HC1
m2 = PoissonRegression(solver='newton', compute_inference=True, cov_type='hc0')
m2.fit(X, y)
```

**Covariance types**:
- ``'nonrobust'`` (default): model-based, φ·I(β)⁻¹ using expected Fisher information.  Matches ``statsmodels.GLM(..., family=Poisson()).fit()``.
- ``'hc0'``: sandwich H⁻¹·J·H⁻¹ with observed Hessian.
- ``'hc1'``: HC0 × n/(n−k) degrees-of-freedom correction.
- ``'hc2'``, ``'hc3'``, ``'hac'``: not yet implemented for Poisson (raises ``NotImplementedError``).

**Distribution**: z-statistics (asymptotic normal).  P-values are two-sided.
**Dispersion**: φ = 1.0 (Poisson variance = mean).  Pearson dispersion available via metadata.

**Strict inference**: model-based nonrobust Poisson matches statsmodels to machine precision (|bse diff| < 1e-9 for n≥200).
**GPU**: raises ``NotImplementedError`` — CPU only in this release.  No silent fallback.

## Parameters

| Parameter | Default | Description |
|---|---:|---|
| `fit_intercept` | `True` | Whether to fit an intercept |
| `max_iter` | `100` | Maximum IRLS iterations |
| `tol` | `1e-4` | Convergence tolerance |
| `C` | `1.0` | Inverse regularization strength used by the inherited GLM IRLS path |
| `device` | `"auto"` | `cpu` / `cuda` / `torch` / `auto` |
| `solver` | `"auto"` | `auto` / `irls` / `fista` / `newton` / `lbfgs` |
| `n_jobs` | `None` | Number of parallel jobs |
| `gpu_memory_cleanup` | `False` | Best-effort CuPy memory pool cleanup after fit |
| `formula` | `None` | Optional patsy-style formula passed to `fit` |
| `data` | `None` | DataFrame used with `formula` |

## CPU+GPU Examples

```python
from statgpu.linear_model import PoissonRegression

# CPU count model
m_cpu = PoissonRegression(device="cpu", max_iter=100, tol=1e-6)
m_cpu.fit(X, y_count)
mu_cpu = m_cpu.predict(X)

# GPU count model when CUDA backend is available
m_gpu = PoissonRegression(device="cuda", max_iter=100, tol=1e-6)
m_gpu.fit(X_gpu, y_count_gpu)
mu_gpu = m_gpu.predict(X_gpu)
```

Formula usage:

```python
from statgpu.linear_model import PoissonRegression

model = PoissonRegression()
model.fit(formula="count ~ exposure + x1 + C(group)", data=df)
pred = model.predict(df_new)
```

For large GPU workloads, prefer explicit `X, y` arrays because formula parsing is CPU-side convenience.

## strict/approx difference

There is no public strict/approx inference switch for `PoissonRegression`. The release validation focus is coefficient, prediction, objective, and runtime consistency across available backends and external frameworks.

## Outputs

- Coefficients: `intercept_`, `coef_`
- Iterations: `n_iter_`
- Methods: `fit`, `predict`
- Formula metadata is stored internally when fitting with `formula` and `data`

`predict` returns the inverse-link mean response, so for Poisson it returns estimated counts/rates \(\hat\mu\), not the linear predictor.

## FAQ

- When should I use `PoissonRegression` instead of `GeneralizedLinearModel(family="poisson")`? Use `PoissonRegression` when you want the explicit model class and clearer public API. Both share the GLM implementation.
- When should I use `PenalizedPoissonRegression`? Use it when you need L1, L2, ElasticNet, group, or adaptive penalty support.
- Does `PoissonRegression` provide standard errors and p-values? Yes — set ``compute_inference=True``.  Supports model-based (``cov_type='nonrobust'``) and sandwich (``hc0``, ``hc1``) covariance.  Validated against statsmodels to |bse diff| < 1e-9.
- Does `device="cuda"` always guarantee GPU execution? For supported Poisson GLM solver paths, yes: core computation stays on CuPy or raises a clear error. `device="torch"` similarly requires Torch CUDA.

## External Validation

Poisson GLM validation should include:

- CPU/GPU coefficient and prediction consistency.
- Comparison against sklearn `PoissonRegressor` for aligned L2 settings.
- Comparison against statsmodels GLM Poisson for ordinary GLM estimation.
- Runtime benchmarks with warm-up and GPU synchronization on remote CUDA hardware.

Current remote GLM validation entry points:

```bash
python dev/tests/run_remote_v10_accuracy.py
python dev/benchmarks/run_remote_v10_benchmark.py
```

## References

- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman & Hall/CRC.
- Cameron, A. C., & Trivedi, P. K. (2013). *Regression Analysis of Count Data* (2nd ed.). Cambridge University Press.
- scikit-learn PoissonRegressor documentation: [https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html)
- statsmodels GLM documentation: [https://www.statsmodels.org/stable/glm.html](https://www.statsmodels.org/stable/glm.html)
