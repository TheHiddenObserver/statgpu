# Panel

> Language: English  
> Last updated: 2026-05-28  
> This page: Model documentation  
> Switch: [Chinese](../../models/panel.md)

Language switch: [Chinese](../../models/panel.md)

## Overview

The `panel` module provides panel data models for longitudinal/panel data. `PanelOLS` estimates fixed effects (entity and/or time effects) with non-robust, HC1 robust, and clustered standard errors. `RandomEffects` implements feasible GLS random effects using the Swamy-Arora variance component estimator. Both classes support CPU, CuPy, and PyTorch backends with automatic device detection.

## Path

- `statgpu.panel.PanelOLS`
- `statgpu.panel.RandomEffects`
- `statgpu.panel.clustered_covariance`
- `statgpu.panel.two_way_clustered_covariance`

## Objective Function

**PanelOLS** solves within-transformation OLS. The dependent variable and regressors are demeaned to sweep out fixed effects. For entity effects the transformation is:

$$
y_{it}^{within} = y_{it} - \bar{y}_{i\cdot}
$$

For two-way (entity + time) fixed effects the double-demeaned residual is:

$$
y_{it}^{within} = y_{it} - \bar{y}_{i\cdot} - \bar{y}_{\cdot t} + \bar{y}_{\cdot\cdot}
$$

where \(\bar{y}_{i\cdot}\) is the entity mean, \(\bar{y}_{\cdot t}\) is the time mean, and \(\bar{y}_{\cdot\cdot}\) is the grand mean. The same transformation is applied column-by-column to \(X\).

**RandomEffects** estimates a variance-components model:

$$
y_{it} = \alpha + X_{it}'\beta + a_i + \epsilon_{it}
$$

where \(a_i \sim \text{iid}(0, \sigma^2_a)\) is the individual random effect and \(\epsilon_{it} \sim \text{iid}(0, \sigma^2_e)\) is the idiosyncratic error. The Swamy-Arora estimator obtains \(\hat{\sigma}^2_e\) from the within estimator and \(\hat{\sigma}^2_a\) from the between estimator, then applies feasible GLS.

## Estimating Equation

**PanelOLS** fits OLS on the demeaned data:

$$
\hat{\beta} = (X_d^\top X_d)^{-1} X_d^\top y_d
$$

where \(X_d\) and \(y_d\) are the entity- (and optionally time-) demeaned regressors and dependent variable.

**RandomEffects** proceeds in six steps:

1. **Between estimation**: compute group means \(\bar{y}_i, \bar{X}_i\) and run OLS on the between data to obtain \(\hat{\beta}_{between}\).

2. **Within estimation**: entity-demean OLS to obtain \(\hat{\beta}_{within}\) and the residual sum of squares \(RSS_{within}\).

3. **Variance components**:
   \[
   \hat{\sigma}^2_e = \frac{RSS_{within}}{N - n_{entities} - k}, \qquad
   \hat{\sigma}^2_a = \max\!\left(0,\; \frac{RSS_{between}}{n_{entities}} - \frac{\hat{\sigma}^2_e}{\bar{T}}\right)
   \]
   where \(\bar{T}\) is the average number of observations per entity.

4. **GLS weight** per entity:
   \[
   \theta_i = 1 - \sqrt{\frac{\hat{\sigma}^2_e}{\hat{\sigma}^2_e + T_i\,\hat{\sigma}^2_a}}
   \]

5. **Quasi-demeaned OLS**: apply the partial demeaning transformation
   \[
   y^*_{it} = y_{it} - \theta_i\,\bar{y}_{i\cdot}, \qquad
   X^*_{it} = X_{it} - \theta_i\,\bar{X}_{i\cdot}
   \]
   and run OLS on the transformed data.

6. **Inference**: compute the OLS covariance on the quasi-demeaned data.

## Covariance/Inference

The `cov_type` parameter on `PanelOLS` selects the inference method:

- **`nonrobust`**: classical OLS covariance \(\hat{\sigma}^2 (X_d^\top X_d)^{-1}\). P-values from the \(t\)-distribution with \(df_{resid}\) degrees of freedom.
- **`robust`**: HC1 sandwich estimator (White 1980, with finite-sample \(n/(n-k)\) correction). P-values from the standard normal.
- **`clustered`**: cluster-robust sandwich estimator (Cameron & Miller 2015). P-values from the standard normal. For two-way clustering, pass a 2-column cluster array; the variance is computed via the Cameron, Gelbach & Miller (2011) method.

`RandomEffects` uses nonrobust OLS inference on the quasi-demeaned data by default.

Outputs after `fit()`: `coef_`, `bse_`, `tvalues_`, `pvalues_`, `conf_int_`, `rsquared_within` (PanelOLS).

## Parameters

### PanelOLS

| Parameter | Default | Description |
|---|---:|---|
| `entity_effects` | `False` | Include entity (individual) fixed effects |
| `time_effects` | `False` | Include time fixed effects |
| `cov_type` | `'nonrobust'` | Covariance type: `'nonrobust'`, `'robust'`, or `'clustered'` |
| `device` | `"auto"` | Computation device: `"cpu"`, `"cuda"`, or `"auto"` |

### RandomEffects

| Parameter | Default | Description |
|---|---:|---|
| `device` | `"auto"` | Computation device: `"cpu"`, `"cuda"`, or `"auto"` |

## CPU+GPU Examples

```python
from statgpu.panel import PanelOLS, RandomEffects
import numpy as np

# Generate panel data
n_entities, n_times = 50, 10
n = n_entities * n_times
entity_ids = np.repeat(np.arange(n_entities), n_times)
time_ids = np.tile(np.arange(n_times), n_entities)
X = np.random.randn(n, 3)
y = X @ [1.0, -0.5, 0.3] + np.random.randn(n) * 0.1

# --- Fixed effects (CPU) ---

fe = PanelOLS(entity_effects=True, cov_type='robust', device='cpu')
fe.fit(y, X, entity_ids=entity_ids)
print(f"Coef: {fe.coef_}, SE: {fe.bse_}")
print(f"R-squared (within): {fe.rsquared_within:.4f}")

# Two-way fixed effects with clustered SE
fe2 = PanelOLS(entity_effects=True, time_effects=True,
               cov_type='clustered', device='cpu')
fe2.fit(y, X, entity_ids=entity_ids, time_ids=time_ids,
        cluster=entity_ids)
print(f"Two-way FE coef: {fe2.coef_}")

# --- Random effects (CPU) ---

re = RandomEffects(device='cpu')
re.fit(y, X, entity_ids=entity_ids)
print(f"RE coef: {re.coef_}, theta: {re.theta_}")
print(f"Variance components: {re.variance_components_}")

# --- Random effects (GPU) ---

re_gpu = RandomEffects(device='cuda')
re_gpu.fit(y, X, entity_ids=entity_ids)
print(f"GPU RE coef: {re_gpu.coef_}, theta: {re_gpu.theta_}")

# --- GPU with PyTorch tensors ---

import torch
y_torch = torch.from_numpy(y).cuda().float()
X_torch = torch.from_numpy(X).cuda().float()
fe_torch = PanelOLS(entity_effects=True, cov_type='robust', device='cuda')
fe_torch.fit(y_torch, X_torch, entity_ids=entity_ids)
print(f"Torch FE coef: {fe_torch.coef_}")
```

## strict/approx difference

There is no strict/approx mode for panel models. The `cov_type` parameter controls the inference method:

- `'nonrobust'`: classical OLS standard errors, assumes homoskedasticity and no within-cluster correlation. Uses the \(t\)-distribution for p-values.
- `'robust'`: HC1 heteroskedasticity-robust standard errors (White sandwich). Uses the normal distribution for p-values.
- `'clustered'`: cluster-robust standard errors allowing arbitrary within-cluster correlation. Uses the normal distribution for p-values. Supports one-way and two-way clustering.

## Outputs

### Fitted attributes

| Attribute | Shape | Description |
|---|---|---|
| `coef_` | `(k,)` | Estimated coefficients |
| `bse_` | `(k,)` | Standard errors |
| `tvalues_` | `(k,)` | T-statistics (or Z-statistics for robust/clustered) |
| `pvalues_` | `(k,)` | P-values |
| `conf_int_` | `(k, 2)` | 95% confidence intervals |
| `rsquared_within` | scalar | Within R-squared (PanelOLS only) |
| `theta_` | scalar | GLS transformation weight (RandomEffects only) |
| `variance_components_` | dict | `{'sigma2_e': float, 'sigma2_a': float}` (RandomEffects only) |
| `nobs` | int | Number of observations |
| `df_resid` | int | Residual degrees of freedom |

### Methods

| Method | Returns | Description |
|---|---|---|
| `fit(y, X, entity_ids, ...)` | `self` | Fit the panel model. Requires `entity_ids` (1-D array of entity labels). Optional: `time_ids`, `cluster`. |
| `predict(X, entity_ids)` | `ndarray` | Predicted values |
| `summary()` | str | Formatted summary table |

## FAQ

**When should I use FE vs RE?**
Use Fixed Effects (`PanelOLS`) when entity effects may be correlated with regressors. The FE estimator is consistent regardless of this correlation. Use Random Effects (`RandomEffects`) for efficiency when the effects are uncorrelated with regressors. A Hausman test can help decide: if the Hausman statistic is significant, prefer FE.

**How do I do two-way clustering?**
Pass `cluster` as a 2-column array (or list of two arrays) to `PanelOLS.fit()`. Each column defines one clustering dimension. The two-way clustered variance is computed via the Cameron, Gelbach & Miller (2011) method, which projects onto the union of the two cluster sets.

**What is the difference from `linearmodels`?**
The statistical methods are the same as `linearmodels.panel.PanelOLS` and `linearmodels.panel.RandomEffects`. The main difference is GPU acceleration: statgpu dispatches core linear algebra to CuPy or PyTorch backends, providing speedups on large panel datasets when a GPU is available.

**Can I pass CuPy or PyTorch arrays directly?**
Yes. Pass a CuPy ndarray or PyTorch tensor as `y` or `X` and the backend is auto-detected from the input type. You can also set `device="cuda"` explicitly with NumPy input to force GPU computation.

**What happens with unbalanced panels?**
Both `PanelOLS` and `RandomEffects` handle unbalanced panels. Entity and time identifiers define the structure; each entity can have a different number of observations \(T_i\). The GLS weight \(\theta_i\) in RandomEffects varies by entity to account for differing \(T_i\).

## External Validation

Validated against `linearmodels.panel.PanelOLS` and `linearmodels.panel.RandomEffects` (from the `linearmodels` package). Coefficient estimates, standard errors, and variance components match to relative error < 1e-12 on test datasets. Consistency checks are maintained in `dev/tests/test_external_consistency.py`.

## References

- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press.
- Cameron, A. C., & Miller, D. L. (2015). A practitioner's guide to cluster-robust inference. *Journal of Human Resources*, 50(2), 317-372. [https://doi.org/10.3368/jhr.50.2.317](https://doi.org/10.3368/jhr.50.2.317)
- Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011). Robust inference with multiway clustering. *Journal of Business & Economic Statistics*, 29(3), 238-249. [https://doi.org/10.1198/jbes.2010.07136](https://doi.org/10.1198/jbes.2010.07136)
- Swamy, P. A. V. B., & Arora, S. S. (1972). The exact finite sample properties of the estimators of coefficients in the error components regression models. *Econometrica*, 40(2), 261-275. [https://doi.org/10.2307/1909125](https://doi.org/10.2307/1909125)
- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*, 48(4), 817-838. [https://doi.org/10.2307/1912934](https://doi.org/10.2307/1912934)
