# Panel

> Language: English  
> Last updated: 2026-07-12  
> This page: Model documentation  
> Switch: [Chinese](../../models/panel.md)

Language switch: [Chinese](../../models/panel.md)

## Overview

The `panel` module provides panel data models for longitudinal/panel data. `PanelOLS` estimates fixed effects (entity and/or time effects) with non-robust, HC1 robust, and clustered standard errors. `RandomEffects` implements feasible GLS random effects using the Swamy-Arora variance component estimator. `PooledOLS` runs OLS on the stacked panel without demeaning. `BetweenOLS` collapses to group means and runs OLS on the between data. `FirstDifferenceOLS` takes first differences within entities to remove fixed effects. `FamaMacBeth` implements the two-pass cross-sectional regression approach common in asset pricing. All classes support CPU, CuPy, and PyTorch backends with automatic device detection.

## Path

- `statgpu.panel.PanelOLS`
- `statgpu.panel.RandomEffects`
- `statgpu.panel.PooledOLS`
- `statgpu.panel.BetweenOLS`
- `statgpu.panel.FirstDifferenceOLS`
- `statgpu.panel.FamaMacBeth`
- `statgpu.panel.clustered_covariance`
- `statgpu.panel.two_way_clustered_covariance`
- `statgpu.panel.hac_covariance`

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

**PooledOLS** runs standard OLS on the stacked panel data without any transformation:

$$
y_{it} = \alpha + X_{it}'\beta + u_{it}
$$

All observations are pooled together and OLS is applied directly. An intercept is added automatically.

**BetweenOLS** collapses the data to group (entity) means and runs OLS on the reduced dataset:

$$
\bar{y}_{i\cdot} = \alpha + \bar{X}_{i\cdot}'\beta + \bar{u}_{i\cdot}
$$

where \(\bar{y}_{i\cdot} = T_i^{-1} \sum_t y_{it}\) and \(\bar{X}_{i\cdot} = T_i^{-1} \sum_t X_{it}\). The effective sample size is the number of entities \(n_{entities}\).

**FirstDifferenceOLS** removes entity fixed effects by taking first differences within each entity:

$$
\Delta y_{it} = \Delta X_{it}'\beta + \Delta u_{it}, \qquad \Delta y_{it} = y_{it} - y_{i,t-1}
$$

The intercept is eliminated by differencing. Data must be sorted by entity and time; entities with fewer than two observations are dropped.

**FamaMacBeth** implements the two-pass regression (Fama & MacBeth 1973):

1. **Step 1**: For each time period \(t\), run a cross-sectional OLS regression to obtain a coefficient vector \(\hat{\beta}_t\).
2. **Step 2**: Average the time-series of coefficients \(\bar{\beta} = T^{-1} \sum_t \hat{\beta}_t\) and compute standard errors from the time-series variation of \(\hat{\beta}_t\), optionally with a Newey-West HAC correction for serial correlation.

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

**PooledOLS** fits OLS on the raw (stacked) data:

$$
\hat{\beta} = (X^\top X)^{-1} X^\top y
$$

where \(X\) includes an automatically added intercept column.

**BetweenOLS** collapses to entity means then runs OLS:

$$
\hat{\beta} = (\bar{X}^\top \bar{X})^{-1} \bar{X}^\top \bar{y}
$$

where \(\bar{X}\) and \(\bar{y}\) are the entity-mean matrices of dimension \((n_{entities}, k)\).

**FirstDifferenceOLS** applies first differencing within each entity and runs OLS:

$$
\hat{\beta} = (\Delta X^\top \Delta X)^{-1} \Delta X^\top \Delta y
$$

where \(\Delta X\) and \(\Delta y\) are the first-differenced regressors and dependent variable (no intercept).

**FamaMacBeth** runs two passes:

1. For each period \(t\): \(\hat{\beta}_t = (X_t^\top X_t)^{-1} X_t^\top y_t\)
2. Average: \(\bar{\beta} = \frac{1}{T} \sum_{t=1}^{T} \hat{\beta}_t\)

Standard errors are computed from the time-series of \(\hat{\beta}_t\). With `cov_type='newey-west'`, the Newey-West HAC estimator with Bartlett kernel is applied to the \(\hat{\beta}_t\) series to correct for serial correlation.

## Covariance/Inference

The `cov_type` parameter selects the inference method:

- **`nonrobust`**: classical OLS covariance \(\hat{\sigma}^2 (X^\top X)^{-1}\). P-values from the \(t\)-distribution with \(df_{resid}\) degrees of freedom.
- **`robust`**: HC1 sandwich estimator (White 1980, with finite-sample \(n/(n-k)\) correction). P-values from the standard normal.
- **`clustered`**: cluster-robust sandwich estimator (Cameron & Miller 2015). P-values from the standard normal. For two-way clustering, pass a 2-column cluster array; the variance is computed via the Cameron, Gelbach & Miller (2011) method.
- **`hac`**: Newey-West HAC estimator (Newey & West 1987) with Bartlett kernel. P-values from the standard normal. Automatic bandwidth selection via the Newey-West (1994) rule: \(bw = \lfloor 4 (n/100)^{2/9} \rfloor\). Used with `PooledOLS` (via `time_index`) and `FamaMacBeth` (applied to the \(\hat{\beta}_t\) time-series).

Supported `cov_type` values by model:

| Model | nonrobust | robust | clustered | hac / newey-west |
|---|---|---|---|---|
| PanelOLS | yes | yes | yes | -- |
| RandomEffects | yes | -- | -- | -- |
| PooledOLS | yes | yes | yes | yes |
| BetweenOLS | yes | yes | yes | -- |
| FirstDifferenceOLS | yes | yes | -- | -- |
| FamaMacBeth | yes | -- | -- | yes (`newey-west`) |

`RandomEffects` uses nonrobust OLS inference on the quasi-demeaned data by default.

Outputs after `fit()`: `coef_`, `bse_`, `tvalues_`, `pvalues_`, `conf_int_`, `rsquared_within` (PanelOLS). `FamaMacBeth` additionally stores `betas_` (the T-by-k matrix of per-period coefficients) and `n_periods`.

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

### PooledOLS

| Parameter | Default | Description |
|---|---:|---|
| `cov_type` | `'nonrobust'` | Covariance type: `'nonrobust'`, `'robust'`, `'clustered'`, or `'hac'` |
| `alpha` | `0.05` | Significance level for confidence intervals |
| `bandwidth` | `None` | HAC bandwidth; `None` uses Newey-West (1994) rule |
| `kernel` | `'bartlett'` | HAC kernel function |
| `device` | `"auto"` | Computation device: `"cpu"`, `"cuda"`, or `"auto"` |

**fit()**: `fit(X, y, cluster=None, time_index=None)`. An intercept is added automatically. `cluster` is required for `cov_type='clustered'`. `time_index` is used for HAC estimation.

### BetweenOLS

| Parameter | Default | Description |
|---|---:|---|
| `cov_type` | `'nonrobust'` | Covariance type: `'nonrobust'`, `'robust'`, or `'clustered'` |
| `alpha` | `0.05` | Significance level for confidence intervals |
| `device` | `"auto"` | Computation device: `"cpu"`, `"cuda"`, or `"auto"` |

**fit()**: `fit(X, y, entity_ids)`. An intercept is added automatically. `entity_ids` is required.

### FirstDifferenceOLS

| Parameter | Default | Description |
|---|---:|---|
| `cov_type` | `'nonrobust'` | Covariance type: `'nonrobust'` or `'robust'` |
| `alpha` | `0.05` | Significance level for confidence intervals |
| `device` | `"auto"` | Computation device: `"cpu"`, `"cuda"`, or `"auto"` |

**fit()**: `fit(X, y, entity_ids, time_ids=None)`. No intercept is added (differencing removes it). `entity_ids` is required. `time_ids` is optional; if omitted, data is assumed sorted by time within each entity.

### FamaMacBeth

| Parameter | Default | Description |
|---|---:|---|
| `cov_type` | `'newey-west'` | Covariance type: `'nonrobust'` or `'newey-west'` |
| `bandwidth` | `None` | Newey-West bandwidth; `None` uses Newey-West (1994) rule |
| `alpha` | `0.05` | Significance level for confidence intervals |
| `min_obs_per_period` | `1` | Minimum observations per time period to include |
| `device` | `"auto"` | Computation device: `"cpu"`, `"cuda"`, or `"auto"` |

**fit()**: `fit(X, y, time_ids)`. An intercept is added automatically. `time_ids` is required.

## CPU+GPU Examples

```python
from statgpu.panel import (PanelOLS, RandomEffects, PooledOLS,
                            BetweenOLS, FirstDifferenceOLS, FamaMacBeth)
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
fe_torch = PanelOLS(entity_effects=True, cov_type='robust', device='torch')
fe_torch.fit(y_torch, X_torch, entity_ids=entity_ids)
print(f"Torch FE coef: {fe_torch.coef_}")

# --- PooledOLS ---

pooled = PooledOLS(cov_type='clustered', device='cpu')
pooled.fit(X, y, cluster=entity_ids)
print(f"Pooled coef: {pooled.coef_}, R-squared: {pooled.rsquared:.4f}")

# PooledOLS with HAC standard errors
pooled_hac = PooledOLS(cov_type='hac', device='cpu')
pooled_hac.fit(X, y, time_index=time_ids)
print(f"Pooled HAC coef: {pooled_hac.coef_}, SE: {pooled_hac.bse_}")

# --- BetweenOLS ---

between = BetweenOLS(cov_type='robust', device='cpu')
between.fit(X, y, entity_ids=entity_ids)
print(f"Between coef: {between.coef_}, nobs: {between.nobs}")

# --- FirstDifferenceOLS ---

fd = FirstDifferenceOLS(cov_type='robust', device='cpu')
fd.fit(X, y, entity_ids=entity_ids, time_ids=time_ids)
print(f"FD coef: {fd.coef_}, R-squared: {fd.rsquared:.4f}")

# --- FamaMacBeth ---

fm = FamaMacBeth(cov_type='newey-west', device='cpu')
fm.fit(X, y, time_ids=time_ids)
print(f"FM coef: {fm.coef_}, SE: {fm.bse_}")
print(f"FM periods: {fm.n_periods}, betas shape: {fm.betas_.shape}")
```

## Backend execution and metadata boundary

For array input, `FamaMacBeth` keeps cross-sectional regressions, coefficient paths,
Newey-West covariance, inference arrays, and prediction on NumPy, CuPy, or Torch.
Panel formula construction and categorical/time/cluster label factorization remain CPU
metadata operations; only compact integer codes are copied to the numerical backend.
Scalar t/normal CDF and quantile evaluations are also intentional CPU boundaries.

Formula-side arrays are aligned to Patsy's retained rows after missing-value deletion.
NumPy/Torch-CPU parity is tested for Fama–MacBeth HAC fit and prediction; physical CUDA
validation remains pending.

## strict/approx difference

There is no strict/approx mode for panel models. The `cov_type` parameter controls the inference method:

- `'nonrobust'`: classical OLS standard errors, assumes homoskedasticity and no within-cluster correlation. Uses the \(t\)-distribution for p-values.
- `'robust'`: HC1 heteroskedasticity-robust standard errors (White sandwich). Uses the normal distribution for p-values.
- `'clustered'`: cluster-robust standard errors allowing arbitrary within-cluster correlation. Uses the normal distribution for p-values. Supports one-way and two-way clustering.
- `'hac'` / `'newey-west'`: Newey-West HAC standard errors with Bartlett kernel, robust to heteroskedasticity and autocorrelation. Uses the normal distribution for p-values. Automatic bandwidth via the Newey-West (1994) rule: \(bw = \lfloor 4 (n/100)^{2/9} \rfloor\).

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
| `rsquared` | scalar | R-squared (PooledOLS, BetweenOLS, FirstDifferenceOLS) |
| `theta_` | scalar | GLS transformation weight (RandomEffects only) |
| `variance_components_` | dict | `{'sigma2_e': float, 'sigma2_a': float}` (RandomEffects only) |
| `betas_` | `(T, k)` | Time-series of per-period coefficients from Step 1 (FamaMacBeth only) |
| `n_periods` | int | Number of time periods used (FamaMacBeth only) |
| `nobs` | int | Number of observations |
| `df_resid` | int | Residual degrees of freedom |

### Methods

| Method | Returns | Description |
|---|---|---|
| `fit(y, X, entity_ids, ...)` | `self` | Fit the panel model. Requires `entity_ids` (1-D array of entity labels). Optional: `time_ids`, `cluster`. |
| `fit(X, y, ...)` | `self` | Fit PooledOLS/BetweenOLS/FirstDifferenceOLS/FamaMacBeth. Arguments differ by model (see Parameters above). |
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
Both `PanelOLS` and `RandomEffects` handle unbalanced panels. Entity and time identifiers define the structure; each entity can have a different number of observations \(T_i\). The GLS weight \(\theta_i\) in RandomEffects varies by entity to account for differing \(T_i\). `BetweenOLS` and `FirstDifferenceOLS` also handle unbalanced panels naturally: `BetweenOLS` computes entity means over whatever observations each entity has, and `FirstDifferenceOLS` drops single-observation entities.

**When should I use FamaMacBeth vs PanelOLS?**
`FamaMacBeth` is the standard approach in asset pricing and factor model research. It runs cross-sectional regressions per period and averages the coefficients, which is intuitive when the cross-section is the dimension of interest. `PanelOLS` with fixed effects is preferred when you want to control for unobserved heterogeneity. Use `FamaMacBeth` with `cov_type='newey-west'` to correct for serial correlation in the coefficient estimates.

**When is HAC inference appropriate?**
Use `cov_type='hac'` (PooledOLS) or `cov_type='newey-west'` (FamaMacBeth) when residuals exhibit autocorrelation -- for example, in time-series or panel data with persistent shocks. The Bartlett kernel downweights higher-order autocovariances, and the Newey-West (1994) bandwidth rule provides a data-driven lag length.

**What is the hac_covariance function?**
`hac_covariance` is a standalone function that computes the Newey-West HAC covariance matrix for OLS estimates. It is used internally by `PooledOLS` (when `cov_type='hac'`) and `FamaMacBeth` (when `cov_type='newey-west'`), but can also be called directly on any OLS design matrix and residuals.

## External Validation

Validated against `linearmodels.panel.PanelOLS`, `linearmodels.panel.RandomEffects`, `linearmodels.panel.PooledOLS`, `linearmodels.panel.BetweenOLS`, `linearmodels.panel.FirstDifferenceOLS`, and `linearmodels.panel.FamaMacBeth` (from the `linearmodels` package). Coefficient estimates, standard errors, and variance components match to relative error < 1e-12 on test datasets. The `hac_covariance` function is validated against `statsmodels` Newey-West standard errors. Consistency checks are maintained in `dev/tests/test_panel_p2.py`.

## References

- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press.
- Cameron, A. C., & Miller, D. L. (2015). A practitioner's guide to cluster-robust inference. *Journal of Human Resources*, 50(2), 317-372. [https://doi.org/10.3368/jhr.50.2.317](https://doi.org/10.3368/jhr.50.2.317)
- Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011). Robust inference with multiway clustering. *Journal of Business & Economic Statistics*, 29(3), 238-249. [https://doi.org/10.1198/jbes.2010.07136](https://doi.org/10.1198/jbes.2010.07136)
- Swamy, P. A. V. B., & Arora, S. S. (1972). The exact finite sample properties of the estimators of coefficients in the error components regression models. *Econometrica*, 40(2), 261-275. [https://doi.org/10.2307/1909125](https://doi.org/10.2307/1909125)
- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*, 48(4), 817-838. [https://doi.org/10.2307/1912934](https://doi.org/10.2307/1912934)
- Fama, E. F., & MacBeth, J. D. (1973). Risk, return, and equilibrium: Empirical tests. *Journal of Political Economy*, 81(3), 607-636. [https://doi.org/10.1086/260061](https://doi.org/10.1086/260061)
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703-708. [https://doi.org/10.2307/1913610](https://doi.org/10.2307/1913610)
- Newey, W. K., & West, K. D. (1994). Automatic lag selection in covariance matrix estimation. *Review of Economic Studies*, 61(4), 631-653. [https://doi.org/10.2307/2297912](https://doi.org/10.2307/2297912)
