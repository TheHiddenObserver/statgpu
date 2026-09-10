# Linear regression inference and complete API

> Language: English
> Last updated: 2026-09-03
> Switch: [简体中文](../../cn/models/linear-regression-inference.md)

This is the advanced companion to the [beginner-oriented LinearRegression guide](linear-regression.md). Use the beginner page for model intuition and a first fit; use this page when you need an exact covariance convention, every public input and output, or backend implementation boundaries.

## Notation

Let $Z$ be the fitted design matrix, including the intercept column when present. Write $n$ for the number of observations, $r=\operatorname{rank}(Z)$, $k$ for the number of design columns, $\hat e_i$ for residual $i$, and

$$
B=(Z^\top Z)^{-1},
\qquad
h_{ii}=z_i^\top Bz_i.
$$

For a rank-deficient design, statgpu uses a least-squares or pseudoinverse path where required, and the residual degrees of freedom are $n-r$. The formulas below show the ordinary full-rank form.

With `sample_weight=w`, fitting is weighted least squares. Read $Z$ and $\hat e$ below as the transformed quantities $\widetilde Z=W^{1/2}Z$ and $\widetilde e=W^{1/2}e$.

## Covariance estimators

### `nonrobust`: classical homoskedastic covariance

$$
\hat\sigma^2=\frac{\hat e^\top\hat e}{n-r},
\qquad
\widehat{\operatorname{Var}}(\hat\beta)=\hat\sigma^2B.
$$

This is efficient under conditionally homoskedastic, uncorrelated errors. Coefficient tests use a Student $t$ reference distribution with $n-r$ degrees of freedom.

### `hc0`: White sandwich covariance

$$
\widehat{\operatorname{Var}}_{\mathrm{HC0}}(\hat\beta)
=B\left(\sum_{i=1}^{n}\hat e_i^2z_iz_i^\top\right)B.
$$

HC0 permits unknown heteroskedasticity but has no finite-sample degrees-of-freedom correction.

### `hc1`: degrees-of-freedom-corrected HC0

$$
\widehat{\operatorname{Var}}_{\mathrm{HC1}}(\hat\beta)
=\frac{n}{n-r}\widehat{\operatorname{Var}}_{\mathrm{HC0}}(\hat\beta).
$$

HC1 is a common general-purpose robust choice and is close to the convention used by many econometric packages.

### `hc2`: leverage-adjusted covariance

$$
\widehat{\operatorname{Var}}_{\mathrm{HC2}}(\hat\beta)
=B\left(\sum_{i=1}^{n}\frac{\hat e_i^2}{1-h_{ii}}z_iz_i^\top\right)B.
$$

HC2 increases the contribution of observations with high leverage.

### `hc3`: stronger leverage adjustment

$$
\widehat{\operatorname{Var}}_{\mathrm{HC3}}(\hat\beta)
=B\left(\sum_{i=1}^{n}\frac{\hat e_i^2}{(1-h_{ii})^2}z_iz_i^\top\right)B.
$$

HC3 is often a conservative default for smaller samples or influential designs. statgpu clips computed leverage below 1 to prevent division by zero; this numerical guard does not make a nearly singular design statistically reliable.

### `hac`: Bartlett/Newey–West covariance

Define the per-observation score $s_t=z_t\hat e_t$ and lag products

$$
\Gamma_\ell=\sum_{t=\ell+1}^{n}s_ts_{t-\ell}^\top.
$$

For maximum lag $L$, statgpu uses Bartlett weights $w_\ell=1-\ell/(L+1)$:

$$
\Omega
=\Gamma_0+\sum_{\ell=1}^{L}w_\ell
\left(\Gamma_\ell+\Gamma_\ell^\top\right),
\qquad
\widehat{\operatorname{Var}}_{\mathrm{HAC}}(\hat\beta)=B\Omega B.
$$

If `hac_maxlags=None`,

$$
L=\left\lfloor4\left(\frac{n}{100}\right)^{2/9}\right\rfloor,
$$

clipped to $0\le L\le n-1$. HAC uses row order, so sort observations into their meaningful time order before fitting. It does not infer a time variable from the DataFrame.

## From covariance to the reported table

For coefficient $j$,

$$
\operatorname{se}(\hat\beta_j)
=\sqrt{\widehat{\operatorname{Var}}(\hat\beta)_{jj}},
\qquad
q_j=\frac{\hat\beta_j}{\operatorname{se}(\hat\beta_j)}.
$$

`nonrobust` uses a $t_{n-r}$ calibration. HC and HAC p-values and intervals use a standard normal calibration. For backward compatibility, the stored statistic is currently named `_tvalues` and `summary()` labels its column `t` even when robust inference uses normal calibration; treat it as a generic coefficient statistic outside `nonrobust`.

## Which covariance should I choose?

| Error structure | Starting choice | Important limit |
|---|---|---|
| independent and plausibly constant variance | `nonrobust` | invalid under heteroskedasticity |
| independent, large sample, possible heteroskedasticity | `hc1` | does not handle serial or cluster dependence |
| smaller sample or notable leverage | `hc3` | can become large around extreme leverage |
| ordered observations with short-range serial dependence | `hac` | depends on correct row order and lag choice |

Changing `cov_type` changes uncertainty estimates, not `coef_`. It cannot repair a nonlinear mean, omitted confounders, dependence across clusters, or a misspecified design.

## Complete API reference

### Constructor

```python
LinearRegression(
    fit_intercept=True,
    device="auto",
    n_jobs=None,
    compute_inference=True,
    gpu_memory_cleanup=False,
    cov_type="nonrobust",
    hac_maxlags=None,
)
```

| Parameter | Accepted values and effect |
|---|---|
| `fit_intercept` | Boolean. Adds an intercept unless Formula syntax explicitly removes it |
| `device` | `"auto"`, `"cpu"`, `"cuda"` (CuPy), `"torch"` (Torch CUDA), or a `Device` value |
| `n_jobs` | Optional estimator-level parallelism hint; not every linear algebra backend consumes it |
| `compute_inference` | Computes covariance, tests, intervals, and summary inputs when `True` |
| `gpu_memory_cleanup` | Releases backend cache blocks after public GPU work when `True`; can reduce repeated-fit throughput |
| `cov_type` | `"nonrobust"`, `"hc0"`, `"hc1"`, `"hc2"`, `"hc3"`, or `"hac"` |
| `hac_maxlags` | Non-negative integer or `None` for the sample-size rule above |

### `fit`

```python
fit(X=None, y=None, sample_weight=None, formula=None, data=None)
```

Pass either `X` plus `y`, or `formula` plus a pandas `data` frame. `sample_weight` must be one-dimensional, finite, non-negative, and have positive total weight. Formula evaluation drops rows according to Patsy; weights are aligned to the retained row positions.

### Fitted outputs

| Name | Meaning |
|---|---|
| `coef_` | Feature coefficients; shape `(p,)` or `(n_targets, p)` |
| `intercept_` | Scalar or one value per target |
| `rank_` | Numerical rank of the fitted design |
| `rsquared`, `rsquared_adj` | $R^2$ and adjusted $R^2$ |
| `fvalue`, `f_pvalue` | Overall regression F statistic and p-value |
| `llf`, `aic`, `bic` | Gaussian log-likelihood and information criteria; multi-output information criteria are not reported |
| `_bse` | Coefficient standard errors, intercept first when present |
| `_tvalues` | Stored coefficient statistic; see calibration note above |
| `_pvalues` | Two-sided coefficient p-values |
| `_conf_int` | 95% coefficient intervals with shape `(k, 2)` |
| `_inference_result` | Structured inference result with parameters, statistic metadata, feature names, and DataFrame conversion |

The underscore-prefixed arrays are part of the current compatibility surface but remain underscore-prefixed. Prefer `_inference_result` when writing reusable reporting code because it records the covariance and reference-distribution metadata.

### Methods

| Method | Contract |
|---|---|
| `predict(X_new)` | Returns fitted conditional means; a DataFrame reuses stored Formula encoding |
| `score(X, y)` | Returns $R^2$; multi-output scores are averaged |
| `summary()` | Prints a single-output coefficient and fit-statistics table; requires inference |
| `get_params()` / `set_params()` | sklearn-style estimator configuration inherited from the base estimator |

## Backend implementation notes

- CPU uses NumPy least squares.
- CuPy and Torch form the same design on the requested CUDA backend, use a Cholesky solve when possible, and fall back to backend least squares only for a genuine rank failure.
- Classical and robust covariance calculations run on the selected numerical backend. User-facing coefficient and inference metadata are transferred to NumPy at the reporting boundary.
- Large CPU HAC workloads may benchmark a small score-matrix probe and use a cached mixed-precision accumulation path when it is measurably faster.
- Explicit `"cuda"` or `"torch"` requests fail if that backend is unavailable; they do not silently execute on CPU.

See [CPU/GPU acceleration internals](../guides/acceleration-internals.md) for lifecycle, transfer, dtype, synchronization, and reproducibility details.

## Validation and source map

- Estimator: `statgpu/linear_model/wrappers/_linear.py`
- Shared inference results: `statgpu/inference/_results.py`
- Cross-framework consistency: `dev/tests/test_external_consistency.py`
- Formula and inference contracts: `dev/tests/test_gaussian_inference_formula_cleanup_contract.py`

These paths describe the current implementation. Private helper names are not public API.
