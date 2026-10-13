# PooledOLS

> Language: English  
> Last updated: 2026-10-13<br>
> Switch: [Chinese](../../cn/panel/pooled-ols.md)

<a id="overview"></a>

## When one pooled relationship is appropriate

Use `PooledOLS` when the target is one common intercept and one set of slopes across all entity-period observations. It fits ordinary least squares on the stacked rows, so it is a useful baseline when a single pooled relationship is scientifically appropriate.

Pooling does not remove entity or time effects. For the usual interpretation of the slope, the composite error, including any unmodeled entity differences, must have conditional mean zero given the regressor history. If persistent entity differences are related to your regressors, consider [PanelOLS](panel-ols.md) with entity effects. Changing standard errors cannot fix that mean-model problem.

The model is $y_{it}=\alpha+x_{it}^{\top}\beta+u_{it}$: $i$ indexes entities, $t$ indexes periods, $y_{it}$ is the response, and $x_{it}$ is the predictor vector in that row of `X`. The intercept $\alpha$ and slopes $\beta$ are shared across all rows; $u_{it}$ includes any entity or time differences left outside the model. Fitting minimizes the sum of squared residuals $\sum_{i,t}(y_{it}-\alpha-x_{it}^{\top}\beta)^2$ over the stacked observations.

## Prepare the stacked observations

Stack your panel in long form, with one row per entity-period observation:

- `X`: shape `(n_observations, n_features)`, containing regressors without an intercept column. `PooledOLS` always adds the intercept itself.
- `y`: shape `(n_observations,)`, in the same row order as `X`.
- Optional `entity_ids`: one entity label per row. It enables panel-specific fit statistics, including within/between measures, and the Breusch-Pagan LM diagnostic without changing the coefficients. It does not automatically cluster standard errors; see [Choose standard errors for your panel](#choose-standard-errors-for-your-panel).
- Optional `time_index`: one period label per row. It is required for Driscoll-Kraay covariance and can define the chronological order for HAC. This argument is named `time_index`, not `time_ids`.

In the example, `X.shape == (60, 1)` represents 12 entities with 5 observations each. The data share a common intercept of 1 and slope of 2, with independent noise.

<a id="cpu-and-gpu-example"></a>

## Fit a pooled baseline

<a id="cpu-example"></a>

<!-- example: cpu-basic -->
```python
import numpy as np
from statgpu.panel import PooledOLS

rng = np.random.default_rng(7)
entity_ids = np.repeat(np.arange(12), 5)
time_ids = np.tile(np.arange(5), 12)
x = rng.normal(size=60)
X = x[:, None]
y = 1.0 + 2.0 * x + 0.2 * rng.normal(size=60)

model = PooledOLS(device="cpu").fit(X, y, entity_ids=entity_ids)
print("coef:", np.round(model.coef_, 3))
print("standard errors:", np.round(model.bse_, 3))
print("nobs:", model.nobs)
```
<!-- example-end: cpu-basic -->

`coef_` is ordered as intercept, then the slope for the single column of `X`. The estimated slope is close to the generating value 2; `bse_` gives the standard errors in the same order. `nobs=60` counts all stacked rows. This agreement in simulated data does not establish exogeneity for a real panel.

To run this example on a GPU, replace `device="cpu"` with `device="cuda"` (CuPy/CUDA) or `device="torch"` (Torch CUDA). An unavailable explicitly requested backend raises an error instead of switching to CPU.

<a id="outputs"></a>

## Interpret coefficients, uncertainty, and predictions

Read `coef_[0]` as the common intercept and the remaining coefficients in `X` column order. They summarize the pooled relationship across rows; they are not within-entity effects.

`bse_` is the uncertainty of each coefficient, in the same order as `coef_`; it is not the spread of individual predictions. `conf_int_` gives one lower/upper interval per coefficient (`alpha=0.05` means 95%). `tvalues_` is the coefficient-to-standard-error ratio: the default `nonrobust` fit uses a Student-$t$ reference with `df_resid`, while the other supported covariance choices use a normal reference for p-values and intervals. See [Panel covariance](covariance.md) before comparing inference across settings.

`rsquared` is the pooled fit's $R^2$. `fit_statistics_` contains the standardized fit statistics, including within/between $R^2$ when `entity_ids` was supplied; see [fit statistics](fit-statistics.md). `nobs` is the original row count and `df_resid` is the residual degrees of freedom. `summary()` returns a structured panel summary.

For new rows, `model.predict(X_new)` adds the fitted common intercept and applies the slopes. Supply the same feature columns and order as at fit time, without a constant column. The result is a NumPy array with one prediction per row; entity/time labels are not needed and no entity-specific effect is added. After formula fitting, you can instead pass a DataFrame containing the required regressor columns to `predict()`.

<a id="covariance-and-inference"></a>

## Choose standard errors for your panel

The default `cov_type="nonrobust"` uses classical standard errors. Use an HC option for heteroskedasticity, or `cov_type="clustered"` with `cluster=entity_ids` when the intended clusters are entities. Cluster labels may describe a different grouping if your error structure calls for it; `group_debias=True` optionally applies the small-number-of-clusters correction.

`cov_type` changes the standard-error calculation, not the OLS coefficient estimate. In addition to nonrobust and HC covariance, `PooledOLS` supports clustered covariance and two time-dependent choices:

- `cov_type="hac"` treats the observations as one ordered sequence and applies Bartlett/Newey-West HAC. If `time_index` is supplied, the rows are first sorted by its chronology; otherwise the input row order is used. Numeric and datetime labels use their natural order. An ordered pandas categorical uses the category order declared by the user. Plain string labels use lexical order; other non-categorical object labels follow their sorted comparable-value order and fail if they are not mutually comparable. Labels such as `t1, t2, t10` should therefore be supplied as an ordered categorical (or a numeric/datetime key) when lexical string order is not the intended chronology.
- `cov_type="driscoll-kraay"` groups observations by `time_index` and combines their contributions within each period before applying lag weights.

These two choices are therefore not interchangeable. HAC neither resets lags at entity boundaries nor aggregates repeated periods, so sorting a stacked panel does not make it general panel HAC. Check the [ordered-sequence applicability conditions](covariance.md#ordered-sequence-hac) before using it.

## Formula Example

Assume `df` contains `y`, `x1`, and `x2` columns.

```python
from statgpu.panel import PooledOLS

model = PooledOLS().fit(
    formula="y ~ x1 + x2",
    data=df,
)
```

`PooledOLS` always includes an intercept, so explicit no-intercept formulas are rejected.

<a id="numerical-and-strict-behavior"></a>

## Check these limits before using the results

Check that your specification includes only the intercept added by `PooledOLS`; do not add a second constant column to `X`. If the design is exactly rank deficient, fitted values remain available but the coefficients are not unique, so coefficient-level standard errors, tests, p-values, and confidence intervals are disabled. See [Panel covariance](covariance.md).

Required metadata must be present and aligned: clustered covariance needs `cluster`, Driscoll-Kraay needs `time_index`, and malformed cluster arrays raise an error. HAC rejects missing or non-finite time metadata rather than guessing the order.

A new `fit()` attempt invalidates previous fitted results. If refitting fails, partially computed results are also cleared; `predict()` and `summary()` then report an unfitted estimator. Explicit GPU requests fail if the requested backend is unavailable.

At extreme response scales, the intercept calculation protects representable small contributions from cancellation between large positive and negative responses. The legacy `rsquared` calculation also uses range-safe centering when direct subtraction of the response mean would overflow. These safeguards preserve the same least-squares target.

## FAQ

**Does supplying `entity_ids` change coefficients?**  No. It only enables panel-aware fit statistics and diagnostics.

**Is `hac` the same as Driscoll-Kraay?**  No. HAC treats observations as one ordered sequence; Driscoll-Kraay first aggregates observation contributions within each supplied time period.

## Statistical Model and Identification

The pooled linear model can be written as

$$
y_{it}=\alpha+x_{it}^{\top}\beta+u_{it}.
$$

A common sufficient condition for interpreting $\beta$ as the pooled conditional-mean slope is

$$
E(u_{it}\mid X_i)=0,
$$

where $X_i=(x_{i1},\ldots,x_{iT_i})$. The model therefore treats any unmodeled entity or time heterogeneity as part of the composite error $u_{it}$.

For example, if the underlying data satisfy

$$
y_{it}=\alpha+x_{it}^{\top}\beta+a_i+\varepsilon_{it},
$$

then pooled OLS does not remove $a_i$. To recover the same structural $\beta$, the combined error $a_i+\varepsilon_{it}$ must still be orthogonal to the regressors. If entity heterogeneity is correlated with the regressor history, pooled OLS generally does not identify the same slope as a fixed-effects estimator.

Covariance choices such as HC, clustering, HAC, or Driscoll-Kraay change how uncertainty is estimated; they do not repair a failure of this mean-model exogeneity condition.

## Estimator

With $Z=[\mathbf 1,X]$,

$$
\widehat\beta_{\mathrm{pooled}}
=\arg\min_\beta\|y-Z\beta\|_2^2
=(Z^\top Z)^+Z^\top y.
$$

Thus the coefficient estimate is the same OLS fit you would obtain from treating the stacked panel rows as one regression sample.

## Parameters

| Parameter | Default | Allowed / Constraint | Meaning |
|---|---:|---|---|
| `cov_type` | `"nonrobust"` | `nonrobust`, `robust`/`hc1`, `hc0`, `hc2`, `hc3`, `clustered`, `hac`, `driscoll-kraay`/`dk`/`kernel` | How coefficient standard errors are computed. |
| `alpha` | `0.05` | finite and strictly between 0 and 1 | Confidence-interval significance level; `0.05` gives 95% intervals. |
| `bandwidth` | `None` | `None` or a non-negative integer | HAC/DK lag or smoothing bandwidth. Legacy HAC uses at most $n-1$ lags; DK follows the observed-period rule in [Panel covariance](covariance.md). |
| `kernel` | `"bartlett"` | `hac` requires Bartlett; DK also accepts Parzen and QS aliases | Kernel used by HAC/DK covariance. |
| `device` | `"auto"` | `auto`, `cpu`, `cuda`, `torch` | Where numerical computation runs. |
| `n_jobs` | `None` | integer or `None` | Shared parallelism hint. |
| `group_debias` | `False` | boolean; clustered covariance only | Apply the small-number-of-clusters correction. |

### Fitting and other methods

```python
model.fit(X=None, y=None, cluster=None, time_index=None,
          formula=None, data=None, entity_ids=None)
```

Supply arrays or `formula` with `data`; `fit()` returns the fitted estimator. ID/time arrays have one label per observation, and `cluster` accepts the one- or two-way grouping described in [Panel covariance](covariance.md). `entity_ids` enables panel-aware statistics; `time_index` is required for DK and optional for HAC. Formula-side metadata must align with the retained observations; see [the formula example above](#formula-example).

- `predict(X)`: return pooled level predictions as a NumPy array.
- `summary()`: return the structured panel summary.
- `breusch_pagan_lm_test()`: return the random-effects LM diagnostic when `entity_ids` was supplied; check [applicability conditions](diagnostics.md).
- `get_params(deep=True)` / `set_params(**params)`: inspect or change constructor settings. `set_params()` clears fitted results; fit again after a change.

Fitted attributes: `coef_`, `bse_`, `tvalues_`, `pvalues_`, `conf_int_`, `rsquared`, `fit_statistics_`, `nobs`, `df_resid`, and `rank_` (the fitted design's numerical rank). Coefficient-inference arrays are unavailable for an exactly rank-deficient design.

The inherited `adjust_pvalues`, `combine_pvalues`, `bootstrap_statistic`, and `permutation_test` helpers have complete signatures in the [shared estimator API](../reference/estimator-api.md). They do not automatically select a panel-valid resampling scheme.

<a id="external-validation"></a>

## Comparing with Other Packages

To compare with `linearmodels` 7.0, use the same stacked sample and explicit intercept. For Driscoll-Kraay, align the time labels, kernel, bandwidth, and degrees-of-freedom convention; for clustered covariance, align cluster groups and `group_debias`. These choices affect covariance and standard errors even when the OLS coefficients agree. See [comparing covariance definitions](covariance.md#comparing-covariance-definitions).

## References

- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). The MIT Press.

Covariance-specific sources are collected in [Panel covariance](covariance.md).
