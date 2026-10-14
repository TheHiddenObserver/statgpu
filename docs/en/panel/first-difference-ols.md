# FirstDifferenceOLS

> Language: English  
> Last updated: 2026-10-13<br>
> Switch: [Chinese](../../cn/panel/first-difference-ols.md)

<a id="overview"></a>

## When changes answer your question

`FirstDifferenceOLS` removes a time-invariant entity effect by subtracting each entity's previous **observed** value from its current observed value. It then regresses the differenced outcome on the differenced predictors without an intercept.

This estimator can be applied to the same one-way fixed-parameter entity-effect model as a fixed-effects estimator; the difference is the transformation used to eliminate the entity effect.

Choose it when changes within the same entity answer your question and you want to remove a time-invariant entity effect. It cannot estimate effects of time-invariant predictors. As with [fixed effects](panel-ols.md), a common sufficient assumption is strict exogeneity: errors have conditional mean zero given the full regressor history and the entity effect. Differencing alone does not establish that assumption.

Start from $y_{it}=\alpha+x_{it}^{\top}\beta+a_i+\varepsilon_{it}$. Here $i$ indexes entities, $t$ indexes periods, $y_{it}$ is the response, $x_{it}$ is the predictor vector, $\alpha$ is a common intercept, $\beta$ is the common slope vector, $a_i$ is a time-invariant entity effect, and $\varepsilon_{it}$ is the remaining error. Differencing within an entity gives $\Delta y_{it}=\Delta x_{it}^{\top}\beta+\Delta\varepsilon_{it}$, removing both $\alpha$ and $a_i$.

Here $\Delta$ means the current value minus the previous observed value for the same entity, without dividing by the calendar gap. Fitting minimizes squared residuals in this differenced equation, without an intercept; the original levels are the inputs to `.fit()`.

## Prepare levels and set the time order

Use long-form panel data: each row is one entity-period observation. Pass `X` with shape `(n, p)`, `y` with shape `(n,)`, and `entity_ids` of length `n`, all in the same row order. A one-dimensional `X` is accepted as a single predictor. IDs identify the entity to which each row belongs; they are not predictor columns. Supply finite numeric values.

Pass the original levels without a constant column; `.fit()` performs the differencing. Supply `time_ids` of length `n` to establish chronology. Without it, each entity’s rows are differenced in their original input order. An entity needs at least two rows to contribute a difference.

Supplying `time_ids` lets statgpu put each entity's rows in chronological order; each `(entity_id, time_id)` pair must then be unique. Numeric and datetime time labels use natural order and ordered categoricals use their declared category order. Plain string labels use lexical order; other non-categorical object labels follow their sorted comparable-value order and fail if they are not mutually comparable. Use an ordered categorical or numeric/datetime key when lexical string order is not the intended chronology.

<a id="cpu-and-gpu-example"></a>

## Fit a first model on CPU

Each row is one entity in one period: 12 entities with 5 chronologically labeled observations each. Differencing removes both the common intercept and the time-invariant entity effect.

<a id="cpu-example"></a>

<!-- example: cpu-basic -->
```python
import numpy as np
from statgpu.panel import FirstDifferenceOLS

rng = np.random.default_rng(7)
entity_ids = np.repeat(np.arange(12), 5)
time_ids = np.tile(np.arange(5), 12)
x = np.repeat(np.linspace(-2, 2, 12), 5) + rng.normal(size=60)
X = x[:, None]
entity_effect = np.repeat(np.linspace(-1, 1, 12), 5)
y = 1.0 + 2.0 * x + entity_effect + 0.2 * rng.normal(size=60)

model = FirstDifferenceOLS(device="cpu").fit(
    X, y, entity_ids=entity_ids, time_ids=time_ids
)
print("slope:", np.round(model.coef_, 3))
print("nobs, df_resid:", model.nobs, model.df_resid)
```
<!-- example-end: cpu-basic -->

<a id="outputs"></a>

## Read the differenced fit and predict changes

`coef_` contains only the slope, close to the generating value 2. Each entity contributes four adjacent-observation differences, so `nobs=48` and `df_resid=47`. Even independent level errors become serially correlated after differencing; the available HC choices address heteroskedasticity, not general serial correlation.

Public results include `coef_`, `bse_`, `tvalues_`, `pvalues_`, `conf_int_`, `rsquared`, `fit_statistics_`, `nobs`, and `df_resid`. `nobs` refers to the differenced observations used in the final regression, so it is smaller than the original panel row count.

`predict(X_new)` returns the linear product `X_new @ coef_` as a NumPy array. It does not difference new inputs or restore the entity effect. To predict outcome changes, first construct the desired predictor differences yourself and pass those rows. Calling it on levels gives only the slope component, not a full level prediction. For a formula with transformations or interactions, difference the resulting design columns and pass that numeric matrix; differencing raw dataframe columns before rebuilding the formula can describe a different model.

`bse_` reports coefficient standard errors and `conf_int_` has one lower/upper pair per coefficient. `nonrobust` uses Student-t inference with `df_resid`; the HC choices use a normal reference, so `tvalues_` then contains z-statistics. Use `print(model.summary())` to view estimates and uncertainty together. Public coefficient/inference arrays are NumPy arrays. For standardized within/between/overall measures, consult [fit statistics](fit-statistics.md); they need not equal the transformed-regression `rsquared`.

<a id="covariance-and-inference"></a>

## Choose uncertainty for the differenced errors

Standard errors are computed from the differenced regression $(\Delta y,\Delta X)$. Supported choices are nonrobust and HC0/HC1/HC2/HC3; see [Panel covariance](covariance.md).

Changing the covariance estimator changes uncertainty for the differenced regression; it does not replace the exogeneity condition needed for the usual first-difference interpretation of $\beta$.

There is no cluster or HAC option on this estimator. With more than two periods, adjacent differences share level errors; even independent level errors produce serially correlated differenced errors. Select a model and covariance strategy appropriate to that dependence rather than treating an HC choice as a serial-correlation correction.

<a id="numerical-and-strict-behavior"></a>

## Check the sample and numerical limits

When `time_ids` is supplied, duplicate observations for the same entity and time raise an error because the chronological difference would be ambiguous. Calendar gaps are left as gaps: statgpu differences adjacent observed rows and does not insert missing periods or rescale by elapsed time.

If finite-precision arithmetic cannot reliably resolve a nonzero coefficient in the differenced regression, `.fit()` raises `FloatingPointError` rather than returning an unreliable coefficient. This is distinct from exact collinearity. If the differenced predictors are exactly collinear, fitted values can still be computed but the coefficient vector is not unique, so coefficient-level standard errors, tests, p-values, and confidence intervals are disabled.

Legacy `rsquared` uses range-safe centering to avoid overflow when subtracting the mean of an extreme differenced response. This preserves the same scale-invariant $R^2$ definition. Invalid covariance choices or unavailable explicitly requested GPU backends raise clear errors.

The differenced regression needs positive residual degrees of freedom: more retained differences than design rank.

<a id="formula-example"></a>

## Fit from dataframe columns

Assume `df` contains `y`, `x1`, `x2`, `entity`, and `time` columns.

```python
from statgpu.panel import FirstDifferenceOLS

model = FirstDifferenceOLS().fit(
    formula="y ~ x1 + x2 - 1",
    data=df,
    entity_ids=df["entity"],
    time_ids=df["time"],
)
```

The formula removes the intercept because `FirstDifferenceOLS` does not estimate one after differencing.

## Run the same model on a GPU

To run this example on a GPU, replace `device="cpu"` with `device="cuda"` (CuPy/CUDA) or `device="torch"` (Torch CUDA). An unavailable explicitly requested backend raises an error instead of switching to CPU.

<a id="statistical-model-and-identification"></a>

## Why differencing removes the entity effect

Start from the one-way fixed-parameter panel model

$$
y_{it}=x_{it}^{\top}\beta+a_i+\varepsilon_{it},
$$

where $a_i$ is a fixed but unknown time-invariant entity effect. As with the fixed-effects model, $a_i$ need not be orthogonal to the regressor history. A common sufficient exogeneity condition for a static model is

$$
E\!\left(\varepsilon_{it}\mid X_i,a_i\right)=0,
\qquad
X_i=(x_{i1},\ldots,x_{iT_i}).
$$

Taking first differences removes $a_i$ exactly:

$$
\Delta y_{it}=\Delta x_{it}^{\top}\beta+\Delta\varepsilon_{it}.
$$

The slope is therefore identified from within-entity changes. A regressor that is constant over time within an entity differences to zero and cannot identify a slope in this specification. A common intercept in the level model is also removed by differencing, which is why the final regression has no intercept.

<a id="estimator"></a>

## The adjacent-observation least-squares problem

For consecutive observed periods within entity $i$,

$$
\Delta y_{it}=y_{it}-y_{i,t^-},
\qquad
\Delta x_{it}=x_{it}-x_{i,t^-},
$$

and

$$
\widehat\beta_{\mathrm{FD}}
=\arg\min_\beta\|\Delta y-\Delta X\beta\|_2^2
=(\Delta X^\top\Delta X)^+\Delta X^\top\Delta y.
$$

Here $t^-$ means the previous **observed** time for that entity. Missing calendar periods are not created, and the difference is not divided by the length of a calendar gap.

<a id="parameters"></a>

## Complete API reference

| Parameter | Default | Allowed / Constraint | Meaning |
|---|---:|---|---|
| `cov_type` | `"nonrobust"` | `nonrobust`, `robust`/`hc1`, `hc0`, `hc2`, `hc3` | How standard errors are computed after differencing. |
| `alpha` | `0.05` | finite and strictly between 0 and 1 | Confidence-interval significance level; `0.05` gives 95% intervals. |
| `device` | `"auto"` | `auto`, `cpu`, `cuda`, `torch` | Where numerical computation runs. |
| `n_jobs` | `None` | integer or `None` | Shared parallelism hint. |

### Fit inputs

`fit(X=None, y=None, entity_ids=None, time_ids=None, formula=None, data=None)` returns the fitted estimator.

| Argument | Meaning |
|---|---|
| `X`, `y` | Array interface: predictors of shape `(n, p)` and response of shape `(n,)`; a one-dimensional `X` is treated as one predictor. Use finite numeric values. |
| `entity_ids` | Required entity labels aligned with the rows of `X` and `y`. |
| `time_ids` | Optional chronological labels; omitted labels leave the within-entity input order unchanged. |
| `formula`, `data` | Alternative formula string and dataframe interface. Requires the optional `pandas`/`patsy` dependencies. Formula-retained rows are also used to align the supplied ID arrays. |

Supply either arrays or `formula` with `data`. The [dataframe example](#formula-example) shows the model-specific intercept convention.

### Other public methods

- `predict(X)`: return one prediction per supplied row, with the interpretation described above; the feature count and column order must match training.
- `summary()`: return a printable coefficient and fit summary; use `print(model.summary())`.
- `get_params(deep=True)` and `set_params(**params)`: inspect or update constructor settings; refit after changing settings.
- The inherited `adjust_pvalues`, `combine_pvalues`, `bootstrap_statistic`, and `permutation_test` helpers are documented with complete signatures in the [shared estimator API](../reference/estimator-api.md). They do not automatically select a panel-valid resampling scheme.

<a id="faq"></a>

## Questions about gaps and intercepts

**Does a two-period calendar gap create a two-step difference?**  No. The method subtracts consecutive observed rows after chronological ordering, regardless of the calendar gap length.

**Is an intercept estimated after differencing?**  No.

<a id="external-validation"></a>

## Comparing with Other Packages

To compare with an OLS package such as statsmodels, construct the identical differenced sample after applying the same within-entity chronology. Fit without an intercept and align the HC0/HC2/HC3 choice. Calendar gaps remain gaps in both samples; do not replace adjacent-observation differences with differences divided by elapsed time. See [comparing covariance definitions](covariance.md#comparing-covariance-definitions).

## References

- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). The MIT Press.
