# BetweenOLS

> Language: English  
> Last updated: 2026-10-13<br>
> Switch: [Chinese](../../cn/panel/between-ols.md)

<a id="overview"></a>

## When to compare entity averages

`BetweenOLS` first averages each variable within entity, leaving one observation per entity, and then runs an OLS regression with an intercept. It is useful when the relationship of interest is driven by differences **between** entities rather than changes within the same entity over time.

Because every retained entity contributes one averaged observation, the effective sample size is the number of retained entities, not the number of original panel rows.

Choose it to compare long-run averages across firms, people, or regions. If you instead want to remove time-invariant differences between entities, consider [PanelOLS](panel-ols.md) or [FirstDifferenceOLS](first-difference-ols.md). Averaging does not remove an entity effect: interpreting the slope as a common structural effect requires the combined entity effect and averaged error to be orthogonal to the entity means of the predictors.

## Prepare the panel rows

Use long-form panel data: each row is one entity-period observation. Pass `X` with shape `(n, p)`, `y` with shape `(n,)`, and `entity_ids` of length `n`, all in the same row order. A one-dimensional `X` is accepted as a single predictor. IDs identify the entity to which each row belongs; they are not predictor columns. Supply finite numeric values.

Pass the original observations, not precomputed means. The estimator averages within each entity and adds an intercept, so do not add a constant column to `X`. Different entities may have different observation counts; each retained entity still contributes one equally weighted mean. `time_ids` is not needed for this fit.

<a id="cpu-and-gpu-example"></a>

## Fit a first model on CPU

Each row is an entity-period observation, with 12 entities and 5 periods per entity. Here the mean of `x` varies across entities and the outcome has no omitted entity effect.

<a id="cpu-example"></a>

<!-- example: cpu-basic -->
```python
import numpy as np
from statgpu.panel import BetweenOLS

rng = np.random.default_rng(7)
entity_ids = np.repeat(np.arange(12), 5)
time_ids = np.tile(np.arange(5), 12)
x = np.repeat(np.linspace(-2, 2, 12), 5) + rng.normal(size=60)
X = x[:, None]
y = 1.0 + 2.0 * x + 0.2 * rng.normal(size=60)

model = BetweenOLS(device="cpu").fit(X, y, entity_ids=entity_ids)
print("coef:", np.round(model.coef_, 3))
print("nobs, df_resid:", model.nobs, model.df_resid)
```
<!-- example-end: cpu-basic -->

<a id="outputs"></a>

## Read the between fit and make predictions

`coef_` gives the intercept and the slope on the entity mean of `x`, close to 1 and 2. Although the input has 60 rows, `nobs=12` and `df_resid=10` describe the 12 entity means and two fitted coefficients. Averaging does not remove an omitted entity effect if real data contain one.

Public results include `coef_`, `bse_`, `tvalues_`, `pvalues_`, `conf_int_`, `rsquared`, `fit_statistics_`, `nobs`, and `df_resid`. Here `nobs` is the number of entity means used in the final regression.

`predict(X_new)` evaluates the fitted intercept plus `X_new @ coef_[1:]` row by row and returns a NumPy array. It does not average the new data by entity. To predict entity means, pass a predictor-mean row for each entity; predictions on individual level rows use the between slope and do not estimate entity-specific effects.

`bse_` reports coefficient standard errors and `conf_int_` has one lower/upper pair per coefficient. `nonrobust` uses Student-t inference with `df_resid`; the HC choices use a normal reference, so `tvalues_` then contains z-statistics. Use `print(model.summary())` to view estimates and uncertainty together. Public coefficient/inference arrays are NumPy arrays. For standardized within/between/overall measures, consult [fit statistics](fit-statistics.md); they need not equal the transformed-regression `rsquared`.

<a id="covariance-and-inference"></a>

## Choose uncertainty for the entity means

Standard errors are computed from that entity-mean regression. `cov_type="nonrobust"` uses the usual homoskedastic OLS covariance; `robust`/`hc1` and HC0/HC2/HC3 provide heteroskedasticity-consistent alternatives. The shared formulas are given in [Panel covariance](covariance.md).

These covariance choices change inference for the between regression; they do not replace the orthogonality condition needed to interpret its slope as the structural $\beta$ from the underlying panel model.

<a id="numerical-and-strict-behavior"></a>

## Check the sample and numerical limits

Extreme response scales and cancellation receive numerical safeguards that preserve representable intercept contributions without changing the entity-mean regression target. The legacy `rsquared` calculation also uses range-safe centering when direct subtraction of the entity-mean response average would overflow. These calculations remain subject to float64 precision.

If the entity-level regressors are exactly collinear, statgpu can still compute fitted values using a least-squares solution, but the coefficient vector is not unique. For that fit, coefficient-level standard errors, tests, p-values, and confidence intervals are disabled rather than being computed from an arbitrary coefficient representation.

Invalid covariance choices raise an error. Likewise, an explicitly requested GPU backend must be available; statgpu does not silently run the model on CPU instead.

The transformed regression needs positive residual degrees of freedom: more entity means than design rank.

<a id="formula-example"></a>

## Fit from dataframe columns

Assume `df` contains `y`, `x1`, `x2`, and `entity` columns.

```python
from statgpu.panel import BetweenOLS

model = BetweenOLS().fit(
    formula="y ~ x1 + x2",
    data=df,
    entity_ids=df["entity"],
)
```

`BetweenOLS` always includes an intercept; explicit no-intercept formulas are rejected.

## Run the same model on a GPU

To run this example on a GPU, replace `device="cpu"` with `device="cuda"` (CuPy/CUDA) or `device="torch"` (Torch CUDA). An unavailable explicitly requested backend raises an error instead of switching to CPU.

<a id="statistical-model-and-identification"></a>

## Why entity averaging needs an extra assumption

The algebra of the between transformation can be seen by starting from the one-way entity-effect equation

$$
y_{it}=\alpha+x_{it}^{\top}\beta+a_i+\varepsilon_{it}.
$$

Averaging over time within entity gives

$$
\bar y_i=\alpha+\bar x_i^{\top}\beta+a_i+\bar\varepsilon_i.
$$

If $a_i$ is treated as an unrestricted fixed parameter, as in the classical fixed-effects formulation, averaging does **not** eliminate it. Therefore the fixed-parameter model alone does not justify interpreting the between-regression slope as the same structural $\beta$.

To give the between slope a population interpretation across entities, one typically adds a superpopulation or cross-sectional moment assumption. For example, if $(a_i,\bar x_i,\bar\varepsilon_i)$ are viewed as varying randomly across entities, a common sufficient condition is

$$
E\!\left(a_i+\bar\varepsilon_i\mid \bar x_i\right)=0.
$$

A weaker route, sufficient for identifying the corresponding linear-projection slope, is to impose the relevant orthogonality moment directly, for example $E[\bar x_i(a_i+\bar\varepsilon_i)]=0$ after handling the intercept. Under such an additional restriction, the between regression can recover the same $\beta$ in the underlying panel equation.

If persistent entity heterogeneity is related to the entity's average regressors, `BetweenOLS` still estimates the cross-sectional linear projection of $\bar y_i$ on $\bar x_i$, but that projection generally differs from the fixed-effects or first-difference slope.

<a id="estimator"></a>

## The entity-mean least-squares problem

For entity $i$,

$$
\bar y_i=T_i^{-1}\sum_t y_{it},
\qquad
\bar x_i=T_i^{-1}\sum_t x_{it},
$$

and with $\bar Z_i=(1,\bar x_i^\top)^\top$,

$$
\widehat\beta_B
=\arg\min_\beta\sum_i(\bar y_i-\bar Z_i^\top\beta)^2
=(\bar Z^\top\bar Z)^+\bar Z^\top\bar y.
$$

In other words, the original panel is reduced to an ordinary cross-sectional regression of entity means.

<a id="parameters"></a>

## Complete API reference

| Parameter | Default | Allowed / Constraint | Meaning |
|---|---:|---|---|
| `cov_type` | `"nonrobust"` | `nonrobust`, `robust`/`hc1`, `hc0`, `hc2`, `hc3` | How coefficient standard errors are computed after entity averaging. |
| `alpha` | `0.05` | finite and strictly between 0 and 1 | Confidence-interval significance level; `0.05` gives 95% intervals. |
| `device` | `"auto"` | `auto`, `cpu`, `cuda`, `torch` | Where numerical computation runs. |
| `n_jobs` | `None` | integer or `None` | Shared parallelism hint. |

### Fit inputs

`fit(X=None, y=None, entity_ids=None, time_ids=None, formula=None, data=None)` returns the fitted estimator.

| Argument | Meaning |
|---|---|
| `X`, `y` | Array interface: predictors of shape `(n, p)` and response of shape `(n,)`; a one-dimensional `X` is treated as one predictor. Use finite numeric values. |
| `entity_ids` | Required entity labels aligned with the rows of `X` and `y`. |
| `time_ids` | Accepted for API compatibility; not used in the between transformation. |
| `formula`, `data` | Alternative formula string and dataframe interface. Requires the optional `pandas`/`patsy` dependencies. Formula-retained rows are also used to align the supplied ID arrays. |

Supply either arrays or `formula` with `data`. The [dataframe example](#formula-example) shows the model-specific intercept convention.

### Other public methods

- `predict(X)`: return one prediction per supplied row, with the interpretation described above; the feature count and column order must match training.
- `summary()`: return a printable coefficient and fit summary; use `print(model.summary())`.
- `get_params(deep=True)` and `set_params(**params)`: inspect or update constructor settings; refit after changing settings.
- The inherited `adjust_pvalues`, `combine_pvalues`, `bootstrap_statistic`, and `permutation_test` helpers are documented with complete signatures in the [shared estimator API](../reference/estimator-api.md). They do not automatically select a panel-valid resampling scheme.

<a id="faq"></a>

## Questions about entity averages

**Why is `nobs` smaller than the original row count?**  `nobs` is the number of entity-mean observations used in the between regression.

**Are entities with more time observations automatically weighted more heavily?**  No. Each retained entity contributes one mean observation to the final OLS regression.

<a id="external-validation"></a>

## Comparing with Other Packages

To compare with an OLS package such as statsmodels, first construct the identical entity-mean sample and include the same intercept. Each entity contributes one observation to this regression. Align the HC0/HC2/HC3 choice and compute covariance and standard errors from this transformed sample, rather than the original panel rows. See [comparing covariance definitions](covariance.md#comparing-covariance-definitions).

## References

- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). The MIT Press.
