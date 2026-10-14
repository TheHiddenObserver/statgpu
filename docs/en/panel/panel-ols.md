# PanelOLS

> Language: English  
> Last updated: 2026-10-13<br>
> Switch: [Chinese](../../cn/panel/panel-ols.md)

<a id="overview"></a>

## When to use fixed effects

Use `PanelOLS` when you want to estimate how changes in a regressor relate to changes in an outcome after accounting for persistent differences between entities, shocks shared by a time period, or both. For example, entity effects compare a firm's changing inputs and outcomes after removing its stable baseline.

Entity effects remove entity-specific levels, so a regressor that never changes within an entity cannot have a separately identified slope. Time effects remove period-specific levels shared across entities; two-way effects account for both kinds of differences.

Fixed effects may be correlated with the regressor history. They do not remove time-varying confounding: the usual static-model interpretation still requires the idiosyncratic error to have conditional mean zero given that history and the included effects. The slope needs variation that survives removal of all selected effects.

Write the model as $y_{it}=x_{it}^{\top}\beta+a_i+\gamma_t+\varepsilon_{it}$, omitting an effect when it is not selected. Here $i$ indexes entities, $t$ indexes periods, $y_{it}$ is the response, $x_{it}$ is the row's predictor vector, and $\beta$ is the common slope vector. The entity level $a_i$ and period level $\gamma_t$ are fixed but unknown; $\varepsilon_{it}$ is the remaining observation-level error.

For the entity-only example below, subtracting each entity's observed-period means gives $y_{it}-\bar y_i=(x_{it}-\bar x_i)^{\top}\beta+(\varepsilon_{it}-\bar\varepsilon_i)$. Least squares on these demeaned rows estimates the slopes; the entity level, including any common intercept, disappears. Bars denote averages over that entity's observed periods.

## Prepare your panel data

Use long-form data: one row per entity-period observation, not one column per period.

- `X` has shape `(n_observations, n_features)` and `y` has shape `(n_observations,)`. Keep rows aligned across all arrays.
- `entity_ids` identifies which entity produced each row; `time_ids` identifies its period. Each supplied ID array has one label per row. Labels are grouping metadata, not extra regressors.
- Set `entity_effects=True` and provide `entity_ids` for entity effects; set `time_effects=True` and provide `time_ids` for time effects. Enable both and supply both ID arrays for two-way effects. Both options default to `False`: supplying IDs alone does not turn on an effect in the array API.
- The array API does not add an intercept. With fixed effects, leave out a constant column because the effects already absorb it. Without effects, include a constant yourself if the level regression should have an intercept.

The example below has `X.shape == (60, 1)`: 12 entities observed in 5 periods. The entity effect is related to the entity-specific level of `x`; within-entity changes still identify the slope.

<a id="cpu-and-gpu-example"></a>

## Fit an entity-effects model

<a id="cpu-example"></a>

<!-- example: cpu-basic -->
```python
import numpy as np
from statgpu.panel import PanelOLS

rng = np.random.default_rng(7)
entity_ids = np.repeat(np.arange(12), 5)
time_ids = np.tile(np.arange(5), 12)
x = np.repeat(np.linspace(-2, 2, 12), 5) + rng.normal(size=60)
X = x[:, None]
entity_effect = np.repeat(np.linspace(-1, 1, 12), 5)
y = 1.0 + 2.0 * x + entity_effect + 0.2 * rng.normal(size=60)

model = PanelOLS(entity_effects=True, device="cpu").fit(
    X, y, entity_ids=entity_ids
)
print("slope:", np.round(model.coef_, 3))
print("nobs, df_resid:", model.nobs, model.df_resid)
```
<!-- example-end: cpu-basic -->

`coef_` contains the single slope, close to the generating value 2. The common intercept and entity levels are absorbed by the fixed effects, so there is no separate intercept in this coefficient array. `nobs=60` retains the original row count, while `df_resid=47` subtracts the 12 entity-effect directions and one slope.

To run this example on a GPU, replace `device="cpu"` with `device="cuda"` (CuPy/CUDA) or `device="torch"` (Torch CUDA). An unavailable explicitly requested backend raises an error instead of switching to CPU.

<a id="outputs"></a>

## Read the results and make predictions

`coef_` follows the columns of `X`; in the example it is a within-entity slope. A one-unit change in `x` within the same entity is associated with an estimated change of about 2 in `y`, under the model's assumptions. Fixed-effect levels are not separate entries in `coef_`.

`bse_` is the uncertainty of each coefficient, in the same order as `coef_`; it is not the spread of individual predictions. `conf_int_` gives one lower/upper interval per coefficient (`alpha=0.05` means 95%). `tvalues_` is the coefficient-to-standard-error ratio: the default `nonrobust` fit uses a Student-$t$ reference with `df_resid`, while the other supported covariance choices use a normal reference for p-values and intervals. See [Panel covariance](covariance.md) before comparing inference across settings.

`nobs` counts original observations. `df_resid` also subtracts the dimensions occupied by the fixed effects, so it is not just the row count minus the number of visible slope coefficients. `rsquared_within` reports fit on the transformed data; `fit_statistics_` provides the standardized within/between/overall $R^2$ and available joint-test statistics described in [fit statistics](fit-statistics.md). `summary()` prints and returns the panel summary.

### Predict with or without fitted effects

`model.predict(X_new)` returns only the linear part $X_{new}\widehat\beta$. To add fitted entity levels in the example, use `model.predict(X_new, entity_ids=new_entity_ids)`. Keep the same feature columns and order used for fitting; predictions are returned as a NumPy array.

For a one-way fit, a supplied known label adds its fitted effect; an omitted or unseen label adds none. For a two-way fit, supply both `entity_ids` and `time_ids` to include the effects. Both labels must be known and belong to the same connected component of the fitted entity-time pattern. A known label paired with an unknown or omitted label, or labels from different components, raises an error. If neither label is known (including when both arrays are omitted), prediction uses only the linear part. This fallback is not an estimate of the missing entity or period effect.

Think of entities and periods as two kinds of nodes. Each training observation draws an edge between its entity and period. Labels reachable from one another along these edges form a connected component. For example, observations `(A, 1)`, `(A, 2)`, `(B, 1)`, and `(B, 2)` connect firms A/B to periods 1/2. A separate group `(C, 3)`, `(C, 4)`, `(D, 3)`, and `(D, 4)` connects C/D to 3/4. With no observation linking these groups, `(A, 3)` is forbidden for prediction with effects even though both labels were seen. The relative effect levels of the two groups cannot be determined from the training data. A pair need not itself have occurred in training: known labels in the same component are sufficient, subject to the rules above.

After formula fitting, `predict(new_df)` rebuilds the regressor design from the formula, but fixed-effect prediction IDs still need to be passed explicitly.

## Choose effects and standard errors

Use `entity_effects=True` for within-entity comparisons, `time_effects=True` to account for common period shocks, or both for two-way effects. Both options default to `False`. Without either effect, this is a level regression; [PooledOLS](pooled-ols.md) offers a pooled model with an automatic intercept.

Choose the effects from the comparison you want to make, then choose uncertainty settings for the remaining errors:

- `cov_type="nonrobust"` is the default classical covariance. HC0–HC3 are also available, but ordinary HC0/HC1 on the within regression can be inconsistent with fixed short $T>2$, even without serial correlation in the original errors. HC2/HC3 are not a general fix; see [HC applicability](covariance.md#nonrobust-and-hc-covariance).
- For errors that may be correlated within an entity, use `cov_type="clustered"` and pass `cluster=entity_ids` to `fit()`. An effect ID does not automatically select the clustering level. `group_debias=True` is an optional small-number-of-clusters correction, available only with clustered covariance.
- For Driscoll-Kraay covariance, use `cov_type="driscoll-kraay"` and supply `time_ids`, even if no time fixed effects are included. `bandwidth` and `kernel` control its time smoothing. This estimator does not support the separate `hac` option offered by `PooledOLS`.

Covariance settings change standard errors and related inference, not which fixed effects are removed. See [Panel covariance](covariance.md) for assumptions, multiway clustering, and bandwidth rules. [Pooling F and Hausman tests](diagnostics.md) can inform a comparison of specifications; they do not replace the substantive exogeneity assumptions.

## Formula Example

Assume `df` contains `y`, `x1`, `x2`, `entity`, and `time` columns.

```python
from statgpu.panel import PanelOLS

# Two-way fixed effects with pipe syntax.
two_way = PanelOLS().fit(
    formula="y ~ x1 + x2 | entity + time",
    data=df,
)

# The same fixed-effect structure with effect tokens.
two_way_tokens = PanelOLS().fit(
    formula="y ~ x1 + x2 + EntityEffects + TimeEffects",
    data=df,
)

# Ordinary level regression without an intercept.
level_no_intercept = PanelOLS().fit(
    formula="y ~ 0 + x1 + x2",
    data=df,
)
```

Use either pipe syntax or effect tokens for fixed effects, not both. Pipe syntax accepts at most two fixed-effect variables. Pipe-named entity/time columns are authoritative: if `entity_ids` or `time_ids` is also supplied explicitly, it must match the retained formula column exactly; a conflicting duplicate source raises an error.

<a id="numerical-and-strict-behavior"></a>

## Limits to check before interpreting a fit

Two-way demeaning must actually converge. If the requested tolerance is not reached within `demean_max_iter`, statgpu raises an error instead of using the last iterate as an approximate fit.

If the transformed regressors are exactly collinear, fitted values remain well defined but the coefficient vector is not unique. statgpu therefore disables coefficient-level standard errors, tests, p-values, and confidence intervals for that fit instead of reporting inference from an arbitrary coefficient representation; see [Panel covariance](covariance.md).

Formula parsing also raises clear errors for unsupported requests: mixing pipe and effect-token syntax, specifying more than two pipe fixed effects, or supplying a fixed-effect formula with no non-intercept regressor.

## FAQ

**Why can two-way demeaning raise instead of returning the last iterate?**  Because using a non-converged transformation would change the fitted regression. The requested tolerance is therefore enforced before results are returned.

**Does robust covariance change `fit_statistics_.f_statistic` into a robust Wald test?**  No. This field remains the classical joint test of the fitted slope regressors; see [fit statistics](fit-statistics.md).

## Statistical Model and Identification

For entity and time fixed effects, the classical fixed-parameter model is

$$
y_{it}=x_{it}^{\top}\beta+a_i+\gamma_t+\varepsilon_{it}.
$$

Here $a_i$ and $\gamma_t$ are **fixed but unknown nuisance parameters**. They are not assigned a probability distribution, and the fixed-effects model does not require an orthogonality assumption between these nuisance effects and the regressor history $X_i=(x_{i1},\ldots,x_{iT_i})$.

For a static linear panel model, a common sufficient exogeneity condition for the usual fixed-effects interpretation is

$$
E\!\left(\varepsilon_{it}\mid X_i,a_i,\gamma_1,\ldots,\gamma_T\right)=0.
$$

In the fixed-parameter formulation this expectation is understood conditional on the included effects. The key restriction is therefore on the idiosyncratic error $\varepsilon_{it}$, not on the relationship between $a_i$ and the regressors.

The slope $\beta$ is identified only from regressor variation that remains after the included effects are removed. For example, a time-invariant regressor is absorbed by entity fixed effects and cannot have a separately identified slope in that specification. When both `entity_effects=False` and `time_effects=False`, the model reduces to an ordinary level regression and the fixed-effects interpretation above no longer applies.

## Estimator

statgpu applies these transformations directly instead of building a large dummy-variable matrix. The statistical fixed-effects model is unchanged.

Let $F$ denote the included fixed-effect design and $M_F=I-F(F^\top F)^+F^\top$. The slope estimate solves

$$
\widehat\beta_{\mathrm{FE}}
=\arg\min_\beta\|M_F(y-X\beta)\|_2^2
=(X^\top M_FX)^+X^\top M_Fy.
$$

For one-way entity effects, this is ordinary within-entity demeaning. For two-way effects, statgpu repeatedly removes entity and time means until the requested tolerance is reached. If that iteration does not converge within `demean_max_iter`, `.fit()` raises an error instead of returning a partially demeaned regression.

## Covariance and Inference

Standard errors are computed from the same transformed regressors and residuals used to estimate $\widehat\beta_{\mathrm{FE}}$. Writing the transformed design as $Z=M_FX$ and the transformed residual as $e=M_F(y-X\widehat\beta_{\mathrm{FE}})$, the shared nonrobust, HC, clustered, and Driscoll-Kraay formulas are given in [Panel covariance](covariance.md).

For Driscoll-Kraay, the degrees-of-freedom adjustment must also count the fixed effects. The corresponding fixed-effect rank is

$$
r_F=\begin{cases}N,&\text{entity only},\\T,&\text{time only},\\N+T-C,&\text{two way},\end{cases}
$$

where $C$ is the number of connected components in the observed entity-time incidence graph.

The public residual degrees of freedom use this same nuisance-effect rank,

$$
df_{\mathrm{resid}}=n-\operatorname{rank}(Z)-r_F.
$$

This single definition drives the reported `df_resid`, the nonrobust residual-variance scale, the HC1 (`robust`) finite-sample correction, and nonrobust Student-$t$ inference. It is therefore identical to the residual df of the corresponding explicit fixed-effect dummy regression rather than an `N-1`/`T-1` shortcut applied after within transformation.

## Parameters

| Parameter | Default | Allowed / Constraint | Meaning |
|---|---:|---|---|
| `entity_effects` | `False` | boolean | Remove entity-specific level effects. |
| `time_effects` | `False` | boolean | Remove time-specific level effects. |
| `cov_type` | `"nonrobust"` | `nonrobust`, `robust`/`hc1`, `hc0`, `hc2`, `hc3`, `clustered`, `driscoll-kraay`/`dk`/`kernel` | How coefficient standard errors are computed. |
| `alpha` | `0.05` | finite and strictly between 0 and 1 | Confidence-interval significance level; `0.05` gives 95% intervals. |
| `device` | `"auto"` | `auto`, `cpu`, `cuda`, `torch` | Where numerical computation runs. |
| `n_jobs` | `None` | integer or `None` | Shared parallelism hint. |
| `bandwidth` | `None` | `None` or a non-negative integer; DK only | Driscoll-Kraay smoothing bandwidth. `None` uses the documented automatic rule. |
| `kernel` | `"bartlett"` | Bartlett/Newey-West, Parzen/Gallant, or QS/Quadratic-Spectral/Andrews aliases | Driscoll-Kraay kernel. |
| `group_debias` | `False` | boolean; clustered covariance only | Apply the small-number-of-clusters correction. |
| `demean_max_iter` | `1_000_000` | positive integer | Maximum iterations allowed for two-way demeaning. |
| `demean_tol` | `1e-10` | finite and positive | Convergence tolerance for two-way demeaning. |

### Fitting and other methods

```python
model.fit(X=None, y=None, entity_ids=None, time_ids=None,
          cluster=None, formula=None, data=None)
```

Supply arrays or `formula` with `data`; `fit()` returns the fitted estimator. ID arrays have one label per observation, and `cluster` supplies one- or two-way grouping as described in [Panel covariance](covariance.md). Formula-side metadata must align with the retained observations; see [the formula example above](#formula-example).

- `predict(X, entity_ids=None, time_ids=None)`: linear predictions, with eligible supplied fitted effects added as described above.
- `summary()`: print and return the structured panel summary.
- `pooling_f_test()`: compare the included fixed effects with the pooled model.
- `hausman_test(random_effects_model)`: compare with a compatible fitted random-effects model; see [applicability conditions](diagnostics.md).
- `get_params(deep=True)` / `set_params(**params)`: inspect or change constructor settings. Changing settings with `set_params()` clears fitted results; fit again before predicting or reading a summary.

Fitted attributes: `coef_`, `bse_`, `tvalues_`, `pvalues_`, `conf_int_`, `rsquared_within`, `fit_statistics_`, `nobs`, and `df_resid`. Coefficient-inference arrays are unavailable for an exactly rank-deficient transformed design.

The inherited `adjust_pvalues`, `combine_pvalues`, `bootstrap_statistic`, and `permutation_test` helpers have complete signatures in the [shared estimator API](../reference/estimator-api.md). They do not automatically select a panel-valid resampling scheme.

<a id="external-validation"></a>

## Comparing with Other Packages

For one- or two-way fixed effects, align the sample, included effects, transformed design, and absorbed-effect degrees of freedom before comparing with linearmodels. The same alignment is needed for one-way comparisons with R plm. Driscoll-Kraay comparisons additionally require matching time order, kernel, and bandwidth. With no fixed effects, compare against an ordinary OLS regression such as statsmodels using the same intercept specification. See [comparing covariance definitions](covariance.md#comparing-covariance-definitions).

## References

- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). The MIT Press.

For covariance- and diagnostic-specific sources, see [Panel covariance](covariance.md) and [Panel diagnostics](diagnostics.md).
