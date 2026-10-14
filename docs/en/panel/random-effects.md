# RandomEffects

> Language: English  
> Last updated: 2026-10-13<br>
> Switch: [Chinese](../../cn/panel/random-effects.md)

<a id="overview"></a>

## When a random intercept is appropriate

Use `RandomEffects` for a one-way random-intercept panel model when both within-entity changes and between-entity differences should contribute to the common slope. For example, it can combine changes within a firm with comparisons across firms, while accounting for their different baseline levels.

First separate what the regressors do not explain into two parts: a persistent deviation $a_i$ shared by all observations of an entity, and an observation-specific error $\varepsilon_{it}$. The latter is also called the idiosyncratic error. The regressor history means that entity's predictor values across all its periods, not an extra input array.

The key assumption is stronger than for fixed effects: given this entire regressor history, the random effect $a_i$ must have conditional mean zero. For example, the assumption can fail if firms with persistently higher inputs also have persistently higher, unrecorded management ability. Given both that history and $a_i$, each $\varepsilon_{it}$ must also have conditional mean zero. The classical error-components structure further assumes constant component variances, serially uncorrelated idiosyncratic errors, and orthogonality between the two error components. The precise conditions appear below.

If stable entity differences are systematically related to the regressors, consider [PanelOLS](panel-ols.md) with entity effects. Robust standard errors do not remove this identification problem. `RandomEffects` models one random intercept per entity, not time random effects or random slopes.

The model uses the estimated variances of the two error components to decide how much of each entity mean to subtract from a row. The fit estimates Swamy-Arora variance components, subtracts part of each entity mean (quasi-demeaning), then estimates the slopes by feasible GLS.

The model is $y_{it}=x_{it}^{\top}\beta+a_i+\varepsilon_{it}$, where $i$ indexes entities and $t$ indexes periods. The response is $y_{it}$, $x_{it}$ is the row's predictor vector (including a constant if supplied), and $\beta$ contains the common coefficients. The random intercept $a_i$ is shared by all observations of entity $i$, while $\varepsilon_{it}$ is the observation-specific error.

Quasi-demeaning forms $y_{it}^{*}=y_{it}-\theta_i\bar y_i$ and $x_{it}^{*}=x_{it}-\theta_i\bar x_i$, then fits least squares to these transformed rows. Bars denote means over the entity's observed periods; $\theta_i$ controls how much of each mean is removed, using the estimated variance components and the entity's observation count.

## Prepare entity groups and the design matrix

Use long-form observations, with one row per entity and period:

- `X` has shape `(n_observations, n_features)` and `y` has shape `(n_observations,)`; keep all rows aligned.
- `entity_ids` is required, with one label per row. It identifies the entities whose means and variance components enter the transformation.
- The array API does not add a constant. If you want a common intercept, include a column of ones in `X`.
- `time_ids` is needed for Driscoll-Kraay covariance, and `cluster` is needed for clustered covariance. Passing these labels does not add another random effect.

The example uses 12 entities over 5 periods. Its entity effect is drawn independently of `x`, matching the random-effects orthogonality assumption. `X.shape == (60, 2)` includes the explicit constant and one regressor.

<a id="cpu-and-gpu-example"></a>

## Fit the random-intercept model

<a id="cpu-example"></a>

<!-- example: cpu-basic -->
```python
import numpy as np
from statgpu.panel import RandomEffects

rng = np.random.default_rng(7)
entity_ids = np.repeat(np.arange(12), 5)
time_ids = np.tile(np.arange(5), 12)
x = rng.normal(size=60)
X = np.column_stack([np.ones(60), x])
entity_effect = rng.normal(scale=0.6, size=12)
y = 1.0 + 2.0 * x + entity_effect[entity_ids] + 0.2 * rng.normal(size=60)

model = RandomEffects(device="cpu").fit(X, y, entity_ids=entity_ids)
print("coef:", np.round(model.coef_, 3))
print("theta:", round(float(model.theta_), 3))
print("nobs:", model.nobs)
```
<!-- example-end: cpu-basic -->

`coef_` follows the columns of `X`: intercept, then slope. The slope is close to the generating value 2. `theta_` reports the quasi-demeaning factor here because all entities have the same number of observations; a value between 0 and 1 removes part of each entity mean. `nobs=60` still counts the original rows. This small simulation illustrates the output, not a test of the random-effects assumption.

To run this example on a GPU, replace `device="cpu"` with `device="cuda"` (CuPy/CUDA) or `device="torch"` (Torch CUDA). An unavailable explicitly requested backend raises an error instead of switching to CPU.

<a id="outputs"></a>

## Read the coefficients and variance components

`coef_` follows the original level-design columns. In the example, the first coefficient is a common intercept and the second is the common slope estimated from within- and between-entity variation.

`bse_` is the uncertainty of each coefficient, in the same order as `coef_`; it is not the spread of individual predictions. `conf_int_` gives one lower/upper interval per coefficient (`alpha=0.05` means 95%). `tvalues_` is the coefficient-to-standard-error ratio: the default `nonrobust` fit uses a Student-$t$ reference with `df_resid`, while the other supported covariance choices use a normal reference for p-values and intervals. See [Panel covariance](covariance.md) before comparing inference across settings.

`variance_components_` is a dictionary with `sigma2_e` (idiosyncratic-error variance) and `sigma2_a` (random-intercept variance). `theta_` summarizes how much of each entity mean is removed. For an unbalanced panel, the fit uses entity-specific factors; `theta_` is their average across entities, not a single factor applied to every row. Equivalently, factors for entities of the same observation count are weighted by the number of those entities.

`nobs` counts original rows, `df_resid` is the final transformed regression's residual degrees of freedom, and `fit_statistics_` contains standardized fit measures; see [fit statistics](fit-statistics.md). `summary()` prints and returns a structured panel summary.

### Predict the common part of the outcome

`model.predict(X_new)` returns the original level design times `coef_`, as a NumPy array. It does not return a quasi-demeaned outcome and does not estimate or add an entity-specific random intercept. No entity IDs are accepted by `predict()`.

Use the same design columns and order as at fit time, including the explicit constant used in this example. After formula fitting, a DataFrame with the required regressors can be passed instead. A shortened array can omit the fitted explicit constant only when its identity is unambiguous; other feature-count mismatches are rejected. Keeping the full training design is the clearest array convention.

## Choose uncertainty settings and refit

A different `cov_type` changes uncertainty after that fit, not the variance components or coefficients.

Start with the random-effects assumptions above, then select the uncertainty model. The default is `cov_type="nonrobust"`; HC0/1/2/3, clustered, and Driscoll-Kraay alternatives are also available.

- For entity-clustered errors, use `cov_type="clustered"` with `cluster=entity_ids` in `fit()`. Merely supplying `entity_ids` does not select clustered standard errors. `group_debias=True` optionally adds the small-number-of-clusters correction.
- Driscoll-Kraay needs `time_ids`; use `bandwidth` and `kernel` to choose the time smoothing. There is no separate `hac` option for this model.
- To compare covariance choices, fit again with the same data and other settings. `set_params()` resets fitted results, and direct attribute assignment does not recompute them.

These choices affect covariance for the same quasi-demeaned GLS regression. They do not alter the Swamy-Arora transformation or repair correlation between the random effect and the regressors. The classical [Hausman comparison](diagnostics.md) is available only under its documented applicability conditions.

## Formula Example

Assume `df` contains `y`, `x1`, `x2`, and `entity` columns.

```python
from statgpu.panel import RandomEffects

with_intercept = RandomEffects().fit(
    formula="y ~ x1 + x2 | entity",
    data=df,
)

without_intercept = RandomEffects().fit(
    formula="y ~ 0 + x1 + x2 | entity",
    data=df,
)
```

The first pipe variable is the entity grouping column. A second pipe variable is accepted only with `cov_type="driscoll-kraay"`, where it supplies `time_ids`; otherwise the fit fails clearly instead of silently ignoring that variable. If explicit `entity_ids`/`time_ids` are also supplied, they must match any pipe-named columns. Fixed-effect magic tokens (`EntityEffects`, `TimeEffects`, `FixedEffects`) are rejected for `RandomEffects`; use the pipe syntax to provide grouping metadata instead.

<a id="numerical-and-strict-behavior"></a>

## Check whether the fit is supported by your data

The within and between auxiliary regressions must both have positive residual degrees of freedom. In particular, the number of entities must exceed the identified rank of the between regression; having many rows from only a few entities is not enough.

If the final transformed design is exactly rank deficient, fitted values may still be available but the coefficient vector is not unique. Coefficient-level standard errors, tests, p-values, and confidence intervals are then disabled.

Numerical safeguards in the auxiliary regressions and final GLS preserve the same Swamy-Arora and least-squares definitions at extreme response scales or with cancellation. If float64 cannot reliably represent the within/between residual-variance contributions or the quasi-demeaned data, `fit()` raises `FloatingPointError`. Finite input and full rank alone do not guarantee representable results.

A new fit attempt clears the previous fitted state; a failed fit does not retain partial results as usable output. Invalid covariance inputs or unavailable explicitly requested GPU backends also raise errors.

## FAQ

**Does `cov_type` change the Swamy-Arora coefficient estimate?**  No. With the same data and other settings, refitting with another covariance choice changes only the standard errors and related inference reported after the GLS fit.

**Why can $\widehat\sigma_a^2$ be zero?**  The raw Swamy-Arora estimate can be negative in finite samples; statgpu truncates that variance estimate at zero because a variance cannot be negative.

## Statistical Model and Identification

A standard one-way random-effects model is

$$
y_{it}=x_{it}^{\top}\beta+a_i+\varepsilon_{it},
$$

where $x_{it}$ may include a constant. The entity effect $a_i$ is now a random variable rather than a fixed nuisance parameter. The classical error-components interpretation uses

$$
E(a_i)=0,
\qquad
\operatorname{Var}(a_i)=\sigma_a^2,
\qquad
\operatorname{Var}(\varepsilon_{it})=\sigma_e^2.
$$

A key distinction from fixed effects is the orthogonality restriction on the random effect. A common sufficient condition is

$$
E(a_i\mid X_i)=0,
\qquad
E(\varepsilon_{it}\mid X_i,a_i)=0,
$$

where $X_i=(x_{i1},\ldots,x_{iT_i})$. The classical one-way error-components covariance structure additionally treats the idiosyncratic errors as serially uncorrelated within entity, for example

$$
\operatorname{Cov}(\varepsilon_{it},\varepsilon_{is}\mid X_i)=0,
\qquad t\ne s,
$$

with the random effect orthogonal to the idiosyncratic error. These assumptions produce the familiar random-intercept covariance structure that motivates the Swamy-Arora variance-component transformation.

Under this interpretation, both within-entity and between-entity variation can be used to estimate the common slope $\beta$. If $a_i$ is systematically related to the regressor history, the random-effects GLS calculation can still be carried out numerically, but its coefficient generally need not identify the same structural $\beta$ as a fixed-effects estimator. This is the substantive distinction behind the classical FE-versus-RE Hausman comparison.

## Estimator

Swamy-Arora first estimates

$$
\widehat\sigma_e^2=\frac{RSS_W}{df_W},
\qquad
\bar T_H=\frac{N}{\sum_{i=1}^N T_i^{-1}},
$$

and

$$
\widehat\sigma_a^2=
\max\left\{0,\frac{RSS_B/df_B-\widehat\sigma_e^2}{\bar T_H}\right\}.
$$

Here $df_W=n-r_W-r_E$, with $r_E=N$ when the level design has an explicit constant and $r_E=N-1$ otherwise; $df_B=N-r_B$. If an auxiliary regression contains redundant columns, the numerical rank is used so those redundant directions are not counted as additional degrees of freedom.

For entity $i$,

$$
\theta_i=1-\sqrt{\frac{\widehat\sigma_e^2}{\widehat\sigma_e^2+T_i\widehat\sigma_a^2}},
$$

$$
y_{it}^*=y_{it}-\theta_i\bar y_i,
\qquad
x_{it}^*=x_{it}-\theta_i\bar x_i.
$$

This quasi-demeaning subtracts part of each entity mean. The amount depends on the estimated within-entity and between-entity variance components and on the entity's number of observations.

The feasible-GLS estimator is then

$$
\widehat\beta_{\mathrm{RE}}
=\arg\min_\beta\|y^*-X^*\beta\|_2^2
=(X^{*\top}X^*)^+X^{*\top}y^*.
$$

## Covariance and Inference

Standard errors are based on the quasi-demeaned data $(y^*,X^*)$ used in the GLS regression. In particular,

$$
\widehat V_{\mathrm{nonrobust}}
=\widehat\sigma_*^2(X^{*\top}X^*)^+,
\qquad
\widehat\sigma_*^2=\frac{e^{*\top}e^*}{n-\operatorname{rank}(X^*)},
$$

where $e^*=y^*-X^*\widehat\beta_{\mathrm{RE}}$. HC0/1/2/3, clustered, and Driscoll-Kraay alternatives use the same transformed regression; see [Panel covariance](covariance.md).

Robust covariance choices can relax assumptions used to estimate the **reported covariance after the GLS transformation**, but they do not change the Swamy-Arora transformation itself and do not remove the mean-model orthogonality condition $E(a_i\mid X_i)=0$ needed for the usual random-effects interpretation of $\beta$.

## Parameters

| Parameter | Default | Allowed / Constraint | Meaning |
|---|---:|---|---|
| `cov_type` | `"nonrobust"` | `nonrobust`, `robust`/`hc1`, `hc0`, `hc2`, `hc3`, `clustered`, `driscoll-kraay`/`dk`/`kernel` | How standard errors are computed after the random-effects GLS transformation. |
| `alpha` | `0.05` | finite and strictly between 0 and 1 | Confidence-interval significance level; `0.05` gives 95% intervals. |
| `device` | `"auto"` | `auto`, `cpu`, `cuda`, `torch` | Where numerical computation runs. |
| `n_jobs` | `None` | integer or `None` | Shared parallelism hint. |
| `bandwidth` | `None` | `None` or a non-negative integer; DK only | Driscoll-Kraay smoothing bandwidth. |
| `kernel` | `"bartlett"` | Bartlett/Newey-West, Parzen/Gallant, or QS/Quadratic-Spectral/Andrews aliases | Driscoll-Kraay kernel. |
| `group_debias` | `False` | boolean; clustered covariance only | Apply the small-number-of-clusters correction. |

### Fitting and other methods

```python
model.fit(X=None, y=None, entity_ids=None, time_ids=None,
          formula=None, data=None, cluster=None)
```

Supply arrays or `formula` with `data`; `fit()` returns the fitted estimator. Entity labels are required and have one label per observation. DK also requires `time_ids`; clustered covariance requires `cluster` with the one- or two-way grouping described in [Panel covariance](covariance.md). Formula-side metadata must align with the retained observations; see [the formula example above](#formula-example).

- `predict(X)`: return the common level prediction, without entity-specific random intercepts.
- `summary()`: print and return the structured panel summary.
- `hausman_test(fixed_effects_model)`: compare with a compatible fitted fixed-effects model; see [Panel diagnostics](diagnostics.md).
- `get_params(deep=True)` / `set_params(**params)`: inspect or change constructor settings; refit after `set_params()` clears the fitted state.

Fitted attributes: `coef_`, `bse_`, `tvalues_`, `pvalues_`, `conf_int_`, `theta_`, `variance_components_`, `fit_statistics_`, `nobs`, and `df_resid`. Coefficient-inference arrays are unavailable for an exactly rank-deficient transformed design.

The inherited `adjust_pvalues`, `combine_pvalues`, `bootstrap_statistic`, and `permutation_test` helpers have complete signatures in the [shared estimator API](../reference/estimator-api.md). They do not automatically select a panel-valid resampling scheme.

<a id="external-validation"></a>

## Comparing with Other Packages

Random-effects coefficient estimates are **not** guaranteed to match another package because statgpu uses its own Swamy-Arora variance-component construction. A like-for-like covariance comparison uses statgpu's quasi-demeaned $(X^*,y^*)$ regression: robust and Driscoll-Kraay covariance can be compared with linearmodels, and HC2/HC3 with statsmodels, after aligning covariance settings and degrees of freedom. Such a comparison concerns uncertainty for that transformed regression; it does not establish coefficient equivalence between different random-effects implementations. See [comparing covariance definitions](covariance.md#comparing-covariance-definitions).

## References

- Swamy, P. A. V. B., & Arora, S. S. (1972). The exact finite sample properties of the estimators of coefficients in the error components regression models. *Econometrica*, 40(2), 261-275. [https://doi.org/10.2307/1909405](https://doi.org/10.2307/1909405)
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). The MIT Press.
