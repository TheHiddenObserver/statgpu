# PanelOLS

> Language: English  
> Last updated: 2026-08-15  
> Switch: [Chinese](../../cn/panel/panel-ols.md)

## Overview

`PanelOLS` fits linear panel regressions with no fixed effects, one-way fixed effects, or two-way fixed effects. With fixed effects, the coefficient estimate is identified from variation left after removing the selected entity and/or time effects.

The implementation performs these transformations directly rather than constructing a large dummy-variable matrix. This changes the numerical representation, not the statistical fixed-effects model.

## Path

Implementation: `statgpu/panel/_fixed_effects.py`.

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

```python
model.fit(X, y, entity_ids=None, time_ids=None, cluster=None)
```

When an effect is enabled through constructor arguments, provide the matching `entity_ids` and/or `time_ids`. Clustered covariance additionally needs `cluster`, and Driscoll-Kraay needs `time_ids`.

## CPU and GPU Example

```python
from statgpu.panel import PanelOLS

cpu = PanelOLS(entity_effects=True, device="cpu").fit(X, y, entity_ids=entity_ids)
cuda = PanelOLS(entity_effects=True, device="cuda").fit(X, y, entity_ids=entity_ids)
torch = PanelOLS(entity_effects=True, device="torch").fit(X, y, entity_ids=entity_ids)
```

`device="cuda"` requires CuPy/CUDA and `device="torch"` requires Torch CUDA. If the requested backend is unavailable, `.fit()` raises an error rather than switching to CPU.

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

## Outputs

Common public results include `coef_`, `bse_`, `tvalues_`, `pvalues_`, `conf_int_`, `rsquared_within`, `fit_statistics_`, `nobs`, and `df_resid`. `summary()` returns the panel summary object. Pooling F and Hausman tests are described in [Panel diagnostics](diagnostics.md).

## Numerical and Strict Behavior

Two-way demeaning must actually converge. If the requested tolerance is not reached within `demean_max_iter`, statgpu raises an error instead of using the last iterate as an approximate fit.

If the transformed regressors are exactly collinear, fitted values remain well defined but the coefficient vector is not unique. statgpu therefore disables coefficient-level standard errors, tests, p-values, and confidence intervals for that fit instead of reporting inference from an arbitrary coefficient representation; see [Panel covariance](covariance.md).

For two-way fixed-effect prediction, stored entity and time effects can be combined only when the requested labels are identified by the fitted data. A known label paired with an unknown label, or labels from incompatible fitted components, raises an error. If both labels are unseen, prediction falls back to the linear $X\widehat\beta$ part because no fitted fixed effect is available for either label.

Formula parsing also raises clear errors for unsupported requests: mixing pipe and effect-token syntax, specifying more than two pipe fixed effects, or supplying a fixed-effect formula with no non-intercept regressor.

## FAQ

**Why can two-way demeaning raise instead of returning the last iterate?**  Because using a non-converged transformation would change the fitted regression. The requested tolerance is therefore enforced before results are returned.

**Does robust covariance change `fit_statistics_.f_statistic` into a robust Wald test?**  No. This field remains the classical joint test of the fitted slope regressors; see [fit statistics](fit-statistics.md).

## External Validation

We compare one- and two-way fixed-effect Driscoll-Kraay results with `linearmodels==7.0`: coefficients use `rtol=2e-10, atol=2e-11`, and covariance/BSE use `rtol=5e-9, atol=5e-11`. The no-fixed-effect OLS path is compared with `statsmodels==0.14.6`, and one-way fixed-effect coefficients are also checked against R `plm==2.6-7`. Shared covariance and R tolerances are summarized in the [validation matrix](covariance.md#validation-matrix).

GPU consistency is tested separately by comparing CuPy and Torch outputs with NumPy using default `rtol=5e-6, atol=5e-7`; observed maximum differences are stored in `results/pr126_p100_fresh/panel_stage_c_correctness_p100.json`.

## References

- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). The MIT Press.

For covariance- and diagnostic-specific sources, see [Panel covariance](covariance.md) and [Panel diagnostics](diagnostics.md).
