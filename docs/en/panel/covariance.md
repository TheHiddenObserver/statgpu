# Panel Covariance Estimators

> Language: English  
> Last updated: 2026-10-13<br>
> Switch: [Chinese](../../cn/panel/covariance.md)

## Overview

Choose covariance after choosing the mean model. It changes uncertainty estimates, not which omitted effects the regression removes or whether the regressors are exogenous. For a first fit, start with the [model guide](README.md); use this page to select an option or check its exact definition.

## Choose the error structure

| What you need to allow | Option and scope | Metadata to prepare |
|---|---|---|
| Model-specific classical homoskedastic covariance assumptions | `nonrobust` for the five residual-OLS models below | Model-specific entity/time labels |
| Heteroskedasticity, without allowing arbitrary dependence between rows | `robust`/`hc1`, `hc0`, `hc2`, `hc3` for those five models | No extra covariance labels; HC2/HC3 also require valid leverage |
| Dependence within supplied groups | `clustered` for `PanelOLS`, `PooledOLS`, `RandomEffects` | One or two cluster columns aligned with rows; at least two groups per dimension |
| Cross-entity dependence and weak temporal dependence with a sufficiently long time dimension | `driscoll-kraay` for those same three models | `time_ids` for FE/RE, `time_index` for pooled OLS; kernel and bandwidth |
| Serial dependence in a single ordered observation sequence | `hac` for `PooledOLS` only | Optional `time_index`; otherwise input order is used |
| Uncertainty in average cross-sectional coefficients | `nonrobust` or `newey-west` for `FamaMacBeth` | Its period coefficient series, not a stacked residual covariance |

Choosing an available option is not enough to justify inference: its assumptions must hold for the relevant regression or coefficient series. HC does not handle arbitrary within-entity serial correlation, and ordinary within-regression HC has additional limitations for fixed short panels, explained below. Cluster-robust covariance needs sufficiently many clusters; time-series covariance needs a sufficiently long series with suitable dependence conditions; two groups merely meet the input minimum. `group_debias=True` is a finite-sample adjustment, not a guarantee of reliable few-cluster inference. `BetweenOLS` and `FirstDifferenceOLS` expose only nonrobust/HC options; do not pass clustered or DK options to them.

Select `cov_type` when constructing the estimator, provide the required metadata to `fit()`, and refit after changing covariance settings. Read `bse_`, `tvalues_`, `pvalues_`, and `conf_int_` only when coefficient inference is available. The [public API section](#public-api-and-aliases) lists aliases and standalone helpers; the sections below define the estimators and their limits.

## Regression representation

The panel estimators do not all run OLS on the same data representation: fixed effects use demeaned data, random effects use quasi-demeaned data, and first differences use differenced data. Covariance must therefore be computed from the **same transformed regression that produced the coefficients**.

To write the shared formulas once, let $Z$ denote that model-specific regression design and $e$ its residual vector, and define

$$
B=(Z^\top Z)^+,
\qquad
\psi_i=Bz_i e_i.
$$

For each estimator, $Z$ means:

| Model | Regression used for covariance |
|---|---|
| `PooledOLS` | original level design |
| `PanelOLS` | design after removing the selected fixed effects |
| `RandomEffects` | quasi-demeaned design $X^*$ |
| `BetweenOLS` | one entity-mean observation per entity |
| `FirstDifferenceOLS` | first-differenced design |

`FamaMacBeth` is different: its uncertainty is computed from the time series of period-specific coefficient estimates, as described on the [FamaMacBeth](fama-macbeth.md) page. It requires every retained period design to have full column rank and raises an error if that condition is not met.

For the residual-OLS families listed in the table above, an exactly rank-deficient fit-space design can still have uniquely defined fitted values even though its coefficient vector is not unique. In that situation statgpu keeps the fitted results but disables coefficient-level BSE, tests, p-values, and confidence intervals for that fit rather than reporting inference from an arbitrary coefficient representation.

## Nonrobust and HC Covariance

`nonrobust` uses the classical model-based covariance under the relevant homoskedastic variance assumptions. For fixed effects, homoskedastic and uncorrelated original idiosyncratic errors, conditional on the regressors and effects, justify the usual covariance with the appropriate residual degrees of freedom. Demeaning itself correlates transformed errors; it does not require them to be independent for the classical fixed-effects variance formula to apply. Random effects additionally relies on its error-components specification and GLS transformation. This option does not generally protect against heteroskedasticity or misspecified dependence.

HC0-HC3 use observation-level squared residuals from the regression shown above. Under the appropriate independent-observation asymptotics, they allow heteroskedasticity; HC1 adds a degrees-of-freedom correction, while HC2 and HC3 adjust for leverage. Their availability does not make them valid for every panel transformation.

**Short-panel fixed effects need particular care.** With entity effects, fixed $T>2$, and an increasing number of entities, conventional HC0/HC1 applied to the within regression can be inconsistent even when the original idiosyncratic errors are serially uncorrelated (Stock and Watson, 2008, Section 1). The HC2/HC3 leverage adjustments implemented here are not a general remedy for that fixed-$T$ problem. When entities are independent and sufficiently numerous, entity clustering allows within-entity dependence and is often the relevant alternative; it still does not allow arbitrary cross-entity dependence.

For `BetweenOLS`, the independent sampling units are entity means, not the original panel rows. HC allows heteroskedasticity across those means, including variation associated with unequal numbers of observations per entity, but does not correct dependence between entities. For `FirstDifferenceOLS`, adjacent differences share a level error: even independent level errors generally yield correlated differenced errors when an entity has more than two periods. Neither nonrobust nor HC automatically handles that induced serial correlation. These two estimators expose no clustered, HAC, or DK option; select another model/covariance strategy when their supported assumptions are inappropriate.

$$
\widehat V_{\mathrm{nonrobust}}=\widehat\sigma^2 B,
\qquad
\widehat\sigma^2=\frac{e^\top e}{df_{\mathrm{resid}}}.
$$

$$
\widehat V_{\mathrm{HC0}}=\sum_i\psi_i\psi_i^\top,
\qquad
\widehat V_{\mathrm{HC1}}=\frac{n}{df_{\mathrm{resid}}}\widehat V_{\mathrm{HC0}}.
$$

With leverage $h_i=z_i^\top Bz_i$,

$$
\widehat V_{\mathrm{HC2}}=\sum_i\frac{\psi_i\psi_i^\top}{1-h_i},
\qquad
\widehat V_{\mathrm{HC3}}=\sum_i\frac{\psi_i\psi_i^\top}{(1-h_i)^2}.
$$

HC2 and HC3 require $1-h_i$ to be numerically positive. For a full-rank estimator fit or a direct covariance-primitive call, an observation with leverage effectively equal to 1 therefore raises rather than returning an infinite or unstable variance. If the estimator fit-space is already rank deficient, coefficient-level inference is unavailable regardless of covariance type; in that case statgpu keeps the fitted values and does not attempt an HC2/HC3 coordinate covariance that can be undefined at unit leverage.

Nonrobust coefficient inference uses a Student-t reference. HC, clustered, and Driscoll-Kraay inference use the asymptotic normal reference used by the panel API. Positive covariance diagonal entries are used without an absolute variance floor, so rescaling the response rescales coefficients and standard errors without changing finite t/z statistics. At an exactly zero diagonal variance, a zero coefficient has statistic 0 and a nonzero coefficient has signed-infinite statistic; p-values and confidence intervals are then derived from that explicit result rather than from a fabricated tiny denominator.

## Clustered Covariance

Choose the cluster labels from the dependence you need to allow, not simply from the fixed effects included in the mean model:

- **Entity clustering** allows heteroskedasticity and arbitrary dependence over time within an entity, while requiring independent clusters across entities (or an appropriate asymptotic weakening of independence). It does not cover a common shock correlating different entities.
- **Time clustering** allows arbitrary dependence across entities within a period, while requiring independent period clusters. It does not cover serial correlation between periods.
- **Entity-and-time clustering** allows observations to be dependent when they share either an entity or a period. It does not allow arbitrary dependence between observations sharing neither, such as unrestricted cross-entity spillovers across different periods.

Conventional cluster-robust inference needs sufficiently many clusters in **every** clustering dimension, suitable moment conditions, and no dominant cluster. Many entities do not compensate for very few time clusters in a two-way fit. There is no universal cluster-count cutoff; imbalance and leverage matter too. The API minimum of two groups and `group_debias=True` do not establish reliable few-cluster inference, and the reported reference remains asymptotic normal rather than a few-cluster correction.

For cluster $g$, define $s_g=\sum_{i\in g}\psi_i$. Then

$$
\widehat V_G=\sum_g s_gs_g^\top.
$$

Clustered inference requires at least two distinct groups in every supplied clustering dimension. With only one cluster the grouped score collapses to the full-sample estimating-equation score, so the cluster-robust variance is not estimable; statgpu therefore raises an error even when `group_debias=False`.

With `group_debias=True`, each cluster component is multiplied by the small-number-of-clusters correction

$$
\frac{G}{G-1}\frac{n-1}{n}.
$$

Numerical scaling reduces avoidable overflow and underflow when group contributions differ greatly in magnitude. It does not change the covariance definition or provide arbitrary-precision recovery of small remainders after severe cancellation.

Two-way clustering combines the two one-way cluster covariances and subtracts the covariance for the paired cluster labels. If one clustering dimension is nested in the other, the matching marginal and intersection terms cancel, leaving the covariance for the coarser partition. This depends on which observations belong to each group, not on the particular integer labels used to encode the groups. At extreme scales, a nonzero covariance contribution may not be reliably representable in float64; statgpu raises `FloatingPointError` rather than silently treating that contribution as zero.

The estimator is then

$$
\widehat V_{1,2}=\widehat V_1+\widehat V_2-\widehat V_{12}.
$$

## Driscoll-Kraay

Driscoll-Kraay is useful when dependence extends across entities as well as across time and the time dimension is sufficiently long. Its justification is a large-$T$ argument for the time-aggregated score series, with regularity conditions such as finite moments and sufficiently weak temporal dependence (Driscoll and Kraay, 1998). It is not a guarantee for arbitrary dependence, nonstationary common shocks, or a short panel merely because the number of entities is large.

Bandwidth must balance omitted serial dependence against noisy estimates at long lags. Under standard HAC consistency conditions it grows with the time dimension at a suitably controlled rate, rather than occupying a fixed large fraction of the series. Check sensitivity to plausible bandwidths and the available number of periods; an automatic bandwidth is a rule of thumb, not evidence that these conditions hold.

statgpu first combines observation contributions within each observed period,

$$
g_t=\sum_{i:t_i=t}\psi_i,
$$

and then applies kernel weights across time lags. For weights $w_\ell$,

$$
\widehat V_{\mathrm{DK}}=
\frac{n}{n-\mathrm{extra\_df}-r_Z}
\left[
\sum_tg_tg_t^\top+
\sum_{\ell=1}^{T-1}w_\ell
\sum_{t=\ell+1}^{T}(g_tg_{t-\ell}^\top+g_{t-\ell}g_t^\top)
\right].
$$

Here $r_Z$ is the number of identified regression directions: it equals the number of columns of $Z$ at full column rank and $\operatorname{rank}(Z)$ otherwise. `PanelOLS` also counts the absorbed fixed effects through `extra_df`; `PooledOLS` and `RandomEffects` use zero for this term.

HAC and Driscoll-Kraay use range-aware arithmetic to reduce avoidable overflow and underflow without changing the displayed covariance definitions. These calculations remain subject to float64 precision: a representable final result alone does not guarantee that arbitrarily ill-conditioned cancellation can be resolved accurately.

With `bandwidth=None`, statgpu uses $\lfloor4(T/100)^{2/9}\rfloor$. Bartlett and Parzen kernels assign zero weight beyond the bandwidth. Quadratic Spectral instead treats bandwidth as a smoothing scale and, when positive, gives weights to all observed lags.

Time order matters. Numeric and datetime labels use their natural order. Ordered pandas categoricals use the category order declared by the user. Plain string labels use lexical order; other non-categorical object labels follow their sorted comparable-value order and are rejected if they are not mutually comparable. When lexical string order is not the intended chronology (for example, `t1, t2, t10`), supply a numeric/datetime key or an ordered categorical instead.

<a id="ordered-sequence-hac"></a>

## Pooled HAC is one ordered sequence

`PooledOLS(cov_type="hac")` applies Bartlett/Newey-West covariance to a single sequence of observation-level scores. Supply `time_index` to sort the rows; otherwise input order defines the lags. A useful application is a single time-series regression with meaningful observation spacing, sufficiently weak serial dependence, enough observations, and an appropriate bandwidth.

This calculation does not reset lags at entity boundaries or aggregate observations sharing a period. Simply sorting a stacked panel by time does not turn it into general panel HAC: adjacent rows can belong to different entities, and repeated time labels still occupy separate sequence positions. Use entity clustering for dependence confined within independent entities, two-way clustering for the corresponding two-way pattern, or DK when its long-time-dimension conditions are justified. `PanelOLS` and `RandomEffects` do not accept the separate `hac` option.

<a id="coefficient-series-conditions"></a>

## FamaMacBeth uses a time series of coefficients

For `FamaMacBeth`, `nonrobust` uses the sample variation of period coefficients for the variance of their average, treating period estimates as independent. `newey-west` instead estimates the long-run covariance of that coefficient series, requiring meaningful time order, sufficiently weak serial dependence, suitable moment/stability conditions, an appropriate bandwidth, and sufficiently many retained periods. Two valid periods meet an input minimum, not a reliability guarantee.

Neither option repairs period-level endogeneity, unidentified cross-sectional regressions, or arbitrary structural changes in the coefficient series. A single period with many entities cannot replace a long enough time series for this inference. See [the FamaMacBeth guide](fama-macbeth.md) for its average-slope target, period filtering, and reference distributions.

## Supply two clustering dimensions

The following self-contained example places one entity label and one time label in each row of `cluster`. The fitted mean model includes entity effects; adding time clustering does not add time fixed effects. The simulated common period shocks motivate time clustering alongside entity clustering. The sample sizes illustrate the API, not a minimum number of clusters that guarantees valid inference.

<!-- example: covariance-two-way -->
```python
import numpy as np
from statgpu.panel import PanelOLS

rng = np.random.default_rng(42)
n_entities, n_periods = 80, 60
entity_ids = np.repeat(np.arange(n_entities), n_periods)
time_ids = np.tile(np.arange(n_periods), n_entities)
X = rng.normal(size=(len(entity_ids), 1))
entity_shocks = rng.normal(size=n_entities)
time_shocks = rng.normal(size=n_periods)
y = (2.0 * X[:, 0] + entity_shocks[entity_ids]
     + time_shocks[time_ids] + rng.normal(size=len(entity_ids)))
cluster = np.column_stack([entity_ids, time_ids])
model = PanelOLS(
    entity_effects=True, cov_type="clustered", device="cpu"
).fit(X, y, entity_ids=entity_ids, time_ids=time_ids, cluster=cluster)
print("coef:", np.round(model.coef_, 3))
print("standard errors:", np.round(model.bse_, 3))
```
<!-- example-end: covariance-two-way -->

The two columns select entity and time partitions separately. Passing paired entity-time labels as a single column would instead define one-way clustering on their intersections, a different covariance estimator.

## Public API and Aliases

The public covariance helpers exported by `statgpu.panel` are `clustered_covariance`, `two_way_clustered_covariance`, `hac_covariance`, and `driscoll_kraay_covariance`. For model-based inference, choose a `cov_type` supported by that estimator and read `bse_`, `tvalues_`, `pvalues_`, and `conf_int_`. Among these panel estimators, `FamaMacBeth` additionally exposes its full coefficient covariance as `cov_params_`; the five residual-OLS estimators do not expose that attribute. Consult the individual model guide for its supported covariance options.

For estimator `cov_type` values, `hc1` is an alias of `robust`; `dk` and `kernel` are aliases of `driscoll-kraay`. Driscoll-Kraay kernel aliases include Bartlett/Newey-West, Parzen/Gallant, and QS/Quadratic-Spectral/Andrews. `PooledOLS(cov_type="hac")` remains a separate ordered-sequence Bartlett/Newey-West calculation and should not be confused with Driscoll-Kraay; when `time_index` is supplied, PooledOLS sorts the sequence by that index before applying HAC.

<a id="validation-matrix"></a>

## Comparing Covariance Definitions

A package comparison is meaningful only when both fits use the same observations, transformed regression, intercept, residual degrees of freedom, and covariance settings. In particular:

- For HC0-HC3, compare OLS covariance on the same design and residuals, including the same leverage and finite-sample adjustments.
- For one- or two-way clustering, use the same group partitions and `group_debias` setting.
- For Driscoll-Kraay, align time ordering, kernel, bandwidth, and the degrees of freedom consumed by fixed effects.
- For `BetweenOLS` and `FirstDifferenceOLS`, construct the same entity-mean or differenced sample before comparing with an OLS package such as statsmodels.
- For `RandomEffects`, compare covariance on statgpu's quasi-demeaned $(X^*,y^*)$ regression. Packages using a different variance-component construction need not produce the same GLS coefficients.
- For `FamaMacBeth`, align coefficient-series covariance and the inference reference distribution separately; see [FamaMacBeth comparisons](fama-macbeth.md#comparing-with-other-packages).

Agreement in coefficients or covariance does not by itself imply equal p-values and confidence intervals when the reference distribution or its degrees of freedom differ.

## References

- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*, 48(4), 817-838. [https://doi.org/10.2307/1912934](https://doi.org/10.2307/1912934)
- MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties. *Journal of Econometrics*, 29(3), 305-325. [https://doi.org/10.1016/0304-4076(85)90158-7](https://doi.org/10.1016/0304-4076(85)90158-7)
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703-708. [https://doi.org/10.2307/1913610](https://doi.org/10.2307/1913610)
- Andrews, D. W. K. (1991). Heteroskedasticity and autocorrelation consistent covariance matrix estimation. *Econometrica*, 59(3), 817-858. [https://doi.org/10.2307/2938229](https://doi.org/10.2307/2938229)
- Driscoll, J. C., & Kraay, A. C. (1998). Consistent covariance matrix estimation with spatially dependent panel data. *The Review of Economics and Statistics*, 80(4), 549-560. [https://doi.org/10.1162/003465398557825](https://doi.org/10.1162/003465398557825)
- Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011). Robust inference with multiway clustering. *Journal of Business & Economic Statistics*, 29(2), 238-249. [https://doi.org/10.1198/jbes.2010.07136](https://doi.org/10.1198/jbes.2010.07136)

- Stock, J. H., & Watson, M. W. (2008). Heteroskedasticity-robust standard errors for fixed effects panel data regression. *Econometrica*, 76(1), 155–174. [doi:10.1111/j.0012-9682.2008.00821.x](https://doi.org/10.1111/j.0012-9682.2008.00821.x). [Author manuscript](https://www.princeton.edu/~mwatson/papers/Stock_Watson_Ecta_2008.pdf).
