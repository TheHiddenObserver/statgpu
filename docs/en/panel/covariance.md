# Panel Covariance Estimators

> Language: English  
> Last updated: 2026-10-09<br>
> Switch: [Chinese](../../cn/panel/covariance.md)

## Overview

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

`nonrobust` is the usual homoskedastic OLS covariance. HC0-HC3 are heteroskedasticity-consistent alternatives that progressively adjust the contribution of observations with high leverage.

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

Clustered covariance allows observations within the same supplied cluster to have correlated errors. For cluster $g$, define $s_g=\sum_{i\in g}\psi_i$. Then

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

Driscoll-Kraay is a time-indexed covariance estimator for panel regressions. statgpu first combines observation contributions within each observed period,

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

## Public API and Aliases

The public covariance helpers exported by `statgpu.panel` are `clustered_covariance`, `two_way_clustered_covariance`, `hac_covariance`, and `driscoll_kraay_covariance`. `ols_covariance` is an internal shared dispatcher used by panel estimators; it is not part of the public `statgpu.panel` export surface.

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