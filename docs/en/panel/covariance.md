# Panel Covariance Estimators

> Language: English  
> Last updated: 2026-08-14  
> Switch: [Chinese](../../cn/panel/covariance.md)

## Overview and Path

Shared residual-based panel covariance is implemented in `statgpu/panel/_covariance.py`. Let $Z$ be the estimator's actual fit-space design, $e$ its residual vector, and

$$
B=(Z^\top Z)^+,
\qquad
\psi_i=Bz_i e_i.
$$

| Model | $Z$ |
|---|---|
| `PooledOLS` | level design |
| `PanelOLS` | within-transformed design |
| `RandomEffects` | quasi-demeaned $X^*$ |
| `BetweenOLS` | entity-mean design |
| `FirstDifferenceOLS` | first-difference design |

`FamaMacBeth` instead uses covariance of its coefficient time series. For exact rank deficiency, OLS-style panel models retain fitted values and fit-space quantities, but coordinate-wise BSE/test/p-value/CI output is unavailable because the original coefficient representation is not unique.

## Nonrobust and HC Covariance

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

A numerically unit leverage makes HC2/HC3 undefined and raises. Nonrobust coefficient inference uses the maintained Student-t reference; HC/cluster/DK inference uses the maintained asymptotic-normal reference.

## Clustered Covariance

For cluster $g$, define $s_g=\sum_{i\in g}\psi_i$. Then

$$
\widehat V_G=\sum_g s_gs_g^\top.
$$

With `group_debias=True`, each cluster component is multiplied by

$$
\frac{G}{G-1}\frac{n-1}{n}.
$$

Two-way clustering uses exact paired-label inclusion-exclusion,

$$
\widehat V_{1,2}=\widehat V_1+\widehat V_2-\widehat V_{12}.
$$

## Driscoll-Kraay

Aggregate influence scores by ordered observed time, $g_t=\sum_{i:t_i=t}\psi_i$. For kernel weights $w_\ell$,

$$
\widehat V_{\mathrm{DK}}=
\frac{n}{n-\mathrm{extra\_df}-r_Z}
\left[
\sum_tg_tg_t^\top+
\sum_{\ell=1}^{T-1}w_\ell
\sum_{t=\ell+1}^{T}(g_tg_{t-\ell}^\top+g_{t-\ell}g_t^\top)
\right].
$$

At full column rank $r_Z$ is the number of columns of $Z$; in the rank-deficient extension it is $\operatorname{rank}(Z)$. `PanelOLS` supplies the fixed-effect nuisance rank as `extra_df`; `PooledOLS` and `RandomEffects` use zero.

`bandwidth=None` uses $\lfloor4(T/100)^{2/9}\rfloor$. Bartlett and Parzen are truncated at the bandwidth. Quadratic Spectral treats bandwidth as a smoothing scale and weights all observed lags when positive. Numeric/datetime labels use natural order; ordered pandas categoricals preserve declared chronology.

## Public API and Aliases

Public covariance helpers include `ols_covariance`, `clustered_covariance`, `two_way_clustered_covariance`, `hac_covariance`, and `driscoll_kraay_covariance`. `hc1` aliases `robust`; `dk` and `kernel` alias `driscoll-kraay`. `PooledOLS(cov_type="hac")` remains the separate row-order Bartlett/Newey-West path.

## External Validation

Pinned Python references are `linearmodels==7.0` and `statsmodels==0.14.6`; pinned R references are `plm==2.6-7` and `sandwich==3.1-3`. Maintained checks live in `dev/tests/test_panel_stage_c_external.py`, `dev/tests/test_panel_stage_c_external_defaults.py`, `dev/tests/test_panel_stage_c_linearmodels_estimators.py`, and `dev/tests/test_panel_stage_c_r_external.py`. Physical CuPy/Torch validation is summarized in `results/pr126_p100_fresh/validation_summary.txt`.

## References

White (1980); Newey and West (1987); Andrews (1991); Driscoll and Kraay (1998); Cameron, Gelbach, and Miller (2011).
