# Panel Covariance Estimators

> Language: English  
> Last updated: 2026-08-13  
> Switch: [Chinese](../../cn/panel/covariance.md)

## Common fit-space notation

Let $Z$ be the estimator's actual fit-space design, $e$ its residual vector, and

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

## Nonrobust and HC covariance

$$
\widehat V_{\mathrm{nonrobust}}
=\widehat\sigma^2 B,
\qquad
\widehat\sigma^2=\frac{e^\top e}{df_{\mathrm{resid}}}.
$$

$$
\widehat V_{\mathrm{HC0}}
=\sum_i\psi_i\psi_i^\top,
\qquad
\widehat V_{\mathrm{HC1}}
=\frac{n}{df_{\mathrm{resid}}}\widehat V_{\mathrm{HC0}}.
$$

With leverage $h_i=z_i^\top Bz_i$,

$$
\widehat V_{\mathrm{HC2}}
=\sum_i\frac{\psi_i\psi_i^\top}{1-h_i},
\qquad
\widehat V_{\mathrm{HC3}}
=\sum_i\frac{\psi_i\psi_i^\top}{(1-h_i)^2}.
$$

A numerically unit leverage makes HC2/HC3 undefined and raises. Nonrobust coefficient inference uses the maintained Student-t reference; HC/cluster/DK inference uses the maintained asymptotic-normal reference.

## Clustered covariance

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

Aggregate influence scores by ordered observed time,

$$
g_t=\sum_{i:t_i=t}\psi_i.
$$

For kernel weights $w_\ell$,

$$
\widehat V_{\mathrm{DK}}
=
\frac{n}{n-\mathrm{extra\_df}-r_Z}
\left[
\sum_tg_tg_t^\top+
\sum_{\ell=1}^{T-1}w_\ell
\sum_{t=\ell+1}^{T}
\left(g_tg_{t-\ell}^\top+g_{t-\ell}g_t^\top\right)
\right].
$$

At full column rank $r_Z$ is the number of columns of $Z$; in the documented rank-deficient extension it is $\operatorname{rank}(Z)$. `PanelOLS` uses the fixed-effect nuisance rank as `extra_df`; `PooledOLS` and `RandomEffects` use zero.

`bandwidth=None` uses $\lfloor4(T/100)^{2/9}\rfloor$. Bartlett and Parzen are truncated at the bandwidth. Quadratic Spectral treats bandwidth as a smoothing scale and weights all observed lags when it is positive. Numeric/datetime labels use natural order; ordered pandas categoricals preserve declared chronology.

## Legacy pooled HAC

`PooledOLS(cov_type="hac")` is deliberately separate from Driscoll-Kraay: it applies Bartlett/Newey-West HAC directly to row-ordered influence scores, using `time_index` to define stable order.

## References

White (1980); Newey and West (1987); Andrews (1991); Driscoll and Kraay (1998).
