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

Public covariance helpers include `ols_covariance`, `clustered_covariance`, `two_way_clustered_covariance`, `hac_covariance`, and `driscoll_kraay_covariance`. `hc1` aliases `robust`; `dk` and `kernel` alias `driscoll-kraay`. DK kernel aliases are Bartlett/Newey-West, Parzen/Gallant, and QS/Quadratic-Spectral/Andrews. `PooledOLS(cov_type="hac")` remains the separate row-order Bartlett/Newey-West path.

## Validation Matrix

External-framework accuracy and physical backend precision are distinct validation layers.

| Layer | Reference | Maintained comparison | Assertion tolerance |
|---|---|---|---|
| HC primitives | `statsmodels==0.14.6` | HC2/HC3 on full-rank OLS fit space | `rtol=5e-12`, `atol=5e-14` |
| Cluster / DK primitives | `linearmodels==7.0` | one-/two-way group-debiased cluster; Bartlett/Parzen/QS weights and DK covariance; default bandwidth and `extra_df` | covariance `rtol=5e-12`, `atol=5e-14`; weights `rtol=5e-14`, `atol=5e-15` |
| PooledOLS / PanelOLS integration | `linearmodels==7.0` | coefficients plus DK covariance/BSE; PooledOLS group-debiased cluster covariance | coefficient `rtol=2e-10`, `atol=2e-11`; covariance/BSE `rtol=5e-9`, `atol=5e-11` |
| BetweenOLS / FirstDifferenceOLS integration | `statsmodels==0.14.6` | coefficients plus HC0/HC2/HC3 covariance/BSE on identical transformed fit spaces | coefficient `rtol=5e-10`, `atol=5e-12`; covariance/BSE `rtol=5e-9`, `atol=5e-11` |
| RandomEffects fit-space integration | `linearmodels==7.0`, `statsmodels==0.14.6` | robust/HC2/HC3/DK covariance on statgpu's own Swamy-Arora $X^*,y^*$ fit space; no coefficient-parity claim | covariance `rtol=5e-9`, `atol=5e-11` |
| R external gate | `plm==2.6-7`, `sandwich==3.1-3` | HC0/HC2/HC3 covariance and one-way FE coefficients | covariance `rtol=5e-9`, `atol=5e-11`; FE coefficient `rtol=5e-10`, `atol=5e-11` |
| Physical GPU | NumPy reference | 35 estimator cases + 12 public covariance primitives per CuPy/Torch backend | default `rtol=5e-6`, `atol=5e-7` |

The no-FE level-OLS `PanelOLS` regression is also checked against `statsmodels==0.14.6`, including coefficients, covariance/BSE, $R^2$, adjusted $R^2$, and model F statistics.

The ill-conditioned full-rank stress tests intentionally use scale-aware tolerances: HC0 against statsmodels uses `rtol=2e-6, atol=5e-3`, while stable HC2/HC3 leverage checks use `rtol=5e-11, atol=5e-3` because variances can exceed $10^{10}$.

The external CI assertions are pass/fail tolerances; they are not persisted as observed error summaries. The physical P100 payload does persist per-field `max_abs_differences` in `results/pr126_p100_fresh/panel_stage_c_correctness_p100.json`, with the audit summary in `results/pr126_p100_fresh/validation_summary.txt`.

Maintained external tests are `dev/tests/test_panel_stage_c_external.py`, `dev/tests/test_panel_stage_c_external_defaults.py`, `dev/tests/test_panel_stage_c_linearmodels_estimators.py`, and `dev/tests/test_panel_stage_c_r_external.py`.

## References

- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*, 48(4), 817-838. [https://doi.org/10.2307/1912934](https://doi.org/10.2307/1912934)
- MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties. *Journal of Econometrics*, 29(3), 305-325. [https://doi.org/10.1016/0304-4076(85)90158-7](https://doi.org/10.1016/0304-4076(85)90158-7)
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703-708. [https://doi.org/10.2307/1913610](https://doi.org/10.2307/1913610)
- Andrews, D. W. K. (1991). Heteroskedasticity and autocorrelation consistent covariance matrix estimation. *Econometrica*, 59(3), 817-858. [https://doi.org/10.2307/2938229](https://doi.org/10.2307/2938229)
- Driscoll, J. C., & Kraay, A. C. (1998). Consistent covariance matrix estimation with spatially dependent panel data. *The Review of Economics and Statistics*, 80(4), 549-560. [https://doi.org/10.1162/003465398557825](https://doi.org/10.1162/003465398557825)
- Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011). Robust inference with multiway clustering. *Journal of Business & Economic Statistics*, 29(2), 238-249. [https://doi.org/10.1198/jbes.2010.07136](https://doi.org/10.1198/jbes.2010.07136)
