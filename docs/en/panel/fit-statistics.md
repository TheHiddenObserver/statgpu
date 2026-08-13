# Panel Fit Statistics

> Language: English  
> Last updated: 2026-08-13  
> Switch: [Chinese](../../cn/panel/fit-statistics.md)

`fit_statistics_` uses parameter-based panel $R^2$ definitions. For fitted coefficient vector $\widehat\beta$,

$$
R^2_{\mathrm{overall}}
=1-\frac{\sum_{it}(y_{it}-x_{it}^\top\widehat\beta)^2}{TSS_{\mathrm{overall}}},
$$

$$
R^2_{\mathrm{between}}
=1-\frac{\sum_i(\bar y_i-\bar x_i^\top\widehat\beta)^2}{TSS_{\mathrm{between}}},
$$

and $R^2_{\mathrm{within}}$ applies the same definition to entity-demeaned $y$ and $X$. Overall/between TSS is centered only when the actual level design contains an identified constant.

For an OLS-style estimator's primary fit space, the classical joint-slope statistic is

$$
F=
\frac{(RSS_R-RSS_U)/q}{RSS_U/df_{\mathrm{resid}}},
$$

where $q$ is the effective restriction rank. Robust covariance choices do not turn this field into a robust Wald statistic.

For `PanelOLS`, standardized diagnostics use

$$
df_{\mathrm{resid,diag}}=n-r_X-r_F,
\qquad
df_{\mathrm{total,diag}}=n-r_F,
$$

with fixed-effect nuisance rank $r_F=N$, $T$, or $N+T-C$ as appropriate. The legacy public `PanelOLS.df_resid` and `PanelOLS.rsquared_within` remain compatibility fields and are not silently redefined by the standardized statistics.

`FamaMacBeth` reports parameter-based within/between/overall $R^2$ but does not report a residual-OLS adjusted $R^2$ or model F statistic.
