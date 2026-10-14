# Panel Fit Statistics

> Language: English  
> Last updated: 2026-10-13<br>
> Switch: [Chinese](../../cn/panel/fit-statistics.md)

## Overview

Panel estimators expose a common `fit_statistics_` object so that within-, between-, and overall goodness of fit can be interpreted consistently across models. These quantities answer different questions:

- **overall $R^2$** measures fit to the observed outcome levels;
- **between $R^2$** measures fit to differences in entity means;
- **within $R^2$** measures fit to changes around each entity's own mean.

Before comparing two fits, ask whether you care about changes within an entity, differences in entity averages, or outcome levels. Compare the same statistic on the same sample and with the same intercept convention. A high $R^2$ does not validate exogeneity, and changing covariance does not change the fitted coefficients or make a classical F statistic robust.

## Availability and Outputs

When entity metadata are available, `fit_statistics_` provides standardized within, between, and overall $R^2$. OLS-style estimators also report adjusted $R^2$ and the classical model F statistic when those quantities are defined for the fitted regression.

`FamaMacBeth` reports parameter-based within, between, and overall $R^2$, but it does not report the residual-OLS adjusted $R^2$ or model F because its estimator is an average of period-by-period regressions rather than one pooled residual regression.

`fit_statistics_.f_statistic` is `None` when there are no testable slope restrictions or the F ratio is undefined; a missing statistic is not acceptance of the null. At extreme scales, RSS metadata in original squared units can be `inf` while the corresponding dimensionless F statistic is finite.

Read the standardized fields from `model.fit_statistics_`; consult the model page for legacy fields such as `model.rsquared`. For `PanelOLS`, `model.rsquared_within` uses the selected fixed-effect transformation, whereas `model.fit_statistics_.rsquared_within` uses entity-demeaned level data when entity labels are present. In particular, a two-way fit can report different values without either being an error. See [the FE example](panel-ols.md#cpu-example) for the fitted model and [diagnostics](diagnostics.md) for specification comparisons.

## Definitions

For fitted coefficient vector $\widehat\beta$,

$$
R^2_{\mathrm{overall}}
=1-\frac{\sum_{it}(y_{it}-x_{it}^\top\widehat\beta)^2}{TSS_{\mathrm{overall}}},
$$

$$
R^2_{\mathrm{between}}
=1-\frac{\sum_i(\bar y_i-\bar x_i^\top\widehat\beta)^2}{TSS_{\mathrm{between}}},
$$

and $R^2_{\mathrm{within}}$ applies the same idea after subtracting entity means from $y$ and $X$.

The total sum of squares is centered around a mean only when the fitted level regression actually contains an identified constant. This keeps the $R^2$ definition consistent with intercept and no-intercept models.

For OLS-style estimators, the reported model F statistic is the classical comparison of the fitted regression with a restricted regression:

$$
F=\frac{(RSS_R-RSS_U)/q}{RSS_U/df_{\mathrm{resid}}},
$$

where $q$ is the number of independently testable restrictions. Choosing a robust covariance estimator changes coefficient standard errors, but it does **not** convert `fit_statistics_.f_statistic` into a robust Wald test.

For `PanelOLS`, fixed effects also consume degrees of freedom. The standardized diagnostic counts use

$$
df_{\mathrm{resid,diag}}=n-r_X-r_F,
\qquad
df_{\mathrm{total,diag}}=n-r_F,
$$

where $r_F=N$ for entity effects, $T$ for time effects, and $N+T-C$ for two-way effects, with $C$ the number of connected components in the observed entity-time graph. Each observed row connects its entity node to its period node; $C$ counts the groups that cannot be reached from one another along those connections. See the [prediction example](panel-ols.md#predict-with-or-without-fitted-effects) for two disconnected groups.

`PanelOLS.df_resid` uses the residual degrees of freedom shown above. The two within-$R^2$ fields described under [outputs](#availability-and-outputs) use different transformations; keep that distinction when comparing models.

## References

- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). The MIT Press.

Model-specific conventions are documented on the estimator pages.
