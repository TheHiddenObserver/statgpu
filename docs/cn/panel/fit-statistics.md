# 面板 Fit Statistics

> 语言：中文  
> 最后更新：2026-08-13  
> 切换：[English](../../en/panel/fit-statistics.md)

`fit_statistics_` 使用 parameter-based panel $R^2$。对拟合系数 $\widehat\beta$，

$$
R^2_{\mathrm{overall}}
=1-\frac{\sum_{it}(y_{it}-x_{it}^\top\widehat\beta)^2}{TSS_{\mathrm{overall}}},
$$

$$
R^2_{\mathrm{between}}
=1-\frac{\sum_i(\bar y_i-\bar x_i^\top\widehat\beta)^2}{TSS_{\mathrm{between}}},
$$

$R^2_{\mathrm{within}}$ 则对 entity-demeaned $y,X$ 使用同一定义。只有实际 level design 含有 identified constant 时，overall/between TSS 才中心化。

OLS-style estimator 的 classical joint-slope statistic 为

$$
F=
\frac{(RSS_R-RSS_U)/q}{RSS_U/df_{\mathrm{resid}}},
$$

其中 $q$ 为有效 restriction rank；robust covariance 不会把该字段改成 robust Wald statistic。

`PanelOLS` 的标准 diagnostics 使用

$$
df_{\mathrm{resid,diag}}=n-r_X-r_F,
\qquad
df_{\mathrm{total,diag}}=n-r_F,
$$

其中 $r_F$ 依次为 $N$、$T$ 或 $N+T-C$。历史 public `PanelOLS.df_resid` 与 `PanelOLS.rsquared_within` 继续作为兼容字段，不会被 standardized statistics 静默重定义。

`FamaMacBeth` 提供 parameter-based within/between/overall $R^2$，但不提供 residual-OLS adjusted $R^2$ 或 model F。
