# 面板 Fit Statistics

> 语言：中文  
> 最后更新：2026-08-14  
> 切换：[English](../../en/panel/fit-statistics.md)

## Overview and Path

标准化 panel fit statistics 通过 `fit_statistics_` 暴露，并由 `statgpu/panel/` 下的 diagnostic/statistics helpers 实现。

## Definitions

对拟合系数 $\widehat\beta$，

$$
R^2_{\mathrm{overall}}
=1-\frac{\sum_{it}(y_{it}-x_{it}^\top\widehat\beta)^2}{TSS_{\mathrm{overall}}},
$$

$$
R^2_{\mathrm{between}}
=1-\frac{\sum_i(\bar y_i-\bar x_i^\top\widehat\beta)^2}{TSS_{\mathrm{between}}},
$$

$R^2_{\mathrm{within}}$ 对 entity-demeaned $y,X$ 使用同一定义。只有实际 level design 含 identified constant 时，overall/between TSS 才中心化。

OLS-style estimator 的 primary fit space 上，

$$
F=\frac{(RSS_R-RSS_U)/q}{RSS_U/df_{\mathrm{resid}}},
$$

其中 $q$ 为 effective restriction rank。robust covariance 不会把该字段改成 robust Wald statistic。

`PanelOLS` 的 standardized diagnostics 使用

$$
df_{\mathrm{resid,diag}}=n-r_X-r_F,
\qquad
df_{\mathrm{total,diag}}=n-r_F,
$$

其中 fixed-effect nuisance rank $r_F$ 依次为 $N$、$T$ 或 $N+T-C$。历史 public `PanelOLS.df_resid` 与 `PanelOLS.rsquared_within` 继续作为 compatibility fields，不会被静默重定义。

## Availability and Outputs

在所需 entity metadata 存在时，`fit_statistics_` 提供 standardized within/between/overall $R^2$；具有相应 residual-OLS fit-space 定义的 estimator 还提供 adjusted $R^2$ 与 classical model-F fields。`FamaMacBeth` 提供 parameter-based within/between/overall $R^2$，但不提供 residual-OLS adjusted $R^2$ 或 model F。

## Validation

panel fit-statistic regressions 由 full CPU suite 与 estimator-level external alignment 覆盖。robust covariance choice 不会被解释为 robust-Wald model F；该契约与 covariance construction 分开测试。

## References

Wooldridge (2010), *Econometric Analysis of Cross Section and Panel Data*；model-specific conventions 见各 estimator 页面。
