# 面板 Fit Statistics

> 语言：中文  
> 最后更新：2026-08-15  
> 切换：[English](../../en/panel/fit-statistics.md)

## 概述与路径

panel estimator 通过统一的 `fit_statistics_` 对象提供 goodness-of-fit 统计量。within、between 与 overall $R^2$ 回答的是三个不同问题：

- **overall $R^2$**：模型对实际 outcome level 的解释程度；
- **between $R^2$**：模型对不同 entity 平均水平差异的解释程度；
- **within $R^2$**：模型对同一 entity 围绕自身平均水平变化的解释程度。

这些计算由 `statgpu/panel/` 下的 panel diagnostic/statistics helpers 实现。

## 定义

对拟合系数 $\widehat\beta$，

$$
R^2_{\mathrm{overall}}
=1-\frac{\sum_{it}(y_{it}-x_{it}^\top\widehat\beta)^2}{TSS_{\mathrm{overall}}},
$$

$$
R^2_{\mathrm{between}}
=1-\frac{\sum_i(\bar y_i-\bar x_i^\top\widehat\beta)^2}{TSS_{\mathrm{between}}},
$$

$R^2_{\mathrm{within}}$ 使用同样的定义，但先从 $y$ 与 $X$ 中减去各 entity 的均值。

只有实际拟合的 level regression 中存在可识别的 constant 时，total sum of squares 才围绕均值中心化。这样 intercept model 与 no-intercept model 使用各自一致的 $R^2$ 定义。

对 OLS-style estimator，`fit_statistics_` 中的 model F statistic 是 fitted regression 与 restricted regression 的 classical comparison：

$$
F=\frac{(RSS_R-RSS_U)/q}{RSS_U/df_{\mathrm{resid}}},
$$

其中 $q$ 是可独立检验的 restriction 数。选择 robust covariance 会改变 coefficient standard error，但**不会**把 `fit_statistics_.f_statistic` 自动改成 robust Wald test。

对 `PanelOLS`，fixed effects 也会占用 degrees of freedom。standardized diagnostic 使用

$$
df_{\mathrm{resid,diag}}=n-r_X-r_F,
\qquad
df_{\mathrm{total,diag}}=n-r_F,
$$

其中 entity effects 时 $r_F=N$，time effects 时 $r_F=T$，双向 effects 时 $r_F=N+T-C$；$C$ 是观测 entity-time graph 的 connected-component 数。

为保持 backward compatibility，历史 public fields `PanelOLS.df_resid` 与 `PanelOLS.rsquared_within` 保持原有含义。标准化后的统计量单独放在 `fit_statistics_` 中，而不是静默改变旧字段的定义。

## 可用范围与输出

提供 entity metadata 后，`fit_statistics_` 可以给出 standardized within、between 与 overall $R^2$。对具有普通 residual-OLS 定义的 estimator，还会提供 adjusted $R^2$ 与 classical model F statistic。

`FamaMacBeth` 提供 parameter-based within/between/overall $R^2$，但不报告 residual-OLS adjusted $R^2$ 或 model F，因为它的 estimator 是多个 period regressions 的 coefficient average，而不是单个 pooled residual regression。

## 验证

这些统计量由完整 CPU regression suite 与 estimator-level external comparison 覆盖。tests 会特别确认：选择 robust covariance 只改变 coefficient inference，不会悄悄改变 classical model F statistic 的定义。

## 参考（References）

- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). The MIT Press.

各模型的特殊约定见对应 estimator 页面。
