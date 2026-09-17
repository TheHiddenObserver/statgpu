# 面板拟合统计量

> 语言：中文  
> 最后更新：2026-09-17  
> 切换：[English](../../en/panel/fit-statistics.md)

## 概述

面板估计器通过统一的 `fit_statistics_` 对象提供拟合优度统计量。overall、between 和 within $R^2$ 分别回答不同问题：

- **overall $R^2$**：模型对实际响应水平差异的解释程度；
- **between $R^2$**：模型对不同个体平均水平差异的解释程度；
- **within $R^2$**：模型对同一个体围绕自身平均水平变化的解释程度。

这些统计量不能简单互换，因为它们对应不同的数据变异来源。

## $R^2$ 定义

对拟合系数 $\widehat\beta$，overall $R^2$ 写为

$$
R^2_{\mathrm{overall}}
=1-\frac{\sum_{it}(y_{it}-x_{it}^\top\widehat\beta)^2}{TSS_{\mathrm{overall}}}.
$$

个体间 $R^2$ 使用个体均值：

$$
R^2_{\mathrm{between}}
=1-\frac{\sum_i(\bar y_i-\bar x_i^\top\widehat\beta)^2}{TSS_{\mathrm{between}}}.
$$

个体内 $R^2$ 使用相同形式，但先从 $y$ 和 $X$ 中减去各个体自身的均值。

只有当实际拟合的水平回归中存在可识别截距时，总平方和才围绕均值中心化。这样，含截距模型与无截距模型会使用与各自统计定义一致的 $R^2$。

## 调整 $R^2$ 与模型 F 统计量

对可以表示为单个 OLS 型残差回归的模型，`fit_statistics_` 还可以提供调整 $R^2$ 和经典模型 F 统计量。

经典模型 F 统计量比较完整模型与受限模型：

$$
F=\frac{(RSS_R-RSS_U)/q}{RSS_U/df_{\mathrm{resid}}},
$$

其中 $q$ 是可独立检验的斜率约束数。

选择稳健协方差只会改变系数标准误及相关推断，**不会**把 `fit_statistics_.f_statistic` 自动改成稳健 Wald 检验。

## `PanelOLS` 中的固定效应自由度

对 `PanelOLS`，固定效应同样占用自由度。标准化统计量使用

$$
df_{\mathrm{resid,diag}}=n-r_X-r_F,
\qquad
df_{\mathrm{total,diag}}=n-r_F,
$$

其中 $r_X$ 是变换后解释变量设计的秩，$r_F$ 是固定效应占用的维度：

$$
r_F=\begin{cases}
N,&\text{仅个体固定效应},\\
T,&\text{仅时间固定效应},\\
N+T-C,&\text{双向固定效应},
\end{cases}
$$

其中 $C$ 是已观测个体—时间关联图的连通分量数。

为保持既有 API 行为，历史公开字段与 `fit_statistics_` 中的标准化统计量可能承担不同职责。需要解释具体字段时，应以对应模型页为准，而不要假设所有面板估计器共享完全相同的 $R^2$ 定义。

## 不同模型的可用性

提供个体信息后，`fit_statistics_` 可以在适用模型上给出 standardized within、between 和 overall $R^2$。

对于普通残差 OLS 型估计器，还可以给出调整 $R^2$ 和经典模型 F 统计量。

`FamaMacBeth` 的结构不同：它先在每个时期进行横截面回归，再对时期系数取平均。因此它可以提供基于参数预测的 within/between/overall $R^2$，但不报告单个残差 OLS 回归意义下的调整 $R^2$ 或模型 F 统计量。

## 数值解释

拟合优度中的平方和和中心化在极端尺度下会使用更稳定的数值计算，以避免可避免的中间溢出或下溢。只要最终统计量在 float64 中可表示，这些数值保护不会改变 $R^2$ 或 F 统计量的统计定义。

如果模型本身由于秩亏或输入限制而不能定义某个系数级统计量，`fit_statistics_` 也不会通过另一种未经声明的定义绕过该限制。

## 相关文档

- [PanelOLS](panel-ols.md)
- [PooledOLS](pooled-ols.md)
- [BetweenOLS](between-ols.md)
- [FirstDifferenceOLS](first-difference-ols.md)
- [RandomEffects](random-effects.md)
- [FamaMacBeth](fama-macbeth.md)
- [面板诊断](diagnostics.md)
- [面板协方差](covariance.md)

## 参考文献

- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). The MIT Press.
