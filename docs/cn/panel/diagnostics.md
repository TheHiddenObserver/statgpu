# 面板模型诊断

> 语言：中文  
> 最后更新：2026-09-17  
> 切换：[English](../../en/panel/diagnostics.md)

## 概述

面板诊断用于回答几类不同的模型比较问题，例如：是否需要固定效应、合并回归的误差结构是否足够，以及随机效应设定是否与固定效应估计相容。

每个检验都返回 `PanelTestResult`，其中包含检验统计量、p 值、参考分布、自由度、原假设与备择假设，以及该检验当前是否适用。

## Pooling F 检验

Pooling F 检验用于判断 `PanelOLS` 中加入的固定效应能否整体删除。原假设是所包含的固定效应联合为 0。

$$
F=\frac{(RSS_R-RSS_U)/q}{RSS_U/df_U},
$$

其中 $RSS_R$ 和 $RSS_U$ 分别来自受限与非受限模型，$q$ 是可以独立检验的固定效应约束数。

```python
result = fe.pooling_f_test()
print(result.statistic, result.pvalue)
```

较小的 p 值表示数据不支持把相应固定效应整体删去。

## Breusch–Pagan LM 检验

对提供 `entity_ids` 的 `PooledOLS`，单向 Breusch–Pagan LM 检验用于判断是否需要个体层面的随机误差成分：

$$
H_0:\sigma_a^2=0.
$$

较小的 p 值意味着简单的合并误差结构可能不足。对不平衡面板，statgpu 使用相应的 Baltagi–Li 形式。

```python
result = pooled.breusch_pagan_lm_test()
print(result.statistic, result.pvalue)
```

## Hausman 固定效应—随机效应检验

Hausman 检验比较固定效应与随机效应的系数估计。经典原假设下，随机效应估计量应当一致且更有效率；如果两种估计之间存在系统性差异，则随机效应设定可能与数据不相容。

$$
H=(\widehat\beta_{\mathrm{FE}}-\widehat\beta_{\mathrm{RE}})^\top
(V_{\mathrm{FE}}-V_{\mathrm{RE}})^+
(\widehat\beta_{\mathrm{FE}}-\widehat\beta_{\mathrm{RE}}).
$$

```python
result = fe.hausman_test(re)
# 或
result = re.hausman_test(fe)
```

当前 `hausman_test()` 实现的是经典的单向个体固定效应与随机效应比较。使用时需要满足：

- FE 与 RE 基于同一组对齐后的观测和解释变量；
- 两个模型都使用 `cov_type="nonrobust"`；
- 两个模型的系数向量都能够唯一识别；
- 协方差差矩阵允许定义经典 Hausman 二次型。

如果这些条件不满足，结果返回 `applicable=False` 并通过 `reason` 说明原因，而不是在同一个方法名下静默切换成另一种检验。

数值上令

$$
D=V_{\mathrm{FE}}-V_{\mathrm{RE}}.
$$

如果 $D$ 存在明显负特征值，经典二次型不适用。若 $D$ 是奇异的半正定矩阵，则只有当系数差位于 $\operatorname{range}(D)$ 中时，statgpu 才使用 Moore–Penrose 广义逆 $D^+$。

## `PanelTestResult`

常用字段包括：

- `statistic`：检验统计量；
- `pvalue`：p 值；
- 参考分布与自由度；
- 原假设与备择假设文本；
- `applicable`：当前检验是否适用；
- `reason`：不适用时的原因说明。

如果某项检验按照其统计定义无法计算，statgpu 会明确返回“不适用”或报出相应输入错误，不会用同一个接口返回另一种统计检验。

## 极端数值尺度

Pooling F、Breusch–Pagan LM 以及相关的经典 F 统计量在极端但仍可表示的数据尺度下会使用稳定的归一化与求和方式，以减少中间溢出或下溢造成的伪 `0`、`NaN` 或 `inf`。

这些数值保护只用于更稳定地计算同一个统计量，不会改变检验的原假设、备择假设或参考分布。

## 如何选择检验

- **比较合并回归与固定效应模型**：使用 Pooling F；
- **检查合并模型是否需要个体随机成分**：使用 Breusch–Pagan LM；
- **比较经典 FE 与 RE 估计是否相容**：使用 Hausman 检验。

这些检验回答的问题不同，不能仅根据 p 值大小相互替代。

## 相关文档

- [PanelOLS](panel-ols.md)
- [PooledOLS](pooled-ols.md)
- [RandomEffects](random-effects.md)
- [面板协方差](covariance.md)
- [面板拟合统计量](fit-statistics.md)

## 参考文献

- Hausman, J. A. (1978). Specification tests in econometrics. *Econometrica*, 46(6), 1251-1271.
- Breusch, T. S., & Pagan, A. R. (1980). The Lagrange multiplier test and its applications to model specification in econometrics. *The Review of Economic Studies*, 47(1), 239-253.
- Baltagi, B. H., & Li, Q. (1990). A Lagrange multiplier test for the error components model with incomplete panels. *Econometric Reviews*, 9(1), 103-107.
