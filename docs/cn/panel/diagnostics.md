# 面板模型诊断

> 语言：中文  
> 最后更新：2026-10-13  
> 切换：[English](../../en/panel/diagnostics.md)

## 概述

面板诊断用于回答几类不同的模型比较问题，例如：是否需要固定效应、合并回归的误差结构是否足够，以及随机效应设定是否与固定效应估计相容。

每个检验都返回 `PanelTestResult`，其中包含检验统计量、p 值、参考分布、自由度、原假设与备择假设，以及该检验当前是否适用。

## 如何选择检验

- **比较合并回归与固定效应模型**：使用 Pooling F；
- **检查合并模型是否需要个体随机成分**：使用 Breusch–Pagan LM；
- **比较经典 FE 与 RE 估计是否相容**：使用 Hausman 检验。

这些检验回答的问题不同，不能仅根据 p 值大小相互替代。

这些检验不是自动选出正确模型的顺序流程。未拒绝原假设不能证明模型假设成立。Hausman 尤其需要同样本、同系数设计、可唯一识别且采用非稳健协方差的单向个体 FE 与 RE 拟合。

## 在同一组数据上运行比较

本例让三种模型使用相同的 30 个个体，每个个体有 8 条观测。FE 与 RE 都使用同一列斜率解释变量，不包含水平常数列；合并 OLS 自行添加截距。这组模拟数据用于示范怎样阅读检验，不替代真实研究中的假设判断。

<!-- example: cpu-diagnostics -->
```python
import numpy as np
from statgpu.panel import PanelOLS, PooledOLS, RandomEffects

rng = np.random.default_rng(17)
entity_ids = np.repeat(np.arange(30), 8)
X = rng.normal(size=(240, 1))
y = (2.0 * X[:, 0] + np.repeat(rng.normal(size=30), 8)
     + rng.normal(scale=0.4, size=240))

fe = PanelOLS(entity_effects=True, cov_type="nonrobust", device="cpu").fit(
    X, y, entity_ids=entity_ids
)
pooled = PooledOLS(device="cpu").fit(X, y, entity_ids=entity_ids)
re = RandomEffects(cov_type="nonrobust", device="cpu").fit(
    X, y, entity_ids=entity_ids
)
tests = {
    "pooling F": fe.pooling_f_test(),
    "Breusch-Pagan LM": pooled.breusch_pagan_lm_test(),
    "Hausman": fe.hausman_test(re),
}
for name, result in tests.items():
    if result.applicable:
        print(name, result.statistic, result.pvalue, result.null)
    else:
        print(name, "not applicable:", result.reason)
```
<!-- example-end: cpu-diagnostics -->

Pooling F 约为 40.26，LM 约为 564.32；在这组模拟数据中，两者都强烈反对各自的合并模型原假设。Hausman 则**不适用**，因为估计出的协方差差矩阵不是半正定矩阵。这不支持也不反对 RE，应报告原因，而不是编造或解释一个 p 值。[固定效应](panel-ols.md#cpu-example)、[合并 OLS](pooled-ols.md#cpu-example)和[随机效应](random-effects.md#cpu-example)页进一步解释各自的拟合结果。

## `PanelTestResult`

先看 `result.applicable`。若为假，应读 `result.reason`，不要解释缺失的 p 值；适用时，再将原假设、参考分布、统计量和 p 值一起阅读。

常用字段包括：

- `statistic`：检验统计量；
- `pvalue`：p 值；
- 参考分布与自由度；
- 原假设与备择假设文本；
- `applicable`：当前检验是否适用；
- `reason`：不适用时的原因说明。

如果某项检验按照其统计定义无法计算，statgpu 会明确返回“不适用”或报出相应输入错误，不会用同一个接口返回另一种统计检验。

## Pooling F 检验

Pooling F 检验用于判断 `PanelOLS` 中加入的个体和/或时间效应，相对含公共截距的合并模型是否具有额外解释力。原假设是相对这一公共水平不存在额外的效应差异；在通常的识别约束下，等价于额外固定效应系数联合为 0。未拒绝原假设表示该检验未发现反对合并模型的证据，并不证明合并模型正确。

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

## 极端数值尺度

Pooling F、Breusch–Pagan LM 以及相关的经典 F 统计量在极端但仍可表示的数据尺度下会使用稳定的归一化与求和方式，以减少中间溢出或下溢造成的伪 `0`、`NaN` 或 `inf`。

这些数值保护只用于更稳定地计算同一个统计量，不会改变检验的原假设、备择假设或参考分布。公开的 RSS 元数据使用原始平方单位，若该平方量本身超出 float64 范围，可能为 `inf`；这并不意味着尺度不变的检验统计量也为无穷。

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
