# ANOVA

> 语言：中文  
> 最后更新：2026-07-24  
> 切换：[English](../../en/models/anova.md)

## 概览

ANOVA 模块提供单因素 ANOVA、平衡双因素 ANOVA、Welch ANOVA、Tukey HSD、
Bonferroni 校正的两两 Welch 检验以及效应量辅助函数。分组归约支持
NumPy、CuPy 和 Torch 后端。

## 路径

- `statgpu.anova.f_oneway`、`statgpu.anova.AnovaResult`
- `statgpu.anova.f_twoway`、`statgpu.anova.TwoWayAnovaResult`
- `statgpu.anova.f_welch`
- `statgpu.anova.tukey_hsd`、`statgpu.anova.TukeyResult`
- `statgpu.anova.bonferroni`、`statgpu.anova.PosthocResult`
- `statgpu.anova.cohens_f`
- `statgpu.anova.partial_eta_squared`

## 单因素 ANOVA

设第 $i$ 组样本量为 $n_i$、均值为 $\bar y_i$，总样本量
$N=\sum_i n_i$，总体均值为

$$
\bar y = \frac{\sum_i n_i\bar y_i}{N}.
$$

组间平方和与组内平方和分别为

$$
SSB = \sum_i n_i(\bar y_i-\bar y)^2,
\qquad
SSW = \sum_i\sum_j(y_{ij}-\bar y_i)^2.
$$

检验统计量为

$$
F = \frac{SSB/(k-1)}{SSW/(N-k)}.
$$

`f_oneway` 通过后端原生归约直接计算这些量，不需要迭代求解器。p 值来自
F 分布生存函数。Eta-squared 为

$$
\eta^2 = \frac{SSB}{SSB+SSW}.
$$

### 参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `*groups` | 必填 | 至少两个一维样本 |
| `backend` | `"auto"` | `"auto"`、`"numpy"`、`"cupy"` 或 `"torch"` |
| `dtype` | `None` | 函数公开该参数时使用的计算 dtype |

### 输出

`AnovaResult` 包含：

| 字段 | 说明 |
|---|---|
| `statistic` | F 统计量 |
| `pvalue` | F 分布尾概率 |
| `df_between` | 分子自由度 |
| `df_within` | 分母自由度 |
| `eta_squared` | 单因素效应量 |

## 双因素 ANOVA

`f_twoway` 用于平衡双因素设计，可检验因子 A、因子 B 以及可选的交互项。
在公共 API 明确 Type I、II 或 III 平方和约定之前，不平衡单元格会被拒绝。
当 `interaction=False` 时，使用加性模型，剩余交互变异进入残差项。

### 参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `data` | 必填 | 包含观测值的 `(a, b)` 嵌套单元格 |
| `interaction` | `True` | 是否拟合和检验交互项 |
| `backend` | `"auto"` | 数值后端 |
| `dtype` | `None` | 计算 dtype |

### 输出

`TwoWayAnovaResult` 给出因子 A、因子 B 和可选交互项的统计量、p 值、自由度、
eta-squared，以及残差自由度和残差平方和。

## Welch ANOVA

`f_welch` 是允许组间方差不等的单因素检验。分母自由度使用
Welch-Satterthwaite 公式，通常为小数，因此返回的
`AnovaResult.df_within` 是浮点数。普通合并方差 ANOVA 的 eta-squared 并非
相应的 Welch 估计目标，所以 `eta_squared` 返回 `NaN`。

## 事后比较

### Tukey HSD

`tukey_hsd` 使用 studentized-range 分布进行全部均值两两比较，控制族错误率，
并返回同时置信区间。`TukeyResult` 包含比较列表、显著性水平、组数、残差自由度
和合并均方误差。每项比较包含组索引、均值差、校正 p 值、置信区间和拒绝结论。

### Bonferroni 两两 Welch 检验

`bonferroni` 对每一对组执行 Welch t 检验并进行 Bonferroni 校正，不要求等方差。
`PosthocResult` 给出全部两两比较、族显著性水平和比较数量。

## 效应量

- `partial_eta_squared(ss_effect, ss_error)` 计算
  $ss_{effect}/(ss_{effect}+ss_{error})$，并验证平方和有限且非负。
- `cohens_f(*groups)` 根据 eta-squared 计算 Cohen's $f$：

$$
f = \sqrt{\frac{\eta^2}{1-\eta^2}}.
$$

## CPU 与 GPU 示例

### NumPy

```python
import numpy as np
from statgpu.anova import f_oneway, f_welch, tukey_hsd

rng = np.random.default_rng(7)
g1 = rng.normal(0.0, 1.0, 100)
g2 = rng.normal(0.5, 1.0, 100)
g3 = rng.normal(-0.2, 2.0, 80)

result = f_oneway(g1, g2, backend="numpy")
welch = f_welch(g1, g2, g3, backend="numpy")
posthoc = tukey_hsd(g1, g2, alpha=0.05, backend="numpy")
```

### CuPy

```python
import cupy as cp
from statgpu.anova import f_oneway

rng = cp.random.RandomState(7)
g1 = rng.standard_normal(100, dtype=cp.float64)
g2 = rng.standard_normal(100, dtype=cp.float64) + 0.5
result = f_oneway(g1, g2, backend="cupy")
```

### Torch CUDA

```python
import torch
from statgpu.anova import f_oneway

torch_device = torch.device("cuda")
g1 = torch.randn(100, device=torch_device, dtype=torch.float64)
g2 = torch.randn(100, device=torch_device, dtype=torch.float64) + 0.5
result = f_oneway(g1, g2, backend="torch")
```

## 后端与执行边界

均值、方差、平方和和分组归约保留在所选后端。若 GPU 后端没有所需函数，
F、t、正态或 studentized-range 分布的最终标量计算可能跨到 CPU。不会仅为了
计算 p 值而把完整分组向量转移到 NumPy。

`backend="cupy"` 选择 CuPy，`backend="torch"` 选择 Torch。显式后端请求不会
静默切换到其他后端。

## strict 与 approximate

ANOVA 函数没有独立的 strict/approximate 统计模式，所有后端使用相同的检验定义。
CPU 标量分布调用只是执行边界，不是另一套近似 ANOVA 公式。

## 限制与失败行为

- 单因素和 Welch 检验至少需要两个非空组。
- 双因素 ANOVA 当前要求平衡单元格。
- 维护中的公共验证路径会拒绝非有限观测值。
- Tukey HSD 依赖 studentized-range 分布，可能使用 CPU 标量实现。
- 效应量辅助函数对非法平方和显式报错，而不是返回误导性的有限结果。

## 外部验证

维护测试将 Welch ANOVA 与 `statsmodels.stats.oneway.anova_oneway` 对齐，并覆盖
NumPy/Torch 一致性、自由度语义、平衡设计限制、效应量验证和后端执行边界。
所有验证结论仅适用于记录中的具体函数、后端、环境和 commit。

## FAQ

### Torch 输入是否必须指定 `backend="torch"`？

显式 Torch 执行应使用 `backend="torch"`。`"auto"` 可以根据输入类型推断，
但测试和 benchmark 中推荐显式指定。

### 为什么返回的 p 值可能是 Python 标量？

ANOVA 结果对象将统计摘要暴露为标量。用于获得这些摘要的充分统计量会一直保留
在所选后端，直到最终标量分布边界。

### 为什么拒绝不平衡双因素设计？

不平衡设计中的不同平方和约定检验不同假设。实现选择显式失败，而不是静默采用
某一种约定。

## 参考文献

- Fisher, R. A. (1925). *Statistical Methods for Research Workers*.
- Welch, B. L. (1951). On the comparison of several mean values.
- Tukey, J. W. (1949). Comparing individual means in the analysis of variance.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*.
