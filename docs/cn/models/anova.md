# ANOVA

> 语言：中文  
> 最后更新：2026-07-24  
> 切换：[English](../../../en/models/anova.md)

## 概览

ANOVA 模块提供：

- `f_oneway`
- `f_twoway`
- `f_welch`
- `tukey_hsd`
- `bonferroni`
- `cohens_f`
- `partial_eta_squared`

组内归约支持 NumPy、CuPy 与 Torch。若所选 GPU 后端缺少需要的分布函数，
会在完成后端原生充分统计量计算后，仅对标量进行 CPU 分布求值。

## 单因素 ANOVA

对于组大小 $n_i$ 与组均值 $\bar y_i$，总体均值为

$$
\bar y = \frac{\sum_i n_i\bar y_i}{\sum_i n_i}.
$$

F 统计量为

$$
F = \frac{SSB/(k-1)}{SSW/(N-k)},
$$

其中

$$
SSB = \sum_i n_i(\bar y_i-\bar y)^2,
\qquad
SSW = \sum_i\sum_j(y_{ij}-\bar y_i)^2.
$$

`AnovaResult` 返回 `statistic`、`pvalue`、`df_between`、`df_within` 与
`eta_squared`。

## 双因素、Welch 与事后检验

- `f_twoway` 支持平衡双因素设计，可使用完整交互模型或加性模型。非平衡设计在
  Type I/II/III 平方和语义得到明确支持前会报错。
- `f_welch` 用于异方差组，并保留 Welch–Satterthwaite 小数分母自由度。
- `tukey_hsd` 使用 studentized-range 分布。
- `bonferroni` 执行 Bonferroni 校正的两两 Welch 检验。

## 参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `*groups` | 必填 | 两个或更多一维样本 |
| `backend` | `"auto"` | `"auto"`、`"numpy"`、`"cupy"` 或 `"torch"` |

函数特有参数见公开 API docstring。

## 示例

```python
import numpy as np
from statgpu.anova import f_oneway

g1 = np.random.randn(100)
g2 = np.random.randn(100) + 0.5
result = f_oneway(g1, g2, backend="numpy")
print(result.statistic, result.pvalue, result.eta_squared)
```

```python
import cupy as cp
from statgpu.anova import f_oneway

g1 = cp.random.randn(100)
g2 = cp.random.randn(100) + 0.5
result = f_oneway(g1, g2, backend="cupy")
```

```python
import torch
from statgpu.anova import f_oneway

g1 = torch.randn(100, device="cuda", dtype=torch.float64)
g2 = torch.randn(100, device="cuda", dtype=torch.float64) + 0.5
result = f_oneway(g1, g2, backend="torch")
```

## 执行边界

均值、方差、平方和及组内归约保留在所选后端。只有在 CuPy 或 Torch 缺少所需
函数时，studentized-range、t、normal 或 F 分布的标量求值才可能转到 CPU；
不会仅为计算 p 值而把完整组数组传回 NumPy。

## 验证说明

本页不维护全局 GPU 完成标记。验证证据应限定到维护测试或硬件特定 artifact
所记录的具体函数、后端、硬件与 commit。

## 参考文献

- Fisher, R. A. (1925). *Statistical Methods for Research Workers*.
- Welch, B. L. (1951). On the comparison of several mean values.
- Tukey, J. W. (1949). Comparing individual means in the analysis of variance.
