# PanelOLS

> 语言：中文  
> 最后更新：2026-09-17  
> 切换：[English](../../en/panel/panel-ols.md)

## 概述

`PanelOLS` 可以拟合无固定效应、单向固定效应或双向固定效应的线性面板回归。加入固定效应后，系数由去除相应个体效应和/或时间效应之后仍然存在的解释变量变化来识别。

statgpu 直接对数据执行去均值等变换，而不是显式生成大量虚拟变量。这改变的是数值实现方式，不改变固定效应模型本身的统计含义。

## 统计模型与识别

含个体和时间固定效应的经典模型可以写为

$$
y_{it}=x_{it}^{\top}\beta+a_i+\gamma_t+\varepsilon_{it}.
$$

其中 $a_i$ 与 $\gamma_t$ 是固定但未知的干扰参数，不要求来自某个概率分布。对静态线性面板模型，一个常见的充分外生性条件是

$$
E\!\left(\varepsilon_{it}\mid X_i,a_i,\gamma_1,\ldots,\gamma_T\right)=0.
$$

因此，核心限制落在个体特异误差 $\varepsilon_{it}$ 上，而不是要求固定效应与解释变量独立。

$\beta$ 只能由去除固定效应后仍然存在的解释变量变化识别。例如，仅包含个体固定效应时，一个在个体内部完全不随时间变化的解释变量会被固定效应吸收，无法单独识别其斜率。

当 `entity_effects=False` 且 `time_effects=False` 时，`PanelOLS` 退化为普通水平回归，此时不再使用上述固定效应解释。

## 估计量

令 $F$ 表示纳入模型的固定效应设计矩阵，并定义

$$
M_F=I-F(F^\top F)^+F^\top.
$$

则

$$
\widehat\beta_{\mathrm{FE}}
=\arg\min_\beta\|M_F(y-X\beta)\|_2^2
=(X^\top M_FX)^+X^\top M_Fy.
$$

仅有个体固定效应时，这就是通常的个体内去均值。双向固定效应下，statgpu 交替去除个体均值和时间均值，直到满足 `demean_tol`。如果在 `demean_max_iter` 次迭代内仍未收敛，`.fit()` 会直接报错，而不会返回尚未充分去均值的近似结果。

## 协方差与统计推断

标准误使用与系数估计完全相同的变换后解释变量和残差。记

$$
Z=M_FX,
\qquad
e=M_F(y-X\widehat\beta_{\mathrm{FE}}),
$$

则 `nonrobust`、HC、聚类稳健和 Driscoll–Kraay 协方差的统一定义见 [面板协方差](covariance.md)。

Driscoll–Kraay 和残差自由度都需要计入固定效应占用的维度。固定效应秩为

$$
r_F=\begin{cases}
N,&\text{仅个体固定效应},\\
T,&\text{仅时间固定效应},\\
N+T-C,&\text{双向固定效应},
\end{cases}
$$

其中 $C$ 是已观测个体—时间关联图的连通分量数。残差自由度使用

$$
df_{\mathrm{resid}}
=n-\operatorname{rank}(Z)-r_F.
$$

这一自由度同时用于公开的 `df_resid`、同方差残差方差尺度、HC1 修正以及相应的 Student-t 推断。

如果去除固定效应后的设计矩阵精确秩亏，拟合值仍可能唯一，但系数向量不唯一。此时 statgpu 保留拟合结果，同时不发布系数级标准误、检验、p 值和置信区间。

## 参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `entity_effects` | `False` | 是否去除个体固定效应 |
| `time_effects` | `False` | 是否去除时间固定效应 |
| `cov_type` | `"nonrobust"` | `nonrobust`、`robust`/`hc1`、`hc0`、`hc2`、`hc3`、`clustered`、`driscoll-kraay`/`dk`/`kernel` |
| `alpha` | `0.05` | 置信区间显著性水平 |
| `device` | `"auto"` | `auto`、`cpu`、`cuda`、`torch` |
| `n_jobs` | `None` | 适用路径的并行任务数 |
| `bandwidth` | `None` | Driscoll–Kraay 带宽；`None` 使用自动规则 |
| `kernel` | `"bartlett"` | Driscoll–Kraay 核函数 |
| `group_debias` | `False` | 聚类稳健协方差是否使用小簇数修正 |
| `demean_max_iter` | `1_000_000` | 双向去均值的最大迭代次数 |
| `demean_tol` | `1e-10` | 双向去均值的收敛容差 |

打开相应固定效应后，需要提供对应的 `entity_ids` 和/或 `time_ids`：

```python
model.fit(
    X,
    y,
    entity_ids=entity_ids,
    time_ids=time_ids,
    cluster=cluster,
)
```

聚类稳健协方差还需要 `cluster`；Driscoll–Kraay 需要时间索引。

## 使用示例

### 数组接口

```python
from statgpu.panel import PanelOLS

model = PanelOLS(
    entity_effects=True,
    device="cpu",
).fit(X, y, entity_ids=entity_ids)
```

`device="cuda"` 需要 CuPy/CUDA，`device="torch"` 需要 Torch CUDA。显式请求的后端不可用时直接报错，不会自动改用 CPU。

### 公式接口

```python
from statgpu.panel import PanelOLS

# 双向固定效应：pipe 语法
two_way = PanelOLS().fit(
    formula="y ~ x1 + x2 | entity + time",
    data=df,
)

# 等价的固定效应标记写法
two_way_tokens = PanelOLS().fit(
    formula="y ~ x1 + x2 + EntityEffects + TimeEffects",
    data=df,
)

# 无固定效应、无截距的水平回归
level_no_intercept = PanelOLS().fit(
    formula="y ~ 0 + x1 + x2",
    data=df,
)
```

固定效应可以使用 pipe 语法，也可以使用 `EntityEffects` / `TimeEffects` 等固定效应标记，但同一个公式中不能混用。如果同时从公式和显式参数提供个体/时间标签，两者必须在公式筛选后的样本上完全一致。

## 输出与解释

常用结果包括：

- `coef_`：固定效应变换后回归的斜率系数；
- `bse_`、`tvalues_`、`pvalues_`、`conf_int_`：可用时的系数推断；
- `rsquared_within`、`fit_statistics_`：拟合优度统计量；
- `nobs`、`df_resid`：样本量和残差自由度；
- `summary()`：面板回归摘要。

Pooling F 检验和 Hausman 检验见 [面板诊断](diagnostics.md)。

## 预测与数值边界

双向固定效应预测时，只有当请求的个体/时间标签能够由已拟合数据中的固定效应唯一确定时，才会加入相应固定效应。若标签组合与已拟合的效应结构不兼容，会直接报错；若两个标签都从未出现，则预测只使用线性部分 $X\widehat\beta$。

公式接口会对不支持的结构明确报错，例如混用两种固定效应语法、在 pipe 中指定超过两个固定效应变量，或固定效应模型中没有任何非截距解释变量。

## 常见问题

**为什么双向去均值不能在未收敛时直接返回最后一次迭代？**  
因为未充分去均值会改变实际拟合的回归问题。只有达到 `demean_tol` 后结果才会发布。

**使用稳健协方差后，`fit_statistics_.f_statistic` 会自动变成稳健 Wald 检验吗？**  
不会。该字段仍表示经典的斜率联合检验；见 [面板拟合统计量](fit-statistics.md)。

**固定效应会改变斜率的识别来源吗？**  
会。系数只由去除相应固定效应后仍存在的解释变量变化识别。

## 相关文档

- [面板模型总览](../models/panel.md)
- [面板协方差](covariance.md)
- [面板诊断](diagnostics.md)
- [面板拟合统计量](fit-statistics.md)

## 参考文献

- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). The MIT Press.
