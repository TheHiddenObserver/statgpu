# PooledOLS

> 语言：中文  
> 最后更新：2026-09-17  
> 页面定位：面板模型文档  
> 切换：[English](../../en/panel/pooled-ols.md)

## 概览

`PooledOLS` 把所有面板观测直接堆叠起来，拟合一个具有公共截距和公共斜率的普通线性回归。它**不会**消除个体效应或时间效应，因此适用于本来就希望所有观测共享同一个条件均值关系的场景。

`entity_ids` 是可选的：提供它不会改变系数估计，但可以额外计算面板专属拟合统计量，并启用 Breusch–Pagan LM 诊断。

## 统计模型与识别

合并线性模型可以写成

$$
y_{it}=\alpha+x_{it}^{\top}\beta+u_{it}.
$$

若要把 $\beta$ 解释为合并条件均值中的斜率，一个常见的充分条件是

$$
E(u_{it}\mid X_i)=0,
$$

其中 $X_i=(x_{i1},\ldots,x_{iT_i})$。没有显式建模的个体或时间异质性都会并入复合误差项 $u_{it}$。

例如，如果真实数据满足

$$
y_{it}=\alpha+x_{it}^{\top}\beta+a_i+\varepsilon_{it},
$$

那么 pooled OLS 并不会消除 $a_i$。如果仍希望识别同一个结构参数 $\beta$，复合误差 $a_i+\varepsilon_{it}$ 必须与解释变量正交。若个体异质性与解释变量历史相关，PooledOLS 一般不会识别与固定效应估计量相同的斜率。

HC、聚类稳健、HAC 或 Driscoll–Kraay 等协方差选择只改变不确定性的计算方式，不能修复条件均值模型中外生性假设失效造成的识别问题。

## 估计量

令 $Z=[\mathbf1,X]$，则

$$
\widehat\beta_{\mathrm{pooled}}
=
\arg\min_\beta\|y-Z\beta\|_2^2
=
(Z^\top Z)^+Z^\top y.
$$

因此，系数估计就是把所有面板观测看作一个普通回归样本后得到的 OLS 结果。

## 协方差与推断

`cov_type` 只改变标准误的计算方式，不改变 OLS 系数估计。除经典协方差与 HC0–HC3 外，`PooledOLS` 还支持聚类稳健协方差，以及两种处理时间相关性的方式：

- `cov_type="hac"`：把观测看作一条有顺序的序列，并使用 Bartlett/Newey–West HAC。若提供 `time_index`，会先按时间顺序排序；否则使用输入数据的行顺序；
- `cov_type="driscoll-kraay"`：先按照 `time_index` 把观测分配到各时期，在时期内聚合协方差贡献，再对跨期滞后进行核加权。

因此，HAC 与 Driscoll–Kraay 不能互换解释。完整公式见 [面板模型协方差](covariance.md)。

### 时间标签顺序

当协方差计算需要时间顺序时：

- 数值与 datetime 标签使用自然顺序；
- 有序 pandas categorical 使用用户声明的类别顺序；
- 普通字符串按字典序排序；
- 其他可比较对象按其自然比较顺序排序；
- 标签无法相互比较时会报错。

如果 `t1, t2, t10` 这类字符串不应按字典序解释，应改用数值/datetime 键，或显式使用有序 categorical。

## 参数

| 参数 | 默认值 | 可选值 / 约束 | 含义 |
|---|---:|---|---|
| `cov_type` | `"nonrobust"` | `nonrobust`、`robust`/`hc1`、`hc0`、`hc2`、`hc3`、`clustered`、`hac`、`driscoll-kraay`/`dk`/`kernel` | 系数标准误的计算方式 |
| `alpha` | `0.05` | 有限且严格位于 `(0,1)` | 置信区间显著性水平；`0.05` 对应 95% 区间 |
| `bandwidth` | `None` | `None` 或非负整数 | HAC/DK 的滞后阶或平滑带宽 |
| `kernel` | `"bartlett"` | HAC 只允许 Bartlett；DK 还支持 Parzen 与 QS 别名 | HAC/DK 使用的核函数 |
| `device` | `"auto"` | `auto` / `cpu` / `cuda` / `torch` | 数值计算的执行后端与设备 |
| `n_jobs` | `None` | 整数或 `None` | 共享并行参数 |
| `group_debias` | `False` | 布尔值；仅聚类协方差使用 | 是否应用小聚类数修正 |

拟合接口：

```python
model.fit(X, y, cluster=None, time_index=None, entity_ids=None)
```

- 使用聚类稳健协方差时传入 `cluster`；
- 使用 Driscoll–Kraay 时必须传入 `time_index`；
- 对 HAC 而言，`time_index` 可选；提供后，它决定 HAC 使用的观测顺序；
- 如果还希望得到标准化的组内/组间 $R^2$ 或 Breusch–Pagan LM 检验，可以提供 `entity_ids`。

## CPU 与 GPU 示例

```python
from statgpu.panel import PooledOLS

cpu = PooledOLS(device="cpu").fit(X, y)
cuda = PooledOLS(device="cuda").fit(X, y)
torch = PooledOLS(device="torch").fit(X, y)
```

若显式请求的 GPU 后端不可用，`.fit()` 会直接报错，而不会静默切换到 CPU。

## 公式接口

假设 `df` 包含 `y`、`x1` 与 `x2`：

```python
from statgpu.panel import PooledOLS

model = PooledOLS().fit(
    formula="y ~ x1 + x2",
    data=df,
)
```

`PooledOLS` 始终包含截距，因此显式无截距公式会被拒绝。

## 输出

常用结果包括：

- `coef_`、`bse_`、`tvalues_`、`pvalues_`、`conf_int_`；
- `rsquared`、`fit_statistics_`；
- `nobs`、`df_resid`；
- 提供 `entity_ids` 后可使用 `breusch_pagan_lm_test()`，见 [面板诊断](diagnostics.md)。

## 数值行为与失败语义

用户可以依赖以下公开行为：

- 自动加入的截距会使用稳定的数值处理，避免极端响应尺度下可表示的低阶截距贡献被静默抹除；
- 聚类稳健协方差缺少 `cluster`、Driscoll–Kraay 缺少 `time_index`，或聚类数组长度/形状不匹配时，会直接报错，而不会自动切换到另一种协方差；
- HAC 的时间元数据含缺失值或非有限值时会报错，不会猜测排序；
- 每次新的 `fit()` 都会先失效上一轮的已拟合/推断状态。如果重拟合在后续任意阶段失败，对象保持未拟合状态；
- 如果设计矩阵精确秩亏，拟合值仍可能有意义，但系数向量不唯一。此时不会继续发布依赖唯一系数表示的标准误、检验、p 值与置信区间；
- 显式 `device="cuda"` 或 `device="torch"` 要求对应后端可用，否则会报错而不是切换到 CPU。

这些公开行为不要求用户依赖内部具体使用哪一种 SVD/BLAS 归约、数值缩放或验证脚本。

## 常见问题

**提供 `entity_ids` 会改变系数吗？**  
不会；它只额外启用面板专属的拟合统计量与诊断。

**`hac` 与 Driscoll–Kraay 相同吗？**  
不同。HAC 把观测当作一条有顺序的序列；Driscoll–Kraay 则先按时间时期聚合观测贡献，再计算跨期相关。

## 相关文档

- [面板模型总览](../models/panel.md) — Panel 模型选择与解释
- [面板模型协方差](covariance.md) — HC、cluster、HAC 与 Driscoll–Kraay 公式
- [面板诊断](diagnostics.md) — Breusch–Pagan LM 等诊断
- [设备与 GPU 内存](../guides/device-and-memory.md) — 后端与设备语义

## 参考文献

- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). The MIT Press.
