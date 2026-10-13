# PooledOLS

> 语言：中文  
> 最后更新：2026-10-13  
> 页面定位：面板模型文档  
> 切换：[English](../../en/panel/pooled-ols.md)

<a id="概览"></a>

## 什么时候适合合并回归

当研究目标是让所有个体—时期观测共享一个截距和一组斜率时，可以使用 `PooledOLS`。它在堆叠后的全部观测上做普通最小二乘；如果共同的合并关系符合研究问题，它可以作为直接的基准模型。

合并回归不会消除个体效应或时间效应。通常的斜率解释要求：给定解释变量历史后，复合误差（包括未建模的个体差异）的条件均值为零。若个体的长期差异与解释变量相关，可考虑使用带个体效应的 [PanelOLS](panel-ols.md)。仅更换标准误不能解决条件均值模型的问题。

提供 `entity_ids` 不会改变系数，只会启用面板专属拟合统计量与 Breusch–Pagan LM 诊断，也不会自动启用聚类标准误。

模型写作 $y_{it}=\alpha+x_{it}^{\top}\beta+u_{it}$：$i$ 表示个体，$t$ 表示时期，$y_{it}$ 是响应，$x_{it}$ 是 `X` 中该行的解释变量向量。公共截距 $\alpha$ 和斜率 $\beta$ 由所有观测共享；$u_{it}$ 包含模型未显式控制的个体或时期差异。拟合在全部堆叠观测上最小化残差平方和 $\sum_{i,t}(y_{it}-\alpha-x_{it}^{\top}\beta)^2$。

## 准备堆叠后的观测

把面板整理成长表，每行对应一个个体在一个时期的观测：

- `X`：形状为 `(n_observations, n_features)`，只放解释变量，不要放截距列。`PooledOLS` 始终自行加入截距。
- `y`：形状为 `(n_observations,)`，与 `X` 的行顺序一致。
- 可选的 `entity_ids`：每行一个个体标签，用于组内/组间拟合统计量与 LM 诊断。
- 可选的 `time_index`：每行一个时期标签；Driscoll–Kraay 必须提供它，HAC 也可用它指定时间顺序。这个参数叫 `time_index`，不是 `time_ids`。

示例中 `X.shape == (60, 1)`，表示 12 个个体各有 5 次观测；数据具有公共截距 1、斜率 2 和独立噪声。

<a id="cpu-与-gpu-示例"></a>

## 拟合一个合并基准模型

<a id="cpu-example"></a>

<!-- example: cpu-basic -->
```python
import numpy as np
from statgpu.panel import PooledOLS

rng = np.random.default_rng(7)
entity_ids = np.repeat(np.arange(12), 5)
time_ids = np.tile(np.arange(5), 12)
x = rng.normal(size=60)
X = x[:, None]
y = 1.0 + 2.0 * x + 0.2 * rng.normal(size=60)

model = PooledOLS(device="cpu").fit(X, y, entity_ids=entity_ids)
print("coef:", np.round(model.coef_, 3))
print("standard errors:", np.round(model.bse_, 3))
print("nobs:", model.nobs)
```
<!-- example-end: cpu-basic -->

`coef_` 依次是截距和 `X` 唯一一列的斜率。估计斜率接近生成数据时设定的 2；`bse_` 按相同顺序给出标准误。`nobs=60` 表示所有堆叠后的观测行数。模拟结果接近设定值，并不能证明真实面板满足外生性条件。

在 GPU 上运行时，把 `device="cpu"` 换成 `device="cuda"`（CuPy/CUDA）或 `device="torch"`（Torch CUDA）。显式请求的后端不可用时会报错，不会切换到 CPU。

<a id="输出"></a>

## 解读系数、标准误与预测

`coef_[0]` 是公共截距，后续系数按 `X` 的列顺序排列。它们描述所有观测的合并关系，不是个体内效应。

`bse_` 按 `coef_` 的顺序衡量各系数的不确定性，并不是单次预测的波动大小。`conf_int_` 每行给出一个系数的区间下限与上限；`alpha=0.05` 对应 95% 区间。`tvalues_` 是系数与标准误之比：默认 `nonrobust` 拟合使用自由度为 `df_resid` 的 Student-t 参考分布，其他支持的协方差类型使用正态参考分布计算 p 值与区间。比较推断结果前，请核对[面板协方差](covariance.md)中的约定。

`rsquared` 是合并拟合的 $R^2$。`fit_statistics_` 包含标准化的拟合统计量；提供 `entity_ids` 后也包含组内/组间 $R^2$，定义见[拟合统计量](fit-statistics.md)。`nobs` 是原始行数，`df_resid` 是残差自由度。`summary()` 返回结构化面板摘要。

对新观测调用 `model.predict(X_new)` 时，会加入已拟合的公共截距并应用斜率。特征列及顺序应与拟合一致，不要加入常数列。结果为每行一个预测值的 NumPy 数组；不需要个体/时间标签，也不会加入个体专属效应。公式拟合后，也可向 `predict()` 传入包含所需解释变量列的 DataFrame。

<a id="协方差与推断"></a>

## 为面板选择标准误

默认 `cov_type="nonrobust"` 使用经典标准误。存在异方差时可选择 HC 类型；若需要按个体聚类，可设置 `cov_type="clustered"` 并传入 `cluster=entity_ids`。聚类也可以按误差结构选择其他分组；`group_debias=True` 可选地应用小簇数修正。

`cov_type` 只改变标准误的计算方式，不改变 OLS 系数估计。除经典协方差与 HC0–HC3 外，`PooledOLS` 还支持聚类稳健协方差，以及两种处理时间相关性的方式：

- `cov_type="hac"`：把观测看作一条有顺序的序列，并使用 Bartlett/Newey–West HAC。若提供 `time_index`，会先按时间顺序排序；否则使用输入数据的行顺序；
- `cov_type="driscoll-kraay"`：先按照 `time_index` 把观测分配到各时期，在时期内聚合协方差贡献，再对跨期滞后进行核加权。

因此，HAC 与 Driscoll–Kraay 不能互换解释。HAC 不会在个体边界重置滞后，也不会聚合同期多行；对堆叠面板排序不等于一般的面板 HAC。使用前请检查[单条有序序列的适用条件](covariance.md#ordered-sequence-hac)。

### 时间标签顺序

当协方差计算需要时间顺序时：

- 数值与 datetime 标签使用自然顺序；
- 有序 pandas categorical 使用用户声明的类别顺序；
- 普通字符串按字典序排序；
- 其他可比较对象按其自然比较顺序排序；
- 标签无法相互比较时会报错。

如果 `t1, t2, t10` 这类字符串不应按字典序解释，应改用数值/datetime 键，或显式使用有序 categorical。

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

<a id="数值行为与失败语义"></a>

## 使用结果前检查这些边界

确认截距只由 `PooledOLS` 自动加入，不要向 `X` 再添加一列常数。如果设计矩阵精确秩亏，拟合值仍可计算，但系数不唯一，因而系数级标准误、检验、p 值与置信区间不可用；见[面板协方差](covariance.md)。

所需元数据必须完整且对齐：聚类协方差需要 `cluster`，Driscoll–Kraay 需要 `time_index`，聚类数组形状不正确会报错。HAC 的时间标签含缺失值或非有限值时，也会报错，而不会猜测顺序。

新的 `fit()` 尝试会先失效旧结果。若重拟合失败，部分计算结果也会清除；之后 `predict()` 与 `summary()` 会报告模型尚未拟合。显式请求的 GPU 后端不可用时会报错。

极端响应尺度下，截距计算会保护大幅正负响应抵消后仍可表示的微小贡献；当直接减去响应均值可能溢出时，原有 `rsquared` 计算也会采用避免溢出的中心化。这些数值保护不改变最小二乘目标。

## 常见问题

**提供 `entity_ids` 会改变系数吗？**  
不会；它只额外启用面板专属的拟合统计量与诊断。

**`hac` 与 Driscoll–Kraay 相同吗？**  
不同。HAC 把观测当作一条有顺序的序列；Driscoll–Kraay 则先按时间时期聚合观测贡献，再计算跨期相关。

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

## 参数

| 参数 | 默认值 | 可选值 / 约束 | 含义 |
|---|---:|---|---|
| `cov_type` | `"nonrobust"` | `nonrobust`、`robust`/`hc1`、`hc0`、`hc2`、`hc3`、`clustered`、`hac`、`driscoll-kraay`/`dk`/`kernel` | 系数标准误的计算方式 |
| `alpha` | `0.05` | 有限且严格位于 `(0,1)` | 置信区间显著性水平；`0.05` 对应 95% 区间 |
| `bandwidth` | `None` | `None` 或非负整数 | HAC/DK 的滞后阶或平滑带宽；HAC 最多使用 $n-1$ 阶，DK 遵循[协方差文档](covariance.md)中的观测时期规则 |
| `kernel` | `"bartlett"` | HAC 只允许 Bartlett；DK 还支持 Parzen 与 QS 别名 | HAC/DK 使用的核函数 |
| `device` | `"auto"` | `auto` / `cpu` / `cuda` / `torch` | 数值计算的执行后端与设备 |
| `n_jobs` | `None` | 整数或 `None` | 共享并行参数 |
| `group_debias` | `False` | 布尔值；仅聚类协方差使用 | 是否应用小聚类数修正 |

### 拟合与其他方法

```python
model.fit(X=None, y=None, cluster=None, time_index=None,
          formula=None, data=None, entity_ids=None)
```

可传入数组，或配合 `data` 使用 `formula`；`fit()` 返回已拟合的估计器。个体/时间标签与观测一一对应；`cluster` 接受[面板协方差](covariance.md)说明的一维或二维分组。`entity_ids` 启用面板统计量；DK 必须提供 `time_index`，HAC 可选择提供。公式旁路元数据需要与保留观测对齐，见[本页公式接口](#公式接口)。

- `predict(X)`：返回 NumPy 数组形式的合并水平预测。
- `summary()`：返回结构化面板摘要。
- `breusch_pagan_lm_test()`：提供 `entity_ids` 后可调用随机效应 LM 诊断；须检查[适用条件](diagnostics.md)。
- `get_params(deep=True)` / `set_params(**params)`：读取或修改构造参数。`set_params()` 会清除拟合结果，修改后须重新拟合。

拟合属性：`coef_`、`bse_`、`tvalues_`、`pvalues_`、`conf_int_`、`rsquared`、`fit_statistics_`、`nobs`、`df_resid`、`rank_`（拟合设计矩阵的数值秩）。设计精确秩亏时，系数推断数组不可用。

继承的 `adjust_pvalues`、`combine_pvalues`、`bootstrap_statistic` 和 `permutation_test` 的完整签名见[共享估计器 API](../reference/estimator-api.md)。这些工具不会自动选择适用于面板依赖结构的重采样方案。

<a id="external-validation"></a>

<a id="comparing-with-other-packages"></a>

## 与其他软件包比较

与 `linearmodels` 7.0 比较时，应使用同一组堆叠观测并显式包含截距。Driscoll–Kraay 比较还需对齐时间标签、核、带宽与自由度约定；聚类稳健协方差需对齐聚类分组和 `group_debias`。即使 OLS 系数一致，这些选择仍会影响协方差和标准误。详见[协方差定义的比较](covariance.md#comparing-covariance-definitions)。

## 相关文档

- [面板模型总览](README.md) — Panel 模型选择与解释
- [面板模型协方差](covariance.md) — HC、cluster、HAC 与 Driscoll–Kraay 公式
- [面板诊断](diagnostics.md) — Breusch–Pagan LM 等诊断
- [设备与 GPU 内存](../guides/device-and-memory.md) — 后端与设备语义

## 参考文献

- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). The MIT Press.
