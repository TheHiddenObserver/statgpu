# BetweenOLS

> 语言：中文  
> 最后更新：2026-10-13  
> 切换：[English](../../en/panel/between-ols.md)

<a id="概述"></a>

## 何时比较个体的长期均值

`BetweenOLS` 先在每个个体内部对响应和解释变量取时间均值，把面板数据压缩为“每个个体一行”，再对这些个体均值进行带截距的 OLS 回归。它关注的是**不同个体长期平均水平之间的关系**，而不是同一个体随时间的变化。

因此，最终回归的有效样本量是进入均值回归的个体数量，而不是原始面板数据的总行数。

适合比较企业、个人或地区的长期平均水平。若你希望消除个体间不随时间变化的差异，应考虑[固定效应](panel-ols.md)或[一阶差分](first-difference-ols.md)。取均值并不会消除个体效应；若要把斜率解释为共同的结构效应，还需个体效应与平均误差的合成项和解释变量的个体均值正交。

## 准备面板观测行

采用长表格式，每行是一条个体—时期观测。`X` 的形状为 `(n, p)`，`y` 为 `(n,)`，`entity_ids` 的长度为 `n`，三者按同一行序对齐。一维 `X` 也可作为单个解释变量。ID 用于标识观测属于哪个个体，不是解释变量列。数组中的数值必须有限。

传入原始观测即可，不必预先取均值。估计器按个体求均值并自动加截距，所以不要给 `X` 添加常数列。各个体的观测次数可以不同；最终每个保留个体仍只贡献一个等权均值。该拟合不需要 `time_ids`。

<a id="使用示例"></a>
<a id="数组接口"></a>

## 先在 CPU 上拟合一个模型

每行是一个个体—时期观测，共 12 个个体，每个个体有 5 个时期。这里 `x` 的均值在个体之间变化，响应中没有遗漏的个体效应。

<a id="cpu-example"></a>

<!-- example: cpu-basic -->
```python
import numpy as np
from statgpu.panel import BetweenOLS

rng = np.random.default_rng(7)
entity_ids = np.repeat(np.arange(12), 5)
time_ids = np.tile(np.arange(5), 12)
x = np.repeat(np.linspace(-2, 2, 12), 5) + rng.normal(size=60)
X = x[:, None]
y = 1.0 + 2.0 * x + 0.2 * rng.normal(size=60)

model = BetweenOLS(device="cpu").fit(X, y, entity_ids=entity_ids)
print("coef:", np.round(model.coef_, 3))
print("nobs, df_resid:", model.nobs, model.df_resid)
```
<!-- example-end: cpu-basic -->

<a id="输出与解释"></a>

## 理解均值回归并进行预测

`coef_` 依次给出截距和 `x` 个体均值的斜率，分别接近 1 和 2。输入虽有 60 行，`nobs=12` 与 `df_resid=10` 对应的却是 12 个个体均值和两个拟合系数。若真实数据包含遗漏的个体效应，取均值并不会消除它。

常用结果包括：

- `coef_`：个体均值回归的系数；
- `bse_`、`tvalues_`、`pvalues_`、`conf_int_`：可用时的系数推断；
- `rsquared`、`fit_statistics_`：拟合优度统计量；
- `nobs`：最终进入回归的个体均值观测数；
- `df_resid`：残差自由度。

`predict(X_new)` 对每行计算截距加 `X_new @ coef_[1:]`，返回 NumPy 数组；它不会对新数据自动按个体取均值。预测个体均值时，应自行为每个个体提供一行解释变量均值。输入个体—时期水平行时，结果仍使用个体间斜率，不包含单独估计的个体效应。

`bse_` 是系数标准误，`conf_int_` 每行为相应系数的区间下限和上限。`nonrobust` 使用自由度为 `df_resid` 的 Student-t 推断；HC 选项使用正态参考分布，此时 `tvalues_` 实际存放 z 统计量。用 `print(model.summary())` 可同时查看估计与不确定性。公开系数和推断数组均为 NumPy 数组。标准化的组内/组间/总体指标见[拟合统计量](fit-statistics.md)，不应假定它们等于变换后回归的 `rsquared`。

<a id="协方差与统计推断"></a>

## 选择个体均值回归的标准误

标准误直接基于个体均值回归计算。当前支持：

- `cov_type="nonrobust"`：通常的同方差 OLS 协方差；
- `robust` / `hc1`：HC1 异方差稳健协方差；
- `hc0`、`hc2`、`hc3`：相应的 HC 稳健协方差。

这些选项只改变个体均值回归的不确定性估计，不能替代把斜率解释为基础面板模型结构参数时所需的识别条件。统一公式见 [面板协方差](covariance.md)。

## 检查样本和数值限制

最终均值回归需要正的残差自由度，即个体均值数大于设计矩阵的秩。

极端响应尺度和抵消会采用数值保护，以保留可表示的截距贡献，但不改变个体均值回归目标。原有 `rsquared` 使用避免溢出的中心化计算，定义不变；这些计算仍受 float64 精度限制。

若个体均值设计矩阵精确秩亏，仍可通过最小二乘解计算拟合值，但系数向量不唯一。此时系数级标准误、检验、p 值和置信区间不可用，而不会根据任意一个系数表示计算它们。

无效协方差选项或不可用的显式 GPU 后端会报错，不会静默换成其他选项。

<a id="公式接口"></a>

## 从数据框列拟合

假设 `df` 已包含下面公式及分组参数中使用的列。

```python
from statgpu.panel import BetweenOLS

model = BetweenOLS().fit(
    formula="y ~ x1 + x2",
    data=df,
    entity_ids=df["entity"],
)
```

`BetweenOLS` 始终包含截距，因此显式无截距公式会被拒绝。

## 在 GPU 上运行同一个模型

在 GPU 上运行时，把 `device="cpu"` 换成 `device="cuda"`（CuPy/CUDA）或 `device="torch"`（Torch CUDA）。显式请求的后端不可用时会报错，不会切换到 CPU。

<a id="统计模型与识别"></a>

## 为什么取均值还需要额外假设

可以从单向个体效应模型出发：

$$
y_{it}=\alpha+x_{it}^{\top}\beta+a_i+\varepsilon_{it}.
$$

在个体内部对时间取平均后得到

$$
\bar y_i=\alpha+\bar x_i^{\top}\beta+a_i+\bar\varepsilon_i.
$$

如果把 $a_i$ 视为不受限制的固定参数，取时间均值并不会消除它。因此，仅由固定效应模型本身，不能推出 BetweenOLS 的斜率一定等于原始结构参数 $\beta$。

若希望把个体间回归的斜率解释为基础模型中的同一个 $\beta$，还需要额外的跨个体矩条件。若把 $(a_i,\bar x_i,\bar\varepsilon_i)$ 视为在个体总体中随机变化，一个常见的充分条件是

$$
E\!\left(a_i+\bar\varepsilon_i\mid \bar x_i\right)=0.
$$

如果只关心线性投影参数，也可以直接施加相应矩条件，例如处理截距后要求 $E[\bar x_i(a_i+\bar\varepsilon_i)]=0$。在这类额外限制下，均值回归才能恢复基础面板方程中的同一个 $\beta$。若长期个体异质性与平均解释变量相关，`BetweenOLS` 仍然定义了 $\bar y_i$ 对 $\bar x_i$ 的横截面线性投影，但该投影通常不会与固定效应或一阶差分估计的斜率相同。

<a id="估计量"></a>

## 个体均值的最小二乘问题

对个体 $i$，定义

$$
\bar y_i=T_i^{-1}\sum_t y_{it},
\qquad
\bar x_i=T_i^{-1}\sum_t x_{it}.
$$

令 $\bar Z_i=(1,\bar x_i^\top)^\top$，则

$$
\widehat\beta_B
=\arg\min_\beta\sum_i(\bar y_i-\bar Z_i^\top\beta)^2
=(\bar Z^\top\bar Z)^+\bar Z^\top\bar y.
$$

也就是说，原始面板数据最终被转换为一个以个体均值为观测单位的普通横截面 OLS 问题。

<a id="参数"></a>

## 完整 API 参考

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `cov_type` | `"nonrobust"` | `nonrobust`、`robust`/`hc1`、`hc0`、`hc2`、`hc3` |
| `alpha` | `0.05` | 有限且严格位于 `(0,1)`；置信区间显著性水平；`0.05` 对应 95% 区间 |
| `device` | `"auto"` | `auto`、`cpu`、`cuda`、`torch` |
| `n_jobs` | `None` | 整数或 `None`；共享并行参数 |

### 拟合输入

`fit(X=None, y=None, entity_ids=None, time_ids=None, formula=None, data=None)` 返回拟合后的估计器。

| 参数 | 含义 |
|---|---|
| `X`、`y` | 数组接口：解释变量为 `(n, p)`，响应为 `(n,)`；一维 `X` 视为单个解释变量。数值必须有限。 |
| `entity_ids` | 必需，与 `X`、`y` 按行对齐的个体标签。 |
| `time_ids` | 为 API 兼容而接受；不参与个体均值变换。 |
| `formula`、`data` | 可替代数组输入的公式字符串和 dataframe；需要可选的 `pandas`/`patsy` 依赖。提供的 ID 数组会与公式保留的观测行对齐。 |

使用数组输入或 `formula` 加 `data` 中的一种。模型的截距约定见[数据框示例](#公式接口)。

### 其他公开方法

- `predict(X)`：每条输入行返回一个预测值，具体解释见前文；特征数量和列顺序必须与训练时一致。
- `summary()`：返回可打印的系数与拟合摘要，使用 `print(model.summary())` 查看。
- `get_params(deep=True)` 与 `set_params(**params)`：读取或更新构造参数；改变设置后应重新拟合。
- 继承的 `adjust_pvalues`、`combine_pvalues`、`bootstrap_statistic` 和 `permutation_test` 的完整签名见[共享估计器 API](../reference/estimator-api.md)。这些工具不会自动选择适用于面板依赖结构的重采样方案。

<a id="常见问题"></a>

## 关于个体均值的常见疑问

**为什么 `nobs` 小于原始数据行数？**  
因为最终回归以“个体均值”为观测单位，每个保留个体只贡献一行。

**时间观测更多的个体会自动获得更高权重吗？**  
不会。当前 BetweenOLS 的最终 OLS 中，每个保留个体贡献一个均值观测。

**BetweenOLS 与固定效应回归可以互换吗？**  
不能。两者利用的数据变异来源和识别条件不同：BetweenOLS 使用个体间长期平均差异，固定效应回归使用去除固定效应后的个体内变化。

<a id="external-validation"></a>
<a id="comparing-with-other-packages"></a>

## 与其他软件包比较

与 statsmodels 等 OLS 实现比较时，先构造完全相同的个体均值样本，并采用相同的截距设定；每个个体只贡献一条均值观测。对齐 HC0/HC2/HC3 等协方差选项后，使用该变换后样本计算协方差与标准误，不能直接使用原始面板行。详见[协方差定义的比较](covariance.md#comparing-covariance-definitions)。

<a id="相关文档"></a>

## 继续阅读

- [面板模型总览](README.md)
- [面板协方差](covariance.md)
- [面板拟合统计量](fit-statistics.md)

<a id="参考文献"></a>

## 统计参考文献

- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). The MIT Press.
