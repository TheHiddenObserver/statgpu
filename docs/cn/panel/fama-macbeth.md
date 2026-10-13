# Fama–MacBeth

> 语言：中文  
> 最后更新：2026-10-13  
> 页面定位：面板模型文档  
> 切换：[English](../../en/panel/fama-macbeth.md)

<a id="概览"></a>

## 何时关注平均横截面斜率

`FamaMacBeth` 在每个时期分别进行一次横截面回归，再对各时期得到的系数取平均。它与其他面板估计器的关键区别在于：目标参数是**平均横截面系数**，而标准误来自这些时期系数**随时间的波动**。

因此，Fama–MacBeth 的推断不是基于一个堆叠残差回归直接构造 HC/cluster/Driscoll–Kraay 协方差，而是把各时期的系数序列作为推断对象。

适合回答“各时期横截面关系的平均水平是多少”，并允许截距和斜率随时间变化。它不是消除遗漏个体效应的固定效应变换。通常的时期斜率解释仍需每期满足横截面外生性或相应的投影正交条件；每个保留时期都必须能识别所有系数，推断也需要足够多的时期来描述系数序列的波动。

每期模型写作 $y_{it}=\alpha_t+x_{it}^{\top}\beta_t+\varepsilon_{it}$。其中 $i$ 表示个体，$t$ 表示时期，$y_{it}$ 是响应，$x_{it}$ 是 `X` 中该行的解释变量向量。截距 $\alpha_t$ 和斜率 $\beta_t$ 可以随时期变化；$\varepsilon_{it}$ 是观测层面的误差。

对每个保留时期，用该期原始解释变量行加上截距做 OLS，再对斜率估计取平均：$\widehat\beta_{\mathrm{FM}}=T^{-1}\sum_{t=1}^{T}\widehat\beta_t$，其中 $T$ 只计保留时期；截距估计也按同样方式平均。这样估计的是等权目标 $\beta_{\mathrm{FM}}=T^{-1}\sum_{t=1}^{T}\beta_t$，不对 `X` 去均值或做差分。

## 准备数据并检查每个时期

采用长表格式，每行是一条个体—时期观测：`X` 的形状为 `(n, p)`，`y` 为 `(n,)`，必需的 `time_ids` 长度为 `n`。所有数组保持相同行序，数值必须有限。一维 `X` 也可作为单个解释变量。每个时间标签定义一次横截面回归；模型自动加入时期截距，不要再给 `X` 添加常数列。可选的 `entity_ids` 仅用于标准化组内/组间 $R^2$。

拟合前先检查每期样本数与解释变量的变化。若有 $p$ 个解释变量，设计矩阵宽度为 $k=p+1$；观测数少于 `max(min_obs_per_period, k)` 的时期会被排除。其余时期都必须满列秩；保留时期秩亏时会抛出 `ValueError`，不会被悄悄跳过。最后至少要剩两期。各保留时期等权进入平均，与每期观测数无关。

<a id="时间顺序"></a>

### 为系数序列确定时间顺序

时期系数序列的顺序由 `time_ids` 决定，因此时间标签的解释直接影响 Newey–West 滞后协方差。

- 数值和 datetime 标签使用自然顺序；
- 有序 pandas categorical 使用用户声明的类别顺序；
- 普通字符串按字典序排序；
- 其他可以比较的对象标签按其自然比较顺序排序；
- 标签无法相互比较时会报错。

如果 `t1, t2, t10` 这类字符串不应按字典序解释，应改用数值/datetime 键，或显式使用有序 categorical。

## 先在 CPU 上拟合一个短系数序列

每行是 12 个个体之一在 5 个时期之一的观测。每期包含截距和 `x` 的设计矩阵均满列秩；生成数据时设定的斜率随时期从 1.6 变化到 2.4。

<a id="cpu-example"></a>

<!-- example: cpu-basic -->
```python
import numpy as np
from statgpu.panel import FamaMacBeth

rng = np.random.default_rng(7)
entity_ids = np.repeat(np.arange(12), 5)
time_ids = np.tile(np.arange(5), 12)
x = rng.normal(size=60)
X = x[:, None]
period_slopes = np.linspace(1.6, 2.4, 5)
y = 1.0 + period_slopes[time_ids] * x + 0.2 * rng.normal(size=60)

model = FamaMacBeth(device="cpu", cov_type="newey-west", bandwidth=1).fit(
    X, y, time_ids=time_ids, entity_ids=entity_ids
)
print("average coef:", np.round(model.coef_, 3))
print("standard errors:", np.round(model.bse_, 3))
print("periods, coefficient shape:", model.n_periods, model.betas_.shape)
```
<!-- example-end: cpu-basic -->

<a id="输出"></a>

## 理解时期系数平均值和预测

`coef_` 依次包含各时期截距和斜率的平均值；斜率接近生成斜率的平均值 2。`betas_.shape=(5, 2)` 表示五期各有两个系数。`bse_` 使用一阶 Newey–West 滞后估计系数序列平均值的不确定性，并非观测级残差的 HC 协方差。五个时期仅用于演示输出；可靠的时间序列推断通常需要更多时期。

常用结果包括：

- `coef_`：各保留时期系数的平均；
- `betas_`：每个保留时期的系数；
- `bse_`、`tvalues_`、`pvalues_`、`conf_int_`：系数推断结果；
- `cov_params_`：系数协方差矩阵；
- `fit_statistics_`：拟合统计量；
- `nobs`、`n_periods`、`df_resid`：样本量与自由度信息。

`n_periods` 是保留的回归时期数，`df_resid=n_periods-1`。`nobs` 记录公式删行后的输入观测总数，包括因时期样本不足而被排除的行；它不是系数序列推断的有效样本量。`coef_` 的形状为 `(p+1,)`，`betas_` 为 `(n_periods, p+1)`，`cov_params_` 为 `(p+1, p+1)`，`conf_int_` 为 `(p+1, 2)`。

`predict(X_new)` 自动加截距，再对每条输入行使用平均后的 `coef_`。它不接受时期参数，也不会根据时期选取 `betas_` 中的某一行。预测结果保留在拟合后端；通过公式拟合时，可传入 dataframe 使用已保存的公式设计。

## 为时期系数序列选择推断方式

默认 `cov_type="newey-west"` 允许保留时期的系数序列存在序列相关。用 `bandwidth` 指定纳入的滞后阶数，`None` 使用后文给出的自动规则。`cov_type="nonrobust"` 把各期系数估计视为独立。两种选择都不是观测级的 HC、cluster、HAC 或 Driscoll–Kraay 协方差；即使名称与其他模型相同，这里的不确定性针对的也是时期系数的平均值。

`bse_` 给出系数标准误，`pvalues_` 给出双侧 p 值，`conf_int_` 给出置信区间。`cov_type="newey-west"` 时，`tvalues_` 存放使用正态参考分布的 z 统计量；`cov_type="nonrobust"` 时，它存放使用自由度为 $T-1$ 的 Student-t 参考分布的 t 统计量。使用 `print(model.summary())` 可一起查看系数估计、不确定性、协方差选择和样本量。

这些公开推断数组和 `coef_` 均保留在选定的拟合后端；Torch 的 p 值与置信区间在实际张量设备上计算。

<a id="数值行为与失败语义"></a>

## 检查时期和数值限制

用户可以依赖以下公开行为：

- 过滤后至少需要两个有效时期，否则无法估计时期系数序列的波动，`.fit()` 会报错；
- 每个保留时期的设计矩阵必须满列秩；秩亏时期会在推断前抛出 `ValueError`；
- 满秩并不保证在任意极端尺度下都能可靠计算系数；当 float64 精度不足时，会明确抛出 `FloatingPointError` 并指出受影响的时期；
- 真正的零系数允许存在，不会仅因为数值结果为 0 就被当成失败；
- 最终协方差含非有限值或对角方差为负时，推断会报错；对角方差精确为零时，零系数对应统计量 0，非零系数对应带符号无穷；
- 每次新的拟合尝试都会先失效旧的已拟合/推断状态；如果新拟合失败，对象保持未拟合状态。

计算使用 float64，而非任意精度运算。即使最终均值或协方差可表示，剧烈抵消后的微小余量也不一定能准确恢复。

不会为零方差人为添加正下限；不满足上述要求时也不会静默改用其他推断方法。显式请求的 `cuda` 或 `torch` 后端不可用时会报错。

<a id="公式接口"></a>

## 从数据框列拟合

假设 `df` 包含 `y`、`x1`、`x2` 和 `time`：

```python
from statgpu.panel import FamaMacBeth

model = FamaMacBeth().fit(
    formula="y ~ x1 + x2",
    data=df,
    time_ids=df["time"],
)
```

`FamaMacBeth` 与数组接口一致，始终包含时期专属截距。因此，`y ~ 0 + x1 + x2` 或 `y ~ x1 + x2 - 1` 这类显式无截距公式会得到清晰的 `ValueError`，而不会被静默改写成带截距模型。

<a id="后端与设备"></a>

## 让计算保留在选定后端

在 GPU 上运行时，把 `device="cpu"` 换成 `device="cuda"`（CuPy/CUDA）或 `device="torch"`（Torch CUDA）。显式请求的后端不可用时会报错，不会切换到 CPU。`device="auto"` 允许自动选择后端，NumPy/CuPy/Torch 原生输入可能保留其后端。显式设备请求具有优先权，拟合和预测输入会转换到请求的或已拟合的后端。时期回归、系数平均、协方差、p 值和置信区间均使用该后端；公开数组保持为相应数组或张量，Torch 使用实际张量设备。

<a id="统计模型与目标参数"></a>

## 时期专属模型与平均目标

一个自然的时期专属模型是

$$
y_{it}=\alpha_t+x_{it}^{\top}\beta_t+\varepsilon_{it},
$$

其中截距和斜率都允许随时期变化。为了赋予每个时期横截面回归通常的统计解释，可以要求

$$
E(\varepsilon_{it}\mid x_{it},t)=0,
$$

或采用相应的横截面正交条件，把 $\beta_t$ 定义为时期 $t$ 的线性投影系数。

对最终保留的 $T$ 个时期，Fama–MacBeth 的直接目标是这些时期系数的等权平均：

$$
\beta_{\mathrm{FM}}
=
\frac1T\sum_{t=1}^T\beta_t.
$$

这个定义本身不要求先为序列 $\{\beta_t\}$ 指定概率模型。如果进一步把这些时期理解为从某个时间总体中抽取的样本，在相应抽样假设下，可以把平均系数解释为类似 $E_t(\beta_t)$ 的总体量。常系数模型 $\beta_t\equiv\beta$ 是其中的特殊情形。

这一解释要求每个保留时期的横截面设计能够唯一识别系数。statgpu 在样本量筛选后检查含截距的 $X_t$ 是否满列秩；秩亏时会在系数平均和推断前抛出 `ValueError`，而不报告非唯一系数表示的逐系数结果。

<a id="估计量"></a>

## 保留时期的系数如何估计

对每个保留时期，令 $X_t$ 表示加入时期截距后的横截面设计矩阵，则

$$
\widehat\beta_t
=
\arg\min_\beta\|y_t-X_t\beta\|_2^2,
\qquad
\widehat\beta_{\mathrm{FM}}
=
\frac1T\sum_{t=1}^T\widehat\beta_t.
$$

时期只有在满足 `min_obs_per_period` 以及模型的最小样本量要求后才会进入最终平均；设包含截距后的设计矩阵宽度为 $k$，实际保留时期还需要满足 $n_t\ge k$。

每个保留时期的横截面设计矩阵必须能够唯一识别系数。如果某个时期的设计矩阵秩亏，`.fit()` 会在系数平均和推断之前抛出 `ValueError`，而不会对非唯一的系数向量报告逐系数标准误。

如果设计矩阵满秩，但某个系数在 float64 下已经小到无法可靠区分其数值贡献，statgpu 也可能抛出 `FloatingPointError`。这表示有限精度下无法可靠计算该系数，与设计矩阵秩亏不同。真正的零系数允许存在；剧烈抵消导致非零贡献丢失是另一种情况。

<a id="协方差与推断"></a>

## 系数序列的协方差公式

定义各时期系数相对于最终平均系数的偏差

$$
\nu_t
=
\widehat\beta_t-\widehat\beta_{\mathrm{FM}}.
$$

<a id="cov_typenonrobust"></a>

### 独立时期系数的协方差

当把各时期系数视为相互独立时，协方差估计为

$$
\widehat V_{\mathrm{nonrobust}}
=
\frac{1}{T(T-1)}
\sum_{t=1}^T \nu_t\nu_t^\top.
$$

这一模式使用自由度 $T-1$ 的 Student-t 参考分布。

<a id="cov_typenewey-west"></a>

### 序列相关时期系数的协方差

如果允许时期系数序列存在序列相关，定义

$$
\widehat\Gamma_\ell
=
\frac1T\sum_{t=\ell+1}^{T}
\nu_t\nu_{t-\ell}^\top,
\qquad \ell=0,\ldots,L,
$$

以及 Bartlett 权重

$$
w_\ell=1-\frac{\ell}{L+1}.
$$

则长期协方差与最终平均系数的协方差分别为

$$
\widehat\Omega_{\mathrm{NW}}
=
\widehat\Gamma_0
+
\sum_{\ell=1}^{L}w_\ell
\left(
\widehat\Gamma_\ell+
\widehat\Gamma_\ell^\top
\right),
$$

$$
\widehat V_{\mathrm{NW}}
(\widehat\beta_{\mathrm{FM}})
=
\frac1T\widehat\Omega_{\mathrm{NW}}.
$$

当 `bandwidth=None` 时，statgpu 先使用

$$
L=
\left\lfloor4(T/100)^{2/9}\right\rfloor,
$$

再把结果限制在 $0\le L\le T-1$。

这里的 Newey–West 作用于**时期系数序列**，因此与 [面板模型协方差](covariance.md) 中基于观测级残差/得分的 HAC 或 Driscoll–Kraay 不同。`newey-west` 使用渐近正态参考分布。

<a id="参数"></a>

## 完整 API 参考

| 参数 | 默认值 | 约束 | 含义 |
|---|---:|---|---|
| `cov_type` | `"newey-west"` | `nonrobust` 或 `newey-west` | 是否忽略时期系数的跨期相关，或使用 Newey–West 修正 |
| `bandwidth` | `None` | `None` 或非负整数，最终不超过 $T-1$ | Bartlett Newey–West 带宽 $L$ |
| `alpha` | `0.05` | 有限且严格位于 `(0,1)` | 置信区间显著性水平；`0.05` 对应 95% 区间 |
| `min_obs_per_period` | `1` | 正整数 | 初步的最小时期样本量；最终还要满足 $n_t\ge k$ 且设计矩阵满秩 |
| `device` | `"auto"` | `auto` / `cpu` / `cuda` / `torch` | 数值计算的执行设备 |
| `n_jobs` | `None` | 整数或 `None` | 共享并行参数 |

### 拟合输入

`fit(X=None, y=None, time_ids=None, formula=None, data=None, entity_ids=None)` 返回拟合后的估计器。

| 参数 | 含义 |
|---|---|
| `X`、`y` | 数组接口：解释变量为 `(n, p)`，响应为 `(n,)`；一维 `X` 视为单个解释变量。数值必须有限。 |
| `entity_ids` | 可选，仅用于组内/组间拟合统计量。 |
| `time_ids` | 必需，用来划分各时期的横截面回归。 |
| `formula`、`data` | 可替代数组输入的公式字符串和 dataframe；需要可选的 `pandas`/`patsy` 依赖。提供的 ID 数组会与公式保留的观测行对齐。 |

使用数组输入或 `formula` 加 `data` 中的一种。模型的截距约定见[数据框示例](#公式接口)。

### 其他公开方法

- `predict(X)`：每条输入行返回一个预测值，具体解释见前文；特征数量和列顺序必须与训练时一致。
- `summary()`：返回可打印的系数与拟合摘要，使用 `print(model.summary())` 查看。
- `get_params(deep=True)` 与 `set_params(**params)`：读取或更新构造参数；改变设置后应重新拟合。
- 继承的 `adjust_pvalues`、`combine_pvalues`、`bootstrap_statistic` 和 `permutation_test` 的完整签名见[共享估计器 API](../reference/estimator-api.md)。这些工具不会自动选择适用于面板依赖结构的重采样方案。

<a id="常见问题"></a>

## 关于时期筛选和推断的常见疑问

**为什么 Fama–MacBeth 的协方差不直接列入 HC/cluster/Driscoll–Kraay？**  
这些方法通常基于观测级回归残差或得分；Fama–MacBeth 推断则基于各时期的系数 $\widehat\beta_t$ 所形成的时间序列。

**样本过少的时期如何处理？**  
会在形成最终系数平均之前排除。如果最后不足两个有效时期，拟合会报错。

**如果某个保留时期秩亏会怎样？**  
拟合会抛出 `ValueError`，不会把非唯一的时期系数向量纳入平均后再报告逐系数标准误。

**如果时期设计满秩，但某个系数低于可靠的 float64 分辨率呢？**  
当有限精度计算无法可靠求出该系数时，拟合会抛出 `FloatingPointError`，说明系数分辨率不足，而不会把它误报成秩亏。真正的零系数仍然有效。

<a id="external-validation"></a>
<a id="comparing-with-other-packages"></a>

## 与其他软件包比较

与 `linearmodels` 7.0 比较时，应对齐显式时期截距、保留观测、满秩时期设计、时间顺序和系数集合。先将 `betas_` 与 linearmodels 的 `all_params` 比较，再比较平均系数。

对于 `cov_type="nonrobust"`，linearmodels 中对应的协方差设置是 `cov_type="unadjusted", debiased=True`。两者的系数、协方差、标准误和系数 t 统计量定义一致，但 p 值和置信区间仍可能不同：statgpu 使用保留时期数对应的自由度 $T-1$，而 linearmodels 在 `debiased=True` 时使用堆叠面板的残差自由度。

对于 `cov_type="newey-west"`，在 linearmodels 中使用 `cov_type="kernel", kernel="bartlett", bandwidth=L, debiased=False`，并固定相同的 $L$。这样可对齐系数序列的核协方差与正态参考分布推断。该方法仍不同于[面板协方差](covariance.md)中的观测级残差协方差估计。

<a id="相关文档"></a>

## 继续阅读

- [面板模型总览](README.md) — Panel 模型族与选择建议
- [面板模型架构](architecture.md) — 共享层和各估计器的职责
- [面板模型协方差](covariance.md) — HC、cluster、HAC 与 Driscoll–Kraay
- [设备与 GPU 内存](../guides/device-and-memory.md) — 后端与设备语义

<a id="参考文献"></a>

## 统计参考文献

- Fama, E. F., & MacBeth, J. D. (1973). Risk, return, and equilibrium: Empirical tests. *Journal of Political Economy*, 81(3), 607-636. [https://doi.org/10.1086/260061](https://doi.org/10.1086/260061)
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703-708. [https://doi.org/10.2307/1913610](https://doi.org/10.2307/1913610)
