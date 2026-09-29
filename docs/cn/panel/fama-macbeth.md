# Fama–MacBeth

> 语言：中文  
> 最后更新：2026-09-17  
> 页面定位：面板模型文档  
> 切换：[English](../../en/panel/fama-macbeth.md)

## 概览

`FamaMacBeth` 在每个时期分别进行一次横截面回归，再对各时期得到的系数取平均。它与其他面板估计器的关键区别在于：目标参数是**平均横截面系数**，而标准误来自这些时期系数**随时间的波动**。

因此，Fama–MacBeth 的推断不是基于一个堆叠残差回归直接构造 HC/cluster/Driscoll–Kraay 协方差，而是把各时期的系数序列作为推断对象。

## 统计模型与目标参数

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

## 估计量

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

每个保留时期的横截面设计矩阵必须能够唯一识别系数。如果某个时期的设计矩阵秩亏，`.fit()` 会在系数平均和推断之前抛出 `ValueError`，而不会把一个非唯一的系数表示继续带入坐标级推断。

如果设计矩阵满秩，但某个系数在 float64 下已经小到无法可靠区分其数值贡献，statgpu 也可能抛出 `FloatingPointError`。这类失败表示有限精度下无法可靠认证系数，而不是把问题错误归类为秩亏。

## 协方差与推断

定义各时期系数相对于最终平均系数的偏差

$$
\nu_t
=
\widehat\beta_t-\widehat\beta_{\mathrm{FM}}.
$$

### `cov_type="nonrobust"`

当把各时期系数视为相互独立时，协方差估计为

$$
\widehat V_{\mathrm{nonrobust}}
=
\frac{1}{T(T-1)}
\sum_{t=1}^T \nu_t\nu_t^\top.
$$

这一模式使用自由度 $T-1$ 的 Student-t 参考分布。

### `cov_type="newey-west"`

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

## 时间顺序

时期系数序列的顺序由 `time_ids` 决定，因此时间标签的解释直接影响 Newey–West 滞后协方差。

- 数值和 datetime 标签使用自然顺序；
- 有序 pandas categorical 使用用户声明的类别顺序；
- 普通字符串按字典序排序；
- 其他可以比较的对象标签按其自然比较顺序排序；
- 标签无法相互比较时会报错。

如果 `t1, t2, t10` 这类字符串不应按字典序解释，应改用数值/datetime 键，或显式使用有序 categorical。

## 后端与设备

`FamaMacBeth` 支持公开文档所列的 NumPy、CuPy 与 Torch 路径。显式设备请求具有优先权：

```python
from statgpu.panel import FamaMacBeth

cpu = FamaMacBeth(device="cpu").fit(X, y, time_ids=time_ids)
cuda = FamaMacBeth(device="cuda").fit(X, y, time_ids=time_ids)
torch = FamaMacBeth(device="torch").fit(X, y, time_ids=time_ids)
```

如果显式请求的 CUDA/Torch 后端不可用，`.fit()` 会报错，而不会静默切换到其他后端。`device="auto"` 则允许 statgpu 根据输入和可用后端自动选择。

在受支持的后端上，时期回归、系数平均、协方差和参考分布计算都在实际数值后端完成；统一结果容器所需要的小型 NumPy 结果只在数值推断结束后整理。

## 参数

| 参数 | 默认值 | 约束 | 含义 |
|---|---:|---|---|
| `cov_type` | `"newey-west"` | `nonrobust` 或 `newey-west` | 是否忽略时期系数的跨期相关，或使用 Newey–West 修正 |
| `bandwidth` | `None` | `None` 或非负整数，最终不超过 $T-1$ | Bartlett Newey–West 带宽 $L$ |
| `alpha` | `0.05` | 有限且严格位于 `(0,1)` | 置信区间显著性水平；`0.05` 对应 95% 区间 |
| `min_obs_per_period` | `1` | 正整数 | 初步的最小时期样本量；最终还要满足 $n_t\ge k$ 且设计矩阵满秩 |
| `device` | `"auto"` | `auto` / `cpu` / `cuda` / `torch` | 数值计算的执行设备 |
| `n_jobs` | `None` | 整数或 `None` | 共享并行参数 |

拟合接口：

```python
model.fit(X, y, time_ids=time_ids, entity_ids=None)
```

`time_ids` 必需，用于定义各时期的横截面回归；`entity_ids` 可选，用于标准化的组内/组间 $R^2$。

## 公式接口

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

## 输出

常用结果包括：

- `coef_`：各保留时期系数的平均；
- `betas_`：每个保留时期的系数；
- `bse_`、`tvalues_`、`pvalues_`、`conf_int_`：系数推断结果；
- `cov_params_`：系数协方差矩阵；
- `fit_statistics_`：拟合统计量；
- `nobs`、`n_periods`、`df_resid`：样本量与自由度信息；
- `_inference_result`：统一的结构化推断结果容器。

## 数值行为与失败语义

用户可以依赖以下公开行为：

- 过滤后至少需要两个有效时期，否则无法估计时期系数序列的波动，`.fit()` 会报错；
- 每个保留时期必须满足共享的 Panel 秩判定要求；秩亏时期会在推断前报错；
- 满秩并不保证任意极端尺度都能在 float64 中可靠认证；当系数分辨率不足时，会明确抛出 `FloatingPointError`；
- 真正的零系数允许存在，不会仅因为数值结果为 0 就被当成失败；
- 协方差如果在最终尺度上无法可靠表示，推断会报错，而不是静默返回伪造的 0、无穷大或负方差；
- 每次新的拟合尝试都会先失效旧的已拟合/推断状态；如果新拟合失败，对象保持未拟合状态。

这些行为并不要求用户了解内部使用哪一种 Gram/SVD 快速路径、数值缩放或归约策略。那些实现细节和硬件验证证据属于工程层，而不是本模型页的稳定接口。

## 常见问题

**为什么 Fama–MacBeth 的协方差不直接列入 HC/cluster/Driscoll–Kraay？**  
这些方法通常基于观测级回归残差或得分；Fama–MacBeth 推断则基于各时期的系数 $\widehat\beta_t$ 所形成的时间序列。

**样本过少的时期如何处理？**  
会在形成最终系数平均之前排除。如果最后不足两个有效时期，拟合会报错。

**如果某个保留时期秩亏会怎样？**  
拟合会抛出 `ValueError`，不会把非唯一的时期系数向量纳入平均后再发布坐标级标准误。

**如果时期设计满秩，但某个系数低于可靠的 float64 分辨率呢？**  
当现有有限精度计算无法可靠认证该坐标时，拟合会抛出带有系数分辨率含义的 `FloatingPointError`，而不会把它误报成秩亏。

## 相关文档

- [面板模型总览](../models/panel.md) — Panel 模型族与选择建议
- [面板模型架构](architecture.md) — 共享层和各估计器的职责
- [面板模型协方差](covariance.md) — HC、cluster、HAC 与 Driscoll–Kraay
- [设备与 GPU 内存](../guides/device-and-memory.md) — 后端与设备语义

## 参考文献

- Fama, E. F., & MacBeth, J. D. (1973). Risk, return, and equilibrium: Empirical tests. *Journal of Political Economy*, 81(3), 607-636. [https://doi.org/10.1086/260061](https://doi.org/10.1086/260061)
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703-708. [https://doi.org/10.2307/1913610](https://doi.org/10.2307/1913610)
