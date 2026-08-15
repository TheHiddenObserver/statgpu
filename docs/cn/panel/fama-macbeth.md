# FamaMacBeth

> 语言：中文  
> 最后更新：2026-08-16  
> 切换：[English](../../en/panel/fama-macbeth.md)

## Overview

`FamaMacBeth` 在每个 time period 分别做一次 cross-sectional regression，再对这些 period-specific coefficient 取平均。它与其他 panel estimator 的一个关键区别是：目标参数是平均 cross-sectional slope，standard error 则来自这些 slope **随时间的波动**。

## Path

实现：`statgpu/panel/_fama_macbeth.py`。

## Statistical Model and Target

一个自然的 period-specific model 是

$$
y_{it}=\alpha_t+x_{it}^{\top}\beta_t+\varepsilon_{it},
$$

其中 intercept 与 slope 都允许随时期变化。为了赋予每个 cross-sectional regression 通常的统计解释，一个常见的充分条件是

$$
E(\varepsilon_{it}\mid x_{it},t)=0,
$$

或者更一般地，使用相应的 cross-sectional orthogonality condition，将 $\beta_t$ 定义为 period-$t$ 的 linear projection coefficient。

对 estimator 最终保留的 $T$ 个时期，直接目标是 retained-period 的等权平均

$$
\beta_{\mathrm{FM}}=\frac{1}{T}\sum_{t=1}^{T}\beta_t.
$$

定义这个 target 并不要求先给 sequence $\{\beta_t\}$ 指定概率模型。如果进一步把这些 retained periods 看成从某个 time superpopulation 中抽取，那么在相应 sampling assumptions 下，这个平均量还可以获得类似 $E_t(\beta_t)$ 的 population interpretation。constant-slope model $\beta_t\equiv\beta$ 是其中的特殊情况。

上述 coefficient interpretation 要求每个最终保留时期的 cross-sectional design 都能唯一识别相应 coefficient vector。完成 observation-count filtering 后，statgpu 会按照共享的 panel SVD rank policy 检查加入 intercept 后的 $X_t$。如果某个 retained period rank deficient，`.fit()` 会在 coefficient averaging 与 inference 之前直接抛出 `ValueError`，而不会基于非唯一的 coefficient representation 继续做 coordinate-level inference。

## Estimator

对每个保留时期，令 $X_t$ 表示已加入 intercept 的 period design。在要求 full column rank 的契约下，

$$
\widehat\beta_t
=\arg\min_\beta\|y_t-X_t\beta\|_2^2
=(X_t^\top X_t)^{-1}X_t^\top y_t,
\qquad
\widehat\beta_{\mathrm{FM}}=T^{-1}\sum_{t=1}^T\widehat\beta_t.
$$

时期只有在满足 `min_obs_per_period` 且通过实现中的最小样本量规则 $n_t\ge k+1$ 时才会保留，其中 $k$ 是含 intercept 的 design width。随后每个 retained period 还必须通过 full-rank 检查。最终 `coef_` 是所有保留时期 coefficient 的简单平均。

## Covariance and Inference

定义每个 period coefficient 与最终平均 coefficient 的偏差

$$
\nu_t=\widehat\beta_t-\widehat\beta_{\mathrm{FM}}.
$$

当 `cov_type="nonrobust"` 时，

$$
\widehat V_{\mathrm{nonrobust}}
=\frac{1}{T(T-1)}\sum_{t=1}^T \nu_t\nu_t^\top.
$$

这相当于把保留时期的 coefficient 看作相互独立的 coefficient series。若使用 `cov_type="newey-west"`，则允许该 coefficient series 存在 serial dependence。定义

$$
\widehat\Gamma_\ell
=\frac{1}{T}\sum_{t=\ell+1}^{T}\nu_t\nu_{t-\ell}^\top,
\qquad \ell=0,\ldots,L,
$$

以及 Bartlett weight

$$
w_\ell=1-\frac{\ell}{L+1}.
$$

coefficient series 的 long-run covariance 与 Fama-MacBeth 平均 coefficient 的 covariance 分别为

$$
\widehat\Omega_{\mathrm{NW}}
=\widehat\Gamma_0+
\sum_{\ell=1}^{L}w_\ell
\left(\widehat\Gamma_\ell+\widehat\Gamma_\ell^\top\right),
\qquad
\widehat V_{\mathrm{NW}}(\widehat\beta_{\mathrm{FM}})
=\frac{1}{T}\widehat\Omega_{\mathrm{NW}}.
$$

若 `bandwidth=None`，statgpu 先取

$$
L=\left\lfloor4(T/100)^{2/9}\right\rfloor,
$$

再截断到 $0\le L\le T-1$。这里的 Newey-West 是作用在 period coefficient sequence 上，因此与 [面板 covariance](covariance.md) 中基于 observation residual 的 HAC/Driscoll-Kraay 不同。

retained coefficient series 的顺序由 `time_ids` 决定。numeric 与 datetime labels 使用自然顺序；ordered pandas categorical 会保留用户明确声明的 category order，不会在形成 Newey-West lag covariance 之前被转换为字符串字典序。

## Parameters

| 参数 | 默认值 | 可选值 / 约束 | 含义 |
|---|---:|---|---|
| `cov_type` | `"newey-west"` | `nonrobust` 或 `newey-west` | 是否忽略 period coefficient 的跨期相关性，或用 Newey-West 进行修正。 |
| `bandwidth` | `None` | `None` 或非负整数；最终不超过 $T-1$ | Bartlett Newey-West bandwidth $L$。 |
| `alpha` | `0.05` | 有限且严格位于 0 与 1 之间 | 置信区间显著性水平；`0.05` 对应 95% 区间。 |
| `min_obs_per_period` | `1` | 正整数 | 初步的最小 period size；最终保留时期还必须满足 $n_t\ge k+1$，其中 $k$ 是含 intercept 的 design width，并且必须 full column rank。 |
| `device` | `"auto"` | `auto`、`cpu`、`cuda`、`torch` | 数值计算运行在哪个 backend/device。 |
| `n_jobs` | `None` | integer 或 `None` | 共享并行参数。 |

```python
model.fit(X, y, time_ids=time_ids, entity_ids=None)
```

`time_ids` 必需，用于定义每个 cross-sectional regression。`entity_ids` 可选，只用于 standardized within/between $R^2$。

## CPU and GPU Example

```python
from statgpu.panel import FamaMacBeth

cpu = FamaMacBeth(device="cpu").fit(X, y, time_ids=time_ids)
cuda = FamaMacBeth(device="cuda").fit(X, y, time_ids=time_ids)
torch = FamaMacBeth(device="torch").fit(X, y, time_ids=time_ids)
```

若显式指定的 GPU backend 不可用，`.fit()` 会直接报错，而不是切换到 CPU。

## Formula Example

假设 `df` 包含 `y`、`x1`、`x2` 与 `time` 列。

```python
from statgpu.panel import FamaMacBeth

model = FamaMacBeth().fit(
    formula="y ~ x1 + x2",
    data=df,
    time_ids=df["time"],
)
```

`FamaMacBeth` 与 array API 一致，始终包含 period-specific intercept。因此 `y ~ 0 + x1 + x2` 或 `y ~ x1 + x2 - 1` 这类显式 no-intercept formula 会得到清晰的 `ValueError`，而不会被静默改写成带 intercept 的模型。

## Outputs

常用结果包括 `coef_`、`bse_`、`tvalues_`、`pvalues_`、`conf_int_`、`betas_`、`cov_params_`、`fit_statistics_`、`nobs`、`n_periods` 与 `df_resid`。`betas_` 保存各保留时期的 coefficient，`coef_` 则是这些 coefficient 的平均。

## Numerical and Strict Behavior

过滤后至少需要两个有效 period，否则 `.fit()` 会报错，因为少于两个 period 无法估计 coefficient series 的波动。每个 retained period 还必须在共享 panel SVD cutoff 下 full column rank；若某个 retained period rank deficient，会在 inference 之前 fail closed。

`cov_type="newey-west"` 使用 asymptotic-normal inference；`cov_type="nonrobust"` 使用自由度 $T-1$ 的 Student-t reference。如果这些条件不满足，statgpu 不会在后台切换成另一套 inference 方法。

显式指定 `device="cuda"` 或 `device="torch"` 时也要求对应 backend 可用，否则直接报错而不是切换到 CPU。

## FAQ

**为什么它的 covariance 不列在 HC/cluster/Driscoll-Kraay 中？**  这些方法基于 observation-level regression residual。Fama-MacBeth inference 则基于各时期 coefficient $\widehat\beta_t$ 形成的 time series。

**样本过少的 period 如何处理？**  在形成 coefficient average 之前排除。若最终不足两个有效 period，fit 会报错。

**如果某个 retained period rank deficient 会怎样？**  fit 会抛出 `ValueError`。statgpu 不会把非唯一的 period coefficient vector 纳入平均后再给出 coordinate-level standard error。

## External Validation

目前没有针对该 estimator coefficient-series covariance 的 maintained cross-package comparison，因此文档不宣称 `linearmodels` 或其他 package 会产生完全相同的 Fama-MacBeth standard error。maintained regression tests 覆盖 formula intercept contract、array/formula 两条路径上的 ordered-categorical 与 numeric chronology、formula missing-row alignment、retained-period rank rejection，以及该 rank contract 的 Torch-CPU parity。

标准 full-rank、numeric-time 的 `fama_macbeth_newey_west` case 仍由 `dev/benchmarks/validate_panel_stage_a_gpu.py` 单独做 GPU consistency 验证，使用默认 `rtol=5e-6, atol=5e-7` 比较 CuPy、Torch 与 NumPy。Fama-MacBeth 不属于 Stage-C residual-covariance matrix，因为它的 covariance 来自 coefficient series，而不是 observation residual。

## 参考（References）

- Fama, E. F., & MacBeth, J. D. (1973). Risk, return, and equilibrium: Empirical tests. *Journal of Political Economy*, 81(3), 607-636. [https://doi.org/10.1086/260061](https://doi.org/10.1086/260061)
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703-708. [https://doi.org/10.2307/1913610](https://doi.org/10.2307/1913610)
