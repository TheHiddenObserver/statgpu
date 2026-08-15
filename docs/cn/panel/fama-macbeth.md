# FamaMacBeth

> 语言：中文  
> 最后更新：2026-08-15  
> 切换：[English](../../en/panel/fama-macbeth.md)

## Overview

`FamaMacBeth` 在每个 time period 分别做一次 cross-sectional regression，再对这些 period-specific coefficient 取平均。它与其他 panel estimator 的一个关键区别是：standard error 来自 **coefficient 随时间的波动**，而不是来自一个 pooled regression 的 residual covariance。

## Path

实现：`statgpu/panel/_fama_macbeth.py`。

## Objective and Estimator

对每个保留时期，令 $X_t$ 表示已加入 intercept 的 period design，则

$$
\widehat\beta_t
=\arg\min_\beta\|y_t-X_t\beta\|_2^2
=(X_t^\top X_t)^+X_t^\top y_t,
\qquad
\widehat\beta_{\mathrm{FM}}=T^{-1}\sum_{t=1}^T\widehat\beta_t.
$$

只有同时满足 `min_obs_per_period`，且 observation 数足以估计该 period regression 的时期才会保留。最终 `coef_` 是所有保留时期 coefficient 的简单平均。

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

## Parameters

| 参数 | 默认值 | 可选值 / 约束 | 含义 |
|---|---:|---|---|
| `cov_type` | `"newey-west"` | `nonrobust` 或 `newey-west` | 是否忽略 period coefficient 的跨期相关性，或用 Newey-West 进行修正。 |
| `bandwidth` | `None` | `None` 或非负整数；最终不超过 $T-1$ | Bartlett Newey-West bandwidth $L$。 |
| `alpha` | `0.05` | 有限且严格位于 0 与 1 之间 | 置信区间显著性水平；`0.05` 对应 95% 区间。 |
| `min_obs_per_period` | `1` | 正整数 | 初步的最小 period size；最终保留时期还必须满足 $n_t\ge k+1$，其中 $k$ 是含 intercept 的 design width。 |
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

## Outputs

常用结果包括 `coef_`、`bse_`、`tvalues_`、`pvalues_`、`conf_int_`、`betas_`、`cov_params_`、`fit_statistics_`、`nobs`、`n_periods` 与 `df_resid`。`betas_` 保存各保留时期的 coefficient，`coef_` 则是这些 coefficient 的平均。

## Numerical and Strict Behavior

过滤后至少需要两个有效 period，否则 `.fit()` 会报错，因为少于两个 period 无法估计 coefficient series 的波动。

`cov_type="newey-west"` 使用 asymptotic-normal inference；`cov_type="nonrobust"` 使用自由度 $T-1$ 的 Student-t reference。如果这些条件不满足，statgpu 不会在后台切换成另一套 inference 方法。

显式指定 `device="cuda"` 或 `device="torch"` 时也要求对应 backend 可用，否则直接报错而不是切换到 CPU。

## FAQ

**为什么它的 covariance 不列在 HC/cluster/Driscoll-Kraay 中？**  这些方法基于 observation-level regression residual。Fama-MacBeth inference 则基于各时期 coefficient $\widehat\beta_t$ 形成的 time series。

**样本过少的 period 如何处理？**  在形成 coefficient average 之前排除。若最终不足两个有效 period，fit 会报错。

## External Validation

目前没有针对该 estimator coefficient-series covariance 的 maintained cross-package comparison，因此文档不宣称 `linearmodels` 或其他 package 会产生完全相同的 Fama-MacBeth standard error。estimator、covariance、period filtering 与 formula behavior 由 `dev/tests/` 下的 panel regression tests 覆盖。

`fama_macbeth_newey_west` 的 GPU 一致性由 `dev/benchmarks/validate_panel_stage_a_gpu.py` 单独验证，使用默认 `rtol=5e-6, atol=5e-7` 比较 CuPy、Torch 与 NumPy。Fama-MacBeth 不属于 Stage-C residual-covariance matrix，因为它的 covariance 来自 coefficient series，而不是 observation residual。

## 参考（References）

- Fama, E. F., & MacBeth, J. D. (1973). Risk, return, and equilibrium: Empirical tests. *Journal of Political Economy*, 81(3), 607-636. [https://doi.org/10.1086/260061](https://doi.org/10.1086/260061)
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703-708. [https://doi.org/10.2307/1913610](https://doi.org/10.2307/1913610)
