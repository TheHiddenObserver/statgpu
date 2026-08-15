# FamaMacBeth

> 语言：中文  
> 最后更新：2026-08-15  
> 切换：[English](../../en/panel/fama-macbeth.md)

## Overview

`FamaMacBeth` 在每个保留时期分别运行 cross-sectional regression，再对 period-specific coefficient vector 求平均。其 inference 基于 coefficient time series，而不是 residual-based panel covariance。

## Path

实现：`statgpu/panel/_fama_macbeth.py`。

## Objective and Estimator

对每个保留时期，令 $X_t$ 表示**已加入自动 intercept** 的 period design，则

$$
\widehat\beta_t
=\arg\min_\beta\|y_t-X_t\beta\|_2^2
=(X_t^\top X_t)^+X_t^\top y_t,
\qquad
\widehat\beta_{\mathrm{FM}}=T^{-1}\sum_{t=1}^T\widehat\beta_t.
$$

低于 `min_obs_per_period` 或不足以支撑 period design 的时期会被过滤。

## Covariance and Inference

令

$$
u_t=\widehat\beta_t-\widehat\beta_{\mathrm{FM}}.
$$

当 `cov_type="nonrobust"` 时，

$$
\widehat V_{\mathrm{nonrobust}}
=\frac{1}{T(T-1)}\sum_{t=1}^T u_tu_t^\top.
$$

当 `cov_type="newey-west"` 时，定义 lag-$\ell$ coefficient covariance

$$
\widehat\Gamma_\ell
=\frac{1}{T}\sum_{t=\ell+1}^{T}u_tu_{t-\ell}^\top,
\qquad \ell=0,\ldots,L,
$$

以及 Bartlett weight

$$
w_\ell=1-\frac{\ell}{L+1}.
$$

则 coefficient-series long-run covariance 与 Fama-MacBeth 平均 coefficient 的 covariance 分别为

$$
\widehat\Omega_{\mathrm{NW}}
=\widehat\Gamma_0+
\sum_{\ell=1}^{L}w_\ell
\left(\widehat\Gamma_\ell+\widehat\Gamma_\ell^\top\right),
\qquad
\widehat V_{\mathrm{NW}}(\widehat\beta_{\mathrm{FM}})
=\frac{1}{T}\widehat\Omega_{\mathrm{NW}}.
$$

若 `bandwidth=None`，实现先取

$$
L=\left\lfloor4(T/100)^{2/9}\right\rfloor,
$$

再截断到 $0\le L\le T-1$。该 coefficient-series HAC 路径与 [面板 covariance](covariance.md) 中的 residual-based 定义不同。

## Parameters

| 参数 | 默认值 | 可选值 / 约束 | 含义 |
|---|---:|---|---|
| `cov_type` | `"newey-west"` | `nonrobust` 或 `newey-west` | coefficient-series covariance estimator。 |
| `bandwidth` | `None` | `None` 或非负整数；最终不超过 $T-1$ | Bartlett HAC bandwidth $L$。 |
| `alpha` | `0.05` | 有限且严格位于 0 与 1 之间 | 置信区间显著性水平；`0.05` 对应 95% 区间。 |
| `min_obs_per_period` | `1` | 正整数 | 初步的最小 period size；最终保留时期还必须满足 $n_t\ge k+1$，其中 $k$ 是含 intercept 的 design width。 |
| `device` | `"auto"` | `auto`、`cpu`、`cuda`、`torch` | 数值 backend/device。 |
| `n_jobs` | `None` | integer 或 `None` | 共享并行参数。 |

```python
model.fit(X, y, time_ids=time_ids, entity_ids=None)
```

`time_ids` 必需；`entity_ids` 只用于 standardized within/between $R^2$。

## CPU and GPU Example

```python
from statgpu.panel import FamaMacBeth

cpu = FamaMacBeth(device="cpu").fit(X, y, time_ids=time_ids)
cuda = FamaMacBeth(device="cuda").fit(X, y, time_ids=time_ids)
torch = FamaMacBeth(device="torch").fit(X, y, time_ids=time_ids)
```

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

public 结果包括 `coef_`、`bse_`、`tvalues_`、`pvalues_`、`conf_int_`、`betas_`、`cov_params_`、`fit_statistics_`、`nobs`、`n_periods` 与 `df_resid`。

## Numerical and Strict Behavior

过滤后至少需要两个 period。Newey-West 路径使用 asymptotic-normal coefficient inference；nonrobust 路径使用自由度 $T-1$ 的 Student-t reference。不存在 separate approximate estimator/inference path，显式 GPU 请求也不会静默 fallback 到 CPU。

## FAQ

**为什么它的 covariance 不列在 HC/cluster/DK 中？**  Fama-MacBeth inference 来自 $\widehat\beta_t$ 的时间序列，而不是 observation residual。

**样本过少的 period 如何处理？**  在 coefficient average 前被排除；若没有有效 period 或保留不足两个 period，则 fit 报错。

## External Validation

当前没有针对该 estimator beta-series covariance 的 maintained external-framework parity gate，因此文档不宣称其与 `linearmodels` 或其他 package 一致。`dev/tests/test_panel_p2.py` 维护 estimator/covariance/filtering regression，formula coverage 位于 `dev/tests/test_panel_formula.py`。

`dev/benchmarks/validate_panel_stage_a_gpu.py` 中的 `fama_macbeth_newey_west` case 会用默认 `rtol=5e-6, atol=5e-7` 比较 CuPy/Torch 与 NumPy。Stage-C residual-covariance physical matrix 是另一套 gate，不包含 FamaMacBeth。

## 参考（References）

- Fama, E. F., & MacBeth, J. D. (1973). Risk, return, and equilibrium: Empirical tests. *Journal of Political Economy*, 81(3), 607-636. [https://doi.org/10.1086/260061](https://doi.org/10.1086/260061)
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703-708. [https://doi.org/10.2307/1913610](https://doi.org/10.2307/1913610)
