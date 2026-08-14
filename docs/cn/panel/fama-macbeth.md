# FamaMacBeth

> 语言：中文  
> 最后更新：2026-08-14  
> 切换：[English](../../en/panel/fama-macbeth.md)

## Overview

`FamaMacBeth` 在每个保留时期分别运行 cross-sectional regression，再对 period-specific coefficient vector 求平均。其 inference 基于 coefficient time series，而不是 residual-based panel covariance。

## Path

实现：`statgpu/panel/_fama_macbeth.py`。

## Objective and Estimator

对每个保留时期，

$$
\widehat\beta_t
=\arg\min_\beta\|y_t-X_t\beta\|_2^2
=(X_t^\top X_t)^+X_t^\top y_t,
\qquad
\widehat\beta_{\mathrm{FM}}=T^{-1}\sum_{t=1}^T\widehat\beta_t.
$$

低于 `min_obs_per_period` 或不足以支撑 period design 的时期会被过滤。

## Covariance and Inference

令 $u_t=\widehat\beta_t-\widehat\beta_{\mathrm{FM}}$。当 `cov_type="nonrobust"` 时，

$$
\widehat V_{\mathrm{FM}}
=\frac{1}{T}\left[\frac{1}{T-1}\sum_{t=1}^T u_tu_t^\top\right].
$$

`cov_type="newey-west"` 将括号内替换为 coefficient series 的 Bartlett HAC long-run covariance。该路径与 [面板 covariance](covariance.md) 中的 residual-based 定义刻意分离。

## Parameters

| 参数 | 含义 |
|---|---|
| `cov_type` | `nonrobust` 或 `newey-west`。 |
| `bandwidth` | coefficient-series Bartlett HAC bandwidth。 |
| `min_obs_per_period` | design-size filtering 前的最小 period sample size。 |
| `alpha` | 置信区间显著性水平。 |
| `device` | `auto`、`cpu`、`cuda` 或 `torch`。 |
| `n_jobs` | 共享并行参数。 |

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

## Outputs

public 结果包括 `coef_`、`bse_`、`tvalues_`、`pvalues_`、`conf_int_`、`betas_`、`cov_params_`、`fit_statistics_`、`nobs`、`n_periods` 与 `df_resid`。

## Numerical and Strict Behavior

过滤后至少需要两个 period。Newey-West 路径使用 asymptotic-normal coefficient inference；nonrobust 路径使用自由度 $T-1$ 的 Student-t reference。显式 GPU 请求不会静默 fallback 到 CPU。

## FAQ

**为什么它的 covariance 不列在 HC/cluster/DK 中？**  Fama-MacBeth inference 来自 $\widehat\beta_t$ 的时间序列，而不是 observation residual。

**样本过少的 period 如何处理？**  在 coefficient average 前被排除；若没有有效 period 或保留不足两个 period，则 fit 报错。

## External Validation

`dev/tests/test_panel_p2.py` 维护 estimator、covariance、filtering 与 prediction regressions，formula coverage 位于 `dev/tests/test_panel_formula.py`。Stage-C residual-covariance external suite **不宣称**本 estimator 与 `linearmodels` parity，因为当前 beta-series covariance 是 estimator-specific contract。

## References

Fama and MacBeth (1973), *Risk, Return, and Equilibrium: Empirical Tests*；Newey and West (1987)。
