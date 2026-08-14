# BetweenOLS

> 语言：中文  
> 最后更新：2026-08-14  
> 切换：[English](../../en/panel/between-ols.md)

## Overview

`BetweenOLS` 先将 panel 压缩为 entity mean，再对 entities 进行带截距的 OLS。其有效样本量是保留的 entity 数量。

## Path

实现：`statgpu/panel/_between.py`。

## Objective and Estimator

对 entity $i$，

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

## Covariance and Inference

covariance 在 entity-mean fit space 上计算。nonrobust 与 HC0/1/2/3 见 [面板 covariance](covariance.md)。

## Parameters

| 参数 | 默认值 | 可选值 / 约束 | 含义 |
|---|---:|---|---|
| `cov_type` | `"nonrobust"` | `nonrobust`、`robust`/`hc1`、`hc0`、`hc2`、`hc3` | entity-mean fit space 上的 covariance estimator。 |
| `alpha` | `0.05` | 有限且严格位于 0 与 1 之间 | 置信区间显著性水平；`0.05` 对应 95% 区间。 |
| `device` | `"auto"` | `auto`、`cpu`、`cuda`、`torch` | 数值 backend/device。 |
| `n_jobs` | `None` | integer 或 `None` | 共享并行参数。 |

```python
model.fit(X, y, entity_ids=entity_ids)
```

`entity_ids` 必需。模型始终包含截距；显式 no-intercept formula 会被拒绝。

## CPU and GPU Example

```python
from statgpu.panel import BetweenOLS

cpu = BetweenOLS(device="cpu").fit(X, y, entity_ids=entity_ids)
cuda = BetweenOLS(device="cuda").fit(X, y, entity_ids=entity_ids)
torch = BetweenOLS(device="torch").fit(X, y, entity_ids=entity_ids)
```

## Outputs

public 结果包括 `coef_`、`bse_`、`tvalues_`、`pvalues_`、`conf_int_`、`rsquared`、`fit_statistics_`、`nobs` 与 `df_resid`。

## Numerical and Strict Behavior

回归在 entity-mean fit space 中使用 rank-aware linear algebra。不存在 silent approximate-inference 或 backend fallback；不支持的 covariance choice 与不可用的显式 GPU backend 会报错。

## FAQ

**为什么 `nobs` 小于原始 row 数？**  `nobs` 是 between regression 实际使用的 entity-mean observation 数。

**观测更多的 entity 会自动获得更高权重吗？**  不会；当前 estimator 对每个 entity mean 使用一个 OLS observation。

## External Validation

`statsmodels==0.14.6` 在相同 entity-mean design 上检查 HC0/HC2/HC3。coefficient 使用 `rtol=5e-10, atol=5e-12`；covariance 与 BSE 使用 `rtol=5e-9, atol=5e-11`。共享 covariance definition 检查见 [validation matrix](covariance.md#validation-matrix)。

Stage-C 物理 runner 另行使用默认 `rtol=5e-6, atol=5e-7` 比较 BetweenOLS 的 CuPy/Torch 与 NumPy。

## 参考（References）

- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). The MIT Press.
