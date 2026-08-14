# FirstDifferenceOLS

> 语言：中文  
> 最后更新：2026-08-14  
> 切换：[English](../../en/panel/first-difference-ols.md)

## Overview

`FirstDifferenceOLS` 对每个 entity 的连续已观测时期做差分，从而消除 time-invariant entity effects，再在无截距的 differenced fit space 上做 OLS。

## Path

实现：`statgpu/panel/_first_diff.py`。

## Objective and Estimator

对 entity $i$ 内相邻的已观测时期，

$$
\Delta y_{it}=y_{it}-y_{i,t^-},
\qquad
\Delta x_{it}=x_{it}-x_{i,t^-}.
$$

且

$$
\widehat\beta_{\mathrm{FD}}
=\arg\min_\beta\|\Delta y-\Delta X\beta\|_2^2
=(\Delta X^\top\Delta X)^+\Delta X^\top\Delta y.
$$

## Covariance and Inference

covariance 使用 $Z=\Delta X$，见 [面板 covariance](covariance.md)。支持 nonrobust 与 HC0/1/2/3。

## Parameters

| 参数 | 默认值 | 可选值 / 约束 | 含义 |
|---|---:|---|---|
| `cov_type` | `"nonrobust"` | `nonrobust`、`robust`/`hc1`、`hc0`、`hc2`、`hc3` | differenced fit space 上的 covariance estimator。 |
| `alpha` | `0.05` | 显著性水平；`0.05` 对应 95% 置信区间。 | 置信区间 level 控制。 |
| `device` | `"auto"` | `auto`、`cpu`、`cuda`、`torch` | 数值 backend/device。 |
| `n_jobs` | `None` | integer 或 `None` | 共享并行参数。 |

```python
model.fit(X, y, entity_ids=entity_ids, time_ids=None)
```

`entity_ids` 必需。提供 `time_ids` 时 `(entity_id, time_id)` 必须唯一。差分使用相邻的**已观测**时间；calendar gap 不会被补齐，也不会除以 gap 长度。ordered categorical time label 保留声明顺序。

## CPU and GPU Example

```python
from statgpu.panel import FirstDifferenceOLS

cpu = FirstDifferenceOLS(device="cpu").fit(X, y, entity_ids=entity_ids, time_ids=time_ids)
cuda = FirstDifferenceOLS(device="cuda").fit(X, y, entity_ids=entity_ids, time_ids=time_ids)
torch = FirstDifferenceOLS(device="torch").fit(X, y, entity_ids=entity_ids, time_ids=time_ids)
```

## Outputs

public 结果包括 `coef_`、`bse_`、`tvalues_`、`pvalues_`、`conf_int_`、`rsquared`、`fit_statistics_`、`nobs` 与 `df_resid`。

## Numerical and Strict Behavior

当前 transformation 不会虚构缺失 calendar period。提供 `time_ids` 时 duplicate entity-time observation 会 fail closed。不存在 silent approximate-inference 或 backend fallback。

## FAQ

**跨两个 calendar period 的 gap 会形成 two-step difference 吗？**  不会；按时间排序后只对连续已观测 rows 做差。

**差分后是否估计 intercept？**  不估计。

## External Validation

`statsmodels==0.14.6` 在完全相同的 differenced sample 上检查 HC0/HC2/HC3。coefficient 使用 `rtol=5e-10, atol=5e-12`；covariance 与 BSE 使用 `rtol=5e-9, atol=5e-11`。共享 covariance definition 检查见 [validation matrix](covariance.md#validation-matrix)。

Stage-C 物理 runner 另行使用默认 `rtol=5e-6, atol=5e-7` 比较 FirstDifferenceOLS 的 CuPy/Torch 与 NumPy。

## References

Wooldridge (2010), *Econometric Analysis of Cross Section and Panel Data*。
