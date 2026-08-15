# PooledOLS

> 语言：中文  
> 最后更新：2026-08-15  
> 切换：[English](../../en/panel/pooled-ols.md)

## Overview

`PooledOLS` 对 stacked panel 拟合一个公共线性回归，并始终包含截距。`entity_ids` 只是 panel fit statistics 与 diagnostics 的可选 metadata，不参与 coefficient estimation。

## Path

实现：`statgpu/panel/_pooled.py`。

## Objective and Estimator

令 $Z=[\mathbf 1,X]$，则

$$
\widehat\beta_{\mathrm{pooled}}
=\arg\min_\beta\|y-Z\beta\|_2^2
=(Z^\top Z)^+Z^\top y.
$$

## Covariance and Inference

covariance 使用 level design $Z$，见 [面板 covariance](covariance.md)。`cov_type="hac"` 是 legacy row-order Bartlett/Newey-West HAC；Driscoll-Kraay 则先按 `time_index` 聚合 influence score。

## Parameters

| 参数 | 默认值 | 可选值 / 约束 | 含义 |
|---|---:|---|---|
| `cov_type` | `"nonrobust"` | `nonrobust`、`robust`/`hc1`、`hc0`、`hc2`、`hc3`、`clustered`、`hac`、`driscoll-kraay`/`dk`/`kernel` | covariance estimator。 |
| `alpha` | `0.05` | 有限且严格位于 0 与 1 之间 | 置信区间显著性水平；`0.05` 对应 95% 区间。 |
| `bandwidth` | `None` | `None` 或非负整数 | HAC/DK bandwidth。legacy HAC 的有效 lag 最多为 $n-1$；DK 使用 [面板 covariance](covariance.md) 中的 observed-period 规则。 |
| `kernel` | `"bartlett"` | `hac` 只允许 Bartlett；DK 还支持 Parzen 与 QS aliases | HAC/DK kernel。 |
| `device` | `"auto"` | `auto`、`cpu`、`cuda`、`torch` | 数值 backend/device。 |
| `n_jobs` | `None` | integer 或 `None` | 共享并行参数。 |
| `group_debias` | `False` | boolean；仅 clustered covariance 使用 | small-group cluster correction。 |

```python
model.fit(X, y, cluster=None, time_index=None, entity_ids=None)
```

clustered covariance 需要 `cluster`；Driscoll-Kraay 需要 `time_index`。`entity_ids` 用于 standardized within/between $R^2$ 与 Breusch-Pagan LM。

## CPU and GPU Example

```python
from statgpu.panel import PooledOLS

cpu = PooledOLS(device="cpu").fit(X, y)
cuda = PooledOLS(device="cuda").fit(X, y)
torch = PooledOLS(device="torch").fit(X, y)
```

显式 GPU 请求不会静默 fallback 到 CPU。

## Formula Example

假设 `df` 包含 `y`、`x1` 与 `x2` 列。

```python
from statgpu.panel import PooledOLS

model = PooledOLS().fit(
    formula="y ~ x1 + x2",
    data=df,
)
```

`PooledOLS` 始终包含截距，因此显式 no-intercept formula 会被拒绝。

## Outputs

public 结果包括 `coef_`、`bse_`、`tvalues_`、`pvalues_`、`conf_int_`、`rsquared`、`fit_statistics_`、`nobs` 与 `df_resid`。提供 `entity_ids` 后可调用 `breusch_pagan_lm_test()`；见 [面板 diagnostics](diagnostics.md)。

## Numerical and Strict Behavior

不支持的 covariance 输入不会走 approximate fallback：缺少所需 `cluster`/`time_index`、cluster shape 无效或显式 GPU backend 不可用都会报错。精确 rank deficiency 使用 [面板 covariance](covariance.md) 的统一 inference 契约。

## FAQ

**提供 `entity_ids` 会改变 coefficient 吗？**  不会，只启用 panel-aware statistics 与 diagnostics。

**`hac` 与 Driscoll-Kraay 相同吗？**  不同。legacy HAC 是 row-order Newey-West；DK 先按时间标签聚合 score。

## External Validation

`linearmodels==7.0` 检查 public estimator 的 Driscoll-Kraay coefficient、covariance、BSE，以及 group-debiased clustered covariance。coefficient 使用 `rtol=2e-10, atol=2e-11`；covariance/BSE 使用 `rtol=5e-9, atol=5e-11`。definition-level HC、cluster、DK、default-bandwidth 与 R `sandwich` 检查见 [validation matrix](covariance.md#validation-matrix)。

Stage-C 物理 runner 另行使用默认 `rtol=5e-6, atol=5e-7` 比较 PooledOLS 的 CuPy/Torch 与 NumPy；实际 `max_abs_differences` 保存在 `results/pr126_p100_fresh/panel_stage_c_correctness_p100.json`。

## 参考（References）

- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). The MIT Press.

covariance 专门参考文献见 [面板 covariance](covariance.md)。
