# PooledOLS

> 语言：中文  
> 最后更新：2026-08-14  
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

| 参数 | 含义 |
|---|---|
| `cov_type` | nonrobust、HC、clustered、legacy HAC 或 DK。 |
| `bandwidth`, `kernel` | 适用时的 HAC/DK 参数。 |
| `group_debias` | cluster small-group correction。 |
| `alpha` | 置信区间显著性水平。 |
| `device` | `auto`、`cpu`、`cuda` 或 `torch`。 |
| `n_jobs` | 共享并行参数。 |

```python
model.fit(X, y, cluster=None, time_index=None, entity_ids=None)
```

clustered covariance 需要 `cluster`；Driscoll-Kraay 需要 `time_index`。`entity_ids` 用于 standardized within/between $R^2$ 与 Breusch-Pagan LM。Formula 必须保留模型截距。

## CPU and GPU Example

```python
from statgpu.panel import PooledOLS

cpu = PooledOLS(device="cpu").fit(X, y)
cuda = PooledOLS(device="cuda").fit(X, y)
torch = PooledOLS(device="torch").fit(X, y)
```

显式 GPU 请求不会静默 fallback 到 CPU。

## Outputs

public 结果包括 `coef_`、`bse_`、`tvalues_`、`pvalues_`、`conf_int_`、`rsquared`、`fit_statistics_`、`nobs` 与 `df_resid`。提供 `entity_ids` 后可调用 `breusch_pagan_lm_test()`；见 [面板 diagnostics](diagnostics.md)。

## Numerical and Strict Behavior

不支持的 covariance 输入不会走 approximate fallback：缺少所需 `cluster`/`time_index`、cluster shape 无效或显式 GPU backend 不可用都会报错。精确 rank deficiency 使用 [面板 covariance](covariance.md) 的统一 inference 契约。

## FAQ

**提供 `entity_ids` 会改变 coefficient 吗？**  不会，只启用 panel-aware statistics 与 diagnostics。

**`hac` 与 Driscoll-Kraay 相同吗？**  不同。legacy HAC 是 row-order Newey-West；DK 先按时间标签聚合 score。

## External Validation

Estimator-level DK 与 clustered covariance 在 `dev/tests/test_panel_stage_c_linearmodels_estimators.py` 中与 `linearmodels==7.0` 比较；重合的 OLS 定义也使用 `statsmodels==0.14.6`。R `plm==2.6-7` / `sandwich==3.1-3` 检查位于 `dev/tests/test_panel_stage_c_r_external.py`。GPU 物理验证见 `results/pr126_p100_fresh/validation_summary.txt`。

## References

Wooldridge (2010), *Econometric Analysis of Cross Section and Panel Data*；covariance references 见 [面板 covariance](covariance.md)。
