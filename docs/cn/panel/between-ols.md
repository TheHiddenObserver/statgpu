# BetweenOLS

> 语言：中文  
> 最后更新：2026-08-15  
> 切换：[English](../../en/panel/between-ols.md)

## Overview

`BetweenOLS` 先在每个 entity 内对各变量取时间均值，把面板数据压缩成“每个 entity 一行”，再对这些 entity 均值做带截距的 OLS 回归。它适合研究主要来自 **entity 之间长期平均差异** 的关系，而不是同一 entity 随时间的变化。

因此，最终回归的有效样本量是保留下来的 entity 数量，而不是原始 panel 的总行数。

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

也就是说，原始 panel 最终被转换成一个以 entity 均值为观测的普通横截面 OLS。

## Covariance and Inference

标准误直接基于这个 entity-mean 回归计算。`cov_type="nonrobust"` 使用通常的同方差 OLS covariance；`robust`/`hc1` 以及 HC0/HC2/HC3 提供异方差稳健版本。统一公式见 [面板 covariance](covariance.md)。

## Parameters

| 参数 | 默认值 | 可选值 / 约束 | 含义 |
|---|---:|---|---|
| `cov_type` | `"nonrobust"` | `nonrobust`、`robust`/`hc1`、`hc0`、`hc2`、`hc3` | entity 均值回归完成后如何计算 coefficient standard error。 |
| `alpha` | `0.05` | 有限且严格位于 0 与 1 之间 | 置信区间显著性水平；`0.05` 对应 95% 区间。 |
| `device` | `"auto"` | `auto`、`cpu`、`cuda`、`torch` | 数值计算运行在哪个 backend/device。 |
| `n_jobs` | `None` | integer 或 `None` | 共享并行参数。 |

```python
model.fit(X, y, entity_ids=entity_ids)
```

`entity_ids` 必需，因为模型需要先在每个 entity 内计算均值。

## CPU and GPU Example

```python
from statgpu.panel import BetweenOLS

cpu = BetweenOLS(device="cpu").fit(X, y, entity_ids=entity_ids)
cuda = BetweenOLS(device="cuda").fit(X, y, entity_ids=entity_ids)
torch = BetweenOLS(device="torch").fit(X, y, entity_ids=entity_ids)
```

若显式指定 `device="cuda"` 或 `device="torch"`，但对应 GPU backend 不可用，`.fit()` 会直接报错，而不是自动切换到 CPU。

## Formula Example

假设 `df` 包含 `y`、`x1`、`x2` 与 `entity` 列。

```python
from statgpu.panel import BetweenOLS

model = BetweenOLS().fit(
    formula="y ~ x1 + x2",
    data=df,
    entity_ids=df["entity"],
)
```

`BetweenOLS` 始终包含截距，因此显式 no-intercept formula 会被拒绝。

## Outputs

常用结果包括 `coef_`、`bse_`、`tvalues_`、`pvalues_`、`conf_int_`、`rsquared`、`fit_statistics_`、`nobs` 与 `df_resid`。其中 `nobs` 表示最终进入回归的 entity 均值个数。

## Numerical and Strict Behavior

如果 entity-level regressors 存在完全共线，statgpu 仍可用 least-squares solution 得到 fitted values，但 coefficient vector 不唯一。对于该次拟合，statgpu 会整体关闭 coefficient-level standard error、检验、p-value 与 confidence interval，而不是从任意一种 coefficient representation 中继续做推断。

不支持的 `cov_type` 会报错。类似地，若用户显式要求某个 GPU backend，该 backend 必须实际可用；statgpu 不会悄悄改用 CPU。

## FAQ

**为什么 `nobs` 小于原始数据行数？**  因为 `nobs` 是 between regression 使用的 entity 均值观测数，而不是原始 panel 行数。

**时间观测更多的 entity 会自动获得更高权重吗？**  不会。每个保留的 entity 在最终 OLS 中只贡献一个均值观测。

## External Validation

我们先在 statgpu 与 `statsmodels==0.14.6` 中构造完全相同的 entity-mean 回归，再比较 coefficient 以及 HC0/HC2/HC3 covariance 和 standard error。coefficient 使用 `rtol=5e-10, atol=5e-12`；covariance/BSE 使用 `rtol=5e-9, atol=5e-11`。共享 covariance 检查见 [validation matrix](covariance.md#validation-matrix)。

GPU 一致性单独验证：Stage-C physical validation 使用 `rtol=5e-6, atol=5e-7` 比较 CuPy、Torch 与 NumPy。

## 参考（References）

- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). The MIT Press.
