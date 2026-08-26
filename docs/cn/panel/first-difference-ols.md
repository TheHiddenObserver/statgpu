# FirstDifferenceOLS

> 语言：中文  
> 最后更新：2026-08-19  
> 切换：[English](../../en/panel/first-difference-ols.md)

## Overview

`FirstDifferenceOLS` 对同一 entity 的相邻**已观测**时期做一阶差分，用当前值减去上一条已观测值，从而消除 time-invariant entity effect；随后对差分后的 outcome 和 predictors 做无截距 OLS。

它可以和 fixed-effects estimator 从同一个 one-way fixed-parameter entity-effect model 出发；两者的区别在于使用什么 transformation 来消除 entity effect。

## Path

实现：`statgpu/panel/_first_diff.py`。

## Statistical Model and Identification

从 one-way fixed-parameter panel model 出发：

$$
y_{it}=x_{it}^{\top}\beta+a_i+\varepsilon_{it},
$$

其中 $a_i$ 是 fixed but unknown、且不随时间变化的 entity effect。与 fixed-effects model 一样，不要求 $a_i$ 与 regressor history 正交。对于 static model，一个常见的充分外生性条件是

$$
E\!\left(\varepsilon_{it}\mid X_i,a_i\right)=0,
\qquad
X_i=(x_{i1},\ldots,x_{iT_i}).
$$

做一阶差分后，$a_i$ 被精确消除：

$$
\Delta y_{it}=\Delta x_{it}^{\top}\beta+\Delta\varepsilon_{it}.
$$

因此 slope 由同一 entity 内随时间发生的变化识别。一个在 entity 内始终不变的 regressor 差分后恒为 0，无法在该 specification 中识别 slope。level model 中的公共 intercept 也会被差分消掉，所以最终回归不估计 intercept。

## Estimator

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

这里 $t^-$ 表示该 entity 的上一条**已观测**时间。缺失的 calendar period 不会被补出来，差分也不会再除以 calendar gap 的长度。

## Covariance and Inference

standard error 基于差分后的回归 $(\Delta y,\Delta X)$ 计算。支持 nonrobust 以及 HC0/HC1/HC2/HC3；统一公式见 [面板 covariance](covariance.md)。

改变 covariance estimator 只改变 differenced regression 的 uncertainty，并不能替代将 $\beta$ 解释为基础 panel model slope 时所需的外生性条件。

## Parameters

| 参数 | 默认值 | 可选值 / 约束 | 含义 |
|---|---:|---|---|
| `cov_type` | `"nonrobust"` | `nonrobust`、`robust`/`hc1`、`hc0`、`hc2`、`hc3` | 差分后如何计算 coefficient standard error。 |
| `alpha` | `0.05` | 有限且严格位于 0 与 1 之间 | 置信区间显著性水平；`0.05` 对应 95% 区间。 |
| `device` | `"auto"` | `auto`、`cpu`、`cuda`、`torch` | 数值计算运行在哪个 backend/device。 |
| `n_jobs` | `None` | integer 或 `None` | 共享并行参数。 |

```python
model.fit(X, y, entity_ids=entity_ids, time_ids=None)
```

`entity_ids` 必需。提供 `time_ids` 后，statgpu 会先按时间对每个 entity 的记录排序；此时每个 `(entity_id, time_id)` 组合必须唯一。numeric/datetime `time_ids` 使用自然顺序，ordered categorical 使用用户声明的 category 顺序。普通 string labels 使用字符串字典序；其他非 categorical object labels 按其可比较值排序，不能相互比较时会直接报错。若字符串字典序不是实际 chronology，应改用 ordered categorical 或 numeric/datetime key。

## CPU and GPU Example

```python
from statgpu.panel import FirstDifferenceOLS

cpu = FirstDifferenceOLS(device="cpu").fit(X, y, entity_ids=entity_ids, time_ids=time_ids)
cuda = FirstDifferenceOLS(device="cuda").fit(X, y, entity_ids=entity_ids, time_ids=time_ids)
torch = FirstDifferenceOLS(device="torch").fit(X, y, entity_ids=entity_ids, time_ids=time_ids)
```

若显式指定的 GPU backend 不可用，`.fit()` 会直接报错，而不是切换到 CPU。

## Formula Example

假设 `df` 包含 `y`、`x1`、`x2`、`entity` 与 `time` 列。

```python
from statgpu.panel import FirstDifferenceOLS

model = FirstDifferenceOLS().fit(
    formula="y ~ x1 + x2 - 1",
    data=df,
    entity_ids=df["entity"],
    time_ids=df["time"],
)
```

这里显式去掉 intercept，因为 `FirstDifferenceOLS` 在差分后的回归中不估计截距。

## Outputs

常用结果包括 `coef_`、`bse_`、`tvalues_`、`pvalues_`、`conf_int_`、`rsquared`、`fit_statistics_`、`nobs` 与 `df_resid`。其中 `nobs` 是实际进入最终回归的差分观测数，因此会小于原始 panel 行数。

## Numerical and Strict Behavior

提供 `time_ids` 时，如果同一 entity 在同一 time 出现重复观测，模型会报错，因为无法唯一确定时间排序后的差分。calendar gap 会原样保留：statgpu 只对相邻的已观测 rows 做差，不会插入缺失时期，也不会按经过的时间长度重新缩放差分。

差分回归使用共享的 certified panel least-squares policy。若 response projection 存在 cancellation risk，会进入 magnitude-tiered reduction；若某个非零 coefficient 已低于 float64 projection 可可靠分辨的尺度，并且候选解显著违反 least-squares stationarity，`.fit()` 会抛出 `FloatingPointError`，而不是返回有限但不可靠的 coefficient。这与精确共线不同：若差分 predictors 精确共线，fitted values 仍可能得到，但 coefficient vector 不唯一，因此 coefficient-level standard error、检验、p-value 与 confidence interval 会关闭。

legacy `rsquared` 也采用 range-safe centering。当物理 subtraction $\Delta y-\overline{\Delta y}$ 在 float64 边界附近会溢出时，response 与 residual 会先放到同一个 dimensionless centering scale，再形成 scale-invariant 的 $R^2$ ratio；普通尺度下的 centering 不变。不支持的 covariance choice 或不可用的显式 GPU backend 仍直接报错。

## FAQ

**跨两个 calendar period 的 gap 会形成 two-step difference 吗？**  不会。按时间排序后，模型只用当前观测减去上一条已观测记录，不论两者在 calendar 上相隔多久。

**差分后是否估计 intercept？**  不估计。

## External Validation

我们在 `statsmodels==0.14.6` 中构造完全相同的差分样本，再比较 coefficient 以及 HC0/HC2/HC3 covariance 和 standard error。coefficient 使用 `rtol=5e-10, atol=5e-12`；covariance/BSE 使用 `rtol=5e-9, atol=5e-11`。共享 covariance 检查见 [validation matrix](covariance.md#validation-matrix)。

GPU 一致性单独验证：CuPy 与 Torch 分别和 NumPy 比较，默认容差为 `rtol=5e-6, atol=5e-7`。当前 exact-head physical gate 还会验证 shared coefficient-resolution fail-closed，以及一个物理 centering 会超出 float64 range 的极端 differenced-response case。

## 参考（References）

- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). The MIT Press.
