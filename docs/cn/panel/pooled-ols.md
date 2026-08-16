# PooledOLS

> 语言：中文  
> 最后更新：2026-08-16  
> 切换：[English](../../en/panel/pooled-ols.md)

## Overview

`PooledOLS` 把所有 panel observation 直接堆叠起来，拟合一个具有公共截距和公共斜率的普通线性回归。它**不会**去除 entity effect 或 time effect，因此适用于本来就希望所有观测共享同一个 pooled conditional-mean relationship 的场景。

`entity_ids` 是可选的：提供它不会改变 coefficient estimate，但可以额外计算 panel-specific fit statistics，并启用 Breusch-Pagan LM diagnostic。

## Path

实现：`statgpu/panel/_pooled.py`。

## Statistical Model and Identification

Pooled linear model 可以写成

$$
y_{it}=\alpha+x_{it}^{\top}\beta+u_{it}.
$$

要把 $\beta$ 解释为 pooled conditional-mean slope，一个常见的充分条件是

$$
E(u_{it}\mid X_i)=0,
$$

其中 $X_i=(x_{i1},\ldots,x_{iT_i})$。因此，任何没有显式建模的 entity 或 time heterogeneity 都会被并入 composite error $u_{it}$。

例如，如果真实数据满足

$$
y_{it}=\alpha+x_{it}^{\top}\beta+a_i+\varepsilon_{it},
$$

那么 pooled OLS 并不会消除 $a_i$。若希望仍然识别同一个 structural $\beta$，combined error $a_i+\varepsilon_{it}$ 必须与 regressors 正交。如果 entity heterogeneity 与 regressor history 相关，pooled OLS 一般不会识别与 fixed-effects estimator 相同的 slope。

HC、clustered、HAC 或 Driscoll-Kraay 等 covariance choice 只改变 uncertainty 的计算方式，不能修复 mean model 中外生性条件失效造成的识别问题。

## Estimator

令 $Z=[\mathbf 1,X]$，则

$$
\widehat\beta_{\mathrm{pooled}}
=\arg\min_\beta\|y-Z\beta\|_2^2
=(Z^\top Z)^+Z^\top y.
$$

因此，coefficient estimate 就是把所有 panel rows 当作一个普通回归样本后得到的 OLS 结果。

## Covariance and Inference

`cov_type` 只改变 standard error 的计算方式，不改变 OLS coefficient estimate。除了 nonrobust 和 HC covariance 外，`PooledOLS` 还支持 clustered covariance，以及两种考虑时间相关性的方式：

- `cov_type="hac"` 把 observations 看作一条有顺序的序列并应用 Bartlett/Newey-West HAC。若提供 `time_index`，会先按其时间顺序排序；否则直接使用输入数据的行顺序。numeric 与 datetime label 使用自然顺序；ordered pandas categorical 使用用户声明的 category order，因此 `t1, t2, t10` 这类标签不会被静默改成字符串词典序。
- `cov_type="driscoll-kraay"` 按 `time_index` 将 observations 分到各 period，并先在 period 内聚合其 covariance contribution，再对跨期 lag 加权。

因此这两种 covariance 不能互换理解。完整公式见 [面板 covariance](covariance.md)。

## Parameters

| 参数 | 默认值 | 可选值 / 约束 | 含义 |
|---|---:|---|---|
| `cov_type` | `"nonrobust"` | `nonrobust`、`robust`/`hc1`、`hc0`、`hc2`、`hc3`、`clustered`、`hac`、`driscoll-kraay`/`dk`/`kernel` | coefficient standard error 的计算方式。 |
| `alpha` | `0.05` | 有限且严格位于 0 与 1 之间 | 置信区间显著性水平；`0.05` 对应 95% 区间。 |
| `bandwidth` | `None` | `None` 或非负整数 | HAC/DK 的 lag 或 smoothing bandwidth。legacy HAC 最多使用 $n-1$ 个 lag；DK 的规则见 [面板 covariance](covariance.md)。 |
| `kernel` | `"bartlett"` | `hac` 只允许 Bartlett；DK 还支持 Parzen 与 QS aliases | HAC/DK 使用的 kernel。 |
| `device` | `"auto"` | `auto`、`cpu`、`cuda`、`torch` | 数值计算运行在哪个 backend/device。 |
| `n_jobs` | `None` | integer 或 `None` | 共享并行参数。 |
| `group_debias` | `False` | boolean；仅 clustered covariance 使用 | 是否应用 small-number-of-clusters correction。 |

```python
model.fit(X, y, cluster=None, time_index=None, entity_ids=None)
```

使用 clustered covariance 时传入 `cluster`；使用 Driscoll-Kraay 时必须传入 `time_index`。对 legacy HAC，`time_index` 是可选的；提供后，它决定 HAC 使用的 observation ordering。若还希望得到 standardized within/between $R^2$ 或 Breusch-Pagan LM test，则提供 `entity_ids`。

## CPU and GPU Example

```python
from statgpu.panel import PooledOLS

cpu = PooledOLS(device="cpu").fit(X, y)
cuda = PooledOLS(device="cuda").fit(X, y)
torch = PooledOLS(device="torch").fit(X, y)
```

若显式指定的 GPU backend 不可用，`.fit()` 会直接报错，而不是切换到 CPU。

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

常用结果包括 `coef_`、`bse_`、`tvalues_`、`pvalues_`、`conf_int_`、`rsquared`、`fit_statistics_`、`nobs` 与 `df_resid`。提供 `entity_ids` 后还可调用 `breusch_pagan_lm_test()`；见 [面板 diagnostics](diagnostics.md)。

## Numerical and Strict Behavior

covariance 所需的附加信息会在计算前检查。例如，clustered covariance 缺少 `cluster`、Driscoll-Kraay 缺少 `time_index`，或 cluster 数组长度/形状不匹配时，都会直接报错，而不会自动改用另一种 covariance。legacy HAC 的 time metadata 若含 missing/non-finite value 也会 fail closed，而不会猜测排序。

每次新的 `fit()` 都会先失效上一轮 fitted/inference state。如果 refit 在后续任意阶段失败，已经部分写入的新输出也会被清理；此后 `predict()` 与 `summary()` 会把 estimator 视为未拟合状态。

如果 design matrix 精确 rank deficient，模型仍可能得到 fitted values，但 coefficient vector 不唯一。statgpu 会对该次拟合整体关闭 coefficient-level standard error、检验、p-value 与 confidence interval，而不是从任意一种 coefficient representation 中继续做推断；详见 [面板 covariance](covariance.md)。

显式指定 `device="cuda"` 或 `device="torch"` 时也要求对应 backend 可用，否则直接报错而不是切换到 CPU。

## FAQ

**提供 `entity_ids` 会改变 coefficient 吗？**  不会；它只额外启用 panel-aware fit statistics 与 diagnostics。

**`hac` 与 Driscoll-Kraay 相同吗？**  不同。HAC 把 observations 当作一条有顺序的序列；Driscoll-Kraay 则先在用户提供的各 time period 内聚合 observation contribution。

## External Validation

我们将 `PooledOLS` 与 `linearmodels==7.0` 比较，覆盖 Driscoll-Kraay coefficient、covariance、BSE，以及 group-debiased clustered covariance。coefficient 使用 `rtol=2e-10, atol=2e-11`；covariance/BSE 使用 `rtol=5e-9, atol=5e-11`。HC、cluster、Driscoll-Kraay、default bandwidth 与 R `sandwich` 的定义级检查见 [validation matrix](covariance.md#validation-matrix)。

GPU 一致性单独验证：CuPy 与 Torch 输出分别和 NumPy 比较，默认容差为 `rtol=5e-6, atol=5e-7`；实际最大差异保存在 PR #126 的 physical validation artifacts 中。专用的 `dev/benchmarks/validate_panel_hac_chronology_gpu.py` gate 还会在最终 exact source 上验证 ordered-categorical legacy-HAC chronology、lexical-order negative control、formula missing-row alignment，以及 requested/executed CuPy/Torch backend identity。

## 参考（References）

- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). The MIT Press.

covariance 专门参考文献见 [面板 covariance](covariance.md)。
