# PanelOLS

> 语言：中文  
> 最后更新：2026-08-15  
> 切换：[English](../../en/panel/panel-ols.md)

## Overview

`PanelOLS` 可以拟合无 fixed effects、单向 fixed effects 或双向 fixed effects 的线性面板回归。加入 fixed effects 后，coefficient 由去除所选 entity 和/或 time effects 后仍然存在的 variation 来识别。

实现时 statgpu 直接对数据做相应变换，而不是显式生成大量 dummy columns。这只改变数值实现方式，不改变 fixed-effects model 的统计含义。

## Path

实现：`statgpu/panel/_fixed_effects.py`。

## Statistical Model and Identification

对于 entity 与 time fixed effects，经典 fixed-parameter model 写为

$$
y_{it}=x_{it}^{\top}\beta+a_i+\gamma_t+\varepsilon_{it}.
$$

这里的 $a_i$ 与 $\gamma_t$ 是 **fixed but unknown nuisance parameters**。它们不需要来自某个概率分布，fixed-effects model 也不要求这些 nuisance effects 与 regressor history $X_i=(x_{i1},\ldots,x_{iT_i})$ 正交。

对于 static linear panel model，一个常见的充分外生性条件是

$$
E\!\left(\varepsilon_{it}\mid X_i,a_i,\gamma_1,\ldots,\gamma_T\right)=0.
$$

在 fixed-parameter formulation 下，这里的条件期望理解为给定已经包含在模型中的 fixed effects。核心限制因此落在 idiosyncratic error $\varepsilon_{it}$ 上，而不是要求 $a_i$ 与 regressors 独立或不相关。

$\beta$ 只能由去除 fixed effects 后仍然存在的 regressor variation 识别。例如，在 entity fixed effects 下，完全不随时间变化的 regressor 会被 entity effect 吸收，因此无法单独识别其 slope。当 `entity_effects=False` 且 `time_effects=False` 时，模型退化为普通 level regression，此时上面的 fixed-effects interpretation 不再适用。

## Estimator

令 $F$ 为包含的 fixed-effect design，$M_F=I-F(F^\top F)^+F^\top$。则

$$
\widehat\beta_{\mathrm{FE}}
=\arg\min_\beta\|M_F(y-X\beta)\|_2^2
=(X^\top M_FX)^+X^\top M_Fy.
$$

只有 entity fixed effects 时，这就是通常的 within-entity demeaning。双向 fixed effects 时，statgpu 会反复去除 entity mean 与 time mean，直到满足 `demean_tol`。如果在 `demean_max_iter` 次迭代内仍未收敛，`.fit()` 会直接报错，而不是返回尚未充分去均值的近似结果。

## Covariance and Inference

standard error 使用与 coefficient estimation 完全相同的 transformed regressors 和 residuals。记变换后的 design 为 $Z=M_FX$，residual 为 $e=M_F(y-X\widehat\beta_{\mathrm{FE}})$，则 nonrobust、HC、clustered 与 Driscoll-Kraay 的统一公式见 [面板 covariance](covariance.md)。

Driscoll-Kraay 的 degrees-of-freedom correction 还需要计入被 fixed effects 占用的维度，对应的 fixed-effect rank 为

$$
r_F=\begin{cases}N,&\text{仅 entity},\\T,&\text{仅 time},\\N+T-C,&\text{双向},\end{cases}
$$

其中 $C$ 是观测 entity-time incidence graph 的 connected-component 数。

公开的 residual degrees of freedom 使用同一个 nuisance-effect rank：

$$
df_{\mathrm{resid}}=n-\operatorname{rank}(Z)-r_F.
$$

这一统一定义同时决定公开的 `df_resid`、nonrobust residual-variance scale、HC1 (`robust`) finite-sample correction，以及 nonrobust Student-$t$ inference。因此它与对应的显式 fixed-effect dummy regression 的 residual df 完全一致，而不是在 within transformation 后再使用 `N-1`/`T-1` 的简化计数。

## Parameters

| 参数 | 默认值 | 可选值 / 约束 | 含义 |
|---|---:|---|---|
| `entity_effects` | `False` | boolean | 是否去除 entity-specific level effect。 |
| `time_effects` | `False` | boolean | 是否去除 time-specific level effect。 |
| `cov_type` | `"nonrobust"` | `nonrobust`、`robust`/`hc1`、`hc0`、`hc2`、`hc3`、`clustered`、`driscoll-kraay`/`dk`/`kernel` | coefficient standard error 的计算方式。 |
| `alpha` | `0.05` | 有限且严格位于 0 与 1 之间 | 置信区间显著性水平；`0.05` 对应 95% 区间。 |
| `device` | `"auto"` | `auto`、`cpu`、`cuda`、`torch` | 数值计算运行在哪个 backend/device。 |
| `n_jobs` | `None` | integer 或 `None` | 共享并行参数。 |
| `bandwidth` | `None` | `None` 或非负整数；仅 DK 使用 | Driscoll-Kraay smoothing bandwidth；`None` 使用文档中的自动规则。 |
| `kernel` | `"bartlett"` | Bartlett/Newey-West、Parzen/Gallant、QS/Quadratic-Spectral/Andrews aliases | Driscoll-Kraay kernel。 |
| `group_debias` | `False` | boolean；仅 clustered covariance 使用 | 是否应用 small-number-of-clusters correction。 |
| `demean_max_iter` | `1_000_000` | 正整数 | 双向去均值最多允许的迭代次数。 |
| `demean_tol` | `1e-10` | 有限且严格为正 | 双向去均值的收敛容差。 |

```python
model.fit(X, y, entity_ids=None, time_ids=None, cluster=None)
```

通过 constructor 打开某个 fixed effect 时，需要提供对应的 `entity_ids` 和/或 `time_ids`。clustered covariance 还需要 `cluster`；Driscoll-Kraay 需要 `time_ids`。

## CPU and GPU Example

```python
from statgpu.panel import PanelOLS

cpu = PanelOLS(entity_effects=True, device="cpu").fit(X, y, entity_ids=entity_ids)
cuda = PanelOLS(entity_effects=True, device="cuda").fit(X, y, entity_ids=entity_ids)
torch = PanelOLS(entity_effects=True, device="torch").fit(X, y, entity_ids=entity_ids)
```

`device="cuda"` 需要 CuPy/CUDA，`device="torch"` 需要 Torch CUDA。如果显式要求的 backend 不可用，`.fit()` 会报错，而不是自动切换到 CPU。

## Formula Example

假设 `df` 包含 `y`、`x1`、`x2`、`entity` 与 `time` 列。

```python
from statgpu.panel import PanelOLS

# 使用 pipe syntax 的双向 fixed effects。
two_way = PanelOLS().fit(
    formula="y ~ x1 + x2 | entity + time",
    data=df,
)

# 使用 effect token 表示相同的 fixed-effect structure。
two_way_tokens = PanelOLS().fit(
    formula="y ~ x1 + x2 + EntityEffects + TimeEffects",
    data=df,
)

# 无截距的普通 level regression。
level_no_intercept = PanelOLS().fit(
    formula="y ~ 0 + x1 + x2",
    data=df,
)
```

fixed effects 可以使用 pipe syntax，也可以使用 effect token，但同一个 formula 中不能混用。pipe syntax 最多接受两个 fixed-effect variables。pipe 中明确命名的 entity/time 列是权威来源：如果同时显式传入 `entity_ids` 或 `time_ids`，它们必须与 formula 过滤后保留的对应列完全一致；冲突的重复来源会直接报错。

## Outputs

常用结果包括 `coef_`、`bse_`、`tvalues_`、`pvalues_`、`conf_int_`、`rsquared_within`、`fit_statistics_`、`nobs` 与 `df_resid`。`summary()` 返回 panel summary。Pooling F 与 Hausman test 见 [面板 diagnostics](diagnostics.md)。

## Numerical and Strict Behavior

双向去均值必须真正收敛。如果在 `demean_max_iter` 内没有达到 `demean_tol`，statgpu 会报错，而不会把最后一次迭代当作近似结果返回。

如果去除 fixed effects 后的 regressors 精确共线，fitted values 仍可能是确定的，但 coefficient vector 不唯一。statgpu 会对该次拟合整体关闭 coefficient-level standard error、检验、p-value 与 confidence interval，而不是从任意一种 coefficient representation 中继续做推断；详见 [面板 covariance](covariance.md)。

对于双向 fixed-effect prediction，只有当请求的 entity/time label 能由已拟合数据中的 fixed effects 唯一确定时，才会加入 stored effects。一个已知 label 与一个未知 label 的组合，或来自不兼容 fitted components 的 label 组合，会直接报错。如果两个 label 都从未见过，则没有任何已估计 fixed effect 可加入，prediction 只使用线性部分 $X\widehat\beta$。

Formula 也会对不支持的写法明确报错，例如混用 pipe 与 effect-token syntax、通过 pipe 指定超过两个 fixed effects，或 fixed-effect formula 中没有任何 non-intercept regressor。

## FAQ

**为什么双向 demeaning 不直接返回最后一个 iterate？**  因为没有收敛的去均值会改变实际拟合的回归。只有达到用户要求的 tolerance 后结果才会返回。

**robust covariance 会把 `fit_statistics_.f_statistic` 改成 robust Wald test 吗？**  不会。该字段仍表示 classical joint test of slopes；见 [fit statistics](fit-statistics.md)。

## External Validation

我们将单向和双向 fixed-effect Driscoll-Kraay 结果与 `linearmodels==7.0` 比较：coefficient 使用 `rtol=2e-10, atol=2e-11`，covariance/BSE 使用 `rtol=5e-9, atol=5e-11`。无 fixed effects 的 level OLS 路径与 `statsmodels==0.14.6` 比较；单向 fixed-effect coefficient 还与 R `plm==2.6-7` 比较。共享 covariance 与 R tolerance 见 [validation matrix](covariance.md#validation-matrix)。

GPU 一致性单独验证：CuPy 与 Torch 输出分别和 NumPy 比较，默认容差为 `rtol=5e-6, atol=5e-7`；实际最大差异保存在 `results/pr126_p100_fresh/panel_stage_c_correctness_p100.json`。

## 参考（References）

- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). The MIT Press.

covariance 与 diagnostic 的专门参考文献见 [面板 covariance](covariance.md) 和 [面板 diagnostics](diagnostics.md)。
