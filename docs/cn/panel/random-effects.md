# RandomEffects

> 语言：中文  
> 最后更新：2026-08-15  
> 切换：[English](../../en/panel/random-effects.md)

## Overview

`RandomEffects` 使用 Swamy-Arora 方法估计单向 random-intercept model 的 variance components，再通过 feasible GLS 得到 coefficient。与 fixed effects 不同，它把 entity-specific effect 看作随机成分，而不是为每个 entity 单独估计一个固定截距。

`cov_type` 只改变 GLS 拟合之后报告的 standard error 和检验，不会改变 Swamy-Arora variance components，也不会改变 coefficient estimate。

## Path

实现：`statgpu/panel/_random_effects.py`。

## Model and Estimator

$$
y_{it}=x_{it}^{\top}\beta+a_i+\varepsilon_{it},
\qquad
\operatorname{Var}(a_i)=\sigma_a^2,
\qquad
\operatorname{Var}(\varepsilon_{it})=\sigma_e^2.
$$

Swamy-Arora 先估计

$$
\widehat\sigma_e^2=\frac{RSS_W}{df_W},
\qquad
\bar T_H=\frac{N}{\sum_{i=1}^N T_i^{-1}},
$$

再计算

$$
\widehat\sigma_a^2=
\max\left\{0,\frac{RSS_B/df_B-\widehat\sigma_e^2}{\bar T_H}\right\}.
$$

其中 $df_W=n-r_W-r_E$。level design 有显式常数列时 $r_E=N$，否则 $r_E=N-1$；同时 $df_B=N-r_B$。如果辅助回归中存在重复或线性依赖的列，自由度按实际 numerical rank 计算，避免把同一个有效方向重复计数。

对 entity $i$，

$$
\theta_i=1-\sqrt{\frac{\widehat\sigma_e^2}{\widehat\sigma_e^2+T_i\widehat\sigma_a^2}},
$$

$$
y_{it}^*=y_{it}-\theta_i\bar y_i,
\qquad
x_{it}^*=x_{it}-\theta_i\bar x_i.
$$

这个 quasi-demeaning 会从每个观测中减去一部分 entity mean；减去多少由估计得到的 within/between variance components 以及该 entity 的观测数共同决定。

随后在 $(y^*,X^*)$ 上做 feasible GLS：

$$
\widehat\beta_{\mathrm{RE}}
=\arg\min_\beta\|y^*-X^*\beta\|_2^2
=(X^{*\top}X^*)^+X^{*\top}y^*.
$$

## Covariance and Inference

standard error 基于实际用于 GLS 的 quasi-demeaned data $(y^*,X^*)$ 计算。特别地，

$$
\widehat V_{\mathrm{nonrobust}}
=\widehat\sigma_*^2(X^{*\top}X^*)^+,
\qquad
\widehat\sigma_*^2=\frac{e^{*\top}e^*}{n-\operatorname{rank}(X^*)},
$$

其中 $e^*=y^*-X^*\widehat\beta_{\mathrm{RE}}$。HC0/1/2/3、clustered 与 Driscoll-Kraay 也都在同一个 transformed regression 上计算；见 [面板 covariance](covariance.md)。

## Parameters

| 参数 | 默认值 | 可选值 / 约束 | 含义 |
|---|---:|---|---|
| `cov_type` | `"nonrobust"` | `nonrobust`、`robust`/`hc1`、`hc0`、`hc2`、`hc3`、`clustered`、`driscoll-kraay`/`dk`/`kernel` | random-effects GLS transformation 完成后如何计算 standard error。 |
| `alpha` | `0.05` | 有限且严格位于 0 与 1 之间 | 置信区间显著性水平；`0.05` 对应 95% 区间。 |
| `device` | `"auto"` | `auto`、`cpu`、`cuda`、`torch` | 数值计算运行在哪个 backend/device。 |
| `n_jobs` | `None` | integer 或 `None` | 共享并行参数。 |
| `bandwidth` | `None` | `None` 或非负整数；仅 DK 使用 | Driscoll-Kraay smoothing bandwidth。 |
| `kernel` | `"bartlett"` | Bartlett/Newey-West、Parzen/Gallant、QS/Quadratic-Spectral/Andrews aliases | Driscoll-Kraay kernel。 |
| `group_debias` | `False` | boolean；仅 clustered covariance 使用 | 是否应用 small-number-of-clusters correction。 |

```python
model.fit(X, y, entity_ids=entity_ids, time_ids=None, cluster=None)
```

`entity_ids` 必需，因为 variance-component estimation 与 quasi-demeaning 都按 entity 进行。Driscoll-Kraay 还需要 `time_ids`；clustered covariance 需要 `cluster`。

## CPU and GPU Example

```python
from statgpu.panel import RandomEffects

cpu = RandomEffects(device="cpu").fit(X, y, entity_ids=entity_ids)
cuda = RandomEffects(device="cuda").fit(X, y, entity_ids=entity_ids)
torch = RandomEffects(device="torch").fit(X, y, entity_ids=entity_ids)
```

若显式指定的 GPU backend 不可用，`.fit()` 会直接报错，而不是切换到 CPU。

## Formula Example

假设 `df` 包含 `y`、`x1`、`x2` 与 `entity` 列。

```python
from statgpu.panel import RandomEffects

with_intercept = RandomEffects().fit(
    formula="y ~ x1 + x2 | entity",
    data=df,
)

without_intercept = RandomEffects().fit(
    formula="y ~ 0 + x1 + x2 | entity",
    data=df,
)
```

## Outputs

常用结果包括 `coef_`、`bse_`、`tvalues_`、`pvalues_`、`conf_int_`、`theta_`、`variance_components_`、`fit_statistics_`、`nobs` 与 `df_resid`。`variance_components_` 保存 $\widehat\sigma_e^2$ 与 $\widehat\sigma_a^2$；`theta_` 是拟合中各 entity quasi-demeaning factor 按 entity 数量加权后的平均值。

## Numerical and Strict Behavior

改变 `cov_type` 不会重新拟合 random-effects model：variance components 与 coefficient 保持不变，只改变报告的不确定性。

如果 transformed design 精确 rank deficient，fitted values 仍可能得到，但 coefficient vector 不唯一。statgpu 会对该次拟合整体关闭 coefficient-level standard error、检验、p-value 与 confidence interval，而不是从任意一种 coefficient representation 中继续做推断。

Classical Hausman comparison 只在 [面板 diagnostics](diagnostics.md) 说明的条件下可用。不合法的 covariance 输入或不可用的显式 GPU backend 会直接报错。

## FAQ

**`cov_type` 会改变 Swamy-Arora coefficient estimate 吗？**  不会；它只改变 GLS 拟合后的 standard error 与相关 inference。

**为什么 $\widehat\sigma_a^2$ 可能等于 0？**  finite sample 下 raw Swamy-Arora estimate 可能为负；由于 variance 不能为负，statgpu 会将该估计截断为 0。

## External Validation

我们**不宣称** RandomEffects coefficient 与其他 package 完全一致，因为 statgpu 使用自身的 Swamy-Arora variance-component construction。验证时先取 statgpu 得到的 quasi-demeaned $(X^*,y^*)$：robust 与 Driscoll-Kraay covariance 和 `linearmodels==7.0` 比较，HC2/HC3 covariance 和 `statsmodels==0.14.6` 比较。covariance comparison 使用 `rtol=5e-9, atol=5e-11`；见 [validation matrix](covariance.md#validation-matrix)。

GPU 一致性单独验证：CuPy 与 Torch 输出分别和 NumPy 比较，默认容差为 `rtol=5e-6, atol=5e-7`；实际差异保存在 `results/pr126_p100_fresh/panel_stage_c_correctness_p100.json`。

## 参考（References）

- Swamy, P. A. V. B., & Arora, S. S. (1972). The exact finite sample properties of the estimators of coefficients in the error components regression models. *Econometrica*, 40(2), 261-275. [https://doi.org/10.2307/1909405](https://doi.org/10.2307/1909405)
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). The MIT Press.
