# PanelOLS

> 语言：中文  
> 最后更新：2026-08-14  
> 切换：[English](../../en/panel/panel-ols.md)

## Overview

`PanelOLS` 用于 level、单向 fixed effects 或双向 fixed effects 的线性面板回归。Fixed effects 在实际数值 fit space 中消除，而不是显式构造稠密 dummy matrix。

## Path

实现：`statgpu/panel/_fixed_effects.py`。

## Model and Objective

对 entity 与 time effects，

$$
y_{it}=x_{it}^{\top}\beta+\alpha_i+\gamma_t+\varepsilon_{it}.
$$

令 $F$ 为包含的 fixed-effect design，$M_F=I-F(F^\top F)^+F^\top$。则

$$
\widehat\beta_{\mathrm{FE}}
=\arg\min_\beta\|M_F(y-X\beta)\|_2^2
=(X^\top M_FX)^+X^\top M_Fy.
$$

单向 entity effect 等价于 within demeaning。双向 effects 使用 backend-native alternating projection；若在 `demean_max_iter` 内没有达到 `demean_tol`，则 fail closed。

## Covariance and Inference

fit space 为 $Z=M_FX$，residual 为 $e=M_F(y-X\widehat\beta_{\mathrm{FE}})$。nonrobust、HC、clustered 与 Driscoll-Kraay 的统一公式见 [面板 covariance](covariance.md)。

Driscoll-Kraay 使用的 fixed-effect nuisance rank 为

$$
r_F=\begin{cases}N,&\text{仅 entity},\\T,&\text{仅 time},\\N+T-C,&\text{双向},\end{cases}
$$

其中 $C$ 为观测 entity-time incidence graph 的 connected-component 数。

## Parameters

| 参数 | 默认值 | 可选值 / 约束 | 含义 |
|---|---:|---|---|
| `entity_effects` | `False` | boolean | 是否加入 entity fixed effects。 |
| `time_effects` | `False` | boolean | 是否加入 time fixed effects。 |
| `cov_type` | `"nonrobust"` | `nonrobust`、`robust`/`hc1`、`hc0`、`hc2`、`hc3`、`clustered`、`driscoll-kraay`/`dk`/`kernel` | covariance estimator。 |
| `alpha` | `0.05` | 显著性水平；`0.05` 对应 95% 置信区间。 | 置信区间 level 控制。 |
| `device` | `"auto"` | `auto`、`cpu`、`cuda`、`torch` | 数值 backend/device。 |
| `n_jobs` | `None` | integer 或 `None` | 共享并行参数。 |
| `bandwidth` | `None` | `None` 或非负整数；仅 DK 使用 | Driscoll-Kraay smoothing bandwidth；`None` 使用文档中的自动规则。 |
| `kernel` | `"bartlett"` | Bartlett/Newey-West、Parzen/Gallant、QS/Quadratic-Spectral/Andrews aliases | Driscoll-Kraay kernel。 |
| `group_debias` | `False` | boolean；仅 clustered covariance 使用 | small-group cluster correction。 |
| `demean_max_iter` | `1_000_000` | 正整数 | 双向 alternating projection 的最大迭代次数。 |
| `demean_tol` | `1e-10` | 有限且严格为正 | 双向 alternating projection 的收敛容差。 |

```python
model.fit(X, y, entity_ids=None, time_ids=None, cluster=None)
```

Formula 可使用 additive `EntityEffects` / `TimeEffects` token 或 pipe fixed-effect syntax，但不能混用。无 fixed effects 时普通 formula intercept 会保留，`0 +` / `-1` 表示 no-intercept level regression；有 fixed effects 时公共 intercept 被 effect space 吸收。

## CPU and GPU Example

```python
from statgpu.panel import PanelOLS

cpu = PanelOLS(entity_effects=True, device="cpu").fit(X, y, entity_ids=entity_ids)
cuda = PanelOLS(entity_effects=True, device="cuda").fit(X, y, entity_ids=entity_ids)
torch = PanelOLS(entity_effects=True, device="torch").fit(X, y, entity_ids=entity_ids)
```

`cuda` 需要 CuPy/CUDA，`torch` 需要 Torch CUDA；显式 GPU 请求不会静默 fallback 到 CPU。

## Outputs

常用 public 结果包括 `coef_`、`bse_`、`tvalues_`、`pvalues_`、`conf_int_`、`rsquared_within`、`fit_statistics_`、`nobs` 与 `df_resid`。`summary()` 返回 panel summary。Pooling F 与 Hausman 见 [面板 diagnostics](diagnostics.md)。

## Numerical and Strict Behavior

不存在 silent approximate-inference fallback。精确 rank deficient 时保留 fit-space quantities，但 coefficient coordinate inference 不可用，见 [面板 covariance](covariance.md)。双向 stored-effect prediction 要求 entity/time labels 已识别且属于同一个 fitted incidence component；单边、known-plus-unknown 与 cross-component 组合 fail closed；若两个 label 都未见过，则使用 linear-only fallback。

## FAQ

**为什么双向 demeaning 不直接返回最后一个 iterate？**  收敛容差属于数值契约，未达到时直接 fail closed。

**robust covariance 会把 `fit_statistics_.f_statistic` 改成 robust Wald test 吗？**  不会。该字段仍是 classical fit-space joint-slope statistic，见 [fit statistics](fit-statistics.md)。

## External Validation

外部框架对齐与物理 GPU backend 精度是两套独立 gate。`linearmodels==7.0` 检查单向/双向 FE Driscoll-Kraay integration：coefficient 使用 `rtol=2e-10, atol=2e-11`，covariance/BSE 使用 `rtol=5e-9, atol=5e-11`。`statsmodels==0.14.6` 检查 no-FE level OLS；R `plm==2.6-7` 检查单向 FE coefficient。共享 covariance definition 与 R tolerance 见 [validation matrix](covariance.md#validation-matrix)。

Stage-C 物理 runner 另行使用默认 `rtol=5e-6, atol=5e-7` 比较 CuPy/Torch 与 NumPy；实际 `max_abs_differences` 保存在 `results/pr126_p100_fresh/panel_stage_c_correctness_p100.json`。

## 参考（References）

- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). The MIT Press.

covariance 与 diagnostic 专门参考文献见 [面板 covariance](covariance.md) 和 [面板 diagnostics](diagnostics.md)。
