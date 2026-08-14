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

| 参数 | 含义 |
|---|---|
| `entity_effects`, `time_effects` | 选择 fixed-effect 维度。 |
| `cov_type` | `nonrobust`、HC0/1/2/3、`clustered` 或 DK aliases。 |
| `bandwidth`, `kernel` | Driscoll-Kraay 参数。 |
| `group_debias` | cluster small-group correction。 |
| `demean_max_iter`, `demean_tol` | 双向 alternating projection 控制。 |
| `alpha` | 置信区间显著性水平。 |
| `device` | `auto`、`cpu`、`cuda` 或 `torch`。 |
| `n_jobs` | 共享并行参数。 |

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

在定义重合处，维护测试与 `linearmodels==7.0`、`statsmodels==0.14.6` 以及 R `plm==2.6-7` / `sandwich==3.1-3` 对齐；见 `dev/tests/test_panel_stage_c_linearmodels_estimators.py` 与 `dev/tests/test_panel_stage_c_r_external.py`。CuPy/Torch 物理验证见 `results/pr126_p100_fresh/validation_summary.txt`。

## References

Wooldridge (2010), *Econometric Analysis of Cross Section and Panel Data*；covariance 与 diagnostic 参考文献见 [面板 covariance](covariance.md) 和 [面板 diagnostics](diagnostics.md)。
