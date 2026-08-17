# FamaMacBeth

> 语言：中文  
> 最后更新：2026-08-17  
> 切换：[English](../../en/panel/fama-macbeth.md)

## Overview

`FamaMacBeth` 在每个 time period 分别做一次 cross-sectional regression，再对这些 period-specific coefficient 取平均。它与其他 panel estimator 的一个关键区别是：目标参数是平均 cross-sectional slope，standard error 则来自这些 slope **随时间的波动**。

## Path

实现：`statgpu/panel/_fama_macbeth.py`。

## Statistical Model and Target

一个自然的 period-specific model 是

$$
y_{it}=\alpha_t+x_{it}^{\top}\beta_t+\varepsilon_{it},
$$

其中 intercept 与 slope 都允许随时期变化。为了赋予每个 cross-sectional regression 通常的统计解释，一个常见的充分条件是

$$
E(\varepsilon_{it}\mid x_{it},t)=0,
$$

或者更一般地，使用相应的 cross-sectional orthogonality condition，将 $\beta_t$ 定义为 period-$t$ 的 linear projection coefficient。

对 estimator 最终保留的 $T$ 个时期，直接目标是 retained-period 的等权平均

$$
\beta_{\mathrm{FM}}=\frac{1}{T}\sum_{t=1}^{T}\beta_t.
$$

定义这个 target 并不要求先给 sequence $\{\beta_t\}$ 指定概率模型。如果进一步把这些 retained periods 看成从某个 time superpopulation 中抽取，那么在相应 sampling assumptions 下，这个平均量还可以获得类似 $E_t(\beta_t)$ 的 population interpretation。constant-slope model $\beta_t\equiv\beta$ 是其中的特殊情况。

上述 coefficient interpretation 要求每个最终保留时期的 cross-sectional design 都能唯一识别相应 coefficient vector。完成 observation-count filtering 后，statgpu 会按照共享的 panel SVD rank policy 检查加入 intercept 后的 $X_t$。如果某个 retained period rank deficient，`.fit()` 会在 coefficient averaging 与 inference 之前直接抛出 `ValueError`，而不会基于非唯一的 coefficient representation 继续做 coordinate-level inference。

## Estimator

对每个保留时期，令 $X_t$ 表示已加入 intercept 的 period design。在要求 full column rank 的契约下，

$$
\widehat\beta_t
=\arg\min_\beta\|y_t-X_t\beta\|_2^2
=(X_t^\top X_t)^{-1}X_t^\top y_t,
\qquad
\widehat\beta_{\mathrm{FM}}=T^{-1}\sum_{t=1}^T\widehat\beta_t.
$$

时期只有在满足 `min_obs_per_period` 且通过实现中的最小样本量规则 $n_t\ge k$ 时才会保留，其中 $k$ 是含 intercept 的 design width；随后每个 retained period 仍必须满足 full-rank contract。NumPy reference path 保留原有 serial rank-revealing SVD policy。GPU 路径按真实 $n_t$ 做 exact-size grouping，不使用 zero padding，并先批量构造 $G_t=X_t^\top X_t$ 与 $X_t^\top y_t$。只有当 backend-native Gram spectrum 满足 $\lambda_{\min}(G_t)/\lambda_{\max}(G_t)>10^{-4}$ 时，该时期才允许使用 batched Gram solve candidate，因此 fast path 只覆盖明显 well-conditioned 的设计。这个 certificate 只是 performance gate，并没有替换 rank 定义：所有 uncertified period 都回退到原有 $\max(n_t,k)\epsilon s_{\max,t}$ SVD cutoff。Torch 对 unsafe subset 可以使用其有文档保证的 stacked-SVD，CuPy 则继续使用受支持的二维 SVD fallback。这样 near-rank-boundary 与 rank-deficient 行为仍由原 SVD policy 决定，而 clearly well-conditioned 的 GPU periods 可以避免更昂贵的 rank-revealing SVD。

## Covariance and Inference

定义每个 period coefficient 与最终平均 coefficient 的偏差

$$
\nu_t=\widehat\beta_t-\widehat\beta_{\mathrm{FM}}.
$$

当 `cov_type="nonrobust"` 时，

$$
\widehat V_{\mathrm{nonrobust}}
=\frac{1}{T(T-1)}\sum_{t=1}^T \nu_t\nu_t^\top.
$$

这相当于把保留时期的 coefficient 看作相互独立的 coefficient series。若使用 `cov_type="newey-west"`，则允许该 coefficient series 存在 serial dependence。定义

$$
\widehat\Gamma_\ell
=\frac{1}{T}\sum_{t=\ell+1}^{T}\nu_t\nu_{t-\ell}^\top,
\qquad \ell=0,\ldots,L,
$$

以及 Bartlett weight

$$
w_\ell=1-\frac{\ell}{L+1}.
$$

coefficient series 的 long-run covariance 与 Fama-MacBeth 平均 coefficient 的 covariance 分别为

$$
\widehat\Omega_{\mathrm{NW}}
=\widehat\Gamma_0+
\sum_{\ell=1}^{L}w_\ell
\left(\widehat\Gamma_\ell+\widehat\Gamma_\ell^\top\right),
\qquad
\widehat V_{\mathrm{NW}}(\widehat\beta_{\mathrm{FM}})
=\frac{1}{T}\widehat\Omega_{\mathrm{NW}}.
$$

若 `bandwidth=None`，statgpu 先取

$$
L=\left\lfloor4(T/100)^{2/9}\right\rfloor,
$$

再截断到 $0\le L\le T-1$。这里的 Newey-West 是作用在 period coefficient sequence 上，因此与 [面板 covariance](covariance.md) 中基于 observation residual 的 HAC/Driscoll-Kraay 不同。

retained coefficient series 的顺序由 `time_ids` 决定。numeric 与 datetime labels 使用自然顺序；ordered pandas categorical 会保留用户明确声明的 category order。普通 string labels 按字符串字典序排序；其他非 categorical object labels 按其可比较值排序，不能相互比较时会直接报错。因此 `t1, t2, t10` 这类字符串语义标签若不应按字典序解释，应改用 ordered categorical 或 numeric/datetime key，再形成 Newey-West lag covariance。

成功拟合还会发布 inference-capable statgpu estimator 共用的 `ParameterInferenceResult`。公开的 `coef_`、`bse_`、`tvalues_`、`pvalues_` 与 `conf_int_` 继续保持 backend-native。distribution inference 跟随实际 fit backend：NumPy 使用 NumPy inference backend，CuPy 使用 CuPy inference backend，Torch 则在实际 tensor device 上使用 Torch inference backend。`_inference_result` 以及 `_params`、`_bse`、`_tvalues`/`_zvalues`、`_pvalues`、`_conf_int` 保存的 NumPy snapshot 仅用于统一 inference/reporting contract，不参与 p-value 或 confidence interval 的数值计算。对于 GPU fit，reporting fields 会先在 active backend 上打包，再在 numerical inference 完成后一次性形成小型 NumPy snapshot。`newey-west` 标记为 `z`/normal inference，`nonrobust` 标记为自由度 $T-1$ 的 Student-t inference。

## Parameters

| 参数 | 默认值 | 可选值 / 约束 | 含义 |
|---|---:|---|---|
| `cov_type` | `"newey-west"` | `nonrobust` 或 `newey-west` | 是否忽略 period coefficient 的跨期相关性，或用 Newey-West 进行修正。 |
| `bandwidth` | `None` | `None` 或非负整数；最终不超过 $T-1$ | Bartlett Newey-West bandwidth $L$。 |
| `alpha` | `0.05` | 有限且严格位于 0 与 1 之间 | 置信区间显著性水平；`0.05` 对应 95% 区间。 |
| `min_obs_per_period` | `1` | 正整数 | 初步的最小 period size；最终保留时期还必须满足 $n_t\ge k$，其中 $k$ 是含 intercept 的 design width，并且必须 full column rank。 |
| `device` | `"auto"` | `auto`、`cpu`、`cuda`、`torch` | 数值计算运行在哪个 backend/device。 |
| `n_jobs` | `None` | integer 或 `None` | 共享并行参数。 |

```python
model.fit(X, y, time_ids=time_ids, entity_ids=None)
```

`time_ids` 必需，用于定义每个 cross-sectional regression。`entity_ids` 可选，只用于 standardized within/between $R^2$。

## CPU and GPU Example

```python
from statgpu.panel import FamaMacBeth

cpu = FamaMacBeth(device="cpu").fit(X, y, time_ids=time_ids)
cuda = FamaMacBeth(device="cuda").fit(X, y, time_ids=time_ids)
torch = FamaMacBeth(device="torch").fit(X, y, time_ids=time_ids)
```

若显式指定的 GPU backend 不可用，`.fit()` 会直接报错，而不是切换到 CPU。

## Formula Example

假设 `df` 包含 `y`、`x1`、`x2` 与 `time` 列。

```python
from statgpu.panel import FamaMacBeth

model = FamaMacBeth().fit(
    formula="y ~ x1 + x2",
    data=df,
    time_ids=df["time"],
)
```

`FamaMacBeth` 与 array API 一致，始终包含 period-specific intercept。因此 `y ~ 0 + x1 + x2` 或 `y ~ x1 + x2 - 1` 这类显式 no-intercept formula 会得到清晰的 `ValueError`，而不会被静默改写成带 intercept 的模型。

## Outputs

常用结果包括 `coef_`、`bse_`、`tvalues_`、`pvalues_`、`conf_int_`、`betas_`、`cov_params_`、`fit_statistics_`、`nobs`、`n_periods` 与 `df_resid`。`betas_` 保存各保留时期的 coefficient，`coef_` 则是这些 coefficient 的平均；`_inference_result` 提供统一的 inference container，供 statgpu 的通用 reporting/downstream tooling 使用。

## Numerical and Strict Behavior

过滤后至少需要两个有效 period，否则 `.fit()` 会报错，因为少于两个 period 无法估计 coefficient series 的波动。每个 retained period 还必须在共享 panel SVD cutoff 下 full column rank；若某个 retained period rank deficient，会在 inference 之前 fail closed。GPU 上的 Gram certificate 被刻意放在远离 numerical rank boundary 的区域，它只决定 coefficient solve 是否可以使用 fast path；uncertified periods 继续采用原有 SVD cutoff 与 fail-closed 语义。所有路径都保持 `betas_` chronology；若多个时期 rank deficient，公开错误仍定位 chronology 上最早的 deficient retained period。

维护中的 physical scaling matrix 保留三个 resident-array workload：micro（64×128×4；8,192 rows）、medium（128×1,024×8；131,072 rows）以及 large（128×4,096×16；524,288 rows）。在 numerical source `8c60db00f5ea986aed96b1f1dce3f5c3b4f0bcd4` 上的 fresh Tesla P100 evidence 中，CuPy/Torch 的 GPU-over-NumPy median-time ratio 分别为：micro **0.549/0.343**、medium **0.204/0.168**、large **0.114/0.109**，对应约 1.82×/2.92×、4.91×/5.97× 和 8.75×/9.16× speedup。所有 GPU backend × scale case 都是一个 `gram-certified` exact-size batch、一次 control synchronization、零 SVD fallback。这些结果说明维护中的 P100 resident-array protocol 已经在三个 scale 上全部 crossover；它们并不是对所有硬件与数据搬运场景的普遍性能承诺。

`dev/benchmarks/benchmark_fama_macbeth_scaling_gpu.py` 记录同步后的 NumPy/CuPy/Torch median time、rows/sec、GPU/NumPy ratio、speedup、solver provenance、numerical parity、backend version 和 thread-environment provenance。GPU input array 在 warmup/timing 之前已经转移到 device，因此 benchmark 测量的是 resident-array `fit()` 性能，并明确**不包含** host-to-device input-transfer time。accepted P100 scaling artifact 的数值差异仍然很小：coefficient/beta/prediction 接近 machine precision，最大的 statistic difference 低于 $4\times10^{-11}$。

p-value 与 critical value 通过所选 inference backend 计算，不再把 statistic vector 转到 NumPy/SciPy。normal inference 使用 backend 的 two-sided normal routine；一般 Student-t inference 使用 backend Student-t routine。小自由度边界同样保持 backend-native：df=1 使用精确 Cauchy identity，df=2 使用精确的 elementary two-sided tail 与 quantile 公式，从而避免缺少 native `betainc` 的 Torch 版本降低维护中的高精度契约。只有 inference 已经完成之后，为统一 `ParameterInferenceResult` reporting contract 才会保存 NumPy snapshot。

`cov_type="newey-west"` 使用 asymptotic-normal inference；`cov_type="nonrobust"` 使用自由度 $T-1$ 的 Student-t reference。如果这些条件不满足，statgpu 不会在后台切换成另一套 inference 方法。

每次新的 fit attempt 都会先失效旧的 fitted/inference state。如果新一次拟合失败，对象会保持 unfitted，而不会继续暴露上一份数据的 stale coefficient 或 standard error。

显式指定 `device="cuda"` 或 `device="torch"` 时也要求对应 backend 可用，否则直接报错而不是切换到 CPU。

## FAQ

**为什么它的 covariance 不列在 HC/cluster/Driscoll-Kraay 中？**  这些方法基于 observation-level regression residual。Fama-MacBeth inference 则基于各时期 coefficient $\widehat\beta_t$ 形成的 time series。

**样本过少的 period 如何处理？**  在形成 coefficient average 之前排除。若最终不足两个有效 period，fit 会报错。

**如果某个 retained period rank deficient 会怎样？**  fit 会抛出 `ValueError`。statgpu 不会把非唯一的 period coefficient vector 纳入平均后再给出 coordinate-level standard error。

## External Validation

`dev/tests/test_fama_macbeth_linearmodels_external.py` 提供维护中的 pinned `linearmodels==7.0` definition-alignment gate。fixture 在两边使用相同的显式 period intercept、full-rank balanced panel、period ordering 与 coefficient set；两种 covariance mode 都会比较 period-by-period coefficient（`betas_` 对 `all_params`）以及最终平均 coefficient。

对 `cov_type="nonrobust"`，statgpu covariance 对齐 linearmodels 的 `cov_type="unadjusted", debiased=True`：维护中的测试比较 covariance、standard error 和 coefficient t-statistic。这里**不会**强行比较 p-value/CI，因为虽然 covariance definition 对齐，两套 API 对 post-estimation inference 使用的 reference df 不同：statgpu 按 retained-period 定义使用 $T-1$，而 linearmodels 在 `debiased=True` 时使用 stacked-panel residual degrees of freedom。

对 `cov_type="newey-west"`，external gate 使用 linearmodels `cov_type="kernel", kernel="bartlett", bandwidth=L, debiased=False`，并固定完全相同的 $L$。此时 coefficient-series kernel covariance 与 normal-reference inference 都对齐，因此会继续比较 covariance、standard error、test statistic、p-value 和 confidence interval。专用 `Panel Stage C external covariance` workflow 会安装固定 reference 版本，并在相关 PR/source change 上运行该测试。

内部 maintained regression 还覆盖 formula intercept、array/formula 两条路径的 ordered-categorical 与 numeric chronology、formula missing-row alignment、retained-period rank rejection、failed-refit invalidation、SVD rank-boundary policy、conservative Gram-certificate acceptance/rejection、backend-native distribution routing、df=1/df=2 的精确小自由度 inference boundary、balanced 与 shuffled-unbalanced exact-size GPU grouping、chronological rank-error reporting、SVD fallback ownership、direct-fit finite-validation ownership，以及 packed reporting snapshot。

标准 full-rank、numeric-time 的 `fama_macbeth_newey_west` case 仍由 `dev/benchmarks/validate_panel_stage_a_gpu.py` 做 GPU consistency 验证。历史 focused gate `dev/benchmarks/validate_fama_macbeth_review_fix_gpu.py` 继续作为 chronology/formula/rank/inference 的详细 correctness oracle。final optimized-source physical acceptance 使用 `dev/benchmarks/validate_fama_macbeth_optimized_gpu.py`，crossover evidence 由 `dev/benchmarks/benchmark_fama_macbeth_scaling_gpu.py` 记录。PR126 accepted P100 evidence 统一锚定 numerical source `8c60db00...`，并由同一 source 的 Stage-C matrix 和 HAC-chronology runner 补齐 broad physical acceptance。Fama-MacBeth 不属于 Stage-C residual-covariance matrix，因为它的 covariance 来自 coefficient series，而不是 observation residual。

## 参考（References）

- Fama, E. F., & MacBeth, J. D. (1973). Risk, return, and equilibrium: Empirical tests. *Journal of Political Economy*, 81(3), 607-636. [https://doi.org/10.1086/260061](https://doi.org/10.1086/260061)
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703-708. [https://doi.org/10.2307/1913610](https://doi.org/10.2307/1913610)