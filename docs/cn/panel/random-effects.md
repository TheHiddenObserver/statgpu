# RandomEffects

> 语言：中文  
> 最后更新：2026-08-14  
> 切换：[English](../../en/panel/random-effects.md)

## Overview

`RandomEffects` 实现单向 Swamy-Arora error-components estimator，并在其后进行 feasible GLS。不同 covariance 选择只改变 quasi-demeaned fit space 上的 inference，不改变 variance-component construction。

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

其中 $df_W=n-r_W-r_E$。level design 有显式常数列时 $r_E=N$，否则 $r_E=N-1$；同时 $df_B=N-r_B$。对 entity $i$，

$$
\theta_i=1-\sqrt{\frac{\widehat\sigma_e^2}{\widehat\sigma_e^2+T_i\widehat\sigma_a^2}},
$$

$$
y_{it}^*=y_{it}-\theta_i\bar y_i,
\qquad
x_{it}^*=x_{it}-\theta_i\bar x_i.
$$

feasible GLS 的 objective 与 estimator 为

$$
\widehat\beta_{\mathrm{RE}}
=\arg\min_\beta\|y^*-X^*\beta\|_2^2
=(X^{*\top}X^*)^+X^{*\top}y^*.
$$

rank-deficient extension 中，辅助回归的原始列数由 identified numerical rank 替代。

## Covariance and Inference

所有 residual-based covariance 都使用 $Z=X^*$ 与 $e^*=y^*-X^*\widehat\beta_{\mathrm{RE}}$。特别地，

$$
\widehat V_{\mathrm{nonrobust}}
=\widehat\sigma_*^2(X^{*\top}X^*)^+,
\qquad
\widehat\sigma_*^2=\frac{e^{*\top}e^*}{n-\operatorname{rank}(X^*)}.
$$

HC0/1/2/3、clustered 与 Driscoll-Kraay 使用 [面板 covariance](covariance.md) 的统一定义，其中 fit-space design 取 $X^*$。

## Parameters

| 参数 | 含义 |
|---|---|
| `cov_type` | `nonrobust`、HC0/1/2/3、`clustered` 或 DK aliases。 |
| `bandwidth`, `kernel` | Driscoll-Kraay 参数。 |
| `group_debias` | cluster small-group correction。 |
| `alpha` | 置信区间显著性水平。 |
| `device` | `auto`、`cpu`、`cuda` 或 `torch`。 |
| `n_jobs` | 共享并行参数。 |

```python
model.fit(X, y, entity_ids=entity_ids, time_ids=None, cluster=None)
```

`entity_ids` 必需；Driscoll-Kraay 需要 `time_ids`，clustered covariance 需要 `cluster`。Formula 保留正常 R/Patsy intercept；`0 +` 或 `-1` 表示 no-intercept random-effects model。

## CPU and GPU Example

```python
from statgpu.panel import RandomEffects

cpu = RandomEffects(device="cpu").fit(X, y, entity_ids=entity_ids)
cuda = RandomEffects(device="cuda").fit(X, y, entity_ids=entity_ids)
torch = RandomEffects(device="torch").fit(X, y, entity_ids=entity_ids)
```

显式 GPU 请求要求对应 backend 可用，不会静默 fallback 到 CPU。

## Outputs

public 结果包括 `coef_`、`bse_`、`tvalues_`、`pvalues_`、`conf_int_`、`theta_`、`variance_components_`、`fit_statistics_`、`nobs` 与 `df_resid`。`variance_components_` 保存 $\widehat\sigma_e^2$ 与 $\widehat\sigma_a^2$。

## Numerical and Strict Behavior

variance components 与 coefficient estimate 不随 covariance choice 改变。不存在 silent approximate-inference fallback。精确 rank deficient 时使用 documented identified-rank extension，但 coordinate-wise inference 仍要求 coefficient representation 可识别。Classical FE-versus-RE Hausman 只在文档规定的适用条件下执行。

## FAQ

**`cov_type` 会改变 Swamy-Arora coefficient estimate 吗？**  不会；它只改变 quasi-demeaned GLS 拟合后的 inference。

**为什么 $\widehat\sigma_a^2$ 可能等于 0？**  当前 estimator 会把 raw variance-component estimate 截断在 0。

## External Validation

`dev/tests/test_panel_stage_c_linearmodels_estimators.py` 在 statgpu 自身 Swamy-Arora fit space 上，将 HC/cluster/DK covariance 与 pinned `linearmodels==7.0` 和 `statsmodels==0.14.6` 定义比较。R panel covariance 检查独立维护；CuPy/Torch 物理验证见 `results/pr126_p100_fresh/validation_summary.txt`。

## References

Swamy and Arora (1972), error-components feasible GLS；Wooldridge (2010), *Econometric Analysis of Cross Section and Panel Data*。
