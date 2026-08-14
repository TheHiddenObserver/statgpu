# 面板 Covariance Estimators

> 语言：中文  
> 最后更新：2026-08-14  
> 切换：[English](../../en/panel/covariance.md)

## Overview and Path

共享 residual-based panel covariance 实现在 `statgpu/panel/_covariance.py`。令 $Z$ 为 estimator 实际 fit-space design，$e$ 为 residual，并定义

$$
B=(Z^\top Z)^+,
\qquad
\psi_i=Bz_i e_i.
$$

| 模型 | $Z$ |
|---|---|
| `PooledOLS` | level design |
| `PanelOLS` | within-transformed design |
| `RandomEffects` | quasi-demeaned $X^*$ |
| `BetweenOLS` | entity-mean design |
| `FirstDifferenceOLS` | first-difference design |

`FamaMacBeth` 使用独立的 coefficient-series covariance。精确 rank deficient 时，OLS-style panel model 的 fitted values 与 fit-space quantities 仍有效，但原始 coefficient coordinate 不唯一，因此 BSE/test/p-value/CI 不可用。

## Nonrobust and HC Covariance

$$
\widehat V_{\mathrm{nonrobust}}=\widehat\sigma^2B,
\qquad
\widehat\sigma^2=\frac{e^\top e}{df_{\mathrm{resid}}}.
$$

$$
\widehat V_{\mathrm{HC0}}=\sum_i\psi_i\psi_i^\top,
\qquad
\widehat V_{\mathrm{HC1}}=\frac{n}{df_{\mathrm{resid}}}\widehat V_{\mathrm{HC0}}.
$$

若 $h_i=z_i^\top Bz_i$，

$$
\widehat V_{\mathrm{HC2}}=\sum_i\frac{\psi_i\psi_i^\top}{1-h_i},
\qquad
\widehat V_{\mathrm{HC3}}=\sum_i\frac{\psi_i\psi_i^\top}{(1-h_i)^2}.
$$

leverage 在数值上等于 1 时 HC2/HC3 无定义并报错。nonrobust coefficient inference 使用 Student-t reference；HC/cluster/DK 使用 asymptotic-normal reference。

## Clustered Covariance

对 cluster $g$，令 $s_g=\sum_{i\in g}\psi_i$，则

$$
\widehat V_G=\sum_gs_gs_g^\top.
$$

`group_debias=True` 时每个 cluster component 乘以

$$
\frac{G}{G-1}\frac{n-1}{n}.
$$

双向 clustering 使用 exact paired-label inclusion-exclusion：

$$
\widehat V_{1,2}=\widehat V_1+\widehat V_2-\widehat V_{12}.
$$

## Driscoll-Kraay

先按有序观测时间聚合 $g_t=\sum_{i:t_i=t}\psi_i$。对 kernel weights $w_\ell$，

$$
\widehat V_{\mathrm{DK}}=
\frac{n}{n-\mathrm{extra\_df}-r_Z}
\left[
\sum_tg_tg_t^\top+
\sum_{\ell=1}^{T-1}w_\ell
\sum_{t=\ell+1}^{T}(g_tg_{t-\ell}^\top+g_{t-\ell}g_t^\top)
\right].
$$

满列秩时 $r_Z$ 为 $Z$ 的列数；rank-deficient extension 使用 $\operatorname{rank}(Z)$。`PanelOLS` 将 fixed-effect nuisance rank 传入 `extra_df`；`PooledOLS` 与 `RandomEffects` 使用 0。

`bandwidth=None` 使用 $\lfloor4(T/100)^{2/9}\rfloor$。Bartlett/Parzen 在 bandwidth 处截断；Quadratic Spectral 在 bandwidth 为正时对全部观测 lag 赋权。numeric/datetime label 使用自然顺序；ordered pandas categorical 保留声明 chronology。

## Public API and Aliases

public helpers 包括 `ols_covariance`、`clustered_covariance`、`two_way_clustered_covariance`、`hac_covariance` 与 `driscoll_kraay_covariance`。`hc1` alias `robust`；`dk` 与 `kernel` alias `driscoll-kraay`。`PooledOLS(cov_type="hac")` 继续表示独立的 row-order Bartlett/Newey-West path。

## External Validation

pinned Python references 为 `linearmodels==7.0` 与 `statsmodels==0.14.6`；pinned R references 为 `plm==2.6-7` 与 `sandwich==3.1-3`。维护检查位于 `dev/tests/test_panel_stage_c_external.py`、`dev/tests/test_panel_stage_c_external_defaults.py`、`dev/tests/test_panel_stage_c_linearmodels_estimators.py` 与 `dev/tests/test_panel_stage_c_r_external.py`。CuPy/Torch 物理验证见 `results/pr126_p100_fresh/validation_summary.txt`。

## References

White (1980)；Newey and West (1987)；Andrews (1991)；Driscoll and Kraay (1998)；Cameron, Gelbach, and Miller (2011)。
