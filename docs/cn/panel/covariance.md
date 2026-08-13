# 面板 Covariance Estimators

> 语言：中文  
> 最后更新：2026-08-13  
> 切换：[English](../../en/panel/covariance.md)

## 统一 fit-space 记号

令 $Z$ 为估计器实际使用的 fit-space design，$e$ 为 residual，并定义

$$
B=(Z^\top Z)^+,
\qquad
\psi_i=Bz_i e_i.
$$

不同模型对应：`PooledOLS` 使用 level design；`PanelOLS` 使用 within-transformed design；`RandomEffects` 使用 quasi-demeaned $X^*$；`BetweenOLS` 使用 entity-mean design；`FirstDifferenceOLS` 使用 first-difference design。`FamaMacBeth` 使用独立的 coefficient-series covariance。

精确 rank deficient 时，OLS-style panel model 的 fitted values 与 fit-space quantities 仍有效，但原始 coefficient coordinate 不唯一，因此 BSE/test/p-value/CI 不可用。

## Nonrobust 与 HC

$$
\widehat V_{\mathrm{nonrobust}}
=\widehat\sigma^2B,
\qquad
\widehat\sigma^2=\frac{e^\top e}{df_{\mathrm{resid}}}.
$$

$$
\widehat V_{\mathrm{HC0}}=\sum_i\psi_i\psi_i^\top,
\qquad
\widehat V_{\mathrm{HC1}}=\frac{n}{df_{\mathrm{resid}}}\widehat V_{\mathrm{HC0}}.
$$

若 $h_i=z_i^\top Bz_i$，则

$$
\widehat V_{\mathrm{HC2}}
=\sum_i\frac{\psi_i\psi_i^\top}{1-h_i},
\qquad
\widehat V_{\mathrm{HC3}}
=\sum_i\frac{\psi_i\psi_i^\top}{(1-h_i)^2}.
$$

leverage 在数值上等于 1 时 HC2/HC3 无定义并直接报错。nonrobust coefficient inference 使用维护的 Student-t reference；HC/cluster/DK 使用维护的 asymptotic-normal reference。

## Clustered covariance

对 cluster $g$，令 $s_g=\sum_{i\in g}\psi_i$，则

$$
\widehat V_G=\sum_gs_gs_g^\top.
$$

`group_debias=True` 时，每个组成部分乘以

$$
\frac{G}{G-1}\frac{n-1}{n}.
$$

双向 clustering 使用精确 paired-label inclusion-exclusion：

$$
\widehat V_{1,2}=\widehat V_1+\widehat V_2-\widehat V_{12}.
$$

## Driscoll-Kraay

先按有序观测时间聚合

$$
g_t=\sum_{i:t_i=t}\psi_i,
$$

再计算

$$
\widehat V_{\mathrm{DK}}
=
\frac{n}{n-\mathrm{extra\_df}-r_Z}
\left[
\sum_tg_tg_t^\top+
\sum_{\ell=1}^{T-1}w_\ell
\sum_{t=\ell+1}^{T}
\left(g_tg_{t-\ell}^\top+g_{t-\ell}g_t^\top\right)
\right].
$$

满列秩时 $r_Z$ 为 $Z$ 的列数；rank-deficient extension 中使用 $\operatorname{rank}(Z)$。`PanelOLS` 的 `extra_df` 为 fixed-effect nuisance rank；`PooledOLS` 与 `RandomEffects` 为 0。

`bandwidth=None` 使用 $\lfloor4(T/100)^{2/9}\rfloor$。Bartlett/Parzen 在 bandwidth 处截断；Quadratic Spectral 在 bandwidth 为正时对全部观测 lag 赋权。ordered pandas categorical 保留声明的时间顺序。

## Legacy pooled HAC

`PooledOLS(cov_type="hac")` 与 Driscoll-Kraay 分开定义：它直接在按 `time_index` 稳定排序后的 influence score 上计算 Bartlett/Newey-West HAC。

## References

White (1980)；Newey and West (1987)；Andrews (1991)；Driscoll and Kraay (1998)。
