# 面板 Covariance Estimators

> 语言：中文  
> 最后更新：2026-08-18<br>
> 切换：[English](../../en/panel/covariance.md)

## Overview and Path

不同 panel estimator 实际用于回归的数据并不相同：fixed effects 会先去均值，random effects 会做 quasi-demeaning，first difference 会先做差分。因此 standard error 也必须基于**真正用于 coefficient estimation 的那组 transformed data**计算，而不是统一拿原始 $X$ 和 $y$ 计算。

为了用一套符号写出共享公式，下面用 $Z$ 表示某个模型实际用于最终回归的 design，用 $e$ 表示对应 residual，并定义

$$
B=(Z^\top Z)^+,
\qquad
\psi_i=Bz_i e_i.
$$

对不同模型，$Z$ 分别表示：

| 模型 | 用于 covariance 的回归 |
|---|---|
| `PooledOLS` | 原始 level regression |
| `PanelOLS` | 去除所选 fixed effects 后的 regression |
| `RandomEffects` | quasi-demeaned design $X^*$ |
| `BetweenOLS` | 每个 entity 一个均值观测的 regression |
| `FirstDifferenceOLS` | first-differenced regression |

`FamaMacBeth` 不使用这套 residual-based covariance；它根据各时期 coefficient 的 time series 计算 uncertainty，见 [FamaMacBeth](fama-macbeth.md)。它要求每个 retained period design 都 full column rank，不满足时会 fail closed。

对于上表中的 residual-OLS families，如果 fit-space design 精确 rank deficient，fitted values 仍可能是唯一的，但 coefficient vector 不唯一。此时 statgpu 会保留可解释的 fitted result，同时对该次拟合整体关闭 coefficient-level BSE、检验、p-value 与 confidence interval，而不是从任意一种 coefficient representation 中继续做推断。

实现：`statgpu/panel/_covariance.py`。

## Nonrobust and HC Covariance

`nonrobust` 是通常的同方差 OLS covariance。HC0-HC3 是异方差稳健版本；HC2/HC3 会进一步调整 high-leverage observation 的影响。

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

令 leverage 为 $h_i=z_i^\top Bz_i$，则

$$
\widehat V_{\mathrm{HC2}}=\sum_i\frac{\psi_i\psi_i^\top}{1-h_i},
\qquad
\widehat V_{\mathrm{HC3}}=\sum_i\frac{\psi_i\psi_i^\top}{(1-h_i)^2}.
$$

HC2/HC3 要求 $1-h_i$ 在数值上为正。对于 full-rank estimator fit 或直接调用 covariance primitive，如果某个 observation 的 leverage 在数值上等于 1，statgpu 会直接报错，而不是返回无穷大或不稳定的 variance。若 estimator 的 fit-space 本身已经 rank deficient，则 coefficient-level inference 无论选择哪种 covariance 都不可用；此时 statgpu 保留 fitted values，并且不会再强行构造可能在 unit leverage 下无定义的 HC2/HC3 coordinate covariance。

nonrobust coefficient inference 使用 Student-t reference；HC、clustered 与 Driscoll-Kraay 使用 panel API 中的 asymptotic-normal reference。正的 covariance diagonal 不再使用绝对 variance floor，因此整体缩放 response 会按同一比例缩放 coefficient 与 standard error，而不会改变有限 t/z statistic。若 diagonal variance 精确为 0，则零 coefficient 的 statistic 为 0，非零 coefficient 的 statistic 为带符号无穷；p-value 与 confidence interval 直接由这一显式结果得到，而不是通过伪造 tiny denominator。

## Clustered Covariance

clustered covariance 允许同一用户指定 cluster 内的 observations 具有相关误差。对 cluster $g$，令 $s_g=\sum_{i\in g}\psi_i$，则

$$
\widehat V_G=\sum_gs_gs_g^\top.
$$

clustered inference 要求每个用户提供的 clustering dimension 至少包含两个不同 group。只有一个 cluster 时，grouped score 会退化为全样本 estimating-equation score，cluster-robust variance 无法被估计；因此即使 `group_debias=False`，statgpu 也会直接报错。

`group_debias=True` 时会应用 small-number-of-clusters correction：

$$
\frac{G}{G-1}\frac{n-1}{n}.
$$

对极端但仍有限的 score，grouped reduction 只在同号 partial sum 存在溢出风险的 group/coordinate 上使用 group-size working scale，并先分别累计正项与负项，再做最终 cancellation；不会因为某个危险 group 而对其他安全 group 做全局 magnitude normalization。和一般 float64 线性代数一样，这并不承诺在上游已经发生灾难性病态 cancellation 后恢复任意微小 remainder。

双向 clustering 将两个 one-way cluster covariance 相加，再减去 paired cluster labels 对应的 covariance：

$$
\widehat V_{1,2}=\widehat V_1+\widehat V_2-\widehat V_{12}.
$$

## Driscoll-Kraay

Driscoll-Kraay 是按 time index 构造的 panel covariance。statgpu 先把同一 observed period 内各 observation 对 covariance 的贡献聚合起来：

$$
g_t=\sum_{i:t_i=t}\psi_i,
$$

再通过 kernel weights 对不同 time lags 加权。对权重 $w_\ell$，

$$
\widehat V_{\mathrm{DK}}=
\frac{n}{n-\mathrm{extra\_df}-r_Z}
\left[
\sum_tg_tg_t^\top+
\sum_{\ell=1}^{T-1}w_\ell
\sum_{t=\ell+1}^{T}(g_tg_{t-\ell}^\top+g_{t-\ell}g_t^\top)
\right].
$$

这里 $r_Z$ 表示回归中实际可识别的 regression directions：满列秩时等于 $Z$ 的列数，rank deficient 时等于 $\operatorname{rank}(Z)$。`PanelOLS` 还需要通过 `extra_df` 计入被吸收的 fixed effects；`PooledOLS` 与 `RandomEffects` 的该项为 0。

对 symmetric covariance combination，statgpu 使用 range-aware arithmetic：最终 symmetrization 会避免有限同号平均值在相加阶段先溢出；two-way inclusion-exclusion 会在可能时先减去同号 intersection component。HAC/Driscoll-Kraay 会先形成 symmetric lag average，而不是先构造可能溢出的未加权 symmetric sum；随后对整个 lag sequence 只在某个 entry 的 transient partial sum 存在溢出风险时使用 per-entry reduction-length working scale，安全与 subnormal entry 保持原尺度。只要最终 float64 结果可表示，这些重排与上面的统计定义代数等价；与其他数值路径一样，并不宣称可以用更高精度恢复任意病态 cancellation。

`bandwidth=None` 时使用 $\lfloor4(T/100)^{2/9}\rfloor$。Bartlett 与 Parzen kernel 在 bandwidth 之外权重为 0；Quadratic Spectral 将 bandwidth 作为 smoothing scale，在 bandwidth 为正时会对全部 observed lags 赋权。

time ordering 会影响 Driscoll-Kraay。numeric 和 datetime labels 使用自然顺序；ordered pandas categorical 使用用户声明的 category 顺序。普通 string labels 按字符串字典序排序；其他非 categorical object labels 按其可比较值的排序顺序处理，若 labels 不能相互比较则直接报错。如果字符串字典序不是实际 chronology（例如 `t1, t2, t10`），应改用 numeric/datetime key 或 ordered categorical。

## Public API and Aliases

`statgpu.panel` 公开导出的 covariance helpers 是 `clustered_covariance`、`two_way_clustered_covariance`、`hac_covariance` 与 `driscoll_kraay_covariance`。`ols_covariance` 是 panel estimator 内部复用的 shared dispatcher，不属于公开的 `statgpu.panel` export surface。

在 estimator 的 `cov_type` 中，`hc1` 是 `robust` 的 alias；`dk` 与 `kernel` 是 `driscoll-kraay` 的 alias。Driscoll-Kraay kernel aliases 包括 Bartlett/Newey-West、Parzen/Gallant 与 QS/Quadratic-Spectral/Andrews。`PooledOLS(cov_type="hac")` 仍是独立的 ordered-sequence Bartlett/Newey-West calculation，不应与 Driscoll-Kraay 混为一谈；若提供 `time_index`，PooledOLS 会先按该 index 排序再计算 HAC。

## Validation Matrix

下表记录这些 statistical definitions 如何与独立实现进行比较。GPU consistency 另外与 NumPy 比较，这样“和外部统计 package 的定义一致”与“CPU/GPU 计算一致”不会混在同一个 validation 中。

| Layer | Reference | 比较内容 | Assertion tolerance |
|---|---|---|---|
| HC primitives | `statsmodels==0.14.6` | full-rank OLS regression 上的 HC2/HC3 | `rtol=5e-12`, `atol=5e-14` |
| Cluster / DK primitives | `linearmodels==7.0` | one-/two-way group-debiased clustering；Bartlett/Parzen/QS weights 与 DK covariance；default bandwidth 与 fixed-effect df adjustment | covariance `rtol=5e-12`, `atol=5e-14`；weights `rtol=5e-14`, `atol=5e-15` |
| PooledOLS / PanelOLS | `linearmodels==7.0` | coefficient、DK covariance/BSE，以及 PooledOLS group-debiased cluster covariance | coefficient `rtol=2e-10`, `atol=2e-11`；covariance/BSE `rtol=5e-9`, `atol=5e-11` |
| BetweenOLS / FirstDifferenceOLS | `statsmodels==0.14.6` | 使用相同 averaging/differencing transformation 后的 coefficient 与 HC0/HC2/HC3 covariance/BSE | coefficient `rtol=5e-10`, `atol=5e-12`；covariance/BSE `rtol=5e-9`, `atol=5e-11` |
| RandomEffects transformed regression | `linearmodels==7.0`, `statsmodels==0.14.6` | statgpu Swamy-Arora quasi-demeaned $X^*,y^*$ 上的 robust/HC2/HC3/DK covariance；不宣称 coefficient parity | covariance `rtol=5e-9`, `atol=5e-11` |
| R external checks | `plm==2.6-7`, `sandwich==3.1-3` | HC0/HC2/HC3 covariance 与 one-way FE coefficient | covariance `rtol=5e-9`, `atol=5e-11`；FE coefficient `rtol=5e-10`, `atol=5e-11` |
| Physical GPU | NumPy reference | 每个 CuPy/Torch backend 的 35 个 estimator cases + 12 个 covariance-primitive cases | 默认 `rtol=5e-6`, `atol=5e-7` |

无 fixed effects 的 `PanelOLS` level regression 还会与 `statsmodels==0.14.6` 比较 coefficient、covariance/BSE、$R^2$、adjusted $R^2$ 与 model F statistics。

ill-conditioned 但 full-rank 的 stress tests 使用 scale-aware tolerance，因为 covariance entries 可能非常大：HC0 对 statsmodels 使用 `rtol=2e-6, atol=5e-3`；stable HC2/HC3 leverage checks 在 variance 可能超过 $10^{10}$ 时使用 `rtol=5e-11, atol=5e-3`。

表中的 CI tolerance 是 pass/fail threshold，不是实际观测误差。P100 physical validation 另外保存每个字段实际的 `max_abs_differences`，位于 `results/pr126_p100_fresh/panel_stage_c_correctness_p100.json`；summary 位于 `results/pr126_p100_fresh/validation_summary.txt`。

对应的 external tests 为 `dev/tests/test_panel_stage_c_external.py`、`dev/tests/test_panel_stage_c_external_defaults.py`、`dev/tests/test_panel_stage_c_linearmodels_estimators.py` 与 `dev/tests/test_panel_stage_c_r_external.py`。

## 参考（References）

- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*, 48(4), 817-838. [https://doi.org/10.2307/1912934](https://doi.org/10.2307/1912934)
- MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties. *Journal of Econometrics*, 29(3), 305-325. [https://doi.org/10.1016/0304-4076(85)90158-7](https://doi.org/10.1016/0304-4076(85)90158-7)
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703-708. [https://doi.org/10.2307/1913610](https://doi.org/10.2307/1913610)
- Andrews, D. W. K. (1991). Heteroskedasticity and autocorrelation consistent covariance matrix estimation. *Econometrica*, 59(3), 817-858. [https://doi.org/10.2307/2938229](https://doi.org/10.2307/2938229)
- Driscoll, J. C., & Kraay, A. C. (1998). Consistent covariance matrix estimation with spatially dependent panel data. *The Review of Economics and Statistics*, 80(4), 549-560. [https://doi.org/10.1162/003465398557825](https://doi.org/10.1162/003465398557825)
- Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011). Robust inference with multiway clustering. *Journal of Business & Economic Statistics*, 29(2), 238-249. [https://doi.org/10.1198/jbes.2010.07136](https://doi.org/10.1198/jbes.2010.07136)