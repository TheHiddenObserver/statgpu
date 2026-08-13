# RandomEffects

> 语言：中文  
> 最后更新：2026-08-13  
> 切换：[English](../../en/panel/random-effects.md)

## 模型与估计量

单向随机效应模型为

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
\theta_i=1-\sqrt{
\frac{\widehat\sigma_e^2}
{\widehat\sigma_e^2+T_i\widehat\sigma_a^2}},
$$

$$
y_{it}^*=y_{it}-\theta_i\bar y_i,
\qquad
x_{it}^*=x_{it}-\theta_i\bar x_i,
$$

因此 feasible GLS 估计量为

$$
\widehat\beta_{\mathrm{RE}}
=(X^{*\top}X^*)^+X^{*\top}y^*.
$$

rank-deficient extension 中，辅助回归的原始列数由 identified numerical rank 替代。

## Covariance

所有 residual-based covariance 都在 quasi-demeaned fit space

$$
Z=X^*,
\qquad
e^*=y^*-X^*\widehat\beta_{\mathrm{RE}}
$$

上计算。例如

$$
\widehat V_{\mathrm{nonrobust}}
=\widehat\sigma_*^2(X^{*\top}X^*)^+,
\qquad
\widehat\sigma_*^2=
\frac{e^{*\top}e^*}{n-\operatorname{rank}(X^*)}.
$$

HC0/HC1/HC2/HC3、clustered 与 Driscoll-Kraay 的统一公式见 [面板 covariance](covariance.md)，这里只需将 fit-space design 取为 $X^*$。

## API

```python
from statgpu.panel import RandomEffects

model.fit(X, y, entity_ids=entity_ids, time_ids=None, cluster=None)
```

`entity_ids` 必需；Driscoll-Kraay 需要 `time_ids`，clustered covariance 需要 `cluster`。主要选项为 `cov_type`、`bandwidth`、`kernel`、`group_debias`、`alpha` 和 `device`。

`variance_components_` 保存 $\widehat\sigma_e^2$ 与 $\widehat\sigma_a^2$。Classical FE-versus-RE Hausman 检验见 [面板 diagnostics](diagnostics.md)。
