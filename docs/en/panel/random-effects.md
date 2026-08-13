# RandomEffects

> Language: English  
> Last updated: 2026-08-13  
> Switch: [Chinese](../../cn/panel/random-effects.md)

## Model and Estimator

The one-way random-effects model is

$$
y_{it}=x_{it}^{\top}\beta+a_i+\varepsilon_{it},
\qquad
\operatorname{Var}(a_i)=\sigma_a^2,
\qquad
\operatorname{Var}(\varepsilon_{it})=\sigma_e^2.
$$

Swamy-Arora first estimates

$$
\widehat\sigma_e^2=\frac{RSS_W}{df_W},
\qquad
\bar T_H=\frac{N}{\sum_{i=1}^N T_i^{-1}},
$$

and then

$$
\widehat\sigma_a^2=
\max\left\{0,\frac{RSS_B/df_B-\widehat\sigma_e^2}{\bar T_H}\right\}.
$$

Here $df_W=n-r_W-r_E$, with $r_E=N$ when the level design has an explicit constant and $r_E=N-1$ otherwise; $df_B=N-r_B$. For entity $i$,

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

so the feasible-GLS estimator is

$$
\widehat\beta_{\mathrm{RE}}
=(X^{*\top}X^*)^+X^{*\top}y^*.
$$

The documented rank-deficient extension replaces raw auxiliary-regression column counts by their identified numerical ranks.

## Covariance

All residual-based covariance estimators use the quasi-demeaned fit space

$$
Z=X^*,
\qquad
e^*=y^*-X^*\widehat\beta_{\mathrm{RE}}.
$$

For example,

$$
\widehat V_{\mathrm{nonrobust}}
=\widehat\sigma_*^2(X^{*\top}X^*)^+,
\qquad
\widehat\sigma_*^2=
\frac{e^{*\top}e^*}{n-\operatorname{rank}(X^*)}.
$$

HC0/HC1/HC2/HC3, clustered, and Driscoll-Kraay use the shared definitions in [Panel covariance](covariance.md) with $X^*$ as the fit-space design.

## API

```python
from statgpu.panel import RandomEffects

model.fit(X, y, entity_ids=entity_ids, time_ids=None, cluster=None)
```

`entity_ids` is required. Driscoll-Kraay requires `time_ids`; clustered covariance requires `cluster`. Main options are `cov_type`, `bandwidth`, `kernel`, `group_debias`, `alpha`, and `device`.

Formula input retains the normal R/Patsy intercept; `0 +` or `-1` requests a no-intercept random-effects model.

`variance_components_` stores $\widehat\sigma_e^2$ and $\widehat\sigma_a^2$. Classical FE-versus-RE Hausman testing is described in [Panel diagnostics](diagnostics.md).
