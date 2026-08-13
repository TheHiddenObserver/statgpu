# FamaMacBeth

> Language: English  
> Last updated: 2026-08-13  
> Switch: [Chinese](../../cn/panel/fama-macbeth.md)

For retained period $t$,

$$
\widehat\beta_t=(X_t^\top X_t)^+X_t^\top y_t,
\qquad
\widehat\beta_{\mathrm{FM}}=T^{-1}\sum_{t=1}^T\widehat\beta_t.
$$

Let $u_t=\widehat\beta_t-\widehat\beta_{\mathrm{FM}}$. With `cov_type="nonrobust"`,

$$
\widehat V_{\mathrm{FM}}
=
\frac{1}{T}
\left[
\frac{1}{T-1}\sum_{t=1}^T u_tu_t^\top
\right].
$$

`cov_type="newey-west"` replaces the bracketed term with a Bartlett HAC long-run covariance of the coefficient series. This is distinct from residual-based panel covariance.

```python
from statgpu.panel import FamaMacBeth

model.fit(X, y, time_ids=time_ids, entity_ids=None)
```

`time_ids` is required. `entity_ids` only enables standardized within/between $R^2$. Main options are `cov_type`, `bandwidth`, `alpha`, `min_obs_per_period`, and `device`. `betas_` stores the retained period-specific estimates.
