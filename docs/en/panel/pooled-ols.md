# PooledOLS

> Language: English  
> Last updated: 2026-08-13  
> Switch: [Chinese](../../cn/panel/pooled-ols.md)

`PooledOLS` always includes an intercept. With

$$
Z=[\mathbf 1,X],
$$

the estimator is

$$
\widehat\beta_{\mathrm{pooled}}
=(Z^\top Z)^+Z^\top y.
$$

Covariance uses the level design $Z$; see [Panel covariance](covariance.md). `cov_type="hac"` is the legacy row-order Bartlett/Newey-West HAC, while Driscoll-Kraay first groups influence scores by `time_index`.

```python
from statgpu.panel import PooledOLS

model.fit(X, y, cluster=None, time_index=None, entity_ids=None)
```

`cluster` is required for clustered covariance and `time_index` for Driscoll-Kraay. `entity_ids` does not change coefficients; it enables standardized within/between $R^2$ and the Breusch-Pagan LM test. Formula input must retain the model intercept.

Main options are `cov_type`, `alpha`, `bandwidth`, `kernel`, `group_debias`, and `device`. See [Panel diagnostics](diagnostics.md).
