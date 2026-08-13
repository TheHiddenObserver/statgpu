# BetweenOLS

> Language: English  
> Last updated: 2026-08-13  
> Switch: [Chinese](../../cn/panel/between-ols.md)

For entity $i$,

$$
\bar y_i=T_i^{-1}\sum_t y_{it},
\qquad
\bar x_i=T_i^{-1}\sum_t x_{it}.
$$

With $\bar Z_i=(1,\bar x_i^\top)^\top$,

$$
\widehat\beta_B
=(\bar Z^\top\bar Z)^+\bar Z^\top\bar y.
$$

Covariance is computed on the entity-mean fit space; nonrobust and HC0/HC1/HC2/HC3 are defined in [Panel covariance](covariance.md).

```python
from statgpu.panel import BetweenOLS
model.fit(X, y, entity_ids=entity_ids)
```

`entity_ids` is required. The model always includes an intercept. Main options are `cov_type`, `alpha`, and `device`.
