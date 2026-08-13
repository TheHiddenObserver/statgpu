# FirstDifferenceOLS

> Language: English  
> Last updated: 2026-08-13  
> Switch: [Chinese](../../cn/panel/first-difference-ols.md)

For consecutive observed periods within entity $i$,

$$
\Delta y_{it}=y_{it}-y_{i,t^-},
\qquad
\Delta x_{it}=x_{it}-x_{i,t^-},
$$

and

$$
\widehat\beta_{\mathrm{FD}}
=(\Delta X^\top\Delta X)^+\Delta X^\top\Delta y.
$$

Covariance uses $Z=\Delta X$; see [Panel covariance](covariance.md).

```python
from statgpu.panel import FirstDifferenceOLS

model.fit(X, y, entity_ids=entity_ids, time_ids=None)
```

`entity_ids` is required. When `time_ids` is supplied, `(entity_id, time_id)` pairs must be unique. Differences use consecutive **observed** times; calendar gaps are neither filled nor divided out. Ordered categorical time labels preserve declared chronology.

Main options are `cov_type`, `alpha`, and `device`.
