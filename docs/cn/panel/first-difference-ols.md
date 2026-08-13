# FirstDifferenceOLS

> 语言：中文  
> 最后更新：2026-08-13  
> 切换：[English](../../en/panel/first-difference-ols.md)

对 entity $i$ 内相邻的已观测时期，

$$
\Delta y_{it}=y_{it}-y_{i,t^-},
\qquad
\Delta x_{it}=x_{it}-x_{i,t^-},
$$

且

$$
\widehat\beta_{\mathrm{FD}}
=(\Delta X^\top\Delta X)^+\Delta X^\top\Delta y.
$$

covariance 使用 $Z=\Delta X$，见 [面板 covariance](covariance.md)。

```python
from statgpu.panel import FirstDifferenceOLS

model.fit(X, y, entity_ids=entity_ids, time_ids=None)
```

`entity_ids` 必需。提供 `time_ids` 时，`(entity_id, time_id)` 必须唯一。差分使用相邻的**已观测**时间；calendar gap 不会被补齐，也不会除以 gap 长度。ordered categorical time label 保留声明的顺序。

主要选项为 `cov_type`、`alpha` 和 `device`。
