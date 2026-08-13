# BetweenOLS

> 语言：中文  
> 最后更新：2026-08-13  
> 切换：[English](../../en/panel/between-ols.md)

对 entity $i$，

$$
\bar y_i=T_i^{-1}\sum_t y_{it},
\qquad
\bar x_i=T_i^{-1}\sum_t x_{it}.
$$

令 $\bar Z_i=(1,\bar x_i^\top)^\top$，则

$$
\widehat\beta_B
=(\bar Z^\top\bar Z)^+\bar Z^\top\bar y.
$$

covariance 在 entity-mean fit space 上计算；nonrobust 与 HC0/HC1/HC2/HC3 的公式见 [面板 covariance](covariance.md)。

```python
from statgpu.panel import BetweenOLS
model.fit(X, y, entity_ids=entity_ids)
```

`entity_ids` 必需。模型始终包含截距。主要选项为 `cov_type`、`alpha` 和 `device`。
