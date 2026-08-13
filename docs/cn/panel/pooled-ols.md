# PooledOLS

> 语言：中文  
> 最后更新：2026-08-13  
> 切换：[English](../../en/panel/pooled-ols.md)

`PooledOLS` 始终包含截距。令

$$
Z=[\mathbf 1,X],
$$

则

$$
\widehat\beta_{\mathrm{pooled}}
=(Z^\top Z)^+Z^\top y.
$$

covariance 使用 level design $Z$，见 [面板 covariance](covariance.md)。`cov_type="hac"` 是 legacy row-order Bartlett/Newey-West HAC；Driscoll-Kraay 则先按 `time_index` 聚合 influence score。

```python
from statgpu.panel import PooledOLS

model.fit(X, y, cluster=None, time_index=None, entity_ids=None)
```

clustered covariance 需要 `cluster`；Driscoll-Kraay 需要 `time_index`。`entity_ids` 不改变 coefficient，只启用 standardized within/between $R^2$ 与 Breusch-Pagan LM。Formula 必须保留模型截距。

主要选项为 `cov_type`、`alpha`、`bandwidth`、`kernel`、`group_debias` 和 `device`。见 [面板 diagnostics](diagnostics.md)。
