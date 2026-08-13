# FamaMacBeth

> 语言：中文  
> 最后更新：2026-08-13  
> 切换：[English](../../en/panel/fama-macbeth.md)

对保留的时期 $t$，

$$
\widehat\beta_t=(X_t^\top X_t)^+X_t^\top y_t,
\qquad
\widehat\beta_{\mathrm{FM}}=T^{-1}\sum_{t=1}^T\widehat\beta_t.
$$

令 $u_t=\widehat\beta_t-\widehat\beta_{\mathrm{FM}}$。当 `cov_type="nonrobust"` 时，

$$
\widehat V_{\mathrm{FM}}
=
\frac{1}{T}
\left[
\frac{1}{T-1}\sum_{t=1}^T u_tu_t^\top
\right].
$$

`cov_type="newey-west"` 将括号内部分替换为 coefficient series 的 Bartlett HAC long-run covariance；这与 residual-based panel covariance 不同。

```python
from statgpu.panel import FamaMacBeth

model.fit(X, y, time_ids=time_ids, entity_ids=None)
```

`time_ids` 必需。`entity_ids` 仅用于 standardized within/between $R^2$。主要选项为 `cov_type`、`bandwidth`、`alpha`、`min_obs_per_period` 和 `device`。`betas_` 保存各期 coefficient estimate。
