# 逐节点 Lasso 推断调参迁移

> 最后更新：2026-09-17  
> 切换：[English](../../en/guides/nodewise-alpha-migration.md)

稀疏 Gaussian debiased inference 对 `Lasso`、`ElasticNet`、相应 Gaussian penalized interface，以及 `LassoCV` / `ElasticNetCV` 的 final-refit inference 暴露 public `nodewise_alpha` 控制。

## `alpha` 与 `nodewise_alpha` 是不同参数

- `alpha` 控制 penalized prediction/selection fit；
- `nodewise_alpha` 只控制 debiased inference 中用于估计近似 design precision matrix 的 node-wise Lasso regression。

调用者给出的有限正值会直接使用；`nodewise_alpha=None` 请求 statgpu 的自动规则。

## 自动规则

自动 node-wise penalty 定义在标准化后的 centered/weighted working design 上，其尺度为

$$
\lambda_{\mathrm{nw}}
=
\sqrt{\frac{2\log(\max(p,2))}{n_{\mathrm{nw}}}},
$$

无 analytic weight 时 `n_nw=n`；非均匀 analytic weight 下使用 Kish-style effective sample size。

$\sqrt{\log(p)/n}$ 的量级来自高维 node-wise regression 的理论动机；具体常数和 effective-sample-size convention 是 statgpu 的默认选择，并不是某个定理规定的唯一形式。

此前内部使用的 response-scale-dependent rule 不再作为 compatibility option 暴露。如果需要固定的 node-wise penalty，应在标准化 node-wise scale 上显式设置 `nodewise_alpha=`。

## Precision matrix 构造

令 `X_w` 表示 sparse Gaussian inference 使用的 centered/weighted working design。statgpu 通过

$$
d_j^2
=
\frac{1}{n}\sum_i X_{w,ij}^2,
\qquad
Z=X_wD^{-1}
$$

对列进行标准化。

对每个 feature，在 `Z` 上求解 node-wise Lasso，并采用

$$
\hat\tau_j^2
=
\frac{\lVert r_j\rVert_2^2}{n}
+
\lambda_{\mathrm{nw}}\lVert\hat\gamma_j\rVert_1
$$

作为 residual normalizer。标准化尺度上的近似 precision matrix 最后再变换回 working-feature scale。

如果 feature scale 退化、precision state 非有限、normalizer 无效，或 post-solve optimality check 失败，inference 会报错，而不是发布 placeholder precision row。

当 `p=1` 时没有 nuisance node-wise regression；statgpu 直接使用一维解析 precision，并令 `nodewise_alpha_` 保持为 `None`。

## 实际使用的值

多 feature debiased inference 会通过 `nodewise_alpha_` 暴露实际采用的值。这样用户可以检查自动 tuning 结果，而无需把 node-wise solver 的内部 stopping setting 变成额外 public tuning parameter。

## Cross-validation

`LassoCV(nodewise_alpha=...)` 与 `ElasticNetCV(nodewise_alpha=...)` 把 `nodewise_alpha` 视为**仅属于 final-refit inference 的配置**。

它不会进入：

- 主 alpha grid；
- fold scoring；
- `alpha_` 的选择；
- `l1_ratio_` 的选择。

只有 CV 已经选出 prediction model，并在全部数据上进行 final refit 后执行 inference 时，`nodewise_alpha` 才参与计算。

## Backend 与 analytic weights

在 debiased route 支持的范围内，NumPy、CuPy 与 Torch 使用同一个标准化统计定义。显式 CUDA/Torch inference 请求不会被静默替换成 CPU 数值计算。

Analytic weights 继续使用 sparse Gaussian average-loss convention；尤其是所有正权重同乘一个常数不会改变自动 node-wise tuning 的统计 target。

## 迁移建议

如果过去并没有依赖某个内部 node-wise penalty 数值，保留 `nodewise_alpha=None`，直接使用新的标准化自动规则即可。

如果可复现性要求固定 node-wise tuning value，应显式设置：

```python
from statgpu.linear_model import Lasso

model = Lasso(
    alpha=0.05,
    compute_inference=True,
    inference_method="debiased",
    nodewise_alpha=0.08,
)
model.fit(X, y)

print(model.nodewise_alpha_)
```

不要机械地把主模型 `alpha` 复制到 `nodewise_alpha`：两者对应不同优化问题，并且工作尺度也不同。

## 相关文档

- [推断模式](inference-modes.md) — debiased、post-selection 与 bootstrap 的选择
- [交叉验证](cross-validation.md) — selection 与 final-refit 语义
- [Lasso](../models/lasso.md) 与 [ElasticNet](../models/elastic-net.md) — 模型专属 inference 控制

## 参考文献

- van de Geer, S., Buhlmann, P., Ritov, Y., & Dezeure, R. (2014). On asymptotically optimal confidence regions and tests for high-dimensional models. *Annals of Statistics*, 42(3), 1166-1202.
- Zhang, C.-H., & Zhang, S. S. (2014). Confidence intervals for low-dimensional parameters in high-dimensional linear models. *JRSS B*, 76(1), 217-242.
- Javanmard, A., & Montanari, A. (2014). Confidence intervals and hypothesis testing for high-dimensional regression. *JMLR*, 15, 2869-2909.
