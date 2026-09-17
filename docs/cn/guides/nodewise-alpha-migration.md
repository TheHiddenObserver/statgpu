# 逐节点 Lasso 推断调参迁移

> 最后更新：2026-09-17  
> 切换：[English](../../en/guides/nodewise-alpha-migration.md)

稀疏 Gaussian 模型的去偏推断，在 `Lasso`、`ElasticNet`、相应的 Gaussian 惩罚模型接口，以及 `LassoCV` / `ElasticNetCV` 的最终重拟合推断中，都提供公开的 `nodewise_alpha` 控制参数。

## `alpha` 与 `nodewise_alpha` 是不同参数

- `alpha` 控制主惩罚模型的预测/变量选择拟合；
- `nodewise_alpha` 只控制去偏推断中用于估计近似设计精度矩阵的逐节点 Lasso 回归。

调用者给出的有限正值会直接使用；`nodewise_alpha=None` 表示采用 statgpu 的自动规则。

## 自动规则

自动逐节点惩罚强度定义在标准化后的中心化/带权工作设计矩阵上，其尺度为

$$
\lambda_{\mathrm{nw}}
=
\sqrt{\frac{2\log(\max(p,2))}{n_{\mathrm{nw}}}}.
$$

没有解析权重时，`n_nw=n`；存在非均匀解析权重时，使用 Kish 型有效样本量。

$\sqrt{\log(p)/n}$ 的量级来自高维逐节点回归的理论动机；具体常数以及有效样本量的定义方式是 statgpu 的默认选择，并不是某个定理规定的唯一形式。

此前内部使用的、依赖响应变量尺度的规则不再作为兼容选项公开。如果需要固定逐节点惩罚强度，应在标准化后的逐节点尺度上显式设置 `nodewise_alpha=`。

## 精度矩阵的构造

令 `X_w` 表示稀疏 Gaussian 推断使用的中心化/带权工作设计矩阵。statgpu 先通过

$$
d_j^2
=
\frac{1}{n}\sum_i X_{w,ij}^2,
\qquad
Z=X_wD^{-1}
$$

对各列进行标准化。

随后，对每个特征在 `Z` 上求解逐节点 Lasso，并采用

$$
\hat\tau_j^2
=
\frac{\lVert r_j\rVert_2^2}{n}
+
\lambda_{\mathrm{nw}}\lVert\hat\gamma_j\rVert_1
$$

作为残差归一化量。标准化尺度上的近似精度矩阵最后再变换回原工作特征尺度。

如果特征尺度退化、精度矩阵状态出现非有限值、归一化量无效，或者求解后的最优性检查失败，推断会报错，而不是发布占位用的精度矩阵行。

当 `p=1` 时不存在需要控制的其他特征，因此不需要逐节点回归；statgpu 直接使用一维解析精度值，并令 `nodewise_alpha_` 保持为 `None`。

## 实际采用的值

当特征数大于 1 时，去偏推断会通过 `nodewise_alpha_` 暴露实际采用的逐节点惩罚强度。这样用户可以检查自动调参结果，而不需要把逐节点求解器内部的停止参数进一步公开成新的调参接口。

## 交叉验证

`LassoCV(nodewise_alpha=...)` 与 `ElasticNetCV(nodewise_alpha=...)` 把 `nodewise_alpha` 视为**只属于最终重拟合推断的配置**。

它不会参与：

- 主模型的 `alpha` 网格；
- 各数据折的评分；
- `alpha_` 的选择；
- `l1_ratio_` 的选择。

只有在 CV 已经选出预测模型，并在全部数据上完成最终重拟合之后，执行推断时 `nodewise_alpha` 才会参与计算。

## 后端与解析权重

在去偏推断受支持的范围内，NumPy、CuPy 与 Torch 使用同一个标准化统计定义。显式 CUDA/Torch 推断请求不会被静默替换成 CPU 数值计算。

解析权重继续遵循稀疏 Gaussian 模型的平均损失约定；特别地，把所有正权重同时乘上同一个常数，不会改变自动逐节点调参所对应的统计目标。

## 迁移建议

如果过去并没有依赖某个内部逐节点惩罚强度，保留 `nodewise_alpha=None`，直接使用新的标准化自动规则即可。

如果可复现性要求固定逐节点调参值，应显式设置：

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

不要机械地把主模型的 `alpha` 复制到 `nodewise_alpha`：两者对应不同的优化问题，工作尺度也不同。

## 相关文档

- [推断模式](inference-modes.md) — `debiased`、`post_selection_ols` 与 `bootstrap` 的选择
- [交叉验证](cross-validation.md) — 选择与最终重拟合的语义
- [Lasso](../models/lasso.md) 与 [ElasticNet](../models/elastic-net.md) — 模型专属推断控制

## 参考文献

- van de Geer, S., Buhlmann, P., Ritov, Y., & Dezeure, R. (2014). On asymptotically optimal confidence regions and tests for high-dimensional models. *Annals of Statistics*, 42(3), 1166-1202.
- Zhang, C.-H., & Zhang, S. S. (2014). Confidence intervals for low-dimensional parameters in high-dimensional linear models. *JRSS B*, 76(1), 217-242.
- Javanmard, A., & Montanari, A. (2014). Confidence intervals and hypothesis testing for high-dimensional regression. *JMLR*, 15, 2869-2909.
