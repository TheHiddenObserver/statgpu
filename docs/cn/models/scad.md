# SCAD 回归

> 语言：简体中文
> 最后更新：2026-09-03
> 切换：[English](../../en/models/scad.md)

## 它解决什么问题？

`SCADRegression` 使用平滑截断绝对偏差（SCAD）惩罚拟合稀疏线性模型。与 Lasso 一样，它可以把不重要的系数压到零；与 Lasso 不同，它会逐渐减小对大系数的惩罚，从而降低强信号的收缩偏差。

## 什么时候使用？

以下情况可以考虑 SCAD：

- 响应连续，候选特征中可能有许多无关变量；
- 希望得到稀疏且可解释的模型；
- Lasso 能选中有效变量，但对大系数收缩过强；
- 可以使用留出数据或交叉验证调整惩罚强度。

如果更看重凸目标与稳定的全局优化路径，优先使用 Lasso 或 Elastic Net；如果大多数变量都可能具有小效应，或特征高度相关，可以考虑 Ridge。自动特征选择不能替代因果识别设计。

## 直观理解

SCAD 根据系数大小改变拉向零的力度：

1. 小系数受到类似 L1 的惩罚，可能被压到零；
2. 中等系数受到逐渐减弱的收缩；
3. 足够大的系数不再受到额外收缩。

这种形状试图同时获得稀疏性与较低的大信号偏差。代价是目标函数非凸：不同初值或调参结果可能进入不同的局部解。

## 模型与惩罚

SCAD 最小化

$$
\frac{1}{2n}\lVert y-\beta_0\mathbf 1-X\beta\rVert_2^2
+
\sum_{j=1}^{p}p_{\lambda,a}(|\beta_j|),
$$

其中 $\lambda$ 对应 `alpha`，且

$$
p_{\lambda,a}(\theta)=
\begin{cases}
\lambda\theta,
&0\leq\theta\leq\lambda,\\
\dfrac{2a\lambda\theta-\theta^2-\lambda^2}{2(a-1)},
&\lambda<\theta\leq a\lambda,\\
\dfrac{(a+1)\lambda^2}{2},
&\theta>a\lambda.
\end{cases}
$$

常用值为 $a=3.7$。在正则条件与合适调参序列下，SCAD 可以具有选择一致性和 oracle 风格的渐近性质；这不是对每个有限样本的无条件保证。

## 最小可运行示例

下面的数据包含 20 个标准化特征，其中前 4 个是真实信号。

```python
import numpy as np
from statgpu.linear_model import SCADRegression

rng = np.random.default_rng(2)
X = rng.normal(size=(600, 20))
X = (X - X.mean(axis=0)) / X.std(axis=0)

true_coef = np.zeros(20)
true_coef[:4] = [2.5, -1.8, 1.2, 0.8]
y = 0.7 + X @ true_coef + rng.normal(scale=0.8, size=600)

model = SCADRegression(
    alpha=0.08,
    a=3.7,
    device="cpu",
    compute_inference=False,
).fit(X, y)

selected = np.flatnonzero(np.abs(model.coef_) > 1e-8)
print("入选特征下标：", selected)
print("系数：", model.coef_)
print("训练集 R²：", model.score(X, y))
```

前 4 个特征通常会被保留，但具体结果取决于噪声与 `alpha`。训练集上入选并不能证明该特征在样本外仍有用。

## 如何读取结果？

- 等于零或非常接近零的系数表示该特征未入选。
- 非零系数可按线性回归系数解释，但解释以当前入选模型和特征尺度为条件。
- `intercept_` 不受惩罚。
- `score(X, y)` 返回 $R^2$。
- 入选集合可能在不同重采样中变化，应检查稳定性，而不是把单次结果当作确定事实。

## 关键参数怎么选？

| 参数 | 默认值 | 选择建议 |
|---|---:|---|
| `alpha` | `1.0` | 值越大越稀疏；在对数尺度网格上用交叉验证或验证集选择 |
| `a` | `3.7` | 通常保留文献默认值，除非进行了明确的敏感性分析 |
| `fit_intercept` | `True` | 通常保持开启 |
| `max_iter` | `1000` | 求解器未收敛时增加 |
| `tol` | `1e-4` | 对系数精度要求更高时收紧，同时接受更大计算量 |
| `device` | `"auto"` | 根据规模和后端可用性选择 `cpu`、`cuda` 或 `torch` |
| `solver` | `"auto"` | 除非需要复现特定配置，否则让 statgpu 选择兼容路径 |
| `compute_inference` | `False` | 类型化的新手路径保持关闭；高级用法见下方推断说明 |

调整 `alpha` 前应先标准化特征，否则相同惩罚强度会把测量单位差异误当成重要性差异。

## 与其他方法比较

| 性质 | Lasso | Elastic Net | SCAD |
|---|---|---|---|
| 目标函数 | 凸 | 凸 | 非凸 |
| 精确零系数 | 有 | 有 | 有 |
| 大信号收缩 | 持续收缩 | 持续收缩 | 超过 SCAD 阈值后减弱 |
| 高相关特征 | 可能不稳定地只选一个 | 通常更稳定 | 仍可能不稳定 |
| 全局最优保证 | 对凸目标有 | 对凸目标有 | 无 |

## CPU、GPU 与推断支持

兼容路径中，SCAD 估计支持 CPU、CuPy CUDA 与 Torch CUDA：

```python
gpu_model = SCADRegression(
    alpha=0.08,
    device="cuda",
    compute_inference=False,
).fit(X, y)
```

类型化 `SCADRegression` 构造器暴露了 `compute_inference`，但没有暴露 `inference_method` 选择器。因此，只写 `SCADRegression(compute_inference=True)` **不是**文档推荐的推断入口：继承的默认推断方法与 SCAD 不兼容。

设备驻留的 LLA/FISTA、同步、精度以及 CPU active-set 重拟合边界见 [CPU/GPU 加速实现](../guides/acceleration-internals.md)。

如需经过当前实现路径核对的 active-set 推断，应显式使用通用惩罚估计器：

```python
from statgpu.linear_model import PenalizedLinearRegression

inferential_model = PenalizedLinearRegression(
    penalty="scad",
    penalty_kwargs={"a": 3.7},
    alpha=0.08,
    device="cpu",
    compute_inference=True,
    inference_method="oracle",
).fit(X, y)

print(inferential_model._bse)
print(inferential_model._pvalues)
```

该过程会在 active set 上重新拟合无惩罚模型。推断以成功选择为条件，并依赖 oracle-property 假设，不等于一般的有限样本选择后推断。即使最初在 GPU 上估计，active-set 重拟合当前仍在 CPU 上完成。

## 常见误区

- 调整 `alpha` 时不能使用最终测试集。
- 应检查不同折或重采样中的选择稳定性。
- 不要写成“SCAD 总是具有 oracle property”；该结论依赖假设与合适的调参序列。
- 目标函数非凸，应检查收敛以及结果对调参的敏感性。
- 稀疏预测模型不能证明未入选变量在科学意义上确实为零。

## API 与验证

主要路径：`statgpu.linear_model.SCADRegression`

高级路径：`statgpu.linear_model.PenalizedLinearRegression(penalty="scad")`

主要估计输出为 `coef_`、`intercept_`、`fit`、`predict` 和 `score`。通用 oracle 推断路径还会为 active-set 重拟合填充 `_bse`、`_zvalues`、`_pvalues` 与 `_conf_int`。

验证覆盖非凸求解器行为、CPU/GPU 一致性、收敛契约与推断分发。性能数据应从版本化[基准面板](/dashboard/)读取，而不是把某次测速写成永久结论。

## 参考文献

- Fan, J., & Li, R. (2001). Variable selection via nonconcave penalized likelihood and its oracle properties. *Journal of the American Statistical Association*, 96(456), 1348-1360.
- Wang, H., Li, R., & Tsai, C.-L. (2007). Tuning parameter selectors for the smoothly clipped absolute deviation method. *Biometrika*, 94(3), 553-568.
- Zou, H., & Li, R. (2008). One-step sparse estimates in nonconcave penalized likelihood models. *Annals of Statistics*, 36(4), 1509-1533.
