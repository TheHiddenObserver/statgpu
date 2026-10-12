# 固定收缩：用适度偏差换取稳定性

> 语言：中文
> 最后更新：2026-10-12
> 切换：[English](../../en/models/covariance-shrunk.md)

## 为什么协方差能求逆，却仍可能不可靠？

如果两个特征几乎相同，它们之差的波动就很小。经验协方差在这个方向上会出现很小的特征值，求逆又会放大微小的估计误差，使距离或下游计算变得不稳定。

`ShrunkCovariance` 允许你主动把估计拉向一个更简单的矩阵，通过 `shrinkage` 控制保留多少样本细节。当强度需要由敏感性分析、领域要求或你自己的验证流程决定时，它很合适。[协方差概览](covariance.md) 介绍其他可选估计器。

## 收缩到底改变了什么？

设 $S$ 是以 $n$ 为分母的[经验协方差](covariance-empirical.md)，$\mu=\operatorname{tr}(S)/p$ 是平均特征方差。给定 $\alpha\in[0,1]$，

$$
\widehat\Sigma=(1-\alpha)S+\alpha\mu I.
$$

这里的 $\alpha$ 对应 `shrinkage`，默认值为 `0.1`。

- 取 `0` 时，协方差就是经验估计。
- 介于 `0` 和 `1` 之间时，非对角线协方差乘以 $1-\alpha$，各个方差向平均方差靠拢。
- 取 `1` 时，结果为 $\mu I$：各特征方差相等，交叉协方差为零。

经验特征值 $\lambda_j$ 变为 $(1-\alpha)\lambda_j+\alpha\mu$。只要 $\mu>0$，正的收缩强度就会使最小特征值为正，即使原来的 $S$ 秩亏也如此。矩阵的迹保持不变。不过，条件数改善的代价是引入偏差：真实的方差差异和变量关系也会被削弱。这里的目标是缩放单位矩阵，既不是用户自定义目标，也不会保留每一个原始方差。

## 选择固定强度，还是自动估计？

需要明确指定强度，或想按实际任务的验证指标比较一组强度时，使用固定收缩。不想搜索验证网格时，可用 [Ledoit–Wolf](covariance-ledoit-wolf.md) 解析估计强度；高斯假设合理且样本有限时，可比较 [OAS](covariance-oas.md)。固定收缩和自动收缩都不是为了消除离群点的影响而设计的。

## CPU 示例：稳定近乎重复的特征

下面的自包含示例生成 80 行数据，两个特征仅相差少量噪声。我们比较收缩前后的特征值，而不是把成功求逆当作估计可靠的证据。

<!-- example: covariance-shrunk -->
```python
import numpy as np
from statgpu.covariance import EmpiricalCovariance, ShrunkCovariance

np.set_printoptions(precision=3, suppress=True)
rng = np.random.default_rng(42)
z = rng.normal(size=80)
X = np.column_stack([z, z + 0.05 * rng.normal(size=80)])
```

在完全相同的观测上拟合两个估计器。

```python
empirical = EmpiricalCovariance(device="cpu").fit(X)
model = ShrunkCovariance(shrinkage=0.2, device="cpu").fit(X)
```

比较特征值与条件数。

```python
print(f"shrinkage: {model.shrinkage_:.3f}")
print("empirical eigenvalues:", np.round(
    np.linalg.eigvalsh(empirical.covariance_), 3
))
print("shrunk eigenvalues:", np.round(
    np.linalg.eigvalsh(model.covariance_), 3
))
print(f"condition numbers: {np.linalg.cond(empirical.covariance_):.1f}"
      f" -> {np.linalg.cond(model.covariance_):.1f}")
```
<!-- example-end: covariance-shrunk -->

输出：

```text
shrinkage: 0.200
empirical eigenvalues: [0.001 1.195]
shrunk eigenvalues: [0.12  1.075]
condition numbers: 1117.4 -> 8.9
```

小特征值对应两个特征之差的微小波动。收缩提高了小特征值，同时降低了大特征值。这里的条件数就是最大与最小特征值之比，它明显下降。`shrinkage_` 记录的是你指定的 `0.2`，不是估计器学习或优化得到的结果。

这个示例展示了数值稳定性的改善，并不说明 `0.2` 最优。模拟中的近乎完全相关是真实结构，因此过强的收缩即使改善条件数，也可能丢失有用信息。

## 如何选择强度

1. 先判断特征的原始单位是否适合采用共同方差目标。如果需要标准化，只用训练数据确定变换，并将同一变换应用到验证集和测试集。
2. 在相同训练数据上拟合 `[0, 1]` 内的候选强度。如果目标是高斯密度估计，可以比较验证数据的 `score`，越大越好。对于其他下游任务，应使用该任务的验证指标。
3. 用验证结果选定强度后，在可用训练数据上重新拟合，把最终测试集留给最后评估。这个类没有内置的交叉验证选择器。

默认的 `0.1` 只是起点，不是根据当前数据自动给出的建议。除非有意将中心固定为零，否则保留 `assume_centered=False`。各候选模型应采用相同的中心化方式和特征尺度，才能公平比较。

## 限制与后续选择

`shrinkage` 必须是 `[0, 1]` 内的有限值，否则会抛出 `ValueError`。拟合至少需要两个观测、一个特征，且所有输入值都必须为有限数。如果中心化后所有观测均为零，那么 $\mu=0$，收缩无法创造变异。条件数良好也不保证总体估计准确、能够抵抗离群点，或得到稀疏精度矩阵。

这些其他目标可参阅[协方差概览](covariance.md)。[完整 API 参考](../reference/covariance-api.md) 列出全部参数、方法和拟合属性；设备设置见其中的[设备与数据类型](../reference/covariance-api.md#device-and-dtype)。

## 参考资料

- [scikit-learn：ShrunkCovariance](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.ShrunkCovariance.html)，固定缩放单位矩阵目标的收缩估计。
- Ledoit, O., & Wolf, M. (2004).
  [A well-conditioned estimator for large-dimensional covariance matrices](https://doi.org/10.1016/S0047-259X(03)00096-4).
