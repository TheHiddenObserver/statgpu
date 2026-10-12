# MinCovDet：减轻异常观测对中心和协方差的影响

> 语言：中文
> 最后更新：2026-10-12
> 切换：[English](../../en/models/covariance-min-cov-det.md)

## 少量观测扭曲了整个协方差，怎么办？

一小组异常观测就可能把普通均值拉离主要数据云，并放大协方差。若再用这个被
放大的协方差计算距离，这些观测自身的异常程度反而可能被掩盖。`MinCovDet`
先从较集中的子集估计位置和协方差，再根据稳健距离重新选择参与估计的观测。

当多数行来自同一个近似椭圆分布的总体，而少数行可能受到污染时，它比较适用。
它不是通用聚类方法：真实存在的第二个群体，也可能像记录错误一样被降低影响。
通用数据和设备行为见[协方差指南](covariance.md)；全部参数、方法和已拟合属性
见[完整 API 参考](../reference/covariance-api.md)。

## 最小体积子集的直觉

对包含 $h$ 行的子集 $H$，定义

$$
\overline x_H=\frac1h\sum_{i\in H}x_i,
\qquad
S_H=\frac1h\sum_{i\in H}
(x_i-\overline x_H)(x_i-\overline x_H)^\top.
$$

最小协方差行列式的目标为

$$
\widehat H\in\arg\min_{H:\,|H|=h}\det(S_H).
$$

行列式衡量多维分散程度。寻找较小的行列式，相当于寻找较集中的一组观测，
避免让远处的数据点把估计拉长。穷举所有子集通常不可行；FAST-MCD 从多组初始
子集出发，通过集中步骤逐步改进。这个搜索是近似的，不保证找到全局最小值。
统计背景见 [FAST-MCD 原论文](https://wis.kuleuven.be/stat/robust/papers/1999/rousseeuwvandriessen-fastalgorithmformcd-technomet.pdf)。

实现先对原始子集协方差做一致性校正，再计算稳健距离平方，以自由度为 $p$ 的
卡方分布 `0.975` 分位数作为重加权阈值。随后，用选出的支持集重新估计位置和
校正后的协方差。如果保留的行数过少，则改用原始子集及其校正后的协方差。
设置 `assume_centered=True` 时，整个过程中都使用零作为中心。

## 在 CPU 上比较普通估计和稳健估计

下面的示例可以独立运行。它生成 180 行接近 `[0, 0]` 的数据，以及 20 行接近
`[6, -6]` 的明显偏移数据。只有因为这是模拟，我们才知道每一组的真实标签。
两种估计器都会接收全部 200 行，不使用标签。

<!-- example: covariance-min-cov-det -->
导入普通协方差估计器和稳健估计器。

```python
import numpy as np
from statgpu.covariance import EmpiricalCovariance, MinCovDet
```

生成干净数据与偏移组，再把两组按行合并。

```python
rng = np.random.default_rng(42)
clean = rng.multivariate_normal(
    [0.0, 0.0], [[1.0, 0.6], [0.6, 1.0]], size=180
)
contamination = rng.normal(loc=[6.0, -6.0], scale=0.5, size=(20, 2))
X = np.vstack([clean, contamination])
```

在同一份输入上拟合两个模型。

```python
ordinary = EmpiricalCovariance(device="cpu").fit(X)
robust = MinCovDet(random_state=42, device="cpu").fit(X)
```

查看两个中心、稳健支持集，以及模拟中两个已知组别的距离。

```python
print("ordinary center:", np.round(ordinary.location_, 3))
print("robust center:", np.round(robust.location_, 3))
print("robust covariance:")
print(np.round(robust.covariance_, 3))
print("final support:", int(robust.support_.sum()))
print("shifted rows in support:", int(robust.support_[180:].sum()))
print("median clean squared distance:", round(float(
    np.median(robust.mahalanobis(clean))
), 3))
print("median shifted squared distance:", round(float(
    np.median(robust.mahalanobis(contamination))
), 3))
```
<!-- example-end: covariance-min-cov-det -->

输出保留三位小数：

```text
ordinary center: [ 0.597 -0.635]
robust center: [0.056 0.027]
robust covariance:
[[0.95  0.606]
 [0.606 0.857]]
final support: 174
shifted rows in support: 0
median clean squared distance: 1.391
median shifted squared distance: 238.589
```

稳健中心仍接近模拟中干净总体的中心，稳健协方差也保留了正向关联。最终支持集
包含 174 行，全部来自干净组；这意味着它同时排除了六行真正的干净数据。
因此，`support_ == False` 不能直接解释成“这一行有问题”。这个掩码描述的是
哪些行参与了稳健拟合，不会从 `X` 中删除行，也不是具有校准错误率的异常分类器。

偏移组的距离平方中位数明显更大。`mahalanobis` 和 `predict` 都返回这种距离
平方；`dist_` 保存训练行在最终估计下的距离平方。支持集是在最终重拟合之前，
根据校正后的原始估计确定的，所以对 `dist_` 再应用同一个阈值，不一定能得到
相同的 `support_`。

## 如何选择子集大小、检查假设？

- 默认原始子集大小为 $h=\lceil(n+p+1)/2\rceil$，上限为 $n$。它是搜索
  子集的大小，不一定等于最终重加权支持集的大小。
- `support_fraction` 取值范围为 `(0, 1]`。给定后，使用
  $\lceil n\,\text{support_fraction}\rceil$ 行，并将其限制在 $[p+1,n]$。
  更大的子集通常能够利用更多干净数据，但对污染的抵抗能力会减弱。不要把它
  当作指定最终异常比例的参数。
- `random_state` 控制随机初始子集。重要分析应尝试不同随机种子，检查结果
  是否稳定，而不是只依赖一次搜索。
- 实现要求 `n_samples > n_features`。这只是支持集协方差非奇异的必要条件，
  并不充分；重复特征或线性相关特征仍可能使拟合无法得到可用的结果。
- 拟合前需处理缺失值和无穷值。单一且占主导的椭圆形数据云，比多峰、强偏斜
  或受低维流形约束的数据更符合方法的假设。

`raw_location_` 是所选原始子集的中心，`raw_covariance_` 是以 $h$ 为分母、
尚未校正的子集协方差。实际使用最终稳健估计时，应读取 `location_` 和
`covariance_`。`score(X)` 仍是平均高斯对数似然，不是稳健损失分数或异常检测
准确率。稳健估计有意限制污染行的影响，因此不一定使污染数据上的似然最大。

## 替代方案与限制

对于干净、样本充足的数据，[EmpiricalCovariance](covariance-empirical.md)
更简单，也不需要随机子集搜索。当 $p$ 接近或超过 $n$ 时，
[LedoitWolf](covariance-ledoit-wolf.md) 和 [OAS](covariance-oas.md) 可以提供
正则化，但不具备 MCD 式的抗污染能力。如果需要的是稀疏条件依赖图，应使用
[GraphicalLasso](covariance-graphical-lasso.md)；它回答的问题与稳健位置、
协方差估计不同。

MCD 的计算开销高于直接计算协方差。对逆矩阵做数值稳定化，并不能让退化的
支持集获得原本缺失的统计信息。将距离用于后续分析前，应检查输出是否有限、
协方差的条件数，以及结果对子集大小的敏感性。高斯模型下的卡方阈值可以作为
参考，但不能自动保证任意真实数据上的拟合距离具有指定的误报率。

## 参考文献与后续阅读

- Rousseeuw, P. J., & Van Driessen, K. (1999).
  [A fast algorithm for the minimum covariance determinant estimator](https://wis.kuleuven.be/stat/robust/papers/1999/rousseeuwvandriessen-fastalgorithmformcd-technomet.pdf).
  *Technometrics*, 41(3), 212–223.
- [协方差概览与通用行为](covariance.md)。
- [完整协方差 API](../reference/covariance-api.md)。
