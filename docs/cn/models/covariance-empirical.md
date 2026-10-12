# 经验协方差：理解未经正则化的基线

> 语言：中文
> 最后更新：2026-10-12
> 切换：[English](../../en/models/covariance-empirical.md)

## 多个变量如何一起变化？

假设每一行都是同一次观测中的多项测量。你想知道各个变量的波动大小，以及它们是否一起升高或降低，并不需要预测某个目标变量。`EmpiricalCovariance` 直接从样本估计这些关系。当观测数远多于特征数、没有近乎重复的列，也没有影响很大的离群点时，它是一个直观的基线。

[协方差概览](covariance.md) 帮助你选择估计方法。本页重点解释经验估计的含义，以及如何读懂它给出的距离。

## 直觉与公式

方差描述单个变量的分散程度；协方差描述两个变量相对于各自均值的偏离是否同向。对包含 $n$ 个观测、$p$ 个特征的数据，默认估计为

$$
\bar x=\frac{1}{n}\sum_{i=1}^n x_i,
\qquad
S=\frac{1}{n}\sum_{i=1}^n(x_i-\bar x)(x_i-\bar x)^\top.
$$

分母是 **$n$，不是 $n-1$**。这是高斯模型的极大似然协方差估计，并非常见的无偏样本协方差。它与 `np.cov(X, rowvar=False, ddof=0)` 一致。计算协方差本身不要求数据服从高斯分布；但 `score` 使用的是高斯模型的对数似然，解释结果时要考虑这一模型假设是否合理。

`covariance_` 保存 $S$：对角线是各特征的方差，非对角线是特征之间的协方差。协方差的单位是两个特征单位的乘积，因此数值大于一并不意味着结果出错，它不是相关系数。`location_` 保存 $\bar x$。设置 `assume_centered=True` 时，估计器将中心固定为零，不再减去样本均值，而是直接计算 $X^\top X/n$。

## 何时使用，何时换一种方法

如果你需要容易解释的基线、描述性协方差，或样本已经充足、不希望额外引入正则化偏差，可以使用经验估计。

- 特征较多或很小的特征值使求逆不稳定时，比较 [Ledoit–Wolf](covariance-ledoit-wolf.md) 与 [OAS](covariance-oas.md)。
- 希望自行控制正则化强度时，使用[固定收缩](covariance-shrunk.md)。
- 样本受到离群点污染时，可考虑 [MinCovDet](covariance-min-cov-det.md)，不要期待经验协方差具有抗离群点能力。

## CPU 示例：估计波动并计算距离

下面的完整示例生成 400 行、两个相关特征。总体均值为 `[10, -3]`，总体协方差为 `[[4, 1.2], [1.2, 1]]`。不需要响应变量，也不需要手动中心化。

<!-- example: covariance-empirical -->
```python
import numpy as np
from statgpu.covariance import EmpiricalCovariance

np.set_printoptions(precision=3, suppress=True)
rng = np.random.default_rng(42)
transform = np.array([[2.0, 0.0], [0.6, 0.8]])
X = rng.normal(size=(400, 2)) @ transform.T + [10.0, -3.0]
```

用这些观测拟合协方差和中心。

```python
model = EmpiricalCovariance(device="cpu").fit(X)
```

查看估计矩阵、核对分母约定，再计算两个新点的距离。

```python
print("location:", np.round(model.location_, 3))
print("covariance:\n", np.round(model.covariance_, 3))
print("matches divisor n:", np.allclose(
    model.covariance_, np.cov(X, rowvar=False, ddof=0)
))
queries = np.array([[10.0, -3.0], [14.0, -3.0]])
print("squared distances:", np.round(model.mahalanobis(queries), 3))
```
<!-- example-end: covariance-empirical -->

输出：

```text
location: [ 9.96  -3.042]
covariance:
 [[4.027 1.454]
 [1.454 1.104]]
matches divisor n: True
squared distances: [0.002 7.522]
```

估计均值和方差接近总体值，但有限样本得到的协方差仍有抽样误差。正的交叉协方差表示两个特征倾向于一起增大。

`mahalanobis` 返回的是**平方**距离，
$d^2(x)=(x-\bar x)^\top\texttt{precision\_}(x-\bar x)$。
第一个查询点接近估计中心。第二个点的第一个特征偏高，而第二个特征没有相应升高，因此在这个相关结构下更不典型。`7.522` 既不是概率，也不是自动校准的离群点标签。`predict` 返回相同的平方距离，并不预测特征值。

## 使用选择与常见误区

- 除非零是你有意采用的已知中心，或预处理已经完成中心化，否则保留 `assume_centered=False`。样本均值较小不代表可以忽略中心化。对新数据沿用训练数据确定的预处理，不要单独重新中心化评估数据。
- 本方法没有收缩强度可调。在下游使用精度矩阵前，应检查条件数以及留出数据上的表现。
- 估计均值后，协方差的秩最多为 $\min(p,n-1)$。因此 $p\ge n$ 时必然秩亏；即使观测数更多，共线性也可能导致秩亏。求逆时的数值稳定化只是计算保障，不会改变 `covariance_`，也不会凭空补充缺失的信息。
- `score(X_test)` 是每个观测的平均高斯对数似然。在相同留出数据和特征尺度上，越大越好。奇异协方差可能得到 `-inf`；拟合成功不等于得到了可用的高斯密度模型。
- 拟合至少需要两个观测、一个特征，且所有输入值都必须为有限数。拟合前处理缺失值，评估时保持列顺序一致。

## API 与参考资料

[完整协方差 API](../reference/covariance-api.md) 列出全部参数、拟合属性、输入约定和方法。设备设置见[设备与数据类型](../reference/covariance-api.md#device-and-dtype)，得分、距离和精度矩阵的行为见[共同方法说明](../reference/covariance-api.md#fit-score-predict-and-mahalanobis)。

- [scikit-learn：EmpiricalCovariance](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EmpiricalCovariance.html)，高斯极大似然协方差估计的参考实现。
