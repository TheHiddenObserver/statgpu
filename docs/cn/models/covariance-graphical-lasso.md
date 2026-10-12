# GraphicalLasso：用稀疏精度矩阵理解条件关联

> 语言：中文
> 最后更新：2026-10-12
> 切换：[English](../../en/models/covariance-graphical-lasso.md)

## 考虑其他变量后，哪些关联仍然存在？

两个测量值一起变化，可能只是因为它们都与第三个变量相关。普通协方差描述
边际关联；协方差的逆矩阵，即精度矩阵，则有助于描述在给定其他已测量变量后
仍然存在的关联。

`GraphicalLasso` 对精度矩阵的非对角元素施加 L1 惩罚，让其中一些元素变为零，
从而得到较简洁的无向图。在多元高斯模型中，一个非对角元素为零，表示对应的
两个变量在给定其余变量后条件独立。拟合得到的零元素描述的是所估计模型中的
关系，不能据此判断因果方向。

[协方差概览](covariance.md) 介绍基本概念和方法选择；输入、距离、评分、全部
公共参数和已拟合属性见[完整 API 参考](../reference/covariance-api.md)，
设备设置见[设备与数据类型](../reference/covariance-api.md#device-and-dtype)。

## 什么时候适合稀疏假设？

如果少量条件关联就能合理描述变量关系，或者需要对精度矩阵估计进行正则化，
可以考虑 Graphical Lasso。精度矩阵稀疏不意味着协方差矩阵也稀疏：稀疏矩阵的
逆矩阵通常仍是稠密的。

- 样本数明显多于特征数时，可用 [EmpiricalCovariance](covariance-empirical.md)
  作为无惩罚基线。
- 如果主要目标是获得稳定的协方差，而不需要稀疏图，可选
  [LedoitWolf](covariance-ledoit-wolf.md) 或 [OAS](covariance-oas.md)。
- 如果需要根据留出数据的高斯似然选择惩罚强度，可用
  [GraphicalLassoCV](covariance-graphical-lasso-cv.md)。
- 如果主要问题是异常观测污染，可考虑 [MinCovDet](covariance-min-cov-det.md)。
  L1 正则化本身不会使 Graphical Lasso 对异常行具有稳健性。

## 优化目标是什么？

设训练数据有 $n$ 行，$S$ 是以 $n$ 为分母计算的经验协方差。除非设置
`assume_centered=True`，计算前会减去训练数据的列均值。在正定精度矩阵中
求解以下优化问题：

$$
\widehat\Theta
=\arg\min_{\Theta\succ0}
\left\{\operatorname{tr}(S\Theta)-\log\det\Theta
+\alpha\sum_{j\ne k}|\Theta_{jk}|\right\}.
$$

前两项衡量高斯协方差模型的拟合，最后一项按非对角元素的绝对值总和施加惩罚；
对角元素不受 L1 惩罚。增大 `alpha` 通常会使估计更稀疏。这里的 `alpha` 是
惩罚强度，不是显著性水平。当矩阵非奇异时，协方差估计为
$\widehat\Sigma=\widehat\Theta^{-1}$。

对两个不同特征 $j,k$，偏相关系数可以写成

$$
\widehat\rho_{jk\mid\mathrm{others}}
=-\frac{\widehat\Theta_{jk}}
{\sqrt{\widehat\Theta_{jj}\widehat\Theta_{kk}}}.
$$

这些关系帮助解释图中的边，但不会自动给出边的 p 值或置信区间。
统计定义和算法背景可见 [Graphical Lasso 原论文](https://hastie.su.domains/Papers/graph.pdf)。

## 在 CPU 上拟合四变量关系图

下面是一个可独立运行的示例。模拟总体的精度矩阵是链状的，共有三条边。
前 300 行用于训练，后 100 行用于评价。生成的特征处于相近尺度，因此没有
额外做标准化。

<!-- example: covariance-graphical-lasso -->
导入 NumPy 和估计器。

```python
import numpy as np
from statgpu.covariance import GraphicalLasso
```

生成链状关系的数据，并留出测试行。每行是一条观测，每列是一个特征。

```python
rng = np.random.default_rng(12)
true_precision = np.array([
    [1.0, -0.5, 0.0, 0.0],
    [-0.5, 1.0, -0.4, 0.0],
    [0.0, -0.4, 1.0, -0.3],
    [0.0, 0.0, -0.3, 1.0],
])
X = rng.multivariate_normal(
    np.zeros(4), np.linalg.inv(true_precision), size=400
)
X_train, X_test = X[:300], X[300:]
```

使用固定惩罚，仅拟合训练行。

```python
model = GraphicalLasso(
    alpha=0.1, max_iter=200, tol=1e-6, device="cpu"
).fit(X_train)
```

查看精度矩阵、按数值阈值统计边数，并评价尚未参与拟合的测试行。

```python
print(np.round(model.precision_, 3))
edge_count = np.count_nonzero(
    np.triu(np.abs(model.precision_) > 1e-4, k=1)
)
print("edges:", edge_count)
print("iterations:", model.n_iter_)
print("test score:", round(model.score(X_test), 3))
print("squared distances:", np.round(model.mahalanobis(X_test[:3]), 3))
```
<!-- example-end: covariance-graphical-lasso -->

输出保留三位小数：

```text
[[ 0.929 -0.34  -0.074 -0.   ]
 [-0.34   0.851 -0.219  0.   ]
 [-0.074 -0.219  0.915 -0.242]
 [-0.     0.    -0.242  0.872]]
edges: 4
iterations: 2
test score: -6.036
squared distances: [2.358 1.085 6.911]
```

拟合图有四条边，比模拟总体多了特征 0 与特征 2 之间的一条边。有限样本和某个
给定惩罚强度并不能保证恢复真实关系图。这里的 `1e-4` 只是明确指定的数值展示
阈值，不是统计检验；四舍五入后出现的负零也没有统计含义。

`score(X_test)` 是测试集上按行平均的高斯对数似然。在相同数据、相同特征尺度上比较
模型时，分数越高越好，但它不是准确率。`mahalanobis` 和 `predict` 都返回
马氏距离的平方，不返回类别。第三行的距离大于前两行，说明它在这个度量下离
拟合中心更远；仅凭距离不能认定该行有误。

## 如何选参数、检查拟合？

- 默认 `alpha=0.01`，其作用取决于变量单位。若需要标准化，只在训练数据上
  估计均值和尺度，再原样应用到评价数据。应先移除常量列。
- `alpha=0` 使用经验协方差和伪逆，不再是稀疏正则化估计；伪逆也不会为
  秩亏方向补充统计信息。
- 默认 `max_iter=100`、`tol=1e-4` 控制迭代。更小的容差可能增加计算量。
  可查看 `n_iter_`，并比较增加迭代次数或收紧容差后结果是否稳定。
- 当前实现在达到 `max_iter` 后返回最后的估计，不发出未收敛警告。
  成功返回或迭代次数很少，都不等于所有数值条件、最优性条件已经满足。
- 对近共线数据或很小的惩罚，应额外检查输出是否有限、协方差是否正定。
  对当前求解器而言，`alpha` 为正并不是所有输入上的数值保证。

精确求逆失败时可能采用数值稳定化，详见 API 参考中的
[求逆、距离与评分规则](../reference/covariance-api.md#fit-score-predict-and-mahalanobis)。
如果目的是科学研究中的关系图恢复，还应考察惩罚强度和抽样波动的影响。
预测似然较好与边的选择可靠，是两个不同的目标。

## 参考文献与后续阅读

- Friedman, J., Hastie, T., & Tibshirani, R. (2008).
  [Sparse inverse covariance estimation with the graphical lasso](https://hastie.su.domains/Papers/graph.pdf).
  *Biostatistics*, 9(3), 432–441.
- [用 GraphicalLassoCV 选择惩罚](covariance-graphical-lasso-cv.md)。
- [协方差概览与方法选择](covariance.md)。
- [完整协方差 API](../reference/covariance-api.md)。
