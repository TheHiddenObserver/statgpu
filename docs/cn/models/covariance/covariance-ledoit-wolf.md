# Ledoit–Wolf：让数据决定收缩强度

> 语言：中文
> 最后更新：2026-10-12
> 切换：[English](../../../en/models/covariance/covariance-ledoit-wolf.md)

## 应该多大程度相信带有噪声的协方差？

当特征数相对于观测数较大时，经验协方差容易过度贴合样本中偶然出现的关系。求逆会进一步放大估计方差很小的方向上的噪声。[固定收缩](covariance-shrunk.md) 可以缓解这一问题，却需要自行选择强度。

`LedoitWolf` 在一次拟合中根据训练数据估计收缩强度。如果你需要较稳定的稠密协方差或精度矩阵，又不想搜索收缩强度网格，可以把它作为[经验协方差](covariance-empirical.md) 的首个比较对象。[协方差概览](README.md) 还介绍了稳健和稀疏估计方法。

## 直觉：权衡估计噪声与协方差结构

估计器将经验矩阵 $S$ 与缩放单位矩阵混合：

$$
\widehat\Sigma=(1-\alpha)S+\alpha\mu I,
\qquad \mu=\operatorname{tr}(S)/p.
$$

目标矩阵保留平均方差，但假定方差相等、交叉协方差为零。$S$ 的不确定性越大，就越需要收缩；估计结构越明显地偏离目标，就越应保留 $S$。Ledoit–Wolf 根据协方差平方误差准则推导强度，并不是最大化训练数据上的高斯得分。

对中心化观测 $z_i$，令 $S=n^{-1}\sum_i z_i z_i^\top$，本实现的计算可写为

$$
\beta=\frac{\sum_i\lVert z_i\rVert_2^4-n\lVert S\rVert_F^2}{n^2},
\qquad
\delta=\lVert S\rVert_F^2-\frac{\operatorname{tr}(S)^2}{p},
\qquad
\alpha=\operatorname{clip}(\beta/\delta,0,1).
$$

Frobenius 范数的平方是所有矩阵元素的平方和。$\delta$ 衡量矩阵偏离目标的程度，$\beta$ 估计抽样波动。当 $\delta\le0$ 时，本实现取 $\alpha=1$。若 `assume_centered=True`，则直接使用 $z_i=x_i$，并将拟合中心设为零。

## 哪些情况下应比较其他估计器？

- 高斯假设合理，尤其是观测较少时，可比较 [OAS](covariance-oas.md)。应根据实际应用评估，不能预期某一种方法在所有样本中都更好。
- 如果需要预先规定强度，或按下游任务指标选强度，使用 [ShrunkCovariance](covariance-shrunk.md)。
- 数据充足时，保留 [EmpiricalCovariance](covariance-empirical.md) 作为基线。若关心污染数据或稀疏条件关系，参阅[概览](README.md) 中的稳健和稀疏选择。

## CPU 示例：在新观测上评估

我们生成 20 个相关特征，列序号相差越大，相关性越低。拟合只使用 40 行，另用 300 行独立生成的数据评估高斯模型对新观测的描述能力。

<!-- example: covariance-ledoit-wolf -->
```python
import numpy as np
from statgpu.covariance import EmpiricalCovariance, LedoitWolf

rng = np.random.default_rng(42)
p = 20
population = 0.6 ** np.abs(np.arange(p)[:, None] - np.arange(p)[None, :])
L = np.linalg.cholesky(population)
X_train = rng.normal(size=(40, p)) @ L.T
X_test = rng.normal(size=(300, p)) @ L.T
```

仅用训练观测拟合两个模型。

```python
empirical = EmpiricalCovariance(device="cpu").fit(X_train)
model = LedoitWolf(device="cpu").fit(X_train)
```

查看估计得到的强度，并比较条件数与留出数据上的得分。

```python
print(f"shrinkage: {model.shrinkage_:.3f}")
print(f"condition numbers: {np.linalg.cond(empirical.covariance_):.1f}"
      f" -> {np.linalg.cond(model.covariance_):.1f}")
print(f"held-out scores: empirical={empirical.score(X_test):.3f},"
      f" Ledoit-Wolf={model.score(X_test):.3f}")
```
<!-- example-end: covariance-ledoit-wolf -->

输出：

```text
shrinkage: 0.311
condition numbers: 162.7 -> 10.3
held-out scores: empirical=-38.391, Ledoit-Wolf=-26.268
```

在这份样本中，最终估计约有 31% 的权重来自目标矩阵。条件数降低表示特征值之间的比例不再那么悬殊。较高的留出得分 `-26.268` 表示在相同测试观测上的平均高斯对数似然更高，并不是 26% 的准确率。本例的收缩也改善了样本外表现，但一次模拟不能保证在其他结构或数据上仍然如此。

## 仍然需要你选择什么？

`LedoitWolf` 没有供用户设定的 `shrinkage` 参数；拟合后查看 `shrinkage_` 即可。你仍然需要选择特征、单位、预处理、中心化方式和评估指标。默认行为会估计训练均值。除非有意把零当作已知中心，否则不要仅因为总体均值大致为零就设置 `assume_centered=True`。

共同方差目标依赖特征单位。如果标准化适合你的问题，就在训练数据上拟合变换，并在新观测上复用。如果根据留出得分选择多个估计器之一，这些观测就属于验证集，还需要另外的数据做最终评估。

## 限制

- 这是向缩放单位矩阵进行的线性收缩，并不能估计任意指定的结构目标。真实的相关性和不相等的方差也可能被削弱。
- 它不依赖 OAS 特有的高斯推导，但不具备抗离群点能力。极端观测可能明显影响四阶矩计算。
- 收缩不会主动把精度矩阵中的元素压到精确的零，也不提供系数的 p 值。较大的 `shrinkage_` 不是“总体协方差等于目标”的显著性检验。
- 只有目标方差为正时，正收缩强度才给出正特征值。完全没有变异的数据仍然退化。拟合至少需要两个观测、一个特征，且所有输入值都必须为有限数。

## API 与参考资料

[完整协方差 API](../../reference/covariance-api.md) 列出参数、方法、属性及共同的输入规则。设备设置见[设备与数据类型](../../reference/covariance-api.md#device-and-dtype)，得分、距离和精度矩阵的行为见[共同方法说明](../../reference/covariance-api.md#fit-score-predict-and-mahalanobis)。

- Ledoit, O., & Wolf, M. (2004).
  [A well-conditioned estimator for large-dimensional covariance matrices](https://doi.org/10.1016/S0047-259X(03)00096-4).
