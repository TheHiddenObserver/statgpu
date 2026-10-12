# OAS：利用高斯假设进行小样本收缩

> 语言：中文
> 最后更新：2026-10-12
> 切换：[English](../../en/models/covariance-oas.md)

## 观测较少时，高斯假设能提供帮助吗？

特征多、观测少时，即使每个观测都有效，经验协方差仍然可能包含较多噪声。`OAS`（Oracle Approximating Shrinkage，近似预言收缩）利用基于高斯分布的推导，估计应多大程度把协方差拉向缩放单位矩阵。如果观测近似高斯的假设合理，且样本量相对维度较小，它值得比较。

这里的“预言”指一种需要知道未知总体协方差才能使用的理想收缩规则。OAS 用样本中的统计量来近似这个规则，并不知晓真实协方差，也不保证对你的数据最优。[协方差概览](covariance.md) 介绍更广泛的方法选择。

## 估计形式与实际采用的公式

与 [Ledoit–Wolf](covariance-ledoit-wolf.md) 一样，输出为

$$
\widehat\Sigma=(1-\alpha)S+\alpha\mu I,
\qquad S=\frac{1}{n}\sum_{i=1}^n z_i z_i^\top,
\qquad \mu=\frac{\operatorname{tr}(S)}{p}.
$$

默认使用 $z_i=x_i-\bar x$。设置 `assume_centered=True` 时，使用 $z_i=x_i$，并把中心固定为零。记矩阵元素平方的平均值为 $q=\lVert S\rVert_F^2/p^2$，本实现计算

$$
\alpha=\operatorname{clip}\left(
\frac{q+\mu^2}{(n+1)(q-\mu^2/p)},0,1\right).
$$

分母非正时取 $\alpha=1$。

这是 **scikit-learn 采用的简化 OAS 公式**，省略了 Chen 等人（2010）论文公式 (23) 中的 $2/p$ 修正项。因此，直接代入论文中的完整公式，不一定能复现 `shrinkage_`，尤其在 $p$ 较小时。[scikit-learn 的 OAS 说明](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.OAS.html#notes) 也明确区分了两种公式。

## 应与哪些方法比较？

- [Ledoit–Wolf](covariance-ledoit-wolf.md) 不依赖 OAS 特有的高斯推导，是自动选择强度的实用基线。
- [ShrunkCovariance](covariance-shrunk.md) 允许直接控制强度，也方便根据下游任务的验证指标选择强度。
- 数据较充足时，[EmpiricalCovariance](covariance-empirical.md) 可展示未经收缩的结果。

OAS 处理的是抽样噪声，不是针对污染数据的稳健估计器；它也不会主动构造稀疏精度矩阵。

## CPU 示例：比较两种自动选择

下面的完整示例使用 25 个训练观测和 20 个相关高斯特征。它既核对简化公式，也在 300 个新观测上比较 OAS 与 Ledoit–Wolf。测试数据不参与模型拟合，也不参与收缩强度的计算。

<!-- example: covariance-oas -->
```python
import numpy as np
from statgpu.covariance import LedoitWolf, OAS

rng = np.random.default_rng(42)
p = 20
population = 0.6 ** np.abs(np.arange(p)[:, None] - np.arange(p)[None, :])
L = np.linalg.cholesky(population)
X_train = rng.normal(size=(25, p)) @ L.T
X_test = rng.normal(size=(300, p)) @ L.T
```

在训练样本上拟合两种自动估计器。

```python
oas = OAS(device="cpu").fit(X_train)
lw = LedoitWolf(device="cpu").fit(X_train)
```

从经验协方差重新计算简化 OAS 公式中的强度。

```python
S = np.cov(X_train, rowvar=False, ddof=0)
mu = np.trace(S) / p
q = np.mean(S * S)
denominator = (len(X_train) + 1) * (q - mu * mu / p)
alpha = (1.0 if denominator <= 0 else
         np.clip((q + mu * mu) / denominator, 0.0, 1.0))
```

核对公式，并在新观测上比较拟合结果。

```python
print(f"shrinkage: OAS={oas.shrinkage_:.3f}, Ledoit-Wolf={lw.shrinkage_:.3f}")
print("formula agrees:", np.isclose(oas.shrinkage_, alpha))
print(f"held-out scores: OAS={oas.score(X_test):.3f},"
      f" Ledoit-Wolf={lw.score(X_test):.3f}")
print(f"smallest OAS eigenvalue: {np.linalg.eigvalsh(oas.covariance_)[0]:.3f}")
```
<!-- example-end: covariance-oas -->

输出：

```text
shrinkage: OAS=0.448, Ledoit-Wolf=0.400
formula agrees: True
held-out scores: OAS=-26.540, Ledoit-Wolf=-26.557
smallest OAS eigenvalue: 0.441
```

OAS 在这份样本中采用了略强的收缩。核对公式时，`ddof=0` 保持了以 $n$ 为分母的约定；NumPy 默认的 $n-1$ 分母对应另一种协方差尺度。最小特征值为正，说明本例收缩后的协方差不存在零方差方向，可以求逆。

两个得分都是平均高斯对数似然；在相同观测和尺度上，越高越好。OAS 在本例中只稍好一点，不能据此断言它总会胜过 Ledoit–Wolf。若根据这些得分选择方法，还应保留独立的最终测试集。

## 参数、假设与限制

OAS 没有用户指定的收缩强度，也不会搜索交叉验证网格。`shrinkage_` 是输出，`assume_centered` 则控制是否估计训练均值。除非零是预期中心，否则保留默认的 `False`。预处理与特征选择也应只使用训练数据来确定。

目标矩阵在所选单位下假设各特征方差相等。标准化会改变这一目标，因而改变统计问题；如果选择标准化，请复用训练数据确定的尺度。明显偏离高斯的重尾、离群点，或与目标相差很大的协方差结构，都可能使 OAS 的假设不适合当前任务。

强度为 `1` 只表示估计完全采用 $\mu I$，不是数据证明了独立性。只有 $\mu>0$ 时，正收缩强度才保证特征值为正；完全没有变异的数据仍然退化。拟合至少需要两个有限值观测和一个特征。稳健与稀疏替代方法以及共同的求逆行为见[概览](covariance.md)。

## API 与参考资料

[完整协方差 API](../reference/covariance-api.md) 说明全部参数、属性、方法、输入要求和设备行为。

- Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O. (2010).
  [Shrinkage algorithms for MMSE covariance estimation](https://arxiv.org/abs/0907.4698).
- [scikit-learn：OAS](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.OAS.html)，特别是与原论文公式的差异说明。
