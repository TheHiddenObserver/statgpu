# GraphicalLassoCV：用留出数据选择稀疏模型的惩罚

> 语言：中文
> 最后更新：2026-10-12
> 切换：[English](../../../en/models/covariance/covariance-graphical-lasso-cv.md)

## 应该让关系图稀疏到什么程度？

Graphical Lasso 的惩罚过小，可能保留噪声造成的关联；惩罚过大，又可能抹去
有用的依赖关系。`GraphicalLassoCV` 将输入数据随机分为若干折，轮流留出一折
做验证、用其余折拟合，以此比较各候选惩罚的平均验证分数。选出分数最高的惩罚后，
再用全部输入数据重新拟合。它是同一个
[Graphical Lasso 模型](covariance-graphical-lasso.md)的参数选择过程，
不是另一种协方差模型。

如果观测适合随机分折，且希望按留出数据的平均高斯对数似然选择惩罚，就可以
使用这个估计器。如果惩罚已经由研究方案或外部依据确定，直接使用
`GraphicalLasso` 即可。若只想稳定估计协方差而不需要稀疏图，可考虑
[LedoitWolf](covariance-ledoit-wolf.md) 或 [OAS](covariance-oas.md)。
方法选择见[协方差概览](README.md)，输入规则和完整接口见
[API 参考](../../reference/covariance-api.md)，设备设置见
[设备与数据类型](../../reference/covariance-api.md#device-and-dtype)。

## 交叉验证在最大化什么？

在每个训练折内，$S$ 是以该折训练行数为分母计算的经验协方差；除非设置
`assume_centered=True`，计算前会减去该折训练数据的列均值。优化变量 $\Theta$
是正定精度矩阵，即逆协方差。对每个候选 $\alpha$，拟合的目标与 Graphical Lasso 相同：

$$
\operatorname{tr}(S\Theta)-\log\det\Theta
+\alpha\sum_{j\ne k}|\Theta_{jk}|,
\qquad \Theta\succ0.
$$

前两项衡量高斯拟合；L1 项只惩罚 $\Theta$ 的非对角元素，鼓励稀疏的条件关联，
对角元素不受惩罚。这里的 $\alpha$ 控制惩罚强度。

设有 $p$ 个特征、$K$ 个验证折。对验证折 $V_k$，使用其余折拟合得到的中心
$\widehat\mu_k$、协方差 $\widehat\Sigma_k$ 和精度矩阵 $\widehat\Theta_k$，
计算平均高斯对数似然：

$$
s_k(\alpha)=-\frac12\left[
 p\log(2\pi)+\log\det\widehat\Sigma_k
 +\frac{1}{|V_k|}\sum_{i\in V_k}
 (x_i-\widehat\mu_k)^\top\widehat\Theta_k(x_i-\widehat\mu_k)
\right].
$$

选择使 $K^{-1}\sum_{k=1}^K s_k(\alpha)$ 最大的惩罚，即对各折分数取等权平均，
越大越好。验证分数不包含 L1 惩罚项，它衡量的是模型对未见数据的高斯拟合。
最终的 `covariance_`、`precision_` 和 `location_` 来自选定惩罚在全部输入
数据上的重拟合，不是各折矩阵的平均。

## 在 CPU 上比较五个惩罚值

下面的独立示例生成六变量高斯链，并保留 60 行完全不参与参数选择。这样，
最终测试分数与选择 `alpha` 时使用的分数来自不同数据。

<!-- example: covariance-graphical-lasso-cv -->
导入本例所需的工具。

```python
import numpy as np
from statgpu.covariance import GraphicalLassoCV
```

生成数据，并保留独立测试集。

```python
rng = np.random.default_rng(23)
true_precision = np.eye(6)
for j in range(5):
    true_precision[j, j + 1] = -0.3
    true_precision[j + 1, j] = -0.3
X = rng.multivariate_normal(
    np.zeros(6), np.linalg.inv(true_precision), size=240
)
X_train, X_test = X[:180], X[180:]
```

在训练数据内部做四折交叉验证，选择惩罚后自动进行最终重拟合。

```python
model = GraphicalLassoCV(
    alphas=[0.01, 0.03, 0.1, 0.3, 1.0],
    cv=4,
    random_state=7,
    max_iter=200,
    tol=1e-6,
    device="cpu",
).fit(X_train)
```

比较候选分数，并查看最终模型在独立测试集上的分数。

```python
for result in model.cv_results_:
    print(f"alpha={result['alpha']:.2f}, mean score={result['mean_score']:.3f}")
print("selected alpha:", model.alpha_)
print("refit rows:", model.n_samples_)
print("test score:", round(model.score(X_test), 3))
```
<!-- example-end: covariance-graphical-lasso-cv -->

分数保留三位小数，输出为：

```text
alpha=0.01, mean score=-8.829
alpha=0.03, mean score=-8.828
alpha=0.10, mean score=-8.833
alpha=0.30, mean score=-8.940
alpha=1.00, mean score=-9.037
selected alpha: 0.03
refit rows: 180
test score: -8.782
```

`0.03` 对应的平均分数最高，但与 `0.01` 几乎持平。因此，应查看各折的分数
波动及结果对分折方式的敏感性，不宜把获选值当作精确确定的常数。最终模型使用
全部 180 行训练数据；`-8.782` 是另外 60 行上的平均高斯对数似然，既不是准确率，
也不能证明选出的图就是真实关系图。

`cv_results_` 是字典列表，每个候选对应一个字典，包含 `alpha`、`mean_score`
和 `scores`；其中 `scores` 保存各折分数。它不是 scikit-learn 常见的
“字典中保存各列数组”的结果格式。

## 明确选择调参网格和分折方式

- 整数 `alphas=m` 恰好生成 `np.logspace(-2, 0, m)`。当 `m > 1` 时，
  这是从 `0.01` 到 `1.0` 的固定对数网格；当 `m == 1` 时只有 `[0.01]`。
  它不会根据数据自适应确定范围，也不会逐轮细化。还可以传入非空的显式序列，
  每个惩罚必须有限且非负，允许包含 `0`。
- 若最优值位于边界，可以向该方向扩展网格，同时检查数值表现。惩罚的作用
  依赖特征尺度，默认范围不适用于所有数据。
- `cv` 必须是从 2 到样本行数之间的整数。实现先随机打乱行，再分成指定折数；
  `random_state` 用于复现分折。每个训练折仍须至少有两行，才能拟合 Graphical Lasso。
- 不接受自定义分折对象、分组折或时间序列折。对于重复测量、有关联的研究对象
  或时间相关数据，应自行构造适当的训练集和验证集，再逐次拟合 `GraphicalLasso`。
- 若标准化、插补等变换需要从数据中估计参数，就应在每个训练折内分别估计。
  `GraphicalLassoCV` 没有折内预处理接口；需要在每折内完成这些变换时，应使用
  `GraphicalLasso` 编写手动验证循环，避免验证数据参与预处理参数的估计。
- `n_jobs` 为保留参数，增大它不会在这里启用并行交叉验证。候选和折数越多，
  需要的拟合次数越多，选择后还要再做一次最终重拟合。

## 限制与诊断

该类与 `GraphicalLasso` 有相同的数值限制。惩罚较小、协方差病态时，拟合可能较困难。
`max_iter` 和 `tol` 作用于每一折及最终拟合；达到迭代次数上限目前不会发出
未收敛警告。`GraphicalLassoCV` **不提供 `n_iter_`**。若需要查看迭代次数，
可以在相同数据和设置下单独拟合 `GraphicalLasso(alpha=model.alpha_, ...)`；
这个检查只针对该次直接拟合，不能替代对之前各折的检查。

采用所选惩罚前，应检查所有候选的各折分数是否有限。完成参数选择并不保证每次
拟合的数值结果都可靠。报告调参后的性能时，应保留独立测试集，或采用合适的
外层验证方案。
`predict` 和 `mahalanobis` 仍返回距离平方，不会给出图标签或异常类别。

## 参考文献与后续阅读

- Friedman, J., Hastie, T., & Tibshirani, R. (2008).
  [Sparse inverse covariance estimation with the graphical lasso](https://hastie.su.domains/Papers/graph.pdf).
  *Biostatistics*, 9(3), 432–441.
- [GraphicalLasso：解释精度矩阵与偏相关](covariance-graphical-lasso.md)。
- [协方差概览与方法选择](README.md)。
- [完整协方差 API](../../reference/covariance-api.md)。
