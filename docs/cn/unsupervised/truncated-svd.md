# TruncatedSVD

> 语言：中文
> 最后更新：2026-10-14
> 切换：[English](../../en/unsupervised/truncated-svd.md)
> 路径：`statgpu.unsupervised.TruncatedSVD`

## 概览

在文档—词项矩阵中，每一行是一篇文档，每一列记录一个词的权重。你希望用更少的坐标概括词语共同出现的模式，同时近似原来的矩阵。只保留几个常见词会丢掉它们之间的联合结构；把每行求和又只剩一个总量。如果减去每个词的平均权重，问题则变成了“这篇文档与平均文档有何不同”。

TruncatedSVD 学习几个共享模式，每个模式都为各个词分配一个权重。用一篇文档自身的坐标缩放这些模式，再把它们相加，就能近似还原这篇文档的词权重。因此，每个坐标概括的是多个词的组合，而不只是某一个词。

这里不先减去每列的平均词权重。因此，如果所有文档都大量使用某些相同的词，第一个成分可能主要反映这种共同用词，而不是文档之间的差别。一个成分也不一定对应可以命名的主题或独立的真实来源。

正式地说，这些模式是保留的奇异方向，给出未经中心化的矩阵的低秩近似。用于文档—词项数据时，这也是潜在语义分析（LSA）的基本思路。同样的运算也适用于其他数值矩阵，并不要求输入是文本或非负数。

## 何时使用

希望近似未经中心化的原矩阵时，可使用 TruncatedSVD；关注相对各特征均值的变化时，应选择 [PCA](pca.md)，因为中心化后再做 SVD 就得到 PCA 问题。如果模式及其贡献都必须非负，可考虑 [NMF](nmf.md)；TruncatedSVD 允许正负系数。

statgpu 的当前实现只接受稠密输入。LSA 的应用背景不代表它支持稀疏文档—词项矩阵；如果稠密矩阵难以容纳，应选择支持稀疏输入的实现。可结合重构误差和降维坐标的实际用途选择秩，不要仅凭成分顺序就给它们赋予主题含义。

## 导入路径

从 `statgpu.unsupervised` 导入：

```python
from statgpu.unsupervised import TruncatedSVD
```

当前估计器支持 CPU、CuPy/CUDA 与 Torch CUDA 上的稠密输入。

## 目标函数

给定秩 `k`，Truncated SVD 求解：

$$
\min_{\operatorname{rank}(Z) \le k} \|X - Z\|_F^2.
$$

返回的成分是 `X` 的前几个右奇异向量。

## 估计方程

精确路径计算：

$$
X = U \Sigma V^\top.
$$

随机化路径先把 `X` 投影到随机的低维子空间，每轮幂迭代后重新正交化，再对投影后的小矩阵做 SVD，最后按确定性约定统一成分的符号。

## 参数

`n_components`、`algorithm`、`n_iter`、`n_oversamples`、`random_state`、`device`。

## 一个可独立运行的 CPU 示例

<!-- learner-example: truncated-svd -->
```python
import numpy as np
from statgpu.unsupervised import TruncatedSVD

rng = np.random.default_rng(0)
X = rng.normal(size=(60, 5)) + 2.0
model = TruncatedSVD(n_components=2, algorithm="full", device="cpu")
Z = model.fit_transform(X)
X_hat = model.inverse_transform(Z)
print(Z.shape, X_hat.shape, model.explained_variance_ratio_.sum())
```

60 个输入行各有 5 个测量值；每个输出行有两个数，表示该行在两个已学模式上的坐标，因此 `Z` 为 `(60, 2)`。若输入是文档—词项数据，就是每篇文档对应两个坐标。这里 `Z = X @ model.components_.T`，`X_hat = Z @ model.components_`，两步都不减去或加回均值。解释方差是投影坐标中心化后的方差（除以 `n`），尽管拟合本身不中心化；它不等同于保留奇异值平方和占原矩阵平方范数的比例。

安装了相应 GPU 后端后，可新建估计器并指定 `device="cuda"`（CuPy）或 `device="torch"`（Torch CUDA）。数组通常留在该后端；输出所在设备及主机端步骤见 [API 参考](api-reference.md#truncatedsvd)。显式请求的 GPU 不可用时会报错。

## 近似与解释边界

`algorithm="full"` 是稠密数据上的精确 SVD；`algorithm="randomized"` 是近似算法，比较时应使用对符号和子空间不变的指标。

## 输出

`components_`、`explained_variance_`、`explained_variance_ratio_`、`singular_values_`、`n_components_`、`n_features_in_`。

## FAQ

不支持稀疏输入和 ARPACK。


## 完整 API 参考

构造默认值、全部公开方法、输出形状与限制见 [TruncatedSVD API 参考](api-reference.md#truncatedsvd)。

<a id="references"></a>

## 参考文献

- Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. *SIAM Review*, 53(2), 217-288. https://doi.org/10.1137/090771806
- scikit-learn Developers. [TruncatedSVD：非中心化投影与 LSA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html)。
