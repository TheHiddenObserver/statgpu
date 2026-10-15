# TruncatedSVD

> 语言：中文
> 最后更新：2026-10-14
> 切换：[English](../../en/unsupervised/truncated-svd.md)
> 路径：`statgpu.unsupervised.TruncatedSVD`

## 概览

TruncatedSVD 用指定数量的共享模式近似一个数值矩阵，不先减去各列均值。每个模式为所有输入特征分配权重；每行数据得到一组坐标，用来缩放这些模式。将缩放后的模式相加，就能近似重构该行。模式与坐标都可以包含正数或负数。

算法保留最强的奇异方向，也就是在原矩阵中包含最多平方数值总量的方向。只保留其中几个，便得到低秩近似，以及每行更短的坐标向量。由于拟合不做中心化，多数行共同具有的较大模式可能主导结果，即使它对行与行之间差异的解释很少。一个成分不一定代表独立的真实来源。

例如，在文档—词项矩阵中，每一行是一篇文档，每一列记录一个词的权重。只保留几个常见词会丢掉它们之间的联合结构；把每行求和又只剩一个总量。TruncatedSVD 概括的是共同出现的词语组合。这里不先减去每列的平均词权重，因此第一个成分可能主要反映共同用词。一个成分不一定对应可以命名的主题。这种文本应用称为潜在语义分析（LSA），但方法本身不要求输入为文本或非负数。

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
