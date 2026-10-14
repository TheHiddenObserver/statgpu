# TSNE

> 语言：中文
> 最后更新：2026-10-13
> 切换：[English](../../en/unsupervised/tsne.md)
> 路径：`statgpu.unsupervised.TSNE`

## 概览

假设一批图像各用数百个数值特征表示，你想在没有预设类别的情况下，看看哪些图像的特征相似。只画两列特征的散点图会丢掉其余信息；[PCA](pca.md) 的前两个主成分虽然概括了主要变异方向，但线性投影仍可能把曲面上不同的邻域叠在一起。t-SNE 关注一个更局部的问题：在给定特征空间中相似的观测，能否在二维图中仍然靠近？

t-SNE 不尝试保留每一对观测的距离，而是把距离转成近邻亲和度，再在图中尽量重现这些亲和度。每个观测都有自己的邻域尺度；低维 Student-t 亲和度比高维高斯亲和度具有更重的尾部，让不相似的观测有更多空间分开，缓解拥挤。这是[原始 t-SNE 方法](https://jmlr.org/papers/v9/vandermaaten08a.html)的基本思路。statgpu 实现的是稠密欧氏距离上的精确 t-SNE，因此需要先确认：特征及其缩放方式是否让欧氏距离能表达你关心的相似性。

## 何时使用

当目标是查看一批固定观测的局部相似关系时，可以使用 t-SNE，例如从相邻点中挑选有代表性的图像查看，或追查某个观测为何远离其他点。

- 如果几个原始变量或线性概括已经能回答问题，先用原始特征散点图或 PCA。需要解释方差、重构数据，或用已拟合的映射变换后续观测时，优先考虑 PCA。
- 如果想探索选定近邻构成的图，并分别调节邻域大小和布局，可比较 [UMAP](umap.md)。t-SNE 匹配的是成对亲和度，用 `perplexity` 控制有效邻域尺度。statgpu 的两种实现都不支持新数据 `transform`。
- 如果需要簇分配，应另行选择并验证聚类方法。t-SNE 图中的一个岛状区域，不能证明存在一个独立群体，也不能替你决定簇数。

## 导入路径

从 `statgpu.unsupervised` 导入：

```python
from statgpu.unsupervised import TSNE
```

## 目标函数

用 $p_{ij}$ 表示一对观测在输入空间中的亲和度，用 $q_{ij}$ 表示它们在图中的亲和度。如果 $p_{ij}$ 很大而图中 $q_{ij}$ 很小，这一对观测会对损失产生较大贡献：拆散重要近邻的代价较高。t-SNE 最小化 KL 散度：

$$
\operatorname{KL}(P \| Q)
= \sum_{i \ne j} p_{ij}\log\frac{p_{ij}}{q_{ij}}.
$$

## 估计方程

对每个观测，高斯条件概率会给距离较近的其他观测更大权重。当 $j\ne i$ 时，

$$
p_{j\mid i}=\frac{\exp(-\beta_i\|x_i-x_j\|_2^2)}
{\sum_{\ell\ne i}\exp(-\beta_i\|x_i-x_\ell\|_2^2)},
\qquad p_{i\mid i}=0,
\qquad \beta_i=\frac{1}{2\sigma_i^2}.
$$

$$
\operatorname{Perp}_i=\exp\left(-\sum_{j\ne i}p_{j\mid i}\log p_{j\mid i}\right),
\qquad p_{ij}=\frac{p_{j\mid i}+p_{i\mid j}}{2n},
\qquad p_{ii}=0.
$$

其中 $n$ 为观测数，$\sigma_i$ 为每行单独确定的带宽。二分搜索调整 $\beta_i$，尝试达到 `perplexity` 指定的目标困惑度；对称化后得到目标函数使用的联合亲和度 $P$。不同点对的低维亲和度定义为：

$$
q_{ij} =
\frac{(1+\|y_i-y_j\|_2^2)^{-1}}
{\sum_{a \ne b}(1+\|y_a-y_b\|_2^2)^{-1}}.
$$

令 $q_{ii}=0$。带宽搜索成功时，两个联合亲和度矩阵在全部不同观测构成的有序点对上求和均为一。嵌入通过早期亲和度放大（`early_exaggeration`）、动量和逐坐标自适应增益进行优化。

困惑度表示有效邻域大小，并不是实际选取的近邻个数。分布在 `n-1` 个其他观测上的概率，其困惑度在 1 到 `n-1` 之间；并列的最近邻可能进一步抬高下限。API 的范围检查只要求 `0 < perplexity < n`，因此通过检查的设置也未必可达到。宜选择位于可行范围内部、明显小于样本量的数值；重复观测或数值校准失败都可能妨碍目标实现。应在同一数据集上比较多组设置，不要直接比较使用不同目标亲和度算出的 KL 值。

## 参数

`n_components`、`perplexity`、`early_exaggeration`、`learning_rate`、`max_iter`、`init`、`random_state`、`metric`、`device`。

## 一个可独立运行的 CPU 示例

<!-- learner-example: tsne -->
```python
import numpy as np
from statgpu.unsupervised import TSNE

rng = np.random.default_rng(0)
X = np.vstack([rng.normal(-2, 0.4, (20, 3)), rng.normal(2, 0.4, (20, 3))])
model = TSNE(perplexity=5, max_iter=300, init="random", random_state=0, device="cpu")
embedding = model.fit_transform(X)
print(embedding.shape, model.n_iter_, model.kl_divergence_)
```

嵌入形状为 `(40, 2)`，每一行对应一个输入观测；保留行标识，才能回查原始特征或图像。KL 散度描述训练亲和度的拟合效果，不是留出数据的预测评分。即使成对亲和度精确计算，后续优化仍是非凸的，并不保证全局最优嵌入。本例的两组数据是生成时刻意设置的；未知数据上看起来分开的图形，并不提供同样的证据。

安装了相应 GPU 后端后，可新建估计器并指定 `device="cuda"`（CuPy）或 `device="torch"`（Torch CUDA）。数组通常留在该后端；输出所在设备及主机端步骤见 [API 参考](api-reference.md#tsne)。显式请求的 GPU 不可用时会报错。

## 近似与解释边界

这里实现的是稠密数据上的精确 t-SNE，所需成对数组的内存随观测数平方增长。不支持 Barnes-Hut 和 FFT/FIt-SNE 加速。

把图中的现象当作需要回到原始特征中核实的线索。比较多组可达到的困惑度和随机种子，再检查图中相邻点在所选预处理下是否确实相似。坐标轴没有原始特征的单位；岛间空隙和岛的面积不能量化群体间隔或密度。[How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/) 展示了没有真实分簇的数据也可能出现团块。不同设置下的稳定性值得检查，但它既不能证明存在真实簇，也不是统计显著性检验。

## 输出

`embedding_`、`kl_divergence_`、`n_iter_`、`n_features_in_`。

## FAQ

不支持稀疏输入、非欧氏 `metric`、Barnes-Hut、FFT/FIt-SNE，也不支持对新样本调用 `transform`。


## 数值与使用注意事项

所有数值参数都必须有限；例如 `learning_rate=np.nan` 当前不一定会被拒绝，却会产生非有限嵌入。特征尺度过大或过小也可能使当前亲和度带宽搜索失败。拟合前应先以 float64 减去由训练数据确定的特征偏移，再缩放到适中的数值范围。很大的共同偏移可能破坏展开距离的计算，即使亲和度矩阵仍归一化、KL 仍非负，也不代表距离正确。应拒绝非有限嵌入或负的 `kl_divergence_`；负 KL 是无效结果，不是更好的拟合，而非负 KL 本身也不足以验证结果。

## 完整 API 参考

构造默认值、全部公开方法、输出形状与限制见 [TSNE API 参考](api-reference.md#tsne)。

<a id="references"></a>

## 参考文献

- van der Maaten, L., & Hinton, G. (2008). [Visualizing data using t-SNE](https://jmlr.org/papers/v9/vandermaaten08a.html). *Journal of Machine Learning Research*, 9, 2579-2605.
- Wattenberg, M., Viégas, F., & Johnson, I. (2016). [How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/). *Distill*.
- Linderman, G. C., Rachh, M., Hoskins, J. G., Steinerberger, S., & Kluger, Y. (2019). Fast interpolation-based t-SNE for improved visualization of single-cell RNA-seq data. *Nature Methods*, 16, 243-245.
