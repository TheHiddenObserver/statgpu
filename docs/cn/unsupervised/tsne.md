# TSNE

> 语言：中文
> 最后更新：2026-10-06
> 切换：[English](../../en/unsupervised/tsne.md)
> 路径：`statgpu.unsupervised.TSNE`

## 概览

`TSNE` 通过匹配高维高斯亲和度与低维 Student-t 亲和度来学习嵌入（embedding）。实现稠密欧氏距离上的精确 t-SNE。

## 何时使用

t-SNE 适合可视化局部相似关系，不是用于预测的新特征映射。解释岛状结构前，应比较多个困惑度和随机种子。全局距离、坐标轴及区域面积都不能直接量化人群差异。

## 导入路径

从 `statgpu.unsupervised` 导入：

```python
from statgpu.unsupervised import TSNE
```

## 目标函数

t-SNE 最小化 KL 散度：

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

嵌入形状为 `(40, 2)`。KL 散度描述训练亲和度的拟合效果，不是留出数据的预测评分。即使成对亲和度精确计算，后续优化仍是非凸的，并不保证全局最优嵌入。

安装了相应 GPU 后端后，可新建估计器并指定 `device="cuda"`（CuPy）或 `device="torch"`（Torch CUDA）。数组通常留在该后端；输出所在设备及主机端步骤见 [API 参考](api-reference.md#tsne)。显式请求的 GPU 不可用时会报错。

## 近似与解释边界

这里实现的是稠密数据上的精确 t-SNE；Barnes-Hut、FFT/FIt-SNE 与 openTSNE 等加速路径只作为外部对齐基线。

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

- van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. *Journal of Machine Learning Research*, 9, 2579-2605.
- Linderman, G. C., Rachh, M., Hoskins, J. G., Steinerberger, S., & Kluger, Y. (2019). Fast interpolation-based t-SNE for improved visualization of single-cell RNA-seq data. *Nature Methods*, 16, 243-245.
