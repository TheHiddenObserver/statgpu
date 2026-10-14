# IncrementalPCA

> 语言：中文
> 最后更新：2026-10-14
> 切换：[English](../../en/unsupervised/incremental-pca.md)

## 概览

测量数据每天存成一个文件时，一个实际问题是：怎样让不断增加的观测共享同一套紧凑坐标？目标是用少数几个数概括一条观测，而不必保留它的全部测量特征。每个数都用一组共享权重，将各特征相对均值的偏离组合起来；这组权重定义一个方向，保留的几个方向共同构成近似表示观测所用的基。

分别对每天的数据做 PCA，会得到不同的方向和均值，各天的坐标不能直接比较。每来一批数据就把全部历史重新做一遍 PCA，虽能避免这个问题，却需要反复读取历史观测。

IncrementalPCA 保留一份摘要：已经选出的方向、这些方向上的变化强度，以及累计特征统计量。新批次与这份摘要合并，用来更新共享的低维基。旧均值与新批次均值之间的差异也必须计入：批次之间的整体位移属于合并数据的变化，不能通过分别中心化每批数据把它抹掉。

这种做法会损失信息。一旦某个方向从历史摘要中被舍弃，后续更新便无法还原过去观测在该方向上的变化。因此，它近似的是合并全部数据后的 PCA，并不保证与一次性拟合全部行得到相同结果。

## 何时使用

当逐批处理稠密观测的需求，比一次性完整分解更重要时，可选择 IncrementalPCA。数据可以整体处理时，[PCA](pca.md) 是同一中心化重构问题更直接的参照。如果想学习的是非负模式的加和，而不是相对均值的带符号方向，可考虑 [MiniBatchNMF](minibatch-nmf.md)。

各批应保持相同的特征定义和预处理方式，并在代表性数据上与完整 PCA 比较重构误差和保留子空间。当前实现累积历史而不设置遗忘因子，概括的是全部已处理观测，而不只是最近的运行状态。能够流式更新，并不意味着它能自动检测分布漂移。

## 导入路径

```python
from statgpu.unsupervised import IncrementalPCA
```

当前估计器支持 CPU、CuPy/CUDA 与 Torch CUDA 上的稠密输入。

## 目标函数 / 损失函数

对 `k` 个主成分，`IncrementalPCA` 近似求解中心化、秩为 `k` 的 PCA 目标：

$$
\min_{V_k^\top V_k=I}
\left\|X - \bar{X} - (X-\bar{X})V_kV_k^\top\right\|_F^2 .
$$

## 估计方程

每次 `partial_fit` 先更新该批的均值与方差，再把历史的低秩基、当前中心化批次和均值校正行合并为一个紧凑矩阵做 SVD，取前 `n_components` 个右奇异向量作为 `components_`。

## 参数

- `n_components`：保留的主成分个数；`fit` 中为 `None` 时取 `min(n_samples, n_features)`，直接 `partial_fit` 时由首批确定秩，之后不自动增加。
- `batch_size`：`fit` 内部使用的批次大小；`partial_fit` 使用调用方传入的批次。
- `whiten`：将 `transform` 的结果除以解释方差的平方根，并设置很小的方差下限。
- `copy`：sklearn 风格兼容参数。
- `device`：`"auto"`、`"cpu"`、`"cuda"` 或 `"torch"`。

`fit` 会先接收并转换完整输入数组，再分批处理，并不会自动从磁盘流式读取。若不希望一次性载入全部数据，应自行读取稠密批次并传给 `partial_fit`。直接增量拟合时，首批过小会限制默认秩；重新调用 `fit` 会重置它。

## 一个可独立运行的 CPU 示例

<!-- learner-example: incremental-pca -->
```python
import numpy as np
from statgpu.unsupervised import IncrementalPCA

rng = np.random.default_rng(0)
X = rng.normal(size=(60, 5))
model = IncrementalPCA(n_components=2, device="cpu")
for start in range(0, len(X), 15):
    model.partial_fit(X[start:start + 15])
Z = model.transform(X)
X_hat = model.inverse_transform(Z)
print(Z.shape, model.n_samples_seen_, X_hat.shape)
```

最终基把全部 60 行转换为 `(60, 2)`，`n_samples_seen_` 为 60。`mean_` 和 `var_` 汇总全部已处理行，但之前各批算出的坐标不会自动转换到最终基。

安装了相应 GPU 后端后，可新建估计器并指定 `device="cuda"`（CuPy）或 `device="torch"`（Torch CUDA）。数组通常留在该后端；输出所在设备及主机端步骤见 [API 参考](api-reference.md#incrementalpca)。显式请求的 GPU 不可用时会报错。

## 近似与解释边界

`IncrementalPCA` 是批次/流式近似估计器，结果会受批次顺序和批次大小影响；不同后端的浮点分解也可能有差异。应在合理误差范围内比较解释方差、重构与保留子空间；成分符号不唯一，重复奇异值对应的子空间基也不唯一。

## 输出字段

- `components_`
- `mean_`
- `var_`
- `explained_variance_`
- `explained_variance_ratio_`
- `singular_values_`
- `n_components_`
- `n_features_in_`
- `n_samples_seen_`

## FAQ

**支持稀疏输入吗？**
不支持。仅支持稠密的二维浮点数组。


## 完整 API 参考

构造默认值、全部公开方法、输出形状与限制见 [IncrementalPCA API 参考](api-reference.md#incrementalpca)。

<a id="references"></a>

## 参考文献

- Ross, D. A., Lim, J., Lin, R.-S., & Yang, M.-H. (2008). Incremental learning for robust visual tracking. *International Journal of Computer Vision*, 77, 125-141. https://doi.org/10.1007/s11263-007-0075-7
- scikit-learn Developers. `sklearn.decomposition.IncrementalPCA`. scikit-learn documentation. https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html
