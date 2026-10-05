# IncrementalPCA

> 语言：中文
> 最后更新：2026-10-05
> 切换：[English](../../en/unsupervised/incremental-pca.md)

## 概览

`IncrementalPCA` 在稠密的小批量（mini-batch）数据上拟合主成分，同时维护累计均值、方差、样本数以及截断 SVD 基。当前支持 CPU、CuPy/CUDA 与 Torch CUDA。

## 何时使用

需要逐批学习线性基时，可使用 IncrementalPCA。每批后的截断可能丢失完整 PCA 会保留的方向，应在代表性子集上比较效果，并固定预处理。直接使用 `partial_fit` 时，默认秩由首批决定，首批过小会限制本次增量拟合可保留的方向数；重新调用 `fit` 会重置它。

## 导入路径

```python
from statgpu.unsupervised import IncrementalPCA
```

## 目标函数 / 损失函数

对 `k` 个主成分，`IncrementalPCA` 近似求解中心化的 rank-k PCA 目标：

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

## References

- Ross, D. A., Lim, J., Lin, R.-S., & Yang, M.-H. (2008). Incremental learning for robust visual tracking. *International Journal of Computer Vision*, 77, 125-141. https://doi.org/10.1007/s11263-007-0075-7
- scikit-learn Developers. `sklearn.decomposition.IncrementalPCA`. scikit-learn documentation. https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html
