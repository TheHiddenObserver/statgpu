# MiniBatchKMeans

> 语言：中文
> 最后更新：2026-10-05
> 切换：[English](../../en/unsupervised/minibatch-kmeans.md)
> 路径：`statgpu.unsupervised.MiniBatchKMeans`

## 概览

`MiniBatchKMeans` 使用小批量更新聚类中心，避免每一步 Lloyd 迭代都扫描完整数据集。

## 何时使用

需要多次用较小批次更新中心时，可使用 MiniBatchKMeans。`fit` 仍接收完整稠密数据，并在最后执行全数据修正；外部数据流应使用 `partial_fit`。各批次应采用一致预处理，首批应有足够样本初始化所有簇。

## 导入路径

从 `statgpu.unsupervised` 导入：

```python
from statgpu.unsupervised import MiniBatchKMeans
```

## 目标函数

目标仍是 KMeans 的惯性：

$$
\sum_i \min_j \|x_i - c_j\|_2^2.
$$

## 估计方程

对每一批中被分配到簇 `j` 的样本，中心更新为：

$$
c_j \leftarrow c_j + \eta_j(\bar{x}_{B_j} - c_j),
\qquad
\eta_j = \frac{|B_j|}{n_j + |B_j|}.
$$

`fit` 在小批量更新结束后，会对完整稠密数据再做少量精确的 Lloyd 迭代，以降低最终的全数据惯性；主体训练仍采用小批量更新。

## 参数

`n_clusters`、`init`、`n_init`、`batch_size`、`max_iter`、`max_no_improvement`、`tol`、`random_state`、`device`。

## 一个可独立运行的 CPU 示例

<!-- learner-example: minibatch-kmeans -->
```python
import numpy as np
from statgpu.unsupervised import MiniBatchKMeans

rng = np.random.default_rng(0)
X = np.vstack([rng.normal(-2, 0.3, (30, 2)), rng.normal(2, 0.3, (30, 2))])
rng.shuffle(X)
model = MiniBatchKMeans(n_clusters=2, random_state=0, device="cpu")
for start in range(0, len(X), 15):
    model.partial_fit(X[start:start + 15])
labels = model.predict(X)
print(labels.shape, model.labels_.shape, model.n_steps_)
```

全数据预测标签形状为 `(60,)`，而 `labels_` 只覆盖最后一批，形状为 `(15,)`。保存的批次标签基于该次中心更新之前的分配；`predict` 使用当前中心。`partial_fit` 的 `counts_` 累积各批分配计数。

安装了相应 GPU 后端后，可新建估计器并指定 `device="cuda"`（CuPy）或 `device="torch"`（Torch CUDA）。数组通常留在该后端；输出所在设备及主机端步骤见 [API 参考](api-reference.md#minibatchkmeans)。显式请求的 GPU 不可用时会报错。

## 近似与解释边界

该方法属于随机近似优化；公平比较时应固定相同的初始中心、批次顺序、收敛阈值与迭代预算。

## 输出

`cluster_centers_`、`labels_`、`inertia_`、`n_iter_`、`n_steps_`、`counts_`、`n_features_in_`。

## FAQ

仅支持稠密的欧氏输入；不支持稀疏输入、`sample_weight` 和可调用的 `init`。


## 数值与使用注意事项

显式提供的初始中心必须全部有限。当前实现会检查形状，但不一定拒绝 NaN 或无穷值；`partial_fit` 可能返回非有限的中心和惯性。对于 NumPy 初始中心，应在构造前检查 `np.isfinite(initial_centers).all()`，先拒绝或修正无效值，再拟合。普通 `fit` 的最终修正步骤可能掩盖无效初始化，因此输出有限也不能替代这一输入检查。

平方距离通过展开范数计算。很大的共同偏移可能引起消减误差，影响距离、惯性乃至标签。可先减去由训练数据确定的特征偏移，再拟合，并在预测时减去相同偏移；平移不会改变欧氏几何关系。报告中心时可加回偏移，恢复原始单位。

## 完整 API 参考

构造默认值、全部公开方法、输出形状与限制见 [MiniBatchKMeans API 参考](api-reference.md#minibatchkmeans)。

<a id="references"></a>

## 参考文献

- Sculley, D. (2010). Web-scale k-means clustering. *Proceedings of the 19th International Conference on World Wide Web*, 1177-1178.
