# DBSCAN

> 语言：中文
> 最后更新：2026-10-05
> 切换：[English](../../en/unsupervised/dbscan.md)

## 概览

`DBSCAN` 在稠密欧氏（Euclidean）数据上寻找由密度连通关系定义的簇。支持 CPU、CuPy/CUDA 与 Torch CUDA；计算和内存成本取决于密度、维数及是否安装编译扩展，GPU 路径包含主机传输。

## 何时使用

当密集区域可能形状不规则，且允许部分样本不归入任何簇时，可使用 DBSCAN。`eps` 与特征单位一致，应比较多个半径和 `min_samples` 设置；若不同簇的密度差异很大，一个统一半径可能并不合适。

## 导入路径

```python
from statgpu.unsupervised import DBSCAN
```

## 目标函数 / 损失函数

DBSCAN 不是光滑优化问题，没有可微的损失函数；它的准则是密度可达性（density reachability）：

- 如果一个点的闭 `eps` 邻域内至少包含 `min_samples` 个点，该点就是核心点（core point）。
  $$
  \left|\left\{x_j : \left\|x_i - x_j\right\|_2 \le \varepsilon\right\}\right|
  \ge \text{min\_samples}.
  $$
- 由 `eps` 邻接关系连通的核心点构成一个簇。
- 能从核心点集到达、但自身不是核心点的点称为边界点（border point）。
- 其余点是噪声（noise），标签为 `-1`。

## 计算路径与内存

CPU 数据不超过 12 个特征时使用 SciPy 树搜索；更高维数据使用 scikit-learn 的 `NearestNeighbors`，需安装 scikit-learn。编译好的 statgpu Cython 扩展可加速图标签计算，未编译时使用 Python 实现。

GPU 距离计算使用 float32，保存的核心样本使用 float64。Torch 路径在 GPU 上处理图和标签，但会先把最终标签与核心索引复制到主机，再发布后端数组。CuPy 路径还会逐批把边复制到主机，并使用 Python 辅助处理。因此选择 GPU 不代表没有主机传输。`batch_size` 只限制距离计算的批次，不限制整个近邻图；密集邻域仍可能占用二次增长的内存。`eps` 边界附近的 float32 比较可能不同，应检查标签和噪声判定的稳定性。

很大的共同特征偏移还可能在 GPU 转为 float32 或展开计算距离时丢失样本间距，即使这些距离并不接近 `eps` 边界。应在数据仍为 float64 时，先减去由训练数据确定的统一特征偏移，再拟合；平移保持欧氏距离不变，无需改变 `eps`。应保存该偏移，以便把 `components_` 转回原始单位。标签有限并不代表近邻图正确。

## 参数

- `eps`：邻域半径，必须为正且有限。非有限值当前不一定被拒绝，因此应在拟合前自行验证。
- `min_samples`：成为核心点所需的闭邻域样本数。
- `metric`：仅支持 `"euclidean"`。
- `batch_size`：可选的 GPU 距离计算批量大小，不限制全部图边的存储。
- `device`：`"auto"`、`"cpu"`、`"cuda"` 或 `"torch"`。

## 一个可独立运行的 CPU 示例

<!-- learner-example: dbscan -->
```python
import numpy as np
from statgpu.unsupervised import DBSCAN

rng = np.random.default_rng(0)
X = np.vstack([rng.normal(-2, 0.1, (20, 2)), rng.normal(2, 0.1, (20, 2)), [[8., 8.]]])
model = DBSCAN(eps=0.5, min_samples=3, device="cpu")
labels = model.fit_predict(X)
print(labels.shape, np.unique(labels), model.core_sample_indices_.shape)
```

最后一行远离其他样本，标签为表示噪声的 `-1`。`components_` 保存核心样本，不是聚类中心。实现不提供新数据预测规则；把新旧数据一起重新拟合可能改变原有标签。

安装了相应 GPU 后端后，可新建估计器并指定 `device="cuda"`（CuPy）或 `device="torch"`（Torch CUDA）。数组通常留在该后端；输出所在设备及主机端步骤见 [API 参考](api-reference.md#dbscan)。显式请求的 GPU 不可用时会报错。

## 近似与解释边界

算法目标由欧氏邻域决定，不涉及统计推断。当前未编译扩展的 CPU 路径存在一个限制：当核心点之间没有任何连边时，可能错误地把互不连通的核心点合并。遇到这类稀疏核心点配置时，应先与独立实现核对标签；稠密簇示例不能覆盖这个情况。

## 输出字段

- `labels_`
- `core_sample_indices_`
- `components_`
- `n_features_in_`

## FAQ

**能为新样本预测簇吗？**
不能。只提供训练数据的 `fit_predict`；`predict` 会抛出 `NotImplementedError`。

**GPU 一定更快吗？**
速度取决于数据量、密度、传输和可用内存。应测量完整工作流，支持 GPU 本身不代表加速保证。


## 完整 API 参考

构造默认值、全部公开方法、输出形状与限制见 [DBSCAN API 参考](api-reference.md#dbscan)。

<a id="references"></a>

## 参考文献

- Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. In *Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD-96)* (pp. 226-231). AAAI Press. https://aaai.org/papers/kdd96-037-a-density-based-algorithm-for-discovering-clusters-in-large-spatial-databases-with-noise/
- Schubert, E., Sander, J., Ester, M., Kriegel, H.-P., & Xu, X. (2017). DBSCAN revisited, revisited: Why and how you should (still) use DBSCAN. *ACM Transactions on Database Systems*, 42(3), Article 19. https://doi.org/10.1145/3068335
