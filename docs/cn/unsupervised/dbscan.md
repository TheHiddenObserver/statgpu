# DBSCAN

> 语言：简体中文
> 最后更新：2026-09-03
> 切换：[English](../../en/unsupervised/dbscan.md)

## 它解决什么问题？

`DBSCAN` 寻找由相邻观测组成的高密度区域，并把孤立观测标记为噪声。与 K-Means 不同，它不需要预先指定簇数，也可以识别非球形的簇。

statgpu 的实现面向稠密欧氏数据，提供 CPU、CuPy/CUDA 和 Torch CUDA 路径。

## 什么时候使用？

以下情况适合从 DBSCAN 开始：

- 簇由局部密度定义，而不是由到质心的距离定义；
- 簇可能弯曲或形状不规则；
- 噪声识别本身就是任务的一部分；
- 事先不知道簇数；
- 大多数簇可以用一个有意义的邻域尺度描述。

如果不同簇的密度差异很大、高维距离已经失去区分度、每个新观测都必须获得预测，或需要软成员概率，应考虑其他方法。若原始特征上的欧氏距离没有实际意义，应先降维或构造有意义的表示。

## 直观理解

DBSCAN 把训练点分为三类：

- **核心点：** 以自身为中心、半径为 `eps` 的范围内至少有 `min_samples` 个观测，计数包含自身；
- **边界点：** 自身不是核心点，但位于某个核心点的 `eps` 邻域内；
- **噪声点：** 无法从任何核心连通分量通过密度可达关系到达，标签为 `-1`。

彼此连通的核心点组成一个簇，可达的边界点再依附到该簇。因此 DBSCAN 寻找的是足够稠密区域的连通分量，而不是预先划分固定数量的质心区域。

## 密度判据

当

$$
\left|
\left\{
x_j:\lVert x_i-x_j\rVert_2\leq\varepsilon
\right\}
\right|
\geq \text{min\_samples}
$$

时，$x_i$ 是核心点。其中 $\varepsilon$ 对应 `eps`，闭邻域包含 $x_i$ 自身。

DBSCAN 没有需要最小化的可微损失函数；结果由邻域图和密度可达规则决定。

## 先准备数据

`eps` 与特征使用相同单位。一个以“千”为单位的变量可能完全压过取值在 0 到 1 之间的变量。拟合前应标准化连续特征，或使用具有科学含义的变换。

还应检查重复值、缺失值与高维稀疏性。当前实现接受稠密数组，并且只支持 `metric="euclidean"`。

## 最小可运行示例

下面生成三个紧凑簇，并加入均匀分布的噪声候选点。

```python
import numpy as np
from statgpu.unsupervised import DBSCAN

rng = np.random.default_rng(3)
cluster_a = rng.normal(loc=(-2.0, -1.0), scale=0.22, size=(120, 2))
cluster_b = rng.normal(loc=(0.2, 1.8), scale=0.25, size=(140, 2))
cluster_c = rng.normal(loc=(2.2, -0.5), scale=0.20, size=(110, 2))
noise = rng.uniform(low=-4.0, high=4.0, size=(30, 2))
X = np.vstack([cluster_a, cluster_b, cluster_c, noise])

model = DBSCAN(
    eps=0.45,
    min_samples=6,
    device="cpu",
).fit(X)

labels = model.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = int(np.sum(labels == -1))

print("簇数：", n_clusters)
print("噪声点数：", n_noise)
print("前十个标签：", labels[:10])
```

该数据通常得到约 3 个簇。部分均匀采样点可能落在某个簇附近，因此最终被标为噪声的数量不一定正好等于生成的 30 个噪声候选点。

## 如何读取结果？

- `labels_[i]` 是第 `i` 个训练样本所属的簇。
- `0, 1, ...` 只是标识符，数值大小没有顺序含义。
- `-1` 表示噪声。
- `core_sample_indices_` 是核心点的下标。
- `components_` 保存拟合得到的核心样本。
- 簇数等于不同非负标签的数量。

DBSCAN 不支持对未见样本调用 `predict`。处理新观测时，应重新拟合，或根据应用场景定义并验证独立的分配规则。

## 关键参数怎么选？

| 参数 | 默认值 | 选择建议 |
|---|---:|---|
| `eps` | `0.5` | 以特征单位表示的邻域半径；可从领域知识或排序后的 $k$ 近邻距离开始 |
| `min_samples` | `5` | 值越大，对局部密度证据要求越强，也更容易产生噪声；值越小，更容易接受小而嘈杂的簇 |
| `metric` | `"euclidean"` | 当前唯一支持的距离 |
| `algorithm` | `"auto"` | CPU 近邻搜索提示，通常保持自动 |
| `batch_size` | `None` | GPU 距离分批大小；临时距离块显存不足时减小 |
| `device` | `"auto"` | 根据数据规模与已安装后端选择 `cpu`、`cuda` 或 `torch` |

一种实用调参顺序是：

1. 缩放或变换特征；
2. 根据“可信的最小局部支持数”选择 `min_samples`；
3. 计算每个点到第 `min_samples` 个近邻的距离并排序；
4. 在距离曲线明显转折附近尝试多个 `eps`；
5. 比较簇的稳定性与领域意义，而不只是簇数。

密度差异明显时不存在通用的最佳启发式规则。

## 与其他方法比较

| 方法 | 更适合的情况 |
|---|---|
| K-Means | 簇紧凑、近似球形，且簇数已知 |
| Gaussian mixture | 需要软成员概率与椭圆形分量 |
| 层次聚类 | 需要层级结构或树状图 |
| HDBSCAN（当前 API 未提供） | 簇密度不同，且重视层级密度稳定性 |

## CPU 与 GPU 行为

所有路径使用相同的稠密欧氏邻域判据：

- 低维 CPU 数据使用 `cKDTree`，可用时再进入编译的 Cython 标签流水线；
- 较高维 CPU 数据使用优化近邻搜索，并保持相同标签契约；
- CuPy 与 Torch 路径在 GPU 上分批计算距离，并用后端特定流程构造连通分量。

[CPU/GPU 加速实现](../guides/acceleration-internals.md)进一步说明 `batch_size`、设备传输、同步和标签不变验证为何会影响该路径。

```python
# CuPy CUDA 路径
gpu_model = DBSCAN(
    eps=0.45,
    min_samples=6,
    batch_size=2048,
    device="cuda",
).fit(X)

# Torch CUDA 路径
torch_model = DBSCAN(
    eps=0.45,
    min_samples=6,
    device="torch",
).fit(X)
```

CPU 与 GPU 可能使用不同数字标识同一个划分。比较结果时应使用 adjusted Rand index 等不依赖标签编号的指标，而不是要求标签数字逐项相等。恰好位于 `eps` 边界附近的点也可能受浮点比较影响。

当前性能证据统一放在版本化[基准面板](/dashboard/)中，以便同时查看硬件与数据来源。

## 常见误区

- 未缩放特征是邻域含义失真的最常见原因。
- 增大 `eps` 可能合并不同群组；减小它可能把簇切碎并产生更多噪声。
- 增大 `min_samples` 不等于简单地获得“更好”的簇，而是改变密度定义。
- 高维空间中的距离可能趋于相似，通常需要先降维。
- 聚类标签是探索性结构，不是客观真实类别。
- 在完整数据上拟合 DBSCAN 后，并没有内置的样本外预测步骤。

## API 与验证

导入路径：`statgpu.unsupervised.DBSCAN`

方法：`fit` 与 `fit_predict`。按设计，调用 `predict` 会抛出 `NotImplementedError`。

输出：`labels_`、`core_sample_indices_`、`components_` 与 `n_features_in_`。

与 scikit-learn 对齐的欧氏配置验证位于 `dev/tests/test_unsupervised_dbscan.py`。基准生成器会把准确性与运行时间来源独立记录，不与本概念指南混在一起。

## 参考文献

- Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. In *KDD-96* (pp. 226-231).
- Schubert, E., Sander, J., Ester, M., Kriegel, H.-P., & Xu, X. (2017). DBSCAN revisited, revisited: Why and how you should (still) use DBSCAN. *ACM Transactions on Database Systems*, 42(3), Article 19.
