# AgglomerativeClustering

> 语言：中文
> 最后更新：2026-10-05
> 切换：[English](../../en/unsupervised/agglomerative-clustering.md)

## 概览

`AgglomerativeClustering` 为稠密欧氏（Euclidean）数据构建精确的层次聚类树。CPU、CuPy/CUDA 与 Torch CUDA 路径都支持 `"single"`、`"complete"`、`"average"` 和 `"ward"` 四种连接准则。GPU 路径是稠密、精确的实现，面向中小规模数据；显式请求 GPU 时不会静默回退到 CPU。

## 何时使用

需要查看逐层合并结构时，可使用层次聚类。单连接可能通过少数桥接点把组串起来；完全连接关注最大组间距离，平均连接使用成对距离的均值，Ward 倾向紧凑的组。特征尺度会改变层次结构。

## 导入路径

```python
from statgpu.unsupervised import AgglomerativeClustering
```

## 目标函数 / 损失函数

层次聚类是贪心合并过程，不是全局光滑优化问题。每一步都合并连接准则（linkage criterion）取值最小的一对簇。

单连接（`linkage="single"`）：

$$
d(A, B)
=
\min_{x \in A,\; y \in B}
\left\|x - y\right\|_2 .
$$

完全连接（`linkage="complete"`）：

$$
d(A, B)
=
\max_{x \in A,\; y \in B}
\left\|x - y\right\|_2 .
$$

平均连接（`linkage="average"`）：

$$
d(A, B)
=
\frac{1}{|A||B|}
\sum_{x \in A}\sum_{y \in B}
\left\|x - y\right\|_2 .
$$

Ward 连接每一步合并使簇内平方误差增量最小的一对簇：

$$
\Delta(A, B)
=
\frac{|A||B|}{|A|+|B|}
\left\|\bar{x}_A-\bar{x}_B\right\|_2^2 .
$$

## 估计方程

- 从每个样本一个簇开始。
- 反复合并所选连接准则取值最小的两个簇。
- 将合并树保存为 `children_`，将合并距离保存为 `distances_`。
- 按 `n_clusters` 切树并生成 `labels_`。

CPU 路径调用 SciPy 的层次聚类子模块计算精确连接；显式 CuPy/Torch 路径使用 statgpu 自有的、常驻后端的稠密距离矩阵。单连接先构建最小生成树，再在 CPU 上组装合并树；完全连接、平均连接和 Ward 连接则使用 Lance–Williams 公式更新连接距离。

## 参数

- `n_clusters`：切树后的簇数。
- `linkage`：`"single"`、`"complete"`、`"average"` 或 `"ward"`。
- `metric`：仅支持 `"euclidean"`。
- `device`：`"cpu"`、`"cuda"`、`"torch"` 或 `"auto"`。该估计器的 `device="auto"` 仍默认选择 CPU；显式请求 GPU 时会执行稠密、精确的后端实现。

## 一个可独立运行的 CPU 示例

<!-- learner-example: agglomerative-clustering -->
```python
import numpy as np
from statgpu.unsupervised import AgglomerativeClustering

rng = np.random.default_rng(0)
X = np.vstack([rng.normal(-2, 0.3, (15, 2)), rng.normal(2, 0.3, (15, 2))])
model = AgglomerativeClustering(n_clusters=2, linkage="ward", device="cpu")
labels = model.fit_predict(X)
print(labels.shape, model.children_.shape, model.distances_[-3:])
```

30 个样本对应 `(29, 2)` 的合并对。除了最终分组，还应查看合并高度；CPU 的 maxclust 切分在高度相同时可能得到少于请求数量的簇。不提供新样本预测。

安装了相应 GPU 后端后，可新建估计器并指定 `device="cuda"`（CuPy）或 `device="torch"`（Torch CUDA）。标签、合并对与合并高度始终返回 NumPy 数组，即使在 GPU 上拟合也如此；输出及主机端步骤见 [API 参考](api-reference.md#agglomerativeclustering)。显式请求的 GPU 不可用时会报错。

## 近似与解释边界

`AgglomerativeClustering` 没有统计推断意义上的严格模式：对稠密欧氏输入，CPU、CuPy 与 Torch 路径都给出精确计算。GPU 执行会分配稠密距离矩阵；一旦超过配置的距离矩阵内存上限，会明确抛出 `MemoryError`。

GPU 路径通过展开平方范数计算成对距离。很大的共同特征偏移可能导致消减误差并改变层次结构；应在拟合前以 float64 减去由训练数据确定的偏移，这不会改变原本的欧氏几何关系。SciPy 的 CPU 层次聚类路径不使用这一共享 GPU 距离公式。

`distances_` 中的 Ward 合并高度为 $\sqrt{2\Delta(A,B)}$，不是平方误差增加量 $\Delta(A,B)$ 本身。GPU 距离矩阵估计内存会与配置上限比较；默认上限为 1 GiB，可在导入模块前设置 `STATGPU_AGGLOMERATIVE_GPU_MAX_BYTES`。该值不是当前可用显存。

## 输出字段

- `labels_`
- `children_`
- `distances_`
- `n_features_in_`

## FAQ

**什么时候适合用 GPU 路径？**
中小规模的稠密数据可以显式使用 `device="cuda"` 或 `device="torch"`。层次聚类仍是顺序贪心过程且需要稠密距离矩阵；大数据可能更适合 CPU 路径或其他聚类方法。

**能对新样本 predict 吗？**
不能。当前实现不支持对未见样本调用 `predict`。


## 完整 API 参考

构造默认值、全部公开方法、输出形状与限制见 [AgglomerativeClustering API 参考](api-reference.md#agglomerativeclustering)。

<a id="references"></a>

## 参考文献

- Sneath, P. H. A. (1957). The application of computers to taxonomy. *Journal of General Microbiology*, 17(1), 201-226. https://doi.org/10.1099/00221287-17-1-201
- Murtagh, F. (1983). A survey of recent advances in hierarchical clustering algorithms. *The Computer Journal*, 26(4), 354-359. https://doi.org/10.1093/comjnl/26.4.354
- Muellner, D. (2013). fastcluster: Fast hierarchical, agglomerative clustering routines for R and Python. *Journal of Statistical Software*, 53(9), 1-18. https://doi.org/10.18637/jss.v053.i09
- SciPy Developers. `scipy.cluster.hierarchy`: Hierarchical clustering. SciPy documentation. https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html
