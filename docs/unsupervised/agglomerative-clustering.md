# AgglomerativeClustering

> 语言：中文
> 最后更新：2026-05-07
> English: [English](../en/unsupervised/agglomerative-clustering.md)

## 概览

`AgglomerativeClustering` 为 dense Euclidean 数据构建 exact 层次聚类树。当前 CPU 路径支持 `"single"`、`"complete"`、`"average"` 和 `"ward"` linkage。显式 `device="cuda"` 或 `device="torch"` 会抛出 `NotImplementedError`，不会静默回退到 CPU。

## 导入路径

```python
from statgpu.unsupervised import AgglomerativeClustering
```

## 目标函数 / 损失函数

层次聚类是贪心合并过程，不是全局光滑优化问题。每一步都会选择 linkage criterion 最小的一对簇进行合并。

single linkage：

$$
d(A, B)
=
\min_{x \in A,\; y \in B}
\left\|x - y\right\|_2 .
$$

complete linkage：

$$
d(A, B)
=
\max_{x \in A,\; y \in B}
\left\|x - y\right\|_2 .
$$

average linkage：

$$
d(A, B)
=
\frac{1}{|A||B|}
\sum_{x \in A}\sum_{y \in B}
\left\|x - y\right\|_2 .
$$

Ward linkage 合并使 within-cluster squared error 增量最小的一对簇：

$$
\Delta(A, B)
=
\frac{|A||B|}{|A|+|B|}
\left\|\bar{x}_A-\bar{x}_B\right\|_2^2 .
$$

## 估计方程

- 从每个样本一个簇开始。
- 反复合并所选 linkage criterion 最小的两个簇。
- 将合并树保存为 `children_`，将合并距离保存为 `distances_`。
- 按 `n_clusters` 切树并生成 `labels_`。

当前实现使用 SciPy hierarchy routines 计算 exact CPU linkage，不暴露 GPU 路径。

## 参数

- `n_clusters`：切树后的簇数。
- `linkage`：`"single"`、`"complete"`、`"average"` 或 `"ward"`。
- `metric`：仅支持 `"euclidean"`。
- `device`：仅 CPU；显式 `"cuda"` 或 `"torch"` 会抛出 `NotImplementedError`。

## CPU+GPU 示例

```python
import numpy as np
from statgpu.unsupervised import AgglomerativeClustering

X = np.random.default_rng(0).normal(size=(300, 6))

model = AgglomerativeClustering(n_clusters=4, linkage="ward", device="cpu")
labels = model.fit_predict(X)
```

## strict/approx 差异

AgglomerativeClustering 没有 strict inference 模式。已支持的 CPU linkage 对 dense Euclidean 输入是 exact 计算。GPU execution 会明确报错，不会静默降级。

## 输出字段

- `labels_`
- `children_`
- `distances_`
- `n_features_in_`

## FAQ

**为什么不支持 GPU？**
当前目标是提供清晰、可审计的 exact CPU baseline。高效 GPU 层次聚类需要独立实现 linkage 更新和内存策略，不能简单沿用当前 dense matrix 路径。

**能对新样本 predict 吗？**
不能。当前实现不支持对 unseen samples 调用 `predict`。

## 外部验证

- 测试：`dev/tests/test_unsupervised_agglomerative.py`。
- Benchmark：`dev/benchmarks/benchmark_unsupervised_phase3b.py`。
- 最新远程 artifact：`results/unsupervised_phase3b_verify_20260507_003957.json`。
- Baseline：sklearn `AgglomerativeClustering`、SciPy `linkage`，以及参数可对齐时的 R `cluster::agnes`。
- Phase 3B 验证目标：`"single"`、`"complete"`、`"average"`、`"ward"` 的 label permutation-invariant 一致性、ARI，以及可比较场景下的 linkage distance。

## References

- Sneath, P. H. A. (1957). The application of computers to taxonomy. *Journal of General Microbiology*, 17(1), 201-226. https://doi.org/10.1099/00221287-17-1-201
- Murtagh, F. (1983). A survey of recent advances in hierarchical clustering algorithms. *The Computer Journal*, 26(4), 354-359. https://doi.org/10.1093/comjnl/26.4.354
- Muellner, D. (2013). fastcluster: Fast hierarchical, agglomerative clustering routines for R and Python. *Journal of Statistical Software*, 53(9), 1-18. https://doi.org/10.18637/jss.v053.i09
- SciPy Developers. `scipy.cluster.hierarchy`: Hierarchical clustering. SciPy documentation. https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html
