# AgglomerativeClustering

> 语言：中文
> 最后更新：2026-05-02
> English: [English](../en/unsupervised/agglomerative-clustering.md)

## 概览

`AgglomerativeClustering` 为 dense Euclidean data 构造 exact single-linkage hierarchy。Phase 2 仅支持 CPU。

## 导入路径

```python
from statgpu.unsupervised import AgglomerativeClustering
```

## 目标函数 / 损失函数

Single-linkage agglomerative clustering 是贪心层次过程，不是全局光滑优化。每一步合并 pairwise distance 最小的两个簇：

$$
d(A, B)
=
\min_{x \in A,\; y \in B}
\left\|x - y\right\|_2 .
$$

## 估计方程

- 从每个样本一个簇开始。
- 反复合并 single-linkage distance 最小的两个簇。
- 把 merge tree 存为 `children_`，把 merge distances 存为 `distances_`。
- 按 `n_clusters` 切分树并生成 labels。

当前实现使用 SciPy hierarchy routines 计算 exact CPU linkage，不暴露 GPU 路径。

## 参数

- `n_clusters`：切树后的簇数量。
- `linkage`：仅支持 `"single"`。
- `metric`：仅支持 `"euclidean"`。
- `device`：Phase 2 仅支持 CPU；显式 `"cuda"` 或 `"torch"` 会抛出 `NotImplementedError`。

## CPU+GPU 示例

```python
import numpy as np
from statgpu.unsupervised import AgglomerativeClustering

X = np.random.default_rng(0).normal(size=(300, 6))

model = AgglomerativeClustering(n_clusters=4, linkage="single", device="cpu")
labels = model.fit_predict(X)
```

## strict/approx 差异

AgglomerativeClustering 没有 strict inference 模式。Phase 2 对支持的 dense Euclidean 输入使用 exact CPU single linkage。GPU execution 会明确报错，不会静默降级。

## 输出字段

- `labels_`
- `children_`
- `distances_`
- `n_features_in_`

## FAQ

**为什么不支持 GPU？**
Phase 2 先提供清晰的 exact CPU baseline。GPU single-linkage 需要单独的实现计划。

**能对新样本 predict 吗？**
不能。当前 AgglomerativeClustering 不支持 unseen samples 的 `predict`。

## 外部验证

- 测试：`dev/tests/test_unsupervised_agglomerative.py`。
- Benchmark：`dev/benchmarks/benchmark_unsupervised_phase2.py`。
- Baseline：sklearn `AgglomerativeClustering(linkage="single")`、SciPy `linkage(method="single")`，以及可用时的 R `cluster::agnes`。
- 最新远程矩阵：sklearn 和 SciPy labels 与 statgpu CPU 的 ARI 为 `1.0`。

## References

- Sneath, P. H. A. (1957). The application of computers to taxonomy. *Journal of General Microbiology*, 17(1), 201-226. https://doi.org/10.1099/00221287-17-1-201
- Murtagh, F. (1983). A survey of recent advances in hierarchical clustering algorithms. *The Computer Journal*, 26(4), 354-359. https://doi.org/10.1093/comjnl/26.4.354
- Müllner, D. (2013). fastcluster: Fast hierarchical, agglomerative clustering routines for R and Python. *Journal of Statistical Software*, 53(9), 1-18. https://doi.org/10.18637/jss.v053.i09
- SciPy Developers. `scipy.cluster.hierarchy`: Hierarchical clustering. SciPy documentation. https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html
