# AgglomerativeClustering

> 语言：中文
> 最后更新：2026-05-09
> English: [English](../en/unsupervised/agglomerative-clustering.md)

## 概览

`AgglomerativeClustering` 为 dense Euclidean 数据构建 exact 层次聚类树。当前 CPU、CuPy/CUDA 和 Torch CUDA 路径都支持 `"single"`、`"complete"`、`"average"` 和 `"ward"` linkage。GPU 路径是 dense exact v1，面向小中规模数据；显式 GPU device 不会静默回退到 CPU。

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

CPU 路径使用 SciPy hierarchy routines 计算 exact linkage。显式 CuPy/Torch 路径使用 statgpu 自有 backend-resident dense distance matrix 和 Lance-Williams linkage 更新。

## 参数

- `n_clusters`：切树后的簇数。
- `linkage`：`"single"`、`"complete"`、`"average"` 或 `"ward"`。
- `metric`：仅支持 `"euclidean"`。
- `device`：`"cpu"`、`"cuda"`、`"torch"` 或 `"auto"`。该 estimator 的 `device="auto"` 仍默认 CPU；显式 GPU device 使用 dense exact backend execution。

## CPU+GPU 示例

```python
import numpy as np
from statgpu.unsupervised import AgglomerativeClustering

X = np.random.default_rng(0).normal(size=(300, 6))

model = AgglomerativeClustering(n_clusters=4, linkage="ward", device="cpu")
labels = model.fit_predict(X)

# 安装 CuPy 后，GPU 路径应传入 CUDA backend 上的数组。
# import cupy as cp
# X_gpu = cp.asarray(X)
model_gpu = AgglomerativeClustering(n_clusters=4, linkage="ward", device="cuda")
labels_gpu = model_gpu.fit_predict(X_gpu)
```

## strict/approx 差异

AgglomerativeClustering 没有 strict inference 模式。CPU、CuPy 和 Torch 路径对 dense Euclidean 输入都是 exact 计算。GPU execution 会分配 dense distance matrix；如果超过 v1 显存保护阈值，会明确抛出 `MemoryError`。

## 输出字段

- `labels_`
- `children_`
- `distances_`
- `n_features_in_`

## FAQ

**什么时候适合用 GPU 路径？**
小中规模 dense 数据可以显式使用 `device="cuda"` 或 `device="torch"`。层次聚类仍是顺序贪心过程且需要 dense 距离矩阵；大数据可能更适合 CPU 路径或其他聚类方法。

**能对新样本 predict 吗？**
不能。当前实现不支持对 unseen samples 调用 `predict`。

## 外部验证

- 测试：`dev/tests/test_unsupervised_agglomerative.py`。
- Benchmark：`dev/benchmarks/benchmark_unsupervised_phase3b.py`。
- 最新远程 artifact：`results/unsupervised_agglomerative_gpu_verify_20260509_agglo_gpu.json` 和 `results/unsupervised_agglomerative_gpu_verify_summary_20260509_agglo_gpu.md`。
- Baseline：sklearn `AgglomerativeClustering`、SciPy `linkage`，以及参数可对齐时的 R `cluster::agnes`。
- Phase 3B 验证目标：`"single"`、`"complete"`、`"average"`、`"ward"` 的 label permutation-invariant 一致性、ARI，以及可比较场景下的 linkage distance。

## References

- Sneath, P. H. A. (1957). The application of computers to taxonomy. *Journal of General Microbiology*, 17(1), 201-226. https://doi.org/10.1099/00221287-17-1-201
- Murtagh, F. (1983). A survey of recent advances in hierarchical clustering algorithms. *The Computer Journal*, 26(4), 354-359. https://doi.org/10.1093/comjnl/26.4.354
- Muellner, D. (2013). fastcluster: Fast hierarchical, agglomerative clustering routines for R and Python. *Journal of Statistical Software*, 53(9), 1-18. https://doi.org/10.18637/jss.v053.i09
- SciPy Developers. `scipy.cluster.hierarchy`: Hierarchical clustering. SciPy documentation. https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html
