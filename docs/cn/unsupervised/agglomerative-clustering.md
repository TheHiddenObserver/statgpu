# AgglomerativeClustering

> 语言：中文
> 最后更新：2026-09-29
> 切换：[English](../../en/unsupervised/agglomerative-clustering.md)

## 概览

`AgglomerativeClustering` 为稠密欧氏（Euclidean）数据构建精确的层次聚类树。CPU、CuPy/CUDA 与 Torch CUDA 路径都支持 `"single"`、`"complete"`、`"average"` 和 `"ward"` 四种连接准则。GPU 路径是稠密、精确的第一版实现，面向中小规模数据；显式请求 GPU 时不会静默回退到 CPU。

## 导入路径

```python
from statgpu.unsupervised import AgglomerativeClustering
```

## 目标函数 / 损失函数

层次聚类是贪心合并过程，不是全局光滑优化问题。每一步都合并连接准则（linkage criterion）取值最小的一对簇。

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

CPU 路径调用 SciPy 的层次聚类子模块计算精确连接；显式 CuPy/Torch 路径使用 statgpu 自有的、常驻后端的稠密距离矩阵，并用 Lance-Williams 公式增量更新连接距离。

## 参数

- `n_clusters`：切树后的簇数。
- `linkage`：`"single"`、`"complete"`、`"average"` 或 `"ward"`。
- `metric`：仅支持 `"euclidean"`。
- `device`：`"cpu"`、`"cuda"`、`"torch"` 或 `"auto"`。该估计器的 `device="auto"` 仍默认选择 CPU；显式请求 GPU 时会执行稠密、精确的后端实现。

## CPU+GPU 示例

```python
import numpy as np
from statgpu.unsupervised import AgglomerativeClustering

X = np.random.default_rng(0).normal(size=(300, 6))

model = AgglomerativeClustering(n_clusters=4, linkage="ward", device="cpu")
labels = model.fit_predict(X)

model_gpu = AgglomerativeClustering(n_clusters=4, linkage="ward", device="cuda")
labels_gpu = model_gpu.fit_predict(X)  # NumPy 输入会被转到 CUDA 后端

# 若希望全程保持 CuPy 数组，可先显式转换再传入。
# import cupy as cp
# X_gpu = cp.asarray(X)
# labels_gpu = model_gpu.fit_predict(X_gpu)
```

## 严格与近似模式的差别

`AgglomerativeClustering` 没有统计推断意义上的严格模式：对稠密欧氏输入，CPU、CuPy 与 Torch 路径都给出精确计算。GPU 执行会分配稠密距离矩阵；一旦超过第一版实现设定的显存保护阈值，会明确抛出 `MemoryError`。

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

## 外部验证

- 测试脚本：`dev/tests/test_unsupervised_agglomerative.py`。
- 基准测试：`dev/benchmarks/benchmark_unsupervised_phase3b.py`。
- 最新远程验证产物：`results/unsupervised_agglomerative_gpu_verify_20260509_agglo_gpu.json` 和 `results/unsupervised_agglomerative_gpu_verify_summary_20260509_agglo_gpu.md`。
- 对齐基线：sklearn 的 `AgglomerativeClustering`、SciPy 的 `linkage`，以及参数可以对齐时的 R `cluster::agnes`。
- Phase 3B 的验证目标：四种连接准则在标签置换下的一致性、调整兰德指数（ARI），以及可比场景下的连接距离。

## References

- Sneath, P. H. A. (1957). The application of computers to taxonomy. *Journal of General Microbiology*, 17(1), 201-226. https://doi.org/10.1099/00221287-17-1-201
- Murtagh, F. (1983). A survey of recent advances in hierarchical clustering algorithms. *The Computer Journal*, 26(4), 354-359. https://doi.org/10.1093/comjnl/26.4.354
- Muellner, D. (2013). fastcluster: Fast hierarchical, agglomerative clustering routines for R and Python. *Journal of Statistical Software*, 53(9), 1-18. https://doi.org/10.18637/jss.v053.i09
- SciPy Developers. `scipy.cluster.hierarchy`: Hierarchical clustering. SciPy documentation. https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html

## 完整 API 参考

构造默认值、全部公开方法、输出形状与限制见 [AgglomerativeClustering API 参考](api-reference.md#agglomerativeclustering)。
