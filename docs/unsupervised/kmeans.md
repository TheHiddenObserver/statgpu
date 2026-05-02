# KMeans

> 语言：中文
> 最后更新：2026-05-02
> English: [English](../en/unsupervised/kmeans.md)

## 概览

`KMeans` 通过最小化平方 Euclidean 簇内误差，把 dense observations 分成 `n_clusters` 个簇。它支持 CPU、CuPy/CUDA 和 Torch CUDA。

## 导入路径

```python
from statgpu.unsupervised import KMeans
```

## 目标函数 / 损失函数

KMeans 最小化 inertia：

$$
\min_{C, z} \sum_{i=1}^{n} \left\|x_i - c_{z_i}\right\|_2^2 .
$$

其中 `C` 是 cluster centers，`z_i` 是样本 `i` 的簇标签。

## 估计方程

实现使用 Lloyd 迭代：

- 使用 `random` 或 greedy `k-means++` 初始化 centers。
- 用下式计算平方距离，把每个样本分配到最近中心：
  $$
  d_{ij}^2 = \left\|x_i\right\|_2^2 + \left\|c_j\right\|_2^2 - 2 x_i^\top c_j .
  $$
- 把每个 center 更新为该簇样本均值。
  $$
  c_j = \frac{1}{|\{i: z_i = j\}|}\sum_{i:z_i=j} x_i .
  $$
- 空簇用当前距离 assigned center 最远的样本重置。
- 当 squared center movement 不超过 `tol` 或达到 `max_iter` 时停止。
- 运行 `n_init` 次初始化，保留 inertia 最低的结果。

## 参数

- `n_clusters`：簇数量。
- `init`：`"k-means++"` 或 `"random"`；不支持 callable init。
- `n_init`：`"auto"` 对 k-means++ 使用 `1`，对 random 使用 `10`。
- `max_iter`、`tol`、`random_state`。
- `device`：`"auto"`、`"cpu"`、`"cuda"` 或 `"torch"`。

## CPU+GPU 示例

```python
import numpy as np
from statgpu.unsupervised import KMeans

X = np.random.default_rng(0).normal(size=(10000, 32))

km = KMeans(n_clusters=8, random_state=0, device="torch")
labels = km.fit_predict(X)
distances = km.transform(X)
```

## strict/approx 差异

KMeans 是非凸迭代优化器，不是 strict inference estimator。不同初始化可能得到不同局部最优；可复现性取决于 `random_state`、`init`、`n_init`、`max_iter` 和 `tol`。

## 输出字段

- `cluster_centers_`
- `labels_`
- `inertia_`
- `n_iter_`
- `n_features_in_`

## FAQ

**为什么标签 ID 和 sklearn 不同但聚类看起来一样？**
Cluster ID 本身任意。验证应使用 inertia、center matching 或 permutation-invariant label metrics。

**支持 sparse input 或 `sample_weight` 吗？**
不支持。Phase 2 dense KMeans 对 sparse input 和 `sample_weight` 明确报错。

## 外部验证

- 测试：`dev/tests/test_unsupervised_kmeans.py`。
- Benchmark：`dev/benchmarks/benchmark_unsupervised.py`。
- Baseline：sklearn KMeans，对齐 `n_clusters`、初始化、`n_init`、`max_iter`、`tol` 和 seed。

## References

- MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. In *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability* (Vol. 1, pp. 281-297). University of California Press.
- Lloyd, S. P. (1982). Least squares quantization in PCM. *IEEE Transactions on Information Theory*, 28(2), 129-137. https://doi.org/10.1109/TIT.1982.1056489
- Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. In *Proceedings of the Eighteenth Annual ACM-SIAM Symposium on Discrete Algorithms (SODA 2007)* (pp. 1027-1035). Society for Industrial and Applied Mathematics.
