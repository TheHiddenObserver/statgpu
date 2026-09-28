# KMeans

> 语言：中文
> 最后更新：2026-09-28
> 切换：[English](../../en/unsupervised/kmeans.md)

## 概览

`KMeans` 通过最小化平方欧氏簇内误差，把稠密观测分成 `n_clusters` 个簇。支持 CPU、CuPy/CUDA 与 Torch CUDA。

## 导入路径

```python
from statgpu.unsupervised import KMeans
```

## 目标函数 / 损失函数

`KMeans` 最小化惯性（inertia）：

$$
\min_{C, z} \sum_{i=1}^{n} \left\|x_i - c_{z_i}\right\|_2^2 .
$$

其中 `C` 是聚类中心，`z_i` 是样本 `i` 的簇标签。

## 估计方程

实现采用 Lloyd 迭代：

- 用 `random` 或贪心的 `k-means++` 初始化聚类中心。
- 用下式计算平方距离，把每个样本分配到最近中心：
  $$
  d_{ij}^2 = \left\|x_i\right\|_2^2 + \left\|c_j\right\|_2^2 - 2 x_i^\top c_j .
  $$
- 把每个聚类中心更新为该簇样本的均值。
  $$
  c_j = \frac{1}{|\{i: z_i = j\}|}\sum_{i:z_i=j} x_i .
  $$
- 空簇用距离其原属中心最远的样本重置。
- 当中心位移的平方不超过 `tol` 或达到 `max_iter` 时停止。
- 运行 `n_init` 次初始化，保留惯性最低的结果。

## 参数

- `n_clusters`：簇数量。
- `init`：`"k-means++"` 或 `"random"`；不支持可调用（callable）的 `init`。
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

`KMeans` 是非凸迭代优化器，不提供严格的统计推断；不同初始化可能得到不同的局部最优，可复现性取决于 `random_state`、`init`、`n_init`、`max_iter` 和 `tol`。

## 输出字段

- `cluster_centers_`
- `labels_`
- `inertia_`
- `n_iter_`
- `n_features_in_`

## FAQ

**为什么标签 ID 和 sklearn 不同但聚类看起来一样？**
簇编号本身是任意的；验证应使用惯性、中心匹配或对标签置换不变的指标。

**支持稀疏输入或 `sample_weight` 吗？**
不支持。Phase 2 的稠密 KMeans 会对稀疏输入和 `sample_weight` 明确报错。

## 外部验证

- 测试脚本：`dev/tests/test_unsupervised_kmeans.py`。
- 基准测试：`dev/benchmarks/benchmark_unsupervised.py`。
- 对齐基线：sklearn 的 KMeans，对齐 `n_clusters`、初始化方式、`n_init`、`max_iter`、`tol` 与随机种子。

## References

- MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. In *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability* (Vol. 1, pp. 281-297). University of California Press.
- Lloyd, S. P. (1982). Least squares quantization in PCM. *IEEE Transactions on Information Theory*, 28(2), 129-137. https://doi.org/10.1109/TIT.1982.1056489
- Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. In *Proceedings of the Eighteenth Annual ACM-SIAM Symposium on Discrete Algorithms (SODA 2007)* (pp. 1027-1035). Society for Industrial and Applied Mathematics.
