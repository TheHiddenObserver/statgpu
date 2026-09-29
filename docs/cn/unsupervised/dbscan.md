# DBSCAN

> 语言：中文
> 最后更新：2026-09-29
> 切换：[English](../../en/unsupervised/dbscan.md)

## 概览

`DBSCAN` 在稠密欧氏（Euclidean）数据上寻找由密度连通关系定义的簇。CPU 路径由 Cython 加速，低维数据比 sklearn 快 3–4 倍，高维数据与 sklearn 持平。GPU 路径（PyTorch CUDA）全程在设备上执行，没有 GPU→CPU 传输，比 sklearn 快 3–17 倍。

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

## CPU 策略

CPU 路径根据维度选择算法：

| 维度 | 策略 | 说明 |
|---|---|---|
| p ≤ 12 | cKDTree `query_pairs` + Cython | 单次树遍历；`dbscan_labels_from_pairs` 在 C 层完成计数、并查集（Union-Find）与标签分配。 |
| p > 12 | sklearn `radius_neighbors_graph` + Cython | 用 sklearn 的优化 BLAS 例程计算距离；`dbscan_labels_from_csr` 在 C 层处理 CSR 图。 |

如果 Cython 扩展没有编译，两条路径都会退回纯 Python 实现。

### Cython 模块：`_dbscan_cy_fast.pyx`

两个入口函数都在 C 层完成完整的标签流程，没有 Python 对象开销：

- `dbscan_labels_from_pairs(n_samples, min_samples, pairs)` — 接收 `query_pairs` 的原始 `(i, j)` 对。
- `dbscan_labels_from_csr(n_samples, min_samples, indptr, indices)` — 接收 CSR 稀疏图数组。

内部均使用：
- C 级邻居计数
- C 级并查集（路径压缩 + 按秩合并）
- C 级边界点分配

## GPU 策略（PyTorch CUDA）

GPU 路径把所有数据留在设备上：

1. **距离计算**：在 GPU 上批量执行 `float32` 矩阵乘法
2. **邻居计数**：GPU 上 `mask.sum(dim=1)`
3. **稀疏图**：在 GPU 上调用 `torch.nonzero`，边以 GPU 张量存储
4. **连通分量**：在 GPU 上做标签传播（`scatter_reduce_(amin)`）
5. **边界点分配**：在 GPU 上批量计算距离并 scatter

只有最终标签（`n × int64`）会传回 CPU，从而消除逐批 GPU→CPU 传输，并避免重复计算距离造成的显存不足。

### 标签传播算法

```
labels = arange(n_core)                          # 每个 core 点初始独立
for _ in range(50):                              # 通常 2-5 次迭代收敛
    min_labels = minimum(labels[src], labels[dst])  # 全边并行
    labels.scatter_reduce_(amin)                     # 并行 scatter
    if converged: break
```

这种算法非常适合 GPU：每次迭代对所有边完全并行，而 CPU 上的并查集必须逐条处理边。

## 参数

- `eps`：邻域半径，必须为正。
- `min_samples`：成为核心点所需的闭邻域样本数。
- `metric`：仅支持 `"euclidean"`。
- `batch_size`：可选的 GPU 近邻图分块大小，默认目标约 2GB/批。
- `device`：`"auto"`、`"cpu"`、`"cuda"` 或 `"torch"`。

## CPU+GPU 示例

```python
import numpy as np
from statgpu.unsupervised import DBSCAN

X = np.random.default_rng(0).normal(size=(5000, 8))

# CPU（低维：Cython 快速路径）
labels_cpu = DBSCAN(eps=1.0, min_samples=5, device="cpu").fit_predict(X)

# GPU（PyTorch CUDA：全在设备上）
labels_torch = DBSCAN(eps=1.0, min_samples=5, device="torch").fit_predict(X)

# GPU（CuPy：距离在 GPU 计算，标签经 Cython 在 CPU 生成）
labels_cuda = DBSCAN(eps=1.0, min_samples=5, device="cuda", batch_size=1024).fit_predict(X)
```

## 性能

Tesla P100-SXM2-16GB（GPU）与 Intel Xeon（CPU），取 3 次运行的中位数：

| n | p | sklearn CPU | statgpu CPU | statgpu GPU (torch) | GPU / sklearn |
|---|---|---|---|---|---|
| 10000 | 5 | 0.46s | 0.18s | 0.03s | **0.06x** |
| 30000 | 5 | 3.32s | 1.35s | 0.24s | **0.07x** |
| 50000 | 5 | 9.49s | 3.88s | 0.71s | **0.07x** |
| 10000 | 50 | 0.05s | 0.06s | 0.01s | **0.28x** |
| 30000 | 50 | 0.39s | 0.32s | 0.12s | **0.30x** |
| 50000 | 50 | 1.08s | 0.89s | 0.32s | **0.30x** |

所有情形下 ARI = 1.0000，与 sklearn 参考实现完全一致。

## 严格与近似模式的差别

DBSCAN 没有统计推断意义上的严格模式：对受支持的稠密欧氏输入，纯 Python 回退与 Cython 快速路径都给出精确结果。GPU 路径计算相同的稠密近邻关系，但在 `eps` 边界附近仍可能受浮点比较影响。

## 输出字段

- `labels_`
- `core_sample_indices_`
- `components_`
- `n_features_in_`

## FAQ

**生产 DBSCAN 会调用 sklearn 吗？**
CPU 路径中 p > 12 时，用 sklearn 的 `NearestNeighbors` 做基于 BLAS 的距离计算；图处理和标签分配由 statgpu 的 Cython 代码完成。p ≤ 12 时不依赖 sklearn。

**什么时候使用 Cython？**
当 `_dbscan_cy_fast` 扩展已编译时（`python setup.py build_ext --inplace`）；没有 Cython 时使用纯 Python 回退。Cython 模块需要在目标机器上编译。

**为什么 GPU 路径更快？**
GPU 路径把所有中间数据（距离、边、标签）留在设备上；连通分量的标签传播在 GPU 上完全并行，而 CPU 的并查集必须逐条处理边。只有最终标签传回 CPU。

## 外部验证

- 测试脚本：`dev/tests/test_unsupervised_dbscan.py`。
- 基准测试：`dev/benchmarks/benchmark_unsupervised_dbscan_cython.py`。
- 对齐基线：sklearn DBSCAN，并对齐 `eps`、`min_samples` 与欧氏距离。
- 标签与噪声掩码都对照对齐后的参考实现检查（ARI = 1.0）。

## References

- Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. In *Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD-96)* (pp. 226-231). AAAI Press. https://aaai.org/papers/kdd96-037-a-density-based-algorithm-for-discovering-clusters-in-large-spatial-databases-with-noise/
- Schubert, E., Sander, J., Ester, M., Kriegel, H.-P., & Xu, X. (2017). DBSCAN revisited, revisited: Why and how you should (still) use DBSCAN. *ACM Transactions on Database Systems*, 42(3), Article 19. https://doi.org/10.1145/3068335
