# DBSCAN

> 语言：中文
> 最后更新：2026-06-26
> English: [English](../en/unsupervised/dbscan.md)

## 概览

`DBSCAN` 在 dense Euclidean data 上寻找 density-connected components。CPU 路径使用 Cython 加速 pipeline，低维数据比 sklearn 快 3-4 倍，高维数据与 sklearn 持平。GPU 路径（PyTorch CUDA）全在设备上执行，零 GPU→CPU 传输，比 sklearn 快 3-17 倍。

## 导入路径

```python
from statgpu.unsupervised import DBSCAN
```

## 目标函数 / 损失函数

DBSCAN 不是光滑优化问题，没有可微损失函数。它的准则是 density reachability：

- 如果一个点的闭合 `eps` 邻域内至少有 `min_samples` 个点，则它是 core point。
  $$
  \left|\left\{x_j : \left\|x_i - x_j\right\|_2 \le \varepsilon\right\}\right|
  \ge \text{min\_samples}.
  $$
- 由 `eps` 邻接链连通的 core points 组成一个 cluster。
- 能从 core component 到达的非 core points 是 border points。
- 其他点是 noise，标签为 `-1`。

## CPU 策略

CPU 路径根据维度选择算法：

| 维度 | 策略 | 说明 |
|---|---|---|
| p ≤ 12 | cKDTree `query_pairs` + Cython | 单次树遍历；`dbscan_labels_from_pairs` 在 C 中完成计数、Union-Find 和标签分配。 |
| p > 12 | sklearn `radius_neighbors_graph` + Cython | 使用 sklearn 优化 BLAS 计算距离；`dbscan_labels_from_csr` 在 C 中处理 CSR 图。 |

两条路径在 Cython 扩展未编译时都有纯 Python fallback。

### Cython 模块：`_dbscan_cy_fast.pyx`

两个入口函数，均在 C 中运行完整标签 pipeline（无 Python 对象开销）：

- `dbscan_labels_from_pairs(n_samples, min_samples, pairs)` — 接收 `query_pairs` 的原始 `(i, j)` 对。
- `dbscan_labels_from_csr(n_samples, min_samples, indptr, indices)` — 接收 CSR 稀疏图数组。

内部均使用：
- C 级邻居计数
- C 级 Union-Find（路径压缩 + 按秩合并）
- C 级 border 点分配

## GPU 策略（PyTorch CUDA）

GPU 路径所有数据留在设备上：

1. **距离计算**：GPU 上批量 `float32` matmul
2. **邻居计数**：GPU 上 `mask.sum(dim=1)`
3. **稀疏图**：GPU 上 `torch.nonzero`，边存为 GPU tensor
4. **Connected components**：GPU 上 label propagation（`scatter_reduce_(amin)`）
5. **Border 分配**：GPU 上批量距离 + scatter

只有最终标签（`n × int64`）传回 CPU。消除了逐 batch GPU→CPU 传输开销，避免重复计算距离导致的 OOM。

### Label Propagation 算法

```
labels = arange(n_core)                          # 每个 core 点初始独立
for _ in range(50):                              # 通常 2-5 次迭代收敛
    min_labels = minimum(labels[src], labels[dst])  # 全边并行
    labels.scatter_reduce_(amin)                     # 并行 scatter
    if converged: break
```

非常适合 GPU：每次迭代对所有边完全并行，不像 CPU Union-Find 顺序处理每条边。

## 参数

- `eps`：邻域半径，必须为正。
- `min_samples`：成为 core sample 所需的闭合邻域样本数。
- `metric`：仅支持 `"euclidean"`。
- `batch_size`：可选 GPU neighbor graph chunk size。默认目标 ~2GB/batch。
- `device`：`"auto"`、`"cpu"`、`"cuda"` 或 `"torch"`。

## CPU+GPU 示例

```python
import numpy as np
from statgpu.unsupervised import DBSCAN

X = np.random.default_rng(0).normal(size=(5000, 8))

# CPU（低维：Cython fast path）
labels_cpu = DBSCAN(eps=1.0, min_samples=5, device="cpu").fit_predict(X)

# GPU（PyTorch CUDA：全在设备上）
labels_torch = DBSCAN(eps=1.0, min_samples=5, device="torch").fit_predict(X)

# GPU（CuPy：距离在 GPU，标签在 CPU via Cython）
labels_cuda = DBSCAN(eps=1.0, min_samples=5, device="cuda", batch_size=1024).fit_predict(X)
```

## 性能

Tesla P100-SXM2-16GB（GPU）和 Intel Xeon（CPU），3 次中位数：

| n | p | sklearn CPU | statgpu CPU | statgpu GPU (torch) | GPU / sklearn |
|---|---|---|---|---|---|
| 10000 | 5 | 0.46s | 0.18s | 0.03s | **0.06x** |
| 30000 | 5 | 3.32s | 1.35s | 0.24s | **0.07x** |
| 50000 | 5 | 9.49s | 3.88s | 0.71s | **0.07x** |
| 10000 | 50 | 0.05s | 0.06s | 0.01s | **0.28x** |
| 30000 | 50 | 0.39s | 0.32s | 0.12s | **0.30x** |
| 50000 | 50 | 1.08s | 0.89s | 0.32s | **0.30x** |

所有情况 ARI = 1.0000（与 sklearn reference 完全一致）。

## strict/approx 差异

DBSCAN 没有 strict inference 模式。CPU fallback 和 Cython fast path 对支持的 dense Euclidean 输入是 exact。GPU 路径计算相同 dense neighbor relation，但在 `eps` 边界附近仍受浮点比较影响。

## 输出字段

- `labels_`
- `core_sample_indices_`
- `components_`
- `n_features_in_`

## FAQ

**生产 DBSCAN 会调用 sklearn 吗？**
CPU 路径中 p > 12 时，使用 sklearn 的 `NearestNeighbors` 进行优化 BLAS 距离计算。图处理和标签分配由 statgpu Cython 代码完成。p ≤ 12 时无 sklearn 依赖。

**什么时候使用 Cython？**
当 `_dbscan_cy_fast` 扩展已编译时（`python setup.py build_ext --inplace`）。无 Cython 时使用纯 Python fallback。Cython 模块需在目标机器上编译。

**为什么 GPU 路径更快？**
GPU 路径所有中间数据（距离、边、标签）留在设备上。Connected components 的 label propagation 在 GPU 上完全并行，不像 CPU Union-Find 顺序处理每条边。只有最终标签传回 CPU。

## 外部验证

- 测试：`dev/tests/test_unsupervised_dbscan.py`。
- Benchmark：`dev/benchmarks/benchmark_unsupervised_dbscan_cython.py`。
- Baseline：sklearn DBSCAN，对齐 `eps`、`min_samples` 和 Euclidean metric。
- Labels 和 noise mask 与对齐 reference 检查（ARI = 1.0）。

## References

- Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. In *Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD-96)* (pp. 226-231). AAAI Press. https://aaai.org/papers/kdd96-037-a-density-based-algorithm-for-discovering-clusters-in-large-spatial-databases-with-noise/
- Schubert, E., Sander, J., Ester, M., Kriegel, H.-P., & Xu, X. (2017). DBSCAN revisited, revisited: Why and how you should (still) use DBSCAN. *ACM Transactions on Database Systems*, 42(3), Article 19. https://doi.org/10.1145/3068335
