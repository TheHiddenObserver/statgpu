# DBSCAN

> 语言：中文
> 最后更新：2026-05-02
> English: [English](../en/unsupervised/dbscan.md)

## 概览

`DBSCAN` 在 dense Euclidean data 上寻找 density-connected components。它支持 CPU、CuPy/CUDA 和 Torch CUDA。CPU 路径包含精确 NumPy/SciPy fallback，并为 compact dense 场景提供可选的 statgpu 自有 Cython fast path。

## 导入路径

```python
from statgpu.unsupervised import DBSCAN
```

## 目标函数 / 损失函数

DBSCAN 不是光滑优化问题，没有可微损失函数。它的准则是 density reachability：

- 如果一个点的闭合 `eps` 邻域内至少有 `min_samples` 个点，则它是 core point。
- 由 `eps` 邻接链连通的 core points 组成一个 cluster。
- 能从 core component 到达的非 core points 是 border points。
- 其他点是 noise，标签为 `-1`。

## 估计方程

- CPU 先用 `cKDTree` 估计邻域密度。
- compact dense CPU 输入可使用 `_dbscan_cpu.pyx`，扫描 pairwise distances 并 union core-core neighbor pairs。
- CPU fallback 使用 SciPy/NumPy exact 策略：根据密度和内存选择 condensed `pdist`、sparse distance matrix 或 `query_pairs`。
- CuPy/Torch 路径分 batch 构造 dense boolean neighbor graph，识别 core samples，在 core graph 上传播 connected component labels，再分配 border samples。

## 参数

- `eps`：邻域半径，必须为正。
- `min_samples`：成为 core sample 所需的闭合邻域样本数。
- `metric`：仅支持 `"euclidean"`。
- `batch_size`：可选 GPU neighbor graph chunk size。
- `device`：`"auto"`、`"cpu"`、`"cuda"` 或 `"torch"`。

## CPU+GPU 示例

```python
import numpy as np
from statgpu.unsupervised import DBSCAN

X = np.random.default_rng(0).normal(size=(5000, 8))

labels_cpu = DBSCAN(eps=1.0, min_samples=5, device="cpu").fit_predict(X)
labels_cuda = DBSCAN(eps=1.0, min_samples=5, device="cuda", batch_size=1024).fit_predict(X)
```

## strict/approx 差异

DBSCAN 没有 strict inference 模式。CPU fallback 和 Cython fast path 对支持的 dense Euclidean 输入是 exact。GPU 路径计算相同 dense neighbor relation，但在 `eps` 边界附近仍受浮点比较影响。

## 输出字段

- `labels_`
- `core_sample_indices_`
- `components_`
- `n_features_in_`

## FAQ

**生产 DBSCAN 会调用 sklearn 吗？**
不会。sklearn 只用于测试和 benchmark 外部 baseline。

**什么时候使用 Cython？**
只有在可选扩展已编译，且 CPU selector 判断输入是 compact dense 场景时使用。variable-density、sparse/all-noise 和无编译器环境使用 fallback。

**为什么 Cython 仍可能慢于 sklearn？**
Cython 路径是 statgpu 自有实现，目前在 compact dense 场景接近 sklearn，但不总是更快。最新 `n=5000` 运行是 sklearn CPU 的 `1.23x`，因此没有记录为严格速度达标。

## 外部验证

- 测试：`dev/tests/test_unsupervised_dbscan.py`。
- Benchmark：`dev/benchmarks/benchmark_unsupervised_phase2.py` 和 `dev/benchmarks/benchmark_unsupervised_dbscan_cython.py`。
- Baseline：sklearn DBSCAN，对齐 `eps`、`min_samples` 和 Euclidean metric。
- 最新结果：compact `n=5000` Cython CPU `219.62ms`，fallback CPU `379.80ms`，sklearn CPU `178.94ms`，CuPy `21.56ms`，Torch `21.07ms`；labels 的 ARI 为 `1.0`，noise mask 一致。

## References

- Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise.
