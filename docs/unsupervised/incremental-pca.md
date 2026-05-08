# IncrementalPCA

> 语言：中文
> 最后更新：2026-05-07
> English: [English](../en/unsupervised/incremental-pca.md)

## 概览

`IncrementalPCA` 从 dense mini-batches 中拟合主成分，同时维护 running mean、variance、样本数和截断 SVD basis。当前支持 CPU、CuPy/CUDA 和 Torch CUDA。

## 导入路径

```python
from statgpu.unsupervised import IncrementalPCA
```

## 目标函数 / 损失函数

对 `k` 个 components，IncrementalPCA 近似 centered rank-k PCA 目标：

$$
\min_{V_k^\top V_k=I}
\left\|X - \bar{X} - (X-\bar{X})V_kV_k^\top\right\|_F^2 .
$$

## 估计方程

每次 `partial_fit` 先更新 batch mean/variance，再把历史低秩 basis、当前 centered batch 和 mean-correction row 合并成一个 compact matrix 做 SVD，取前 `n_components` 个右奇异向量作为 `components_`。

## 参数

- `n_components`：保留的 components 数；`None` 使用 `n_features`。
- `batch_size`：`fit` 内部使用的 batch size；`partial_fit` 接收调用方提供的 batch。
- `whiten`：按 explained variance 缩放 transform 结果。
- `copy`：sklearn 风格兼容参数。
- `device`：`"auto"`、`"cpu"`、`"cuda"` 或 `"torch"`。

## CPU+GPU 示例

```python
from statgpu.unsupervised import IncrementalPCA

ipca = IncrementalPCA(n_components=8, batch_size=1024, device="cuda")
ipca.fit(X)
Z = ipca.transform(X)
X_hat = ipca.inverse_transform(Z)
```

## strict/approx 差异

IncrementalPCA 是 batch/streaming 近似估计器，结果会受 batch order 和 batch size 影响。相同 batch 设置下 CPU/CuPy/Torch 应保持一致。

## 输出字段

- `components_`
- `mean_`
- `var_`
- `explained_variance_`
- `explained_variance_ratio_`
- `singular_values_`
- `n_components_`
- `n_features_in_`
- `n_samples_seen_`

## FAQ

**v1 支持 sparse input 吗？**
不支持。Phase 3C 仅支持 dense 2D float arrays。

## 外部验证

- 测试：`dev/tests/test_unsupervised_incremental_pca.py`。
- Benchmark：`dev/benchmarks/benchmark_unsupervised_phase3c.py`。
- 最新远程 artifact：`results/unsupervised_phase3c_opt7_20260507_185500.json`。
- Baseline：sklearn `IncrementalPCA`，对齐 `n_components` 和 `batch_size`。

## References

- Ross, D. A., Lim, J., Lin, R.-S., & Yang, M.-H. (2008). Incremental learning for robust visual tracking. *International Journal of Computer Vision*, 77, 125-141. https://doi.org/10.1007/s11263-007-0075-7
- scikit-learn Developers. `sklearn.decomposition.IncrementalPCA`. scikit-learn documentation. https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html
