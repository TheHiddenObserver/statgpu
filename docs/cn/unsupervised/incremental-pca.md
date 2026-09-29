# IncrementalPCA

> 语言：中文
> 最后更新：2026-09-29
> 切换：[English](../../en/unsupervised/incremental-pca.md)

## 概览

`IncrementalPCA` 在稠密的小批量（mini-batch）数据上拟合主成分，同时维护累计均值、方差、样本数以及截断 SVD 基。当前支持 CPU、CuPy/CUDA 与 Torch CUDA。

## 导入路径

```python
from statgpu.unsupervised import IncrementalPCA
```

## 目标函数 / 损失函数

对 `k` 个主成分，`IncrementalPCA` 近似求解中心化的 rank-k PCA 目标：

$$
\min_{V_k^\top V_k=I}
\left\|X - \bar{X} - (X-\bar{X})V_kV_k^\top\right\|_F^2 .
$$

## 估计方程

每次 `partial_fit` 先更新该批的均值与方差，再把历史的低秩基、当前中心化批次和均值校正行合并为一个紧凑矩阵做 SVD，取前 `n_components` 个右奇异向量作为 `components_`。

## 参数

- `n_components`：保留的主成分个数；为 `None` 时使用 `n_features`。
- `batch_size`：`fit` 内部使用的批次大小；`partial_fit` 使用调用方传入的批次。
- `whiten`：按解释方差缩放 `transform` 的结果。
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

## 严格与近似模式的差别

`IncrementalPCA` 是批次/流式近似估计器，结果会受批次顺序和批次大小影响；在相同的批次设置下，CPU/CuPy/Torch 的结果应保持一致。

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

**v1 支持稀疏输入吗？**
不支持。Phase 3C 仅支持稠密的二维浮点数组。

## 外部验证

- 测试脚本：`dev/tests/test_unsupervised_incremental_pca.py`。
- 基准测试：`dev/benchmarks/benchmark_unsupervised_phase3c.py`。
- 最新远程验证产物：`results/unsupervised_phase3c_opt7_20260507_185500.json`。
- 对齐基线：sklearn 的 `IncrementalPCA`，对齐 `n_components` 与 `batch_size`。

## References

- Ross, D. A., Lim, J., Lin, R.-S., & Yang, M.-H. (2008). Incremental learning for robust visual tracking. *International Journal of Computer Vision*, 77, 125-141. https://doi.org/10.1007/s11263-007-0075-7
- scikit-learn Developers. `sklearn.decomposition.IncrementalPCA`. scikit-learn documentation. https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html
