# UMAP

> 语言：中文
> 最后更新：2026-07-23
> 路径：`statgpu.unsupervised.UMAP`

## 概览

`UMAP` 在输入空间构造 fuzzy neighbor graph，并优化低维 embedding。它支持 dense exact Euclidean 邻居，以及内部 NNDescent 邻居搜索选项。

## 后端与主机边界

距离计算、邻居搜索、membership 权重、embedding 优化和负采样均在所选 NumPy、CuPy 或 Torch 后端执行。当前 fuzzy-union graph assembly 是明确披露的主机边界：O(n*k) 的 edge indices 和 weights 会复制到主机内存，通过 SciPy sparse COO/CSR 完成组装，再复制回所选后端。这不是 optimization 的静默 CPU fallback，但尚不是 device-native sparse graph path。exact neighbor 还需要 O(n^2) dense distance 内存；当可接受 approximate-neighbor 取舍时，可使用 `nn_method='nndescent'` 避免该 distance matrix。

## 导入路径

从 `statgpu.unsupervised` 导入：

```python
from statgpu.unsupervised import UMAP
```

## 目标函数

UMAP 优化高维图权重 `w_ij` 与低维 affinity `q_ij` 之间的 fuzzy-set cross-entropy：

$$
\sum_{i,j} w_{ij}\log\frac{w_{ij}}{q_{ij}}
+ (1-w_{ij})\log\frac{1-w_{ij}}{1-q_{ij}}.
$$

## 估计方程

默认通过 dense exact search 选择 `n_neighbors` 个邻居（`nn_method='auto'` 会解析为 `exact`）；也可显式请求内部 NNDescent。随后构造对称 fuzzy membership graph，并对 embedding 做梯度更新。

## 参数

`n_neighbors`、`n_components`、`metric`、`min_dist`、`spread`、`n_epochs`、`learning_rate`、`init`、`negative_sample_rate`、`repulsion_strength`、`random_state`、`device`。

## CPU+GPU 示例

```python
from statgpu.unsupervised import UMAP

embedding = UMAP(n_neighbors=15, device="cpu").fit_transform(X)
embedding_gpu = UMAP(n_neighbors=15, device="cuda").fit_transform(X_gpu)
```

## Strict/Approx Difference

`nn_method='exact'` 对 dense Euclidean neighbor search 是 exact。`nn_method='nndescent'` 是 approximate 且 backend-aware。两种模式均使用上述 SciPy host-side fuzzy-union boundary；完整 device-native sparse graph pipeline 尚未实现。

## 输出

`embedding_`、`graph_`、`n_epochs_`、`n_features_in_`。

## FAQ

不支持 sparse、非 Euclidean metric 和新样本 `transform`。通过 `nn_method='nndescent'` 支持 approximate neighbor；graph assembly 仍需要 SciPy 与 host memory。

## 外部验证

测试：`dev/tests/test_unsupervised_umap.py`。
Benchmark：`dev/benchmarks/benchmark_unsupervised_phase3.py`。
Baseline：`umap-learn`，以及远程可用时的 cuML UMAP。

## References

- McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. *arXiv:1802.03426*.
