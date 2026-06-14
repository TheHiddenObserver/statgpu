# UMAP

> 语言：中文
> 最后更新：2026-05-09
> 路径：`statgpu.unsupervised.UMAP`

## 概览

`UMAP` 在输入空间构造 fuzzy neighbor graph，并优化低维 embedding。Phase 3A 实现 dense exact Euclidean 路径。

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

statgpu 先计算 exact pairwise distance，再选择 `n_neighbors` 个邻居，构造对称 fuzzy membership graph，最后对 embedding 做梯度更新。

## 参数

`n_neighbors`、`n_components`、`metric`、`min_dist`、`spread`、`n_epochs`、`learning_rate`、`init`、`negative_sample_rate`、`repulsion_strength`、`random_state`、`device`。

## CPU+GPU 示例

```python
from statgpu.unsupervised import UMAP

embedding = UMAP(n_neighbors=15, device="cpu").fit_transform(X)
embedding_gpu = UMAP(n_neighbors=15, device="cuda").fit_transform(X_gpu)
```

## Strict/Approx Difference

v1 对 dense Euclidean neighbor search 是 exact，但相对 `umap-learn` 做了简化：不实现 NNDescent 和完整 sparse graph pipeline。

## 输出

`embedding_`、`graph_`、`n_epochs_`、`n_features_in_`。

## FAQ

Phase 3A 不支持 sparse、非 Euclidean metric、approximate neighbor search 和新样本 `transform`。

## 外部验证

测试：`dev/tests/test_unsupervised_umap.py`。
Benchmark：`dev/benchmarks/benchmark_unsupervised_phase3.py`。
Baseline：`umap-learn`，以及远程可用时的 cuML UMAP。

## References

- McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. *arXiv:1802.03426*.
