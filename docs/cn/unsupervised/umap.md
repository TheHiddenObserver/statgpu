# UMAP

> 语言：中文
> 最后更新：2026-09-29
> 路径：`statgpu.unsupervised.UMAP`

## 概览

`UMAP` 在输入空间构造模糊近邻图（fuzzy neighbor graph），并优化低维嵌入（embedding）。它既支持稠密、精确的欧氏近邻，也支持内置的 NNDescent 近似近邻搜索。

## 后端与主机边界

距离计算、近邻搜索、隶属度权重、嵌入优化和负采样都在所选的 NumPy、CuPy 或 Torch 后端上完成。目前明确披露的主机边界是模糊并集图（fuzzy-union graph）的组装：O(n*k) 的边索引和边权重会复制到主机内存，由 SciPy 的稀疏 COO/CSR 结构完成组装，再复制回所选后端。这不是优化过程的静默 CPU 回退，但也不是完全后端原生的稀疏图流水线。精确近邻还需要 O(n²) 的稠密距离矩阵内存；如果可以接受近似近邻的取舍，可用 `nn_method='nndescent'` 避开这个矩阵。

## 导入路径

从 `statgpu.unsupervised` 导入：

```python
from statgpu.unsupervised import UMAP
```

## 目标函数

UMAP 最小化高维图权重 `w_ij` 与低维亲和度 `q_ij` 之间的模糊集交叉熵：

$$
\sum_{i,j} w_{ij}\log\frac{w_{ij}}{q_{ij}}
+ (1-w_{ij})\log\frac{1-w_{ij}}{1-q_{ij}}.
$$

## 估计方程

默认用稠密、精确的搜索选出 `n_neighbors` 个近邻（`nn_method='auto'` 解析为 `exact`）；也可以显式请求内置的 NNDescent。随后构造对称的模糊隶属度图，并对嵌入做梯度更新。

## 参数

`n_neighbors`、`n_components`、`metric`、`min_dist`、`spread`、`n_epochs`、`learning_rate`、`init`、`negative_sample_rate`、`repulsion_strength`、`random_state`、`device`。

## CPU+GPU 示例

```python
from statgpu.unsupervised import UMAP

embedding = UMAP(n_neighbors=15, device="cpu").fit_transform(X)
embedding_gpu = UMAP(n_neighbors=15, device="cuda").fit_transform(X_gpu)
```

## 严格与近似模式的差别

`nn_method='exact'` 对稠密欧氏近邻搜索给出精确结果；`nn_method='nndescent'` 是近似搜索，并按所选后端执行。两种模式都要经过上述基于 SciPy 的主机侧模糊并集组装；完全后端原生的稀疏图流水线尚未实现。

## 输出

`embedding_`、`graph_`、`n_epochs_`、`n_features_in_`。

## FAQ

不支持稀疏输入、非欧氏 `metric`，也不支持对新样本调用 `transform`。近似近邻可以通过 `nn_method='nndescent'` 启用；图的组装仍然需要 SciPy 和主机内存。

## 外部验证

测试脚本：`dev/tests/test_unsupervised_umap.py`。
基准测试：`dev/benchmarks/benchmark_unsupervised_phase3.py`。
对齐基线：`umap-learn`，以及远程环境可用时的 cuML UMAP。

## References

- McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. *arXiv:1802.03426*.
