# TruncatedSVD

> 语言：中文
> 最后更新：2026-05-09
> 路径：`statgpu.unsupervised.TruncatedSVD`

## 概览

`TruncatedSVD` 在不中心化输入矩阵的情况下计算低秩投影，因此不同于 `PCA`，也更适合 dense LSA 类流程。

## 导入路径

从 `statgpu.unsupervised` 导入：

```python
from statgpu.unsupervised import TruncatedSVD
```

## 目标函数

给定 rank `k`，Truncated SVD 求解：

$$
\min_{\operatorname{rank}(Z) \le k} \|X - Z\|_F^2.
$$

## 估计方程

exact 路径计算：

$$
X = U \Sigma V^\top.
$$

randomized 路径先把 `X` 投影到随机低维子空间，每轮 power iteration 后重新正交化，并使用确定性的 component sign convention，再对小矩阵做 SVD。

## 参数

`n_components`、`algorithm`、`n_iter`、`n_oversamples`、`random_state`、`device`。

## CPU+GPU 示例

```python
from statgpu.unsupervised import TruncatedSVD

Z = TruncatedSVD(n_components=10, device="cpu").fit_transform(X)
Z_gpu = TruncatedSVD(n_components=10, device="cuda").fit_transform(X_gpu)
```

## Strict/Approx Difference

`algorithm="full"` 是 dense exact SVD；`algorithm="randomized"` 是近似算法，比较时应使用 sign/subspace invariant 指标。

## 输出

`components_`、`explained_variance_`、`explained_variance_ratio_`、`singular_values_`、`n_components_`、`n_features_in_`。

## FAQ

Phase 3A 不支持 sparse input 和 ARPACK。

## 外部验证

测试：`dev/tests/test_unsupervised_truncated_svd.py`。
Benchmark：`dev/benchmarks/benchmark_unsupervised_phase3.py`。
Baseline：sklearn `TruncatedSVD`、statsmodels PCA-style SVD、可用时 R `svd`。

## References

- Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. *SIAM Review*, 53(2), 217-288.
