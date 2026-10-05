# TruncatedSVD

> 语言：中文
> 最后更新：2026-09-29
> 路径：`statgpu.unsupervised.TruncatedSVD`

## 概览

`TruncatedSVD` 在不对输入矩阵做中心化的前提下计算低秩投影；这一点与 `PCA` 不同，也更适合稠密的 LSA 类流程。

## 导入路径

从 `statgpu.unsupervised` 导入：

```python
from statgpu.unsupervised import TruncatedSVD
```

## 目标函数

给定秩 `k`，Truncated SVD 求解：

$$
\min_{\operatorname{rank}(Z) \le k} \|X - Z\|_F^2.
$$

## 估计方程

精确路径计算：

$$
X = U \Sigma V^\top.
$$

随机化路径先把 `X` 投影到随机的低维子空间，每轮幂迭代后重新正交化，并使用确定性的主成分符号约定，最后对一个小矩阵做 SVD。

## 参数

`n_components`、`algorithm`、`n_iter`、`n_oversamples`、`random_state`、`device`。

## CPU+GPU 示例

```python
from statgpu.unsupervised import TruncatedSVD

Z = TruncatedSVD(n_components=10, device="cpu").fit_transform(X)
Z_gpu = TruncatedSVD(n_components=10, device="cuda").fit_transform(X_gpu)
```

## 严格与近似模式的差别

`algorithm="full"` 是稠密数据上的精确 SVD；`algorithm="randomized"` 是近似算法，比较时应使用对符号和子空间不变的指标。

## 输出

`components_`、`explained_variance_`、`explained_variance_ratio_`、`singular_values_`、`n_components_`、`n_features_in_`。

## FAQ

Phase 3A 不支持稀疏输入和 ARPACK。

## 外部验证

测试脚本：`dev/tests/test_unsupervised_truncated_svd.py`。
基准测试：`dev/benchmarks/benchmark_unsupervised_phase3.py`。
对齐基线：sklearn 的 `TruncatedSVD`、statsmodels 的 PCA 式 SVD，以及可用时的 R `svd`。

## References

- Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. *SIAM Review*, 53(2), 217-288.

## 完整 API 参考

构造默认值、全部公开方法、输出形状与限制见 [TruncatedSVD API 参考](api-reference.md#truncatedsvd)。
