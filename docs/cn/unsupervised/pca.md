# PCA

> 语言：中文
> 最后更新：2026-05-02
> English: [English](../en/unsupervised/pca.md)

## 概览

`PCA` 为 centered dense data 估计一组正交低维基，捕捉最大方差方向。它支持 CPU、CuPy/CUDA 和 Torch CUDA。

## 导入路径

```python
from statgpu.unsupervised import PCA
```

## 目标函数 / 损失函数

对中心化数据 `X_c = X - mean(X)`，PCA 求解：

$$
\begin{aligned}
\max_{W \in \mathbb{R}^{p \times k}} \quad
& \operatorname{tr}\left(W^\top X_c^\top X_c W\right) \\
\text{s.t.} \quad
& W^\top W = I_k .
\end{aligned}
$$

保留 `k` 个 components 时，也等价于在正交投影中最小化 rank-`k` 平方重构误差：

$$
\begin{aligned}
\min_{W \in \mathbb{R}^{p \times k}} \quad
& \left\|X_c - X_c W W^\top\right\|_F^2 \\
\text{s.t.} \quad
& W^\top W = I_k .
\end{aligned}
$$

二者等价，因为中心化后的总方差是固定的。

## 估计方程

- `svd_solver="covariance"` 计算
  $$
  \Sigma = \frac{X_c^\top X_c}{n - 1}
  $$
  再用 `eigh` 求解
  $$
  \Sigma v_j = \lambda_j v_j .
  $$
- `svd_solver="full"` 计算
  $$
  X_c = U S V^\top
  $$
  使用 `V.T` 的行作为 components。
- `svd_solver="auto"` 在 `n_samples >= n_features` 时使用 covariance/eigh，否则使用 full SVD。
- `svd_solver="randomized"` 使用随机投影、power iteration 和小矩阵 SVD 近似 leading right singular vectors。
- explained variance 计算为
  $$
  \operatorname{explained\_variance}_j = \frac{s_j^2}{n - 1}.
  $$
- `explained_variance_ratio_` 是 retained variance 除以 centered total variance。

## 参数

- `n_components`：保留的 components 数量；`None` 保留所有可行 components。
- `svd_solver`：`"auto"`、`"full"`、`"covariance"` 或 `"randomized"`。
- `whiten`：为 `True` 时，transform 后的 scores 会除以 `sqrt(explained_variance_)`。
- `random_state`、`n_oversamples`、`iterated_power`：randomized solver 控制参数。
- `device`：`"auto"`、`"cpu"`、`"cuda"` 或 `"torch"`。

## CPU+GPU 示例

```python
import numpy as np
from statgpu.unsupervised import PCA

X = np.random.default_rng(0).normal(size=(2000, 50))

pca_cpu = PCA(n_components=10, svd_solver="covariance", device="cpu")
Z_cpu = pca_cpu.fit_transform(X)

pca_gpu = PCA(n_components=10, svd_solver="covariance", device="cuda")
Z_gpu = pca_gpu.fit_transform(X)
```

## strict/approx 差异

PCA 没有统计推断意义上的 strict inference 模式。这里的 exact/approx 指分解算法：

- `full` 和 `covariance` 是 dense 输入上的 exact solver，误差来自浮点计算。
- `randomized` 是 approximate truncated SVD，由 `random_state`、`n_oversamples` 和 `iterated_power` 控制。
- component 符号不可识别，`v` 和 `-v` 表示同一主成分。

## 输出字段

- `components_`
- `mean_`
- `explained_variance_`
- `explained_variance_ratio_`
- `singular_values_`
- `n_components_`
- `n_features_in_`

## FAQ

**为什么 components 和 sklearn 差一个符号？**
Eigenvector 和 singular vector 的符号不唯一。验证时应使用 sign-aware comparison 或比较子空间。

**whitening 做了什么？**
它把 transformed scores 按 `1 / sqrt(explained_variance_)` 缩放，使拟合模型下的 component scores 近似单位方差。

## 外部验证

- 测试：`dev/tests/test_unsupervised_pca.py`。
- Benchmark：`dev/benchmarks/benchmark_unsupervised.py`。
- Baseline：sklearn PCA，以及早期 unsupervised matrix 中可用的 statsmodels/R PCA 对比。
- 最新 Phase 2 摘要：`results/unsupervised_phase2_verify_summary_20260502_210000.md`。

## References

- Pearson, K. (1901). On lines and planes of closest fit to systems of points in space. *The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science*, Series 6, 2(11), 559-572. https://doi.org/10.1080/14786440109462720
- Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer Series in Statistics. Springer. https://doi.org/10.1007/b98835
- Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. *SIAM Review*, 53(2), 217-288. https://doi.org/10.1137/090771806
