# PCA

> 语言：中文
> 最后更新：2026-09-29
> 切换：[English](../../en/unsupervised/pca.md)

## 概览

`PCA` 对已中心化的稠密数据估计一组正交低维基，使其捕捉方差最大的方向。支持 CPU、CuPy/CUDA 与 Torch CUDA。

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

保留 `k` 个主成分时，也等价于在正交投影约束下最小化秩为 `k` 的平方重构误差：

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

- `svd_solver="covariance"` 先计算
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
  并把 `V.T` 的各行作为主成分。
- `svd_solver="auto"` 在 `n_samples >= n_features` 时使用协方差矩阵配合 `eigh`，否则使用完整 SVD。
- `svd_solver="randomized"` 通过随机投影、幂迭代（power iteration）和一个小矩阵的 SVD 近似前若干个右奇异向量。
- 解释方差（explained variance）计算为
  $$
  \operatorname{explained\_variance}_j = \frac{s_j^2}{n - 1}.
  $$
- `explained_variance_ratio_` 是保留方差除以中心化后的总方差。

## 参数

- `n_components`：保留的主成分个数；为 `None` 时保留所有可行的主成分。
- `svd_solver`：`"auto"`、`"full"`、`"covariance"` 或 `"randomized"`。
- `whiten`：为 `True` 时，`transform` 后的得分会除以 `sqrt(explained_variance_)`。
- `random_state`、`n_oversamples`、`iterated_power`：控制随机化 SVD 的参数。
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

## 严格与近似模式的差别

`PCA` 没有统计推断意义上的严格推断模式；这里的“精确/近似”指的是分解算法：

- `full` 和 `covariance` 在稠密输入上是精确求解器，误差仅来自浮点运算。
- `randomized` 是近似的截断 SVD，由 `random_state`、`n_oversamples` 和 `iterated_power` 控制。
- 主成分的符号不可识别：`v` 与 `-v` 表示同一个主成分。

## 输出字段

- `components_`
- `mean_`
- `explained_variance_`
- `explained_variance_ratio_`
- `singular_values_`
- `n_components_`
- `n_features_in_`

## FAQ

**为什么主成分和 sklearn 差一个符号？**
特征向量与奇异向量的符号并不唯一；验证时应使用符号感知的比较，或直接比较子空间。

**白化（whitening）做了什么？**
它把变换后的得分按 `1 / sqrt(explained_variance_)` 缩放，使拟合模型下的主成分得分近似具有单位方差。

## 外部验证

- 测试脚本：`dev/tests/test_unsupervised_pca.py`。
- 基准测试：`dev/benchmarks/benchmark_unsupervised.py`。
- 对齐基线：sklearn 的 PCA，以及早期无监督方法矩阵中可用的 statsmodels/R PCA 对比。
- 最新 Phase 2 摘要：`results/unsupervised_phase2_verify_summary_20260502_210000.md`。

## References

- Pearson, K. (1901). On lines and planes of closest fit to systems of points in space. *The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science*, Series 6, 2(11), 559-572. https://doi.org/10.1080/14786440109462720
- Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer Series in Statistics. Springer. https://doi.org/10.1007/b98835
- Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. *SIAM Review*, 53(2), 217-288. https://doi.org/10.1137/090771806

## 完整 API 参考

构造默认值、全部公开方法、输出形状与限制见 [PCA API 参考](api-reference.md#pca)。
