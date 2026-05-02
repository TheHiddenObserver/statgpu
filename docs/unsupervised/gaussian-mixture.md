# GaussianMixture

> 语言：中文
> 最后更新：2026-05-02
> English: [English](../en/unsupervised/gaussian-mixture.md)

## 概览

`GaussianMixture` 使用 expectation-maximization 拟合 diagonal-covariance Gaussian mixture model。它支持 CPU、CuPy/CUDA 和 Torch CUDA。

## 导入路径

```python
from statgpu.unsupervised import GaussianMixture
```

## 目标函数 / 损失函数

对 `K` 个 components 和 diagonal covariance `sigma_k^2`，模型最大化平均 log likelihood：

```text
(1 / n) * sum_i log sum_k pi_k * Normal(x_i | mu_k, diag(sigma_k^2))
```

`reg_covar` 为 diagonal variances 提供下界，提高数值稳定性。

## 估计方程

实现使用 log-domain EM：

- 用 KMeans 或随机样本初始化 means。
- E-step：
  - 计算 weighted log probabilities：`log pi_k + log N(x_i | mu_k, Sigma_k)`；
  - 使用 `logsumexp` 归一化；
  - 得到 responsibilities `r_ik`。
- M-step：
  - `n_k = sum_i r_ik`；
  - `pi_k = n_k / n`；
  - `mu_k = sum_i r_ik x_i / n_k`；
  - `sigma_k^2 = max(sum_i r_ik x_i^2 / n_k - mu_k^2, reg_covar)`。
- 当 lower-bound improvement 小于 `tol` 或达到 `max_iter` 时停止。
- 运行 `n_init` 次初始化，保留 lower bound 最高的结果。

## 参数

- `n_components`：mixture components 数量。
- `covariance_type`：仅支持 `"diag"`。
- `tol`、`reg_covar`、`max_iter`、`n_init`。
- `init_params`：`"kmeans"` 或 `"random"`。
- `random_state`。
- `device`：`"auto"`、`"cpu"`、`"cuda"` 或 `"torch"`。

## CPU+GPU 示例

```python
import numpy as np
from statgpu.unsupervised import GaussianMixture

X = np.random.default_rng(0).normal(size=(4000, 16))

gmm = GaussianMixture(n_components=4, covariance_type="diag", random_state=0, device="torch")
gmm.fit(X)
labels = gmm.predict(X)
proba = gmm.predict_proba(X)
ll = gmm.score(X)
```

## strict/approx 差异

GMM 有 likelihood scores，但没有 strict inference covariance 或 p-value 模式。EM 优化非凸 likelihood，可能收敛到局部最优。结果可复现性取决于初始化、`random_state`、`n_init`、`tol` 和 `max_iter`。

## 输出字段

- `weights_`
- `means_`
- `covariances_`
- `precisions_cholesky_`
- `converged_`
- `n_iter_`
- `lower_bound_`
- `n_features_in_`

## FAQ

**支持 full covariance 吗？**
不支持。Phase 2 仅支持 `covariance_type="diag"`。

**`score`、`score_samples`、`aic`、`bic` 分别是什么？**
`score_samples` 返回每个样本的 log likelihood，`score` 返回均值，`aic`/`bic` 使用 diagonal-GMM 参数数量计算。

## 外部验证

- 测试：`dev/tests/test_unsupervised_gmm.py`。
- Benchmark：`dev/benchmarks/benchmark_unsupervised_phase2.py`。
- Baseline：sklearn `GaussianMixture(covariance_type="diag")`，对齐初始化和收敛参数。
- 最新远程矩阵：CPU/CuPy/Torch scores 差异处于浮点噪声量级；sklearn score diff 约 `2.4e-11`。

## References

- Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm.
