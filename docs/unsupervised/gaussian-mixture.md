# GaussianMixture

> 语言：中文
> 最后更新：2026-05-07
> English: [English](../en/unsupervised/gaussian-mixture.md)

## 概览

`GaussianMixture` 使用 EM 算法拟合高斯混合模型。当前支持 `"diag"`、`"spherical"`、`"tied"` 和 `"full"` 四种协方差类型，并覆盖 CPU、CuPy/CUDA 和 Torch CUDA 三端。

## 导入路径

```python
from statgpu.unsupervised import GaussianMixture
```

## 目标函数 / 损失函数

在给定混合成分数 `K` 时，模型最大化平均对数似然：

$$
\ell(\theta)
=
\frac{1}{n}\sum_{i=1}^{n}
\log\left[
\sum_{k=1}^{K}
\pi_k
\mathcal{N}\left(x_i \mid \mu_k, \Sigma_k\right)
\right].
$$

`covariance_type` 决定 `\Sigma_k` 的形状：每个成分一组对角协方差、每个成分一个球形方差、所有成分共享一个完整协方差，或每个成分各自拥有完整协方差。`reg_covar` 会向协方差估计加入一个很小的对角 ridge，提高数值稳定性。

## 估计方程

实现使用 log-domain EM：

- 初始化：使用 KMeans 或随机样本初始化均值。
- E-step：计算加权成分 log probability：

  $$
  a_{ik}
  =
  \log \pi_k
  +
  \log \mathcal{N}\left(x_i \mid \mu_k, \Sigma_k\right).
  $$

  用 log-sum-exp 归一化：

  $$
  \log p(x_i)
  =
  \operatorname{logsumexp}_{k=1}^{K}\left(a_{ik}\right).
  $$

  responsibility 为：

  $$
  r_{ik}
  =
  \frac{
    \pi_k \mathcal{N}\left(x_i \mid \mu_k, \Sigma_k\right)
  }{
    \sum_{\ell=1}^{K}
    \pi_\ell \mathcal{N}\left(x_i \mid \mu_\ell, \Sigma_\ell\right)
  }.
  $$

- M-step：更新有效样本数、权重、均值和协方差：

  $$
  n_k = \sum_{i=1}^{n} r_{ik},
  \qquad
  \pi_k = \frac{n_k}{n},
  \qquad
  \mu_k = \frac{1}{n_k}\sum_{i=1}^{n} r_{ik}x_i.
  $$

  full covariance：

  $$
  \Sigma_k
  =
  \frac{1}{n_k}\sum_{i=1}^{n}r_{ik}
  (x_i-\mu_k)(x_i-\mu_k)^\top
  +
  \text{reg\_covar}\,I.
  $$

  tied covariance：

  $$
  \Sigma
  =
  \frac{1}{n}\sum_{k=1}^{K}\sum_{i=1}^{n}r_{ik}
  (x_i-\mu_k)(x_i-\mu_k)^\top
  +
  \text{reg\_covar}\,I.
  $$

  diagonal 和 spherical 协方差使用同一责任加权方差的对角或特征平均：

  $$
  \sigma_{kj}^{2}
  =
  \max\left(
    \frac{1}{n_k}\sum_{i=1}^{n} r_{ik} x_{ij}^{2}
    -
    \mu_{kj}^{2},
    \text{reg\_covar}
  \right),
  \qquad
  \sigma_k^2 = \frac{1}{p}\sum_{j=1}^{p}\sigma_{kj}^{2}.
  $$

- 监控的 lower bound 为：

  $$
  \mathcal{L}
  =
  \frac{1}{n}\sum_{i=1}^{n}\log p(x_i).
  $$

  当 lower bound 的提升小于 `tol` 或达到 `max_iter` 时停止。`n_init` 会运行多组初始化，并保留 lower bound 最高的一组。

## 参数

- `n_components`：混合成分数。
- `covariance_type`：`"diag"`、`"spherical"`、`"tied"` 或 `"full"`。
- `tol`、`reg_covar`、`max_iter`、`n_init`。
- `init_params`：`"kmeans"` 或 `"random"`。
- `random_state`。
- `device`：`"auto"`、`"cpu"`、`"cuda"` 或 `"torch"`。

## CPU+GPU 示例

```python
import numpy as np
from statgpu.unsupervised import GaussianMixture

X = np.random.default_rng(0).normal(size=(4000, 16))

gmm = GaussianMixture(n_components=4, covariance_type="full", random_state=0, device="torch")
gmm.fit(X)
labels = gmm.predict(X)
proba = gmm.predict_proba(X)
ll = gmm.score(X)
```

## strict/approx 差异

GMM 提供 likelihood 分数，但没有 strict inference covariance 或 p-value 模式。EM 优化的是非凸似然，可能收敛到局部最优；结果可复现性取决于初始化、`random_state`、`n_init`、`tol` 和 `max_iter`。

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

**应该选择哪种 covariance type？**
`"diag"` 和 `"spherical"` 计算更便宜，适合成分内特征相关性较弱的场景；`"tied"` 在所有成分之间共享一个完整协方差；`"full"` 最灵活，但参数最多，也需要更多样本支撑。

**`score`、`score_samples`、`aic`、`bic` 分别是什么？**
`score_samples` 返回逐样本 log likelihood，`score` 返回平均 log likelihood，`aic`/`bic` 按对应 covariance type 的参数量计算。

## 外部验证

- 测试：`dev/tests/test_unsupervised_gmm.py`。
- Benchmark：`dev/benchmarks/benchmark_unsupervised_phase3b.py`。
- 最新远程 artifact：`results/unsupervised_phase3b_verify_20260507_003957.json`。
- Baseline：sklearn `GaussianMixture`，对齐 `covariance_type`、初始化和收敛参数。
- Phase 3B 验证目标：`"diag"`、`"spherical"`、`"tied"`、`"full"` 在 CPU/CuPy/Torch 三端的 score 一致性，以及与 sklearn 的 log likelihood、AIC/BIC、responsibility 对齐。

## References

- Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society: Series B (Methodological)*, 39(1), 1-22. https://doi.org/10.1111/j.2517-6161.1977.tb01600.x
- McLachlan, G. J., & Peel, D. (2000). *Finite Mixture Models*. Wiley Series in Probability and Statistics. Wiley.
