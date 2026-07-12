# Covariance

> 语言: 中文  
> 最后更新: 2026-07-12  
> 页面定位: 模型文档  
> 切换: [English](../en/models/covariance.md)

语言切换：[English](../en/models/covariance.md)

## 概览（Overview）

`covariance` 模块包含七种估计器：`EmpiricalCovariance`、`LedoitWolf`、`OAS`、`ShrunkCovariance`、稳健的 `MinCovDet`，以及稀疏精度矩阵估计 `GraphicalLasso`/`GraphicalLassoCV`。七者均支持 NumPy、CuPy 和 Torch 后端；Graphical Lasso 的坐标下降和 FAST-MCD 的 C-step 已改为后端原生执行。

## 路径（Path）

- `statgpu.covariance.EmpiricalCovariance`
- `statgpu.covariance.LedoitWolf`
- `statgpu.covariance.OAS`
- `statgpu.covariance.ShrunkCovariance`
- `statgpu.covariance.MinCovDet`
- `statgpu.covariance.GraphicalLasso`
- `statgpu.covariance.GraphicalLassoCV`

## 目标函数（Objective Function）

**EmpiricalCovariance** 计算最大似然样本协方差：

$$
\hat{S} = \frac{1}{n} X^\top X
$$

其中 \(X\) 为中心化数据矩阵（按列减去均值，除非设置 `assume_centered=True`）。

**LedoitWolf** 和 **OAS** 均产生如下形式的收缩协方差：

$$
\hat{\Sigma} = (1 - \alpha)\,\hat{S} + \alpha\,\mu\,I
$$

其中 \(\mu = \operatorname{tr}(\hat{S})/p\) 为样本协方差的平均特征值。两种估计器的差异仅在于最优收缩强度 \(\alpha\) 的计算方式。

**Ledoit-Wolf 收缩强度**（Ledoit & Wolf 2004）：

$$
\alpha = \operatorname{clip}\!\left(\frac{\beta}{\delta},\; 0,\; 1\right)
$$

其中

$$
\beta = \frac{1}{n^2}\left[\sum_{k=1}^{n} \|x_k\|_2^4 - n\,\|\hat{S}\|_F^2\right], \qquad
\delta = \|\hat{S} - \mu I\|_F^2 = \|\hat{S}\|_F^2 - \frac{\operatorname{tr}(\hat{S})^2}{p}
$$

**OAS 收缩强度**（Chen et al. 2010）：

$$
\alpha = \operatorname{clip}\!\left(\frac{\overline{S^2} + \mu^2}{(n+1)\!\left(\overline{S^2} - \mu^2/p\right)},\; 0,\; 1\right)
$$

其中 \(\overline{S^2} = \frac{1}{p^2}\sum_{i,j} S_{ij}^2\) 为 \(\hat{S}\) 元素平方的均值。

### 其他估计器

`ShrunkCovariance` 使用用户指定的收缩强度。`MinCovDet` 通过 FAST-MCD
选择协方差行列式较小的支持子集，并使用卡方阈值进行重加权。
`GraphicalLasso` 求解

$$
\max_{\Theta \succ 0}\; \log\det(\Theta)-\operatorname{tr}(S\Theta)
-\alpha\|\Theta\|_{1,\mathrm{off}},
$$

其中对角元素不受 L1 惩罚；`GraphicalLassoCV` 通过留出对数似然选择
`alpha`。

## 估计方程（Estimating Equation）

经验与收缩估计器使用直接计算；稳健和稀疏估计器使用迭代算法：

- **EmpiricalCovariance**：样本协方差 \(\hat{S} = X^\top X / n\) 直接计算。先尝试精确求逆；仅当求逆失败或结果非有限时才逐步增加对角 jitter。
- **LedoitWolf**：Ledoit-Wolf 的解析公式从中心化数据中闭式求解 \(\alpha\)，然后计算收缩协方差及其逆。
- **OAS**：与 Ledoit-Wolf 相同的闭式方法，但使用 OAS 收缩公式。该公式在高斯假设下推导，当 \(n > p\) 时渐近最优。
- **ShrunkCovariance**：使用用户指定的收缩强度。
- **MinCovDet**：30/50 个 seeded 随机起点，经后端原生 C-step 精炼并重加权。
- **GraphicalLasso**：协方差块坐标下降，内层使用软阈值坐标更新；外层以协方差最大变化量判断收敛。
- **GraphicalLassoCV**：在各 fold 上拟合并按留出高斯对数似然选择 `alpha`。

## 协方差与推断（Covariance/Inference）

所有估计器在 `fit()` 后产生以下拟合属性：

- `covariance_`：估计的协方差矩阵 \(\hat{\Sigma}\)（形状 `(n_features, n_features)`）。
- `precision_`：逆协方差矩阵 \(\hat{\Sigma}^{-1}\)（形状 `(n_features, n_features)`），通过抖动稳定求逆以保证数值稳健性。
- `location_`：估计的均值向量（形状 `(n_features,)`）；若 `assume_centered=True` 则为零向量。
- `shrinkage_`：收缩强度 \(\alpha\)，取值范围 \([0, 1]\) 的浮点数（LedoitWolf、OAS 与 ShrunkCovariance）。

`score()` 方法计算每个观测的平均高斯对数似然：

$$
\ell = -\frac{1}{2}\!\left(p \log(2\pi) + \log\det(\hat{\Sigma}) + \frac{1}{n}\sum_{k=1}^{n}(x_k - \hat{\mu})^\top \hat{\Sigma}^{-1}(x_k - \hat{\mu})\right)
$$

## 参数（Parameters）

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `assume_centered` | `False` | 若为 `True`，跳过均值估计和中心化；假设数据已经中心化 |
| `device` | `"auto"` | 计算设备：`"cpu"`、`"cuda"`、`"torch"` 或 `"auto"`（根据输入数组类型自动检测） |
| `n_jobs` | `None` | 并行任务数（保留参数，当前未启用） |

以上参数由七种估计器共享；`MinCovDet` 另有 `support_fraction`、`random_state`，Graphical Lasso 另有 `alpha`、`max_iter`、`tol`，CV 版本另有 `alphas` 与 `cv`。

## CPU+GPU 示例（CPU+GPU Examples）

```python
from statgpu.covariance import EmpiricalCovariance, LedoitWolf, OAS
import numpy as np

X = np.random.randn(500, 10)

# --- CPU ---

# 经验协方差
emp = EmpiricalCovariance(device="cpu")
emp.fit(X)
print(f"Covariance shape: {emp.covariance_.shape}")  # (10, 10)
print(f"Location shape:   {emp.location_.shape}")     # (10,)

# Ledoit-Wolf 收缩
lw = LedoitWolf(device="cpu")
lw.fit(X)
print(f"Shrinkage: {lw.shrinkage_:.4f}")              # 例如 0.1234

# OAS 收缩
oas = OAS(device="cpu")
oas.fit(X)
print(f"OAS shrinkage: {oas.shrinkage_:.4f}")

# 评分（平均对数似然）
ll = lw.score(X)
print(f"Log-likelihood: {ll:.4f}")

# 马氏距离
dists = lw.mahalanobis(X[:5])
print(f"Mahalanobis distances: {dists}")

# --- GPU (CuPy) ---

lw_gpu = LedoitWolf(device="cuda")
lw_gpu.fit(X)
print(f"GPU shrinkage: {lw_gpu.shrinkage_:.4f}")
print(f"GPU covariance shape: {lw_gpu.covariance_.shape}")

# --- GPU (PyTorch) ---

import torch
X_torch = torch.randn(500, 10, device="cuda", dtype=torch.float64)
lw_torch = LedoitWolf(device="cuda")
lw_torch.fit(X_torch)
print(f"Torch shrinkage: {lw_torch.shrinkage_:.4f}")
```

## 后端执行与验证边界

Graphical Lasso/CV 的中心化、协方差更新、坐标下降、求逆与 fold 评分，以及
MinCovDet 的 C-step、马氏距离、排序、支持集和重加权均保留在 NumPy/CuPy/Torch
后端。CPU 仅处理随机/fold 整数索引、收敛标量和卡方分布标量。

已验证 NumPy 与 Torch-CPU 数值一致性和输出后端；真实 CuPy/Torch CUDA 的
收敛、显存、性能与重复拟合验证仍为 `PARTIAL_REMOTE_PENDING`。

## strict/approx 差异（strict/approx difference）

协方差估计器没有单独的 strict 或 approx 模式。经验/收缩估计器使用直接公式；MinCovDet 使用内部 C-step；GraphicalLasso/CV 使用 `max_iter` 和以协方差最大变化量定义的 `tol`。

`LedoitWolf` 和 `OAS` 提供不同的收缩强度公式。根据使用场景选择：

- **LedoitWolf**：更通用；在各种 \(n/p\) 比率下表现良好。这是收缩协方差估计的标准推荐。
- **OAS**：在高斯假设下推导；当 \(n > p\) 时渐近最优，在该场景下通常比 Ledoit-Wolf 实现更低的均方误差。

## 输出（Outputs）

### 拟合属性

| 属性 | 形状 | 说明 |
|---|---|---|
| `covariance_` | `(n_features, n_features)` | 估计的协方差矩阵 |
| `precision_` | `(n_features, n_features)` | 逆协方差（精度）矩阵 |
| `location_` | `(n_features,)` | 估计的均值向量 |
| `n_samples_` | 标量 | 训练样本数 |
| `n_features_` | 标量 | 特征数 |
| `shrinkage_` | 标量 (float) | 收缩强度，取值 [0, 1]（仅 LedoitWolf/OAS） |

### 方法

| 方法 | 返回值 | 说明 |
|---|---|---|
| `fit(X)` | `self` | 对数据矩阵 X 拟合协方差模型 |
| `predict(X)` | `ndarray (n_samples,)` | X 中观测的马氏距离 |
| `score(X)` | `float` | 每个观测的平均高斯对数似然 |
| `mahalanobis(X)` | `ndarray (n_samples,)` | X 中观测的平方马氏距离 |

## 常见问题（FAQ）

**LedoitWolf 和 OAS 如何选择？**
当 \(n > p\)（样本数多于特征数）时推荐使用 OAS，因为它在高斯假设下推导且在该场景下渐近最优。LedoitWolf 更通用，当不确定或 \(n\) 与 \(p\) 接近时是更安全的默认选择。实际差异通常较小。

**`score()` 返回什么？**
在拟合的协方差和均值下，多元高斯分布的每个观测平均对数似然。值越大表示拟合越好。可用于不同估计器之间的模型比较。

**协方差矩阵奇异时会怎样？**
精度矩阵计算使用抖动稳定求逆：逐步增加对角增量直到获得稳定的逆矩阵。如果遇到持续的奇异性警告，考虑使用 LedoitWolf 或 OAS 替代 EmpiricalCovariance，因为收缩保证了良态估计。

**能否直接传入 CuPy 或 PyTorch 数组？**
可以。传入 CuPy ndarray 或 PyTorch tensor 时，后端根据输入类型自动检测。也可以对 NumPy 输入显式设置 `device="cuda"` 或 `device="torch"` 来强制 GPU 计算。

## 外部验证（External Validation）

七种估计器均有参考实现或结构不变量测试：

- `sklearn.covariance.EmpiricalCovariance`
- `sklearn.covariance.LedoitWolf`
- `sklearn.covariance.OAS`
- `sklearn.covariance.ShrunkCovariance`
- `sklearn.covariance.MinCovDet`
- `sklearn.covariance.GraphicalLasso`
- `sklearn.covariance.GraphicalLassoCV`

经验与收缩估计器在严格容差下对照 scikit-learn；MinCovDet 与 Graphical Lasso 还检查支持集、互逆性、对角线和稀疏结构。NumPy/Torch-CPU parity 见 `dev/tests/test_three_backend_native_followup.py`，尚不宣称完成真实 CUDA parity。

## 参考文献（References）

- Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365-411. [https://doi.org/10.1016/S0047-259X(03)00096-4](https://doi.org/10.1016/S0047-259X(03)00096-4)
- Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O. (2010). Shrinkage algorithms for MMSE covariance estimation. *IEEE Transactions on Signal Processing*, 58(10), 5297-5307. [https://doi.org/10.1109/TSP.2010.2053029](https://doi.org/10.1109/TSP.2010.2053029)
