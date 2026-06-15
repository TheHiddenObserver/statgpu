# Kernel Methods

> 语言: 中文  
> 最后更新: 2026-05-28  
> 页面定位: 模型文档  
> 切换: [English](../en/models/kernel-methods.md)

语言切换：[English](../en/models/kernel-methods.md)

## 概览（Overview）

核方法模块提供核岭回归（`KernelRidge`）、交叉验证核岭回归（`KernelRidgeCV`）以及六种核函数（RBF、多项式、线性、Laplacian、Sigmoid、余弦）。两个估计器均接受 `kernel` 参数，可选择内置核函数或用户自定义的可调用对象。所有计算通过后端无关的数组接口分发，支持 CPU（NumPy）、CuPy 和 PyTorch 后端，`KernelRidgeCV` 还支持自动 CUDA 加速。

## 路径（Path）

```
statgpu.nonparametric.kernel_methods.KernelRidge
statgpu.nonparametric.kernel_methods.KernelRidgeCV
statgpu.nonparametric.kernel_methods.pairwise_kernels
```

各核函数也可单独导入：

```
statgpu.nonparametric.kernel_methods.rbf_kernel
statgpu.nonparametric.kernel_methods.polynomial_kernel
statgpu.nonparametric.kernel_methods.linear_kernel
statgpu.nonparametric.kernel_methods.laplacian_kernel
statgpu.nonparametric.kernel_methods.sigmoid_kernel
statgpu.nonparametric.kernel_methods.cosine_kernel
```

## 目标函数（Objective Function）

**KernelRidge** 求解核岭回归对偶问题。给定从训练数据计算的 \(n \times n\) 核矩阵 \(K\)，对偶空间的目标函数为：

$$
\min_{\boldsymbol{\alpha}} \| \mathbf{y} - K \boldsymbol{\alpha} \|_2^2 + \lambda \| \boldsymbol{\alpha} \|_2^2
$$

其中 \(\lambda\) 为正则化强度（`alpha`）。闭式解为：

$$
\boldsymbol{\alpha} = (K + \lambda I)^{-1} \mathbf{y}
$$

**KernelRidgeCV** 利用核矩阵的特征分解 \(K = Q \Lambda Q^\top\) 高效地在一系列正则化参数上求解，无需为每个参数值重新求解线性系统。对任意 \(\lambda\) 的解为：

$$
\boldsymbol{\alpha}(\lambda) = Q \, \text{diag}\!\left(\frac{1}{\lambda_i + \lambda}\right) Q^\top \mathbf{y}
$$

其中 \(\lambda_i\) 为 \(K\) 的特征值。对网格中每个 \(\lambda\) 计算交叉验证 MSE，选择使平均 CV MSE 最小的值。

## 估计方程（Estimating Equation）

**KernelRidge**：对偶问题的一阶条件导出线性系统

$$
(K + \lambda I) \boldsymbol{\alpha} = \mathbf{y}
$$

通过 `xp.linalg.solve` 直接求解。对新数据 \(X_{\text{test}}\) 的预测计算为：

$$
\hat{\mathbf{y}} = K_{\text{test}} \boldsymbol{\alpha}
$$

其中 \(K_{\text{test}}\) 为测试数据与训练数据之间的核矩阵。

**KernelRidgeCV**：对每个交叉验证折叠，训练核矩阵仅需特征分解一次。所有 alpha 值的对偶系数随后通过单次向量化操作完成计算：

$$
\boldsymbol{\alpha}(\lambda) = Q \, \text{diag}\!\left(\frac{1}{\lambda_i + \lambda}\right) Q^\top \mathbf{y}_{\text{train}}
$$

在 torch CUDA 后端上，该 alpha 扫描通过批量矩阵操作在所有 alpha 值上完全向量化，避免了对 alpha 网格的 Python 级循环。

## 协方差与推断（Covariance/Inference）

核方法为非参数方法，不产生系数级别的推断（无标准误、t 值或 p 值）。模型质量通过以下指标评估：

- **R 方**：决定系数 \(R^2 = 1 - \text{SS}_{\text{res}} / \text{SS}_{\text{tot}}\)，由 `score` 方法计算。
- **交叉验证 MSE**（KernelRidgeCV）：K 折交叉验证的平均均方误差，存储在 `cv_results_["mean_mse"]` 中。

## 参数（Parameters）

**KernelRidge**：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `alpha` | `1.0` | 正则化强度（\(\lambda\)） |
| `kernel` | `"rbf"` | 核函数：`rbf`、`gaussian`、`linear`、`polynomial`、`poly`、`laplacian`、`sigmoid`、`cosine`，或可调用对象 |
| `gamma` | `None` | rbf/多项式/laplacian/sigmoid 的核系数。默认为 `1 / n_features` |
| `degree` | `3` | 多项式核的阶数 |
| `coef0` | `1` | 多项式核和 sigmoid 核的独立项 |
| `kernel_params` | `None` | 传递给核函数的额外关键字参数 |
| `device` | `"auto"` | `cpu` / `cuda` / `auto` |
| `n_jobs` | `None` | 未使用；保留以维持 API 兼容性 |

**KernelRidgeCV**（继承上述所有核相关参数，另有：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `alphas` | `None` | 待评估的正则化强度数组。若为 `None` 则自动生成 100 个点的对数网格 |
| `cv` | `5` | 交叉验证折数 |
| `random_state` | `None` | 折叠洗牌的随机状态 |

**核函数**：

| 函数 | 公式 |
|---|---|
| `rbf_kernel` | \(K(x, y) = \exp(-\gamma \|x - y\|^2)\) |
| `polynomial_kernel` | \(K(x, y) = (\gamma \, x^\top y + c_0)^d\) |
| `linear_kernel` | \(K(x, y) = x^\top y\) |
| `laplacian_kernel` | \(K(x, y) = \exp(-\gamma \|x - y\|_1)\) |
| `sigmoid_kernel` | \(K(x, y) = \tanh(\gamma \, x^\top y + c_0)\) |
| `cosine_kernel` | \(K(x, y) = \frac{x^\top y}{\|x\| \, \|y\|}\) |

所有核函数接受可选的 `xp` 参数用于后端分发（numpy/cupy/torch）。当 `xp` 为 `None` 时默认使用 NumPy。

## CPU+GPU 示例（CPU+GPU Examples）

```python
from statgpu.nonparametric.kernel_methods import KernelRidge, KernelRidgeCV
import numpy as np

X = np.random.randn(500, 10)
y = X @ np.random.randn(10) + 0.1 * np.random.randn(500)

# CPU
kr = KernelRidge(alpha=1.0, kernel="rbf", device="cpu")
kr.fit(X, y)
print(f"R^2: {kr.score(X, y):.4f}")

# GPU + 交叉验证
kr_cv = KernelRidgeCV(cv=5, kernel="rbf", device="cuda")
kr_cv.fit(X, y)
print(f"Best alpha: {kr_cv.alpha_:.6f}, R^2: {kr_cv.best_score_:.4f}")

# 预测
y_pred = kr_cv.predict(X)

# 查看交叉验证结果
print(f"Alpha grid size: {len(kr_cv.cv_results_['alphas'])}")
print(f"CV results shape: {kr_cv.cv_results_['mean_mse'].shape}")
```

使用自定义核函数：

```python
from statgpu.nonparametric.kernel_methods import KernelRidge

# 多项式核 + 自定义参数
kr_poly = KernelRidge(alpha=0.1, kernel="polynomial", degree=4, coef0=2.0, device="auto")
kr_poly.fit(X, y)

# Laplacian 核
kr_lap = KernelRidge(alpha=1.0, kernel="laplacian", gamma=0.5, device="cpu")
kr_lap.fit(X, y)

# 用户自定义核函数
def my_kernel(X, Y=None, xp=None):
    if xp is None:
        xp = np
    if Y is None:
        Y = X
    return xp.tanh(X @ Y.T + 1.0)

kr_custom = KernelRidge(alpha=1.0, kernel=my_kernel, device="cpu")
kr_custom.fit(X, y)
```

## strict/approx 差异（strict/approx difference）

核方法模块没有 strict/approx 模式区分。闭式对偶解直接计算，无迭代近似。

`KernelRidgeCV` 在未提供 `alphas` 时自动生成 100 个点的对数间距 alpha 网格。网格范围从 `max(lambda_min * 1e-3, 1e-8)` 到 `max(lambda_max * 10, 1)`，其中 `lambda_min` 和 `lambda_max` 为训练核矩阵的极值特征值。用户可提供自定义 `alphas` 数组覆盖此行为。

## 输出（Outputs）

**KernelRidge 拟合属性**：

| 属性 | 形状 | 说明 |
|---|---|---|
| `dual_coef_` | `(n_samples,)` 或 `(n_samples, n_targets)` | 核空间中的对偶系数 |
| `X_fit_` | `(n_samples, n_features)` | 用于预测的训练数据 |

**KernelRidgeCV 拟合属性**：

| 属性 | 形状 | 说明 |
|---|---|---|
| `alpha_` | 标量 | 交叉验证选出的最优正则化参数 |
| `best_score_` | 标量 | 最优 alpha 对应的 R 方得分 |
| `cv_results_` | dict | 字典，键包括：`alphas`、`mean_mse`、`mse_table`、`best_alpha`、`best_score` |
| `estimator_` | `KernelRidge` | 使用最优 alpha 拟合的 `KernelRidge` 实例 |
| `dual_coef_` | `(n_samples,)` 或 `(n_samples, n_targets)` | `estimator_.dual_coef_` 的快捷访问 |
| `X_fit_` | `(n_samples, n_features)` | `estimator_.X_fit_` 的快捷访问 |

**方法**（两个类共有）：

| 方法 | 说明 |
|---|---|
| `fit(X, y)` | 拟合模型。返回 `self`。 |
| `predict(X)` | 对新数据预测目标值。 |
| `score(X, y)` | 返回决定系数 R 方。 |

## 常见问题（FAQ）

**问：不提供 alpha 网格时如何生成？**
答：`KernelRidgeCV` 计算训练核矩阵的特征值，创建从 `max(lambda_min * 1e-3, 1e-8)` 到 `max(lambda_max * 10, 1)` 的 100 个点的对数间距网格。如果范围退化（alpha_min >= alpha_max），网格设置为从 `alpha_max * 1e-4` 到 `alpha_max`。

**问：GPU 的优势在哪里？**
答：使用 torch CUDA 后端时，每个折叠训练核矩阵的特征分解以及整个 alpha 扫描（对所有 100 个 alpha 值计算对偶系数和预测）完全在 GPU 上通过批量矩阵操作运行。numpy/CuPy 路径回退到 Python 级的 alpha 循环。

**问：能否使用自定义核函数？**
答：可以。将任意可调用对象作为 `kernel` 参数传入。该可调用对象应接受 `(X, Y, xp=None, **kwargs)` 并返回形状为 `(n_samples_X, n_samples_Y)` 的核矩阵。`xp` 参数提供用于后端分发的数组模块。

**问：不指定 `gamma` 时默认值是多少？**
答：当 `gamma` 为 `None` 时，核函数默认使用 `1 / n_features`，遵循 scikit-learn 的约定。

**问：KernelRidge 是否支持多输出目标？**
答：支持。如果 `y` 的形状为 `(n_samples, n_targets)`，`KernelRidge` 和 `KernelRidgeCV` 会同时拟合所有目标。对偶系数的形状为 `(n_samples, n_targets)`。

## 外部验证（External Validation）

KernelRidge 结果针对 `sklearn.kernel_ridge.KernelRidge` 进行验证，所有支持的核函数类型下相对误差低于 \(10^{-10}\)。一致性检查维护在测试套件中，覆盖 RBF、多项式、线性、Laplacian、Sigmoid 和余弦核在 CPU 和 GPU 后端上的表现。

## 参考文献（References）

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. Chapter 6. [https://hastie.su.domains/ElemStatLearn/](https://hastie.su.domains/ElemStatLearn/)
- Saunders, C., Gammerman, A., & Vovk, V. (1998). Ridge regression learning algorithm in dual variables. *Proceedings of the 15th International Conference on Machine Learning (ICML)*, 515-521.
