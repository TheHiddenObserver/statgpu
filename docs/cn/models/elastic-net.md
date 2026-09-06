# Elastic Net 弹性网络

> Language: Chinese (中文)  
> Last updated: 2026-09-06<br>
> This page: 模型文档  
> Language switch: [English](../../en/models/elastic-net.md)

## 概述

`ElasticNet` 结合 L1 与 L2 正则化，在稀疏特征选择（Lasso）和系数收缩（Ridge）之间取得平衡。支持 CPU、CuPy GPU 与 PyTorch GPU。直接拟合统一使用**与后端无关的 `solver` 接口**；`device` 只负责决定在哪里执行。

## 路径

`statgpu.linear_model.ElasticNet`

## 目标函数

Elastic Net 优化问题为：

$$
\min_{\beta} \frac{1}{2n}\|y - X\beta\|_2^2 + \alpha \cdot \lambda \cdot \|\beta\|_1 + \frac{\alpha}{2} \cdot (1 - \lambda) \cdot \|\beta\|_2^2
$$

其中：
- `alpha` (α) 控制整体正则化强度
- `l1_ratio` (λ) 混合 L1 与 L2：λ=1 为 Lasso，λ=0 为 Ridge
- `1/(2n)` 表示公开 `alpha` 使用 average-loss 尺度

**正则化缩放说明**：`ElasticNet` 与 `Ridge` 使用同一 average-loss convention，因此 `l1_ratio=0` 时，相同公开 `alpha` 下二者使用一致的 L2 penalty scale。

## 估计方程

Elastic Net 满足一阶最优性条件：

$$
\frac{1}{n} X^\top (X\hat{\beta} - y) + \alpha(1-\lambda)\hat{\beta} + \alpha\lambda \cdot \partial\|\hat{\beta}\|_1 = 0.
$$

对**直接单次拟合**，`solver` 是所有后端上的权威算法选择器。`device` 单独控制 CPU/CuPy/Torch 执行位置。历史 `cpu_solver` 参数已进入弃用流程，在统一引擎中不再代表第二套 CPU direct-fit solver。参见 [penalized solver API 迁移指南](../guides/penalized-solver-api-migration.md)。

## 估计算法

默认路径使用 **FISTA**（快速迭代收缩阈值算法），即带 Nesterov 加速的 proximal-gradient 方法。其他求解器是否可用取决于公开 solver compatibility contract。

### 关键优化洞察

Elastic Net 的 L1/L2 由 proximal operator 共同处理：

```python
# average squared-error 梯度
grad = (X.T @ X @ w - X.T @ y) / n

# Elastic Net proximal step
w = soft_threshold(w_tilde, alpha * l1_ratio * step) / (
    1 + alpha * (1 - l1_ratio) * step
)
```

### 收敛判据

`stopping` 提供两种停止模式：

| 模式 | 说明 |
|------|------|
| `coef_delta` | 系数变化低于 `tol` 时停止 |
| `kkt` | KKT 次梯度违反低于配置阈值时停止 |

数值收敛只表示声明的优化问题被求解到相应精度，并不构成另一种统计近似模型。

## 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `alpha` | `1.0` | 总体正则化强度 |
| `l1_ratio` | `0.5` | L1 混合比例：0=Ridge，1=Lasso |
| `fit_intercept` | `True` | 拟合不受惩罚的截距 |
| `max_iter` | `1000` | 最大求解迭代次数 |
| `tol` | `1e-4` | 收敛容差 |
| `stopping` | `"coef_delta"` | `"coef_delta"` 或 `"kkt"` 停止准则 |
| `device` | `"auto"` | `"auto"`、`"cpu"`、`"cuda"`（CuPy）或 `"torch"` |
| `n_jobs` | `None` | 适用 CPU 路径的并行度 |
| `solver` | `"fista"` | 与后端无关的 direct-fit 优化方法 |
| `cpu_solver` | `"fista"` | **Deprecated compatibility parameter**；新代码请使用 `solver` |
| `lipschitz_L` | `None` | 可选的用户指定 Lipschitz 常数 |
| `gpu_memory_cleanup` | `False` | 在支持的后端上于拟合后释放内存池 |
| `compute_inference` | `False` | 计算拟合后系数推断 |
| `inference_method` | `"debiased"` | `"debiased"`、`"cpu_ols"` 或 `"bootstrap"` |
| `cov_type` | `"nonrobust"` | 适用方法中的协方差约定 |
| `hac_maxlags` | `None` | 支持 HAC 时使用的滞后阶数 |

公开 wrapper 不接受单独的 `backend`、`warm_start` 或 `random_state` 构造参数。后端由 `device` 控制；单次 warm start 可通过 `fit(initial_coef=...)` 提供。

## CPU/GPU 示例

```python
from statgpu.linear_model import ElasticNet

# CPU：solver 选择算法，device 选择执行后端。
model_cpu = ElasticNet(
    alpha=0.1,
    l1_ratio=0.5,
    device="cpu",
    solver="fista",
)
model_cpu.fit(X, y)
print(f"R²: {model_cpu.score(X, y):.4f}")

# GPU 仍使用同一个 solver 接口
model_gpu_cupy = ElasticNet(
    alpha=0.1,
    l1_ratio=0.5,
    device="cuda",
    solver="fista",
    gpu_memory_cleanup=True,
)
model_gpu_cupy.fit(X, y)

model_gpu_torch = ElasticNet(
    alpha=0.1,
    l1_ratio=0.5,
    device="torch",
    solver="fista",
)
model_gpu_torch.fit(X, y)
```

后端性能取决于样本量、特征维数、dtype、硬件、数据驻留位置与传输成本。应针对实际工作负载 benchmark。

## 协方差/推断

`ElasticNet` 默认仅进行估计。设置 `compute_inference=True` 后，通过共享 penalized-linear inference engine 执行拟合后推断。默认 `inference_method="debiased"` 使用 nodewise Lasso 构造 bias-corrected estimator、标准误、z 统计量、p 值与置信区间；推断成功后可调用 `summary()`。

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `compute_inference` | `False` | 启用拟合后系数推断 |
| `inference_method` | `"debiased"` | `"debiased"`、`"cpu_ols"` 或 `"bootstrap"` |
| `cov_type` | `"nonrobust"` | 在相应推断方法中使用的协方差约定 |
| `hac_maxlags` | `None` | 所选方法支持 HAC 时使用的滞后阶数 |

Post-selection OLS 是启发式方法，不提供一般 selective-inference coverage。推断条件于已选择的正则化参数，并不会改变 penalized coefficients。

对于 `ElasticNetCV`，`compute_inference=True` 仅作用于 alpha 与 `l1_ratio` 选定后的最终 full-data refit；各折模型仍仅用于估计和评分。

## 求解器与推断语义

对于直接 `ElasticNet.fit`，**CPU 与 GPU 都使用 `solver`**。`device` 决定执行后端，`solver` 决定优化算法。`cpu_solver` 是早期 hardware-split API 的 deprecated compatibility 参数，新代码不应继续使用。

`compute_inference=False` 只返回 penalized estimate；开启推断后保留同一拟合系数，再运行所选 post-fit inference method。

## 输出属性

拟合后可用以下属性：

| 属性 | 说明 |
|------|------|
| `coef_` | 估计系数 |
| `intercept_` | 拟合截距 |
| `n_iter_` | 收敛所需迭代次数 |
| `aic` | 可用时的 Akaike 信息准则 |
| `bic` | 可用时的 Bayesian 信息准则 |

方法：`fit(X, y)`, `predict(X)`, `score(X, y)`, `summary()`

## 数值验证

维护中的回归测试会按 dtype 与 solver path 检查支持后端之间及与参考实现的数值一致性。solver API 迁移行为由 `dev/tests/test_penalized_solver_api_cleanup.py` 覆盖。

## 参考文献

- Zou, H., & Hastie, T. (2005). Regularization and variable selection via the Elastic Net. *Journal of the Royal Statistical Society: Series B*, 67(2), 301-320.
- Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm for linear inverse problems. *SIAM Journal on Imaging Sciences*, 2(1), 183-202.
