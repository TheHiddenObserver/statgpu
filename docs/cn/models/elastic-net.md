# 弹性网络（Elastic Net）

> Language: Chinese (中文)  
> Last updated: 2026-09-04<br>
> This page: 模型文档  
> Language switch: [English](../../en/models/elastic-net.md)

## 概述

`ElasticNet` 结合了 L1 和 L2 正则化，在稀疏特征选择（Lasso）和系数收缩（Ridge）之间取得平衡。支持 CPU、CuPy GPU 和 PyTorch GPU 后端，可灵活选择设备。

## 路径

`statgpu.linear_model.ElasticNet`

## 目标函数

Elastic Net 优化问题为：

$$
\min_{\beta} \frac{1}{2n}\|y - X\beta\|_2^2 + \alpha \cdot \lambda \cdot \|\beta\|_1 + \frac{\alpha}{2} \cdot (1 - \lambda) \cdot \|\beta\|_2^2
$$

其中：
- `alpha` (α) 控制整体正则化强度
- `l1_ratio` (λ) 混合 L1 和 L2：λ=1 为 Lasso，λ=0 为 Ridge
- 损失函数缩放因子 `1/(2n)` 使 `alpha` 的解释与样本量无关

**正则化缩放说明**：`ElasticNet` 与 `Ridge` 均采用相同的平均损失尺度。因此当 `l1_ratio=0` 时，`ElasticNet(alpha)` 等价于 `Ridge(alpha)`；公开参数 `alpha` 不需要再乘以样本量。

## 估计方程

Elastic Net 估计量求解以下一阶最优性（KKT）条件：

$$
\frac{1}{n} X^\top (X\hat{\beta} - y) + \alpha(1-\lambda)\hat{\beta} + \alpha\lambda \cdot \partial\|\hat{\beta}\|_1 = 0
$$

其中 $\partial\|\hat{\beta}\|_1$ 是 L1 范数的次微分：
- 当 $\hat{\beta}_j \neq 0$ 时：$\text{sign}(\hat{\beta}_j)$
- 当 $\hat{\beta}_j = 0$ 时：$[-1, 1]$ 中的任意值

收敛时，KKT 残差（次梯度违反）满足：
$$
\left| \frac{1}{n} X_j^\top(y - X\hat{\beta}) - \alpha(1-\lambda)\hat{\beta}_j \right| \leq \alpha\lambda \quad \forall j
$$

## 估计算法

Elastic Net 通过 **FISTA**（快速迭代收缩阈值算法）求解，这是一种带有 Nesterov 动量加速的 proximal gradient 方法。

### 关键优化洞察

L2 正则化项**仅在 proximal 步中处理**，不在梯度计算中处理：

```python
# 仅计算 RSS 的梯度（L2 单独处理）
grad = (X.T @ X @ w - X.T @ y) / n

# Proximal 步：软阈值 + L2 缩放
w = soft_threshold(w_tilde, alpha * l1_ratio * step) / (1 + alpha * (1 - l1_ratio) * step)
```

这避免了冗余计算并提高了数值稳定性。

### 收敛判据

通过 `stopping` 参数提供两种停止模式：

| 模式 | 说明 |
|------|------|
| `coef_delta` | 当 `||w_new - w_old||_∞ < tol` 时停止 |
| `kkt` | 当 KKT 次梯度违反 < tol 时停止 |

对于 `kkt` 模式，最优性条件为：
- 对于非零系数：`|∇f + α(1-λ)w + αλ·sign(w)| < tol`
- 对于零系数：`|∇f + α(1-λ)w| ≤ αλ`

**注意**：数值优化中 KKT 违反 ~1e-2 是可接受的；不需要精确为零。

## 求解器支持

| `solver` 值 | CPU | CuPy / Torch | 说明 |
|---|:---:|:---:|---|
| `fista`（默认） | 支持 | 支持 | 推荐的近端优化路径 |
| `auto` | FISTA | FISTA | squared-error + Elastic Net 当前的自动分发 |
| `fista_bb` | 支持 | 支持 | 自适应谱步长 |
| `admm` | 支持 | 支持 | 拆分求解替代路径；只支持均匀样本权重 |
| `coordinate_descent` | 支持 | 不支持 | CPU-only 兼容路径 |

Elastic Net 含非光滑 L1 部分，因此会拒绝 `newton`、`lbfgs`、`irls` 与 `exact`。单模型拟合时，`cpu_solver` 不会覆盖 `solver`。完整迭代公式见[求解器指南](../guides/solver-algorithms.md)。

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
| `solver` | `"fista"` | 后端感知的优化方法 |
| `cpu_solver` | `"fista"` | CPU 求解器覆盖选项 |
| `lipschitz_L` | `None` | 可选的用户指定 Lipschitz 常数 |
| `gpu_memory_cleanup` | `False` | 在支持的后端上于拟合后释放内存池 |
| `compute_inference` | `False` | 计算拟合后系数推断 |
| `inference_method` | `"debiased"` | `"debiased"`、`"cpu_ols"` 或 `"bootstrap"` |
| `cov_type` | `"nonrobust"` | 适用方法中的协方差约定 |
| `hac_maxlags` | `None` | 支持 HAC 时使用的滞后阶数 |

公开 wrapper 不接受单独的 `backend`、`warm_start` 或 `random_state`
构造参数。后端由 `device` 控制；单次拟合的 warm start 可通过
`fit(initial_coef=...)` 提供。

## CPU/GPU 示例

```python
from statgpu.linear_model import ElasticNet

# CPU (NumPy)
model_cpu = ElasticNet(alpha=0.1, l1_ratio=0.5, device="cpu")
model_cpu.fit(X, y)
print(f"R²: {model_cpu.score(X, y):.4f}")

# GPU (CuPy)
model_gpu_cupy = ElasticNet(
    alpha=0.1, l1_ratio=0.5, device="cuda",
    gpu_memory_cleanup=True
)
model_gpu_cupy.fit(X, y)

# GPU (PyTorch)
model_gpu_torch = ElasticNet(
    alpha=0.1, l1_ratio=0.5, device="torch"
)
model_gpu_torch.fit(X, y)
```

后端性能取决于样本量、特征维数、dtype、硬件、数据驻留位置和传输成本。不要仅依据固定阈值选择后端；应对实际目标工作负载进行 benchmark。

## 协方差/推断

`ElasticNet` 默认仅进行估计。设置 `compute_inference=True` 后，将通过共享的
penalized-linear 推断引擎执行拟合后推断。默认
`inference_method="debiased"` 使用 nodewise Lasso 构造偏误校正估计量、标准误、
z 统计量、p 值与 95% 置信区间；推断成功后可调用 `summary()`。

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `compute_inference` | `False` | 启用拟合后系数推断 |
| `inference_method` | `"debiased"` | `"debiased"`、`"cpu_ols"` 或 `"bootstrap"` |
| `cov_type` | `"nonrobust"` | 在相应推断方法中使用的协方差约定 |
| `hac_maxlags` | `None` | 所选方法支持 HAC 时使用的滞后阶数 |

NumPy、CuPy 与 Torch 拟合路径均已实现 debiased 推断。CPU 验证属于托管测试套件；
每个精确发布候选仍必须通过 CuPy 与 Torch 的物理 CUDA 远程验证。
Post-selection OLS 只是启发式方法，不保证有效的选择后覆盖率。推断以已选定的正则化
参数为条件，不会改变原 penalized coefficient。

对于 `ElasticNetCV`，`compute_inference=True` 仅作用于 alpha 与 `l1_ratio`
选定后的全数据最终重拟合；各折模型仍仅用于估计和评分。

## 求解器与推断语义

默认估计器使用 FISTA 优化声明的 Elastic Net 目标函数。`stopping` 仅改变
收敛诊断（`coef_delta` 或 KKT violation），并不定义不同的统计近似模式。

`compute_inference=False` 只返回 penalized estimate。设置
`compute_inference=True` 后，原拟合系数保持不变，并在拟合完成后运行所选推断方法。
独立 `ElasticNet` wrapper 与 `ElasticNetCV` 的最终全数据重拟合都直接支持该契约；
用户不需要仅为了 debiased inference 而切换到其他估计器类。

## 输出属性

拟合后可用以下属性：

| 属性 | 说明 |
|------|------|
| `coef_` | 估计的系数（形状：n_features） |
| `intercept_` | 拟合的截距 |
| `n_iter_` | 收敛所需迭代次数 |
| `aic` | 推断结果提供时的 Akaike 信息准则 |
| `bic` | 推断结果提供时的 Bayesian 信息准则 |

方法：`fit(X, y)`, `predict(X)`, `score(X, y)`, `summary()`

## 数值验证

维护中的回归测试会按 dtype 和求解路径选择相应容差，检查支持后端之间以及与参考实现的数值一致性。物理 CUDA 验证仍属于 exact-head handoff；不存在适用于所有工作负载的统一系数误差阈值或加速比。

## 参考文献

- Zou, H., & Hastie, T. (2005). Regularization and variable selection via the Elastic Net. *Journal of the Royal Statistical Society: Series B*, 67(2), 301-320.
- Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm for linear inverse problems. *SIAM Journal on Imaging Sciences*, 2(1), 183-202.
